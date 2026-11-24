"""Last-vision-layer incoming attention observer; no model/backend modification.

Use native returned probabilities when available. Otherwise reconstruct only
last-layer probabilities from its ACTUAL q_proj/k_proj outputs, native scale
and mask (eager-style numerical policy, NOT fused SDPA internal weights).
Only reduced per-projector-token scores survive the hook. No disk tensor dump.
"""
from dataclasses import dataclass
import math
import time
import numpy as np
import torch
from coverpruner_spatial_selection import select_method


def _integer_positions(values, length, device):
    p=torch.as_tensor(values,device=device)
    if p.ndim!=1 or p.numel()==0 or p.dtype not in (torch.int32,torch.int64):
        raise ValueError('Explicit integer patch positions required')
    if p.min()<0 or p.max()>=length or p.unique().numel()!=p.numel():
        raise ValueError('Duplicate/out-of-range patch positions')
    return p.long()


def expected_patch_groups(grid=(32,32),scale=4):
    rows,cols=grid
    if any(isinstance(x,bool) or not isinstance(x,int) or x<=0 for x in (rows,cols,scale)) or rows%scale or cols%scale:
        raise ValueError('Invalid explicit patch grid/connector scale')
    return torch.tensor([[cols*(v*scale+a)+u*scale+b for a in range(scale) for b in range(scale)]
                         for v in range(rows//scale) for u in range(cols//scale)],dtype=torch.long)


def validated_connector_groups(connector,grid=(32,32),scale=4):
    """Execute actual index-only pixel_shuffle on CPU markers, no projection."""
    expected=expected_patch_groups(grid,scale)
    if grid[0]!=grid[1]: raise ValueError('Installed connector requires square patch grid')
    ids=torch.arange(grid[0]*grid[1],dtype=torch.float32).reshape(1,-1,1)
    actual=connector.pixel_shuffle(ids,scale_factor=scale)
    if tuple(actual.shape)!=(1,expected.shape[0],scale**2):
        raise ValueError('Unexpected native connector grouping shape')
    groups=actual[0].long()
    if not torch.equal(groups.sort(-1).values,expected.sort(-1).values):
        raise ValueError('BLOCKER: native connector does not match nominal patch groups')
    if groups.unique().numel()!=grid[0]*grid[1]:
        raise ValueError('BLOCKER: patch duplicated/lost in connector')
    return groups


def reconstruct_attention(q,k,scaling,mask=None):
    """Vision has no GQA/RoPE here; position context is already in native Q/K."""
    if q.ndim!=4 or q.shape!=k.shape or not q.is_floating_point() or q.dtype!=k.dtype or any(x==0 for x in q.shape):
        raise ValueError('Require equal nonempty view/head/patch/head_dim Q/K')
    if not math.isfinite(scaling) or scaling<=0 or not torch.isfinite(q).all() or not torch.isfinite(k).all():
        raise ValueError('Nonfinite Q/K or invalid scale')
    logits=torch.matmul(q,k.transpose(-1,-2))*scaling
    if not torch.isfinite(logits).all(): raise ValueError('Nonfinite logits')
    if mask is not None:
        try:
            if torch.broadcast_shapes(mask.shape,logits.shape)!=logits.shape:
                raise ValueError("Native mask expands Q/K shape")
        except RuntimeError as exc: raise ValueError('Invalid native mask shape') from exc
        if mask.dtype==torch.bool:
            if not mask.any(-1).all(): raise ValueError('All keys masked')
            logits=logits.masked_fill(~mask,-float('inf'))
        elif mask.is_floating_point():
            valid=(mask==0)
            blocked=torch.isneginf(mask)|(mask<=torch.finfo(mask.dtype).min/2)
            if not (valid|blocked).all() or not valid.any(-1).all():
                raise ValueError('Invalid/all-masked native additive mask')
            logits=logits+mask
        else: raise ValueError('Unsupported native mask dtype')
    probabilities=torch.softmax(logits,dim=-1,dtype=torch.float32).to(q.dtype)
    if not torch.isfinite(probabilities).all(): raise ValueError('Nonfinite probabilities')
    return probabilities


def incoming_patch_scores(attention,query_valid,key_valid,*,patch_positions=None):
    """Mean over heads and allowed QUERY rows, score each KEY column.

    Explicit positions can exclude special keys; they are not renormalized.
    Smol observer rejects extra tokens entirely rather than guessing CLS layout.
    """
    if attention.ndim!=4 or any(x==0 for x in attention.shape) or not attention.is_floating_point():
        raise ValueError('Expected nonempty [views,heads,queries,keys] probabilities')
    views,heads,queries,keys=attention.shape
    if query_valid.shape!=(views,queries) or key_valid.shape!=(views,keys) or query_valid.dtype!=torch.bool or key_valid.dtype!=torch.bool:
        raise ValueError('Explicit boolean query/key validity required')
    positions=torch.arange(keys,device=attention.device) if patch_positions is None else _integer_positions(patch_positions,keys,attention.device)
    output=torch.empty((views,len(positions)),dtype=torch.float32,device=attention.device)
    tolerance=5e-3 if attention.dtype in (torch.float16,torch.bfloat16) else 2e-5
    for view in range(views):
        a=attention[view]
        if not torch.isfinite(a).all() or ((a<0)|(a>1)).any(): raise ValueError('Invalid probability values')
        valid_q=query_valid[view]
        if not valid_q.any() or not key_valid[view].any(): raise ValueError('Empty allowed query/key set')
        rows=a if valid_q.all() else a[:,valid_q,:]
        sums=rows.sum(-1,dtype=torch.float32)
        if ((sums-1).abs()>tolerance).any(): raise ValueError('Attention rows not normalized')
        received=rows.sum(1,dtype=torch.float32).mean(0)/int(valid_q.sum())
        if (received[~key_valid[view]]!=0).any(): raise ValueError('Masked key has nonzero attention')
        if not key_valid[view,positions].all(): raise ValueError('Invalid patch key selected')
        output[view]=received[positions]
    return output


def projector_token_scores(patch_scores,groups):
    if patch_scores.ndim!=2 or not patch_scores.is_floating_point() or not torch.isfinite(patch_scores).all():
        raise ValueError('Invalid per-view patch scores')
    p=patch_scores.shape[1]
    if groups.ndim!=2 or groups.shape[1]!=16 or groups.dtype not in (torch.int32,torch.int64):
        raise ValueError('Require exact sixteen-patch groups')
    if groups.numel()!=p or groups.unique().numel()!=p or groups.min()<0 or groups.max()>=p:
        raise ValueError('Patch mapping must cover every patch exactly once')
    return patch_scores[:,groups.to(device=patch_scores.device,dtype=torch.long)].mean(-1,dtype=torch.float32)


@dataclass(frozen=True)
class VisionObservation:
    projector_scores: torch.Tensor  # [views,projector_tokens_per_view], reduced only
    attention_shape: tuple[int,...]
    source: str
    backend: str
    layer_index: int
    hook_wall_seconds: float


class VisionAttentionObserver:
    """One final-self_attn observation per context, preserving its return value."""
    def __init__(self,vision,patch_valid,groups,*,grid=(32,32)):
        if vision.training: raise ValueError('Vision must be eval')
        self.vision=vision;self.groups=groups;self.grid=grid
        self.valid=patch_valid;self.layers=vision.encoder.layers
        self.index=len(self.layers)-1;self.module=self.layers[self.index].self_attn
        self.backend=self.module.config._attn_implementation
        count=grid[0]*grid[1]
        emb=vision.embeddings
        if getattr(emb,'num_patches',None)!=count or emb.position_embedding.num_embeddings!=count:
            raise ValueError('BLOCKER: unverified special-token/patch embedding layout')
        if self.valid.ndim!=2 or self.valid.shape[1]!=count or self.valid.dtype!=torch.bool or not self.valid.all():
            raise ValueError('P0.1 production policy supports only equal all-valid patch views')
        if self.module.is_causal or groups.numel()!=count: raise ValueError('Invalid vision/connector contract')
        self.handles=[];self.values={};self.observation=None;self.calls=0;self.events=None

    def __enter__(self):
        if self.handles: raise RuntimeError('Observer context already active')
        def before(module,args,kwargs):
            self.calls+=1
            if self.calls!=1: raise RuntimeError('Only one observation per context')
            h=kwargs.get('hidden_states',args[0] if args else None)
            if h is None or h.shape[:2]!=self.valid.shape or module.config._attn_implementation!=self.backend:
                raise ValueError('BLOCKER: extra/missing token or backend drift')
            self.values['mask']=kwargs.get('attention_mask',args[1] if len(args)>1 else None)
        def projection(name):
            def hook(module,args,output): self.values[name]=output.detach()
            return hook
        def after(module,args,output):
            start=time.perf_counter()
            raw_q=self.values.get('q');raw_k=self.values.get('k')
            if raw_q is None or raw_k is None: raise RuntimeError('Native Q/K hooks unavailable')
            if raw_q.device.type=='cuda':
                self.events=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                self.events[0].record()
            views,patches,_=raw_q.shape
            q=raw_q.view(views,patches,module.num_heads,module.head_dim).transpose(1,2)
            k=raw_k.view(views,patches,module.num_heads,module.head_dim).transpose(1,2)
            attention=output[1] if isinstance(output,tuple) and len(output)>1 else None
            source='native_returned_attention'
            if attention is None:
                attention=reconstruct_attention(q,k,module.scale,self.values['mask'])
                source='actual_native_QK_eager_form_reconstruction_not_fused_internal_weights'
            if tuple(attention.shape)!=(views,module.num_heads,patches,patches):
                raise ValueError('Unexpected attention shape')
            patch_scores=incoming_patch_scores(attention,self.valid,self.valid)
            scores=projector_token_scores(patch_scores,self.groups)
            if self.events: self.events[1].record()
            self.observation=VisionObservation(scores.detach(),tuple(attention.shape),source,
                                                self.backend,self.index,time.perf_counter()-start)
            self.values.clear()  # do not retain full attention, Q/K or activations
            # Returning None preserves the original module output without replacing it.
        try:
            self.handles.append(self.module.register_forward_pre_hook(before,with_kwargs=True))
            self.handles.append(self.module.q_proj.register_forward_hook(projection('q')))
            self.handles.append(self.module.k_proj.register_forward_hook(projection('k')))
            self.handles.append(self.module.register_forward_hook(after))
        except Exception:
            self.__exit__(None,None,None);raise
        return self

    def __exit__(self,exc_type,exc,tb):
        for handle in self.handles: handle.remove()
        self.handles.clear();self.values.clear()
        return False

    def require_observation(self):
        if self.calls!=1 or self.observation is None: raise RuntimeError('BLOCKER: no reliable final-layer attention')
        if self.module.config._attn_implementation!=self.backend: raise RuntimeError('Backend changed')
        return self.observation

    def cuda_observer_ms(self):
        if self.events is None: return None
        self.events[1].synchronize()
        return float(self.events[0].elapsed_time(self.events[1]))


def select_E_from_observation(problem,observation):
    if problem.layout is None: raise ValueError('E requires verified spatial layout')
    views=problem.layout.crop_rows*problem.layout.crop_cols+1
    if tuple(observation.projector_scores.shape)!=(views,64):
        raise ValueError('BLOCKER: score order/count does not match local crops + final thumbnail')
    scores=observation.projector_scores.detach().float().cpu().numpy().reshape(-1).copy()
    if len(scores)!=problem.layout.n or not np.isfinite(scores).all(): raise ValueError('Invalid visual score vector')
    return select_method('E',problem,vision_scores=scores)