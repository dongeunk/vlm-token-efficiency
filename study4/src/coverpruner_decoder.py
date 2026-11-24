"""Pre-LLM visual compression and compact-position cold-KV greedy decoding.

No selection/probe/vision engine changes. No external cache accepted, no cache
crop/gather/reuse. Probe positions remain original outside this module.
Only unpadded batch1 is currently validated; all nonvisual tokens are kept.
"""
from dataclasses import dataclass
import operator
import time
import torch
from transformers.cache_utils import DynamicCache


def _indices(values,length,device,*,allow_empty=False):
    if isinstance(values,(list,tuple)) and any(isinstance(v,bool) for v in values):
        raise ValueError('Boolean token indices forbidden')
    result=torch.as_tensor(values,device=device)
    if result.ndim!=1 or (not allow_empty and result.numel()==0): raise ValueError('Invalid index vector')
    if result.numel()==0: return result.long()
    if result.dtype not in (torch.int32,torch.int64) or result.min()<0 or result.max()>=length or result.unique().numel()!=result.numel():
        raise ValueError('Duplicate/out-of-range/noninteger indices')
    return result.long()


@dataclass(frozen=True)
class PrunedInput:
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    kept_original_indices: torch.Tensor
    visual_mask: torch.Tensor
    selected_visual_ordinals: tuple[int,...]
    selected_original_positions: tuple[int,...]
    instruction_positions: tuple[int,...]
    original_length: int
    original_visual_count: int
    image_token_id: int

    @property
    def length(self): return int(self.input_ids.shape[1])


def compress_visual_tokens(input_ids,embeddings,validity,visual_positions,selected_ordinals,
                           instruction_positions,*,image_token_id,expected_k=64):
    if input_ids.ndim!=2 or input_ids.shape[0]!=1 or input_ids.dtype!=torch.int64:
        raise ValueError('Batch1 long token IDs required')
    length=input_ids.shape[1]
    if embeddings.ndim!=3 or embeddings.shape[:2]!=input_ids.shape or not embeddings.is_floating_point() or not torch.isfinite(embeddings).all():
        raise ValueError('Invalid original merged embeddings')
    if validity.shape!=input_ids.shape or not (validity==1).all():
        raise ValueError('Only all-valid unpadded batch1 is supported')
    if input_ids.device!=embeddings.device or validity.device!=embeddings.device:
        raise ValueError('Input devices differ')
    visual=_indices(visual_positions,length,embeddings.device)
    native_visual=(input_ids[0]==image_token_id).nonzero().flatten()
    if not torch.equal(visual,native_visual): raise ValueError('Visual positions differ from native placeholder positions/order')
    selected=_indices(selected_ordinals,len(visual),embeddings.device)
    if isinstance(expected_k,bool) or operator.index(expected_k)!=len(selected):
        raise ValueError('Selection does not match required K')
    queries=_indices(instruction_positions,length,embeddings.device)
    if torch.isin(queries,visual).any(): raise ValueError('Instruction overlaps visual positions')
    keep=torch.ones(length,dtype=torch.bool,device=embeddings.device)
    keep[visual]=False;keep[visual[selected]]=True
    kept=keep.nonzero().flatten()
    ids=input_ids.index_select(1,kept).clone()
    hidden=embeddings.index_select(1,kept).clone()
    visual_mask=(ids[0]==image_token_id)
    if int(visual_mask.sum())!=len(selected) or len(kept)!=length-len(visual)+len(selected):
        raise RuntimeError('Visual removal/compact length invariant failed')
    old_to_new=torch.full((length,),-1,dtype=torch.long,device=embeddings.device)
    old_to_new[kept]=torch.arange(len(kept),device=embeddings.device)
    remapped=old_to_new[queries]
    if (remapped<0).any(): raise RuntimeError('Instruction token lost')
    nonvisual=torch.ones(length,dtype=torch.bool,device=embeddings.device);nonvisual[visual]=False
    if not torch.equal(ids[:,~visual_mask],input_ids[:,nonvisual]) or not torch.equal(hidden[:,~visual_mask],embeddings[:,nonvisual]):
        raise RuntimeError('Nonvisual token/embedding changed')
    compact_positions=torch.arange(len(kept),device=embeddings.device,dtype=torch.long)[None]
    selected_original=visual[selected].sort().values
    return PrunedInput(ids,hidden,torch.ones_like(ids),compact_positions,kept,visual_mask,
        tuple(int(i) for i in selected.sort().values.cpu()),tuple(int(i) for i in selected_original.cpu()),
        tuple(int(i) for i in remapped.cpu()),length,len(visual),int(image_token_id))


def validate_compact_input(packet):
    n=packet.length
    if packet.input_ids.dtype!=torch.int64 or packet.visual_mask.dtype!=torch.bool:
        raise ValueError('Invalid packet token/mask dtype')
    if packet.input_ids.shape!=(1,n) or packet.inputs_embeds.shape[:2]!=(1,n) or n==0:
        raise ValueError('Invalid compact packet')
    if packet.attention_mask.shape!=(1,n) or not (packet.attention_mask==1).all(): raise ValueError('Invalid compact attention mask')
    expected=torch.arange(n,device=packet.position_ids.device,dtype=torch.long)[None]
    if packet.position_ids.dtype!=torch.int64 or not torch.equal(packet.position_ids,expected):
        raise ValueError('Decoder positions must be compact, never original-gap')
    if not torch.isfinite(packet.inputs_embeds).all() or packet.visual_mask.shape!=(n,):
        raise ValueError('Invalid compact embeddings/visual mask')
    if int(packet.visual_mask.sum())!=len(packet.selected_visual_ordinals) or not torch.equal(packet.visual_mask,packet.input_ids[0]==packet.image_token_id):
        raise ValueError('Visual count/mask mismatch')
    if n!=packet.original_length-packet.original_visual_count+len(packet.selected_visual_ordinals):
        raise ValueError('Compact length formula mismatch')
    _indices(packet.instruction_positions,n,packet.input_ids.device)
    if packet.visual_mask[list(packet.instruction_positions)].any(): raise ValueError('Question became visual')


def cache_lengths(cache,layer_count):
    return tuple(int(cache.get_seq_length(i)) for i in range(layer_count))


def _sync(tensor):
    if tensor.device.type=='cuda': torch.cuda.synchronize(tensor.device)


@dataclass
class ColdKVSession:
    cache: DynamicCache
    next_logits: torch.Tensor
    prompt_length: int
    trace: list
    prefill_seconds: float
    decode_seconds: float=0.0


@torch.no_grad()
def cold_prefill(text_model,lm_head,packet):
    validate_compact_input(packet)
    if text_model.training or lm_head.training: raise ValueError('Eval mode required')
    count=text_model.config.num_hidden_layers
    cache=DynamicCache(config=text_model.config) # ALWAYS new, no caller cache argument
    before=cache_lengths(cache,count)
    if any(before): raise RuntimeError('Cold cache is not empty')
    _sync(packet.inputs_embeds);started=time.perf_counter()
    output=text_model(inputs_embeds=packet.inputs_embeds,attention_mask=packet.attention_mask,
        position_ids=packet.position_ids,past_key_values=cache,use_cache=True,
        output_attentions=False,output_hidden_states=False,return_dict=True)
    logits=lm_head(output.last_hidden_state[:,-1:,:])
    _sync(logits);elapsed=time.perf_counter()-started
    after=cache_lengths(cache,count)
    if output.past_key_values is not cache or after!=(packet.length,)*count:
        raise RuntimeError('Prefill cache identity/length mismatch')
    if not torch.isfinite(output.last_hidden_state).all() or not torch.isfinite(logits).all():
        raise ValueError('Nonfinite prefill output')
    trace=[dict(operation='prefill',input_length=packet.length,position_start=0,position_end=packet.length-1,
                mask_length=packet.length,cache_before=0,cache_after=packet.length,
                all_layers_equal=True,visual_tokens=int(packet.visual_mask.sum()))]
    return ColdKVSession(cache,logits,packet.length,trace,elapsed)


@torch.no_grad()
def decode_append(text_model,lm_head,session,token_id):
    count=text_model.config.num_hidden_layers
    lengths=cache_lengths(session.cache,count)
    before=lengths[0]
    if lengths!=(before,)*count or before<session.prompt_length:
        raise RuntimeError('Decode cache is inconsistent')
    device=session.next_logits.device
    token=torch.tensor([[operator.index(token_id)]],dtype=torch.long,device=device)
    if token.item()<0 or token.item()>=text_model.config.vocab_size: raise ValueError('Decode token out of vocabulary')
    positions=torch.tensor([[before]],device=device,dtype=torch.long)
    mask=torch.ones((1,before+1),device=device,dtype=torch.long)
    _sync(session.next_logits);started=time.perf_counter()
    output=text_model(input_ids=token,attention_mask=mask,position_ids=positions,
        past_key_values=session.cache,use_cache=True,output_attentions=False,output_hidden_states=False,return_dict=True)
    logits=lm_head(output.last_hidden_state[:,-1:,:]);_sync(logits)
    session.decode_seconds+=time.perf_counter()-started
    after=cache_lengths(session.cache,count)
    if output.past_key_values is not session.cache or after!=(before+1,)*count:
        raise RuntimeError('Decode cache did not append one token to every layer')
    if not torch.isfinite(output.last_hidden_state).all() or not torch.isfinite(logits).all():
        raise ValueError('Nonfinite decode output')
    session.next_logits=logits
    session.trace.append(dict(operation='decode',input_length=1,position_start=before,position_end=before,
        mask_length=before+1,cache_before=before,cache_after=before+1,all_layers_equal=True))
    return logits


@dataclass
class GenerationResult:
    token_ids: tuple[int,...]
    session: ColdKVSession
    prediction_logits: torch.Tensor  # RAM only for parity, never dump to disk
    stopped_on_eos: bool


@torch.no_grad()
def greedy_decode(text_model,lm_head,packet,*,max_new_tokens=10,eos_token_ids=()):
    eos=set(int(x) for x in eos_token_ids)
    if isinstance(max_new_tokens,bool): raise ValueError('Invalid generation budget')
    try: max_new_tokens=operator.index(max_new_tokens)
    except TypeError as exc: raise ValueError('Generation budget must be integer') from exc
    if not 1<=max_new_tokens<=10: raise ValueError('This smoke wrapper permits1..10 tokens only')
    session=cold_prefill(text_model,lm_head,packet)
    generated=[];predictions=[];stopped=False
    for step in range(max_new_tokens):
        predictions.append(session.next_logits.detach().clone())
        token=int(session.next_logits[0,-1].argmax())
        generated.append(token)
        if token in eos: stopped=True;break
        if step+1<max_new_tokens: decode_append(text_model,lm_head,session,token)
    if cache_lengths(session.cache,text_model.config.num_hidden_layers)!=(packet.length+len(generated)-1,)*text_model.config.num_hidden_layers:
        raise RuntimeError('Final cache length differs from consumed generation tokens')
    return GenerationResult(tuple(generated),session,torch.cat(predictions,dim=1),stopped)