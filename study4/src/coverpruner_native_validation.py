"""Validation-only capture of the real unpruned native LlamaAttention.forward.
Not the probe. Native V/output projections run ONLY for this one-layer reference.
No full Decoder layer/FFN/model forward/cache/generation. Patches/hooks restored.
"""
from contextlib import ExitStack
from unittest.mock import patch
import torch
from transformers.models.llama import modeling_llama as native
from coverpruner_attention_probe import native_layout


@torch.no_grad()
def capture_native_first_attention(layer, rotary, embeddings, validity, position_ids):
    attention=layer.self_attn
    if layer.training or attention.config._attn_implementation!='eager':
        raise ValueError('Native reference requires eval/eager')
    pos,mask=native_layout(attention.config,embeddings,validity,position_ids)
    captured={'eager_calls':0,'softmax_calls':0,'position_ids':pos,'mask':mask}
    length=embeddings.shape[1]
    original_eager=native.eager_attention_forward
    original_softmax=native.nn.functional.softmax
    hooks=[]
    def record(name):
        def hook(module,args,output):
            captured[name]=output.detach().clone()
        return hook
    def eager(module,q,k,v,attention_mask,scaling,**kwargs):
        captured['eager_calls']+=1
        captured.update(q_rope=q.detach().clone(),k_rope=k.detach().clone(),
                        native_mask=attention_mask.detach().clone(),scaling=float(scaling))
        return original_eager(module,q,k,v,attention_mask,scaling=scaling,**kwargs)
    def softmax(value,*args,**kwargs):
        captured['softmax_calls']+=1
        captured['pre_softmax']=value.detach().clone()
        result=original_softmax(value,*args,**kwargs)
        captured['softmax_FP32']=result.detach().clone()
        return result
    try:
        hooks=[layer.input_layernorm.register_forward_hook(record('normalized_input')),
               attention.q_proj.register_forward_hook(record('q_projection')),
               attention.k_proj.register_forward_hook(record('k_projection'))]
        with ExitStack() as stack:
            stack.enter_context(patch.object(native,'eager_attention_forward',eager))
            stack.enter_context(patch.object(native.nn.functional,'softmax',softmax))
            normalized=layer.input_layernorm(embeddings)
            cos_sin=rotary(embeddings,position_ids=pos)
            output,weights=attention(normalized,position_embeddings=cos_sin,
                                     attention_mask=mask,past_key_values=None)
        if captured['eager_calls']!=1 or captured['softmax_calls']!=1 or weights is None:
            raise RuntimeError('BLOCKER: trustworthy native first-attention capture unavailable')
        if not torch.isfinite(output).all() or not torch.isfinite(weights).all():
            raise ValueError('Nonfinite native output/attention')
        captured['q_pre_rope']=captured.pop('q_projection').view(1,length,-1,attention.head_dim).transpose(1,2)
        captured['k_pre_rope']=captured.pop('k_projection').view(1,length,-1,attention.head_dim).transpose(1,2)
        captured['position_embeddings']=cos_sin
        captured['attention']=weights.detach()
        return captured
    finally:
        for hook in hooks: hook.remove()
        if native.eager_attention_forward is not original_eager or native.nn.functional.softmax is not original_softmax:
            raise RuntimeError('Native instrumentation restoration failed')