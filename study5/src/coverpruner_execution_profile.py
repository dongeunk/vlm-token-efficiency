"""Strict evaluation contract; padding diagnostic support is a separate decision."""
import torch
from coverpruner_decoder import validate_compact_input

def validate_evaluation_packet(packet,backend,*,pad_token_id=None):
    validate_compact_input(packet)
    if packet.input_ids.shape[0]!=1 or packet.inputs_embeds.dtype!=torch.float16:raise ValueError('Batch1 FP16 required')
    if backend!='sdpa':raise ValueError('Frozen evaluation backend is sdpa')
    if not (packet.attention_mask==1).all():raise ValueError('Evaluation padding forbidden')
    # pad ID may also be a meaningful EOS. Validity/provenance, not ID equality, defines padding.
    return dict(profile='evaluation_batch1_no_padding',backend=backend,dtype='float16',batch_size=1,
        padding_rows=0,input_length=packet.length,visual_tokens=int(packet.visual_mask.sum()),nonvisual_tokens=packet.length-int(packet.visual_mask.sum()),
        position_range=[0,packet.length-1],cold_cache_required=True,greedy=True)