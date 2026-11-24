"""CPU compression/actual tiny native Idefics decoder tests, no pretrained weights."""
import unittest
import torch
from transformers import Idefics3Config,LlamaConfig,GenerationConfig
from transformers.models.idefics3.configuration_idefics3 import Idefics3VisionConfig
from transformers.models.idefics3.modeling_idefics3 import Idefics3ForConditionalGeneration
from transformers.cache_utils import DynamicCache
from coverpruner_decoder import (compress_visual_tokens,validate_compact_input,cache_lengths,
                                 cold_prefill,decode_append,greedy_decode)

METRICS={'native_prefill_parity':None,'teacher_forced_id_parity':None,'actual_question_mapping':None}


def fixture():
    text=LlamaConfig(vocab_size=64,hidden_size=24,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=6,num_key_value_heads=2,head_dim=4,max_position_embeddings=1024,
        pad_token_id=0,bos_token_id=1,eos_token_id=2)
    vision=Idefics3VisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,
        num_attention_heads=4,image_size=16,patch_size=4)
    cfg=Idefics3Config(text_config=text,vision_config=vision,scale_factor=4,image_token_id=63)
    cfg._attn_implementation='eager'
    model=Idefics3ForConditionalGeneration(cfg).eval()
    with torch.no_grad():
        for name,p in model.named_parameters():
            p.copy_(torch.ones_like(p) if name.endswith('norm.weight') or 'layernorm.weight' in name
                    else torch.sin(torch.arange(p.numel()).reshape(p.shape).float())*.03)
    ids=(torch.arange(96)%60)[None].long();ids[0,2:82]=63
    visual=tuple(range(2,82));query=tuple(range(89,93))
    with torch.no_grad(): merged=model.get_input_embeddings()(ids).clone()
    merged[0,2:82]=torch.cos(torch.arange(80*24).reshape(80,24).float())*.2
    selected=tuple(range(79,15,-1)) # deliberately reverse;64 unique
    packet=compress_visual_tokens(ids,merged,torch.ones_like(ids),visual,selected,query,image_token_id=63)
    return model,ids,merged,visual,selected,query,packet


def independent_reference(ids,hidden,visual,selected):
    chosen={visual[j] for j in selected};visual_set=set(visual)
    kept=[j for j in range(ids.shape[1]) if j not in visual_set or j in chosen]
    return torch.stack([hidden[0,j] for j in kept])[None],ids[:,kept],torch.arange(len(kept))[None],torch.ones(1,len(kept),dtype=torch.long)


class DecoderCPUTests(unittest.TestCase):
    def test_K64_physical_compression_and_sorted_order(self):
        _,ids,hidden,visual,selected,query,p=fixture()
        self.assertEqual(p.length,80);self.assertEqual(p.original_visual_count,80)
        self.assertEqual(int(p.visual_mask.sum()),64)
        self.assertEqual(p.selected_visual_ordinals,tuple(range(16,80)))
        self.assertEqual(p.selected_original_positions,tuple(range(18,82)))
        self.assertTrue(torch.equal(p.kept_original_indices,p.kept_original_indices.sort().values))
        ref,refids,_,_=independent_reference(ids,hidden,visual,selected)
        self.assertTrue(torch.equal(p.inputs_embeds,ref));self.assertTrue(torch.equal(p.input_ids,refids))

    def test_all_nonvisual_tokens_and_embedding_values_preserved(self):
        _,ids,hidden,visual,_,_,p=fixture()
        nv=ids[0]!=63
        self.assertTrue(torch.equal(p.input_ids[:,~p.visual_mask],ids[:,nv]))
        self.assertTrue(torch.equal(p.inputs_embeds[:,~p.visual_mask],hidden[:,nv]))
        self.assertEqual(p.instruction_positions,(73,74,75,76))
        self.assertTrue(torch.equal(p.input_ids[:,list(p.instruction_positions)],ids[:,89:93]))

    def test_compact_positions_mask_lengths_and_native_causality(self):
        model,_,_,_,_,_,p=fixture()
        self.assertTrue(torch.equal(p.position_ids,torch.arange(80)[None]))
        self.assertEqual(p.attention_mask.shape,(1,80));self.assertTrue((p.attention_mask==1).all())
        seen=[]
        def inspect(module,args,kwargs):
            mask=kwargs['attention_mask'];positions=kwargs['position_ids']
            self.assertTrue(torch.equal(positions,p.position_ids))
            if mask is not None:
                self.assertEqual(mask.shape[-2:],(80,80))
                self.assertTrue((mask[0,0].tril()==0).all())
                self.assertTrue((mask[0,0].triu(1)[torch.ones(80,80,dtype=torch.bool).triu(1)]<0).all())
            seen.append(args[0].shape[1])
        h=model.model.text_model.layers[0].register_forward_pre_hook(inspect,with_kwargs=True)
        try: cold_prefill(model.model.text_model,model.lm_head,p)
        finally: h.remove()
        self.assertEqual(seen,[80])

    def test_cold_cache_all_layers_empty_then_prefill_decode_append(self):
        model,_,_,_,_,_,p=fixture();text=model.model.text_model
        empty=DynamicCache(config=text.config);self.assertEqual(cache_lengths(empty,2),(0,0))
        session=cold_prefill(text,model.lm_head,p)
        self.assertIsNot(session.cache,empty);self.assertEqual(cache_lengths(empty,2),(0,0))
        self.assertEqual(cache_lengths(session.cache,2),(80,80))
        decode_append(text,model.lm_head,session,5)
        self.assertEqual(cache_lengths(session.cache,2),(81,81))
        decode_append(text,model.lm_head,session,6)
        self.assertEqual(cache_lengths(session.cache,2),(82,82))
        self.assertEqual([(s['cache_before'],s['cache_after']) for s in session.trace],[(0,80),(80,81),(81,82)])
        self.assertEqual(session.trace[-1]['position_start'],81)
        self.assertEqual(session.trace[-1]['mask_length'],82)

    def test_cannot_supply_or_reuse_external_full_cache(self):
        model,_,_,_,_,_,p=fixture();text=model.model.text_model
        with self.assertRaises(TypeError): cold_prefill(text,model.lm_head,p,past_key_values=DynamicCache(config=text.config))
        first=cold_prefill(text,model.lm_head,p);second=cold_prefill(text,model.lm_head,p)
        self.assertIsNot(first.cache,second.cache)
        self.assertEqual(first.trace[0]['cache_before'],0);self.assertEqual(second.trace[0]['cache_before'],0)

    def test_native_pruned_prefill_reference_same_inputs_logits(self):
        model,ids,hidden,visual,selected,_,p=fixture()
        ref,_,pos,mask=independent_reference(ids,hidden,visual,selected)
        session=cold_prefill(model.model.text_model,model.lm_head,p)
        with torch.no_grad(): native=model(inputs_embeds=ref,position_ids=pos,attention_mask=mask,
            past_key_values=None,use_cache=False,logits_to_keep=1,output_attentions=False,return_dict=True)
        error=float((native.logits-session.next_logits).abs().max())
        self.assertLessEqual(error,3e-6);self.assertIsNone(native.past_key_values)
        self.assertTrue(torch.equal(native.logits.argmax(-1),session.next_logits.argmax(-1)))
        METRICS['native_prefill_parity']=dict(max_abs=error,exact_inputs=True)

    def test_teacher_forced_native_trace_ID_check(self):
        model,_,_,_,_,_,p=fixture()
        result=greedy_decode(model.model.text_model,model.lm_head,p,max_new_tokens=3,eos_token_ids=())
        ids=torch.tensor([result.token_ids[:-1]],dtype=torch.long)
        with torch.no_grad():
            context=torch.cat((p.inputs_embeds,model.get_input_embeddings()(ids)),dim=1)
            n=context.shape[1]
            native=model(inputs_embeds=context,attention_mask=torch.ones(1,n,dtype=torch.long),
                position_ids=torch.arange(n)[None],use_cache=False,
                logits_to_keep=torch.arange(p.length-1,n),output_attentions=False,return_dict=True)
        self.assertTrue(torch.allclose(native.logits,result.prediction_logits,atol=3e-6,rtol=0))
        self.assertEqual(tuple(int(x) for x in native.logits[0].argmax(-1)),result.token_ids)
        self.assertEqual(cache_lengths(result.session.cache,2),(82,82))
        METRICS['teacher_forced_id_parity']='3/3; no extra autoregressive generation'

    def test_native_generate_accepts_full_merged_embeddings_and_ids(self):
        model,ids,hidden,_,_,_,_=fixture()
        cfg=GenerationConfig(max_new_tokens=2,do_sample=False,num_beams=1,use_cache=True,
            eos_token_id=None,pad_token_id=0,bos_token_id=1,return_dict_in_generate=True)
        with torch.no_grad(): output=model.generate(input_ids=ids,inputs_embeds=hidden,
            attention_mask=torch.ones_like(ids),generation_config=cfg)
        self.assertEqual(output.sequences.shape,(1,98))
        self.assertTrue(torch.equal(output.sequences[:,:96],ids))
        self.assertEqual(int(output.past_key_values.get_seq_length()),97)

    def test_K_equals_N_identity(self):
        _,ids,hidden,visual,_,query,_=fixture()
        p=compress_visual_tokens(ids,hidden,torch.ones_like(ids),visual,range(80),query,image_token_id=63,expected_k=80)
        self.assertTrue(torch.equal(p.inputs_embeds,hidden));self.assertTrue(torch.equal(p.input_ids,ids))
        self.assertTrue(torch.equal(p.position_ids,torch.arange(96)[None]))


    def test_mixed_gap_positions_and_incorrect_masks_rejected(self):
        model,_,_,_,_,_,p=fixture()
        p.position_ids[0,-1]=95
        with self.assertRaises(ValueError): cold_prefill(model.model.text_model,model.lm_head,p)
        model,_,_,_,_,_,p=fixture();p.attention_mask[0,-1]=0
        with self.assertRaises(ValueError): cold_prefill(model.model.text_model,model.lm_head,p)

    def test_selection_boundary_invalid_inputs_and_no_unmasked_holes(self):
        _,ids,hidden,visual,selected,query,p=fixture()
        for bad in (selected[:-1],(*selected[:-1],selected[0]),[-1]*64,[80]*64,[True]*64):
            with self.assertRaises(ValueError): compress_visual_tokens(ids,hidden,torch.ones_like(ids),visual,bad,query,image_token_id=63)
        for value in (float('nan'),float('inf')):
            bad=hidden.clone();bad[0,0,0]=value
            with self.assertRaises(ValueError): compress_visual_tokens(ids,bad,torch.ones_like(ids),visual,selected,query,image_token_id=63)
        padded=torch.ones_like(ids);padded[0,-1]=0
        with self.assertRaises(ValueError): compress_visual_tokens(ids,hidden,padded,visual,selected,query,image_token_id=63)
        self.assertEqual(int((p.input_ids==63).sum()),64)
        self.assertTrue((p.attention_mask==1).all())

    def test_eos_and_max_budget_cache_semantics(self):
        model,_,_,_,_,_,p=fixture();text=model.model.text_model
        first=cold_prefill(text,model.lm_head,p);eos=int(first.next_logits[0,-1].argmax())
        result=greedy_decode(text,model.lm_head,p,max_new_tokens=10,eos_token_ids=[eos])
        self.assertEqual(len(result.token_ids),1);self.assertTrue(result.stopped_on_eos)
        self.assertEqual(cache_lengths(result.session.cache,2),(80,80))
        for count in (0,-1,11,True,1.5):
            with self.assertRaises(ValueError): greedy_decode(text,model.lm_head,p,max_new_tokens=count)