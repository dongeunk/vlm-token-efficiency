"""Synthetic torch CPU/native-component and one existing Processor input gate."""
import math
import unittest
from unittest.mock import patch
import numpy as np
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding, repeat_kv
from coverpruner_attention_probe import (
    remap_body_span, extract_positions, instruction_span, prepare_processor_input,
    native_layout, query_attention, relevance_from_attention, first_attention_probe,
)
from coverpruner_coverage import demand_weights

PROCESSOR_EVIDENCE = {}


def fixture():
    config = LlamaConfig(hidden_size=24, intermediate_size=32, num_attention_heads=6,
                         num_key_value_heads=2, head_dim=4, num_hidden_layers=1, pad_token_id=0)
    config._attn_implementation = 'eager'
    layer = LlamaDecoderLayer(config, 0).eval()
    with torch.no_grad():
        for name, p in layer.named_parameters():
            p.copy_(torch.ones_like(p) if 'layernorm.weight' in name else
                    torch.sin(torch.arange(p.numel()).reshape(p.shape).float()) * .1)
    rotary = LlamaRotaryEmbedding(config).eval()
    x = torch.sin(torch.arange(240).reshape(1, 10, 24).float())
    valid = torch.ones((1, 10), dtype=torch.long)
    positions, mask = native_layout(config, x, valid)
    return config, layer, rotary, x, valid, positions, mask


def close(a, b, atol=1e-7):
    if a.shape != b.shape or not torch.allclose(a.float(), b.float(), atol=atol, rtol=0):
        raise AssertionError('Tensor mismatch, maxabs=' + str(float((a.float()-b.float()).abs().max())))


class AttentionCPUTests(unittest.TestCase):
    def test_QK_projection_native_dimensions(self):
        _, layer, rotary, x, valid, pos, mask = fixture()
        p = first_attention_probe(layer, rotary, x, valid, pos, mask, [8, 9], [1, 2, 3])
        self.assertEqual(tuple(p.q_pre_rope.shape), (1, 6, 10, 4))
        self.assertEqual(tuple(p.k_pre_rope.shape), (1, 2, 10, 4))
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + layer.input_layernorm.variance_epsilon)
        close(p.normalized_input, norm)
        expected = torch.nn.functional.linear(norm, layer.self_attn.q_proj.weight).view(1,10,6,4).transpose(1,2)
        close(p.q_pre_rope, expected)

    def test_GQA_repeat_mapping(self):
        k = torch.arange(8).reshape(1,2,1,4).float()
        repeated = repeat_kv(k, 3)
        for head in range(6):
            close(repeated[:,head], k[:,head//3])

    def test_native_RoPE_noncompact_positions(self):
        _, layer, rotary, x, valid, _, mask = fixture()
        pos = torch.tensor([[0,2,5,7,9,11,13,15,17,19]])
        p = first_attention_probe(layer, rotary, x, valid, pos, mask, [8,9], [1,2,3])
        cos, sin = p.position_embeddings
        q = p.q_pre_rope
        rotated = torch.cat((-q[...,2:], q[...,:2]), dim=-1)
        close(p.q_rope, q*cos[:,None] + rotated*sin[:,None])
        other = first_attention_probe(layer, rotary, x, valid,
                                      torch.arange(10)[None], mask, [8,9], [1,2,3])
        self.assertFalse(torch.equal(p.q_rope, other.q_rope))

    def test_native_causal_padding_mask(self):
        cfg, _, _, x, valid, _, _ = fixture()
        valid[0,2] = 0
        _, mask = native_layout(cfg, x, valid)
        for query in range(10):
            for key in range(10):
                self.assertEqual(bool(mask[0,0,query,key] == 0), key <= query and key != 2)

    def test_scaled_logits_manual(self):
        q = torch.arange(24).reshape(1,2,6,2).float()/10
        k = torch.arange(12).reshape(1,1,6,2).float()/10
        mask = torch.where(torch.ones(6,6).tril().bool(), 0., -float('inf'))[None,None]
        p = query_attention(q,k,mask,[4,5],[0,1],1/math.sqrt(2))
        expected = torch.tensor([sum(float(q[0,h,t,d])*float(k[0,0,j,d]) for d in range(2))/math.sqrt(2)
                                 for h in range(2) for t in (4,5) for j in range(6)]).reshape(1,2,2,6)
        close(p.scaled_logits, expected, 1e-6)

    def test_CPU_native_attention_parity(self):
        _, layer, rotary, x, valid, pos, mask = fixture()
        p = first_attention_probe(layer, rotary, x, valid, pos, mask, [8,9], [1,2,3])
        with torch.no_grad():
            _, attention = layer.self_attn(layer.input_layernorm(x),
                position_embeddings=rotary(x, position_ids=pos), attention_mask=mask)
        self.assertIsNotNone(attention)
        close(p.attention.probabilities, attention[:,:,[8,9],:], atol=3e-7)

    def test_domains_have_different_scores_and_ranks(self):
        q, k = torch.zeros(1,2,6,2), torch.zeros(1,1,6,2)
        q[0,0,4,0] = 1; q[0,1,4,1] = 1
        k[0,0] = torch.tensor([[3,0],[0,2],[4,0],[4,0],[4,0],[0,0]])
        mask = torch.where(torch.ones(6,6).tril().bool(), 0., -float('inf'))[None,None]
        full = query_attention(q,k,mask,[4],[0,1],1)
        visual = query_attention(q,k,mask,[4],[0,1],1,domain='visual_only')
        mass = full.visual_probabilities.sum(-1, keepdim=True)
        close(full.visual_probabilities, visual.visual_probabilities*mass)
        self.assertEqual(int(relevance_from_attention(full.visual_probabilities).argmax()),1)
        self.assertEqual(int(relevance_from_attention(visual.visual_probabilities).argmax()),0)
        close(full.probabilities.sum(-1), torch.ones(1,2,1))
        close(visual.visual_probabilities.sum(-1), torch.ones(1,2,1))

    def test_future_visual_key_has_zero_probability(self):
        _, layer, rotary, x, valid, pos, mask = fixture()
        p = first_attention_probe(layer,rotary,x,valid,pos,mask,[5],[1,8])
        self.assertEqual(p.attention.allowed_visual_counts,(1,))
        self.assertTrue((p.attention.visual_probabilities[...,1] == 0).all())

    def test_no_causal_visual_access_is_failure(self):
        _, layer, rotary, x, valid, pos, mask = fixture()
        with self.assertRaises(ValueError):
            first_attention_probe(layer,rotary,x,valid,pos,mask,[0],[2,3])

    def test_head_then_instruction_mean_and_weights(self):
        a = torch.tensor([[[[.1,.2],[.3,.4]],[[.5,.6],[.7,.8]]]])
        close(relevance_from_attention(a),torch.tensor([.4,.5]))
        _, layer, rotary, x, valid, pos, mask = fixture()
        p = first_attention_probe(layer,rotary,x,valid,pos,mask,[8,9],[1,2,3])
        self.assertTrue(np.array_equal(p.weights,demand_weights(p.relevance)))
        self.assertTrue((p.weights>=1).all() and (p.weights<=np.exp(np.float32(.3))).all())

    def test_probe_does_not_run_V_output_FFN_or_layer(self):
        _, layer, rotary, x, valid, pos, mask = fixture()
        def prohibited(*a,**k): raise AssertionError('Forbidden full-layer component')
        with patch.object(layer,'forward',prohibited), patch.object(layer.mlp,'forward',prohibited), \
             patch.object(layer.self_attn.v_proj,'forward',prohibited), patch.object(layer.self_attn.o_proj,'forward',prohibited):
            first_attention_probe(layer,rotary,x,valid,pos,mask,[8],[1,2])

    def test_deterministic_finite_and_equal_relevance(self):
        _, layer, rotary, x, valid, pos, mask = fixture()
        a = first_attention_probe(layer,rotary,x,valid,pos,mask,[8,9],[1,2,3])
        b = first_attention_probe(layer,rotary,x,valid,pos,mask,[8,9],[1,2,3])
        self.assertTrue(np.array_equal(a.relevance,b.relevance))
        self.assertTrue(np.isfinite(a.relevance).all())
        self.assertTrue(np.array_equal(demand_weights([0,0]),[1,1]))

    def test_replacement_and_instruction_offsets(self):
        replacements = [{'span':(3,10),'new_span':(3,17)}, {'span':(13,20),'new_span':(20,34)}]
        self.assertEqual(remap_body_span((20,22),replacements),(34,36))
        ids=[1,99,99,2,10,11,3]
        offsets=[(0,1),(1,2),(2,3),(3,4),(4,6),(6,7),(7,8)]
        self.assertEqual(extract_positions(ids,offsets,[1]*7,99,{1,2,3,99},(4,7)),((1,2),(4,5)))
        with self.assertRaises(ValueError): extract_positions(ids,offsets,[1]*7,99,{1,2,3,99},(5,7))
        with self.assertRaises(ValueError): remap_body_span((4,6),replacements)

    def test_bad_QK_mask_domain_and_indices(self):
        q,k=torch.zeros(1,2,6,2),torch.zeros(1,1,6,2)
        mask=torch.where(torch.ones(6,6).tril().bool(),0.,-float('inf'))[None,None]
        for instruction,visual in (([],[0]),([4,4],[0]),([4],[4]),([6],[0]),([True],[0])):
            with self.subTest(indices=(instruction,visual)),self.assertRaises(ValueError):
                query_attention(q,k,mask,instruction,visual,1)
        for badmask in (torch.zeros(6,6),torch.full_like(mask,float('nan')),torch.ones_like(mask)):
            with self.assertRaises(ValueError): query_attention(q,k,badmask,[4],[0],1)
        for scale,domain in ((float('nan'),'full_prefix'),(0,'full_prefix'),(1,'unknown')):
            with self.assertRaises(ValueError): query_attention(q,k,mask,[4],[0],scale,domain=domain)
        for value in (float('nan'),float('inf')):
            bad=q.clone();bad[0,0,0,0]=value
            with self.assertRaises(ValueError): query_attention(bad,k,mask,[4],[0],1)
        with self.assertRaises(ValueError): query_attention(q[:,:0],k,mask,[4],[0],1)

    def test_invalid_positions_padding_training(self):
        cfg,layer,rotary,x,valid,pos,mask=fixture()
        with self.assertRaises(ValueError): native_layout(cfg,x,valid,pos.float())
        with self.assertRaises(ValueError): native_layout(cfg,x,valid,-torch.ones_like(pos))
        valid[0,8]=0
        with self.assertRaises(ValueError): first_attention_probe(layer,rotary,x,valid,pos,mask,[8],[1])
        layer.train()
        with self.assertRaises(ValueError): first_attention_probe(layer,rotary,x,valid,pos,mask,[9],[1])
        with self.assertRaises(ValueError): relevance_from_attention(torch.full((1,2,1,1),float('nan')))

    def test_native_capture_instrumentation_and_restoration(self):
        from coverpruner_native_validation import capture_native_first_attention
        from transformers.models.llama import modeling_llama as native
        cfg,layer,rotary,x,valid,pos,mask=fixture()
        old_eager,old_softmax=native.eager_attention_forward,native.nn.functional.softmax
        p=first_attention_probe(layer,rotary,x,valid,pos,mask,[8,9],[1,2,3])
        captured=capture_native_first_attention(layer,rotary,x,valid,pos)
        self.assertIs(native.eager_attention_forward,old_eager)
        self.assertIs(native.nn.functional.softmax,old_softmax)
        self.assertEqual(captured['eager_calls'],1)
        self.assertEqual(captured['softmax_calls'],1)
        self.assertEqual(captured['scaling'],.5)
        self.assertTrue(torch.equal(captured['q_pre_rope'],p.q_pre_rope))
        self.assertTrue(torch.equal(captured['q_rope'],p.q_rope))
        self.assertTrue(torch.equal(captured['native_mask'],mask))
        close(captured['pre_softmax'][:,:,[8,9]],p.attention.scaled_logits+p.attention.query_mask,atol=5e-7)
        close(captured['attention'][:,:,[8,9]],p.attention.probabilities,atol=3e-7)