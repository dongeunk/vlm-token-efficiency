"""Synthetic CPU plus tiny native Vision component tests; no pretrained weights."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
import torch
from transformers.models.idefics3.configuration_idefics3 import Idefics3VisionConfig
from transformers.models.idefics3.modeling_idefics3 import (
    Idefics3VisionTransformer,Idefics3Connector,eager_attention_forward)
import coverpruner_vision_attention as va
from coverpruner_spatial_layout import layout_from_p0
from coverpruner_spatial_selection import prepare_problem,select_method
import coverpruner_coverage as engine

METRICS={'structures':[],'tiny_feature_parity':{},'native_connector_all1024_patches_once':False}
CONNECTOR=SimpleNamespace(pixel_shuffle=lambda x,scale_factor:Idefics3Connector.pixel_shuffle(None,x,scale_factor))


def close(a,b,atol=1e-7):
    if a.shape!=b.shape or not torch.allclose(a.float(),b.float(),atol=atol,rtol=0):
        raise AssertionError('Tensor comparison failed')


def tiny(backend):
    config=Idefics3VisionConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=2,
        num_attention_heads=4,image_size=16,patch_size=4,attention_dropout=0.)
    config._attn_implementation=backend
    model=Idefics3VisionTransformer(config).eval()
    pixels=torch.linspace(-1,1,2*3*16*16).reshape(2,3,16,16)
    return model,pixels,torch.ones(2,16,dtype=torch.bool),va.expected_patch_groups((4,4))


class VisionAttentionCPUTests(unittest.TestCase):
    def test_incoming_direction_not_outgoing_row_mean(self):
        attention=torch.tensor([[[[.8,.1,.1],[.6,.3,.1],[.1,.2,.7]]]])
        valid=torch.ones(1,3,dtype=torch.bool)
        result=va.incoming_patch_scores(attention,valid,valid)
        close(result,torch.tensor([[.5,.2,.3]]))
        self.assertFalse(torch.allclose(result,torch.ones(1,3)/3))

    def test_head_and_allowed_query_mean_per_view(self):
        a=torch.tensor([[[[.8,.2],[.1,.9]],[[.4,.6],[.7,.3]]],
                        [[[.2,.8],[.6,.4]],[[.4,.6],[.8,.2]]]])
        qvalid=torch.tensor([[True,False],[True,True]])
        result=va.incoming_patch_scores(a,qvalid,torch.ones(2,2,dtype=torch.bool))
        close(result,torch.tensor([[.6,.4],[.5,.5]]))

    def test_special_tokens_explicit_indices_no_renormalization(self):
        a=torch.ones(1,2,5,5)/5
        query=torch.tensor([[False,True,True,True,True]])
        result=va.incoming_patch_scores(a,query,torch.ones(1,5,dtype=torch.bool),patch_positions=[1,2,3,4])
        close(result,torch.full((1,4),.2)) # CLS key discarded, not renormalized to .25

    def test_padding_queries_excluded_keys_zero(self):
        a=torch.tensor([[[[0.,0.,1.],[.3,.7,0.]]]])
        result=va.incoming_patch_scores(a,torch.tensor([[False,True]]),
                                       torch.tensor([[True,True,False]]),patch_positions=[0,1])
        close(result,torch.tensor([[.3,.7]]))
        with self.assertRaises(ValueError):
            va.incoming_patch_scores(a,torch.ones(1,2,dtype=torch.bool),torch.tensor([[True,True,False]]),patch_positions=[0,1])

    def test_reconstruction_matches_native_eager_math(self):
        q=torch.sin(torch.arange(2*3*8*4).reshape(2,3,8,4).float())
        k=torch.cos(torch.arange(q.numel()).reshape(q.shape).float())
        mask=torch.zeros(2,1,8,8);mask[:,:,:,-1]=-float('inf')
        expected=eager_attention_forward(SimpleNamespace(training=False),q,k,torch.zeros_like(k),mask,scaling=.5)[1]
        close(va.reconstruct_attention(q,k,.5,mask),expected,atol=0.)
        close(va.reconstruct_attention(q,k,.5,mask==0),expected,atol=0.)

    def test_native_pixel_shuffle_exact16_groups(self):
        groups=va.validated_connector_groups(CONNECTOR)
        expected=va.expected_patch_groups()
        self.assertEqual(tuple(groups.shape),(64,16))
        self.assertTrue(torch.equal(groups.sort(-1).values,expected.sort(-1).values))
        self.assertEqual(groups.unique().numel(),1024)
        METRICS['native_connector_all1024_patches_once']=True

    def test_patch_to_projector_known_values(self):
        groups=va.validated_connector_groups(CONNECTOR)
        scores=va.projector_token_scores(torch.arange(1024).float()[None],groups)
        self.assertEqual(scores[0,0],49.5);self.assertEqual(scores[0,1],53.5)
        self.assertEqual(scores[0,8],177.5);self.assertEqual(scores[0,63],973.5)
        for token in range(64):
            v,u=divmod(token,8)
            expected=sum(32*(4*v+a)+(4*u+b) for a in range(4) for b in range(4))/16
            self.assertEqual(float(scores[0,token]),expected)

    def test_view_order_N576_832_1088_E_seed25_greedy39(self):
        groups=va.validated_connector_groups(CONNECTOR)
        for n in (576,832,1088):
            layout=layout_from_p0(n);views=layout.crop_rows*layout.crop_cols+1
            patch_scores=torch.arange(1024).float()[None]+torch.arange(views).float()[:,None]*1000
            reduced=va.projector_token_scores(patch_scores,groups)
            self.assertEqual(tuple(reduced.shape),(views,64))
            self.assertEqual(reduced[1,0]-reduced[0,0],1000)
            self.assertEqual(reduced[-1,0],(views-1)*1000+49.5)
            rng=np.random.default_rng(n+117)
            x=rng.normal(size=(n,8)).astype(np.float32);r=rng.uniform(size=n).astype(np.float32)
            problem=prepare_problem(x,r,layout=layout,sequence_indices=np.arange(n)*4+3)
            observation=va.VisionObservation(reduced,(views,12,1024,1024),'synthetic_test','test',11,0.)
            result=va.select_E_from_observation(problem,observation)
            expected=select_method('E',problem,vision_scores=reduced.numpy().reshape(-1))
            self.assertEqual(result.greedy.selection_order,expected.greedy.selection_order)
            self.assertEqual(result.seed_cell_counts,(1,)*25)
            self.assertEqual(len(set(result.greedy.selection_order)),64)
            self.assertEqual(len(result.greedy.added_indices),39)
            self.assertTrue(all(i>=n-64 for i in result.seed_indices)) # last thumbnail view highest in this fixture
            initial=engine.coverage_value(problem.utilities,problem.weights,result.seed_indices)
            self.assertGreater(initial,0);self.assertEqual(initial,result.greedy.objective_values[0])
            self.assertEqual(result.greedy.sorted_sequence_indices,tuple(sorted(result.greedy.selection_order_sequence_indices)))
            METRICS['structures'].append(dict(N=n,views=views,local_global_order='PASS',seed_cells=25,unique_K=64,added39=True))

    def test_uniform_attention_ties_keep_native_cell_rule(self):
        layout=layout_from_p0(576)
        problem=prepare_problem(np.eye(576,8),np.zeros(576),layout=layout)
        scores=torch.full((9,64),1/1024)
        obs=va.VisionObservation(scores,(9,12,1024,1024),'synthetic_uniform','test',11,0.)
        result=va.select_E_from_observation(problem,obs)
        expected=tuple(int(np.flatnonzero(layout.cell_ids==i)[0]) for i in range(25))
        self.assertEqual(result.seed_indices,expected)
        self.assertEqual(len(set(result.greedy.selection_order)),64)

    def _feature_parity(self,backend):
        model,pixels,valid,groups=tiny(backend)
        before_hooks=[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]
        with torch.no_grad():
            baseline=model(pixels,output_attentions=False).last_hidden_state.clone()
            with va.VisionAttentionObserver(model,valid,groups,grid=(4,4)) as observer:
                self.assertEqual(len(model.encoder.layers[0].self_attn._forward_hooks),0)
                observed=model(pixels,output_attentions=False).last_hidden_state
                obs=observer.require_observation()
                self.assertEqual(obs.layer_index,1)
                self.assertEqual(obs.attention_shape,(2,4,16,16))
                self.assertEqual(tuple(obs.projector_scores.shape),(2,1))
                self.assertEqual(observer.values,{})
            restored=model(pixels,output_attentions=False).last_hidden_state
        self.assertTrue(torch.equal(baseline,observed));self.assertTrue(torch.equal(baseline,restored))
        self.assertEqual(model.config._attn_implementation,backend)
        self.assertEqual(before_hooks,[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()])
        if backend=='eager': self.assertEqual(obs.source,'native_returned_attention')
        else: self.assertIn('native_QK',obs.source)
        METRICS['tiny_feature_parity'][backend]='exact; hooks/backend restored'

    def test_native_eager_tiny_feature_invariance(self): self._feature_parity('eager')
    def test_native_sdpa_tiny_feature_invariance(self): self._feature_parity('sdpa')

    def test_hook_cleanup_on_exception(self):
        model,pixels,valid,groups=tiny('sdpa')
        before=[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()]
        with self.assertRaises(RuntimeError),torch.no_grad(),patch.object(va,'projector_token_scores',side_effect=RuntimeError('test error')):
            with va.VisionAttentionObserver(model,valid,groups,grid=(4,4)):
                model(pixels,output_attentions=False)
        self.assertEqual(before,[(len(m._forward_hooks),len(m._forward_pre_hooks)) for m in model.modules()])
        self.assertEqual(model.config._attn_implementation,'sdpa')

    def test_unknown_CLS_or_padding_contract_blocked(self):
        model,pixels,valid,groups=tiny('eager')
        valid[0,0]=False
        with self.assertRaises(ValueError): va.VisionAttentionObserver(model,valid,groups,grid=(4,4))
        valid[:]=True;model.embeddings.num_patches=17
        with self.assertRaises(ValueError): va.VisionAttentionObserver(model,valid,groups,grid=(4,4))

    def test_invalid_attention_finite_and_probability_rows(self):
        valid=torch.ones(1,2,dtype=torch.bool)
        for value in (float('nan'),float('inf'),-.1,1.1):
            a=torch.ones(1,1,2,2)/2;a[0,0,0,0]=value
            with self.assertRaises(ValueError): va.incoming_patch_scores(a,valid,valid)
        with self.assertRaises(ValueError): va.incoming_patch_scores(torch.zeros(1,1,2,2),valid,valid)
        with self.assertRaises(ValueError): va.incoming_patch_scores(torch.ones(1,1,2,2)/2,~valid,valid)
        with self.assertRaises(ValueError): va.incoming_patch_scores(torch.ones(1,1,2,2)/2,valid,valid,patch_positions=[0,0])

    def test_invalid_reconstruction_masks_QK(self):
        q=torch.ones(1,2,4,2)
        for mask in (torch.ones(1,1,4,4)*float('nan'),torch.full((1,1,4,4),-float('inf')),
                     torch.ones(1,1,1,4,4),torch.zeros(1,1,3,3)):
            with self.assertRaises(ValueError): va.reconstruct_attention(q,q,.5,mask)
        for scale in (0,float('inf')):
            with self.assertRaises(ValueError): va.reconstruct_attention(q,q,scale)
        bad=q.clone();bad[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError): va.reconstruct_attention(bad,q,.5)

    def test_bad_patch_groups_no_truncation(self):
        groups=va.expected_patch_groups()
        bad=groups.clone();bad[0,0]=bad[0,1]
        with self.assertRaises(ValueError): va.projector_token_scores(torch.ones(1,1024),bad)
        with self.assertRaises(ValueError): va.projector_token_scores(torch.ones(1,1025),groups)
        with self.assertRaises(ValueError): va.projector_token_scores(torch.ones(1,1024),groups[:,:15])
        with self.assertRaises(ValueError): va.expected_patch_groups((31,32))

    def test_bridge_rejects_wrong_view_count_and_NaN(self):
        problem=prepare_problem(np.zeros((576,4)),np.zeros(576),layout=layout_from_p0(576))
        for scores in (torch.zeros(8,64),torch.full((9,64),float('nan'))):
            obs=va.VisionObservation(scores,(9,12,1024,1024),'invalid','test',11,0.)
            with self.assertRaises(ValueError): va.select_E_from_observation(problem,obs)