from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

import coverpruner_coverage as engine
import coverpruner_spatial_selection as baseline
from coverpruner_decoder import compress_visual_tokens
from coverpruner_spatial_layout import layout_from_p0
from study5_selection import select_all

CASES = {}


def case(n):
    if n not in CASES:
        rng = np.random.default_rng(n)
        layout = layout_from_p0(n)
        scores = rng.integers(0,16,n).astype(np.float32)
        one_cell = np.flatnonzero(layout.cell_ids==12)[:25]
        assert len(one_cell)==25
        scores[one_cell] = 20
        positions = tuple(range(3,n+3))
        problem = baseline.prepare_problem(rng.normal(size=(n,12)),scores,layout=layout,sequence_indices=positions)
        CASES[n] = (problem,select_all(problem),tuple(map(int,one_cell)))
    return CASES[n]


class Study5Tests(unittest.TestCase):
    def test_shared_raw_score_object(self):
        p,_,_=case(576)
        with patch.object(baseline,"maximum_score_seeds",wraps=baseline.maximum_score_seeds) as seed:
            outputs=select_all(p)
        self.assertEqual(seed.call_count,1)
        self.assertIs(seed.call_args.args[0],p.query_relevance)
        rank=sorted(range(576),key=lambda j:(-p.query_relevance[j],j))
        self.assertEqual(outputs["G"].seed_indices,tuple(rank[:25]))
        self.assertEqual(outputs["H"].selection_order,tuple(rank[:64]))

    def test_global_methods_unconstrained_and_ties(self):
        for n in (576,832,1088):
            with self.subTest(n=n):
                p,out,one_cell=case(n)
                self.assertEqual(out["G"].seed_indices,one_cell)
                self.assertEqual(out["H"].selection_order[:25],one_cell)
                self.assertEqual(out["G"].seed_cell_counts[12],25)
                self.assertEqual(sum(v>0 for v in out["G"].seed_cell_counts),1)
                rank=tuple(sorted(range(n),key=lambda j:(-p.query_relevance[j],j)))
                self.assertEqual(out["H"].selection_order,rank[:64])

    def test_i_preserves_exact_d_seed_and_query_completion(self):
        for n in (576,832,1088):
            p,out,_=case(n)
            d,i=out["D"],out["I"]
            self.assertEqual(i.seed_indices,d.seed_indices)
            self.assertEqual(i.seed_cell_counts,(1,)*25)
            rank=sorted(set(range(n))-set(d.seed_indices),key=lambda j:(-p.query_relevance[j],j))
            self.assertEqual(i.selection_order[25:],tuple(rank[:39]))
            self.assertFalse(i.greedy_objectives)

    def test_seed_coverage_oracle_each_greedy_step(self):
        for n in (576,832,1088):
            p,out,_=case(n)
            for arm in ("D","G"):
                r=out[arm]
                used=np.zeros(n,dtype=bool)
                used[list(r.seed_indices)]=True
                c=p.utilities[:,list(r.seed_indices)].max(axis=1)
                self.assertEqual(len(r.greedy_objectives),40)
                self.assertAlmostEqual(r.greedy_objectives[0],float(np.sum(p.weights*c,dtype=np.float32)),places=4)
                for k,j in enumerate(r.selection_order[25:],1):
                    gains=np.sum(p.weights[:,None]*np.maximum(0,p.utilities-c[:,None]),axis=0,dtype=np.float32)
                    gains[used]=-np.inf
                    self.assertEqual(j,int(np.argmax(gains)))
                    used[j]=True
                    c=np.maximum(c,p.utilities[:,j])
                    self.assertAlmostEqual(r.greedy_objectives[k],float(np.sum(p.weights*c,dtype=np.float32)),places=4)

    def test_a_d_exact_original_path(self):
        for n in (576,832,1088):
            p,out,_=case(n)
            for arm in ("A","D"):
                base=baseline.select_method(arm,p)
                row=out[arm]
                self.assertEqual(row.seed_indices,base.seed_indices)
                self.assertEqual(row.selection_order,base.greedy.selection_order)
                self.assertEqual(row.sorted_sequence_indices,base.greedy.sorted_sequence_indices)
                self.assertEqual(row.greedy_objectives,base.greedy.objective_values)
                self.assertEqual(row.final_cell_counts,base.final_cell_counts)

    def test_h_i_skip_greedy_and_all_restore_order(self):
        p,_,_=case(576)
        with patch.object(engine,"global_greedy",wraps=engine.global_greedy) as greedy:
            out=select_all(p)
        self.assertEqual(greedy.call_count,3)
        for n in (576,832,1088):
            p,out,_=case(n)
            for r in out.values():
                self.assertEqual(len(set(r.selection_order)),64)
                self.assertEqual(r.sorted_sequence_indices,tuple(sorted(p.sequence_indices[j] for j in r.selection_order)))
                self.assertEqual(sum(r.final_cell_counts),64)

    def test_all_methods_pass_original_decoder_compression(self):
        for n in (576,832,1088):
            p,out,_=case(n)
            ids=torch.ones(1,n+6,dtype=torch.long)
            ids[0,3:n+3]=63
            embeddings=torch.arange((n+6)*4,dtype=torch.float32).reshape(1,n+6,4)
            for row in out.values():
                packet=compress_visual_tokens(ids,embeddings,torch.ones_like(ids),p.sequence_indices,
                    row.sorted_visual_indices,(n+3,),image_token_id=63)
                self.assertEqual(packet.selected_original_positions,row.sorted_sequence_indices)
                self.assertEqual(int(packet.visual_mask.sum()),64)
                self.assertEqual(packet.length,70)
                self.assertTrue(torch.equal(packet.input_ids[0,~packet.visual_mask],ids[ids!=63]))


    def test_spatial_layout_does_not_change_global_methods(self):
        p,out,_=case(832)
        moved=replace(p.layout,cell_ids=np.roll(p.layout.cell_ids,1))
        shifted=select_all(replace(p,layout=moved))
        for arm in ("A","G","H"):
            self.assertEqual(shifted[arm].selection_order,out[arm].selection_order)
        self.assertNotEqual(shifted["D"].seed_indices,out["D"].seed_indices)

    def test_transpose_and_frozen_parsers(self):
        from study5_layout import live_layout
        from study5_scoring import parse_answer, official_answer, metrics
        p,_,_=case(832)
        transposed=live_layout(832,4,3,(400,600))
        result=select_all(replace(p,layout=transposed))
        self.assertEqual(result["D"].seed_cell_counts,(1,)*25)
        self.assertEqual(result["I"].seed_indices,result["D"].seed_indices)
        self.assertEqual(parse_answer(" NO! "),"no")
        self.assertEqual(official_answer("NO!"),"yes")
        rows=[dict(label="yes",prediction="invalid"),dict(label="no",prediction="no")]
        self.assertEqual(metrics(rows)["accuracy"],.5)
        self.assertEqual(metrics(rows)["recall"],0.)
