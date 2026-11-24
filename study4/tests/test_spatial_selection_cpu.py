"""P1-C synthetic CPU tests; P0 metadata only, no images/models/attention load."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
import coverpruner_coverage as engine
from coverpruner_spatial_layout import layout_from_p0, DEFAULT_EVIDENCE
from coverpruner_spatial_selection import (METHODS, prepare_problem, select_method, select_all,
    center_seeds, medoid_seeds, maximum_score_seeds, constrained_coverage_seeds)
from test_coverage_cpu import assert_arrays_equal, assert_arrays_close

METRICS={'F_reference_cases':0,'F_marginal_checks':0,'F_max_gain_abs_error':0.,'structures':[]}
CACHE={}


def full_case(n):
    if n not in CACHE:
        rng=np.random.default_rng(n+1717)
        x=rng.normal(size=(n,16)).astype(np.float32)
        r=rng.uniform(size=n).astype(np.float32)
        vision=rng.uniform(size=n).astype(np.float32)
        layout=layout_from_p0(n)
        positions=np.arange(n,dtype=np.int64)*3+7
        problem=prepare_problem(x,r,layout=layout,sequence_indices=positions)
        CACHE[n]=(x,r,vision,problem,select_all(problem,vision_scores=vision))
    return CACHE[n]


def ref_F(a,w,selected):
    return sum(float(w[i])*max((float(a[i,j]) for j in selected),default=0.) for i in range(len(w)))


def ref_F_seeds(a,w,owners,count):
    selected=[];closed=set();gains=[];values=[ref_F(a,w,selected)]
    while len(selected)<count:
        candidates=[j for j in range(len(w)) if owners[j] not in closed]
        scores=[ref_F(a,w,selected+[j])-values[-1] for j in candidates]
        best=max(range(len(candidates)),key=lambda j:scores[j])
        selected.append(candidates[best]);closed.add(owners[selected[-1]])
        gains.append(scores[best]);values.append(ref_F(a,w,selected))
    return tuple(selected),tuple(gains),tuple(values)


class SpatialCPUTests(unittest.TestCase):
    def test_P0_occupancy_N576_832_1088(self):
        for n,shape,minimum in ((576,(2,4),20),(832,(3,4),28),(1088,(4,4),40)):
            with self.subTest(N=n):
                layout=layout_from_p0(n)
                self.assertEqual((layout.crop_rows,layout.crop_cols),shape)
                self.assertEqual(len(layout.cell_ids),n)
                self.assertEqual(min(layout.cell_counts),minimum)
                self.assertEqual(sum(layout.cell_counts),n)
                self.assertEqual(layout.grid_shape,(5,5))
                self.assertFalse(layout.centers_xy.flags.writeable)
                self.assertFalse(layout.cell_ids.flags.writeable)

    def test_nominal_local_and_global_view_order(self):
        layout=layout_from_p0(576)
        assert_arrays_equal(layout.centers_xy[0],[1/64,1/32])
        assert_arrays_equal(layout.centers_xy[64],[(1+1/16)/4,1/32])
        assert_arrays_equal(layout.centers_xy[256],[1/64,(1+1/16)/2])
        assert_arrays_equal(layout.centers_xy[512],[1/16,1/16])
        assert_arrays_equal(layout.centers_xy[575],[15/16,15/16])
        # Local/global representations stay distinct despite overlapping source-image regions.
        self.assertEqual(layout.n,512+64)

    def test_rational_owner_independent_reference(self):
        for n in (576,832,1088):
            layout=layout_from_p0(n);rows,cols=layout.crop_rows,layout.crop_cols
            for ordinal in range(n):
                view,token=divmod(ordinal,64);v,u=divmod(token,8)
                if view==rows*cols:
                    x,y=Fraction(2*u+1,16),Fraction(2*v+1,16)
                else:
                    cr,cc=divmod(view,cols)
                    x,y=(Fraction(cc)+Fraction(2*u+1,16))/cols,(Fraction(cr)+Fraction(2*v+1,16))/rows
                expected=int(5*y)*5+int(5*x)
                self.assertEqual(int(layout.cell_ids[ordinal]),expected)
                self.assertEqual(tuple(layout.centers_xy[ordinal]),(float(x),float(y)))

    def test_reject_unknown_unverified_or_changed_P0(self):
        with self.assertRaises(ValueError): layout_from_p0(640)
        original=json.loads(DEFAULT_EVIDENCE.read_bytes())
        edits=[('pixel_mask_all_valid',False),('crop_marker_exact',False),
               ('installed_merger_order_exact',False),('tokens_per_block',63),
               ('visual_blocks',8),('rows',4),('empty_cells',[3]),('minimum_candidates_per_cell',1)]
        for key,value in edits:
            changed=copy.deepcopy(original);changed['samples'][0][key]=value
            with self.subTest(key=key),patch.object(Path,'read_bytes',return_value=json.dumps(changed).encode()):
                with self.assertRaises(ValueError): layout_from_p0(576)
        changed=copy.deepcopy(original);changed['status']='FAIL'
        with patch.object(Path,'read_bytes',return_value=json.dumps(changed).encode()):
            with self.assertRaises(ValueError): layout_from_p0(576)

    def test_B_nearest_geometric_center(self):
        xy=np.array([[.1,.1],[.25,.5],[.6,.5],[.75,.5]])
        self.assertEqual(center_seeds(xy,[0,0,1,1],(100,100),grid_shape=(1,2)),(1,3))

    def test_B_original_pixel_metric_not_normalized_distance(self):
        xy=np.array([[.6,.5],[.5,.7]])
        self.assertLess(np.sum((xy[0]-.5)**2),np.sum((xy[1]-.5)**2))
        self.assertEqual(center_seeds(xy,[0,0],(1000,100),grid_shape=(1,1)),(1,))

    def test_B_exact_tie_lowest_native(self):
        self.assertEqual(center_seeds([[.25,.5],[.75,.5]],[0,0],(100,100),grid_shape=(1,1)),(0,))

    def test_C_local_cosine_medoid(self):
        m=engine.cosine_similarity([[1,0],[.8,.6],[0,1],[1,0],[-1,0]])
        self.assertEqual(medoid_seeds(m,[0,0,0,1,1],num_cells=2),(1,3))
        scores=np.mean(m[:3,:3],axis=0)
        self.assertEqual(int(scores.argmax()),1)

    def test_C_self_inclusion_and_zero_norm(self):
        m=engine.cosine_similarity([[1,0],[-1,0],[0,0]])
        # Including self gives all-zero means ->0; excluding self would prefer2.
        assert_arrays_equal(m.mean(axis=0),[0,0,0])
        self.assertEqual(medoid_seeds(m,[0,0,0],num_cells=1),(0,))

    def test_D_raw_query_rank_before_weight_quantization(self):
        raw=np.array([0,1e-8,1],dtype=np.float32)
        weights=engine.demand_weights(raw)
        self.assertEqual(weights[0],weights[1])
        self.assertEqual(maximum_score_seeds(raw,[0,0,1],num_cells=2),(1,2))

    def test_E_external_vision_max_not_query(self):
        query=[9,1,8,2];vision=[1,9,2,8];owners=[0,0,1,1]
        self.assertEqual(maximum_score_seeds(query,owners,num_cells=2),(0,2))
        self.assertEqual(maximum_score_seeds(vision,owners,num_cells=2),(1,3))

    def test_F_weighted_global_cells_seed_coverage_and_release(self):
        a=np.array([[1,1,0,0],[0,0,1,0],[0,0,0,1],[0,0,0,0]],dtype=np.float32)
        w=np.array([2,1,1,0],dtype=np.float32);owners=np.array([1,0,0,1])
        stage=constrained_coverage_seeds(a,w,owners,num_cells=2)
        self.assertEqual(stage.indices,(0,2))  # cell1 FIRST, not cell0-first traversal
        self.assertEqual(stage.marginal_gains,(2.,1.))
        assert_arrays_equal(stage.coverage,[1,1,0,0])
        self.assertEqual(engine.marginal_gain(a,w,stage.indices,1),0) # duplicate is covered
        result=engine.global_greedy(a,w,3,seed_indices=stage.indices)
        self.assertEqual(result.selection_order,(0,2,3)) # cell1 reopened, positive gain wins
        self.assertEqual(result.objective_values,(3.,4.))

    def test_F_accounts_for_all_prior_seeds_not_last_only(self):
        a=np.array([[1,0,.5,0],[0,1,.5,0],[0,0,0,1],[0,0,0,0]],dtype=np.float32)
        stage=constrained_coverage_seeds(a,[2,2,1,0],[0,1,2,2],num_cells=3)
        self.assertEqual(stage.indices,(0,1,3))
        self.assertEqual(stage.marginal_gains,(2.,2.,1.))
        self.assertEqual(stage.objective_values,(0.,2.,4.,5.))

    def test_F_ties_global_ordinal_not_cell_number(self):
        owners=np.repeat(np.arange(24,-1,-1),3)
        stage=constrained_coverage_seeds(np.zeros((75,75)),np.ones(75),owners)
        self.assertEqual(stage.indices,tuple(range(0,75,3)))
        self.assertEqual(tuple(owners[list(stage.indices)]),tuple(range(24,-1,-1)))
        self.assertEqual(stage.marginal_gains,(0.,)*25)

    def test_F_slow_reference_order_gain_and_state(self):
        rng=np.random.default_rng(2711)
        for n in range(2,10):
            for count in range(1,min(n,3)+1):
                for repeat in range(2):
                    owners=rng.permutation(np.arange(n)%count)
                    a=rng.integers(0,9,(n,n)).astype(np.float32)/8
                    w=rng.integers(0,5,n).astype(np.float32)/4
                    expected=ref_F_seeds(a,w,owners,count)
                    stage=constrained_coverage_seeds(a,w,owners,num_cells=count)
                    self.assertEqual(stage.indices,expected[0])
                    self.assertEqual(stage.marginal_gains,expected[1])
                    self.assertEqual(stage.objective_values,expected[2])
                    assert_arrays_equal(stage.coverage,engine.coverage_state(a,stage.indices))
                    selected=[];closed=set()
                    for token in stage.indices:
                        before=ref_F(a,w,selected)
                        for j in range(n):
                            if owners[j] not in closed:
                                expected_gain=ref_F(a,w,selected+[j])-before
                                actual=engine.marginal_gain(a,w,selected,j)
                                error=abs(actual-expected_gain)
                                METRICS['F_marginal_checks']+=1
                                METRICS['F_max_gain_abs_error']=max(METRICS['F_max_gain_abs_error'],error)
                                self.assertEqual(actual,expected_gain)
                        selected.append(token);closed.add(owners[token])
                    METRICS['F_reference_cases']+=1

    def test_A_exact_existing_global_without_mapping(self):
        x,r,_,problem,_=full_case(576)
        no_map=prepare_problem(x,r,sequence_indices=problem.sequence_indices)
        result=select_method('A',no_map).greedy
        expected=engine.select_from_embeddings(x,r,64,sequence_indices=problem.sequence_indices)
        self.assertEqual(result.selection_order,expected.selection_order)
        self.assertEqual(result.marginal_gains,expected.marginal_gains)
        self.assertEqual(result.objective_values,expected.objective_values)
        self.assertEqual(result.sorted_sequence_indices,expected.sorted_sequence_indices)
        self.assertIsNone(no_map.layout)

    def test_A_unchanged_by_optional_spatial_mapping(self):
        x,r,_,problem,results=full_case(576)
        baseline=select_method('A',prepare_problem(x,r,sequence_indices=problem.sequence_indices))
        self.assertEqual(results['A'].greedy.selection_order,baseline.greedy.selection_order)

    def test_all_methods_three_structures_and_budgets(self):
        for n in (576,832,1088):
            x,r,vision,problem,results=full_case(n)
            self.assertEqual(tuple(results),METHODS)
            for method,result in results.items():
                self.assertEqual(len(result.greedy.selection_order),64)
                self.assertEqual(len(set(result.greedy.selection_order)),64)
                self.assertEqual(len(result.greedy.sorted_sequence_indices),64)
                self.assertEqual(result.greedy.sorted_sequence_indices,tuple(sorted(result.greedy.selection_order_sequence_indices)))
                if method=='A':
                    self.assertEqual(result.seed_indices,())
                else:
                    self.assertEqual(result.seed_cell_counts,(1,)*25)
                    self.assertEqual(len(result.greedy.added_indices),39)
                    self.assertEqual(result.greedy.selection_order[:25],result.seed_indices)
                    self.assertEqual(sum(result.final_cell_counts),64)
                    self.assertGreaterEqual(max(result.final_cell_counts),3) # final one-per-cell cap released
            METRICS['structures'].append(dict(N=n,methods='A-F',unique_K=64,seed_cells=25,continuation=39,status='PASS'))

    def test_B_to_F_seed_coverage_real_and_not_reset(self):
        _,_,_,problem,results=full_case(576)
        for method in METHODS[1:]:
            result=results[method];seed=result.seed_indices
            state=np.max(problem.utilities[:,list(seed)],axis=1)
            expected_F=float(np.sum(problem.weights*state,dtype=np.float32))
            self.assertGreater(expected_F,0)
            self.assertEqual(result.greedy.objective_values[0],expected_F)
            expected=engine.global_greedy(problem.utilities,problem.weights,64,seed_indices=seed,
                                         sequence_indices=problem.sequence_indices)
            self.assertEqual(result.greedy.selection_order,expected.selection_order)
            self.assertEqual(result.greedy.marginal_gains,expected.marginal_gains)
            self.assertEqual(result.greedy.objective_values,expected.objective_values)
            if method=='F': self.assertEqual(result.seed_objective_values[-1],expected_F)

    def test_same_engine_M_A_weights_and_pool_all_methods(self):
        _,_,vision,problem,_=full_case(576)
        original=engine.global_greedy
        with patch.object(engine,'global_greedy',wraps=original) as spy:
            select_all(problem,vision_scores=vision)
        final_calls=[call for call in spy.call_args_list if call.args[2]==64]
        self.assertEqual(len(final_calls),6)
        for method,call in zip(METHODS,final_calls):
            self.assertIs(call.args[0],problem.utilities)
            self.assertIs(call.args[1],problem.weights)
            self.assertEqual(len(call.kwargs['seed_indices']),0 if method=='A' else 25)
            self.assertEqual(call.kwargs['sequence_indices'],problem.sequence_indices)

    def test_C_E_seed_independent_of_query_and_D_uses_query(self):
        x,r,vision,problem,results=full_case(576)
        changed=prepare_problem(x,-r,layout=problem.layout,sequence_indices=problem.sequence_indices)
        self.assertEqual(select_method('C',changed).seed_indices,results['C'].seed_indices)
        self.assertEqual(select_method('E',changed,vision_scores=vision).seed_indices,results['E'].seed_indices)
        self.assertNotEqual(select_method('D',changed).seed_indices,results['D'].seed_indices)
        assert_arrays_equal(problem.weights,engine.demand_weights(r,tau=.3))

    def test_E_requires_injection_no_model_extractor(self):
        _,_,_,problem,results=full_case(576)
        with self.assertRaises(ValueError): select_method('E',problem)
        self.assertEqual(results['E'].vision_score_source,'external_injected_only')

    def test_original_sequence_indices_gaps_and_reordering(self):
        x,r,_,problem,_=full_case(576)
        positions=(np.arange(576,dtype=np.int64)*7+19)[::-1]
        prepared=prepare_problem(x,r,layout=problem.layout,sequence_indices=positions)
        result=select_method('D',prepared).greedy
        self.assertEqual(result.sorted_sequence_indices,tuple(sorted(positions[list(result.selection_order)])))
        self.assertEqual(result.selection_order_sequence_indices,tuple(positions[list(result.selection_order)]))
        self.assertEqual(tuple(positions[list(result.sorted_visual_indices)]),result.sorted_sequence_indices)
        expected_ordinals=tuple(sorted(result.selection_order,key=lambda i:int(positions[i])))
        self.assertEqual(result.sorted_visual_indices,expected_ordinals)
        assert_arrays_equal(x[list(result.sorted_visual_indices)],x[list(expected_ordinals)])

    def test_all_zero_norm_equal_scores_exact_ties(self):
        problem=prepare_problem(np.zeros((576,4)),np.zeros(576),layout=layout_from_p0(576))
        results=select_all(problem,vision_scores=np.ones(576))
        for method,result in results.items():
            self.assertEqual(len(set(result.greedy.selection_order)),64)
            self.assertEqual(result.greedy.objective_values[-1],0)
            self.assertTrue(np.isfinite(result.greedy.coverage).all())
            if method in ('C','D','E'):
                expected=tuple(int(np.flatnonzero(problem.layout.cell_ids==cell)[0]) for cell in range(25))
                self.assertEqual(result.seed_indices,expected)
        self.assertEqual(results['A'].greedy.selection_order,tuple(range(64)))

    def test_F_determinism_and_input_immutability(self):
        a=np.eye(30,dtype=np.float32);w=np.ones(30,dtype=np.float32);owners=np.arange(30)%3
        original_a,original_w=a.copy(),w.copy()
        first=constrained_coverage_seeds(a,w,owners,num_cells=3)
        second=constrained_coverage_seeds(a,w,owners,num_cells=3)
        self.assertEqual(first.indices,second.indices)
        self.assertEqual(first.marginal_gains,second.marginal_gains)
        assert_arrays_equal(a,original_a);assert_arrays_equal(w,original_w)
        self.assertFalse(first.coverage.flags.writeable)

    def test_prepared_shared_arrays_immutable(self):
        x,r,vision,problem,_=full_case(576)
        before_x,before_r,before_v=x.copy(),r.copy(),vision.copy()
        for value in (problem.raw_cosine,problem.utilities,problem.query_relevance,problem.weights):
            self.assertFalse(value.flags.writeable)
        select_method('E',problem,vision_scores=vision)
        assert_arrays_equal(x,before_x);assert_arrays_equal(r,before_r);assert_arrays_equal(vision,before_v)

    def test_invalid_or_empty_cells_fail_without_fallback(self):
        for ids,count in (([0,0],2),([-1,0],1),([0,2],2),([0.,1.],2),([False,True],2),([0,True],2)):
            with self.subTest(ids=ids):
                with self.assertRaises(ValueError): maximum_score_seeds([1,2],ids,num_cells=count)
                with self.assertRaises(ValueError): constrained_coverage_seeds(np.eye(2),[1,1],ids,num_cells=count)

    def test_invalid_coordinates_and_dimensions(self):
        for xy in ([[np.nan,.5]],[[np.inf,.5]],[[-.1,.5]],[[.5,1.1]],[[1j,.5]],[[.5]]):
            with self.assertRaises(ValueError): center_seeds(xy,[0],(100,100),grid_shape=(1,1))
        for size in ((0,100),(100,-1),(True,100),(1.5,100)):
            with self.assertRaises(ValueError): center_seeds([[.5,.5]],[0],size,grid_shape=(1,1))

    def test_NaN_Inf_embeddings_query_and_external_vision(self):
        x,r,vision,problem,_=full_case(576)
        for bad in (np.nan,np.inf,-np.inf):
            data=x.copy();data[0,0]=bad
            with self.assertRaises(ValueError): prepare_problem(data,r,layout=problem.layout)
            scores=r.copy();scores[0]=bad
            with self.assertRaises(ValueError): prepare_problem(x,scores,layout=problem.layout)
            scores=vision.copy();scores[0]=bad
            with self.assertRaises(ValueError): select_method('E',problem,vision_scores=scores)
        with self.assertRaises(ValueError): medoid_seeds([[np.nan,0],[0,1]],[0,0],num_cells=1)

    def test_invalid_method_layout_and_score_length(self):
        x,r,_,problem,_=full_case(576)
        with self.assertRaises(ValueError): select_method('G',problem)
        with self.assertRaises(ValueError): select_method('B',prepare_problem(x,r))
        with self.assertRaises(ValueError): prepare_problem(x,r,layout=layout_from_p0(832))
        with self.assertRaises(ValueError): select_method('E',problem,vision_scores=[1,2])
        with self.assertRaises(ValueError): select_method('E',problem,vision_scores=np.full(576,1e100))