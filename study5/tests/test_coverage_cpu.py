"""Synthetic CPU tests only. Independent scalar reference never calls prod math."""
import itertools
import math
import unittest

import numpy as np

from coverpruner_coverage import (
    cosine_similarity, coverage_utilities, normalize_relevance, demand_weights,
    coverage_state, coverage_value, marginal_gain, global_greedy,
    select_from_embeddings,
)

def assert_arrays_equal(actual, expected):
    left, right = np.asarray(actual), np.asarray(expected)
    if not np.array_equal(left, right):
        raise AssertionError(f"Array equality failed: shapes {left.shape}/{right.shape}")


def assert_arrays_close(actual, expected, *, atol, rtol):
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(expected, dtype=np.float64)
    if left.shape != right.shape:
        raise AssertionError(f"Shape mismatch {left.shape}/{right.shape}")
    delta = np.abs(left - right)
    if not (np.isfinite(left).all() and np.isfinite(right).all()
            and np.all(delta <= atol + rtol * np.abs(right))):
        raise AssertionError(f"Array closeness failed; max abs error {np.max(delta)}")


METRICS = {"reference_greedy_cases": 0, "reference_gain_checks": 0,
           "max_gain_absolute_error": 0.0, "large_structures": []}


def reference_value(matrix, weights, selected):
    # Python scalar/FP64 accumulation, independently recomputed from ALL selected.
    return sum(float(weights[i]) * max((float(matrix[i][j]) for j in selected), default=0.0)
               for i in range(len(weights)))


def reference_greedy(matrix, weights, k, seed=()):
    selected, gains = list(seed), []
    values = [reference_value(matrix, weights, selected)]
    while len(selected) < k:
        candidates = [j for j in range(len(weights)) if j not in selected]
        deltas = [reference_value(matrix, weights, selected + [j]) - values[-1]
                  for j in candidates]
        # First maximum over canonical ordinal order, same EXACT tie contract.
        best = max(range(len(candidates)), key=lambda i: deltas[i])
        selected.append(candidates[best])
        gains.append(deltas[best])
        values.append(reference_value(matrix, weights, selected))
    return tuple(selected), tuple(gains), tuple(values)


def subsets(n):
    for count in range(n + 1):
        yield from itertools.combinations(range(n), count)


def history_fixture():
    # After0,1, candidate2 has zero gain. A last-token-only state picks2 wrongly.
    return (np.array([[1, 0, .5, 0], [0, 1, .5, 0],
                      [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.float32),
            np.array([2, 2, 1, 0], dtype=np.float32))


class CoverageCPUTests(unittest.TestCase):
    def test_cosine_hand_values(self):
        x = [[3, 4], [0, 5], [-3, -4], [0, 0]]
        expected = [[1, .8, -1, 0], [.8, 1, -.8, 0],
                    [-1, -.8, 1, 0], [0, 0, 0, 0]]
        assert_arrays_close(cosine_similarity(x), expected, atol=2e-7, rtol=0)

    def test_cosine_independent_scalar_reference(self):
        x = np.random.default_rng(173).normal(size=(11, 7)).astype(np.float32)
        expected = []
        for row in x:
            out = []
            for other in x:
                dot = sum(float(a) * float(b) for a, b in zip(row, other))
                norm = math.sqrt(sum(float(a)**2 for a in row))
                other_norm = math.sqrt(sum(float(a)**2 for a in other))
                out.append(dot / max(norm, 1e-12) / max(other_norm, 1e-12))
            expected.append(out)
        assert_arrays_close(cosine_similarity(x), expected, atol=3e-7, rtol=0)

    def test_float32_outputs(self):
        m = cosine_similarity(np.eye(4, dtype=np.float64))
        a = coverage_utilities(m.astype(np.float16))
        for value in (m, a, normalize_relevance([1, 2, 3, 4]),
                      demand_weights([1, 2, 3, 4]), coverage_state(a, [1])):
            self.assertEqual(value.dtype, np.float32)

    def test_row_mean_centering_and_clipping(self):
        matrix = [[1, .5, -.5], [0, 1, 0], [-.5, .5, 1]]
        expected = [[max(0, value - sum(row) / len(row)) for value in row]
                    for row in matrix]
        a = coverage_utilities(matrix)
        assert_arrays_close(a, expected, atol=5e-8, rtol=0)
        self.assertTrue(np.all(a >= 0))
        self.assertNotEqual(float(a[0, 1]), float(a[1, 0]))

    def test_calibration_uses_full_original_N(self):
        a = coverage_utilities([[1, .5, 0, -.5]] * 4)
        assert_arrays_equal(a, [[.75, .25, 0, 0]] * 4)
        before = a.copy()
        global_greedy(a, np.ones(4), 3, seed_indices=[1])
        assert_arrays_equal(a, before)

    def test_relevance_normalization_and_tau03(self):
        r = [-2, 0, 2]
        assert_arrays_equal(normalize_relevance(r), [0, .5, 1])
        assert_arrays_close(demand_weights(r), [1, math.exp(.15), math.exp(.3)],
                                   atol=2e-7, rtol=0)
        assert_arrays_equal(demand_weights(r, tau=0), np.ones(3))

    def test_equal_relevance_policy(self):
        assert_arrays_equal(normalize_relevance([7, 7, 7]), np.zeros(3))
        assert_arrays_equal(demand_weights([7, 7, 7]), np.ones(3))
        # Cast-to-FP32 equality follows the documented FP32 policy.
        assert_arrays_equal(demand_weights([1., 1. + 1e-10]), np.ones(2))

    def test_empty_coverage_and_k0(self):
        a, w = history_fixture()
        assert_arrays_equal(coverage_state(a, []), np.zeros(4))
        self.assertEqual(coverage_value(a, w, []), 0)
        result = global_greedy(a, w, 0)
        self.assertEqual(result.selection_order, ())
        self.assertEqual(result.objective_values, (0.,))

    def test_first_gain_by_hand(self):
        a, w = history_fixture()
        self.assertEqual([marginal_gain(a, w, [], j) for j in range(4)], [2, 2, 2, 1])
        result = global_greedy(a, w, 1)
        self.assertEqual(result.selection_order, (0,))
        self.assertEqual(result.marginal_gains, (2.,))

    def test_full_selected_history_not_last_token(self):
        a, w = history_fixture()
        result = global_greedy(a, w, 4)
        self.assertEqual(result.selection_order, (0, 1, 3, 2))
        self.assertEqual(result.marginal_gains, (2., 2., 1., 0.))
        self.assertEqual(marginal_gain(a, w, [0, 1], 2), 0)
        self.assertEqual(marginal_gain(a, w, [0, 1], 3), 1)

    def test_marginal_equals_F_increment_all_subsets(self):
        rng = np.random.default_rng(177)
        a = rng.integers(0, 9, (6, 6)).astype(np.float32) / 8
        w = np.array([.25, .5, 1, .75, 0, 1.5], dtype=np.float32)
        for selected in subsets(6):
            before = reference_value(a, w, selected)
            self.assertEqual(coverage_value(a, w, selected), before)
            for candidate in range(6):
                expected = reference_value(a, w, (*selected, candidate)) - before
                actual = marginal_gain(a, w, selected, candidate)
                error = abs(actual - expected)
                METRICS["reference_gain_checks"] += 1
                METRICS["max_gain_absolute_error"] = max(
                    METRICS["max_gain_absolute_error"], error)
                self.assertEqual(actual, expected)

    def test_monotone_coverage_each_step(self):
        a = np.random.default_rng(12).uniform(size=(10, 10)).astype(np.float32)
        w = demand_weights(np.arange(10))
        result = global_greedy(a, w, 10)
        previous = np.zeros(10, dtype=np.float32)
        for count in range(1, 11):
            state = coverage_state(a, result.selection_order[:count])
            self.assertTrue(np.all(state >= previous))
            actual_F = coverage_value(a, w, result.selection_order[:count])
            self.assertEqual(actual_F, result.objective_values[count])
            self.assertAlmostEqual(actual_F - result.objective_values[count - 1],
                                   result.marginal_gains[count - 1], delta=3e-6)
            previous = state
        self.assertTrue(all(a <= b for a, b in zip(result.objective_values,
                                                  result.objective_values[1:])))

    def test_reference_greedy_exact_dyadic_cases(self):
        rng = np.random.default_rng(178)
        for n in range(2, 8):
            for _ in range(2):
                a = rng.integers(0, 9, (n, n)).astype(np.float32) / 8
                w = rng.integers(0, 5, n).astype(np.float32) / 4
                for seed in ((), (n - 1,), (n - 1, 0)):
                    for k in range(len(seed), n + 1):
                        expected = reference_greedy(a, w, k, seed)
                        result = global_greedy(a, w, k, seed_indices=seed)
                        self.assertEqual(result.selection_order, expected[0])
                        self.assertEqual(result.marginal_gains, expected[1])
                        self.assertEqual(result.objective_values, expected[2])
                        METRICS["reference_greedy_cases"] += 1

    def test_embedding_pipeline_vs_independent_reference(self):
        x = np.eye(7, dtype=np.float32)
        m = cosine_similarity(x)
        a = coverage_utilities(m)
        w = demand_weights(np.arange(7))
        expected = reference_greedy(a, w, 5)
        result = select_from_embeddings(x, np.arange(7), 5)
        self.assertEqual(result.selection_order, expected[0])
        assert_arrays_close(result.marginal_gains, expected[1], atol=1e-6, rtol=0)
        assert_arrays_close(result.objective_values, expected[2], atol=1e-6, rtol=0)

    def test_exhaustive_optimum_not_greedy_requirement(self):
        a = np.zeros((6, 6), dtype=np.float32)
        a[[0, 1, 2, 3], 0] = 1
        a[[0, 1, 4], 1] = 1
        a[[2, 3, 5], 2] = 1
        w = np.ones(6, dtype=np.float32)
        result = global_greedy(a, w, 2)
        optimum = max((reference_value(a, w, choice), choice)
                      for choice in itertools.combinations(range(6), 2))
        self.assertEqual(result.selection_order, (0, 1))
        self.assertEqual(result.selection_order, reference_greedy(a, w, 2)[0])
        self.assertEqual(result.objective_values[-1], 5)
        self.assertEqual(optimum, (6., (1, 2)))
        METRICS["exhaustive_diagnostic"] = {"greedy_F": 5, "optimal_F": 6,
                                            "gap_is_expected_not_failure": True}

    def test_k64_default_and_K_equals_N_identity(self):
        result = select_from_embeddings(np.eye(64), np.zeros(64))
        self.assertEqual(result.selection_order, tuple(range(64)))
        self.assertEqual(result.sorted_visual_indices, tuple(range(64)))
        self.assertEqual(len(set(result.selection_order)), 64)
        self.assertEqual(result.objective_values[-1], 63)

    def test_N576_832_1088_and_seed25_resume39(self):
        for n in (576, 832, 1088):
            with self.subTest(N=n):
                rng = np.random.default_rng(n)
                x = rng.normal(size=(n, 24)).astype(np.float32)
                a = coverage_utilities(cosine_similarity(x))
                w = demand_weights(rng.uniform(size=n).astype(np.float32))
                positions = np.arange(n, dtype=np.int64) * 5 + 11
                full = global_greedy(a, w, sequence_indices=positions)
                resumed = global_greedy(a, w, seed_indices=full.selection_order[:25],
                                        sequence_indices=positions)
                self.assertEqual(len(full.selection_order), 64)
                self.assertEqual(len(set(full.selection_order)), 64)
                self.assertEqual(resumed.selection_order, full.selection_order)
                self.assertEqual(len(resumed.added_indices), 39)
                self.assertEqual(resumed.marginal_gains, full.marginal_gains[25:])
                self.assertEqual(resumed.objective_values, full.objective_values[25:])
                self.assertEqual(resumed.sorted_sequence_indices,
                                 tuple(sorted(positions[i] for i in full.selection_order)))
                assert_arrays_equal(full.coverage,
                                              coverage_state(a, full.selection_order))
                assert_arrays_equal(resumed.coverage, full.coverage)
                self.assertTrue(np.isfinite(full.coverage).all())
                METRICS["large_structures"].append({"N": n, "K": 64, "unique": 64,
                    "seed_count": 25, "added": 39, "resume_parity": "exact"})

    def test_seed_state_initialization_and_order(self):
        a, w = history_fixture()
        result = global_greedy(a, w, 3, seed_indices=[1, 0])
        assert_arrays_equal(coverage_state(a, [1, 0]), [1, 1, 0, 0])
        self.assertEqual(result.seed_indices, (1, 0))
        self.assertEqual(result.selection_order, (1, 0, 3))
        self.assertEqual(result.added_indices, (3,))
        self.assertEqual(result.objective_values, (4., 5.))
        self.assertEqual(result.marginal_gains, (1.,))

    def test_external_25_seeds_are_not_ignored(self):
        a, w = np.eye(80, dtype=np.float32), np.ones(80)
        seed = tuple(range(1, 50, 2))
        result = global_greedy(a, w, 64, seed_indices=seed)
        self.assertEqual(result.selection_order[:25], seed)
        self.assertEqual(result.objective_values[0], 25)
        self.assertEqual(len(result.added_indices), 39)
        self.assertFalse(set(seed).intersection(result.added_indices))
        self.assertEqual(result.objective_values[-1], 64)
        assert_arrays_equal(result.coverage, coverage_state(a, result.selection_order))

    def test_seed_at_total_budget_no_additions(self):
        a, w = history_fixture()
        result = global_greedy(a, w, 2, seed_indices=[1, 0])
        self.assertEqual(result.selection_order, (1, 0))
        self.assertEqual(result.added_indices, ())
        self.assertEqual(result.marginal_gains, ())
        self.assertEqual(result.objective_values, (4.,))

    def test_sequence_mapping_and_sort(self):
        a, w = history_fixture()
        positions = [70, 10, 90, 30]
        result = global_greedy(a, w, 3, sequence_indices=positions)
        self.assertEqual(result.selection_order, (0, 1, 3))
        self.assertEqual(result.selection_order_sequence_indices, (70, 10, 30))
        self.assertEqual(result.sorted_visual_indices, (1, 3, 0))
        self.assertEqual(result.sorted_sequence_indices, (10, 30, 70))
        x = np.arange(12).reshape(4, 3)
        assert_arrays_equal(x[list(result.sorted_visual_indices)], x[[1, 3, 0]])

    def test_exact_tie_lowest_native_not_sequence_position(self):
        result = global_greedy(np.zeros((5, 5)), np.ones(5), 4, seed_indices=[3],
                               sequence_indices=[90, 80, 70, 60, 50])
        self.assertEqual(result.selection_order, (3, 0, 1, 2))
        self.assertEqual(result.marginal_gains, (0., 0., 0.))
        self.assertEqual(result.sorted_visual_indices, (3, 2, 1, 0))
        result = global_greedy(np.ones((4, 4)), np.zeros(4), 4)
        self.assertEqual(result.selection_order, (0, 1, 2, 3))

    def test_deterministic_repeat_and_input_immutability(self):
        a, w = history_fixture()
        before_a, before_w = a.copy(), w.copy()
        seed, positions = [1], [70, 10, 90, 30]
        first = global_greedy(a, w, 4, seed_indices=seed, sequence_indices=positions)
        second = global_greedy(a, w, 4, seed_indices=seed, sequence_indices=positions)
        self.assertEqual(first.selection_order, second.selection_order)
        self.assertEqual(first.marginal_gains, second.marginal_gains)
        assert_arrays_equal(first.coverage, second.coverage)
        assert_arrays_equal(a, before_a)
        assert_arrays_equal(w, before_w)
        self.assertEqual(seed, [1])
        self.assertEqual(positions, [70, 10, 90, 30])
        with self.assertRaises(ValueError):
            first.coverage[0] = 99

    def test_zero_norm_and_small_norm_policy(self):
        assert_arrays_equal(cosine_similarity(np.zeros((4, 3))), np.zeros((4, 4)))
        actual = cosine_similarity([[1e-13, 0], [0, 0]])
        assert_arrays_close(actual, [[.01, 0], [0, 0]], atol=2e-9, rtol=0)
        result = select_from_embeddings(np.zeros((5, 3)), np.zeros(5), 5)
        self.assertEqual(result.selection_order, tuple(range(5)))
        self.assertEqual(result.marginal_gains, (0.,) * 5)

    def test_nan_inf_rejected_at_all_entrypoints(self):
        for bad in (math.nan, math.inf, -math.inf):
            for function in (
                lambda: cosine_similarity([[bad, 1]]),
                lambda: coverage_utilities([[bad]]),
                lambda: normalize_relevance([0, bad]),
                lambda: demand_weights([0, bad]),
                lambda: coverage_value([[bad]], [1], []),
                lambda: marginal_gain([[1]], [bad], [], 0),
                lambda: global_greedy([[1]], [bad], 1),
            ):
                with self.subTest(bad=bad, function=function), self.assertRaises(ValueError):
                    function()

    def test_overflow_fails_closed(self):
        for function in (
            lambda: cosine_similarity([[1e100]]),
            lambda: cosine_similarity([[1e30]]),
            lambda: coverage_utilities(np.full((8, 8), 1e38)),
            lambda: normalize_relevance([-3e38, 3e38]),
            lambda: demand_weights([0, 1], tau=1000),
            lambda: global_greedy([[3]], [3e38], 1),
        ):
            with self.subTest(function=function), self.assertRaises(ValueError):
                function()

    def test_invalid_shapes_types_and_negative_values(self):
        for function in (
            lambda: cosine_similarity([]), lambda: cosine_similarity([1, 2]),
            lambda: cosine_similarity(np.empty((2, 0))),
            lambda: cosine_similarity([[1j]]), lambda: cosine_similarity([["1"]]),
            lambda: coverage_utilities(np.ones((2, 3))),
            lambda: normalize_relevance([]), lambda: normalize_relevance([[1, 2]]),
            lambda: global_greedy([[-1]], [1], 1),
            lambda: global_greedy([[1]], [-1], 1),
            lambda: global_greedy(np.eye(2), [1], 1),
            lambda: select_from_embeddings(np.eye(3), [0, 1], 2),
        ):
            with self.subTest(function=function), self.assertRaises(ValueError):
                function()

    def test_invalid_k(self):
        for k in (-1, 4, True, 1.5, math.nan):
            with self.subTest(k=k), self.assertRaises(ValueError):
                global_greedy(np.eye(3), np.ones(3), k)
        with self.assertRaises(ValueError):
            global_greedy(np.eye(3), np.ones(3), 1, seed_indices=[0, 1])

    def test_invalid_seed_and_candidate(self):
        for seed in ([0, 0], [-1], [3], [1.], [True], [[1]], [0, True]):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                global_greedy(np.eye(3), np.ones(3), 3, seed_indices=seed)
        for candidate in (-1, 3, True, 1.5):
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                marginal_gain(np.eye(3), np.ones(3), [], candidate)

    def test_invalid_sequence_positions(self):
        for positions in ([0, 1], [0, 0, 1], [-1, 0, 1], [0., 1., 2.],
                          [False, True, False], [0, True, 2], [[0, 1, 2]]):
            with self.subTest(positions=positions), self.assertRaises(ValueError):
                global_greedy(np.eye(3), np.ones(3), 2, sequence_indices=positions)

    def test_invalid_numeric_parameters(self):
        for eps in (0, -1, 1e-100, math.nan, math.inf, True, [1e-12]):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                cosine_similarity(np.eye(3), eps_norm=eps)
        for tau in (-1, math.nan, math.inf, True, [.3]):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                demand_weights([0, 1], tau=tau)