# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dongeunk (original project contributions).
# Licensed under the Apache License, Version 2.0 (the License).
# You may not use this file except in compliance with the License.
# A copy is in the repository LICENSE and at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by law or agreed to in writing, distributed AS IS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for permissions and limitations.
# Publication copy: license/attribution comments added; executable code unchanged.

"""Synthetic tests of the controlled regional-quota allocation ablation."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from src.pruning.matched_allocation import compare_local_allocation
from src.pruning.spatial_floor_coverage import spatial_floor_coverage_select


def independent_regions(rows, cols, token_grid, region_grid):
    """Coordinate-based mapping, independent of the view-major tensor reshape."""
    token_h, token_w = token_grid
    region_h, region_w = region_grid
    height, width = rows * token_h, cols * token_w
    y_edges = [round(i * height / region_h) for i in range(region_h + 1)]
    x_edges = [round(i * width / region_w) for i in range(region_w + 1)]
    regions = [[] for _ in range(region_h * region_w)]
    for index in range(rows * cols * token_h * token_w):
        view, token = divmod(index, token_h * token_w)
        view_y, view_x = divmod(view, cols)
        token_y, token_x = divmod(token, token_w)
        y, x = view_y * token_h + token_y, view_x * token_w + token_x
        r = next(i for i in range(region_h) if y_edges[i] <= y < y_edges[i + 1])
        c = next(i for i in range(region_w) if x_edges[i] <= x < x_edges[i + 1])
        regions[r * region_w + c].append(index)
    return regions


def independent_affinity(embeddings):
    vectors = F.normalize(embeddings.float(), dim=1, eps=1e-12)
    similarity = vectors @ vectors.T
    return (similarity - similarity.mean(dim=1, keepdim=True)).clamp_min(0)


def objective_oracle(affinity, regions, k, budgets=None):
    """Enumerate F(S union {candidate}) over every feasible candidate per step."""
    floor = [max(pool, key=lambda j: sum(float(affinity[i, j]) for i in pool))
             for pool in regions]
    selected = list(floor)
    completion = []
    token_region = {index: region for region, pool in enumerate(regions) for index in pool}
    for _ in range(k - len(floor)):
        counts = [sum(index in selected for index in pool) for pool in regions]
        values = {}
        for candidate in sorted(token_region):
            if candidate in selected:
                continue
            if budgets is not None and counts[token_region[candidate]] >= budgets[token_region[candidate]]:
                continue
            columns = selected + [candidate]
            values[candidate] = sum(max(float(affinity[i, j]) for j in columns)
                                    for i in range(affinity.shape[0]))
        winner = max(values, key=values.get)
        completion.append(winner)
        selected.append(winner)
    return floor, completion, sorted(selected)


class MatchedAllocationTest(unittest.TestCase):
    @staticmethod
    def fixture():
        return torch.randn(12, 6, generator=torch.Generator().manual_seed(0))

    def test_identical_floor_shared_affinity_demand_candidates_and_k(self):
        embeddings = self.fixture()
        affinity = independent_affinity(embeddings)
        before = affinity.clone()
        result = compare_local_allocation(embeddings, 6, 1, 2, (2, 2), (1, 2), [3, 3], affinity)
        self.assertIs(result.affinity, affinity)
        self.assertTrue(torch.equal(affinity, before))
        self.assertIs(result.floor_indices, result.local_only.floor_indices)
        self.assertIs(result.floor_indices, result.laplacian_quota.floor_indices)
        self.assertEqual(result.demand_indices.tolist(), list(range(12)))
        self.assertEqual(result.candidate_indices.tolist(), list(range(8)))
        self.assertEqual(result.region_budgets.tolist(), [[3, 3]])
        for method in (result.local_only, result.laplacian_quota):
            self.assertEqual(method.selected_indices.numel(), 6)
            self.assertEqual((method.selected_local, method.selected_global), (6, 0))

    def test_affinity_is_computed_exactly_once(self):
        embeddings = self.fixture()
        affinity = independent_affinity(embeddings)
        with patch("src.pruning.matched_allocation.calibrated_affinity", return_value=affinity) as helper:
            result = compare_local_allocation(embeddings, 6, 1, 2, (2, 2), (1, 2), [3, 3])
        helper.assert_called_once_with(embeddings)
        self.assertIs(result.affinity, affinity)

    def test_phase_one_matches_existing_spatial_floor_selector(self):
        embeddings = torch.randn(42, 5, generator=torch.Generator().manual_seed(18))
        existing = spatial_floor_coverage_select(embeddings, 12, 2, 3, (3, 2), (3, 4))
        result = compare_local_allocation(embeddings, 12, 2, 3, (3, 2), (3, 4), [1] * 12)
        self.assertTrue(torch.equal(result.floor_indices, existing.floor_indices))
        self.assertEqual(result.local_only.selected_indices.tolist(), existing.selected_indices.tolist())

    def test_both_methods_match_independent_all_demand_greedy_objective(self):
        embeddings = self.fixture()
        affinity = independent_affinity(embeddings)
        regions = independent_regions(1, 2, (2, 2), (1, 2))
        result = compare_local_allocation(embeddings, 6, 1, 2, (2, 2), (1, 2), [3, 3])
        for method, budgets in ((result.local_only, None), (result.laplacian_quota, [3, 3])):
            floor, completion, selected = objective_oracle(affinity, regions, 6, budgets)
            self.assertEqual(method.floor_indices.tolist(), floor)
            self.assertEqual(method.completion_indices.tolist(), completion)
            self.assertEqual(method.selected_indices.tolist(), selected)
        # Greedy order is retained internally; sorting happens only at the end.
        order = result.local_only.completion_indices.tolist()
        self.assertNotEqual(order, sorted(order))

    def test_exact_regional_quotas_and_no_extra_quota_in_local_only(self):
        result = compare_local_allocation(torch.zeros(12, 3), 6, 1, 2, (2, 2), (1, 2), [3, 3])
        self.assertEqual(result.floor_indices.tolist(), [0, 4])
        self.assertEqual(result.local_only.completion_indices.tolist(), [1, 2, 3, 5])
        self.assertEqual(result.local_only.region_counts.tolist(), [[4, 2]])
        self.assertEqual(result.laplacian_quota.completion_indices.tolist(), [1, 2, 5, 6])
        self.assertEqual(result.laplacian_quota.region_counts.tolist(), [[3, 3]])

    def test_global_demands_remain_but_global_candidates_never_compete(self):
        embeddings = torch.zeros(12, 3)
        affinity = torch.zeros(12, 12)
        affinity[:4, 0] = 10
        affinity[4:8, 4] = 10
        affinity[8:, 2] = 2
        affinity[:, 9] = 100  # Would dominate if global candidates leaked in.
        result = compare_local_allocation(embeddings, 3, 1, 2, (2, 2), (1, 2), [2, 1], affinity)
        self.assertEqual(result.floor_indices.tolist(), [0, 4])
        for method in (result.local_only, result.laplacian_quota):
            self.assertEqual(method.completion_indices.tolist(), [2])
            self.assertEqual(method.selected_global, 0)
        # Without global demands all local gains tie and index 1 wins instead.
        _, local_demand_completion, _ = objective_oracle(affinity[:8], [[0, 1, 2, 3], [4, 5, 6, 7]], 3)
        self.assertEqual(local_demand_completion, [1])

    def test_global_coverage_best_is_shared_across_regions(self):
        embeddings = torch.zeros(12, 3)
        affinity = torch.zeros(12, 12)
        affinity[:4, 0] = 10
        affinity[4:8, 4] = 10
        affinity[8, 1] = 6
        affinity[8, 5] = 5  # Redundant after choosing token 1 in another region.
        affinity[9, 6] = 4
        result = compare_local_allocation(embeddings, 4, 1, 2, (2, 2), (1, 2), [2, 2], affinity)
        self.assertEqual(result.laplacian_quota.completion_indices.tolist(), [1, 6])

    def test_non_square_partition_exact_counts_unique_valid_sorted(self):
        embeddings = torch.randn(42, 5, generator=torch.Generator().manual_seed(18))
        regions = independent_regions(2, 3, (3, 2), (3, 4))
        budgets = [min(2, len(pool)) for pool in regions]
        k = sum(budgets)
        result = compare_local_allocation(embeddings, k, 2, 3, (3, 2), (3, 4), budgets)
        self.assertEqual(sorted(index for pool in regions for index in pool), list(range(36)))
        for method in (result.local_only, result.laplacian_quota):
            indices = method.selected_indices.tolist()
            self.assertEqual(len(indices), k)
            self.assertEqual(len(set(indices)), k)
            self.assertEqual(indices, sorted(indices))
            self.assertTrue(all(0 <= index < 36 for index in indices))
            counts = [sum(index in indices for index in pool) for pool in regions]
            self.assertEqual(method.region_counts.flatten().tolist(), counts)
            self.assertTrue(all(count >= 1 for count in counts))
        self.assertEqual(result.laplacian_quota.region_counts.flatten().tolist(), budgets)

    def test_k_equals_regions_and_full_local_capacity(self):
        for k, budgets in ((2, [1, 1]), (8, [4, 4])):
            with self.subTest(k=k):
                result = compare_local_allocation(self.fixture().half(), k, 1, 2, (2, 2), (1, 2), budgets)
                self.assertEqual(result.local_only.selected_indices.tolist(), result.laplacian_quota.selected_indices.tolist())
                if k == 2:
                    self.assertEqual(result.local_only.completion_indices.numel(), 0)
                else:
                    self.assertEqual(result.local_only.selected_indices.tolist(), list(range(8)))

    def test_determinism_and_no_input_mutation(self):
        embeddings = self.fixture()
        before = embeddings.clone()
        a = compare_local_allocation(embeddings, 6, 1, 2, (2, 2), (1, 2), [3, 3])
        b = compare_local_allocation(embeddings, 6, 1, 2, (2, 2), (1, 2), [3, 3])
        self.assertTrue(torch.equal(before, embeddings))
        for method in ("local_only", "laplacian_quota"):
            for field in ("floor_indices", "completion_indices", "selected_indices", "region_counts"):
                self.assertTrue(torch.equal(getattr(getattr(a, method), field), getattr(getattr(b, method), field)))

    def test_invalid_budgets(self):
        for budgets in ([0, 6], [-1, 7], [5, 1], [2, 2], [3], [[3], [3]]):
            with self.subTest(budgets=budgets), self.assertRaises(ValueError):
                compare_local_allocation(self.fixture(), 6, 1, 2, (2, 2), (1, 2), budgets)
        for budgets in ([3.0, 3.0], [True, True], [3 + 0j, 3 + 0j]):
            with self.subTest(budgets=budgets), self.assertRaises(TypeError):
                compare_local_allocation(self.fixture(), 6, 1, 2, (2, 2), (1, 2), budgets)

    def test_invalid_layout_capacity_or_floor(self):
        cases = [
            (12, 1, 1, 2, (2, 2), (1, 2), [1, 1]),  # G>K
            (12, 9, 1, 2, (2, 2), (1, 2), [4, 5]),  # K>N_local
            (11, 6, 1, 2, (2, 2), (1, 2), [3, 3]),
            (12, 6, 1, 2, (2, 2), (3, 1), [2, 2, 2]),  # empty region
            (12, 6, 0, 2, (2, 2), (1, 2), [3, 3]),
        ]
        for n, k, rows, cols, token_grid, region_grid, budgets in cases:
            with self.subTest(case=(n, k, rows, cols, token_grid, region_grid)), self.assertRaises(ValueError):
                compare_local_allocation(torch.ones(n, 3), k, rows, cols, token_grid, region_grid, budgets)

    def test_invalid_embeddings_and_affinity(self):
        invalid_embeddings = [torch.ones(12), torch.ones(12, 3, dtype=torch.long), torch.empty(12, 0),
                              torch.full((12, 3), float("nan")), torch.full((12, 3), float("inf"))]
        for embeddings in invalid_embeddings:
            with self.subTest(shape=tuple(embeddings.shape)), self.assertRaises(ValueError):
                compare_local_allocation(embeddings, 6, 1, 2, (2, 2), (1, 2), [3, 3])
        invalid_affinities = [torch.zeros(12, 11), torch.zeros(12, 12, dtype=torch.float16),
                              torch.full((12, 12), -1.0), torch.full((12, 12), float("inf")),
                              torch.full((12, 12), float("nan"))]
        for affinity in invalid_affinities:
            with self.subTest(shape=tuple(affinity.shape), dtype=affinity.dtype), self.assertRaises(ValueError):
                compare_local_allocation(self.fixture(), 6, 1, 2, (2, 2), (1, 2), [3, 3], affinity)
        for k in (True, 6.0):
            with self.subTest(k=k), self.assertRaises(TypeError):
                compare_local_allocation(self.fixture(), k, 1, 2, (2, 2), (1, 2), [3, 3])


if __name__ == "__main__":
    unittest.main()
