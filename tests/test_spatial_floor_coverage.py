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

"""Synthetic spatial-floor checks; no model weights or image files are needed."""

import unittest

import torch
import torch.nn.functional as F

from src.pruning.spatial_floor_coverage import (
    local_region_indices,
    spatial_floor_coverage_select,
)


def independent_regions(rows, cols, token_grid, region_grid):
    """Assign sequence indices by coordinates, independently of the selector."""
    region_rows, region_cols = ((region_grid, region_grid)
                                if isinstance(region_grid, int) else region_grid)
    token_h, token_w = token_grid
    height, width = rows * token_h, cols * token_w
    row_edges = [round(i * height / region_rows) for i in range(region_rows + 1)]
    col_edges = [round(i * width / region_cols) for i in range(region_cols + 1)]
    result = [[] for _ in range(region_rows * region_cols)]
    for index in range(rows * cols * token_h * token_w):
        view, token = divmod(index, token_h * token_w)
        view_row, view_col = divmod(view, cols)
        token_row, token_col = divmod(token, token_w)
        y, x = view_row * token_h + token_row, view_col * token_w + token_col
        region_row = next(i for i in range(region_rows)
                          if row_edges[i] <= y < row_edges[i + 1])
        region_col = next(i for i in range(region_cols)
                          if col_edges[i] <= x < col_edges[i + 1])
        result[region_row * region_cols + region_col].append(index)
    return result


def independent_affinity(embeddings):
    vectors = F.normalize(embeddings.float(), p=2, dim=1, eps=1e-12)
    similarity = vectors @ vectors.T
    return (similarity - similarity.mean(dim=1, keepdim=True)).clamp_min(0)


def independent_floor(affinity, regions):
    # Candidate loop and explicit summation make the regional objective clear.
    return [max(region, key=lambda j: sum(float(affinity[i, j]) for i in region))
            for region in regions]


def independent_completion(affinity, floor, k):
    """Recompute F(S union {j}) explicitly, without vectorized gain updates."""
    selected = list(floor)
    completion = []
    for _ in range(k - len(floor)):
        candidate_scores = {}
        for candidate in range(affinity.shape[1]):
            if candidate not in selected:
                columns = selected + [candidate]
                candidate_scores[candidate] = sum(
                    max(float(affinity[demand, j]) for j in columns)
                    for demand in range(affinity.shape[0])
                )
        # F(S) is constant for every candidate in a step, so maximizing the
        # new objective is equivalent to maximizing marginal improvement.
        winner = max(candidate_scores, key=candidate_scores.get)
        selected.append(winner)
        completion.append(winner)
    return completion


class SpatialFloorCoverageTest(unittest.TestCase):
    @staticmethod
    def fixture():
        # Local views: indices 0:4 and 4:8. Global view: 8:12.
        # This fixture has unequal extra allocation and unsorted greedy order.
        return torch.randn(12, 6, generator=torch.Generator().manual_seed(0))

    def test_exact_k_unique_valid_sorted_on_non_square_layout(self):
        embeddings = torch.randn(42, 7, generator=torch.Generator().manual_seed(19))
        for k in (12, 16, 31, 42):
            with self.subTest(k=k):
                result = spatial_floor_coverage_select(
                    embeddings, k, rows=2, cols=3,
                    token_grid=(3, 2), region_grid=(3, 4),
                )
                indices = result.selected_indices
                self.assertEqual(indices.shape, (k,))
                self.assertEqual(indices.dtype, torch.long)
                self.assertEqual(indices.device, embeddings.device)
                self.assertEqual(indices.unique().numel(), k)
                self.assertTrue(((indices >= 0) & (indices < 42)).all().item())
                self.assertEqual(indices.tolist(), sorted(indices.tolist()))
                self.assertEqual(result.selected_local + result.selected_global, k)
                self.assertEqual(result.selected_local, int((indices < 36).sum()))
                self.assertEqual(result.selected_global, int((indices >= 36).sum()))

    def test_more_regions_than_budget_fails(self):
        with self.assertRaises(ValueError):
            spatial_floor_coverage_select(torch.eye(8), 3, 1, 1, (2, 2), 2)

    def test_empty_spatial_regions_fail_even_with_enough_budget(self):
        for grid in ((3, 2), (2, 3)):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                spatial_floor_coverage_select(torch.eye(8), 6, 1, 1, (2, 2), grid)

    def test_non_square_mapping_matches_independent_coordinate_partition(self):
        expected = independent_regions(2, 3, (3, 2), (3, 4))
        actual = local_region_indices(2, 3, (3, 2), (3, 4))
        self.assertEqual(len(actual), 12)
        self.assertEqual([region.tolist() for region in actual], expected)
        combined = torch.cat(actual)
        self.assertEqual(combined.unique().numel(), 36)
        self.assertEqual(combined.sort().values.tolist(), list(range(36)))
        # Near-equal partition has unequal capacities when six columns are
        # split into four regions, and view-major order differs from lattice.
        self.assertEqual(sorted({len(region) for region in expected}), [2, 4])

    def test_floor_indices_and_final_region_counts(self):
        embeddings = torch.randn(42, 5, generator=torch.Generator().manual_seed(99))
        result = spatial_floor_coverage_select(embeddings, 20, 2, 3, (3, 2), (3, 4))
        regions = independent_regions(2, 3, (3, 2), (3, 4))
        self.assertEqual(result.floor_indices.numel(), 12)
        self.assertEqual(result.floor_indices.unique().numel(), 12)
        self.assertTrue(((result.floor_indices >= 0) & (result.floor_indices < 36)).all())
        for region, representative in zip(regions, result.floor_indices.tolist()):
            self.assertIn(representative, region)
        selected = set(result.selected_indices.tolist())
        counts = [sum(index in selected for index in region) for region in regions]
        self.assertEqual(result.region_counts.dtype, torch.long)
        self.assertEqual(tuple(result.region_counts.shape), (3, 4))
        self.assertEqual(result.region_counts.flatten().tolist(), counts)
        self.assertTrue((result.region_counts >= 1).all().item())
        self.assertEqual(int(result.region_counts.sum()), result.selected_local)

    def test_k_equals_regions_returns_only_floor(self):
        result = spatial_floor_coverage_select(self.fixture(), 2, 1, 2, (2, 2), (1, 2))
        self.assertEqual(result.completion_indices.numel(), 0)
        self.assertEqual(result.selected_indices.tolist(), sorted(result.floor_indices.tolist()))
        self.assertEqual(result.selected_global, 0)
        self.assertEqual(result.region_counts.tolist(), [[1, 1]])

    def test_completion_can_choose_global_view_tokens(self):
        result = spatial_floor_coverage_select(self.fixture(), 6, 1, 2, (2, 2), (1, 2))
        self.assertGreater(result.selected_global, 0)
        self.assertIn(11, result.completion_indices.tolist())

    def test_completion_has_no_additional_regional_quota(self):
        result = spatial_floor_coverage_select(self.fixture(), 6, 1, 2, (2, 2), (1, 2))
        extras = result.completion_indices.tolist()
        self.assertGreaterEqual(sum(4 <= index < 8 for index in extras), 2)
        self.assertEqual(result.region_counts.tolist(), [[2, 3]])
        self.assertEqual(result.selected_global, 1)

    def test_regional_medoid_matches_explicit_affinity_objective(self):
        embeddings = torch.randn(42, 5, generator=torch.Generator().manual_seed(99))
        affinity = independent_affinity(embeddings)
        regions = independent_regions(2, 3, (3, 2), (3, 4))
        result = spatial_floor_coverage_select(embeddings, 12, 2, 3, (3, 2), (3, 4))
        self.assertEqual(result.floor_indices.tolist(), independent_floor(affinity, regions))

    def test_medoid_ties_choose_lowest_original_visual_index(self):
        for embeddings in (torch.ones(42, 5), torch.zeros(42, 5)):
            with self.subTest(zero_vectors=not embeddings.any().item()):
                regions = independent_regions(2, 3, (3, 2), (3, 4))
                result = spatial_floor_coverage_select(embeddings, 16, 2, 3, (3, 2), (3, 4))
                expected_floor = [min(region) for region in regions]
                self.assertEqual(result.floor_indices.tolist(), expected_floor)
                available = [i for i in range(42) if i not in expected_floor]
                self.assertEqual(result.completion_indices.tolist(), available[:4])

    def test_deterministic_input_has_identical_result(self):
        a = spatial_floor_coverage_select(self.fixture(), 6, 1, 2, (2, 2), (1, 2))
        b = spatial_floor_coverage_select(self.fixture(), 6, 1, 2, (2, 2), (1, 2))
        for field in ("selected_indices", "floor_indices", "completion_indices", "region_counts"):
            self.assertTrue(torch.equal(getattr(a, field), getattr(b, field)))
        self.assertEqual((a.selected_local, a.selected_global),
                         (b.selected_local, b.selected_global))

    def test_completion_matches_explicit_greedy_objective_and_keeps_order(self):
        embeddings = self.fixture()
        affinity = independent_affinity(embeddings)
        regions = independent_regions(1, 2, (2, 2), (1, 2))
        floor = independent_floor(affinity, regions)
        expected = independent_completion(affinity, floor, 6)
        result = spatial_floor_coverage_select(embeddings, 6, 1, 2, (2, 2), (1, 2))
        self.assertEqual(floor, [1, 4])
        self.assertEqual(expected, [2, 11, 6, 5])
        self.assertEqual(result.floor_indices.tolist(), floor)
        self.assertEqual(result.completion_indices.tolist(), expected)
        self.assertNotEqual(expected, sorted(expected))
        self.assertEqual(result.selected_indices.tolist(), sorted(floor + expected))

    def test_full_budget_and_fp16_input(self):
        embeddings = self.fixture().half()
        result = spatial_floor_coverage_select(embeddings, 12, 1, 2, (2, 2), (1, 2))
        self.assertEqual(result.selected_indices.tolist(), list(range(12)))
        self.assertEqual(result.region_counts.tolist(), [[4, 4]])
        self.assertEqual((result.selected_local, result.selected_global), (8, 4))

    def test_integer_region_grid_matches_tuple(self):
        embeddings = torch.randn(20, 5, generator=torch.Generator().manual_seed(12))
        a = spatial_floor_coverage_select(embeddings, 8, 2, 2, (2, 2), 2)
        b = spatial_floor_coverage_select(embeddings, 8, 2, 2, (2, 2), (2, 2))
        self.assertTrue(torch.equal(a.selected_indices, b.selected_indices))
        self.assertTrue(torch.equal(a.floor_indices, b.floor_indices))
        self.assertTrue(torch.equal(a.completion_indices, b.completion_indices))

    def test_invalid_layouts_and_population_fail(self):
        cases = [
            (torch.ones(12, 3), 4, 0, 2, (2, 2), 1),
            (torch.ones(12, 3), 4, 1, -2, (2, 2), 1),
            (torch.ones(12, 3), 4, 1, 2, (0, 2), 1),
            (torch.ones(12, 3), 4, 1, 2, (2, 2), 0),
            (torch.ones(12, 3), 4, 1, 2, (2, 2), (1, 0)),
            (torch.ones(11, 3), 4, 1, 2, (2, 2), (1, 2)),
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments[1:]), self.assertRaises(ValueError):
                spatial_floor_coverage_select(*arguments)

    def test_invalid_embeddings_and_budget_fail(self):
        for k in (-1, 13, 0):
            with self.subTest(k=k), self.assertRaises(ValueError):
                spatial_floor_coverage_select(self.fixture(), k, 1, 2, (2, 2), (1, 2))
        for k in (1.5, True):
            with self.subTest(k=k), self.assertRaises(TypeError):
                spatial_floor_coverage_select(self.fixture(), k, 1, 2, (2, 2), (1, 2))
        invalid = [torch.ones(12), torch.ones(12, 3, dtype=torch.long),
                   torch.empty(12, 0), torch.full((12, 3), float("nan")),
                   torch.full((12, 3), float("inf"))]
        for embeddings in invalid:
            with self.subTest(shape=embeddings.shape), self.assertRaises(ValueError):
                spatial_floor_coverage_select(embeddings, 4, 1, 2, (2, 2), (1, 2))


if __name__ == "__main__":
    unittest.main()
