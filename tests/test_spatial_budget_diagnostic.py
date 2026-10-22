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

"""Synthetic checks for multi-budget spatial statistics; no model inference."""

import math
import statistics
import unittest

import torch

from examples.diagnose_global_coverage import local_selection_lattice
from examples.diagnose_spatial_budgets import (
    BudgetResult, SelectionResult, region_counts, spatial_statistics, summary_row,
)


class SpatialBudgetDiagnosticTest(unittest.TestCase):
    def test_sample_standard_deviation_and_cv(self):
        result = spatial_statistics(torch.tensor([[0, 1], [2, 3]]))
        self.assertEqual((result.zero, result.minimum, result.maximum, result.mean), (1, 0, 3, 1.5))
        self.assertAlmostEqual(result.std, math.sqrt(5 / 3))
        self.assertAlmostEqual(result.cv, math.sqrt(5 / 3) / 1.5)

    def test_uniform_and_global_only_selection(self):
        uniform = spatial_statistics(torch.full((8, 8), 2))
        self.assertEqual((uniform.zero, uniform.std, uniform.cv), (0, 0., 0.))
        # 3x4 crops: [0,768) local and [768,832) global. Select global only.
        lattice = local_selection_lattice(torch.arange(768, 800), 3, 4, (8, 8))
        for side in (4, 8):
            result = spatial_statistics(region_counts(lattice, side))
            self.assertEqual((result.zero, result.mean, result.std), (side**2, 0., 0.))
            self.assertTrue(math.isnan(result.cv))

    def test_both_grids_preserve_non_square_lattice_counts(self):
        lattice = local_selection_lattice(torch.arange(832), 3, 4, (8, 8))
        for side in (4, 8):
            with self.subTest(side=side):
                counts = region_counts(lattice, side)
                self.assertEqual(counts.shape, (side, side))
                self.assertEqual(counts.sum().item(), 768)
                result = spatial_statistics(counts)
                self.assertEqual(result.mean, 768 / side**2)
                self.assertEqual(result.cv, 0.)

    def test_more_regions_than_budget_is_diagnosed(self):
        lattice = torch.zeros(32, 32, dtype=torch.long)
        for index in range(32):
            row, col = divmod(index, 8)
            lattice[row * 4, col * 4] = 1
        counts = region_counts(lattice, 8)
        result = spatial_statistics(counts)
        self.assertEqual(counts.sum().item(), 32)
        self.assertEqual(result.zero, 32)
        self.assertEqual(result.mean, .5)

    def test_summary_averages_per_seed_statistics(self):
        matrices = [torch.tensor([[0, 0], [0, 4]]), torch.ones(2, 2, dtype=torch.long),
                    torch.tensor([[0, 0], [2, 2]]), torch.tensor([[0, 1], [1, 2]]),
                    torch.tensor([[1, 1], [1, 1]])]
        random = []
        for seed, matrix in enumerate(matrices):
            spatial = {}
            for side in (4, 8):
                counts = torch.zeros(side, side, dtype=torch.long)
                counts.flatten()[:4] = matrix.flatten()
                spatial[side] = spatial_statistics(counts)
            random.append(SelectionResult(f"Random {seed}", float(seed), 4, 0, spatial))
        gc = SelectionResult("GC", 5., 4, 0, {side: random[0].spatial[side] for side in (4, 8)})
        budget = BudgetResult(32, [gc] + random, 2., 1., 3., 150.)
        rows = [summary_row("synthetic", budget, side) for side in (4, 8)]
        self.assertEqual(rows[0][4:7], rows[1][4:7])  # coverage independent of grid
        self.assertEqual((rows[0][-1], rows[1][-1]), ("True", "False"))
        self.assertAlmostEqual(float(rows[0][8]), statistics.mean(s.spatial[4].zero for s in random))
        self.assertAlmostEqual(float(rows[0][10]), statistics.mean(s.spatial[4].cv for s in random), places=6)


if __name__ == "__main__":
    unittest.main()
