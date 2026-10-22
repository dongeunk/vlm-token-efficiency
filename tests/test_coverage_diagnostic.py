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

"""Check diagnostic affinity and spatial mapping without model inference."""

from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from examples.diagnose_global_coverage import (
    coarse_region_counts, coverage_affinity, coverage_score,
    local_selection_lattice, near_equal_edges, projected_grid_shape,
)


class CoverageDiagnosticTest(unittest.TestCase):
    def test_affinity_matches_selector_float32_operations(self):
        embeddings = torch.tensor([[1., 2.], [-2., 1.], [0., 0.]], dtype=torch.float16)
        v = F.normalize(embeddings.float(), p=2, dim=1, eps=1e-12)
        similarity = v @ v.T
        expected = (similarity - similarity.mean(dim=1, keepdim=True)).clamp_min_(0)
        self.assertTrue(torch.equal(coverage_affinity(embeddings), expected))

    def test_coverage_objective_uses_all_demand_rows(self):
        affinity = torch.tensor([[2., 1., 0.], [0., 3., 4.], [5., 0., 1.]])
        self.assertEqual(coverage_score(affinity, torch.tensor([0, 2])), 11.)
        self.assertEqual(coverage_score(affinity, torch.empty(0, dtype=torch.long)), 0.)

    def test_non_square_layout_and_global_boundary(self):
        # 2x3 crops, 8x8 tokens/crop: local [0,384), global [384,448).
        indices = torch.tensor([0, 63, 64, 191, 192, 383, 384, 447])
        lattice = local_selection_lattice(indices, 2, 3, (8, 8))
        expected = torch.zeros((16, 24), dtype=torch.long)
        for row, col in ((0, 0), (7, 7), (0, 8), (7, 23), (8, 0), (15, 23)):
            expected[row, col] = 1
        self.assertTrue(torch.equal(lattice, expected))
        self.assertEqual(lattice.sum().item(), 6)  # global indices excluded
        with self.assertRaises(ValueError):
            local_selection_lattice(torch.tensor([448]), 2, 3, (8, 8))

    def test_second_non_square_layout(self):
        lattice = local_selection_lattice(torch.tensor([128, 383, 384]), 3, 2, (8, 8))
        self.assertEqual(lattice.shape, (24, 16))
        self.assertEqual(lattice[8, 0].item(), 1)
        self.assertEqual(lattice[23, 15].item(), 1)
        self.assertEqual(lattice.sum().item(), 2)

    def test_near_equal_partition_and_zero_region_coordinates(self):
        self.assertEqual(near_equal_edges(5), [0, 1, 2, 4, 5])
        self.assertEqual(near_equal_edges(7), [0, 2, 4, 5, 7])
        self.assertTrue(torch.equal(coarse_region_counts(torch.ones(5, 7, dtype=torch.long)),
                                    torch.tensor([[2, 2, 1, 2], [2, 2, 1, 2],
                                                  [4, 4, 2, 4], [2, 2, 1, 2]])))
        lattice = torch.zeros(5, 7, dtype=torch.long)
        lattice[0, 0] = 1
        lattice[4, 6] = 2
        counts = coarse_region_counts(lattice)
        zeros = (counts == 0).nonzero().tolist()
        self.assertEqual(counts.sum().item(), 3)
        self.assertEqual(len(zeros), 14)
        self.assertNotIn([0, 0], zeros)
        self.assertNotIn([3, 3], zeros)

    def test_config_and_observed_token_grid(self):
        config = SimpleNamespace(vision_config=SimpleNamespace(image_size=512, patch_size=16), scale_factor=4)
        self.assertEqual(projected_grid_shape(config, (1, 9, 3, 512, 512), 64), (8, 8))
        for shape, tokens in (((1, 9, 3, 512, 512), 63), ((1, 9, 3, 256, 512), 32)):
            with self.assertRaises(RuntimeError):
                projected_grid_shape(config, shape, tokens)


if __name__ == "__main__":
    unittest.main()
