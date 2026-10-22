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

"""Validate matched sampling quotas and exclusion of global tokens from spatial stats."""

import unittest

import torch

from examples.diagnose_global_coverage import local_selection_lattice
from examples.diagnose_matched_random import describe_selection, matched_random_select


class MatchedRandomDiagnosticTest(unittest.TestCase):
    def test_exact_quotas_unique_range_and_order(self):
        for local_count, global_count in ((7, 3), (0, 3), (12, 0), (12, 4), (0, 0)):
            for seed in range(5):
                with self.subTest(local=local_count, global_count=global_count, seed=seed):
                    indices = matched_random_select(12, 4, local_count, global_count, seed)
                    k = local_count + global_count
                    self.assertEqual(indices.shape, (k,))
                    self.assertEqual(indices.dtype, torch.long)
                    self.assertEqual(indices.unique().numel(), k)
                    self.assertTrue(((indices >= 0) & (indices < 16)).all().item())
                    self.assertTrue(torch.equal(indices, indices.sort().values))
                    self.assertEqual((indices < 12).sum().item(), local_count)
                    self.assertEqual((indices >= 12).sum().item(), global_count)

    def test_deterministic_independent_streams_preserve_default_rng(self):
        state = torch.random.get_rng_state().clone()
        first = matched_random_select(384, 64, 97, 13, 3)
        again = matched_random_select(384, 64, 97, 13, 3)
        self.assertTrue(torch.equal(first, again))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), state))
        without_global = matched_random_select(384, 64, 97, 0, 3)
        without_local = matched_random_select(384, 64, 0, 13, 3)
        self.assertTrue(torch.equal(first[first < 384], without_global))
        self.assertTrue(torch.equal(first[first >= 384], without_local))

    def test_dynamic_boundary_and_global_exclusion(self):
        # Six local views of 64 tokens, followed by one global view.
        indices = matched_random_select(384, 64, 17, 13, 2)
        self.assertEqual((indices >= 384).sum().item(), 13)
        all_lattice = local_selection_lattice(indices, 2, 3, (8, 8))
        local_only = local_selection_lattice(indices[indices < 384], 2, 3, (8, 8))
        self.assertTrue(torch.equal(all_lattice, local_only))
        self.assertEqual(all_lattice.sum().item(), 17)
        result = describe_selection("Matched", indices, 384, 64, 30, 2, 3, (8, 8))
        self.assertEqual((result.local, result.global_count), (17, 13))
        for side in (4, 8):
            self.assertEqual(result.spatial[side].mean, 17 / side**2)

    def test_matched_spatial_mean_is_fixed_across_seeds(self):
        for seed in range(5):
            indices = matched_random_select(768, 64, 23, 9, seed)
            result = describe_selection("Matched", indices, 768, 64, 32, 3, 4, (8, 8))
            for side in (4, 8):
                self.assertEqual(result.spatial[side].mean, 23 / side**2)
                self.assertGreaterEqual(result.spatial[side].zero, max(0, side**2 - 23))

    def test_impossible_quotas_are_rejected(self):
        for local_count, global_count in ((13, 0), (0, 5), (-1, 1), (1, -1)):
            with self.subTest(local=local_count, global_count=global_count), self.assertRaises(ValueError):
                matched_random_select(12, 4, local_count, global_count, 0)


if __name__ == "__main__":
    unittest.main()
