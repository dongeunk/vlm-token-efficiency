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

"""Synthetic correctness checks; no model files or external data are needed."""

import unittest

import torch
import torch.nn.functional as F

from src.pruning.coverage import global_coverage_select


class GlobalCoverageTest(unittest.TestCase):
    def test_exact_count_unique_valid_sorted(self):
        embeddings = torch.randn(13, 5, generator=torch.Generator().manual_seed(42))
        for k in (0, 1, 3, 8, 13):
            with self.subTest(k=k):
                indices = global_coverage_select(embeddings, k)
                self.assertEqual(indices.shape, (k,))
                self.assertEqual(indices.dtype, torch.long)
                self.assertEqual(indices.device, embeddings.device)
                self.assertEqual(indices.unique().numel(), k)
                self.assertTrue(((indices >= 0) & (indices < 13)).all().item())
                self.assertTrue(torch.equal(indices, indices.sort().values))

    def test_duplicate_clusters_cover_distinct_representations(self):
        # Unequal cluster sizes: eight identical, three orthogonal, two opposite.
        centers = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]])
        embeddings = torch.repeat_interleave(centers, torch.tensor([8, 3, 2]), dim=0)
        indices = global_coverage_select(embeddings, 3)
        self.assertEqual(embeddings[indices].unique(dim=0).shape[0], 3)
        self.assertTrue(torch.equal(indices, torch.tensor([0, 8, 11])))

    def test_zero_gain_still_returns_unique_indices(self):
        # Every affinity and gain is zero; the exclusion mask must still work.
        for embeddings in (torch.ones(7, 3), torch.zeros(7, 3)):
            with self.subTest(zero_vectors=not embeddings.any().item()):
                self.assertTrue(torch.equal(
                    global_coverage_select(embeddings, 4), torch.arange(4)
                ))

    def test_full_budget_preserves_all_indices(self):
        embeddings = torch.eye(5, dtype=torch.float16)
        self.assertTrue(torch.equal(global_coverage_select(embeddings, 5), torch.arange(5)))

    def test_empty_input_zero_budget(self):
        self.assertEqual(global_coverage_select(torch.empty(0, 3), 0).numel(), 0)

    def test_invalid_input_and_budget(self):
        for k in (-1, 4):
            with self.subTest(k=k), self.assertRaises(ValueError):
                global_coverage_select(torch.eye(3), k)
        for embeddings in (torch.ones(3), torch.ones(3, 2, dtype=torch.long),
                           torch.empty(3, 0), torch.tensor([[float("nan")]])):
            with self.subTest(shape=embeddings.shape), self.assertRaises(ValueError):
                global_coverage_select(embeddings, 1)
        with self.assertRaises(TypeError):
            global_coverage_select(torch.eye(3), 1.5)

    def test_matches_explicit_coverage_objective(self):
        # Independently evaluate F(S union {j}) - F(S) for each candidate.
        embeddings = torch.tensor([
            [1., 0., 0.], [.9, .2, .1], [0., 1., 0.],
            [0., 0., 1.], [-.4, -.6, .2],
        ])
        v = F.normalize(embeddings, dim=1)
        matrix = v @ v.T
        affinity = (matrix - matrix.mean(dim=1, keepdim=True)).clamp(min=0)
        chosen = []
        for _ in range(3):
            current = affinity[:, chosen].max(dim=1).values.sum().item() if chosen else 0.
            gains = {}
            for candidate in range(len(embeddings)):
                if candidate not in chosen:
                    value = affinity[:, chosen + [candidate]].max(dim=1).values.sum().item()
                    gains[candidate] = value - current
            chosen.append(max(gains, key=gains.get))
        self.assertTrue(torch.equal(
            global_coverage_select(embeddings, 3), torch.tensor(sorted(chosen))
        ))


if __name__ == "__main__":
    unittest.main()
