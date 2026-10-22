# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dongeunk (original project contributions).
# Licensed under the Apache License, Version 2.0 (the License).
# You may not use this file except in compliance with the License.
# A copy is in the repository LICENSE and at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by law or agreed to in writing, distributed AS IS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
# See the License for permissions and limitations.
#
# Greedy-update structure references/adapts the MMTok selector.
# Relevant upstream copyright (greedy-update portions only):
# Copyright (c) 2025 Zoom Communications, Inc.
# Author: Sixun Dong
# Licensed under the Apache License, Version 2.0
# Paper: https://arxiv.org/abs/2508.18264
# Source: https://github.com/Ironieser/MMTok
# Commit: 8e2686836ed5bbe05d064f1d1c20e50c73c70b94
# File: mmtok/core/semantic_selector.py:greedy_merged_jit_kernel
# Local changes: row-centered clipped projected-feature affinity, unit
# visual demands, input checks; no P/temperature/alpha/JIT wrapper.
# See NOTICE and docs/LICENSING.md for attribution and change scope.
# Publication copy: license/attribution comments added; executable code unchanged.

"""Query-independent global representational coverage over projected embeddings."""

import operator

import torch
import torch.nn.functional as F


@torch.no_grad()
def global_coverage_select(visual_embeddings: torch.Tensor, k: int) -> torch.Tensor:
    """Return exactly k unique visual indices, sorted into original sequence order.

    Input is [N, D]. All N embeddings are demands and candidates with unit
    demand weights. Similarities are computed in float32, including for FP16
    inputs. Zero vectors normalize to zero. Ties choose the first available
    index. The result is a LongTensor on the input device.

    Greedy marginal updates follow MMTok's greedy_merged_jit_kernel structure;
    the affinity is row-mean-centered cosine similarity clipped at zero.
    Runtime is O(K*N^2) and working memory is O(N^2).
    """
    if visual_embeddings.ndim != 2 or not visual_embeddings.is_floating_point():
        raise ValueError("visual_embeddings must be a floating-point tensor of shape [N, D]")
    n, dimension = visual_embeddings.shape
    if dimension == 0:
        raise ValueError("Embedding dimension D must be positive")
    if isinstance(k, bool):
        raise TypeError("k must be an integer")
    k = operator.index(k)
    if not 0 <= k <= n:
        raise ValueError(f"k must satisfy 0 <= k <= N={n}; got {k}")
    embeddings = visual_embeddings.float()
    if not torch.isfinite(embeddings).all().item():
        raise ValueError("visual_embeddings must be finite and representable in float32")
    selected = torch.empty(k, dtype=torch.long, device=embeddings.device)
    if k == 0:
        return selected

    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        v = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
        similarity = v @ v.T
        affinity = (similarity - similarity.mean(dim=1, keepdim=True)).clamp_min_(0)
    best = torch.zeros(n, dtype=affinity.dtype, device=affinity.device)
    candidate_mask = torch.zeros_like(best)
    for step in range(k):
        gain = (affinity - best.unsqueeze(1)).clamp_min_(0).sum(dim=0)
        gain.add_(candidate_mask)
        index = torch.argmax(gain).view(1)
        selected[step] = index[0]
        best = torch.maximum(best, affinity.index_select(1, index).squeeze(1))
        candidate_mask.index_fill_(0, index, float("-inf"))

    # Preserve greedy state throughout selection; restore sequence order only now.
    return selected.sort().values
