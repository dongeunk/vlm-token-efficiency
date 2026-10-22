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
# Completion follows the local coverage baseline with local-only
# candidates and externally supplied quota constraints.
# See NOTICE and docs/LICENSING.md for attribution and change scope.
# Publication copy: license/attribution comments added; executable code unchanged.

"""Controlled local-only comparison of free versus regional-quota coverage.

Both methods use the existing calibrated affinity, every visual token as a unit
weight demand, local tokens as candidates, and the same minimum-one medoid
floor. Only the phase-two candidate mask changes: the quota method masks a
region once its prescribed final budget is reached. This is an allocation
ablation, not full S2Prune: it uses neither recursive cells nor ERC scores.
"""

from dataclasses import dataclass
from time import perf_counter

import torch

from src.pruning.spatial_floor_coverage import (
    SpatialFloorCoverageResult,
    _grid_pair,
    _integer,
    _positive,
    calibrated_affinity,
    local_region_indices,
)


@dataclass(frozen=True)
class MatchedAllocationComparison:
    """Shared comparison inputs and the two controlled selection results.

    Affinity is retained by reference and treated as read-only. Demand and
    candidate indices make their different populations explicit. Both method
    results share the exact same floor_indices tensor.
    """

    affinity: torch.Tensor
    demand_indices: torch.Tensor
    candidate_indices: torch.Tensor
    floor_indices: torch.Tensor
    region_budgets: torch.Tensor
    local_only: SpatialFloorCoverageResult
    laplacian_quota: SpatialFloorCoverageResult
    shared_floor_seconds: float
    local_only_seconds: float
    laplacian_quota_seconds: float


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _validate_affinity(affinity, n, device):
    if not isinstance(affinity, torch.Tensor):
        raise TypeError("affinity must be a torch.Tensor")
    if affinity.shape != (n, n) or affinity.dtype != torch.float32:
        raise ValueError(f"affinity must be FP32 with shape [{n}, {n}]")
    if affinity.device != device:
        raise ValueError("affinity and visual_embeddings must be on the same device")
    if not torch.isfinite(affinity).all().item() or (affinity < 0).any().item():
        raise ValueError("affinity must contain finite nonnegative values")


def _validate_budgets(region_budgets, region_shape, capacities, k, device):
    budgets = torch.as_tensor(region_budgets, device=device)
    if budgets.dtype == torch.bool or budgets.is_floating_point() or budgets.is_complex():
        raise TypeError("region_budgets must contain integers")
    g = region_shape[0] * region_shape[1]
    if budgets.shape not in ((g,), region_shape):
        raise ValueError(f"region_budgets must have shape [{g}] or {region_shape}")
    budgets = budgets.to(dtype=torch.long).reshape(-1)
    if (budgets < 1).any().item():
        raise ValueError("Each regional final budget must be at least one")
    if (budgets > capacities).any().item():
        raise ValueError("Regional final budgets must not exceed their candidate capacities")
    if int(budgets.sum().item()) != k:
        raise ValueError(f"Regional final budgets must sum to K={k}")
    return budgets


def _complete(affinity, floor, regions, token_region, k, region_shape, budgets):
    """Run the identical all-demand greedy update, optionally with quota masks."""
    n_local = token_region.numel()
    g = len(regions)
    # Keep all N demand rows, including the global view; only columns are local.
    local_affinity = affinity[:, :n_local]
    best = affinity.index_select(1, floor).max(dim=1).values
    candidate_mask = torch.zeros(n_local, dtype=affinity.dtype, device=affinity.device)
    candidate_mask.index_fill_(0, floor, float("-inf"))
    counts = torch.ones(g, dtype=torch.long, device=affinity.device)
    if budgets is not None:
        candidate_mask.masked_fill_(counts[token_region] >= budgets[token_region], float("-inf"))

    completion = torch.empty(k - g, dtype=torch.long, device=affinity.device)
    for step in range(k - g):
        gain = (local_affinity - best.unsqueeze(1)).clamp_min_(0).sum(dim=0)
        gain.add_(candidate_mask)
        index = gain.argmax().view(1)
        if not torch.isfinite(gain[index]).all().item():
            raise RuntimeError("No feasible unselected local candidate remains")
        completion[step] = index[0]
        # Global best is updated after every choice across every region.
        best = torch.maximum(best, affinity.index_select(1, index).squeeze(1))
        candidate_mask.index_fill_(0, index, float("-inf"))
        counts[token_region[index]] += 1
        if budgets is not None:
            candidate_mask.masked_fill_(counts[token_region] >= budgets[token_region], float("-inf"))

    selected = torch.cat((floor, completion)).sort().values
    if selected.unique().numel() != k or not ((selected >= 0) & (selected < n_local)).all().item():
        raise RuntimeError("Selection violated exact-count, uniqueness, or local-only invariants")
    actual_counts = torch.bincount(token_region[selected], minlength=g)
    if not torch.equal(counts, actual_counts) or not (actual_counts >= 1).all().item():
        raise RuntimeError("Selection violated the shared minimum-one floor")
    if budgets is not None and not torch.equal(actual_counts, budgets):
        raise RuntimeError("Selection did not meet the exact regional final budgets")
    return SpatialFloorCoverageResult(
        selected, floor, completion, k, 0, actual_counts.reshape(region_shape),
    )


@torch.no_grad()
def compare_local_allocation(visual_embeddings: torch.Tensor, k: int, rows: int,
                             cols: int, token_grid, region_grid, region_budgets,
                             affinity: torch.Tensor | None = None) -> MatchedAllocationComparison:
    """Compare the two local-only methods with a shared affinity and floor.

    ``region_budgets`` contains final B_g, including the one floor token. Its
    region order matches local_region_indices (region row-major). All budgets
    must be >=1, <= regional capacity, and sum to k. A supplied affinity must be
    the same FP32 calibrated affinity used for the comparison; it is validated
    and never modified. Otherwise it is computed once using the existing helper.

    Crops and their tokens are view-major; there is one final global view. The
    global view remains a demand and can never be a selection candidate or be
    assigned to a region. Ties choose the lowest original candidate index.
    Floor/completion tensors retain their phase order; only final indices sort.

    Timing fields are descriptive single-run wall times in seconds, with CUDA
    synchronized immediately before and after each phase. There is no warm-up
    or repetition, so these values do not support a method speed comparison.
    Method timings cover their completion and result checks; affinity and the
    shared medoid floor are excluded and the latter is reported separately.
    """
    if not isinstance(visual_embeddings, torch.Tensor):
        raise TypeError("visual_embeddings must be a torch.Tensor")
    if visual_embeddings.ndim != 2 or not visual_embeddings.is_floating_point():
        raise ValueError("visual_embeddings must be floating-point with shape [N, D]")
    n, dimension = visual_embeddings.shape
    if dimension == 0 or not torch.isfinite(visual_embeddings.float()).all().item():
        raise ValueError("visual_embeddings must be finite with positive embedding dimension")
    rows, cols = _positive(rows, "rows"), _positive(cols, "cols")
    token_h, token_w = _grid_pair(token_grid, "token_grid")
    region_shape = _grid_pair(region_grid, "region_grid")
    g = region_shape[0] * region_shape[1]
    k = _integer(k, "k")
    n_local = rows * cols * token_h * token_w
    expected_n = n_local + token_h * token_w
    if n != expected_n:
        raise ValueError(f"Layout requires N={expected_n} local+global tokens; got {n}")
    if not g <= k <= n_local:
        raise ValueError(f"Local-only floor requires G={g} <= K={k} <= N_local={n_local}")
    regions = local_region_indices(
        rows, cols, (token_h, token_w), region_shape, visual_embeddings.device,
    )
    capacities = torch.tensor([pool.numel() for pool in regions], device=visual_embeddings.device)
    budgets = _validate_budgets(region_budgets, region_shape, capacities, k, visual_embeddings.device)
    if affinity is None:
        affinity = calibrated_affinity(visual_embeddings)
    _validate_affinity(affinity, n, visual_embeddings.device)

    # Identical phase 1 and medoid tie rule to the existing floor selector.
    _synchronize(affinity.device)
    phase_start = perf_counter()
    floor = torch.empty(g, dtype=torch.long, device=affinity.device)
    token_region = torch.empty(n_local, dtype=torch.long, device=affinity.device)
    for region, pool in enumerate(regions):
        regional_scores = affinity.index_select(0, pool).index_select(1, pool).sum(dim=0)
        floor[region] = pool[regional_scores.argmax()]
        token_region[pool] = region
    _synchronize(affinity.device)
    shared_floor_seconds = perf_counter() - phase_start

    _synchronize(affinity.device)
    phase_start = perf_counter()
    local_only = _complete(affinity, floor, regions, token_region, k, region_shape, None)
    _synchronize(affinity.device)
    local_only_seconds = perf_counter() - phase_start

    _synchronize(affinity.device)
    phase_start = perf_counter()
    laplacian_quota = _complete(affinity, floor, regions, token_region, k, region_shape, budgets)
    _synchronize(affinity.device)
    laplacian_quota_seconds = perf_counter() - phase_start
    return MatchedAllocationComparison(
        affinity, torch.arange(n, device=affinity.device),
        torch.arange(n_local, device=affinity.device), floor,
        budgets.reshape(region_shape), local_only, laplacian_quota,
        shared_floor_seconds, local_only_seconds, laplacian_quota_seconds,
    )
