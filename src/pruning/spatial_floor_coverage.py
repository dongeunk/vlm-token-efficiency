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
# Completion follows the local coverage baseline; regional floor and
# all-candidate completion are project-specific extensions.
# See NOTICE and docs/LICENSING.md for attribution and change scope.
# Publication copy: license/attribution comments added; executable code unchanged.

"""Local minimum-one medoid floor followed by unrestricted global coverage."""

from dataclasses import dataclass
import operator

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SpatialFloorCoverageResult:
    selected_indices: torch.Tensor
    floor_indices: torch.Tensor
    completion_indices: torch.Tensor
    selected_local: int
    selected_global: int
    region_counts: torch.Tensor


def _integer(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return operator.index(value)


def _positive(value, name):
    value = _integer(value, name)
    if value < 1:
        raise ValueError(f"{name} must be positive; got {value}")
    return value


def _grid_pair(value, name):
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two dimensions")
        return _positive(value[0], name), _positive(value[1], name)
    side = _positive(value, name)
    return side, side


@torch.no_grad()
def calibrated_affinity(visual_embeddings: torch.Tensor) -> torch.Tensor:
    """Match coverage.py exactly: FP32 cosine, row mean subtraction, clamp.

    All local and global tokens are both demands (rows) and candidates (columns).
    After row calibration this matrix is generally not symmetric.
    """
    if visual_embeddings.ndim != 2 or not visual_embeddings.is_floating_point():
        raise ValueError("visual_embeddings must be a floating-point tensor of shape [N, D]")
    if visual_embeddings.shape[0] == 0 or visual_embeddings.shape[1] == 0:
        raise ValueError("Embedding dimensions N and D must be positive")
    embeddings = visual_embeddings.float()
    if not torch.isfinite(embeddings).all().item():
        raise ValueError("visual_embeddings must be finite and representable in float32")
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        v = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
        similarity = v @ v.T
        return (similarity - similarity.mean(dim=1, keepdim=True)).clamp_min_(0)


def local_region_indices(rows, cols, token_grid, region_grid, device="cpu"):
    """Return local candidate pools in region row-major order.

    Near-equal boundaries are round(i * size / parts). Crops and tokens inside
    crops are row-major, but the full sequence is view-major. Each pool is
    ordered by original visual index solely to give medoid ties a fixed rule.
    """
    rows, cols = _positive(rows, "rows"), _positive(cols, "cols")
    token_h, token_w = _grid_pair(token_grid, "token_grid")
    region_h, region_w = _grid_pair(region_grid, "region_grid")
    height, width = rows * token_h, cols * token_w
    if region_h > height or region_w > width:
        raise ValueError(
            f"Region grid {region_h}x{region_w} creates empty regions in lattice {height}x{width}"
        )
    n_local = rows * cols * token_h * token_w
    lattice = torch.arange(n_local, device=device).reshape(rows, cols, token_h, token_w)
    lattice = lattice.permute(0, 2, 1, 3).reshape(height, width)
    row_edges = [round(i * height / region_h) for i in range(region_h + 1)]
    col_edges = [round(i * width / region_w) for i in range(region_w + 1)]
    regions = []
    for row in range(region_h):
        for col in range(region_w):
            pool = lattice[row_edges[row]:row_edges[row + 1], col_edges[col]:col_edges[col + 1]]
            if pool.numel() == 0:
                raise ValueError("Requested region partition contains an empty region")
            regions.append(pool.reshape(-1).sort().values)
    return tuple(regions)


@torch.no_grad()
def spatial_floor_coverage_select(visual_embeddings: torch.Tensor, k: int, rows: int,
                                 cols: int, token_grid, region_grid) -> SpatialFloorCoverageResult:
    """Seed one local medoid per region, then complete K using global gains.

    The layout must describe rows*cols local views followed by one global view.
    Floor indices retain region row-major order; completion indices retain greedy
    order. Only the final union is sorted into original sequence order. No extra
    regional quota is imposed. Runtime O(K*N^2), working memory O(N^2).
    """
    if visual_embeddings.ndim != 2 or not visual_embeddings.is_floating_point():
        raise ValueError("visual_embeddings must be a floating-point tensor of shape [N, D]")
    rows, cols = _positive(rows, "rows"), _positive(cols, "cols")
    token_h, token_w = _grid_pair(token_grid, "token_grid")
    region_h, region_w = _grid_pair(region_grid, "region_grid")
    k = _integer(k, "k")
    n = visual_embeddings.shape[0]
    n_local = rows * cols * token_h * token_w
    expected_n = n_local + token_h * token_w
    if n != expected_n:
        raise ValueError(f"Layout requires N={expected_n} local+global tokens; got N={n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must satisfy 0 <= k <= N={n}; got {k}")
    g = region_h * region_w
    if g > k:
        raise ValueError(f"Spatial floor infeasible: G={g} exceeds K={k}")
    regions = local_region_indices(rows, cols, (token_h, token_w), (region_h, region_w), visual_embeddings.device)
    affinity = calibrated_affinity(visual_embeddings)

    # Phase 1: one actual local token per region, maximizing regional demand sum.
    # Pool indices are ascending, so argmax ties choose the lowest visual index.
    floor = torch.empty(g, device=affinity.device, dtype=torch.long)
    for region, pool in enumerate(regions):
        regional_scores = affinity.index_select(0, pool).index_select(1, pool).sum(dim=0)
        floor[region] = pool[regional_scores.argmax()]

    # Phase 2: every unselected local/global candidate competes for every extra
    # position using the same marginal gain. The floor already covers all demands.
    best = affinity.index_select(1, floor).max(dim=1).values
    candidate_mask = torch.zeros(n, dtype=affinity.dtype, device=affinity.device)
    candidate_mask.index_fill_(0, floor, float("-inf"))
    completion = torch.empty(k - g, device=affinity.device, dtype=torch.long)
    for step in range(k - g):
        gain = (affinity - best.unsqueeze(1)).clamp_min_(0).sum(dim=0)
        gain.add_(candidate_mask)
        index = gain.argmax().view(1)
        completion[step] = index[0]
        best = torch.maximum(best, affinity.index_select(1, index).squeeze(1))
        candidate_mask.index_fill_(0, index, float("-inf"))

    selected = torch.cat((floor, completion)).sort().values
    if selected.unique().numel() != k or floor.unique().numel() != g or not (floor < n_local).all().item():
        raise RuntimeError("Spatial floor selection violated count/uniqueness/local-index invariants")
    selected_local_indices = selected[selected < n_local]
    local_mask = torch.zeros(n_local, dtype=torch.bool, device=affinity.device)
    local_mask[selected_local_indices] = True
    counts = torch.stack([local_mask.index_select(0, pool).sum() for pool in regions]).reshape(region_h, region_w)
    if not (counts >= 1).all().item():
        raise RuntimeError("Final selection failed the minimum-one floor")
    selected_local = selected_local_indices.numel()
    return SpatialFloorCoverageResult(selected, floor, completion, selected_local, k - selected_local, counts)
