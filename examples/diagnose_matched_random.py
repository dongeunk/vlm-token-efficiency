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

"""Separate local/global allocation from spatial allocation within local views.

Compares GC, Standard Random, and Matched-L/G Random on cached SmolVLM
embeddings. No coverage objective evaluation or generation is performed.
Matched sampling uses independent CPU streams: local seed=2*s, global seed=2*s+1.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import statistics
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from examples.diagnose_global_coverage import (
    FILENAMES, MODEL_ID, RANDOM_SEEDS, local_selection_lattice, projected_grid_shape,
)
from examples.diagnose_spatial_budgets import (
    BUDGETS, GRID_SIDES, SpatialStats, print_table, region_counts, spatial_statistics,
)
from src.pruning.coverage import global_coverage_select


def validate_selection(indices, n: int, k: int):
    if indices.ndim != 1 or indices.dtype != torch.long or indices.numel() != k:
        raise RuntimeError("Selection must contain exactly K LongTensor indices")
    if indices.unique().numel() != k:
        raise RuntimeError("Selection contains duplicate indices")
    if not ((indices >= 0) & (indices < n)).all().item():
        raise RuntimeError("Selection contains indices outside [0, N)")
    if not torch.equal(indices, indices.sort().values):
        raise RuntimeError("Selection does not preserve original sequence order")


def matched_random_select(n_local: int, n_global: int, local_count: int,
                          global_count: int, seed: int, device="cpu") -> torch.Tensor:
    """Uniform sampling without replacement from each pool using separate streams."""
    if n_local < 0 or n_global < 0 or not 0 <= local_count <= n_local or not 0 <= global_count <= n_global:
        raise ValueError("Requested local/global quotas must fit their candidate pools")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    local_generator = torch.Generator(device="cpu").manual_seed(2 * seed)
    global_generator = torch.Generator(device="cpu").manual_seed(2 * seed + 1)
    local = torch.randperm(n_local, generator=local_generator)[:local_count]
    global_indices = n_local + torch.randperm(n_global, generator=global_generator)[:global_count]
    indices = torch.cat((local, global_indices)).sort().values.to(device)
    k = local_count + global_count
    validate_selection(indices, n_local + n_global, k)
    if (indices < n_local).sum().item() != local_count or (indices >= n_local).sum().item() != global_count:
        raise RuntimeError("Matched random did not preserve the exact local/global quotas")
    return indices


@dataclass
class AllocationResult:
    label: str
    local: int
    global_count: int
    spatial: dict[int, SpatialStats]


def describe_selection(label, indices, n_local, n_global, k, rows, cols, token_grid):
    validate_selection(indices, n_local + n_global, k)
    local = (indices < n_local).sum().item()
    lattice = local_selection_lattice(indices, rows, cols, token_grid)
    if lattice.sum().item() != local:
        raise RuntimeError("Spatial mapping lost local tokens or counted global tokens")
    spatial = {}
    for side in GRID_SIDES:
        counts = region_counts(lattice, side)
        stats = spatial_statistics(counts)
        if stats.zero < max(0, side**2 - local):
            raise RuntimeError("Zero-region count violates the local token count lower bound")
        spatial[side] = stats
    return AllocationResult(label, local, k - local, spatial)


def summary_row(filename, k, side, n_local, n_global, gc, standard, matched):
    population_ratio = n_local / (n_local + n_global)
    retention_ratio = gc.local / k
    return (
        filename, k, f"{side}x{side}", f"{gc.local}/{gc.global_count}", gc.spatial[side].zero,
        f"{statistics.mean(s.spatial[side].zero for s in standard):.4f}",
        f"{statistics.mean(s.spatial[side].zero for s in matched):.4f}",
        f"{gc.spatial[side].cv:.6f}",
        f"{statistics.mean(s.spatial[side].cv for s in standard):.6f}",
        f"{statistics.mean(s.spatial[side].cv for s in matched):.6f}",
        f"{retention_ratio:.6f}", f"{population_ratio:.6f}",
        f"{(population_ratio - retention_ratio) * 100:+.4f}",
    )


@torch.inference_mode()
def evaluate_budget(filename, embeddings, rows, cols, token_grid, k):
    n_local = rows * cols * token_grid[0] * token_grid[1]
    n_global = token_grid[0] * token_grid[1]
    if embeddings.shape[0] != n_local + n_global or not 0 < k <= embeddings.shape[0]:
        raise ValueError("Candidate count or token budget disagrees with actual view layout")
    gc_indices = global_coverage_select(embeddings, k)
    gc = describe_selection("GC", gc_indices, n_local, n_global, k, rows, cols, token_grid)
    standard, matched = [], []
    for seed in RANDOM_SEEDS:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        standard_indices = torch.randperm(embeddings.shape[0], generator=generator)[:k].sort().values.to(embeddings.device)
        standard.append(describe_selection(
            f"Standard {seed}", standard_indices, n_local, n_global, k, rows, cols, token_grid,
        ))
        matched_indices = matched_random_select(
            n_local, n_global, gc.local, gc.global_count, seed, embeddings.device,
        )
        matched.append(describe_selection(
            f"Matched {seed}", matched_indices, n_local, n_global, k, rows, cols, token_grid,
        ))
    population_ratio = n_local / embeddings.shape[0]
    retention_ratio = gc.local / k
    excess_pp = (population_ratio - retention_ratio) * 100
    print(f"\nK={k}; GC local/global={gc.local}/{gc.global_count}; L/K={retention_ratio:.6f}; "
          f"N_local/N={population_ratio:.6f}")
    print(f"GC global-view fraction minus population global fraction: {excess_pp:+.4f} percentage points")
    print("Matched quota/unique/range/order validation: PASS for seeds 0..4")
    for side in GRID_SIDES:
        print(f"Grid {side}x{side}; G={side**2}; statistics include local tokens only")
        detail = []
        for selection in [gc] + standard + matched:
            s = selection.spatial[side]
            detail.append((selection.label, f"{selection.local}/{selection.global_count}",
                           s.zero, s.minimum, s.maximum, f"{s.mean:.6f}", f"{s.std:.6f}", f"{s.cv:.6f}"))
        print_table(("Selector", "Selected L/G", "Zero", "Min", "Max", "Mean", "Sample std", "CV"), detail)
    sys.stdout.flush()
    return [summary_row(filename, k, side, n_local, n_global, gc, standard, matched) for side in GRID_SIDES]


@torch.inference_mode()
def diagnose_image(processor, model, path, device):
    with Image.open(path) as source:
        original_size = source.size
        image = source.convert("RGB")
    image_inputs = processor.image_processor(
        images=[image], return_row_col_info=True, return_tensors="pt",
    ).to(device)
    rows, cols = int(image_inputs["rows"][0][0]), int(image_inputs["cols"][0][0])
    projected = model.get_image_features(
        pixel_values=image_inputs["pixel_values"], pixel_attention_mask=image_inputs.get("pixel_attention_mask"),
        return_dict=True,
    ).pooler_output
    views, tokens_per_view, _ = projected.shape
    token_grid = projected_grid_shape(model.config, image_inputs["pixel_values"].shape, tokens_per_view)
    if views != rows * cols + 1 or views != image_inputs["pixel_values"].shape[1]:
        raise RuntimeError("Processor metadata disagrees with actual local/global view count")
    embeddings = projected.flatten(0, 1)
    if max(BUDGETS) > embeddings.shape[0]:
        raise RuntimeError(f"{path.name}: N={embeddings.shape[0]} is smaller than K={max(BUDGETS)}")
    print(f"\nImage: {path.name}; size(WxH): {original_size[0]}x{original_size[1]}")
    print(f"Crop: {rows}x{cols}; views: {views}={rows * cols}+1; tokens/view: {tokens_per_view} (8x8 verified)")
    print(f"N={embeddings.shape[0]}; local/global population={rows * cols * tokens_per_view}/{tokens_per_view}", flush=True)
    summary = []
    for k in BUDGETS:
        summary.extend(evaluate_budget(path.name, embeddings, rows, cols, token_grid, k))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=PROJECT_ROOT / "images")
    args = parser.parse_args()
    paths = [args.image_dir / filename for filename in FILENAMES]
    for path in paths:
        if not path.is_file():
            parser.error(f"Image not found: {path}")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, local_files_only=True, dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    print(f"Device: {device}; model/processor loaded once; budgets={BUDGETS}; grids={GRID_SIDES}")
    print("Matched generators (seed: local/global): 0:0/1, 1:2/3, 2:4/5, 3:6/7, 4:8/9")
    print("Positive global excess pp = more global-view tokens than the population proportion.", flush=True)
    summary = []
    for path in paths:
        summary.extend(diagnose_image(processor, model, path, device))
    print("\nSummary: Random zero/CV means average the individual seed statistics.")
    print_table(("image", "K", "grid", "GC L/G", "GC zero", "Standard Random zero mean",
                 "Matched-L/G Random zero mean", "GC CV", "Standard Random CV mean",
                 "Matched-L/G Random CV mean", "GC L/K", "Population N_local/N", "GC global excess pp"), summary)


if __name__ == "__main__":
    main()
