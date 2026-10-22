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

"""Diagnose representational coverage and local spatial allocation at K=256.

Uses five images in images/, cached SmolVLM files, and no language generation.
All reported region coordinates are zero-based. Affinity demands/candidates
include both local and global views; spatial region counts include local only.
"""

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import statistics
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.pruning.coverage import global_coverage_select

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
FILENAMES = ("ditto.webp", "street.avif", "table.jpg", "menu.avif", "people.avif")
K = 256
RANDOM_SEEDS = (0, 1, 2, 3, 4)


@torch.no_grad()
def coverage_affinity(visual_embeddings: torch.Tensor) -> torch.Tensor:
    """Use the same float32 operations and normalization epsilon as coverage.py."""
    embeddings = visual_embeddings.float()
    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        v = F.normalize(embeddings, p=2, dim=1, eps=1e-12)
        similarity = v @ v.T
        return (similarity - similarity.mean(dim=1, keepdim=True)).clamp_min_(0)


def coverage_score(affinity: torch.Tensor, indices: torch.Tensor) -> float:
    """F(S) with every local/global token as a unit-weight demand."""
    if indices.numel() == 0:
        return 0.0
    return affinity.index_select(1, indices).max(dim=1).values.sum().item()


def near_equal_edges(size: int, parts: int = 4) -> list[int]:
    """Non-overlapping partition boundaries using round(i * size / parts)."""
    return [round(i * size / parts) for i in range(parts + 1)]


def local_selection_lattice(indices, rows: int, cols: int, grid_shape: tuple[int, int]):
    """Map view-major token indices into the full local image lattice.

    Crops and projected tokens inside each crop are row-major. The last
    global view is excluded. A flat reshape would misplace adjacent crops.
    """
    grid_h, grid_w = grid_shape
    tokens_per_view = grid_h * grid_w
    local_count = rows * cols * tokens_per_view
    lattice = torch.zeros((rows * grid_h, cols * grid_w), dtype=torch.long)
    for index in indices.tolist():
        if not 0 <= index < local_count + tokens_per_view:
            raise ValueError(f"Visual index out of range: {index}")
        if index >= local_count:
            continue
        view, token = divmod(index, tokens_per_view)
        view_row, view_col = divmod(view, cols)
        token_row, token_col = divmod(token, grid_w)
        lattice[view_row * grid_h + token_row, view_col * grid_w + token_col] += 1
    return lattice


def coarse_region_counts(lattice: torch.Tensor) -> torch.Tensor:
    """Count selected local tokens in a near-equal 4x4 partition."""
    row_edges = near_equal_edges(lattice.shape[0])
    col_edges = near_equal_edges(lattice.shape[1])
    counts = torch.zeros((4, 4), dtype=torch.long)
    for row in range(4):
        for col in range(4):
            counts[row, col] = lattice[
                row_edges[row]:row_edges[row + 1], col_edges[col]:col_edges[col + 1]
            ].sum()
    return counts


def _pair(value):
    return (value, value) if isinstance(value, int) else tuple(value)


def projected_grid_shape(config, pixel_shape, tokens_per_view):
    """Check both configured and actual view dimensions against observed tokens."""
    patch_h, patch_w = _pair(config.vision_config.patch_size)
    config_h, config_w = _pair(config.vision_config.image_size)
    view_h, view_w = pixel_shape[-2:]
    scale = config.scale_factor
    if any((view_h % patch_h, view_w % patch_w,
            (view_h // patch_h) % scale, (view_w // patch_w) % scale)):
        raise RuntimeError("View dimensions are not divisible by patch size and connector scale")
    actual = (view_h // patch_h // scale, view_w // patch_w // scale)
    configured = (config_h // patch_h // scale, config_w // patch_w // scale)
    if actual != configured or actual != (8, 8) or tokens_per_view != math.prod(actual):
        raise RuntimeError(
            f"Expected observed/configured 8x8=64 tokens/view; got "
            f"actual={actual}, configured={configured}, tokens={tokens_per_view}"
        )
    return actual


@dataclass
class DiagnosticResult:
    filename: str
    original_size: tuple[int, int]
    rows: int
    cols: int
    views: int
    n: int
    local_tokens: int
    global_tokens: int
    global_score: float
    random_scores: list[float]
    random_mean: float
    random_std: float
    improvement: float
    improvement_pct: float
    selected_local: int
    selected_global: int
    coarse_counts: torch.Tensor
    zero_regions: list[tuple[int, int]]


@torch.inference_mode()
def diagnose_image(processor, model, path: Path, device: str) -> DiagnosticResult:
    with Image.open(path) as source:
        original_size = source.size
        image = source.convert("RGB")
    # The combined processor uses then removes rows/cols. Read the actual
    # metadata directly from its image processor with the same default settings.
    image_inputs = processor.image_processor(
        images=[image], return_row_col_info=True, return_tensors="pt"
    ).to(device)
    rows = int(image_inputs["rows"][0][0])
    cols = int(image_inputs["cols"][0][0])
    projected = model.get_image_features(
        pixel_values=image_inputs["pixel_values"],
        pixel_attention_mask=image_inputs.get("pixel_attention_mask"),
        return_dict=True,
    ).pooler_output
    views, tokens_per_view, _ = projected.shape
    grid_shape = projected_grid_shape(model.config, image_inputs["pixel_values"].shape, tokens_per_view)
    local_views = rows * cols
    if views != local_views + 1 or image_inputs["pixel_values"].shape[1] != views:
        raise RuntimeError(f"Processor crop metadata disagrees with view count for {path.name}")
    embeddings = projected.flatten(0, 1)
    n = embeddings.shape[0]
    if K > n:
        raise RuntimeError(f"{path.name}: K={K} exceeds N={n}")
    local_tokens = local_views * tokens_per_view
    selected = global_coverage_select(embeddings, K)
    if selected.numel() != K or selected.unique().numel() != K:
        raise RuntimeError("Global selector did not return exactly K unique indices")
    affinity = coverage_affinity(embeddings)
    global_score = coverage_score(affinity, selected)
    random_scores = []
    for seed in RANDOM_SEEDS:
        # Match run_random_pruning.py: CPU randperm, no replacement, final sort.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random_indices = torch.randperm(n, generator=generator)[:K].sort().values.to(device)
        random_scores.append(coverage_score(affinity, random_indices))
    random_mean = statistics.mean(random_scores)
    random_std = statistics.stdev(random_scores)  # sample SD, denominator 5-1
    improvement = global_score - random_mean
    improvement_pct = improvement / random_mean * 100 if random_mean else float("nan")
    selected_local = (selected < local_tokens).sum().item()
    selected_global = K - selected_local
    lattice = local_selection_lattice(selected, rows, cols, grid_shape)
    coarse_counts = coarse_region_counts(lattice)
    if coarse_counts.sum().item() != selected_local or lattice.sum().item() != selected_local:
        raise RuntimeError("Spatial reconstruction lost selected local tokens")
    zero_regions = [tuple(position) for position in (coarse_counts == 0).nonzero().tolist()]
    return DiagnosticResult(
        path.name, original_size, rows, cols, views, n, local_tokens, tokens_per_view,
        global_score, random_scores, random_mean, random_std, improvement,
        improvement_pct, selected_local, selected_global, coarse_counts, zero_regions,
    )


def print_result(result: DiagnosticResult):
    print(f"\nFilename: {result.filename}")
    print(f"Original image size (W x H): {result.original_size[0]} x {result.original_size[1]}")
    print(f"Crop rows x cols: {result.rows} x {result.cols}")
    print(f"Total views: {result.views} = {result.rows * result.cols} local + 1 global")
    print(f"Projected tokens/view: {result.global_tokens}; tensor/config 8x8=64 check: PASS")
    print(f"Total visual N: {result.n}; local tokens: {result.local_tokens}; global tokens: {result.global_tokens}")
    print(f"K: {K}")
    print(f"Global Coverage F(S): {result.global_score:.6f}")
    for seed, score in zip(RANDOM_SEEDS, result.random_scores):
        print(f"Random seed {seed} F(S): {score:.6f}")
    print(f"Random mean +/- sample std: {result.random_mean:.6f} +/- {result.random_std:.6f}")
    print(f"Coverage improvement: {result.improvement:+.6f} ({result.improvement_pct:+.4f}%)")
    print(f"Global selected local/global counts: {result.selected_local} / {result.selected_global}")
    print(f"Local spatial lattice: {result.rows * 8} x {result.cols * 8}")
    print("4x4 coarse-region selected LOCAL-token counts:")
    for row in result.coarse_counts.tolist():
        print("  [" + " ".join(f"{count:3d}" for count in row) + "]")
    print(f"Zero-token regions: {len(result.zero_regions)}; (row, col), zero-based: {result.zero_regions}", flush=True)


def print_summary(results):
    headers = ("Filename", "Size(WxH)", "Crop", "Views(L+G)", "N", "K", "GC F(S)",
               "Random mean +/- SD", "Delta F", "Delta %", "Selected L/G", "Zero regions")
    cells = [headers]
    for r in results:
        cells.append((r.filename, f"{r.original_size[0]}x{r.original_size[1]}", f"{r.rows}x{r.cols}",
                      f"{r.rows * r.cols}+1", str(r.n), str(K), f"{r.global_score:.6f}",
                      f"{r.random_mean:.6f} +/- {r.random_std:.6f}", f"{r.improvement:+.6f}",
                      f"{r.improvement_pct:+.4f}", f"{r.selected_local}/{r.selected_global}", str(len(r.zero_regions))))
    widths = [max(len(row[col]) for row in cells) for col in range(len(headers))]
    print("\nSummary (sample SD; coverage includes local + global, spatial counts local only):")
    for index, row in enumerate(cells):
        print(" | ".join(value.ljust(width) for value, width in zip(row, widths)))
        if index == 0:
            print("-+-".join("-" * width for width in widths))


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
        MODEL_ID, local_files_only=True, dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    print(f"Device: {device}; model and processor loaded once; K={K}", flush=True)
    results = []
    for path in paths:
        print(f"\nInspecting {path.name}...", flush=True)
        result = diagnose_image(processor, model, path, device)
        print_result(result)
        results.append(result)
    print_summary(results)


if __name__ == "__main__":
    main()
