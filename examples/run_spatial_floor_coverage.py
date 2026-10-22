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

"""Compare Pure GC with a spatial medoid floor, then generate from the latter.

Examples:
    python examples/run_spatial_floor_coverage.py --image images/table.jpg --k 64 --region-grid 4
    python examples/run_spatial_floor_coverage.py --image images/table.jpg --k 128 --region-grid 8

--region-grid accepts one side or two dimensions, e.g. --region-grid 4 8.
All model files must already be cached. Generation uses greedy decoding.
"""

import argparse
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from examples.diagnose_global_coverage import coverage_score, projected_grid_shape
from examples.run_global_coverage import measured_generate, print_metrics
from src.pruning.coverage import global_coverage_select
from src.pruning.spatial_floor_coverage import calibrated_affinity, spatial_floor_coverage_select

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def timed_selection(function, device):
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    result = function()
    if device == "cuda":
        torch.cuda.synchronize()
    return result, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=PROJECT_ROOT / "images" / "table.jpg")
    parser.add_argument("--question", default="What is in this image?")
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--region-grid", type=int, nargs="+", default=[4], metavar="SIDE")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if args.k < 1 or args.max_new_tokens < 1:
        parser.error("--k and --max-new-tokens must be positive")
    if len(args.region_grid) not in (1, 2) or min(args.region_grid) < 1:
        parser.error("--region-grid requires one positive side or two positive dimensions")
    region_grid = tuple(args.region_grid * 2) if len(args.region_grid) == 1 else tuple(args.region_grid)
    g = region_grid[0] * region_grid[1]
    if g > args.k:
        parser.error(f"Spatial floor infeasible: G={g} exceeds K={args.k}")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, local_files_only=True, dtype=dtype, attn_implementation="eager",
    ).to(device).eval()
    with Image.open(args.image) as source:
        original_size = source.size
        image = source.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": args.question},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    # As in the diagnostics, metadata comes directly from the actual image
    # processor; combined processing consumes rows/cols while building the prompt.
    image_details = processor.image_processor(
        images=[image], return_row_col_info=True, return_tensors="pt",
    )
    rows, cols = int(image_details["rows"][0][0]), int(image_details["cols"][0][0])
    if image_details["pixel_values"].shape != inputs["pixel_values"].shape:
        raise RuntimeError("Metadata preprocessing disagrees with model pixel input shape")
    input_ids = inputs["input_ids"]
    image_mask = input_ids == model.config.image_token_id
    visual_positions = image_mask[0].nonzero(as_tuple=True)[0]

    with torch.inference_mode():
        projected = model.get_image_features(
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs.get("pixel_attention_mask"), return_dict=True,
        ).pooler_output
        views, tokens_per_view, _ = projected.shape
        token_grid = projected_grid_shape(model.config, inputs["pixel_values"].shape, tokens_per_view)
        if views != rows * cols + 1 or inputs["pixel_values"].shape[1] != views:
            raise RuntimeError("Crop metadata disagrees with actual local/global view count")
        visual_embeddings = projected.flatten(0, 1)
        n = visual_embeddings.shape[0]
        n_local = rows * cols * tokens_per_view
        if visual_positions.numel() != n:
            raise RuntimeError("Image placeholder count does not match projected tokens")
        if args.k > n:
            parser.error(f"--k must be <= N={n}")

        pure, pure_elapsed = timed_selection(
            lambda: global_coverage_select(visual_embeddings, args.k), device,
        )
        floor_result, floor_elapsed = timed_selection(
            lambda: spatial_floor_coverage_select(
                visual_embeddings, args.k, rows, cols, token_grid, region_grid,
            ), device,
        )
        selected = floor_result.selected_indices
        # Both selected sets are scored using this exact same affinity tensor.
        affinity = calibrated_affinity(visual_embeddings)
        pure_score = coverage_score(affinity, pure)
        floor_score = coverage_score(affinity, selected)
        loss = pure_score - floor_score
        loss_pct = loss / pure_score * 100 if pure_score else float("nan")
        indices_unique = selected.unique().numel() == args.k
        original_order = torch.equal(selected, selected.sort().values)
        indices_valid = ((selected >= 0) & (selected < n)).all().item()
        floor_satisfied = (floor_result.region_counts >= 1).all().item()
        if selected.numel() != args.k or not all((indices_unique, original_order, indices_valid, floor_satisfied)):
            raise RuntimeError("Spatial floor selector violated selection invariants")

        # Reuse the validated direct insertion and physical deletion path.
        text_embeddings = model.get_input_embeddings()(input_ids)
        full_embeddings = text_embeddings.clone()
        full_embeddings[image_mask] = visual_embeddings
        keep = (~image_mask[0]).clone()
        keep[visual_positions[selected]] = True
        pruned_ids = input_ids[:, keep]
        pruned_embeddings = full_embeddings[:, keep, :]
        pruned_attention = inputs["attention_mask"][:, keep]
        pruned_image_mask = pruned_ids == model.config.image_token_id
        original_length = input_ids.shape[1]
        pruned_length = pruned_ids.shape[1]
        if pruned_length != original_length - (n - args.k) or pruned_image_mask.sum().item() != args.k:
            raise RuntimeError("Physical sequence shortening does not match N-K deletion")
        text_preserved = (
            torch.equal(pruned_ids[~pruned_image_mask], input_ids[~image_mask])
            and torch.equal(pruned_embeddings[~pruned_image_mask], text_embeddings[~image_mask])
            and torch.equal(pruned_attention[~pruned_image_mask], inputs["attention_mask"][~image_mask])
        )
        visuals_preserved = torch.equal(pruned_embeddings[pruned_image_mask], visual_embeddings[selected])
        if not text_preserved or not visuals_preserved:
            raise RuntimeError("Text/special/visual embedding preservation check failed")

        print(f"Filename: {args.image.name}; question: {args.question}; device: {device}")
        print(f"Original image size (W x H): {original_size[0]} x {original_size[1]}")
        print(f"Crop rows x cols: {rows} x {cols}; projected tokens/view: {tokens_per_view}; grid: {token_grid}")
        print(f"Total visual N: {n}; local/global population counts: {n_local}/{n - n_local}")
        print(f"K: {args.k}; region grid: {region_grid[0]}x{region_grid[1]}; G: {g}; floor ratio G/K: {g / args.k:.4f}")
        print(f"Floor token count: {floor_result.floor_indices.numel()}; completion count: {floor_result.completion_indices.numel()}")
        print(f"Pure GC selected local/global: {(pure < n_local).sum().item()}/{(pure >= n_local).sum().item()}")
        print(f"Spatial Floor selected local/global: {floor_result.selected_local}/{floor_result.selected_global}")
        print(f"Floor indices (region order): {floor_result.floor_indices.tolist()}")
        print(f"Completion indices (greedy order, first 24): {floor_result.completion_indices[:24].tolist()}")
        print(f"Final selected visual indices (sorted, first 24): {selected[:24].tolist()}")
        print("Coarse-region final local-token count matrix:")
        for row in floor_result.region_counts.tolist():
            print("  [" + " ".join(f"{count:3d}" for count in row) + "]")
        print(f"All regions satisfy minimum-one floor: {floor_satisfied}")
        print(f"pure_coverage_score: {pure_score:.6f}")
        print(f"spatial_floor_coverage_score: {floor_score:.6f}")
        print(f"coverage_loss: {loss:+.6f}")
        print(f"coverage_loss_pct: {loss_pct:+.6f}%")
        print(f"Pure GC selector time: {pure_elapsed:.3f} seconds")
        print(f"Spatial Floor + GC selector time: {floor_elapsed:.3f} seconds")
        if loss < 0:
            print("Observation: the floor-seeded greedy run exceeded the pure greedy score; these are heuristic solutions.")
        print(f"Original sequence length: {original_length}; pruned sequence length: {pruned_length}")
        print(f"Text/separator/special tokens preserved: {text_preserved}")
        print(f"Selected visual embeddings preserved: {visuals_preserved}; original order preserved: {original_order}", flush=True)

        direct_inputs = dict(input_ids=pruned_ids, inputs_embeds=pruned_embeddings, attention_mask=pruned_attention)
        settings = dict(max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                        use_cache=True, return_dict_in_generate=True)
        output, elapsed, memory = measured_generate(model, direct_inputs, settings, device)
        print_metrics("Spatial Floor direct", elapsed, memory)
        print("Generation timing excludes visual feature extraction and selector execution.")
        generated_ids = output.sequences[:, pruned_length:]
        eos = model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (tuple, list)) else [eos])
        ended_with_eos = generated_ids[0, -1].item() in eos_ids
        print(f"Generated tokens: {generated_ids.shape[1]}; ended with EOS: {ended_with_eos}; max_new_tokens: {args.max_new_tokens}")
        print("\nFinal generated answer:")
        print(processor.decode(generated_ids[0], skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
