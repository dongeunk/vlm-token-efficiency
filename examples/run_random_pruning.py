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

"""Randomly retain K visual tokens and remove the other sequence positions.

Usage:
    python examples/run_random_pruning.py --k 256 --seed 42
    python examples/run_random_pruning.py --k 1088

Model files must be cached. K=N automatically compares against ordinary
model.generate(), including actual LLM input embeddings and first-step logits.
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


MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_IMAGE = Path(__file__).resolve().parents[1] / "images" / "meta.webp"


def measured_generate(model, inputs, settings, device):
    """Measure only generate(), synchronizing CUDA and resetting peak counters."""
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        starting_memory = torch.cuda.memory_allocated()
    start = time.perf_counter()
    result = model.generate(**inputs, **settings)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    memory = None
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated()
        memory = (peak, peak - starting_memory)
    return result, elapsed, memory


def print_metrics(label, elapsed, memory):
    print(f"{label} generation time: {elapsed:.3f} seconds")
    if memory is not None:
        peak, extra = memory
        print(f"{label} CUDA peak allocated memory: {peak / 2**20:.2f} MiB")
        print(f"{label} CUDA peak allocation above generate() start: {extra / 2**20:.2f} MiB")
    else:
        print(f"{label} CUDA peak memory: N/A (CPU)")


def max_abs_difference(left, right):
    return (left.float() - right.float()).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--question", default="What is in this image?")
    parser.add_argument("--k", type=int, default=256, help="Number of retained visual tokens (0 <= K <= N)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    if args.k < 0 or args.max_new_tokens < 1:
        parser.error("--k must be nonnegative and --max-new-tokens must be positive")
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, local_files_only=True, dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": args.question},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    image_mask = input_ids == model.config.image_token_id
    visual_positions = image_mask[0].nonzero(as_tuple=True)[0]

    with torch.inference_mode():
        projected = model.get_image_features(
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            return_dict=True,
        ).pooler_output
        visual_embeddings = projected.flatten(0, 1)
        n = visual_embeddings.shape[0]
        if visual_positions.numel() != n:
            raise RuntimeError("Image placeholder count does not match visual embedding count")
        if args.k > n:
            parser.error(f"--k must be <= N={n}; got {args.k}")

        # Sample uniformly without replacement on CPU so the same seed selects
        # the same indices on CPU and CUDA. Sort to preserve original order.
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        selected = torch.randperm(n, generator=generator)[:args.k].sort().values.to(device)
        text_embeddings = model.get_input_embeddings()(input_ids)
        full_embeddings = text_embeddings.clone()
        full_embeddings[image_mask] = visual_embeddings

        # Every non-image position stays. Only unselected image positions vanish.
        keep = (~image_mask[0]).clone()
        keep[visual_positions[selected]] = True
        pruned_ids = input_ids[:, keep]
        pruned_embeddings = full_embeddings[:, keep, :]
        pruned_attention = inputs["attention_mask"][:, keep]
        pruned_image_mask = pruned_ids == model.config.image_token_id
        original_length = input_ids.shape[1]
        pruned_length = pruned_ids.shape[1]
        if pruned_length != original_length - (n - args.k):
            raise RuntimeError("Pruned sequence length does not match N-K removal")
        if pruned_image_mask.sum().item() != args.k:
            raise RuntimeError("Pruned sequence does not contain exactly K image positions")
        text_preserved = (
            torch.equal(pruned_ids[~pruned_image_mask], input_ids[~image_mask])
            and torch.equal(pruned_embeddings[~pruned_image_mask], text_embeddings[~image_mask])
            and torch.equal(pruned_attention[~pruned_image_mask], inputs["attention_mask"][~image_mask])
        )
        visuals_preserved = torch.equal(
            pruned_embeddings[pruned_image_mask], visual_embeddings[selected]
        )
        if not text_preserved or not visuals_preserved:
            raise RuntimeError("Token order or preservation check failed")

        print(f"Device: {device}; seed: {args.seed}")
        print(f"Original N: {n}; retained K: {args.k}")
        print(f"Original sequence length: {original_length}")
        print(f"Pruned sequence length: {pruned_length}")
        print(f"Removed visual tokens: {n - args.k}")
        print(f"Selected flattened visual token indices (zero-based, first 24): {selected[:24].tolist()}")
        print(f"Selected original sequence positions (first 24): {visual_positions[selected[:24]].tolist()}")
        print(f"All text/separator/special tokens preserved: {text_preserved}")
        print(f"Selected visual embeddings preserved in sequence order: {visuals_preserved}", flush=True)

        settings = dict(
            max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
            use_cache=True, return_dict_in_generate=True, output_logits=args.k == n,
        )
        original = None
        original_embeddings = None
        if args.k == n:
            def capture_first_llm_input(module, positional_args, kwargs):
                nonlocal original_embeddings
                if original_embeddings is None:
                    original_embeddings = kwargs["inputs_embeds"].detach().clone()

            hook = model.model.text_model.register_forward_pre_hook(
                capture_first_llm_input, with_kwargs=True
            )
            print("K=N: running original model.generate() for comparison...", flush=True)
            try:
                original, elapsed, memory = measured_generate(model, inputs, settings, device)
            finally:
                hook.remove()
            print_metrics("Original", elapsed, memory)
            if original_embeddings is None:
                raise RuntimeError("Original LLM embeddings were not captured")

        # Pixels are not forwarded: get_image_features() already produced the
        # embeddings. generate() derives contiguous position IDs from the shorter
        # attention mask, then continues through the original LLM, head, and cache.
        direct_inputs = dict(
            input_ids=pruned_ids, inputs_embeds=pruned_embeddings,
            attention_mask=pruned_attention,
        )
        print("Running direct embedding generation...", flush=True)
        result, elapsed, memory = measured_generate(model, direct_inputs, settings, device)
        print_metrics("Direct (precomputed visual embeddings)", elapsed, memory)
        print("Timing covers generate() only; visual feature extraction is excluded from the direct path.")
        generated_ids = result.sequences[:, pruned_length:]
        answer = processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        print(f"\nFinal answer:\n{answer}")

        if original is not None:
            original_generated_ids = original.sequences[:, original_length:]
            original_answer = processor.decode(original_generated_ids[0], skip_special_tokens=True).strip()
            embeddings_equal = torch.equal(original_embeddings, pruned_embeddings)
            logits_equal = torch.equal(original.logits[0], result.logits[0])
            ids_equal = torch.equal(original_generated_ids, generated_ids)
            answer_equal = original_answer == answer
            print(f"\nOriginal answer:\n{original_answer}")
            print(f"K=N input embeddings max absolute difference: {max_abs_difference(original_embeddings, pruned_embeddings)}")
            print(f"K=N input embeddings exactly equal: {embeddings_equal}")
            print(f"K=N first-step logits max absolute difference: {max_abs_difference(original.logits[0], result.logits[0])}")
            print(f"K=N first-step logits exactly equal: {logits_equal}")
            print(f"K=N generated token IDs exactly equal: {ids_equal}")
            print(f"K=N final answers equal: {answer_equal}")
            if not all((embeddings_equal, logits_equal, ids_equal, answer_equal)):
                raise RuntimeError("K=N no-op comparison failed")
            print("PASS: K=N matches the original generation path.")


if __name__ == "__main__":
    main()
