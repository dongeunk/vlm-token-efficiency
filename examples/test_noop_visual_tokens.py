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

"""Compare ordinary generation with direct insertion of all 1088 visual tokens.

Usage:
    python examples/test_noop_visual_tokens.py

Uses cached SmolVLM files, images/meta.webp, and "What is in this image?".
Both paths use greedy decoding with the same settings. No files are downloaded.
"""

import argparse
import os
from pathlib import Path
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
IMAGE_PATH = Path(__file__).resolve().parents[1] / "images" / "meta.webp"
QUESTION = "What is in this image?"


def max_abs_difference(left: torch.Tensor, right: torch.Tensor) -> float:
    return (left.float() - right.float()).abs().max().item()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if not IMAGE_PATH.is_file():
        parser.error(f"Image not found: {IMAGE_PATH}")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, local_files_only=True, dtype=dtype, attn_implementation="eager"
    ).to(device).eval()

    with Image.open(IMAGE_PATH) as source:
        image = source.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": QUESTION},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    image_mask = input_ids == model.config.image_token_id
    positions = image_mask[0].nonzero(as_tuple=True)[0]

    generation_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=True,
        return_dict_in_generate=True,
        output_logits=True,
    )

    with torch.inference_mode():
        projected = model.get_image_features(
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            return_dict=True,
        ).pooler_output
        visual_embeddings = projected.flatten(0, 1)
        n = visual_embeddings.shape[0]
        k = n  # No selection, reordering, pooling, or removal of visual tokens.
        if k != 1088 or positions.numel() != n:
            raise RuntimeError(
                f"Expected K=N=1088 and 1088 placeholders; got K={k}, N={n}, "
                f"placeholders={positions.numel()}"
            )

        text_embeddings = model.get_input_embeddings()(input_ids)
        manual_embeddings = text_embeddings.clone()
        # Boolean indexing visits the original placeholder positions in order.
        manual_embeddings[image_mask] = visual_embeddings

        # Capture the actual first LLM input from the ordinary generate path.
        # The manual path above does not call the model's image merger.
        original_embeddings = None

        def capture_first_llm_input(module, positional_args, kwargs):
            nonlocal original_embeddings
            if original_embeddings is None:
                original_embeddings = kwargs["inputs_embeds"].detach().clone()

        hook = model.model.text_model.register_forward_pre_hook(
            capture_first_llm_input, with_kwargs=True
        )
        print(f"Device: {device}; K=N={n}", flush=True)
        print(f"Projected visual embeddings: {tuple(projected.shape)}", flush=True)
        print("Running original model.generate()...", flush=True)
        try:
            original = model.generate(**inputs, **generation_kwargs)
        finally:
            hook.remove()

        if original_embeddings is None:
            raise RuntimeError("The original LLM input was not captured")
        embedding_difference = max_abs_difference(original_embeddings, manual_embeddings)
        embeddings_equal = torch.equal(original_embeddings, manual_embeddings)
        text_preserved = torch.equal(
            manual_embeddings[~image_mask], text_embeddings[~image_mask]
        )
        print(f"LLM input embedding shape: {tuple(manual_embeddings.shape)}", flush=True)
        print(f"Input embeddings max absolute difference: {embedding_difference}", flush=True)
        print(f"Input embeddings exactly equal: {embeddings_equal}", flush=True)
        print(f"Text/separator/special embeddings preserved: {text_preserved}", flush=True)

        print("Running no-op generation from directly inserted embeddings...", flush=True)
        # Keep input_ids for prompt bookkeeping. With no pixel_values or
        # image_hidden_states, the outer model forwards these embeddings directly
        # to the same LLM and LM head. Subsequent steps use the same KV-cache path.
        noop = model.generate(
            input_ids=input_ids,
            inputs_embeds=manual_embeddings,
            attention_mask=inputs["attention_mask"],
            **generation_kwargs,
        )

        first_logits_difference = max_abs_difference(original.logits[0], noop.logits[0])
        first_logits_equal = torch.equal(original.logits[0], noop.logits[0])
        prompt_length = input_ids.shape[1]
        original_ids = original.sequences[:, prompt_length:]
        noop_ids = noop.sequences[:, prompt_length:]
        generated_ids_equal = torch.equal(original_ids, noop_ids)

    print(f"First generation step raw logits max absolute difference: {first_logits_difference}")
    print(f"First generation step raw logits exactly equal: {first_logits_equal}")
    print(f"Generated token IDs exactly equal: {generated_ids_equal}")
    print("\nOriginal answer:")
    print(processor.decode(original_ids[0], skip_special_tokens=True).strip())
    print("\nNo-op answer:")
    print(processor.decode(noop_ids[0], skip_special_tokens=True).strip())
    if not embeddings_equal or not text_preserved or not first_logits_equal or not generated_ids_equal:
        raise RuntimeError("The no-op comparison differs; see the measurements above")
    print("\nPASS: all 1088 visual embeddings, first-step logits, and generated tokens match.")


if __name__ == "__main__":
    main()
