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

"""Inspect the processor sequence and visual embedding positions using cached files.

Usage:
    python examples/inspect_llm_input.py
    python examples/inspect_llm_input.py path/to/image.jpg --question "What is in this image?"

All sequence positions and view/token indices are zero-based. Ranges are [start, end).
No generation is performed, and model/processor files must already be cached.
"""

import argparse
import os
from pathlib import Path
import re

# Keep all Hugging Face access offline, including files referenced by the processor.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
DEFAULT_IMAGE = Path(__file__).resolve().parents[1] / "images" / "meta.webp"
ROW_COL_TOKEN = re.compile(r"<row_(\d+)_col_(\d+)>")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--question", default="What is in this image?")
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

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
    positions = image_mask[0].nonzero(as_tuple=True)[0]

    with torch.inference_mode():
        projected = model.get_image_features(
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            return_dict=True,
        ).pooler_output
        text_embeddings = model.get_input_embeddings()(input_ids)
        # Use the same merger as SmolVLM.forward, immediately before text_model.
        merged = model.model.inputs_merger(input_ids, text_embeddings, projected)

    views, tokens_per_view, dimension = projected.shape
    if positions.numel() != views * tokens_per_view:
        raise RuntimeError("Image placeholder count does not match projected visual tokens")
    ids = input_ids[0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(ids)
    special_ids = set(processor.tokenizer.all_special_ids)
    global_marker = getattr(processor, "global_image_tag", "<global-img>")
    view_positions = positions.reshape(views, tokens_per_view).tolist()
    labels = []
    for block in view_positions:
        if block != list(range(block[0], block[0] + tokens_per_view)):
            raise RuntimeError("Visual placeholders for a view are not contiguous")
        label = "unknown view marker"
        for token in reversed(tokens[:block[0]]):
            match = ROW_COL_TOKEN.fullmatch(token)
            if match:
                label = f"local crop row={match[1]}, column={match[2]} (1-based)"
                break
            if token == global_marker:
                label = "global image (resized full image)"
                break
        labels.append(label)

    print(f"Image: {args.image}")
    print(f"Question: {args.question}")
    print(f"Processor: {type(processor).__name__}; model: {type(model).__name__}")
    print(f"Device: {device}; positions are zero-based; ranges are [start, end)")
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"Total input sequence length: {input_ids.shape[1]}")
    print(f"pixel_values shape: {tuple(inputs['pixel_values'].shape)}")
    print(f"Image placeholder: {processor.image_token!r}, id={model.config.image_token_id}")
    print(f"Image placeholder count: {positions.numel()} = {views} views x {tokens_per_view}")
    print(f"Non-image token count: {len(ids) - positions.numel()}")
    print(f"Projected visual tokens shape: {tuple(projected.shape)}")
    print(f"LLM inputs_embeds shape after merger: {tuple(merged.shape)}")
    print("\nView -> LLM sequence mapping:")
    for view, block in enumerate(view_positions):
        print(
            f"  view {view:2}: {labels[view]}; positions [{block[0]}, {block[-1] + 1}); "
            f"projected[{view}, t, :] -> inputs_embeds[0, {block[0]} + t, :] "
            f"for t=0..{tokens_per_view - 1}, D={dimension}"
        )

    # Show every sequence position, compressing only repeated image placeholders.
    print("\nProcessor token sequence (special tokens and view markers are preserved):")
    start_to_view = {block[0]: view for view, block in enumerate(view_positions)}
    position = 0
    while position < len(ids):
        if position in start_to_view:
            view = start_to_view[position]
            end = position + tokens_per_view
            print(f"  [{position}, {end}) IMAGE_PLACEHOLDERS id={ids[position]} view={view}")
            position = end
            continue
        token = tokens[position]
        if ROW_COL_TOKEN.fullmatch(token) or token == global_marker:
            kind = "VIEW_MARKER"
        elif token == processor.fake_image_token:
            kind = "VIEW_SEPARATOR"
        elif ids[position] in special_ids:
            kind = "SPECIAL"
        else:
            kind = "TEXT/WHITESPACE"
        decoded = processor.tokenizer.decode([ids[position]], skip_special_tokens=False)
        print(
            f"  [{position}, {position + 1}) {kind:16} id={ids[position]:6} "
            f"token={ascii(token)} decoded={ascii(decoded)}"
        )
        position += 1

    visual_match = torch.equal(merged[0, positions, :], projected.flatten(0, 1))
    text_match = torch.equal(merged[~image_mask], text_embeddings[~image_mask])
    print(f"\nVerified visual embedding placement: {visual_match}")
    print(f"Verified separator/special/text embeddings preserved: {text_match}")
    print("The merger replaces image placeholders in place; sequence length is unchanged.")
    if not visual_match or not text_match:
        raise RuntimeError("LLM input embedding verification failed")


if __name__ == "__main__":
    main()
