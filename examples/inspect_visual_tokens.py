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

"""Inspect SmolVLM visual token shapes for one local image.

Usage:
    python examples/inspect_visual_tokens.py path/to/image.jpg

Requires PyTorch, Transformers, and Pillow. The first run may download
HuggingFaceTB/SmolVLM-256M-Instruct model files from Hugging Face.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to one local image")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    with Image.open(args.image) as source:
        image = source.convert("RGB")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    # SmolVLM consumes rows/cols while building the prompt, so request that
    # metadata from its image processor separately for display.
    image_details = processor.image_processor(
        images=[image], return_row_col_info=True, return_tensors="pt"
    )
    rows = image_details["rows"][0][0]
    cols = image_details["cols"][0][0]
    view_count = inputs["pixel_values"].shape[1]

    with torch.inference_mode():
        image_outputs = model.get_image_features(
            pixel_values=inputs["pixel_values"],
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            return_dict=True,
        )

    encoder_tokens = image_outputs.last_hidden_state
    projected_tokens = image_outputs.pooler_output
    # The processor can split one image into several patches. Combine their
    # token axes to report the full sequence of visual embeddings for this image.
    visual_tokens = projected_tokens.flatten(0, 1)

    print(f"Device: {device}")
    print(f"Input image size: {image.width}x{image.height}")
    print(f"pixel_values shape: {tuple(inputs['pixel_values'].shape)}")
    print(f"Image splitting enabled: {processor.image_processor.do_image_splitting}")
    print(f"Resize longest edge: {processor.image_processor.size['longest_edge']}")
    print(f"View edge: {processor.image_processor.max_image_size['longest_edge']}")
    print(f"Local crop layout: {rows} rows x {cols} columns")
    print(f"Views: {view_count} = {rows * cols} local crops + 1 global image")
    for index in range(rows * cols):
        row, col = divmod(index, cols)
        print(f"  view {index}: local crop (row {row + 1}, column {col + 1})")
    print(f"  view {view_count - 1}: global image (resized full image)")
    print(f"Vision encoder output shape: {tuple(encoder_tokens.shape)}")
    print(f"Projected visual token shape: {tuple(projected_tokens.shape)}")
    print(f"Visual tokens for LLM shape: {tuple(visual_tokens.shape)}")
    print(f"N (visual tokens): {visual_tokens.shape[0]}")
    print(f"D (embedding dimension): {visual_tokens.shape[1]}")

    vision_config = model.config.vision_config
    patch_grid_side = vision_config.image_size // vision_config.patch_size
    grid_side = patch_grid_side // model.config.scale_factor
    tokens_per_view = projected_tokens.shape[1]
    expected_tokens = grid_side**2
    print(
        "Projected grid from config: "
        f"({vision_config.image_size} / {vision_config.patch_size} / "
        f"{model.config.scale_factor})^2 = {grid_side}x{grid_side} = {expected_tokens} tokens/view"
    )
    print(
        f"Observed tokens/view: {tokens_per_view}; "
        f"8x8 confirmation: {grid_side == 8 and tokens_per_view == expected_tokens == 64}"
    )


if __name__ == "__main__":
    main()
