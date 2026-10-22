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

"""Ask one question about a local image with SmolVLM on a CUDA GPU.

Usage:
    python examples/run_small_vlm.py path/to/image.jpg "What is in this image?"

Requires a CUDA-enabled PyTorch build, transformers, and Pillow. The first
run downloads HuggingFaceTB/SmolVLM-256M-Instruct model files from Hugging Face.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForMultimodalLM, AutoProcessor


MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a local image")
    parser.add_argument("question", help="Question about the image")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA GPU is required; check your NVIDIA driver and PyTorch installation")
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    with Image.open(args.image) as source:
        image = source.convert("RGB")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to("cuda")
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=80)

    answer_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    print(processor.decode(answer_ids, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
