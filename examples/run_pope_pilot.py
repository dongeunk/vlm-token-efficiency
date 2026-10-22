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

"""Prepare/evaluate a fixed 30-question, full-model COCO POPE pilot.

Uses original output/coco/coco_pope_random.json, unchanged question text,
SmolVLM's standard chat template, cached model files, and batch_size=1.
No downloads or pruning are implemented. Missing data stops before model load.
The balanced, image-diverse pilot and strict invalid-aware parser differ from
the official full-subset positional evaluation; results are not full POPE scores.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.pope import compute_metrics, load_annotations, parse_yes_no, sample_balanced

MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
POPE_COMMIT = "08d957b917e5a378a2f99d35b6293c536a66298b"
ANNOTATION_SHA256 = "ac25245170b975a5bdf9080b23fd431dfe6be458bc038259c1f4f09a6bef7994"
ANNOTATION_URL = (
    f"https://raw.githubusercontent.com/AoiDragon/POPE/{POPE_COMMIT}/"
    "output/coco/coco_pope_random.json"
)
COCO_DOWNLOAD_PAGE = "https://cocodataset.org/#download"
GENERATION = dict(max_new_tokens=16, do_sample=False, num_beams=1,
                  use_cache=True, return_dict_in_generate=True)
OUTPUT_NAMES = ("config.json", "sample_manifest.jsonl", "preflight.json",
                "predictions.jsonl", "summary.json")


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    # Flush each completed response so an interrupted pilot retains its outputs.
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def package_versions():
    result = {"python": sys.version, "executable": sys.executable}
    for name in ("torch", "transformers", "Pillow", "torchvision"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def default_config(args):
    return dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                model=MODEL_ID, method="full_model", pruning=False, retained="K=N for every image",
                model_local_files_only=True, batch_size=1, device="cuda", dtype="float16",
                attention_implementation="eager", seed=args.seed, n_questions=30,
                label_counts={"yes": 15, "no": 15}, subset="original_coco_random",
                annotation_path=str(args.annotations.resolve()), annotation_sha256=None,
                image_dir=str(args.image_dir.resolve()), pope_commit=POPE_COMMIT,
                expected_official_annotation_url=ANNOTATION_URL,
                annotation_source_verified=False,
                sampling="seeded maximum-distinct-image balanced matching, then seeded order",
                question_format="exact annotation text, no appended answer instruction",
                chat_template="processor.apply_chat_template(messages, add_generation_prompt=True)",
                generation=dict(GENERATION), parsing="strict leading yes/no; ambiguity/uncertainty invalid",
                invalid_scoring="incorrect; separate confusion column; invalid yes included in recall denominator",
                quality_review_thresholds=dict(min_accuracy=0.70, max_invalid_ratio=0.10,
                    note="Operational pilot review thresholds, not official POPE requirements; no automatic next experiment"),
                versions=package_versions())


def prepare(config, annotations, image_dir):
    """Select before checking image presence; never resample to available images."""
    preflight = dict(annotation_exists=annotations.is_file(), selected_count=0,
                     selected_label_counts={}, distinct_images=0, missing_images=[],
                     data_ready=False, errors=[], acquisition=dict(
                         annotation=ANNOTATION_URL, coco_download_page=COCO_DOWNLOAD_PAGE,
                         full_image_archive="http://images.cocodataset.org/zips/val2014.zip",
                         individual_image_template="http://images.cocodataset.org/val2014/{filename}",
                         note="No image downloads performed. Use original COCO val2014 JPEGs; no substitutes."))
    if not annotations.is_file():
        preflight["errors"].append("Official original COCO POPE annotation file is missing")
        return [], preflight
    config["annotation_sha256"] = hashlib.sha256(annotations.read_bytes()).hexdigest()
    if config["annotation_sha256"] != ANNOTATION_SHA256:
        preflight["errors"].append("Annotation SHA256 differs from the pinned official original COCO random file")
        return [], preflight
    config["annotation_source_verified"] = True
    provenance_path = annotations.parent / "SOURCE.json"
    if provenance_path.is_file():
        source = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (source.get("sha256") == config["annotation_sha256"]
                and source.get("url") == ANNOTATION_URL and source.get("commit") == POPE_COMMIT):
            config["annotation_source_verified"] = True
            config["annotation_provenance"] = source
    records = load_annotations(annotations)
    selected = sample_balanced(records, per_label=15, seed=config["seed"])
    manifest = []
    for order, record in enumerate(selected):
        path = image_dir / record["image"]
        manifest.append(dict(record, sample_order=order, image_path=str(path.resolve()),
                             image_exists=path.is_file()))
    preflight.update(annotation_count=len(records), selected_count=len(manifest),
                     selected_label_counts=dict(Counter(r["label"] for r in manifest)),
                     distinct_images=len({r["image"] for r in manifest}),
                     missing_images=[dict(image_id=r["image_id"], image=r["image"],
                                          path=r["image_path"]) for r in manifest if not r["image_exists"]])
    if preflight["missing_images"]:
        preflight["errors"].append(f"Missing {len(preflight['missing_images'])} selected original COCO images")
    preflight["data_ready"] = not preflight["errors"]
    return manifest, preflight


def generate_answer(model, processor, image, question, torch_module, device="cuda"):
    """Use the ordinary full multimodal generate path; never insert/prune tokens."""
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)
    prompt_length = inputs["input_ids"].shape[1]
    n = int((inputs["input_ids"] == model.config.image_token_id).sum().item())
    if inputs["input_ids"].shape[0] != 1 or n < 1 or "pixel_values" not in inputs:
        raise RuntimeError("Expected batch_size=1 with real pixels and visual placeholders")
    cuda = torch_module.device(device).type == "cuda"
    if cuda:
        torch_module.cuda.synchronize(device)
        torch_module.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    with torch_module.inference_mode():
        result = model.generate(**inputs, **GENERATION)
    if cuda:
        torch_module.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    sequences = result.sequences
    if (sequences.shape[0] != 1 or sequences.shape[1] < prompt_length
            or not torch_module.equal(sequences[:, :prompt_length], inputs["input_ids"])):
        raise RuntimeError("Generation did not retain the original prompt token prefix")
    generated = sequences[0, prompt_length:]
    if generated.numel() > GENERATION["max_new_tokens"]:
        raise RuntimeError("Generation exceeded the configured token budget")
    raw = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return dict(prompt_text=prompt, input_sequence_length=prompt_length,
                original_visual_tokens=n, retained_visual_tokens=n, generation_seconds=seconds,
                generation_peak_cuda_mib=(torch_module.cuda.max_memory_allocated(device) / 2**20 if cuda else None),
                generated_token_ids=generated.cpu().tolist(), raw_output=raw,
                raw_output_with_special_tokens=processor.decode(
                    generated, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                parsed_answer=parse_yes_no(raw))


def quality_review(metrics, records):
    reasons = []
    if metrics["accuracy"] < 0.70:
        reasons.append("Accuracy below operational pilot threshold 0.70")
    if metrics["invalid_ratio"] > 0.10:
        reasons.append("Invalid ratio above operational pilot threshold 0.10")
    return dict(status="review_required" if reasons else "pilot_checks_passed", reasons=reasons,
                prediction_counts=dict(Counter(r["parsed_answer"] or "invalid" for r in records)),
                invalid_examples=[dict(question_id=r["question_id"], raw_output=r["raw_output"])
                                  for r in records if r["parsed_answer"] is None],
                token_budget_reached_question_ids=[r["question_id"] for r in records
                                                   if len(r["generated_token_ids"]) == GENERATION["max_new_tokens"]],
                notes=["30-question stratified pilot, not a full benchmark or tuning set",
                       "Inspect raw outputs for answer-format mismatch, truncation, yes bias, and object recognition errors",
                       "Reaching 16 tokens alone does not prove truncation: the last token can be EOS",
                       "No next experiment runs automatically, regardless of this status"])


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    occupied = [name for name in OUTPUT_NAMES if (args.output_dir / name).exists()]
    if occupied:
        raise FileExistsError(f"Preserving existing outputs {occupied}; choose a fresh --output-dir")
    config = default_config(args)
    save_json(args.output_dir / "config.json", config)
    try:
        manifest, preflight = prepare(config, args.annotations, args.image_dir)
    except (ValueError, OSError) as error:
        manifest = []
        preflight = dict(data_ready=False, errors=[f"{type(error).__name__}: {error}"])
    save_json(args.output_dir / "config.json", config)
    save_json(args.output_dir / "preflight.json", preflight)
    (args.output_dir / "sample_manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest), encoding="utf-8")
    summary = dict(status="blocked_missing_or_invalid_data", expected_questions=30,
                   completed_questions=0, metrics=None, model_loaded=False, generation_executed=False,
                   errors=preflight["errors"], next_experiment="not_started")
    if not preflight["data_ready"] or args.prepare_only:
        if preflight["data_ready"]:
            summary.update(status="prepared_only", errors=[])
        save_json(args.output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Selected {len(manifest)} questions; distinct images={preflight.get('distinct_images', 0)}; "
              f"missing images={len(preflight.get('missing_images', []))}")
        print(f"Results: {args.output_dir.resolve()}")
        return 0 if preflight["data_ready"] else 2

    # Data checks complete before importing torch/Transformers or allocating GPU memory.
    completed, errors = [], []
    start = time.perf_counter()
    try:
        from PIL import Image

        # Verify real data before importing/model-loading the GPU stack.
        for item in manifest:
            with Image.open(item["image_path"]) as source:
                source.verify()
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU unavailable; environment was not modified")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID, local_files_only=True, dtype=torch.float16, attn_implementation="eager",
        ).to("cuda").eval()
        summary["model_loaded"] = True
        config.update(gpu=torch.cuda.get_device_name(0), cuda_build=torch.version.cuda,
                      chat_template_sha256=hashlib.sha256(str(processor.chat_template).encode()).hexdigest(),
                      model_commit=getattr(model.config, "_commit_hash", None),
                      loaded_generation_config=model.generation_config.to_dict())
        save_json(args.output_dir / "config.json", config)
        for order, item in enumerate(manifest, 1):
            record = dict(item, run_config_file="config.json", generation_success=False)
            try:
                with Image.open(item["image_path"]) as source:
                    image = source.convert("RGB")
                summary["generation_executed"] = True
                answer = generate_answer(model, processor, image, item["text"], torch)
                record.update(answer, correct=answer["parsed_answer"] == item["label"],
                              generation_success=True)
                completed.append(record)
                append_jsonl(args.output_dir / "predictions.jsonl", record)
                print(f"[{order}/30] qid={item['question_id']} gt={item['label']} "
                      f"parsed={answer['parsed_answer']} raw={answer['raw_output']!r}", flush=True)
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
                record.update(raw_output=None, parsed_answer=None, correct=False, error=failure)
                append_jsonl(args.output_dir / "predictions.jsonl", record)
                errors.append(dict(question_id=item["question_id"], error=failure))
                break  # No alternate decoding, resampling, or pruning after a failure/OOM.
    except Exception as error:
        errors.append(dict(error=f"{type(error).__name__}: {error}"))
    complete = len(completed) == 30 and not errors
    summary.update(status="completed" if complete else "runtime_failed", completed_questions=len(completed),
                   metrics=compute_metrics(completed) if complete else None,
                   partial_metrics=compute_metrics(completed) if completed and not complete else None,
                   errors=errors, runtime_seconds=time.perf_counter() - start)
    if complete:
        summary["quality_review"] = quality_review(summary["metrics"], completed)
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {args.output_dir.resolve()}")
    return 0 if complete else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=PROJECT_ROOT / "data/pope/coco_pope_random.json")
    parser.add_argument("--image-dir", type=Path, default=PROJECT_ROOT / "data/coco/val2014")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/pope_pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prepare-only", action="store_true", help="Save selection/data checks without loading the model")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        return run(args)
    except (FileExistsError, OSError, ValueError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
