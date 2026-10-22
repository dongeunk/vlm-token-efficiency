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

"""Gate physical Ours pruning with exact K=N checks, then smoke-test 2..5 POPE items.

Only the unchanged Spatial Floor + Global Coverage selector is executed.
Cached SmolVLM/standard POPE prompts/greedy16/batch1 are retained. No full30
evaluation, alternate selectors, downloads, or result overwrites are supported.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from examples.run_pope_pilot import MODEL_ID, GENERATION
from src.evaluation.pope import parse_yes_no
from src.evaluation.semantic_yes_no import PARSER_VERSION, explain_semantic_yes_no

SPECS = ((64, 4), (128, 8))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_cases(manifest, historical, maximum):
    """Use label/N diversity only, never model correctness or answer content."""
    keys = ("question_id", "image_id", "image", "text", "label")
    if [tuple(r[k] for k in keys) for r in manifest] != [tuple(r[k] for k in keys) for r in historical]:
        raise ValueError("Historical full-model records and original ordered manifest differ")
    sizes = sorted({r["original_visual_tokens"] for r in historical})
    if len(sizes) < 2:
        raise ValueError("Need two observed visual populations for this smoke protocol")
    chosen, used = [], set()
    # First two jointly cover both ground-truth labels and different N.
    for label, n in (("no", sizes[0]), ("yes", sizes[-1]),
                     ("yes", sizes[0]), ("no", sizes[-1])):
        eligible = [r for r in historical if r["question_id"] not in used
                    and r["label"] == label and r["original_visual_tokens"] == n]
        if eligible:
            chosen.append(eligible[0])
            used.add(eligible[0]["question_id"])
    for row in historical:
        if len(chosen) >= maximum:
            break
        if row["question_id"] not in used:
            chosen.append(row)
            used.add(row["question_id"])
    if len(chosen) < maximum:
        raise ValueError("Not enough distinct original questions")
    return chosen[:maximum]


def measured(function, torch):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    start = time.perf_counter()
    value = function()
    torch.cuda.synchronize()
    return value, dict(seconds=time.perf_counter() - start,
                       peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_cuda_increment_mib=(torch.cuda.max_memory_allocated() - before) / 2**20)


def prepare_case(row, model, processor, torch):
    from PIL import Image
    from examples.diagnose_global_coverage import projected_grid_shape

    with Image.open(row["image_path"]) as source:
        image = source.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": row["text"]},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to("cuda")
    details = processor.image_processor(images=[image], return_row_col_info=True, return_tensors="pt")
    rows, cols = int(details["rows"][0][0]), int(details["cols"][0][0])
    metadata_pixels_match = torch.equal(details["pixel_values"], inputs["pixel_values"].cpu())
    if not metadata_pixels_match or rows < 1 or cols < 1:
        raise RuntimeError("Actual metadata preprocessing differs from full prompt pixels")
    del details
    with torch.inference_mode():
        projected, feature_timing = measured(lambda: model.get_image_features(
            pixel_values=inputs["pixel_values"], pixel_attention_mask=inputs.get("pixel_attention_mask"),
            return_dict=True,
        ).pooler_output, torch)
    if projected.ndim != 3:
        raise RuntimeError("Expected [views,tokens,D] projected embeddings")
    views, per_view, dimension = projected.shape
    token_grid = projected_grid_shape(model.config, inputs["pixel_values"].shape, per_view)
    if views != rows * cols + 1 or inputs["pixel_values"].shape[1] != views:
        raise RuntimeError("View count differs from local crop metadata plus final global view")
    visual = projected.flatten(0, 1)
    n = visual.shape[0]
    if n != int((inputs["input_ids"] == model.config.image_token_id).sum().item()):
        raise RuntimeError("Original image placeholder count differs from projected N")
    if n != row["original_visual_tokens"]:
        raise RuntimeError("Actual N differs from the original same-model POPE record")
    if inputs["input_ids"].shape[0] != 1 or not (inputs["attention_mask"] == 1).all().item():
        raise RuntimeError("This smoke audit requires one unpadded prompt")
    return dict(row=row, inputs=inputs, prompt=prompt, visual=visual, n=n,
                rows=rows, cols=cols, views=views, per_view=per_view, dimension=dimension,
                n_local=rows * cols * per_view, token_grid=token_grid,
                feature_timing=feature_timing, metadata_pixels_match=metadata_pixels_match)


def selection_for(data, k, side, torch):
    from src.pruning.spatial_floor_coverage import spatial_floor_coverage_select
    result, timing = measured(lambda: spatial_floor_coverage_select(
        data["visual"], k, data["rows"], data["cols"], data["token_grid"], side,
    ), torch)
    if (result.selected_indices.numel() != k or result.selected_indices.unique().numel() != k
            or not (result.region_counts >= 1).all().item()):
        raise RuntimeError("Unchanged Ours selector violated exact-K/minimum-one invariants")
    return result, timing


def generate_record(data, model, processor, torch, prepared, method, k, side,
                    selection=None, selection_timing=None, full=False, capture=False):
    from src.inference.generation_audit import GenerationAudit

    model_inputs = data["inputs"] if full else prepared.model_inputs
    expected = prepared.model_inputs["inputs_embeds"]
    attention = prepared.model_inputs["attention_mask"]
    settings = dict(GENERATION, output_logits=capture)
    with GenerationAudit(model, expected, attention, capture_embeddings=capture,
                         expect_vision_calls=1 if full else 0) as audit:
        output, generation_timing = measured(lambda: model.generate(**model_inputs, **settings), torch)
    prompt_ids = model_inputs["input_ids"]
    length = prompt_ids.shape[1]
    if not torch.equal(output.sequences[:, :length], prompt_ids):
        raise RuntimeError("Generated sequences no longer retain the prompt prefix")
    generated = output.sequences[0, length:]
    if not 1 <= generated.numel() <= GENERATION["max_new_tokens"]:
        raise RuntimeError("Generated token count is outside greedy16 limits")
    trace = audit.report(int(generated.numel()))
    raw = processor.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    strict = parse_yes_no(raw)
    semantic = explain_semantic_yes_no(raw)
    selected = (selection.selected_indices if selection is not None
                else torch.arange(data["n"], device=data["visual"].device))
    local = int((selected < data["n_local"]).sum().item())
    row = data["row"]
    record = dict(question_id=row["question_id"], image_id=row["image_id"], image=row["image"],
                  question=row["text"], label=row["label"], method=method,
                  n=data["n"], k=k, g=side * side if side else None,
                  region_grid=[side, side] if side else None,
                  crop_rows=data["rows"], crop_cols=data["cols"], total_views=data["views"],
                  tokens_per_view=data["per_view"], token_grid=list(data["token_grid"]),
                  original_local_tokens=data["n_local"], original_global_tokens=data["n"]-data["n_local"],
                  selected_local=local, selected_global=k-local,
                  selected_visual_indices=selected.cpu().tolist(),
                  selected_original_sequence_positions=prepared.visual_positions[selected].cpu().tolist(),
                  original_sequence_length=prepared.original_length, llm_input_sequence_length=length,
                  actual_llm_prefill_length=trace["actual_prefill_length"], removed_visual_tokens=data["n"]-k,
                  sequence_length_delta=prepared.original_length-length,
                  preservation_checks=prepared.checks, generation_audit=trace,
                  metadata_pixels_match=data["metadata_pixels_match"], prompt_text=data["prompt"],
                  generated_token_ids=generated.cpu().tolist(), raw_output=raw,
                  raw_output_with_special_tokens=processor.decode(
                      generated, skip_special_tokens=False, clean_up_tokenization_spaces=False),
                  strict_parsed_answer=strict, strict_correct=strict==row["label"],
                  semantic_parsed_answer=semantic.answer, semantic_correct=semantic.answer==row["label"],
                  semantic_rule_id=semantic.rule_id, semantic_reason=semantic.reason,
                  feature_extraction=data["feature_timing"], selection=selection_timing,
                  generation=generation_timing, error=None)
    if selection is not None:
        if (local != selection.selected_local or k-local != selection.selected_global
                or data["n"]-k != prepared.original_length-length):
            raise RuntimeError("Selected populations or physical length delta disagree")
        record.update(floor_indices=selection.floor_indices.cpu().tolist(),
                      completion_indices=selection.completion_indices.cpu().tolist(),
                      region_counts=selection.region_counts.cpu().tolist())
    aux = dict(prefill_embeddings=audit.prefill_embeddings,
               first_logits=output.logits[0].detach().float().cpu() if capture else None)
    print(f"qid={row['question_id']} {method} N/K={data['n']}/{k} "
          f"L={prepared.original_length}->{length} L/G={local}/{k-local} "
          f"strict={strict} semantic={semantic.answer} raw={raw!r}", flush=True)
    return record, aux


def main_run(args):
    from PIL import Image
    from src.inference.pruned_vlm import build_pruned_inputs

    original_manifest = ROOT / "results/pope_pilot/sample_manifest.jsonl"
    full_predictions = ROOT / "results/pope_pilot/full_model_seed42/predictions.jsonl"
    manifest, historical = read_jsonl(original_manifest), read_jsonl(full_predictions)
    planned = choose_cases(manifest, historical, args.max_questions)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Existing smoke outputs are preserved; choose a fresh --output-dir")
    for row in planned:
        with Image.open(row["image_path"]) as source:
            if source.format != "JPEG":
                raise ValueError("Expected original COCO JPEG")
            source.verify()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected_paths = [original_manifest, full_predictions, ROOT / "src/pruning/spatial_floor_coverage.py",
                       ROOT / "src/evaluation/pope.py", ROOT / "src/evaluation/semantic_yes_no.py"]
    config = dict(created_utc=datetime.now(timezone.utc).isoformat(), model=MODEL_ID,
                  model_local_files_only=True, seed=42, dtype="float16", batch_size=1,
                  attention_implementation="eager", generation=dict(GENERATION),
                  semantic_parser_version=PARSER_VERSION,
                  selector="unchanged spatial_floor_coverage_select; all local+global candidates/demands",
                  specs=[dict(k=k, g=side*side, region_grid=[side, side]) for k, side in SPECS],
                  planned_question_ids=[r["question_id"] for r in planned], initial_question_ids=[r["question_id"] for r in planned[:2]],
                  sampling="metadata-only: no/minN, yes/maxN, yes/minN, no/maxN, then original order; no correctness selection",
                  position_policy="delete then contiguous short-mask positions; original relative token order preserved",
                  cache_policy="new per-generate DynamicCache; no dense cache carried into pruning",
                  timing_note="CUDA-synchronized single run, includes debug hooks; no speed claim. Direct generation excludes shared features; full gate generation includes vision.",
                  original_files_sha256={str(p.resolve()): digest(p) for p in protected_paths},
                  versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "Pillow", "torchvision")})
    save_json(args.output_dir / "config.json", config)
    (args.output_dir / "selected_manifest.jsonl").write_text("".join(json.dumps(dict(
        question_id=r["question_id"], image_id=r["image_id"], image=r["image"], text=r["text"],
        label=r["label"], image_path=r["image_path"], expected_visual_n=r["original_visual_tokens"]
    ), ensure_ascii=False)+"\n" for r in planned), encoding="utf-8")
    summary = dict(status="running", kn_identity_passed=False, initial_two_passed=False,
                   expanded_to_five=False, planned_questions=len(planned), completed_questions=[],
                   ours_generation_count=0, errors=[], further_experiments="not_started")
    stage, current = "model_load", None
    started = time.perf_counter()
    try:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        processor = AutoProcessor.from_pretrained(MODEL_ID, local_files_only=True)
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID, local_files_only=True, dtype=torch.float16, attn_implementation="eager",
        ).to("cuda").eval()
        config.update(gpu=torch.cuda.get_device_name(0), cuda_build=torch.version.cuda,
                      model_commit=getattr(model.config, "_commit_hash", None),
                      loaded_generation_config=model.generation_config.to_dict())
        save_json(args.output_dir / "config.json", config)
        cache = {}
        # Gate BOTH observed populations before any K=64/128 generation.
        for row in planned[:2]:
            current, stage = row["question_id"], "kn_identity"
            print(f"Preparing K=N gate qid={current}...", flush=True)
            data = prepare_case(row, model, processor, torch)
            cache[current] = data
            with torch.inference_mode():
                selected, selection_timing = selection_for(data, data["n"], 4, torch)
                if not torch.equal(selected.selected_indices, torch.arange(data["n"], device="cuda")):
                    raise RuntimeError("K=N unchanged selector did not return original all-token order")
                identity = build_pruned_inputs(model, data["inputs"], data["visual"], selected.selected_indices)
                full, full_aux = generate_record(data, model, processor, torch, identity,
                    "full_model_gate", data["n"], None, full=True, capture=True)
                noop, noop_aux = generate_record(data, model, processor, torch, identity,
                    "ours_k_equals_n", data["n"], 4, selected, selection_timing, capture=True)
            embed_diff = (full_aux["prefill_embeddings"].float()-noop_aux["prefill_embeddings"].float()).abs().max().item()
            logit_diff = (full_aux["first_logits"]-noop_aux["first_logits"]).abs().max().item()
            checks = dict(input_embeddings_exact=torch.equal(full_aux["prefill_embeddings"], noop_aux["prefill_embeddings"]),
                          first_step_logits_exact=torch.equal(full_aux["first_logits"], noop_aux["first_logits"]),
                          generated_token_ids_equal=full["generated_token_ids"]==noop["generated_token_ids"],
                          raw_answer_equal=full["raw_output"]==noop["raw_output"],
                          historical_full_token_ids_equal=full["generated_token_ids"]==row["generated_token_ids"])
            comparison = dict(question_id=current, n=data["n"], passed=all(checks.values()), checks=checks,
                              input_embeddings_max_abs_difference=embed_diff,
                              first_step_logits_max_abs_difference=logit_diff, full=full, noop=noop)
            append_jsonl(args.output_dir / "kn_identity.jsonl", comparison)
            if not comparison["passed"]:
                raise RuntimeError(f"K=N identity failed: {checks}; no smaller-K experiment permitted")
            print(f"K=N PASS qid={current}; embedding/logit maxdiff={embed_diff}/{logit_diff}", flush=True)
        summary["kn_identity_passed"] = True

        def smoke(row):
            nonlocal stage, current
            current, stage = row["question_id"], "ours_pruning"
            data = cache.get(current)
            if data is None:
                data = prepare_case(row, model, processor, torch)
            for k, side in SPECS:
                with torch.inference_mode():
                    selected, selection_timing = selection_for(data, k, side, torch)
                    prepared = build_pruned_inputs(model, data["inputs"], data["visual"], selected.selected_indices)
                    if prepared.pruned_length >= prepared.original_length:
                        raise RuntimeError("K<N did not physically reduce sequence length")
                    result, _ = generate_record(data, model, processor, torch, prepared,
                        "ours_spatial_floor_gc", k, side, selected, selection_timing)
                append_jsonl(args.output_dir / "predictions.jsonl", result)
                summary["ours_generation_count"] += 1
            summary["completed_questions"].append(current)

        for row in planned[:2]:
            smoke(row)
        summary["initial_two_passed"] = True
        save_json(args.output_dir / "initial_two_passed.json", dict(
            passed=True, question_ids=summary["completed_questions"], ours_generations=4,
            note="Both budgets passed all physical/runtime/cache checks before expansion"))
        cache.clear()
        print("Initial two questions passed; expanding only to the remaining planned items...", flush=True)
        for row in planned[2:]:
            smoke(row)
        summary.update(status="completed", expanded_to_five=len(planned)==5)
    except Exception as error:
        summary.update(status="failed_stop", stopped_stage=stage,
                       errors=[dict(question_id=current, stage=stage,
                                    error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())])
        print(traceback.format_exc(), file=sys.stderr, flush=True)
    summary.update(runtime_seconds=time.perf_counter()-started,
                   original_files_unchanged=all(digest(Path(p))==h for p,h in config["original_files_sha256"].items()))
    save_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["status"]=="completed" and summary["original_files_unchanged"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-questions", type=int, choices=(2,3,4,5), default=5)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/pope_pruning_smoke")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    return main_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
