# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 dongeunk (original project contributions).
# See LICENSE for terms. Publication adds comments only; experimental body unchanged.

"""Write-once protocol and bounded source/data integrity records."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/smolvlm_s2_coverage_phase_s3a/phase_s3a_20261114_01"
REVISION = "7e3e67edbbed1bf9888184d9df282b700a323964"
SNAPSHOT = Path.home() / f".cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots/{REVISION}"
ANNOTATION = "data/pope/coco_pope_random.json"
SOURCE = "data/pope/SOURCE.json"
OLD30 = "results/pope_pilot/sample_manifest.jsonl"
OLD80 = "results/pope_spatial_floor_80_preparation/cohort80_five_methods_v2/selected_images.jsonl"
POOL = "results/pope_spatial_floor_80_preparation/candidate_pool.csv"
SCREEN = "results/pope_spatial_floor_80_preparation/cd_screening_470_v1/per_image.csv"
HISTORY = "results/pope_spatial_floor_80_preparation/cohort80_five_methods_v2/cpu_images.jsonl"
ACTUAL = "results/smolvlm_s2_coverage_phase_s2a/phase_s2a_20261114_01/gpu_run_03/ACTUAL_INPUT.json"
METHODS = ("full", "s2_l", "s2_u", "s2_c")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def save_lines(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def protected_paths():
    """Only reviewed contracts, dependencies, exclusion manifests and configs.

    No weights, old prediction logs, repository-wide walk or Qwen SHA rescan.
    """
    files = [ROOT / p for p in (ANNOTATION, SOURCE, OLD30, OLD80, POOL, SCREEN, HISTORY, ACTUAL)]
    for namespace in ("smol_s2_study", "smol_s2_runtime", "smol_s2_pruning_runtime"):
        files += sorted((ROOT / "src" / namespace).glob("*.py"))
    files += sorted((ROOT / "results/smolvlm_s2_coverage_phase_s0/phase_s0_20261114_01").glob("*.md"))
    for p in (
        "results/smolvlm_s2_coverage_phase_s1/phase_s1_20261114_01/REPORT.md",
        "results/smolvlm_s2_coverage_phase_s2a/phase_s2a_20261114_01/REPORT.md",
        "results/smolvlm_s2_coverage_phase_s2b/phase_s2b_20261114_01/REPORT.md",
        "results/smolvlm_s2_coverage_phase_s2b/phase_s2b_20261114_01/S3_ENTRY_CONDITIONS.md",
        "src/evaluation/pope.py", "src/evaluation/semantic_yes_no.py",
        "examples/run_pope80_five_methods.py", "examples/run_pope_local_floor_paired.py",
        "examples/run_small_vlm.py", "examples/run_smol_s2a_gpu.py", "examples/run_smol_s2b_gpu.py",
        "src/pruning/spatial_floor_coverage.py", "src/pruning/coverage.py",
    ):
        files.append(ROOT / p)
    return sorted(set(files))


def capture_protected():
    return dict(created_utc=now(), scope="bounded reviewed dependencies only", full_project_sha_scan=False,
                files=[dict(path=p.relative_to(ROOT).as_posix(), sha256=sha(p), bytes=p.stat().st_size)
                       for p in protected_paths()])


def verify_protected(baseline):
    changed = [row["path"] for row in baseline["files"]
               if not (ROOT / row["path"]).is_file() or sha(ROOT / row["path"]) != row["sha256"]]
    return dict(passed=not changed, checked_files=len(baseline["files"]), changed=changed,
                scope=baseline["scope"], full_project_sha_scan=False,
                unlisted_old_files="not independently rehashed; new output write boundary enforced")


def protocol():
    """Decisions declared BEFORE reading selection outcomes; no score inputs."""
    return dict(
        version="smol_s3a_v1", created_utc=now(), phase="S3-A CPU ONLY",
        model=dict(id="HuggingFaceTB/SmolVLM-256M-Instruct", revision=REVISION,
                   architecture="Idefics3ForConditionalGeneration + Llama", dtype="float16",
                   attention_backend="eager", batch_size=1, local_files_only=True),
        generation=dict(max_new_tokens=16, do_sample=False, num_beams=1, use_cache=True,
                        eos_token_id=49279, pad_token_id=2, bos_token_id=0,
                        logits_processors="none; verify native inactive controls in S3-B"),
        preprocessing=dict(processor="Idefics3Processor", image_processor="Idefics3ImageProcessor",
                           backend="torchvision (S2-A observed); verify S3-B", image_mode="RGB",
                           do_image_splitting=True, longest_edge=2048, max_image_size=512,
                           prompt="<|im_start|>User:<image>{official_question}<end_of_utterance>\nAssistant:",
                           extra_yes_no_instruction=False, prompt_question_unchanged=True),
        pruning=dict(methods=list(METHODS), local_budget=48, global_budget=16, total_budget=64,
                     local_regions=[4, 4], local_base=1, local_extra=32, query_weight=False,
                     cell_and_erc="unchanged S1/S2-B common partition; per-cell Layer0 ERC argmax",
                     global_rule="unchanged single global 8x8 region, recursive 16 cells + ERC",
                     location="after full decoder block 0; gather Layer0 KV and preserve nonvisual tokens",
                     coverage=dict(demand="local+global", candidates="local", dtype="FP32",
                                   calibration="cosine; row mean-center; nonnegative clip", weights=1)),
        selection=dict(seed=42, pilot_images=10, main_images=200, questions_per_image=6,
                       canonical_order="ascending integer image_id; question_id within image",
                       eligibility=["official original COCO POPE Random image ID",
                                    "not in old30 or old80 (image ID union)",
                                    "exactly six unique official question IDs/texts: yes3/no3",
                                    "local JPEG filename/ID/dimensions/decode and historical source SHA match"],
                       fatal_errors=["invalid annotation JSON/field/duplicate question ID",
                                     "source annotation SHA/provenance mismatch", "old exclusion duplicate or overlap",
                                     "candidate manifest vs actual filename ID population mismatch"],
                       incomplete_policy="exclude whole image before sampling; record reason; no repair/download",
                       main_rule="FIRST random.Random(42).sample(sorted eligible IDs,200); retain draw order",
                       pilot_rule="independent random.Random(42); from remaining IDs select one per 832/1088 "
                                  "stratum (prefer stored processor observation; else static geometry prediction), "
                                  "then uniform fill to10; no outcome-based replacement",
                       geometry_status="static arithmetic of installed resize sizes only, NOT observed preprocessing",
                       replacements_after_lock=False, old_selection_flags_ignored=True,
                       annotation_scope="original POPE Random; no COCO annotation support or C/D filtering"),
        scoring=dict(primary_parser="visual_semantic_yes_no_v1 (unchanged existing source)",
                     strict_parser="existing conservative leading yes/no parse_yes_no",
                     primary_metric="pope_f1", primary_comparison="s2_c - s2_l",
                     pope_accuracy_equals_semantic_accuracy=True,
                     official_pope_equivalence=False,
                     invalid="separate prediction column; always wrong for accuracy; invalid_yes enters recall FN",
                     f1="2TP/(2TP+FP+FN+invalid_yes); zero denominator=0",
                     missing_or_runtime_error="fail closed: no complete metrics or bootstrap; no silent drop",
                     secondary=["pope_accuracy", "precision", "recall", "invalid_count", "invalid_rate", "strict_accuracy"],
                     truncation="record EOS and generated IDs; no EOS at16 = potential truncation, not semantic certainty"),
        statistics=dict(unit="image paired cluster (all six questions)", resamples=10000, seed=42,
                        rng="numpy.random.Generator(PCG64(42))", confidence=0.95,
                        interval="percentile [2.5,97.5], numpy linear quantile",
                        aggregation="resample image confusion counts, recompute micro F1; never mean image F1",
                        primary="C-L pope_f1", other_comparisons="exploratory; no multiplicity-adjusted claim"),
        authorization=dict(cuda=False, model_load=False, processor_execution=False, generation=False,
                           downloads=False, installs=False, old_file_modification=False, paid_services=False,
                           git_commit_push=False, s2prune_license="HOLD"),
        python=platform.python_version(),
    )


def precommit():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_new(OUTPUT / "PROTECTED_BEFORE.json", capture_protected())
    p = protocol()
    config_files = {}
    for name in ("config.json", "preprocessor_config.json", "processor_config.json", "chat_template.json",
                 "generation_config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"):
        path = SNAPSHOT / name
        config_files[name] = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
    p["cached_config_files"] = config_files
    p["weights"] = "not opened, loaded or rehashed"
    p["input_sources"] = {name: dict(path=name, sha256=sha(ROOT / name))
                          for name in (ANNOTATION, SOURCE, OLD30, OLD80, POOL, SCREEN, HISTORY, ACTUAL)}
    save_new(OUTPUT / "PROTOCOL_PRECOMMIT.json", p)
    return dict(path=str(OUTPUT), protocol_sha256=sha(OUTPUT / "PROTOCOL_PRECOMMIT.json"))
