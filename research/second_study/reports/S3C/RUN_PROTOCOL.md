> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s3c/phase_s3c_20261115_01/RUN_PROTOCOL.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Phase S3-C frozen main evaluation

This directory contains the main evaluation authorized on 2026-11-15 (Asia/Seoul). The frozen S3-A cohort is exactly 200 images / 1,200 official POPE Random questions, disjoint from the old 110 images and the pilot 10. The original manifests remain in S3-A, protected by their original SHA256 index. No new sample selection occurs here.

## Configuration and implementation

`CONFIG_LOCK.json` copies the pilot's checkpoint/revision, processor, prompt, FP16/eager, greedy generation with max_new_tokens=16, allocation, cells, Layer-0 ERC, global16, RoPE/cache, scoring and statistics settings exactly. Its hash differs because the dataset scope, phase and authorized generation cap differ. `GATE_A.json` records actual CPU processor verification of all 1,200 inputs, 576/832/1088 support, Yes600/No600 and source/config equality. Model loading and CUDA are blocked in CPU preparation/tests.

New code only changes outer data loading, split metadata, the durable attempt ledger and segment checkpointing. Native HF Full and the existing `src/smol_s3_functional/runtime.py::run_pruned_generation` are executed unchanged, as are the allocator, cell/ERC routines, read-only monitor, frozen parser, metrics and bootstrap. Three pruning methods differ only in local region allocation. Every query recomputes allocation and ERC; no new allocation/selection cache or runtime optimization is introduced.

## Execution and recovery

- One local GTX 1660 SUPER, one FP16 model instance, sequential batch1.
- Fixed image order, fixed question order, Full/L/U/C per question. Maximum 4,800 reserved model calls, no warm-up or additional generation.
- Twenty segments, each ten images / 240 calls. Results and evidence are saved per call; each completed segment rehashes its 240 record files and writes `SEGMENT_COMPLETE.json` before the next segment starts.
- Keys are `(image_id, question_id, method, configuration_hash)`. `EVENTS.jsonl` appends and fsyncs a reservation before each call and a record SHA after completion. Resume verifies and reuses completed records. An ambiguous reserved call without a durable validated record is stopped, never silently generated again.
- The first two fixed main questions receive the pilot's full observers (eight calls, included in 4,800); remaining calls use the same read-only light audit. All calls check input tensors, dense prefix/native equality, cell selection, nonvisual preservation, K64, original logical positions, causal/RoPE gather and all30-layer KV append.
- Any mapping, finite ERC, cache, decode, input/config mismatch or OOM stops the worker. No algorithm/parameter adjustment or cohort exclusion follows automatically.
- The process requests temporary Windows sleep prevention while running, cleared at process exit. No persistent power setting changes.

## Complete-only analysis

Progress monitoring reads ledger counts and resource diagnostics only. Quality metrics and CIs are blocked until all 4,800 unique successful results and all scientific evidence hashes are verified. No intermediate quality-driven change is permitted.

Primary contrast: S2-C minus S2-L Semantic F1. Frozen image-cluster paired bootstrap: all six questions per image together, 10,000 resamples, PCG64 seed42, percentile95% CI with linear quantiles and micro confusion recomputation. Additional contrasts are exploratory. Invalid responses stay in the denominator and are wrong for Accuracy; invalid_yes enters the positive F1 recall denominator, and invalid_no never counts as TN. Accuracy, Invalid and Strict Accuracy accompany F1. No EOS at16 is possible truncation, not proof of semantic incompleteness; no longer generation or response replacement occurs.

The 200-image POPE Random subset and frozen semantic parser do not constitute the official full POPE benchmark. Token-structure strata and Coverage/accuracy associations are descriptive, especially the one-image 576-token stratum. Instrumented, fixed-order latency and VRAM are diagnostics, not a controlled speed benchmark.

## Preservation and stop condition

Only this new result directory and new S3-C orchestration/test files are written. Bounded reused source/report/manifest/config hashes and checkpoint file stats are compared before and after; no whole-project census or model-weight SHA scan. S²Prune license HOLD continues. No downloads, installations, paid/remote GPU, git commit or push. Stop after S3-C results/report; ablation ideas require separate authorization.
