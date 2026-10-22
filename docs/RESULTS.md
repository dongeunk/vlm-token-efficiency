# Historical results and evaluation contract

Publication status: these stored numerical results are unchanged. Included aggregate files do not supply the raw inputs or held integrated runner needed to replay the experiment. See [REPRODUCIBILITY](REPRODUCIBILITY.md).

## Scope

The stored experiment completed 300/300 pruning generations: 30 fixed original COCO POPE random questions x five methods x two budgets. Full Model outputs were reused. This is a small pilot, not official full-POPE performance. No new inference or rescoring was run for this document.

## Parser contract

Accuracy includes all questions and counts invalid as wrong. The Yes class is positive. Precision=TP/(TP+FP); recall=TP/(TP+FN+invalid_yes); F1 is their harmonic mean. The stored FN field counts explicit No predictions on Yes labels; invalid Yes examples are a separate field and still reduce recall. Confusion rows are ground truth Yes/No and columns are Yes/No/invalid.

Strict is the original constrained parser; semantic-v1 adds bounded English semantic rules, including explicit negation. Invalid answers are not silently converted to No. Semantic v1 does not take labels/IDs but was developed after raw-output inspection; held-out generalization is not established. Both results and the private raw evidence are preserved.

## Full baseline

Original Strict: 18/30, invalid12. Separate semantic rescoring of the same raw30: 26/30, invalid2. No new Full inference occurred in the five-method comparison. The eight-correct-answer increase is evaluation correction, not model improvement.

## Five-method metrics

| Method | K | Semantic correct | Strict correct | Semantic invalid | Strict invalid | Semantic F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random | 64 | 25/30 | 19/30 | 0 | 8 | 0.800000 |
| Random | 128 | 28/30 | 19/30 | 0 | 11 | 0.928571 |
| Pure GC | 64 | 28/30 | 22/30 | 0 | 6 | 0.928571 |
| Pure GC | 128 | 27/30 | 21/30 | 2 | 9 | 0.965517 |
| Ours | 64 | 25/30 | 23/30 | 3 | 5 | 0.846154 |
| Ours | 128 | 27/30 | 24/30 | 2 | 6 | 0.965517 |
| A: Local-only | 64 | 28/30 | 24/30 | 1 | 5 | 0.965517 |
| A: Local-only | 128 | 26/30 | 22/30 | 3 | 7 | 0.965517 |
| B: Laplacian quota | 64 | 27/30 | 23/30 | 0 | 4 | 0.888889 |
| B: Laplacian quota | 128 | 25/30 | 20/30 | 2 | 7 | 0.896552 |

Machine-readable precision/recall/F1, prediction/invalid ratios and full Yes/No/invalid confusion matrices are retained in [aggregate_metrics.json](../results/pope_controlled_comparison/full/aggregate_metrics.json).

## Paired semantic outcomes

Left-only and right-only mean correct on the same question; invalid remains incorrect.

| Pair (left / right) | K | Left only | Right only | Both correct | Both incorrect | Net left-right |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ours / Pure GC | 64 | 0 | 3 | 25 | 2 | -3 |
| A: Local-only / B: Laplacian quota | 64 | 2 | 1 | 26 | 1 | +1 |
| Ours / A: Local-only | 64 | 1 | 4 | 24 | 1 | -3 |
| Ours / Pure GC | 128 | 2 | 2 | 25 | 1 | +0 |
| A: Local-only / B: Laplacian quota | 128 | 2 | 1 | 24 | 3 | +1 |
| Ours / A: Local-only | 128 | 4 | 3 | 23 | 0 | +1 |

## Interpretation

Ours is not consistently better than Pure GC. Its minimum-one floor guarantees local support but not answer correctness. A exceeds B by one semantic correct answer at each budget on this pilot; this does not establish a universal quota disadvantage. Ours/A changes the completion candidate scope; Ours/B is not a quota-only comparison.

At K64, Ours has more strict-correct answers than GC but fewer semantic-correct answers. Answer-format and semantic content effects must therefore be kept separate.

## Historical inference gates

K=N on N832 and N1088: exact merged input embeddings; first-step logits maxdiff0; identical generated token IDs and raw answers; historical Full token IDs equal. The ordinary model and manual insertion therefore matched on these tested cases. Equality is not VQA correctness.

The recorded Ours smoke shortened sequences 879->111/175 and 1143->119/183 for K64/K128. It preserved nonvisual tokens, retained exactly K visual placeholders/features in original order and audited actual prefill length, mask/position/cache and vision execution.

## Efficiency boundaries

Historical five-method wall runtime: 508.6086037000059 seconds; completion errors: zero. Shared affinity/floor/feature timing must not be charged repeatedly across methods or budgets. Audit-hook generation timing is not a measured end-to-end speedup. CUDA metrics describe allocated PyTorch memory, not all device use.

## Artifact limits

Only selected aggregate derivatives are present. Original questions, ground-truth row labels, raw responses, parsed text, generated token IDs, selected/floor/sequence index lists, and original path-dependent configs are not distributed. Readers cannot independently rescore the private raw responses from these aggregates.

[Provenance and reproduction limits](REPRODUCIBILITY.md) | [Spatial follow-up](HETEROGENEITY.md)

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 dongeunk. Original project text only; external material retains its own rights. -->
