# Structural heterogeneity: exploratory follow-up

Publication status: the historical numerical analysis and photo-free figures below are retained unchanged. The complete image-based H-analysis runner is withheld; see [CODE_AVAILABILITY](CODE_AVAILABILITY.md).

## Question and declared analysis choices

Does image structural heterogeneity explain when a spatial floor helps or hurts Pure GC? This follow-up reads the completed experiment; it does not run a model or selector.

For regional Laplacian-variance complexities C, the primary diagnostic is:

```text
H = std(C, ddof=0) / (mean(C) + 1e-8)
```

The primary input is original-source-image grayscale/full-image Laplacian with the actual local-region layouts (K64: 4x4, K128: 8x8). Raw regional complexity is primary. Normalized complexity, 672-bicubic resizing, and paper-style logical-grid variants are sensitivity analyses. They are not silently substituted for the primary metric.

The S2Prune paper's H notation/preprocessing does not fully fix every implementation choice in the retained analysis. The raw/normalized choice, epsilon, population std and grouping are declared diagnostic choices. This is not a reproduction of the paper's official heterogeneity figure. Its reference grid for K64 is 5x5, whereas this experiment uses 4x4; K128 uses 8x8.

Separately within each K, rank (H_raw, image ID, question ID) and assign positions 0..9 Low, 10..19 Medium, 20..29 High. Correctness/labels are not grouping inputs. Tie-breaking can split identical H values.

## Structural support versus answer quality

| Statistic | K64 / G16 | K128 / G64 |
| --- | ---: | ---: |
| Images where Pure GC has an empty local region | 5/30 | 30/30 |
| Mean Pure GC empty regions | 0.2 | 9.8 |
| Mean Ours empty regions | 0.0 | 0.0 |
| Ours minus Pure GC semantic correct | -3 | 0 |
| Spearman(H, Ours correctness minus GC correctness) | 0.18613852577202558 | 0.0 |
| Mean Ours/GC overlap fraction of K | 0.6197916666666666 | 0.47734375 |
| Mean selected global tokens: Ours | 3.466666666666667 | 9.833333333333334 |
| Mean selected global tokens: Pure GC | 4.966666666666667 | 18.133333333333333 |

At K64, GC already has complete region support on 25/30 cases. The three losses of Ours all occur in this stratum; on the five remaining cases both are correct. This is compatible with an unnecessary or disruptive floor in some cases, but does not establish causation. Floor also changes token identity, density and local/global budget.

At K128, every GC selection has at least one empty region and Ours fills all regions, yet their total semantic correct counts tie. Support improvement alone does not explain accuracy. Two Ours-only wins and two GC-only wins cancel; one case is wrong for both.

The two budgets also change granularity and floor share from G/K=25% to 50%. This confounds a budget-only explanation.

## Group-level semantic correct counts

| K | H group | Random | Pure GC | Ours | A | B |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 64 | Low | 7/10 | 8/10 | 6/10 | 8/10 | 8/10 |
| 64 | Medium | 10/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| 64 | High | 8/10 | 10/10 | 9/10 | 10/10 | 9/10 |
| 128 | Low | 10/10 | 9/10 | 9/10 | 7/10 | 6/10 |
| 128 | Medium | 9/10 | 8/10 | 8/10 | 9/10 | 9/10 |
| 128 | High | 9/10 | 10/10 | 10/10 | 10/10 | 10/10 |

Each entry is correct / 10. These small groups support descriptive inspection only. The data do not establish a monotonic High-H advantage for Ours.

## Priority failure cases: careful interpretation

Historical case IDs are references to private evidence, not redistributed questions or labels.

- q249, q527 and q388 are the K64 Ours-only losses relative to GC. All three occur with zero GC empty regions. The saved analysis reports no valid-Yes/No discordance for this K64 pair; invalid-response transitions account for the disagreements.
- q1603 changes from both wrong at K64 to Ours correct / GC wrong at K128. K=N equality on this image did not make the Full answer correct.
- q2997 and q1799 are A-only wins over B at K64. This is a controlled quota comparison, not proof that local complexity caused the answer difference.

Raw responses, labels and photograph panels are withheld, so readers cannot independently adjudicate the wording from this candidate alone. No token is claimed to cover or miss a particular object without additional object-coordinate evidence.

## Coordinate validity

Processor metadata and runtime audits establish dynamic crop rows/cols, row-major local views, final global view and token-grid counts. Local-token region membership was recomputed against stored counts in the historical analysis.

Nominal crop-grid extents are not exact image resampling coordinates or neural receptive fields. Original-image overlays and object-retention claims are not provided.

## Files and next questions

Retained exports include group_comparison.csv, paired_group_comparison.csv, sensitivity.csv, aggregate_summary.json, per_image_metrics.csv and three photo-free aggregate figures under results/pope_spatial_heterogeneity_30/. They are whitelisted derivatives, not the full historical analysis schema.

Proposed only: cross K and G independently; match local/global budgets for downstream evaluation; prespecify GC-complete versus GC-gap strata; use more images/seeds; verify object-region relationships. No new algorithm or experiment was implemented for this publication preparation.

## Public numeric-case export

The per-image CSV uses anonymous case IDs that stay consistent across K64/K128. Original question/image IDs and ground-truth row labels are not included in the result files. Paired membership lists are reduced to counts; the private audit retains the ID mapping. Historical case IDs mentioned in this discussion are references, not redistributed annotation or response rows.

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 dongeunk. Original project text only; external material retains its own rights. -->
