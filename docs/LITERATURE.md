# Literature inventory and implementation references

## Scope of this note

This is a record of the evidence available for an independent personal
research portfolio. It separates a local literature inventory from the
sources that informed implementation. It is not a systematic literature
review, a claim that every PDF was fully read, or a reproduction of every
listed method.

The working names below are identifiers used during the study, not verified
bibliographic titles. Authors, publication dates, and venues are omitted when
they have not been established. File names and their date-like suffixes were
not treated as bibliographic evidence. PDFs and personal file paths are not
included in this candidate.

## Literature inventory: 20 entries

"Review evidence not established" means that the records inspected do not
establish the relevant review. It does not mean that the researcher never
read the paper. PDF possession, metadata verification, text review, code
review, implementation use, and direct comparison are separate states.

| Working name | Evidence available in the private study records | Scope and remaining uncertainty |
| --- | --- | --- |
| GridPrune | Local PDF inventory entry confirmed | Text review evidence not established. The local reference checkout had a configured remote but no HEAD, README, or implementation. This is a statement about that local checkout, not the upstream repository. |
| MMTok | Local PDF entry, README identification, and official selector source inspection confirmed | The greedy marginal gain structure informed the retained coverage baseline. Full PDF reading and full MMTok reproduction are not established. |
| QCPruner | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| AgilePruner | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| FastV | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| SparseVLM | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| PyramidDrop | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| VisionZip | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| CROP | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| TrimTokenator | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| When token pruning is worse than random | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| DSTP | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| OccamToken | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| IF_Prune | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| QuietPrune | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| SIEVE | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| PACE | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |
| S2Prune | Local PDF entry, README identification, Appendix A.1-A.7 review record, and official allocation source inspection confirmed | The recorded text review is partial. Allocation and external-score selection work informed the private study; the related own module and dependent path are held outside this candidate pending rights review. |
| CoverPruner | Local PDF inventory entry and a row-centered affinity implementation in the study confirmed | Exact equivalence and attribution to the paper have not been established. It is not promoted to a confirmed implementation reference. |
| StackTok | Local PDF inventory entry confirmed | Formal metadata, text review, code review, implementation use, and direct comparison evidence not established. |

## Confirmed implementation reference: MMTok

Upstream: [Ironieser/MMTok](https://github.com/Ironieser/MMTok)

Inspected local upstream commit:

`8e2686836ed5bbe05d064f1d1c20e50c73c70b94`

Source anchors:

- `mmtok/core/semantic_selector.py:24-65`:
  `greedy_merged_jit_kernel`.
- The same file, lines 108-118: text-vision P, vision-vision Q,
  temperature softmax, and the weighted combination.
- The same file, line 146: final sorting into original token order.
- Own implementation evidence: `src/pruning/coverage.py:18`.

The retained own coverage baseline follows the vectorized greedy update
structure: maintain each demand's current best coverage, calculate all
candidate marginal gains, mask selected candidates with negative infinity,
and sort indices only after selection ends.

The baseline uses LLM-visible projected visual embeddings. In float32:

```text
V = L2Normalize(projected_visual_embeddings)
M = V @ V.T
A = clamp(M - row_mean(M), min=0)
F(S) = sum_i max_{j in S} A[i,j]
gain[j] = sum_i max(A[i,j] - best[i], 0)
```

Every visual token has unit demand weight. There is no text-query weighting.
This is not full MMTok: it does not use MMTok's text-vision P, its temperature
softmax, its alpha combination, or its original feature pipeline. The
row-centered affinity should not be attributed to MMTok solely because the
greedy update structure was referenced.

Publication attribution for the referenced greedy portions is implemented in source comments and [NOTICE](../NOTICE). The grant and exclusions are in [LICENSING](LICENSING.md); this is not comprehensive clearance of external materials.

## Confirmed historical reference: S2Prune

Upstream:
[yuanyuanjia71-spec/S2Prune](https://github.com/yuanyuanjia71-spec/S2Prune)

Inspected local upstream commit:

`b0ac197e5e765f8a151669fdcbce10b621bbf44b`

Source anchors:

- `s2prune/allocation.py:46`: coarse spatial partition.
- `s2prune/allocation.py:71`: regional Laplacian variance.
- `s2prune/allocation.py:111`: minimum-one and capped largest-remainder
  allocation.
- `s2prune/allocation.py:175`: recursive local cells.
- `s2prune/allocation.py:209`: cell representative selection from ERC scores.
- `s2prune/qwen.py:365-367`: early representation change across decoder
  Layer 0.
- `s2prune/qwen.py:403-406`: sequence, attention, positions, and cache gather.

The private study records document Appendix A.1-A.7 review, not complete
paper reading. They also document an own allocation/external-ERC core and
reference parity tests. The historical own path is
`src/pruning/s2_style_allocation.py`; it and its dependent publication path
are not distributed in this candidate while rights review is pending.
All upstream source repositories are also excluded.

The private A/B ablation shared affinity, all visual demand rows, local-only
candidates, a common medoid floor, and a marginal gain scorer. B added final
regional quotas. It did not use recursive cells or measured ERC, so it was
an allocation ablation, not full S2Prune. The all-candidate spatial floor
reference additionally allowed global-view candidates and was not a
single-variable comparison against A/B.

The original reference used a different model, visual layout, and pruning
stage. Its K=64 default used a 5x5 grid, while the private controlled protocol
used 4x4. These adaptations prevent a claim of exact paper reproduction.
Holding code is a conservative release decision, not a legal finding that
the implementation is prohibited.

## Pending source questions

- **CoverPruner:** distinguish the implemented row-centered affinity from
  verified paper provenance. A direct source and equation comparison is
  still needed before assigning a confirmed implementation relationship.
- **GridPrune:** distinguish PDF possession from a usable local source
  checkout. No GridPrune implementation use or direct comparison is
  established by the inspected local records.

## Release boundary

The public candidate contains no upstream source tree, paper PDF, dataset,
original photograph, photo-bearing panel, or raw prediction record.
References here describe research history and component provenance; they
do not imply redistribution permission, a license grant, or publication
approval. See [Ethics and release scope](ETHICS_AND_RELEASE_SCOPE.md).

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 dongeunk. Original project text only; external material retains its own rights. -->
