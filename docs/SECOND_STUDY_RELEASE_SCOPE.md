# Second-study additive release scope and provenance

## Preserved baseline

Base: origin/main at66bbfaf44770b99473fbe94fdde2554f5c94f368. Existing LICENSE, NOTICE, docs/LICENSING.md, docs/CODE_AVAILABILITY.md, all phase-one source/results/figures and their notices are preserved. The root README receives an additive navigation/results prefix; its original body remains an exact contiguous suffix. No existing source file is replaced with the unattributed local research copy.

This addendum describes new public candidates. It does not retroactively change the first-release license scope or create rights in upstream material. Included independently authored CPU evaluation/statistics and wrapper contributions follow the repository's Apache-2.0 original-project-contribution convention. No project license is applied to excluded upstream code or third-party assets; the copyright identifiers and existing license instrument are not rewritten. Factual counts/digests are provenance, not an exclusive claim over dataset facts. Any quotation/bibliographic material retains its source rights.

## Included independent subset

- Five original CPU files under src/smol_s3_eval: __init__.py, contracts.py, lock.py, scoring.py, statistics.py. Only license comments are added to these public copies; AST bodies match the existing local research sources. The exposed evaluation entry point uses count vectors. Broader lock/config helper functions still require their explicit local inputs when invoked.
- A public aggregate-only tools/reproduce_summary_cpu.py. It calls frozen count/metric/bootstrap functions and blocks model and HOLD-helper imports; this is packaging glue, not a change to research selection or inference algorithms.
- Authored S0–S4-A report views, first-study80/320 supplementary aggregate evidence, path-redacted config views, ID/digest-only cohorts and photo-free figures.

Raw responses and original questions/photos/annotations/weights/hidden/KV/affinity tensors are withheld. No original POPE question text or model-cache files are introduced as public dataset downloads. Historical reports describe original local artifacts; their commands/raw-evidence and old publication-preparation statements do not override this addendum. Earlier claims about a5920-response replay refer to an earlier local preparation; the new public helper is aggregate-only.

## Rights and dependency HOLD

The existing eleven path classifications in docs/CODE_AVAILABILITY.md remain authoritative for the first-release HOLD. Ten of those paths occurred in the old201-path proposal; all ten are excluded here (the eleventh was already absent).

- examples/analyze_pope_spatial_heterogeneity.py
- examples/compare_allocation_strategies.py
- examples/run_pope_controlled_comparison.py
- src/analysis/spatial_heterogeneity.py
- src/inference/controlled_selectors.py
- src/pruning/s2_style_allocation.py — rights HOLD
- tests/test_controlled_runner.py
- tests/test_controlled_selectors.py
- tests/test_pope_heterogeneity_analysis.py
- tests/test_spatial_heterogeneity.py

Additional conservative exclusions cover all new S²Prune reference/allocator and dependent Smol mapping/cell/ERC/offline/prefill/runtime/runner/test code outside the reviewed CPU evaluation closure. This is not a finding of external borrowing in every excluded original file. It avoids distributing unresolved provenance or a workflow requiring the private helper. No algorithm is renamed/reimplemented to bypass the HOLD. Full GPU reproducibility remains unavailable from this candidate.

## MMTok comparison and resolution

The three existing remote files src/pruning/coverage.py, src/pruning/spatial_floor_coverage.py and src/pruning/matched_allocation.py are retained byte-for-byte. Their AST equals the corresponding local research body; their existing SPDX and Zoom copyright notices remain intact. No restoration from an uncredited local copy is needed.

Pinned local upstream: Ironieser/MMTok revision8e2686836ed5bbe05d064f1d1c20e50c73c70b94. semantic_selector.py states Copyright(c)2025 Zoom Communications, Inc., Author Sixun Dong, Apache-2.0. Existing NOTICE records that revision/source chain. The two LICENSE files differ in a leading blank line and the Appendix's copyright example: upstream inserts Zoom, whereas the remote license instrument retains placeholders. Operative terms match; Zoom is retained in NOTICE and all three source headers. Existing LICENSE is not rewritten and upstream source is not redistributed. This is a record comparison, not an expansion of redistribution rights.

## Future publication

This is a review worktree, not a commit or push. A later publication must use this remote-based allowlist, retain all baseline files/notices, exclude every HOLD path and recheck the remote tip. Never push the older C worktree's unreviewed201-path candidate directly. New experiments require separate authorization.

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 dongeunk. Original project text; external rights unchanged. -->
