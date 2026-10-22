# Code availability and release holds

This selected snapshot keeps unchanged algorithms in the included first-party modules. One report-generation literal and the historical run guide have documented privacy/documentation transformations. No omitted dependency is replaced by a stub.

## Included first-party files

- src/evaluation/controlled_comparison.py
- src/evaluation/pope.py
- src/evaluation/semantic_yes_no.py
- src/inference/generation_audit.py
- src/inference/pruned_vlm.py
- src/pruning/coverage.py
- src/pruning/matched_allocation.py
- src/pruning/spatial_floor_coverage.py
- examples/diagnose_global_coverage.py
- examples/diagnose_matched_random.py
- examples/diagnose_spatial_budgets.py
- examples/download_pope_pilot_images.py
- examples/inspect_llm_input.py
- examples/inspect_visual_tokens.py
- examples/plot_pope_spatial_heterogeneity.py
- examples/README_pope_controlled_comparison.md
- examples/report_pope_spatial_heterogeneity.py
- examples/rescore_pope_semantic.py
- examples/run_global_coverage.py
- examples/run_pope_pilot.py
- examples/run_pope_pruning_smoke.py
- examples/run_random_pruning.py
- examples/run_small_vlm.py
- examples/run_spatial_floor_coverage.py
- examples/test_noop_visual_tokens.py
- tests/test_controlled_comparison.py
- tests/test_coverage.py
- tests/test_coverage_diagnostic.py
- tests/test_generation_audit.py
- tests/test_matched_allocation.py
- tests/test_matched_random_diagnostic.py
- tests/test_pope.py
- tests/test_pope_pilot.py
- tests/test_pruned_vlm.py
- tests/test_semantic_yes_no.py
- tests/test_spatial_budget_diagnostic.py
- tests/test_spatial_floor_coverage.py

## Held first-party files

The eleven held paths have different reasons. The reference module is on rights HOLD. Its parity-test file mixes original tests with upstream fixture/reference material. The other nine paths are original orchestration/analysis/test code held for execution dependencies; this is not a finding of external-code borrowing in each path.

- examples/analyze_pope_spatial_heterogeneity.py
- examples/compare_allocation_strategies.py
- examples/run_pope_controlled_comparison.py
- src/analysis/spatial_heterogeneity.py
- src/inference/controlled_selectors.py
- src/pruning/s2_style_allocation.py
- tests/test_controlled_runner.py
- tests/test_controlled_selectors.py
- tests/test_pope_heterogeneity_analysis.py
- tests/test_s2_style_allocation.py
- tests/test_spatial_heterogeneity.py

## Consequences

Random/GC/Ours selector, direct insertion and evaluator source remain available. matched_allocation accepts externally supplied quotas; the original image-to-Laplacian-quota orchestration is held. Historical five-method and H results are documented but their full runners are absent. Plot/report tools are archival utilities requiring the full historical schema; aggregate exports are not interchangeable inputs.

All third_party trees are excluded. No GridPrune HEAD/source was available in the inspected local clone. The scoped Apache-2.0 grant is documented in [LICENSING](LICENSING.md); it does not cover excluded material. See ETHICS_AND_RELEASE_SCOPE.md for the publication boundary.

## Phase-one classifications

| Held path | Reason |
| --- | --- |
| src/pruning/s2_style_allocation.py | Rights HOLD: code-informed reference implementation |
| tests/test_s2_style_allocation.py | Mixed-provenance HOLD: own tests plus upstream fixture/parity material |
| src/inference/controlled_selectors.py | Dependency HOLD: original selector orchestration |
| src/analysis/spatial_heterogeneity.py | Dependency HOLD: original geometry/H analysis |
| examples/compare_allocation_strategies.py | Dependency HOLD: original allocation experiment runner |
| examples/run_pope_controlled_comparison.py | Dependency HOLD: original five-method runner |
| examples/analyze_pope_spatial_heterogeneity.py | Dependency HOLD: original saved-result H runner |
| tests/test_controlled_selectors.py | Dependency HOLD: original synthetic selector tests |
| tests/test_controlled_runner.py | Dependency HOLD: original runner/approval tests |
| tests/test_spatial_heterogeneity.py | Dependency HOLD: original geometry/complexity tests |
| tests/test_pope_heterogeneity_analysis.py | Dependency HOLD: original H-analysis orchestration tests |

The existing [vlm.py](../vlm.py) is retained as a legacy GPU utility. Attribution-only comments are added to the included Python copies; AST structure is checked against the reviewed candidate. The supplied-quota A/B engine remains available, but the five-method and H-analysis replay paths are absent. A later release requires the relevant rights/dependency review and its own validation; no stub replaces a held method.

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 dongeunk. Original project text only; external material retains its own rights. -->
