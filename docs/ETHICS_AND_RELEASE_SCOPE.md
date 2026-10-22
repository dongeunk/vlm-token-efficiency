# Ethics and release scope

## Purpose and current status

This is a phase-one research snapshot for an independent personal research
portfolio and graduate-school research contact. It is intended to support
technical discussion of visual-token pruning, experimental controls, and
implementation checks. It does not claim an academic affiliation, a
particular student status, a production system, or paper-level reproduction.

The owner authorized preparation and subsequent GitHub publication of this
snapshot. Publication completion is determined by remote verification; these
contents do not assume a successful push. The publication commit is identified
by repository history and is not a historical experiment-execution commit.

The owner approved Apache-2.0 for original contributions within the explicit
scope in [LICENSING](LICENSING.md). A reference to an external paper or project
is not a grant of rights in that material. Excluded S2Prune code, external
models/data and other rights are not licensed by this working copy.

## Material excluded from this candidate

The preparation policy excludes:

- Every external source repository under `third_party/`.
- Paper PDFs and local literature-library paths.
- Model weights, model caches, checkpoints, and downloaded datasets or
  annotations.
- Original photographs, including the local example images and COCO images.
- Figure panels that embed photographs, even if the corresponding original
  image directory is excluded.
- Raw predictions, per-question response records, generated token IDs,
  question manifests, and original JSON/JSONL/CSV result exports.
- Private environment files, credentials, IDE settings, scratch notebooks,
  local Git metadata, and machine-specific absolute paths.

Only selected aggregate derivatives and anonymous case metrics are eligible for inclusion. "Selected derivative"
is a release boundary, not a guarantee that a file is free of identifying
content or rights restrictions. Figures and text must still be checked for
embedded photographs, local paths, and individual records.

## Rights review hold

The historical own module `src/pruning/s2_style_allocation.py` and its
dependent publication path are held outside this candidate pending rights
review. The original private research files are preserved. The hold does
not conclude that publication is legally prohibited; it records that this
release decision has not been cleared.

The own coverage baseline that references MMTok's greedy update structure
remains candidate source. Its provenance is documented in
[Literature inventory and implementation references](LITERATURE.md).
The referenced greedy portions now carry scoped upstream attribution and change notices in the source and [NOTICE](../NOTICE). This does not certify unrelated rights.

Excluding external code, datasets, and photographs does not resolve every
possible rights question about derived material. Anyone distributing or
using external material remains responsible for checking the applicable
terms and permissions. This document does not certify those permissions.

## Privacy and evidence preservation

The private research artifacts contain machine-specific paths and other
metadata that are unsuitable for direct publication. Public-facing copies
should use repository-relative paths or neutral placeholders. Credential
and personal-path scans have limited coverage and cannot prove absence of
all sensitive information.

The candidate is prepared separately from the original study workspace.
Original artifacts and their recorded hashes are not rewritten to make
public-facing copies. Removing or aggregating information for publication
must not silently change the original experimental evidence.

This preparation does not authorize Git history rewriting or changing
already public commit metadata. Any such action requires a separate
decision. No claim is made here that historic public metadata has been
removed.

## Scientific limits

- The work is exploratory and uses a small, fixed evaluation sample.
  It does not establish general benchmark performance or statistical
  significance.
- A higher representational coverage objective is not evidence by itself
  of a better downstream answer.
- Regional token counts use a nominal local spatial lattice. A zero count
  does not show that all visual information about that area is absent:
  global-view tokens and nonlocal vision features may still carry it.
- Local spatial coordinates are not object masks or verified neural
  receptive fields. Object retention or omission is not inferred from a
  region heatmap alone.
- Changing token budget and region grid together confounds budget,
  granularity, and the strength of a minimum-one floor.
- Fixed rule-based yes/no parsers cover a bounded language subset.
  Invalid responses and supported but incorrect yes/no answers must be
  distinguished. Raw answers are not reinterpreted to fit labels.
- Descriptive correlations and within-budget rank groups do not establish
  a causal effect of structural heterogeneity.
- The private Laplacian-quota ablation is not full S2Prune, and the retained
  query-independent baseline is not full MMTok.

Selected aggregate summaries should retain their sample denominator,
scoring definition, and comparison scope. They must not be presented as
large-scale validation, a leaderboard submission, or evidence that a
referenced paper has been reproduced.

## Reproduction boundary

This candidate intentionally omits material used in the private study.
It therefore cannot, by itself, reproduce all historical experiments,
including held allocation paths. External models, datasets, and source
references must be obtained separately under their applicable terms if a
future researcher chooses to use them.

Preparing these documents does not launch a model, run a selector or
generation, download data, install packages, or execute a new experiment.
Future model execution and additional experiments remain separate actions.
GitHub publication is separately authorized and tracked by repository history.

## Anonymous case metrics

Anonymous case metrics omit original question/image IDs, questions, ground-truth labels, responses, generated token IDs, and visual token indices. The original-to-anonymous ID map stays in the private preparation audit, outside this candidate. Some code and research discussion still refer to historical smoke/failure case IDs; they are not original annotation or response records.

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 dongeunk. Original project text only; external material retains its own rights. -->
