# Static publication review

## Scope and methods

This preparation uses standard-library file parsing, token/AST comparison,
hashing and pattern scans. No project module is imported or executed.
No model, selector, generation, unit test suite, installation or experiment ran.

- The inventory contains 72 files, including all 68 reviewed-candidate paths.
- Eleven S2-reference/dependent HOLD paths and all external source trees are absent.
- Included Python parses, and executable AST/tokens are compared with the previous
  candidate; legacy bytes are compared with the existing GitHub/Git blob.
- Internal Python imports and imported names are inspected statically.
- Local Markdown file links/anchors are resolved; external links are inventoried.
- All 15 result files, including three approved photo-free PNGs, retain exact
  reviewed-candidate bytes. JSON/CSV values are compared without rescoring.
- PNG structure, chunk CRCs and textual metadata are checked; source figure
  hashes retain the earlier visual-review lineage. No figure is regenerated.
- Text/structured data is scanned for personal absolute paths, account markers,
  email addresses, recognizable provider credentials, private keys, credential
  URLs, embedded image payloads and forbidden raw-result fields.
- LICENSE equals the official Apache text byte-for-byte. Component scope and
  MMTok notices are checked.
- The publication manifest lists every release file except itself and records
  previous/public hashes separately; its own hash is retained outside the release.
- Original research and permanent archive bytes are checked against a separate
  beginning-of-task snapshot. Existing older-baseline differences are recorded
  privately rather than silently rewriting the archived baseline.

## Rights and privacy boundary

Legacy vlm.py is retained, but assignment of a new license to that file remains HOLD; it is excluded from the new grant.

The owner's original contributions are licensed within
[LICENSING](docs/LICENSING.md); excluded external rights are not granted.
Pattern matching cannot prove the absence of all secrets or settle legal
ownership. Data/model rights remain separate. The private audit contains
machine paths and is outside the publication root.

## What this review does not establish

Static resolution is not a successful runtime import, fresh installation,
passing unit tests, new K=N verification or model benchmark reproduction.
Artifact-dependent checks can skip without original data. Aggregate exports
cannot replace withheld raw responses or full historical schemas.
Only existing approved photo-free figures are retained; structural parsing is
not a new semantic image-classification audit.

Preparation and GitHub publication are authorized. Publication uses a separate
clone, a new commit and a normal push after staged-file verification. Completion
is checked against the remote. New tests/model execution and Notion changes
remain outside this publication work.

<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Copyright 2026 dongeunk. Original project text only; external material retains its own rights. -->
