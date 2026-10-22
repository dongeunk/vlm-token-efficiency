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

"""Standard-library POPE annotation, sampling, parsing, and metric helpers.

The parser deliberately does not reproduce the permissive official heuristic:
only an unquoted leading ``yes``/``no`` followed by end-of-string, whitespace,
or ``.,!?;:`` is accepted, case-insensitively. Explanatory text is allowed, but
any question mark, explicit question/query/prompt marker, echoed existential
question, uncertainty expression listed in ``_UNCERTAINTY_PATTERNS``, or both
standalone answer words makes the answer invalid. Standalone means bounded by
non-word characters (Unicode regex rules); substrings in ``yesterday`` and
``nobody`` are not answers. The parser performs no semantic truth inference.
Raw model outputs must be retained by the caller; these functions never rewrite
them. Invalid answers remain a separate confusion-matrix column.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re


_IMAGE_PATTERN = re.compile(r"COCO_val2014_(\d{12})\.jpg\Z", flags=re.ASCII)
_LEADING_ANSWER = re.compile(r"\A((?a:yes|no))(?=$|[\s.,!?;:])", flags=re.IGNORECASE)
_ANSWER_WORDS = re.compile(r"(?<!\w)((?a:yes|no))(?!\w)", flags=re.IGNORECASE)
_QUESTION_PATTERN = re.compile(
    r"\b(?:question|query|prompt)\s*:|"
    r"\b(?:is|are|was|were|do|does|did|can|could|would|will|has|have)\s+there\b",
    flags=re.IGNORECASE,
)
# These are intentionally conservative lexical exclusions, not model-based
# confidence estimates. Each full expression is applied case-insensitively.
_UNCERTAINTY_PATTERNS = (
    r"\b(?:maybe|perhaps|possibly|probably|likely|unlikely|uncertain|unsure|unclear|unknown|ambiguous)\b",
    r"\b(?:may|might|could|seems?|appears?)\b",
    r"\bnot\s+(?:(?:entirely|completely|fully)\s+)?(?:sure|certain|clear)\b",
    r"\bi\s+(?:think|believe|guess|suspect|assume)\b",
    r"\b(?:do\s+not|don['’]t)\s+know\b",
    r"\b(?:cannot|can['’]t|can\s+not)\s+(?:determine|tell|confirm|identify|see|answer|say)\b",
    r"\bunable\s+to\s+(?:determine|tell|confirm|identify|see|answer)\b",
    r"\b(?:hard|difficult|impossible)\s+to\s+(?:tell|determine|confirm|identify)\b",
    r"\b(?:insufficient|not\s+enough)\s+(?:information|evidence|detail)\b",
    r"\bcannot\s+be\s+(?:determined|confirmed|identified)\b",
    r"\b(?:cannot|can['’]t|can\s+not)\s+be\s+(?:sure|certain)\b",
)
_UNCERTAINTY = tuple(re.compile(pattern, flags=re.IGNORECASE) for pattern in _UNCERTAINTY_PATTERNS)


def _validated_annotation(record, context="annotation"):
    if not isinstance(record, dict):
        raise ValueError(f"{context}: annotation must be a JSON object")
    required = {"question_id", "image", "text", "label"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"{context}: missing fields {sorted(missing)}")
    question_id = record["question_id"]
    if isinstance(question_id, bool) or not isinstance(question_id, int) or question_id < 0:
        raise ValueError(f"{context}: question_id must be a nonnegative integer")
    image = record["image"]
    match = _IMAGE_PATTERN.fullmatch(image) if isinstance(image, str) else None
    if match is None:
        raise ValueError(f"{context}: image must be COCO_val2014_<12 decimal digits>.jpg")
    if not isinstance(record["text"], str) or not record["text"].strip():
        raise ValueError(f"{context}: text must be a nonempty string")
    if record["label"] not in ("yes", "no"):
        raise ValueError(f"{context}: label must be exactly 'yes' or 'no'")
    image_id = int(match.group(1))
    if "image_id" in record:
        supplied = record["image_id"]
        if isinstance(supplied, bool) or not isinstance(supplied, int) or supplied != image_id:
            raise ValueError(f"{context}: image_id disagrees with the image filename")
    result = dict(record)
    result["image_id"] = image_id
    return result


def _validate_records(records):
    validated = []
    question_ids = set()
    for position, record in enumerate(records):
        record = _validated_annotation(record, f"record {position}")
        question_id = record["question_id"]
        if question_id in question_ids:
            raise ValueError(f"Duplicate question_id: {question_id}")
        question_ids.add(question_id)
        validated.append(record)
    return validated


def load_annotations(path):
    """Load official JSONL fields, adding parsed numeric COCO ``image_id``.

    Blank lines are ignored. Original annotation text and additional fields are
    retained. Invalid JSON, invalid fields, duplicate IDs, and empty files raise
    ValueError; filesystem errors propagate. No annotation is silently dropped.
    """
    records = []
    with Path(path).open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
            records.append(_validated_annotation(record, f"{path}:{line_number}"))
    if not records:
        raise ValueError(f"{path}: no annotations found")
    return _validate_records(records)


def sample_balanced(records, per_label=15, seed=42):
    """Select balanced annotations with the maximum feasible image diversity.

    A bipartite matching assigns unique images to yes/no quota slots, maximizing
    the number of different images subject to the label quotas. Any unmatched
    slots are filled with unused annotations of their required label. Canonical
    question/image sorting before seeded random choices makes the result
    independent of input-file order. Each question ID is selected at most once.
    The returned dicts are copies; the input collection is not modified.
    """
    if isinstance(per_label, bool) or not isinstance(per_label, int):
        raise TypeError("per_label must be an integer")
    if per_label < 0:
        raise ValueError("per_label must be nonnegative")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    records = sorted(_validate_records(records), key=lambda record: record["question_id"])
    by_label = {label: [record for record in records if record["label"] == label]
                for label in ("yes", "no")}
    for label in ("yes", "no"):
        if len(by_label[label]) < per_label:
            raise ValueError(f"Need {per_label} '{label}' annotations; found {len(by_label[label])}")
    if per_label == 0:
        return []

    rng = random.Random(seed)
    pools = defaultdict(list)
    for record in records:
        pools[(record["label"], record["image"])].append(record)
    image_candidates = {}
    for label in ("yes", "no"):
        images = sorted({record["image"] for record in by_label[label]})
        rng.shuffle(images)
        image_candidates[label] = images
    slots = [(label, index) for label in ("yes", "no") for index in range(per_label)]
    rng.shuffle(slots)
    matched_images = {}

    def augment(slot, visited):
        for image in image_candidates[slot[0]]:
            if image in visited:
                continue
            visited.add(image)
            previous = matched_images.get(image)
            if previous is None or augment(previous, visited):
                matched_images[image] = slot
                return True
        return False

    for slot in slots:
        augment(slot, set())
    selected = [dict(rng.choice(pools[(slot[0], image)]))
                for image, slot in sorted(matched_images.items())]
    selected_ids = {record["question_id"] for record in selected}
    counts = Counter(record["label"] for record in selected)
    for label in ("yes", "no"):
        available = [record for record in by_label[label] if record["question_id"] not in selected_ids]
        rng.shuffle(available)
        extra = available[:per_label - counts[label]]
        selected.extend(dict(record) for record in extra)
        selected_ids.update(record["question_id"] for record in extra)
    rng.shuffle(selected)
    if len(selected) != 2 * per_label or len(selected_ids) != len(selected):
        raise RuntimeError("Balanced sampler violated annotation-count or uniqueness invariants")
    if Counter(record["label"] for record in selected) != {"yes": per_label, "no": per_label}:
        raise RuntimeError("Balanced sampler violated label quotas")
    return selected


def parse_yes_no(text):
    """Return a conservative leading 'yes'/'no' answer, or None if invalid.

    See the module docstring and uncertainty patterns for the exact lexical
    grammar. Leading quotes, code fences, answer prefixes, question echoes,
    empty/non-string values, and uncertain responses are invalid. The original
    string is neither normalized in place nor returned in altered form.
    """
    if not isinstance(text, str):
        return None
    candidate = text.strip()
    match = _LEADING_ANSWER.match(candidate)
    if match is None:
        return None
    if "?" in candidate or _QUESTION_PATTERN.search(candidate):
        return None
    answers = {word.lower() for word in _ANSWER_WORDS.findall(candidate)}
    if answers != {match.group(1).lower()}:
        return None
    if any(pattern.search(candidate) for pattern in _UNCERTAINTY):
        return None
    return match.group(1).lower()


def compute_metrics(records):
    """Compute all-item metrics without converting invalid answers into 'no'.

    Each record requires label ('yes'/'no') and parsed_answer ('yes'/'no'/None).
    Invalid answers are incorrect for accuracy and class accuracy. Precision is
    TP/(TP+FP); recall is TP/(TP+FN+invalid_yes). Thus invalid positive examples
    count against recall. F1 is the harmonic mean of precision and recall. All
    empty-denominator ratios, including an empty evaluation, are defined as 0.
    Inputs and additional raw-output fields are left untouched.
    """
    confusion = {label: {prediction: 0 for prediction in ("yes", "no", "invalid")}
                 for label in ("yes", "no")}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or "label" not in record or "parsed_answer" not in record:
            raise ValueError(f"record {index}: label and parsed_answer are required")
        label, answer = record["label"], record["parsed_answer"]
        if label not in ("yes", "no") or answer not in ("yes", "no", None):
            raise ValueError(f"record {index}: invalid label or parsed_answer")
        confusion[label]["invalid" if answer is None else answer] += 1
    support = {label: sum(confusion[label].values()) for label in ("yes", "no")}
    n = sum(support.values())
    tp, fn = confusion["yes"]["yes"], confusion["yes"]["no"]
    fp, tn = confusion["no"]["yes"], confusion["no"]["no"]
    invalid_yes, invalid_no = confusion["yes"]["invalid"], confusion["no"]["invalid"]

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn + invalid_yes)
    return dict(
        n=n, accuracy=ratio(tp + tn, n),
        yes_class_accuracy=ratio(tp, support["yes"]), no_class_accuracy=ratio(tn, support["no"]),
        precision=precision, recall=recall, f1=ratio(2 * precision * recall, precision + recall),
        yes_prediction_ratio=ratio(tp + fp, n), invalid_ratio=ratio(invalid_yes + invalid_no, n),
        confusion_matrix=confusion, support=support, tp=tp, fp=fp, fn=fn, tn=tn,
        invalid_yes=invalid_yes, invalid_no=invalid_no,
    )
