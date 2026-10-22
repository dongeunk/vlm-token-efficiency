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

"""Conservative, auditable English visual Yes/No interpretation, without labels.

Only raw text is accepted: no question IDs, image, question, or ground truth.
This is a rule-based parser for explicit binary answers and simple visual
existential clauses, not an unrestricted natural-language inference model.
Unknown scope, uncertainty, mixed assertions, quotes, and incomplete clauses
abstain. Absence of a mention/evidence is never treated as object absence.

Rules are general across noun phrases; there are no pilot IDs/object lists or
ground-truth exceptions. Callers must score decisions only AFTER parsing all
responses, and must retain the original strict decisions and raw strings.
"""

from dataclasses import dataclass
import re


PARSER_VERSION = "visual_semantic_yes_no_v1"
RULES = {
    "version": PARSER_VERSION,
    "inputs": ["raw response text only"],
    "excluded_inputs": ["ground truth", "question", "question ID", "image"],
    "order": ["reject uncertainty, nonvisual negation scope, quotes/questions, and compound scope",
              "recognize explicit yes/no with a compatible complete visual clause or no explanation",
              "recognize a complete simple visual presence/absence clause",
              "otherwise abstain"],
    "visual_grammar": "There is/are no/a/an/some <noun phrase> [present/visible/shown/depicted/seen] in/within the/this image/picture/photo/photograph",
    "safety": "No mention/evidence/guarantee/absence/lack/doubt and uncertainty are not object absence; polarity conflicts abstain",
    "scope_limit": "Conservative English subset; unsupported clear paraphrases can still abstain. Not a learned semantic model.",
}


@dataclass(frozen=True)
class SemanticDecision:
    answer: str | None
    rule_id: str
    evidence: str
    reason: str


FLAGS = re.IGNORECASE | re.ASCII
_LEADING = re.compile(r"\A(?P<answer>yes|no)(?=$|[\s.,!;:])", FLAGS)
_ANSWER_WORDS = re.compile(r"\b(?:yes|no)\b", FLAGS)
_QUESTION = re.compile(r"\?|\b(?:question|query|prompt)\s*:", FLAGS)
_UNCERTAIN = re.compile(
    r"\b(?:maybe|perhaps|possibly|probably|likely|unlikely|uncertain|unsure|unclear|unknown|ambiguous)\b|"
    r"\b(?:may|might|could|seems?|appears?)\b|"
    r"\bnot\s+(?:(?:entirely|completely|fully)\s+)?(?:sure|certain|clear)\b|"
    r"\bi\s+(?:think|believe|guess|suspect|assume)\b|"
    r"\b(?:cannot|can't|can\s+not)\s+(?:determine|tell|confirm|identify|see|answer|say|decide|rule\s+out|be\s+sure|be\s+certain)\b|"
    r"\bunable\s+to\s+(?:determine|tell|confirm|identify|see|answer|decide)\b|"
    r"\b(?:do\s+not|don't)\s+know\b|"
    r"\b(?:insufficient|not\s+enough)\s+(?:information|evidence|detail)\b|"
    r"\b(?:hard|difficult|impossible)\s+to\s+(?:tell|determine|confirm|identify)\b",
    FLAGS,
)
_NONVISUAL_SCOPE = re.compile(
    r"\bno\s+(?:mention|evidence|indication|information|reference|certainty|proof|doubt|reason|guarantee|absence|lack|way|problem|confidence|chance|wonder)\b|"
    r"\bnot\s+(?:only|necessarily)\b", FLAGS,
)
_COMPOUND = re.compile(r"\b(?:and|or|but|however|although|unless|whether|yet|except|if|because)\b", FLAGS)
_VISUAL = re.compile(
    r"\Athere\s+(?:is|are)\s+(?P<determiner>no|a|an|some)\s+"
    r"(?P<entity>[a-z0-9][a-z0-9'\-]*(?:\s+[a-z0-9][a-z0-9'\-]*){0,19}?)"
    r"(?:\s+(?:present|visible|shown|depicted|seen))?\s+(?:in|within)\s+"
    r"(?:the|this)\s+(?:image|picture|photo|photograph)[.!]*\Z", FLAGS,
)
_ABSTRACT_ENTITY = re.compile(
    r"\b(?:mention|evidence|indication|information|reference|certainty|proof|doubt|reason|guarantee|absence|lack|way|problem|confidence|chance|possibility)\b", FLAGS,
)
_NESTED_NEGATION = re.compile(r"\b(?:no|not|never|none|without)\b", FLAGS)


def explain_semantic_yes_no(text) -> SemanticDecision:
    """Interpret supported raw English only; return a polarity and its rule.

    Internal whitespace/curly-apostrophe normalization is for matching only;
    the input is immutable and never overwritten. Reasons are language rules,
    never claims about the image or correctness against a reference answer.
    """
    if not isinstance(text, str) or not text.strip():
        return SemanticDecision(None, "empty_or_nontext", "", "응답이 비어 있거나 문자열이 아님.")
    candidate = text.strip().replace("\u2019", "'")
    if candidate[0] in "\"'`" or _QUESTION.search(candidate):
        return SemanticDecision(None, "quoted_or_question", candidate, "인용/질문을 실제 답변의 단정으로 해석하지 않음.")
    uncertain = _UNCERTAIN.search(candidate)
    if uncertain:
        return SemanticDecision(None, "uncertain", uncertain.group(), "판단 불가 또는 불확실성을 명시하므로 Yes/No를 강제하지 않음.")
    scope = _NONVISUAL_SCOPE.search(candidate)
    if scope:
        return SemanticDecision(None, "nonvisual_negation_scope", scope.group(), "언급/증거 등의 부재나 다른 부정 범위는 대상의 부재가 아님.")
    compound = _COMPOUND.search(candidate)
    if compound:
        return SemanticDecision(None, "compound_or_ambiguous_scope", compound.group(), "복합 대상/접속절의 범위가 지원 문법 밖이므로 보류함.")

    lead = _LEADING.match(candidate)
    explicit = lead.group("answer").lower() if lead else None
    body = candidate[lead.end():].lstrip(" \t\r\n.,!;:") if lead else candidate
    if lead and not body:
        return SemanticDecision(explicit, "explicit_" + explicit, candidate, "명시적인 Yes/No 단독 응답.")
    if lead and len({word.lower() for word in _ANSWER_WORDS.findall(candidate)}) > 1:
        return SemanticDecision(None, "conflicting_polarity", candidate, "하나의 응답에 Yes와 No가 함께 있어 단정하지 않음.")
    match = _VISUAL.fullmatch(body)
    if match is None:
        return SemanticDecision(None, "unsupported_or_incomplete", candidate, "완결된 지원 시각 문장으로 확인할 수 없어 보류함.")
    if _ABSTRACT_ENTITY.search(match.group("entity")):
        return SemanticDecision(None, "nonvisual_entity", match.group("entity"), "추상 대상의 부재를 이미지 속 물체의 부재로 해석하지 않음.")
    if _NESTED_NEGATION.search(match.group("entity")):
        return SemanticDecision(None, "nested_negation_scope", match.group("entity"), "대상 구문 내부의 추가 부정 범위를 임의로 해석하지 않음.")
    answer = "no" if match.group("determiner").lower() == "no" else "yes"
    if explicit is not None and explicit != answer:
        return SemanticDecision(None, "conflicting_polarity", candidate, "명시적 답과 뒤의 존재/부재 진술이 충돌함.")
    rule = "visual_absence" if answer == "no" else "visual_presence"
    reason = ("이미지 속 대상의 부재를 직접 단정하는 완결 문장."
              if answer == "no" else "이미지 속 대상의 존재를 직접 단정하는 완결 문장.")
    if explicit:
        reason = "명시적 " + explicit.upper() + "와 뒤의 시각 진술이 일치함. " + reason
    return SemanticDecision(answer, rule, body, reason)


def parse_semantic_yes_no(text) -> str | None:
    """Return yes/no/None using raw text alone; no reference answer is accepted."""
    return explain_semantic_yes_no(text).answer
