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

"""Re-score existing POPE responses offline; never load or run a model.

Preserves source predictions, strict parser/results, question order, and labels.
All semantic decisions are made from raw text before scoring against labels.
The conservative grammar is frozen/hash-recorded for this retrospective audit.
"""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.pope import compute_metrics, parse_yes_no
from src.evaluation.semantic_yes_no import PARSER_VERSION, RULES, explain_semantic_yes_no


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def table_cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def report_markdown(rows, summary):
    strict, semantic = summary["strict_metrics"], summary["semantic_metrics"]
    report_date = datetime.fromisoformat(summary["created_utc"]).astimezone(timezone(timedelta(hours=9))).date().isoformat()
    lines = ["# POPE raw response semantic 재평가", "", f"{report_date}. 기존 raw30개만 사용했으며 모델/이미지를 다시 실행하지 않았다.", "",
             "## 판정 방법과 label 독립성", "",
             "새 parser의 두 API는 raw text 인자 하나만 받는다. Question ID·질문·image·정답 label 입력은 없다. 전체 raw30개를 먼저 파싱한 후에만 label을 scoring에 결합한다. 물체명·qid별 예외도 없다.", "",
             "- 명시적인 Yes/No 및 완결된 단순 시각적 존재/부재 문장을 해석한다.",
             "- `There is no <대상> present/visible in the image`는 대상의 부재를 단정하므로 No다.",
             "- `no mention/evidence`, `cannot determine`는 부재 단정이 아니므로 invalid다. 불확실성/부정 범위/충돌/인용/미완결은 보수적으로 보류한다.",
             "- Raw-only 독립 전수 검토도 Yes13/No15/불확실2로 일치했다. 이는 새로운 모델 추론이 아니다.",
             "- 이 parser는 제한된 영어 문법을 다루는 규칙 기반 해석기이며 범용 자연어 추론 모델이 아니다. 지원하지 않는 명확한 다른 표현도 abstain할 수 있다.",
             "- 응답 형식을 본 뒤 만든 사후 평가 보정이다. 정답에 맞춘 규칙은 없지만 별도 held-out 응답에서 검증한 일반화 결과는 아니다.", "",
             "## Strict vs semantic", "",
             "모든 지표의 평가 대상은 동일한30문항이다. Invalid는 양쪽 모두 accuracy에서 오답이다. Precision/Recall/F1은 Yes-positive이며 GT Yes의 invalid는 recall 분모에 포함된다.", "",
             "| 지표 | 기존 strict | 새 semantic |", "|---|---:|---:|"]
    names = (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
             ("f1", "F1"), ("invalid_ratio", "Invalid Rate"), ("yes_prediction_ratio", "Yes prediction ratio"),
             ("yes_class_accuracy", "Yes class accuracy"), ("no_class_accuracy", "No class accuracy"))
    for key, name in names:
        lines.append(f"| {name} | {strict[key]*100:.2f}% | {semantic[key]*100:.2f}% |")
    lines += ["", "정답 수: strict18 → semantic26. 새로 No로 해석한10개 중8개는 정답No여서 correct가 되고,2개는 정답Yes여서 실제 오답으로 남는다. 판단 불가능2개는 정답이No이더라도 invalid를 유지한다.", "",
              "### Confusion matrices", "", "행=정답; 열=예측. Invalid를 No로 치환하지 않는다.", "",
              "| Parser | 정답 | Yes | No | Invalid |", "|---|---|---:|---:|---:|"]
    for mode, metrics in (("strict", strict), ("semantic", semantic)):
        for truth in ("yes", "no"):
            counts = metrics["confusion_matrix"][truth]
            lines.append(f"| {mode} | {truth} | {counts['yes']} | {counts['no']} | {counts['invalid']} |")
    lines += ["", "F1이92.86%로 그대로인 이유: Yes-positive TP=13과 recall 분모15가 같다. 기존 invalid_yes2개가 semantic에서는 FN2개가 됐고, 새 TN8개는 Yes-positive F1 산식에 들어가지 않는다. Accuracy 상승은 재추론/모델 개선이 아니라 parser의 부정 서술 인식 보정이다.", "",
              "## 기존 invalid12개 전수 판정", "",
              "변경10개, invalid 유지2개다. Raw는 JSON 문자열 표현으로 표시해 선행 공백도 보존한다. GT는 parser 판정 후 correctness 확인에만 사용한다.", "",
              "| Row | QID | Raw response | Semantic | 판정 근거 | GT | Correct |", "|---:|---:|---|---|---|---|---|"]
    for row in rows:
        if row["strict_parsed_answer"] is None:
            lines.append(f"| {row['source_row']} | {row['question_id']} | {table_cell(json.dumps(row['raw_output'], ensure_ascii=False))} | "
                         f"{row['semantic_parsed_answer'] or 'invalid'} | {table_cell(row['semantic_reason'])} | {row['label']} | {row['semantic_correct']} |")
    lines += ["", "## 전체30개 검토 기록", "",
              "| Row | QID | Raw response | Strict | Semantic | Rule |", "|---:|---:|---|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['source_row']} | {row['question_id']} | {table_cell(json.dumps(row['raw_output'], ensure_ascii=False))} | "
                     f"{row['strict_parsed_answer'] or 'invalid'} | {row['semantic_parsed_answer'] or 'invalid'} | {row['semantic_rule_id']} |")
    lines += ["", "## 저장·검증", "",
              "- 원본 predictions/strict summary/strict parser/manifest/기존 연구 알고리즘을 수정하지 않았다. Source와 parser의 SHA256을 summary와 parser_rules에 기록했다.",
              "- `predictions.jsonl`: raw/strict/semantic/correct/변경 여부와 각 문장의 rule·evidence·reason.",
              "- `summary.json`: 두 parser의 동일-denominator metrics/confusion 및 변경 개수.",
              "- `parser_rules.json`: 버전, 보수적 grammar/scope, source hash, 모든 regex 규칙.",
              "- 다른 물체명·부정 범위·불확실성·충돌·미완결·raw-only API를 synthetic unit test로 검증했다. 원본 VLM inference는0회.", "",
              "명확한 부재10개를 No로 복구하는 것은 타당하지만, 이번 결과만으로 parser가 모든 paraphrase를 정확히 처리한다고 주장할 수 없다. 기존 strict 결과를 함께 보존하며 새 값을 원본 benchmark 결과에 덮어쓰지 않았다. 추가 실험은 진행하지 않았다.", ""]
    return "\n".join(lines)


def run(source, output):
    names = ("predictions.jsonl", "summary.json", "parser_rules.json", "report.md")
    if any((output / name).exists() for name in names):
        raise FileExistsError("Existing review outputs are preserved; choose a fresh --output-dir")
    source_summary = source.parent / "summary.json"
    manifest = ROOT / "results/pope_pilot/sample_manifest.jsonl"
    parser_path = ROOT / "src/evaluation/semantic_yes_no.py"
    strict_path = ROOT / "src/evaluation/pope.py"
    protected = {str(path.resolve()): sha256(path) for path in (source, source_summary, manifest, strict_path)}
    records = read_jsonl(source)
    original = read_jsonl(manifest)
    keys = ("question_id", "image_id", "image", "text", "label", "sample_order")
    if len(records) != 30 or len({r["question_id"] for r in records}) != 30:
        raise ValueError("Expected exactly the original30 unique saved responses")
    if [tuple(r[key] for key in keys) for r in records] != [tuple(r[key] for key in keys) for r in original]:
        raise ValueError("Saved responses differ from the original ordered manifest")
    if not all(r.get("generation_success") and isinstance(r.get("raw_output"), str) for r in records):
        raise ValueError("Only completed original raw responses may be rescored")

    # PARSING PHASE: hand only raw strings to the parser; no label/qid/question.
    raw_responses = [r["raw_output"] for r in records]
    decisions = [explain_semantic_yes_no(text) for text in raw_responses]
    strict_check = [parse_yes_no(text) for text in raw_responses]
    if strict_check != [r["parsed_answer"] for r in records]:
        raise ValueError("Stored strict decisions differ from unchanged strict parser")

    # SCORING PHASE: only after every semantic decision is final, join labels.
    rows = []
    for index, (record, decision) in enumerate(zip(records, decisions), 1):
        if record["correct"] != (record["parsed_answer"] == record["label"]):
            raise ValueError("Stored strict correctness does not match its decision")
        rows.append(dict(source_row=index, question_id=record["question_id"], image_id=record["image_id"],
                         image=record["image"], text=record["text"], label=record["label"],
                         raw_output=record["raw_output"], strict_parsed_answer=record["parsed_answer"],
                         strict_correct=record["correct"], semantic_parsed_answer=decision.answer,
                         semantic_category={"yes": "clear_yes", "no": "clear_no", None: "ambiguous"}[decision.answer],
                         semantic_correct=decision.answer == record["label"],
                         decision_changed=decision.answer != record["parsed_answer"],
                         semantic_rule_id=decision.rule_id, semantic_evidence=decision.evidence,
                         semantic_reason=decision.reason))
    strict = compute_metrics(records)
    semantic = compute_metrics([dict(label=row["label"], parsed_answer=row["semantic_parsed_answer"]) for row in rows])
    if strict != json.loads(source_summary.read_text(encoding="utf-8"))["metrics"]:
        raise ValueError("Strict summary cannot be reproduced exactly")
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), parser_version=PARSER_VERSION,
                   source_predictions=str(source.resolve()), source_sha256=protected[str(source.resolve())],
                   parser_sha256=sha256(parser_path), original_files_sha256=protected,
                   model_inference_executed=False, model_imports=False, raw_only_parser_inputs=True,
                   all_decisions_completed_before_label_scoring=True, reviewed_responses=len(rows),
                   strict_metrics=strict, semantic_metrics=semantic,
                   original_invalid_count=sum(row["strict_parsed_answer"] is None for row in rows),
                   changed_decisions=sum(row["decision_changed"] for row in rows),
                   resolved_invalid_count=sum(row["strict_parsed_answer"] is None and row["semantic_parsed_answer"] is not None for row in rows),
                   remaining_invalid_count=sum(row["semantic_parsed_answer"] is None for row in rows),
                   original_valid_decisions_unchanged=all(not row["decision_changed"] for row in rows if row["strict_parsed_answer"] is not None),
                   accuracy_change_pp=(semantic["accuracy"]-strict["accuracy"])*100,
                   invalid_change_pp=(semantic["invalid_ratio"]-strict["invalid_ratio"])*100,
                   limitations=["Retrospective response-format audit; not new model performance",
                                "Restricted English grammar, not a general semantic reasoner",
                                "No held-out response validation or additional experiment"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    # Export exact pattern strings as well as explanatory rules for reproducibility.
    import src.evaluation.semantic_yes_no as parser_module
    patterns = {name: value.pattern for name, value in vars(parser_module).items()
                if name.startswith("_") and hasattr(value, "pattern")}
    save_json(output / "parser_rules.json", dict(RULES, code_sha256=summary["parser_sha256"], regex_patterns=patterns))
    if not all(sha256(Path(path)) == value for path, value in protected.items()):
        raise RuntimeError("An original protected input changed")
    summary["original_inputs_unchanged"] = True
    save_json(output / "summary.json", summary)
    (output / "report.md").write_text(report_markdown(rows, summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=ROOT / "results/pope_pilot/full_model_seed42/predictions.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/pope_pilot/semantic_review")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    run(args.predictions, args.output_dir)


if __name__ == "__main__":
    main()
