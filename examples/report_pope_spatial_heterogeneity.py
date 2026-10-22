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

"""Write a Korean report from saved, CPU-only POPE spatial diagnostics.

Only the supplied fresh report path is written. No images, annotations, source
results, model, selector or statistical grouping is changed or executed here.
Numerical findings are read from the analysis or aggregated from its rows;
the prespecified photographic notes are qualitative observations only.
"""

from __future__ import annotations

from collections import Counter
import html
import json
import os
from pathlib import Path
from urllib.parse import quote


METHOD_LABELS = {
    "random": "Random",
    "pure_gc": "Pure GC",
    "ours": "Ours (spatial floor + GC)",
    "a_local_only": "A (local-only floor + GC)",
    "b_laplacian_quota": "B (Laplacian quota + GC)",
}
PAIR_LABELS = {
    "ours_vs_pure_gc": "Ours − Pure GC",
    "a_vs_b": "A − B",
    "ours_vs_a": "Ours − A",
}
CATEGORY_LABELS = {
    "left_only_correct": "왼쪽만 정답",
    "right_only_correct": "오른쪽만 정답",
    "both_correct": "양쪽 정답",
    "both_incorrect": "양쪽 오답 또는 invalid",
}
GROUPS = ("Low", "Medium", "High")
FAILURE_QIDS = (249, 527, 388, 1603, 2997, 1799)
PHOTO_NOTES = {
    249: "큰 도심 시계·건물과 화면 아래의 작은 차량들이 함께 보인다. 객체 bounding box가 없으므로 차량 중 어느 것이 질문의 truck인지 단정하지 않는다.",
    527: "접시 위 식사 장면이며 여러 음식의 형태와 질감이 섞여 있다.",
    388: "주방의 냉장고·수납장 등 큰 평면과 경계가 눈에 띈다. 큰 구조가 있다는 관찰과 질문 대상의 존재 여부는 구별해야 한다.",
    1603: "도심의 사람들과 화면 왼쪽 아래의 작은 새가 함께 보인다.",
    2997: "거실의 큰 가구와 작은 탁상 물체들이 함께 보인다.",
    1799: "사람들 주변에 여러 원반이 분산되어 비교적 크게 보인다.",
}
VARIANT_LABELS = {
    "normalized_current": "현재 grid / 원본 픽셀 / normalized H",
    "resized672_raw": "현재 grid / 672 bicubic / raw H",
    "resized672_normalized": "현재 grid / 672 bicubic / normalized H",
    "paper_grid_original_raw": "논문 기본 grid·24×24 lattice / 원본 / raw H",
    "paper_grid_original_normalized": "논문 기본 grid·24×24 lattice / 원본 / normalized H",
}


def _cell(value):
    """Escape display text without changing the source raw response."""
    text = str(value)
    return html.escape(text, quote=False).replace("|", "&#124;").replace("\r", "&#13;").replace("\n", "<br>")


def _raw(value):
    # JSON notation makes leading/trailing whitespace and embedded newlines
    # visible; the source value itself is never stripped or overwritten.
    return _cell(json.dumps(value, ensure_ascii=False))


def _number(value, digits=4):
    return "정의 불가" if value is None else f"{value:.{digits}f}"


def _percent(value):
    return "정의 불가" if value is None else f"{100 * value:.2f}%"


def _average(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _ids(values):
    return ", ".join(str(value) for value in values) or "없음"


def _table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _decision(record):
    parsed = record["semantic_parsed_answer"]
    if parsed is None:
        return "invalid (semantic v1 미해결)"
    return parsed + (" / 정답" if record["semantic_correct"] else " / 유효 yes/no 오답")


def _relative_url(source, report_path):
    relative = os.path.relpath(Path(source).resolve(), report_path.parent.resolve())
    return quote(Path(relative).as_posix(), safe="/.:~_-()")


def _check_schema(analysis):
    for field in ("metadata", "method_order", "rows", "groups", "pair_groups", "summaries", "sensitivity"):
        if field not in analysis:
            raise ValueError(f"Analysis is missing {field}")
    rows = analysis["rows"]
    if not rows:
        raise ValueError("Report requires nonempty saved analysis rows")
    if len({(row["question_id"], row["k"]) for row in rows}) != len(rows):
        raise ValueError("Analysis contains duplicate question/budget rows")
    if set(analysis["method_order"]) != set(METHOD_LABELS) or len(analysis["method_order"]) != len(METHOD_LABELS):
        raise ValueError("Report requires the five saved comparison methods")
    budgets = sorted({row["k"] for row in rows})
    for qid in FAILURE_QIDS:
        if any(not any(row["question_id"] == qid and row["k"] == k for row in rows) for k in budgets):
            raise ValueError(f"Missing prespecified failure-case row for question {qid}")
    return rows, budgets


def _source_section(analysis, rows):
    metadata, plan = analysis["metadata"], analysis["metadata"]["plan"]
    first = rows[0]["metrics"]
    epsilon = plan.get("epsilon_H", first.get("heterogeneity_epsilon"))
    ddof = plan.get("regional_std_ddof", first.get("ddof"))
    return "\n\n".join([
        "## 1. 정의, 전처리와 분석 범위",
        "S²Prune은 regional structural complexity의 변동계수 형태인 "
        "`H = Std({c_g}) / (Mean({c_g}) + ε)`로 structural heterogeneity를 정의한다. "
        "본문의 raw variance와 min-max 정규화 점수 표기, 부록의 표기를 함께 보면 H에 어느 버전을 "
        "넣었는지 명확하지 않다. 공개 저장소에는 논문 Figure 5의 H 산출·grouping 코드가 확인되지 않았다. "
        "논문은 H의 std ddof, epsilon 값, Low/Medium/High 경계와 각 그룹 표본 수도 명시하지 않는다. "
        f"[논문 §3.4·§4.1·부록 A]({metadata['paper_url']}), [공식 저장소]({metadata['repo_url']}).",
        f"이번 진단은 **raw regional Laplacian variance를 primary**, min-max normalized H를 secondary로 "
        f"선언했다. 지역 값의 평균과 표준편차는 FP64에서 계산하고 population std `ddof={ddof}`, "
        f"`ε_H={epsilon}`을 쓴다. 이 수치는 이번 진단의 명시적 선택이며 공식 H 구현이라고 주장하지 않는다. "
        "공식 allocator의 min-max 분모 보호용 FP32 epsilon과 H 분모의 epsilon도 구분한다.",
        "공식 reference의 Laplacian allocation은 **resize 전 원본 RGB 이미지**를 사용한다. "
        "원본을 grayscale로 바꾸고 전체 이미지에 4-neighbor Laplacian을 한 번 계산한 다음 "
        "지역을 잘라 variance를 구한다. 바깥쪽은 replicate padding이며 지역 경계마다 별도로 필터링하지 않는다. "
        "공식 NumPy variance는 기본 ddof=0이다. 모델 입력의 672×672 bicubic resize는 별도 경로이다. "
        "따라서 여기의 672 재계산은 **model-input rescale 민감도 분석**이며 원본 reference allocator의 대체 정의가 아니다. "
        "[allocation.py](https://github.com/yuanyuanjia71-spec/S2Prune/blob/b0ac197e5e765f8a151669fdcbce10b621bbf44b/s2prune/allocation.py#L83-L107), "
        "[qwen.py](https://github.com/yuanyuanjia71-spec/S2Prune/blob/b0ac197e5e765f8a151669fdcbce10b621bbf44b/s2prune/qwen.py#L245-L262).",
        "공식 논문의 Qwen 설정은 24×24 post-merge lattice에서 K=64에 5×5(G=25), K=128에 8×8(G=64)를 "
        "기본값으로 사용한다. 현재 저장된 SmolVLM 실험은 K=64에 4×4(G=16), K=128에 8×8(G=64)이므로 "
        "공식 설정을 그대로 재현한 실험은 아니다. 논문 기본 grid·24×24 lattice의 원본-pixel H도 secondary로 기록한다. "
        "논문 Figure 5는 Qwen2.5-VL-7B의 K=32/64에서 Equal/Adaptive density를 비교하고 "
        "MMBench EN·TextVQA·POPE를 평균한 분석이다. 현재 POPE 30개와 모델·비교법·예산·grouping이 다르다. "
        "**Ours 대 Pure GC는 Figure 5의 Equal Density 대 Adaptive Density 재현이 아니다.** "
        "지표와 층화 관점만 참고했으며 architecture, pruning stage와 selector가 다르다.",
        "K별로 `(H_raw, image_id, question_id)` 오름차순 rank를 세 등분한다. 그룹 이름은 Low, Medium, High이며 "
        "동일한 H도 ID tie-break 때문에 서로 다른 그룹에 들어갈 수 있다. 정답 label과 생성 결과는 그룹을 정할 때 "
        "사용하지 않았다. 이 tertile 규칙은 논문의 미공개 경계를 복원한 것이 아닌 사전 선언한 진단 규칙이다.",
        "원본 JPEG·annotation·질문·저장 raw 답변·선택 index·selector·pruning을 바꾸지 않고 저장 결과만 읽었다. "
        "모델 load, visual embedding 재추출, selector 실행, generation, CUDA 계산은 없다. "
        "저장 processor의 실제 crop rows/cols와 token grid로 local lattice를 복원하고 마지막 global view는 "
        "모든 지역 count에서 제외했다. local view의 row-major 순서와 마지막 global view라는 해석은 "
        "저장된 runtime metadata, 당시와 현재 설치된 processor·connector source SHA 일치 확인에 근거한다. "
        f"저장된 {sum(len(row['methods']) for row in rows)}개 method 설정의 region count를 독립적인 정수 좌표로 "
        "재구성해 원래 count와 일치하는지도 검증했다. 사진과 지역 heatmap은 독립 좌표에 놓았다. 이 대응은 nominal lattice와 "
        "정규화 좌표의 proxy이며 receptive field나 객체 mask가 아니다. **선택 token의 특정 객체 포함 여부를 판정하지 않는다.** "
        "local zero-region은 그 사진 영역의 시각 정보가 완전히 사라졌다는 뜻도 아니다. global-view token과 "
        "vision encoder self-attention에 의한 비국소 정보가 남을 수 있다. 이 heatmap은 local nominal grid의 선택 수만 센다.",
    ])


def _aggregate_section(analysis, rows, budgets, methods):
    lines = ["## 2. 전체 성능과 spatial allocation"]
    aggregate = []
    for k in budgets:
        subset = [row for row in rows if row["k"] == k]
        full = sum(row["full_semantic_correct"] for row in subset)
        for method in methods:
            saved = [row["methods"][method] for row in subset]
            semantic = sum(record["semantic_correct"] for record in saved)
            aggregate.append((k, METHOD_LABELS[method], f"{semantic}/{len(subset)}", semantic - full,
                              sum(record["strict_correct"] for record in saved),
                              sum(record["semantic_invalid"] for record in saved),
                              sum(record["strict_invalid"] for record in saved),
                              _number(_average(record["zero_regions"] for record in saved), 2),
                              _number(_average(record["selected_global"] for record in saved), 2)))
    lines.append(_table(("K", "방법", "semantic 정답/n", "Full 대비 정답 Δ", "strict 정답", "semantic invalid", "strict invalid", "평균 zero", "평균 global tokens"), aggregate))
    lines.append("semantic v1 미해결 응답은 invalid로 남기고 오답 처리했다. strict invalid 중 semantic parser가 "
                 "명확한 yes/no로 해석한 응답은 형식 문제이며 semantic invalid와 같지 않다. 두 판정의 raw 답변은 동일하다. "
                 "valid yes/no 오답과 semantic invalid는 아래에서도 분리한다. 모든 Δ는 표에 적힌 왼쪽−오른쪽 정답 수이다.")
    summary_rows = []
    for summary in analysis["summaries"]:
        subset = [row for row in rows if row["k"] == summary["k"]]
        summary_rows.append((summary["k"], f"{summary['side']}×{summary['side']}", summary["n"],
                             summary["gc_zero_image_count"], _number(summary["mean_gc_zero"], 3),
                             _number(summary["mean_ours_zero"], 3), _percent(summary["mean_overlap"]),
                             _number(_average(row["overlap_jaccard"] for row in subset)),
                             _number(_average(row["newly_covered_regions"] for row in subset), 3),
                             _number(_average(row["floor_new_tokens_vs_gc"] for row in subset), 3),
                             _number(summary["mean_gc_global"], 3), _number(summary["mean_ours_global"], 3)))
    lines.append(_table(("K", "grid", "n", "GC zero>0 이미지", "GC zero 평균", "Ours zero 평균", "공통 index/K", "Jaccard", "새로 덮은 region 평균", "GC 밖 floor token 평균", "GC global 평균", "Ours global 평균"), summary_rows))
    lines.append("공통 index/K는 Ours와 Pure GC의 실제 visual-index 교집합 비율이다. newly covered region은 "
                 "GC count=0이고 Ours count>0인 coarse local region 수이며 representational coverage F(S)가 아니다.")
    strata_rows = []
    for summary in analysis["summaries"]:
        for stratum in summary["gc_coverage_strata"]:
            strata_rows.append((summary["k"], "GC zero=0" if stratum["gc_full_region_coverage"] else "GC zero>0",
                                stratum["n"], stratum["left_correct"], stratum["right_correct"],
                                stratum["left_only_correct"], stratum["right_only_correct"],
                                stratum["both_incorrect"], stratum["net_left_minus_right"],
                                stratum["left_invalid"], stratum["right_invalid"]))
    lines.append("### Pure GC가 이미 모든 region을 덮은 경우\n\n" + _table(
        ("K", "층", "n", "Ours 정답", "GC 정답", "Ours만", "GC만", "양쪽 오답/invalid", "Δ", "Ours invalid", "GC invalid"), strata_rows))
    lines.append("GC zero=0에서는 binary coarse-region support를 새로 늘릴 여지가 없다. 그러나 floor의 medoid "
                 "선택과 그 뒤의 greedy 경로가 token identity·지역 내 밀도·global budget을 바꿀 수 있으므로 두 입력은 동치가 아니다.")
    return "\n\n".join(lines)


def _group_section(analysis):
    lines = ["## 3. H rank 그룹 성능과 paired 비교"]
    lines.append(_table(("K", "group", "방법", "정답/n", "strict 정답", "semantic invalid", "strict invalid", "Full 정답", "GT yes/no", "H min–max", "H 평균", "zero 평균", "global 평균"), [
        (entry["k"], entry["group"], METHOD_LABELS[entry["method"]], f"{entry['semantic_correct']}/{entry['n']}",
         entry["strict_correct"], entry["semantic_invalid"], entry["strict_invalid"], entry["full_semantic_correct"],
         f"{entry['gt_yes']}/{entry['gt_no']}", f"{_number(entry['H_min'])}–{_number(entry['H_max'])}",
         _number(entry["H_mean"]), _number(entry["mean_zero_regions"], 2), _number(entry["mean_selected_global"], 2))
        for entry in analysis["groups"]]))
    lines.append(_table(("K", "group", "왼쪽−오른쪽", "n", "왼쪽만 정답", "오른쪽만 정답", "양쪽 정답", "양쪽 오답/invalid", "순이득", "invalid 왼/오", "유효 yes/no discordance", "Pearson H–Δ", "Spearman H–Δ"), [
        (entry["k"], entry["group"], PAIR_LABELS[entry["pair"]], entry["n"], entry["left_only_correct"],
         entry["right_only_correct"], entry["both_correct"], entry["both_incorrect"], entry["net_left_minus_right"],
         f"{entry['left_invalid']}/{entry['right_invalid']}", entry["valid_yes_no_discordances"],
         _number(entry["pearson_H_delta"]), _number(entry["spearman_H_delta"]))
        for entry in analysis["pair_groups"]]))
    lines.append("각 그룹의 n이 작고 정답 차이 Δ는 −1/0/+1의 이산값이다. correlation이 정의 불가이면 "
                 "표본 수 부족 또는 상수 벡터 때문에 계산할 수 없다는 뜻이다. 유의성 검정·인과 추론은 하지 않았다. "
                 "GT yes/no 구성과 Full 정답도 함께 보여 그룹 난이도·질문 구성의 차이를 숨기지 않는다.")
    return "\n\n".join(lines)


def _categories_section(analysis, rows):
    lines = ["## 4. 모든 paired outcome의 QID와 raw 응답",
             "아래 분류의 correct는 semantic v1 기준이다. `양쪽 오답/invalid`에는 유효 yes/no 오답과 "
             "미해결 응답이 함께 있을 수 있으므로 각 응답의 판정·invalid 쪽을 별도 표시한다. "
             "raw는 JSON 문자열 표기로 개행과 양끝 공백을 드러내며 원본 문자열을 정규화하거나 수정하지 않는다."]
    for summary in analysis["summaries"]:
        for pair, stats in summary["pairs"].items():
            left, right = stats["left_method"], stats["right_method"]
            lines.append(f"### K={summary['k']} · {PAIR_LABELS[pair]}")
            for category, label in CATEGORY_LABELS.items():
                qids = stats[category + "_qids"]
                matching = [row for row in rows if row["k"] == summary["k"] and row["question_id"] in qids]
                lines.append(f"<details>\n<summary>{label}: {len(qids)}개 · QID {_ids(qids)}</summary>\n")
                if matching:
                    lines.append(_table(("QID", "GT", "H / group", "왼쪽 판정", "왼쪽 strict invalid", "왼쪽 raw", "오른쪽 판정", "오른쪽 strict invalid", "오른쪽 raw", "semantic invalid 쪽"), [
                        (row["question_id"], row["label"], f"{_number(row['metrics']['heterogeneity_raw'])} / {row['group']}",
                         _decision(row["methods"][left]), row["methods"][left]["strict_invalid"], _raw(row["methods"][left]["raw_output"]),
                         _decision(row["methods"][right]), row["methods"][right]["strict_invalid"], _raw(row["methods"][right]["raw_output"]),
                         row["pairwise"][pair]["invalid_status"]) for row in matching]))
                else:
                    lines.append("해당 설정 없음.")
                lines.append("</details>")
    return "\n\n".join(lines)


def _failure_section(analysis, rows, budgets, methods, report_path):
    lines = ["## 5. 사전 지정 사례의 원본 사진·heatmap·전체 raw 응답",
             "사례 선택은 아래 QID 목록을 고정했다. 사진의 정성적 관찰은 전체 표본에 대한 공통 원인 검증이 아니다. "
             "객체 bounding box·segmentation·token receptive field 증거 없이 객체 보존·누락을 판정하지 않는다."]
    indexed = {(row["question_id"], row["k"]): row for row in rows}
    for qid in FAILURE_QIDS:
        first = indexed[qid, budgets[0]]
        photo = _relative_url(first["image_path"], report_path)
        lines.extend([f"### QID {qid} · {_cell(first['image'])}",
                      f"[원본 사진]({photo}) · image_id={first['image_id']} · GT={first['label']}\n\n"
                      f"질문: {_raw(first['question'])}\n\n사진 관찰: {PHOTO_NOTES[qid]}",
                      f"Full raw: {_raw(first['full_raw_output'])} · semantic {'정답' if first['full_semantic_correct'] else '오답/invalid'}"])
        for k in budgets:
            row = indexed[qid, k]
            metric = row["metrics"]
            grid, token = row["region_grid"], row["token_grid"]
            panel = f"figures/panels/qid_{qid}_k_{k}.png"
            lines.extend([f"#### K={k} · {grid[0]}×{grid[1]} · {row['group']}",
                          f"[원본 사진과 complexity·선택-count heatmap]({panel})\n\n"
                          f"원본 {metric['image_size'][0]}×{metric['image_size'][1]} px; "
                          f"crop {row['crop_rows']}×{row['crop_cols']}; token grid {token[0]}×{token[1]}; "
                          f"N={row['n']}; H_raw={_number(metric['heterogeneity_raw'], 6)}; "
                          f"H_normalized={_number(metric['heterogeneity_normalized'], 6)}; "
                          f"region μ={_number(metric['mean'])}, σ={_number(metric['std'])}; "
                          f"whole-image Laplacian variance={_number(metric['whole_image_laplacian_variance'])}.\n\n"
                          f"Ours/GC 공통 index={row['overlap_count']}/{k} ({_percent(row['overlap_fraction_of_k'])}); "
                          f"새로 덮은 region={row['newly_covered_regions']}; GC에 없던 floor token={row['floor_new_tokens_vs_gc']}.",
                          _table(("방법", "local/global", "zero regions", "zero 위치 (0-based)", "semantic 판정", "strict invalid", "raw 응답", "semantic rule"), [
                              (METHOD_LABELS[method], f"{row['methods'][method]['selected_local']}/{row['methods'][method]['selected_global']}",
                               row["methods"][method]["zero_regions"], _cell(json.dumps(row["methods"][method]["zero_region_positions"])),
                               _decision(row["methods"][method]), row["methods"][method]["strict_invalid"],
                               _raw(row["methods"][method]["raw_output"]), _cell(row["methods"][method]["semantic_rule_id"]))
                              for method in methods])])
    return "\n\n".join(lines)


def _sensitivity_section(analysis):
    lines = ["## 6. 정의와 해상도·grid 민감도",
             "아래 secondary 정의를 모두 보고하며 성능에 유리한 H나 그룹을 골라 primary를 바꾸지 않았다. "
             "원본-pixel current-grid raw H와의 rank 상관, group 이동 수, 각 secondary tertile의 "
             "Ours−GC 순이득을 함께 제시한다. 672 resize와 논문 grid 분석은 complexity 재계산만 하며 선택 결과는 고정되어 있다."]
    lines.append(_table(("K", "secondary 정의", "primary와 rank ρ", "primary group 이동", "H–Δ Spearman", "Low 정답 왼/오 (Δ)", "Medium 정답 왼/오 (Δ)", "High 정답 왼/오 (Δ)"), [
        (entry["k"], VARIANT_LABELS.get(entry["variant"], entry["variant"]),
         _number(entry["rank_correlation_with_primary"]), entry["changed_primary_group"],
         _number(entry["spearman_H_delta"]),
         *[f"{group['left_correct']}/{group['right_correct']} ({group['net_left_minus_right']:+d})" for group in entry["groups"]])
        for entry in analysis["sensitivity"]]))
    return "\n\n".join(lines)


def _questions_section(analysis, rows, budgets):
    lines = ["## 7. 다섯 질문에 대한 보수적 답변"]
    lines.append("### 1) 실패 사례에 공통된 이미지 속성이 있는가?\n\n"
                 "작은 물체와 큰 배경의 혼재, 복합 음식 질감, 큰 평면과 경계 등 서로 다른 장면이 포함된다. "
                 "지정 사례의 H·그룹·raw 응답은 위에 모두 제시했다. 일부 장면의 작은 객체라는 관찰만으로 "
                 "공통 실패 원인이나 token의 객체 누락을 확정할 수 없다. K별 표본 수와 −1/0/+1의 이산 성능 차이도 고려해야 한다.")
    property_rows = []
    for k in budgets:
        subset = [row for row in rows if row["k"] == k]
        outcomes = Counter(row["pairwise"]["ours_vs_pure_gc"]["delta"] for row in subset)
        property_rows.append((k, len(subset), outcomes[-1], outcomes[0], outcomes[1],
                              _number(min(row["metrics"]["heterogeneity_raw"] for row in subset)),
                              _number(max(row["metrics"]["heterogeneity_raw"] for row in subset))))
    lines.append(_table(("K", "n", "Ours−GC=-1", "=0", "=+1", "H 최소", "H 최대"), property_rows))
    for k in budgets:
        losses = [row for row in rows if row["k"] == k
                  and row["pairwise"]["ours_vs_pure_gc"]["category"] == "right_only_correct"]
        invalid_losses = [row for row in losses if row["methods"]["ours"]["semantic_invalid"]]
        if not invalid_losses:
            continue
        lines.append(f"K={k}에서 GC만 정답인 {len(losses)}개 중 {len(invalid_losses)}개는 Ours가 frozen "
                     "semantic-v1 parser에서 invalid로 남은 경우다. 그러나 invalid는 동일한 언어 현상이나 동일한 "
                     "시각 실패를 뜻하지 않는다. 아래 raw와 고정 rule을 함께 읽어야 한다.")
        invalid_rows = []
        for row in invalid_losses:
            record = row["methods"]["ours"]
            rule = record["semantic_rule_id"]
            if row["question_id"] == 527 and rule == "compound_or_ambiguous_scope":
                interpretation = ("대상을 보이지 않는다고 명시적으로 부정한 뒤 but 접속절이 이어진다. "
                                  "frozen parser가 compound scope를 지원하지 않아 보류한 사례이며 단순 abstention으로 같게 취급하지 않는다.")
            elif rule in ("uncertain", "nonvisual_negation_scope"):
                interpretation = "언급 부재·판단 불가를 표현하는 abstention이며, 물체가 없다는 확정 진술로 바꾸지 않는다."
            else:
                interpretation = "지원 문법 밖인 이유는 고정 rule과 raw에서 확인하며 의미를 임의 보정하지 않는다."
            invalid_rows.append((row["question_id"], _raw(record["raw_output"]), _cell(rule), interpretation))
        lines.append(_table(("QID", "Ours raw", "frozen semantic rule", "해석상의 구분"), invalid_rows))
        lines.append("따라서 이 차이를 전부 단순 formatting error, 전부 확정적 yes/no 오답, 또는 전부 객체 token "
                     "누락으로 동일시할 수 없다. parser를 바꾸거나 정답에 맞춰 재해석하지 않고 기존 invalid 판정과 점수를 유지했다.")
    lines.append("### 2) GC zero=0이면 spatial floor의 이득은 없는가?\n\n"
                 "새로운 coarse region을 덮는 **binary support 이득은 0**이다. 이는 정답 이득이 반드시 0이라는 뜻이 아니다. "
                 "동일한 모든 region을 덮어도 선택 token의 identity와 밀도, global budget은 다를 수 있다. 아래는 그 층의 직접 관측치다.")
    full_rows = []
    for summary in analysis["summaries"]:
        subset = [row for row in rows if row["k"] == summary["k"] and row["methods"]["pure_gc"]["zero_regions"] == 0]
        stat = next(item for item in summary["gc_coverage_strata"] if item["gc_full_region_coverage"])
        full_rows.append((summary["k"], stat["n"], stat["net_left_minus_right"],
                          sum(row["newly_covered_regions"] for row in subset),
                          _percent(_average(row["overlap_fraction_of_k"] for row in subset)),
                          sum(row["overlap_count"] < row["k"] for row in subset),
                          _ids(stat["left_only_correct_qids"]), _ids(stat["right_only_correct_qids"])))
    lines.append(_table(("K", "GC zero=0 n", "Ours−GC 정답 Δ", "새 region 총합", "공통 index/K 평균", "index 집합이 다른 설정", "Ours만 정답 QID", "GC만 정답 QID"), full_rows))
    lines.append("### 3) H가 클수록 Ours가 유리한가?\n\n"
                 "전체 paired 상관과 tertile 순이득을 아래 수치 그대로 읽어야 한다. 그룹 순이득의 비감소 여부는 "
                 "Low→Medium→High의 관측 순서만 요약하며 유의성이나 일반화의 증거가 아니다. K 간 통합 correlation은 계산하지 않았다.")
    correlation_rows = []
    for summary in analysis["summaries"]:
        pair = summary["pairs"]["ours_vs_pure_gc"]
        entries = [next(item for item in analysis["pair_groups"] if item["k"] == summary["k"] and item["group"] == group and item["pair"] == "ours_vs_pure_gc") for group in GROUPS]
        deltas = [entry["net_left_minus_right"] for entry in entries]
        nondecreasing = all(left <= right for left, right in zip(deltas, deltas[1:]))
        correlation_rows.append((summary["k"], _number(pair["pearson_H_delta"]), _number(pair["spearman_H_delta"]),
                                 " / ".join(f"{value:+d}" for value in deltas), "예" if nondecreasing else "아니오"))
    lines.append(_table(("K", "전체 Pearson H–Δ", "전체 Spearman H–Δ", "Low / Medium / High 순이득", "관측 순이득 비감소"), correlation_rows))
    evidence = []
    high_advantages = []
    for summary in analysis["summaries"]:
        pair = summary["pairs"]["ours_vs_pure_gc"]
        high = next(entry for entry in analysis["pair_groups"]
                    if entry["k"] == summary["k"] and entry["group"] == "High"
                    and entry["pair"] == "ours_vs_pure_gc")
        high_advantages.append(high["net_left_minus_right"])
        evidence.append(
            f"K={summary['k']}: 전체 Ours만 정답 {pair['left_only_correct']}개, "
            f"High에서 Ours {high['left_correct']}/{high['n']} 대 GC {high['right_correct']}/{high['n']}, "
            f"전체 Spearman ρ={_number(pair['spearman_H_delta'])}")
    if all(value <= 0 for value in high_advantages):
        conclusion = "**현재 primary 결과는 High-H에서 Ours가 Pure GC보다 유리하다는 근거를 제공하지 않는다.**"
    else:
        conclusion = "현재 primary의 High 그룹에 양의 순이득이 있더라도 작은 고정 표본의 관측이며 일반적인 H 효과를 입증하지 않는다."
    lines.append(conclusion + " " + "; ".join(evidence) + ". "
                 "상관이 양수여도 Low에서의 손실 감소 때문일 수 있고, 관측 순이득 비감소가 모두 동률인 경우에는 "
                 "High-H의 추가 이득을 의미하지 않는다. H에 따른 인과 효과나 논문의 전체 benchmark 경향을 검증한 결과로 해석하지 않는다.")
    lines.append("### 4) K가 달라질 때 floor 효과 차이를 H로 해석해도 되는가?\n\n"
                 "현재 비교에서는 K와 coarse grid G가 함께 변한다. 강제 floor의 G/K 비율, 지역 크기, "
                 "H 자체와 rank group까지 바뀌므로 K 효과·공간 granularity·floor 강도를 분리하지 못한다.")
    lines.append(_table(("K", "grid", "G", "강제 floor G/K", "phase-2 잔여 K−G"), [
        (summary["k"], f"{summary['side']}×{summary['side']}", summary["side"] ** 2,
         _percent(summary["side"] ** 2 / summary["k"]), summary["k"] - summary["side"] ** 2)
        for summary in analysis["summaries"]]))
    lines.append("### 5) 다음에는 무엇을 확인할 것인가?\n\n"
                 "후속 실험의 제안만 남긴다. (a) 동일 K에서 G를 바꾸고 동일 G에서 K를 바꾸는 교차 설계, "
                 "(b) local/global budget을 맞춘 뒤 지역 내 선택 효과를 비교하는 control, "
                 "(c) GT bounding box·segmentation 또는 검증된 receptive-field 대응을 이용한 객체 보존 측정, "
                 "(d) 더 큰 사전 고정 표본과 여러 seed에서 paired 성능·invalid·H 정의 민감도의 재확인, "
                 "(e) GC zero=0과 zero>0을 미리 나눈 분석이 적절하다. 이번 단계에서 어떤 후속 모델 실험도 구현하거나 실행하지 않았다.")
    return "\n\n".join(lines)


def _appendix_section(analysis, rows):
    lines = ["## 부록. 전체 image × budget 행과 산출물"]
    lines.append(_table(("QID", "image_id", "K", "원본 W×H", "crop rows×cols", "N", "grid", "H raw", "H normalized", "region μ", "region σ", "whole-image variance", "group", "GC zero", "Ours zero", "Ours−GC Δ"), [
        (row["question_id"], row["image_id"], row["k"], "×".join(map(str, row["metrics"]["image_size"])),
         f"{row['crop_rows']}×{row['crop_cols']}", row["n"], "×".join(map(str, row["region_grid"])),
         _number(row["metrics"]["heterogeneity_raw"], 6), _number(row["metrics"]["heterogeneity_normalized"], 6),
         _number(row["metrics"]["mean"]), _number(row["metrics"]["std"]),
         _number(row["metrics"]["whole_image_laplacian_variance"]), row["group"],
         row["methods"]["pure_gc"]["zero_regions"], row["methods"]["ours"]["zero_regions"],
         row["pairwise"]["ours_vs_pure_gc"]["delta"]) for row in rows]))
    lines.append("- [전체 수치와 raw 응답 JSON](analysis.json)\n"
                 "- [이미지별 CSV: quantiles·raw/normalized matrix·모든 방법·pairwise 판정](per_image_analysis.csv)\n"
                 "- [그룹 성능 CSV](group_comparison.csv)\n"
                 "- [그룹 paired CSV](paired_group_comparison.csv)\n"
                 "- [정의 민감도 CSV](sensitivity.csv)\n"
                 "- [사전 분석 계획](analysis_plan.json)\n"
                 "- [입력 SHA·변경 여부·실행 범위](provenance.json)")
    lines.append("### 전체 그림\n\n"
                 "![H와 Ours−GC paired 차이](figures/heterogeneity_vs_ours_minus_gc.png)\n\n"
                 "![H 그룹별 정확도](figures/accuracy_by_heterogeneity_group.png)\n\n"
                 "![H와 Pure GC zero regions](figures/gc_zero_regions_vs_heterogeneity.png)")
    lines.append("### 재현 방법과 이번 분석 파일\n\n"
                 "아래 명령은 저장된 결과를 읽는 CPU 분석이다. output directory는 **아직 존재하지 않는 새 경로**로 "
                 "지정해야 하며 기존 분석·보고서는 덮어쓰지 않는다. 모델·selector·generation을 실행하지 않는다.\n\n"
                 "```powershell\n"
                 "$env:PYTHONDONTWRITEBYTECODE='1'\n"
                 "& 'python' -B "
                 "examples/analyze_pope_spatial_heterogeneity.py --output-dir results/pope_spatial_heterogeneity_new\n"
                 "```\n\n"
                 "분석 소스: `src/analysis/spatial_heterogeneity.py`, "
                 "`examples/analyze_pope_spatial_heterogeneity.py`, "
                 "`examples/plot_pope_spatial_heterogeneity.py`, "
                 "`examples/report_pope_spatial_heterogeneity.py`.\n\n"
                 "추가된 관련 테스트: `tests/test_spatial_heterogeneity.py`, "
                 "`tests/test_pope_heterogeneity_analysis.py`.")
    return "\n\n".join(lines)


def write_report(path, analysis) -> None:
    """Write one fresh Markdown report using only the supplied saved analysis.

    Existing report paths are refused. No parser, grouping, selection, model,
    image processing, statistical fitting or external resource is executed.
    """
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Existing report is preserved: {path}")
    rows, budgets = _check_schema(analysis)
    methods = analysis["method_order"]
    unique_images = len({row["image_id"] for row in rows})
    n_predictions = sum(len(row["methods"]) for row in rows)
    baseline = analysis["metadata"]["original_semantic_baseline"]
    sections = [
        "# POPE spatial heterogeneity와 allocation 후속 분석",
        f"원본 고정 {unique_images}개 이미지 × {len(budgets)}개 K = **{len(rows)}개 image-budget 행** 및 "
        f"저장된 **{n_predictions}개 method 응답**을 분석했다. "
        f"FullModel semantic baseline은 {baseline}/{unique_images}이다. "
        f"group 성능 {len(analysis['groups'])}행, paired group {len(analysis['pair_groups'])}행, "
        f"K별 summary {len(analysis['summaries'])}개, secondary sensitivity {len(analysis['sensitivity'])}개를 모두 보고한다. "
        f"semantic parser: `{_cell(analysis['metadata']['semantic_parser_version'])}`. "
        "새 generation·selector 실험은 실행하지 않았다.",
        _source_section(analysis, rows),
        _aggregate_section(analysis, rows, budgets, methods),
        _group_section(analysis),
        _categories_section(analysis, rows),
        _failure_section(analysis, rows, budgets, methods, path),
        _sensitivity_section(analysis),
        _questions_section(analysis, rows, budgets),
        _appendix_section(analysis, rows),
    ]
    report = "\n\n".join(sections) + "\n"
    # Exclusive creation also protects against a target appearing after the
    # initial check. The parent analysis owns creation of the output directory.
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)
