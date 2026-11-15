> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s3a/phase_s3a_20261114_01/EVALUATION_PROTOCOL.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Phase S3-A 평가 프로토콜 v1

2026-11-14. 데이터 고정과 CPU 검증만 수행한다. 실제 SmolVLM/processor/CUDA/generation 실행은 0회다. S²Prune 라이선스 HOLD를 유지한다.

## 연구 범위와 사전 결정

SmolVLM 이식 실험의 Full / S2-L / S2-U / S2-C를 비교한다. Qwen2.5-VL-7B 원본 S²Prune 재현이나 공식 POPE 전체 benchmark 결과로 주장하지 않는다. 1차 연구의 pre-LLM Floor/Coverage 결과와도 구분한다.

선정 전에 `PROTOCOL_PRECOMMIT.json`에 seed42·eligibility·샘플링·판정·통계 규칙을 기록했다. S0의 pilot20/40문항, main80/160문항, Accuracy 주 지표, bootstrap seed20261114 제안은 이번 사용자 지시에 따라 **pilot10/60문항, main200/1200문항, F1 주 지표, bootstrap seed42**로 바뀌었다. S0 원문은 수정하지 않았다.

## 데이터와 추출

공식 original COCO POPE Random annotation은 `data/pope/coco_pope_random.json`, provenance는 `data/pope/SOURCE.json`이다. 500이미지·3000문항·Yes1500/No1500을 파일에서 확인했다. POPEv2는 사용하지 않는다.

- annotation SHA256: `ac25245170b975a5bdf9080b23fd431dfe6be458bc038259c1f4f09a6bef7994`
- upstream commit: `08d957b917e5a378a2f99d35b6293c536a66298b`
- old30: `results/pope_pilot/sample_manifest.jsonl`
- old80: `results/pope_spatial_floor_80_preparation/cohort80_five_methods_v2/selected_images.jsonl`

두 exclusion의 중복0·합집합110을 확인했다. 기존 candidate_pool.csv의 470 ID는 공식500−old30과 일치하고, 실제 470 JPEG 파일 ID와도 일치한다. old80을 추가 제외한 390장 모두 유효했다. 과거 manifest의 `image_exists=false`는 다운로드 전 저장된 값이므로 선정 기준으로 쓰지 않았다. old110의 실제 파일 존재도 확인했다.

사전 eligibility는 (1) 공식 image ID, (2) old110 제외, (3) 이미지당 서로 다른 qid/text 6개와 Yes3/No3, (4) 로컬 JPEG ID·decode·크기·과거 source SHA 일치다. 문항이 불완전하면 이미지 전체를 추출 전에 제외하고 사유를 남긴다. annotation JSON/필드/전역 중복 qid, provenance SHA, exclusion 또는 후보 manifest/inventory 불일치는 중단한다. 이번에는 제외할 불완전 이미지0장이다. COCO object annotation 지지 여부, 복잡도 C/D, 과거 응답/정확도를 선정에 사용하지 않았다.

정렬된 eligible image ID에서 `random.Random(42).sample(ids,200)`으로 **본 평가를 먼저** 추출했다. 따라서 main은 390장 모집단의 단순 무작위 표본이다. 독립적인 `random.Random(42)`를 사용해 남은190장에서 파일럿을 고른다. 832/1088별 저장된 processor 관측을 우선하고, 없으면 정적 geometry 예상값에 따라 한 장씩 뽑고, 나머지8장을 무작위로 채운다. 본 평가 추출에는 geometry strata를 사용하지 않는다.

실제 대상 이미지 파일·크기·SHA, 원문 질문·정답·qid, source annotation 파일 SHA·commit·line number·원문 line SHA·canonical record SHA를 manifest에 보존했다. 원문 line SHA는 줄 끝 CR/LF를 제외한 바이트, canonical SHA는 UTF-8 JSON(sorted keys, compact separators, ensure_ascii=False) 기준이다. prompt text와 UTF-8 SHA도 저장했다. 모델 출력에 따른 표본 교체·reroll·질문 변경을 금지한다. 기존 선정 결과가 있으면 `prepare()`는 덮어쓰기를 거부한다.

| 집합 | 이미지 | 문항 | Yes / No | 방법당 generation | 네 방법 합계 상한 |
|---|---:|---:|---:|---:|---:|
| pilot | 10 | 60 | 30 / 30 | 60 | 240 |
| main | 200 | 1200 | 600 / 600 | 1200 | 4800 |

파일럿과 본 평가, old110은 image ID 기준 모두 disjoint다. 질문6개는 전부 사용한다. 기존80장의 Yes1/No1 선택을 재사용하지 않는다.

## Layout의 증거 수준

이번에는 processor나 tokenizer를 생성하지 않았다. JPEG decode는 파일 검증에만 사용했다. 새 manifest의 `actual_processor_visual_tokens`와 `actual_sequence_length`는 전부 null이다. 설치된 Idefics3 resize 소스와 cached 2048/512 설정을 따른 **크기 산술 예상값**만 별도 `geometry_prediction`에 기록했다. 예상값을 실제 processor 관측으로 표현하지 않는다.

| 집합 | 예상832 | 예상1088 | 예상576 | 새 processor/GPU 관측 |
|---|---:|---:|---:|---:|
| pilot10 | 9 | 1 | 0 | 0 |
| main200 | 172 | 27 | 1 | 0 |

pilot anchors는 image554002(640×425, 예상3×4/832), image134075(640×481, 예상4×4/1088)다. 두 이미지의 실제 구조는 S3-B에서 확인한다. 기존80 CPU metadata에는 image376959의 832, image436127의1088이 저장되어 있으나 둘 다 새 평가에서 제외했다. S2-B 실제 GPU 검증은 376959/N832 한 입력에 국한된다.

main image358255(640×234)는 예상2×4/local512+global64=576이다. 표본은 그대로 둔다. 576의 실제 processor 구조 및 GPU 기능을 검증하기 전에는 해당 입력을 지원한다고 주장하거나 main을 실행하지 않는다. 다른 예상 밖 실제 layout에도 같은 중단 규칙을 적용한다.

## 고정 입력·알고리즘·생성 조건

- checkpoint: `HuggingFaceTB/SmolVLM-256M-Instruct`, revision `7e3e67edbbed1bf9888184d9df282b700a323964`.
- Idefics3Processor/Idefics3ImageProcessor, 기존 torchvision backend, RGB, split/resize2048/512. 실제 class/backend와 cached config SHA는 S3-B 실행 시 대조한다. weights는 이번에 열거나 SHA를 계산하지 않았다.
- batch1, FP16, eager attention, greedy, beams1, max_new_tokens16, use_cache=True. EOS49279/PAD2/BOS0. decoding logits processor는 비활성 상태를 확인한다.
- 기존 chat template: `<|im_start|>User:<image>{공식 질문}<end_of_utterance>\nAssistant:`. 추가 Yes/No 지시나 질문 paraphrase를 붙이지 않는다. 모든 방법이 동일한 token IDs/pixels/원본 visual 집합을 받았는지 hash와 metadata로 대조한다.
- Full은 모든 visual token을 유지한다. Full도 동일한 custom decode/stop/timing 계약을 사용하며 Native Full 및 No-op과의 parity를 Gate A에서 확인한다.
- L/U/C: Local48+Global16=K64, local4×4 region/base1+extra32, 동일 S1 Cell 함수·Layer0 ERC·global16 recursive Cell/ERC, 같은 Layer0 이후 pruning 및 runtime. text/special과 원래1D RoPE 위치 보존, Layer0 KV gather, 상위층 short prefill, 원래 L부터 decode positions.
- C만 all-visual demand/local candidates/FP32 calibrated cosine Coverage로 extra32 예산을 만든다. query weight는 없다. 원본 CoverPruner를 지역 예산으로 변형한 SmolVLM 이식이며 원본 논문 exact reproduction으로 표현하지 않는다.

## Parser와 지표

신규 모듈은 기존 `parse_yes_no`, `explain_semantic_yes_no`, `compute_metrics`를 변경 없이 호출한다. parser 입력은 raw text 하나뿐이며 이미지·질문·정답·qid는 전달하지 않는다.

기존 strict는 leading Yes/No 위주라 `There is no ... in the image.`를 invalid로 처리한다. 반대로 `No mention ...`이나 일부 상충 설명을 No로 받아들일 수 있다. 기존 semantic parser `visual_semantic_yes_no_v1`은 완결된 제한적 시각 존재/부재 문법을 추가하지만, 명백해 보이는 자유로운 paraphrase도 지원 범위 밖이면 abstain한다. 학습된 의미 평가기가 아니다. 두 결과, semantic rule_id/reason/evidence, raw text를 함께 보존하고 사후 문법 확장이나 GT별 예외를 금지한다.

**주 지표 `pope_f1`은 frozen semantic parser로 계산한 POPE-style positive-class F1이다.** `pope_accuracy`는 `semantic_accuracy`와 동일한 alias임을 명시한다. strict_accuracy는 별도다. 원본 POPE의 permissive 문자열 heuristic을 구현·실행한 결과가 아니며 공식 전체 POPE 수치와 직접 등치하지 않는다. 이 선택은 S0/S2-B의 frozen strict+semantic 계약을 유지한다.

| 판정 | 규칙 |
|---|---|
| Semantic Accuracy / POPE Accuracy | (TP+TN)/전체문항; invalid도 분모에 남는다 |
| Precision | TP/(TP+FP) |
| Recall | TP/(TP+FN+invalid_yes) |
| F1 | 2TP/(2TP+FP+FN+invalid_yes) |
| Strict Accuracy | strict가 정답과 일치한 수/전체문항 |
| Invalid | semantic invalid_yes/no와 strict invalid를 모두 count/rate로 보고 |

0 denominator는0이다. Invalid를 No로 변환하거나 정답 처리·제외하지 않는다. positive-class F1 자체는 invalid_no를 직접 벌점화하지 않으므로 Accuracy·Invalid·Strict를 반드시 함께 보고한다. 성공한 생성의 빈 문자열/EOS-only 답변은 invalid다. runtime error/OOM/mapping/cache 문제는 parser invalid와 구분하고, incomplete run에는 완전 평가 지표나 CI를 만들지 않는다. 빠진 문항에 대한 complete-case filtering을 금지한다.

`generated_token_ids` 길이는1~16이다. EOS는 마지막에 한 번만 허용한다. 16번째가 EOS이면 at_limit=True지만 truncation으로 세지 않는다. EOS 없이16이면 `possible_truncation=True`다. 이것은 언어적 답변 미완결의 확정이 아니다. EOS 없이16 미만이면 설명되지 않은 조기 종료로 실패한다. 실제 raw text/IDs/EOS/종료 사유를 보존한다. 길이 민감도16/32/64 등은 추후 별도 사전 고정·승인 실험으로만 수행하며 이번 결과를 보고 문항별 생성 길이를 늘리지 않는다.

## Paired 분석과 통계

image/qid/method 키로 4방법 모두 완전해야 한다. 동일 이미지의 6문항을 묶어 paired image-cluster bootstrap10,000회, PCG64 seed42, percentile95% CI(2.5/97.5%, linear quantile)를 사용한다. 모든 방법에 같은 resampled image indices를 적용하고, 뽑힌 문항들의 confusion counts를 합산한 뒤 **micro F1을 다시 계산**한다. 이미지별 F1 평균을 사용하지 않는다. 6문항의 3yes/3no support를 유지한다.

유일한 primary contrast는 **S2-C−S2-L의 POPE-style semantic F1**이다. Accuracy·Invalid rate/count·Strict Accuracy를 함께 보고한다. 다른 방법 차이와 layout/aspect 분석은 exploratory이며 다중 비교 보정 없는 우월성 주장을 하지 않는다. CI는 현재 부분집합 이미지 cluster에 대한 것이며 전체 POPE·다른 데이터·다른 모델로 일반화하는 보장이 아니다. pilot 통계는 운영 진단이며 main과 합쳐 주 분석으로 만들지 않는다.

## Timing 및 메모리 기록

E2E TTFT는 이미지 decode/전처리 시작~첫 token, decoder TTFT는 merged H0 준비 직후(선택 계산 전)~첫 token이다. 모델 로드는 제외한다. CPU handoff/affinity/allocation/Cell/full Layer0/ERC/gather/suffix/head를 실제 구간대로 포함한다. selection_overhead는 allocation+Cell+ERC+최종 gather에 소요된 별도 벽시계 구간이며 중첩 구간을 TTFT에서 다시 더하지 않는다. 정확한 타이밍 이벤트·CUDA synchronize 경계는 S3-B의 별도 runner review에서 고정해야 한다.

Full/L/U/C 동일 입력마다 fresh KV를 쓰고, 배치·해상도·cache 정책을 바꾸지 않는다. 방법 순서는 예상 run keys의 Full→L→U→C로 고정하고 pilot의 timing은 단일 cold/warm 영향이 섞이는 기술 통계임을 명시한다. 선택 결과/Feature cache의 방법 간 재사용은 금지한다. 정상 allocator의 메모리 재사용은 허용한다. warm-up을 위한 미승인 추가 generation은 하지 않는다. pilot240 외 generation은 Gate A의 별도 예산 안에서만 제안한다. 반복 benchmark와 순서 균형화는 추후 별도 실험이다.

TTFT/selection의 mean·median·p95와 peak allocated/reserved MiB를 보존한다. per-item `reset_peak_memory_stats` 후 synchronize 측정, 실행 전후 메모리 상태, generated token 수/append 수도 필요하다. 이번에 실제 TTFT/VRAM을 측정하지 않았다.
