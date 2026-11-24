# Study5-P1 사전 등록 — Spatial Constraint Ablation

## 질문과 범위

Study4에서 관찰된 D−A 차이는 격자 보장과 질문 점수 기반 대표 선택을 함께 바꾼 비교다. Study5의 주 질문은 질문 Attention으로 최초 25개를 고르고 동일한 Coverage39를 적용할 때, 5×5 셀별 Top-1이 전역 Top-25보다 높은 Strict Accuracy를 보이는가이다. 현재 등록 범위는 방법·설정·분석 계획이다. 독립 평가셋은 별도 적격 판정 후 고정한다. [Study4 공개 보고서](https://github.com/dongeunk/vlm-token-efficiency/blob/main/study4/docs/STUDY4_FINAL_RESEARCH_REPORT.md)

## 다섯 방법과 불변 조건

| ID | 최초 선택 | 후속 선택 |
|---|---|---|
| A | 빈 집합, Study4 원본 | 기존 전역 Coverage Greedy로 K64 |
| D | 기존 5×5 셀별 raw Query Attention Top-1, 셀 순서 seed 25개 | Study4 원본의 seed Coverage로 Greedy39 |
| G | 동일 raw 점수의 전역 Top-25 | D와 동일한 seed-aware Coverage Greedy39 |
| H | 동일 raw 점수의 전역 Top-64 | 추가 단계 없음 |
| I | D와 동일한 셀별 seed 25개를 같은 순서로 유지 | 미선택 후보의 전역 raw Query Attention Top-39 |

높은 점수를 우선하며 정확히 같은 FP32 점수는 낮은 native visual ordinal을 먼저 선택한다. D의 셀 내부 동률과 Coverage 한계이득 동률도 Study4의 낮은 ordinal 규칙을 유지한다. H의 최초 25개는 G의 seed와 일치하는 진단용 prefix다. 각 방법의 선택 64개는 중복 없이 원본 시퀀스 위치로 정렬한 뒤 Decoder에 전달한다.

Vision Encoder·Projector, 원본 후보 모두, 입력과 chat template, 첫 Decoder layer의 Q/K-only Probe와 원본 위치·RoPE·native causal mask·full causal-prefix softmax를 공유한다. 평균 Query-to-Visual score의 원본 FP32 벡터를 D/G/H/I 순위 기준으로 사용한다. A/D/G의 Coverage demand는 동일 벡터의 기존 min–max 정규화와 `exp(0.3 × normalized_score)`를 사용한다. D/G의 차이는 초기 seed의 후보 조건뿐이다. H/I의 최종 Coverage는 선택 이후 같은 목적함수로 계산한다. D/G는 실제 seed 최근접 대표 점수 상태를 유지한 채 39단계의 기존 greedy를 수행한다.

공통 실행 프로파일은 SmolVLM-256M-Instruct snapshot `7e3e67edbbed1bf9888184d9df282b700a323964`, GTX 1660 SUPER, 기존 Conda, Batch1 No-Padding, SDPA·FP16, greedy, max_new_tokens=10이다. 질문에 `Answer with Yes or No only.`를 붙이는 Study4 형식을 유지한다. 합쳐진 입력을 첫 Prefill 이전에 압축하고 모든 비시각 토큰과 순서를 보존한다. Compact Position과 Native Causal Mask에서 빈 KV를 시작한다. Left Padding FP16 parity BLOCKER는 별도 진단 상태다.

## 비교와 통계

- **주 비교**는 동일 문항에서 D−G Strict Accuracy 차이다. 두 방법의 seed 공간 조건을 비교하며, 후속 39개가 바뀌는 영향도 포함하는 정책 비교다.
- **보조 지표**는 Strict F1, Yes Recall, No Recall, TP/FP/TN/FN 및 Invalid다. A 기준 **D−A, G−A**, 설계 대비 **D−H, D−I, G−H, I−H**를 보고한다. D−I와 G−H는 각각 격자 seed와 전역 seed를 공유한 completion 대비이며, I−H는 Query-only completion에서의 seed 공간 조건 대비다. D−H는 두 요소를 함께 바꾼다.
- 동일 이미지의 공식 질문 6개를 cluster로 유지한다. 원본 Manifest 순서의 이미지 cluster를 복원추출하며 모든 방법에 같은 인덱스를 적용한다. `random.Random(20261126)`, 10,000회, 양측 percentile 95% CI, 선형 보간을 사용한다. 효과의 방향은 표기한 앞 방법에서 뒷 방법을 뺀 값이다. 주 비교에서 CI 하한이 0보다 크면 해당 고정 평가셋에서 개선을 지지한다. 보조 비교는 다중 비교 보정 없이 탐색적으로 보고하며, 독립적인 통계적 우월성을 선언하지 않는다.
- Strict Invalid는 Accuracy의 오답이다. Yes/No Recall의 분모는 각 실제 클래스 전체다. Yes를 양성으로 해 Precision·F1을 계산한다. Study4 Official-compatible Parser는 보조 결과로 제공한다. 빈 분모의 비율은 0으로 기록한다.
- seed와 최종 64개의 셀별 점유 수, 비어 있지 않은 셀의 수·비율 및 셀당 최대 토큰 수를 기록한다. H/A의 최초 25개를 비교할 때는 실제 seed와 구분해 prefix로 표기한다. 동일 유틸리티와 수요 가중치에서 최종 가중 Coverage를 기록하고 seed 점유와 Coverage 초기값도 D/G/I에 남긴다. 공간 점유와 Coverage는 별도 현상으로 해석한다.

## 평가 순서와 Freeze

1. **GPU Smoke**는 추후 별도 승인 후 검증된 비평가 N576 이미지 한 장·질문 하나에서 A/D/G/H/I를 각각 한 번 실행한다. 총 최대 5회이며, 이 단계에서는 실행하지 않았다. 하나의 raw score 해시, K64·순서·전처리·Cold-KV·finite logits·생성 계약을 점검한다. N832/1088 실모델 Smoke는 별도 범위에서 진행할 수 있으며, 이번 단계의 검증은 합성 CPU와 후보 이미지 Processor에 국한한다.
2. **Dev** 평가가 별도로 승인되면 Study4 고정 Dev100을 탐색적으로 사용한다. 모든 조건에 같은 문항을 적용하며 Study4의 A/D 공개 수치를 비교 맥락으로 보존한다. 기존 결과 재사용 여부는 원본 입력·모델·런타임과 체크포인트가 일치할 때만 별도로 판정한다.
3. **Holdout**은 RESERVE79의 저장 순서를 유지하고, 사전 정의한 노출·정확 중복·근접 중복 검토·Processor 호환성 Gate를 모두 통과하는 후보 전체를 사용한다. dHash≤10 의심 쌍은 이미지 확인으로 해소되기 전까지 평가 대상에서 보류한다. 노출 범위의 미확인 부분을 해결하고, 제외 사유와 최종 Manifest 및 프로토콜 해시를 결과 조회 전에 고정한다. 평가 시작 후에는 실패 문항·이미지를 방법별로 임의 탈락시키지 않는다. 전체 79장이 적격이면 474문항과 다섯 조건 2,370개 응답이 상한이다. 독립 자격이 해소되지 않으면 탐색 후보로만 보존한다.
4. Holdout 결과를 확인한 뒤 알고리즘·parser·예산·지표·제외 기준을 소급 변경하지 않는다. 변경이 필요하면 새 연구 단계 및 새 독립 표본으로 분리한다.

[노출 감사와 환경](STUDY5_P1_REPORT.md)

