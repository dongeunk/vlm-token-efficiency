# Study5 연구 요약 — 공간 제약과 Query Attention 분리

Study4에서는 5×5 격자에서 질문 Attention으로 토큰을 고른 뒤 Coverage를 적용한 D가 전역 Coverage 기준선 A보다 정확했다. Study5에서는 이 개선이 공간 제약에 따른 것인지 알아보기 위해, 동일한 첫 25개 또는 동일한 후속 39개를 사용하는 기준선을 추가했다. SmolVLM-256M-Instruct의 K64를 유지하며 기록상 비중복인 Exploratory Candidate79의 79장·474문항에서 기존 GPU 응답 2,370개를 재검증했다.

| 방법 | 선택 방식 | Strict Accuracy | F1 |
| --- | ---: | ---: | ---: |
| A | 전역 Coverage64 | 62.24% | 39.32% |
| D | 격자 Query25 + Coverage39 | 65.61% | 47.59% |
| G | 전역 Query25 + Coverage39 | 67.09% | 51.25% |
| H | 전역 Query Top-64 | 75.32% | 67.77% |
| I | 격자 Query25 + 전역 Query39 | 75.32% | 67.95% |

- **공간 제약의 추가 정확도 이점은 확인되지 않았다.** 사전 등록 주 비교 D−G는 −1.48%p(95% CI [−4.01, +1.27]%p)였다. Query-only 방식의 I−H도 0.00%p(95% CI [−1.05, +1.27]%p)였다.
- **같은 초기 토큰에서 Query completion이 더 정확했다.** I는 D보다 9.70%p, H는 G보다 8.23%p 높았다. 두 경우 모두 실제 Yes를 No로 답하는 오류가 크게 줄었다. Query 상위 토큰 보존이 질문 관련 단서를 유지했을 가능성은 있으나 토큰과 객체 위치를 직접 확인한 근거는 없다.
- **H는 가장 단순한 공동 최고 Accuracy 방법이다.** H/I의 다른 예측 6건에서 각각 3건을 맞혔다. I의 TP와 FP가 하나씩 많아 Accuracy는 같고 F1은 0.18%p 높았다.
- **Study4의 D−A 개선 방향은 유지됐다.** Accuracy 차이는 Dev100 +3.33%p, Candidate79 +3.38%p였다. 후속 비교에서는 격자 없이 질문 점수를 쓴 G도 A보다 높았으므로 D−A만으로 격자의 기여를 특정하기 어렵다.

Candidate79의 독립성은 미확정이다. 이번 비교는 단일 SmolVLM·K64·객체 존재 평가다. Study5의 비압축 Full 비교와 End-to-End 가속 검증은 수행하지 않았다. 보조 비교는 다중 비교를 보정하지 않은 탐색 결과이고 Left-Padding FP16 parity BLOCKER가 남는다. 후속 연구에서는 독립 표본과 다른 모델·예산에서 결과를 확인하고 객체 위치와 보존 토큰의 관계를 측정할 수 있다. [최종 보고서](STUDY5_FINAL_RESEARCH_REPORT.md) · [사전 등록](STUDY5_PROTOCOL.md)
