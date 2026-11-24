# Study4 연구 요약

**공간 제약과 질문 관련성을 이용한 시각 토큰 선택 · SmolVLM-256M-Instruct**

## 연구 목적

VLM의 시각 토큰을 64개로 줄일 때, 이미지 전역을 고르게 대표하도록 하는 제약이 객체 존재 판정에 도움이 되는지 조사했다. 또한 같은 공간 제약 아래 어떤 대표 토큰을 선택하는지가 결과를 얼마나 바꾸는지 비교했다.

## 구현 및 검증

CoverPruner의 질문 가중 Coverage를 SmolVLM에 이식했다. 전역 Greedy A와, 5×5 셀에서 하나씩 먼저 선택한 뒤 실제 Seed25 Coverage를 유지해 39개를 추가하는 다섯 방법을 구현했다. 셀별 기준은 중심 거리(B), 코사인 medoid(C), 질문 Attention(D), Vision Attention(E), 제약하의 한계 Coverage(F)다. 시각 토큰 축소를 첫 Decoder Prefill 전에 적용하고, 비시각 토큰과 원래 순서를 보존하는 Cold-KV 경로를 N576·N832·N1088에서 확인했다. [최종 보고서](STUDY4_FINAL_RESEARCH_REPORT.md)

## 핵심 결과

공식 POPE Random Dev100의 100장·600문항에서 Full과 A–F의 총 4,200개 응답을 비교했다.

| 방법 | Strict Accuracy | Strict F1 | Yes Recall |
|---|---:|---:|---:|
| Full | 86.33% | 84.47% | 74.33% |
| A · 전역 Coverage | 65.83% | 48.10% | 31.67% |
| D · 격자별 질문 Attention | 69.17% | 55.42% | 38.33% |
| F · 격자 제약 Coverage | 63.67% | 42.93% | 27.33% |

D는 A 대비 Accuracy **+3.33%p**, F1 **+7.32%p**였다. 이미지 단위 paired bootstrap에서 Accuracy 차이의 95% 구간은 **[+0.67, +6.17]%p**였다. A에서 D로 바꿨을 때 오답→정답 36건과 정답→오답 16건이 관찰됐다. F의 Accuracy는 A보다 **2.17%p 낮아**, 공간 대표를 Coverage만으로 강제하면 성능이 개선될 것이라는 가설은 이 표본에서 지지되지 않았다. 동일한 공간 보장에서도 셀의 대표를 고르는 기준이 달라지면 결과가 달라졌다. [성능 CSV](../results/performance.csv) · [Paired 변화](../results/paired_vs_A.csv) · [신뢰구간](../results/cluster_bootstrap.csv)

## 한계와 후속 방향

가장 좋은 K64 조건 D도 Full보다 Accuracy가 17.17%p 낮았다. 모든 K64 조건에서 False Negative가 늘었으며, 이 패턴은 객체 존재 단서 보존을 후속 연구의 우선 질문으로 제시한다. Prefill 구간은 짧아졌지만 현재 계측은 공유 준비 비용을 배정한 결과라 실제 종단간 가속을 확립하기 어렵다. 연구 범위는 단일 모델·예산·POPE 유형의 탐색적 Dev100이다. 독립 Test400 평가가 남아 있으며 D의 차이를 인과 효과로 확정할 수 없다.

후속 연구에서는 독립 Test에서 고정 비교를 검증하고, K128/K256과 다른 Backbone, 격자 크기, 질문별 토큰 지도 및 객체 단위 오류 분석으로 범위를 넓힐 수 있다.
