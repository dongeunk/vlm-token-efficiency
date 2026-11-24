# Dev100 공개 집계

100장 × 6문항 × 7조건으로 구성된 탐색적 평가다. `performance.csv`, `paired_vs_A.csv`, `cluster_bootstrap.csv`, `coverage_overlap.csv`, `latency_memory.csv`는 완료 로컬 Run의 검증된 요약 CSV다.

`image_cluster_counts.csv`는 공개 검산용으로 이미지 대신 1–100의 순번을 사용한다. 각 이미지에 Yes·No가 각각 3개였고 Invalid가 0이므로, 이미지별 정확도와 Yes Recall에서 TP/TN을 복원하고 각 클래스의 분모로 FP/FN을 계산했다. 복원한 F1을 원본 이미지별 F1과 대조했다. 원본 이미지 ID, Question ID, 질문 텍스트 및 응답을 싣지 않았다. 공유 cluster 순서를 유지해 동일 paired bootstrap을 CPU에서 재계산할 수 있다.

[출처·해시](PROVENANCE.json) · [성능](performance.csv) · [A 대비 변화](paired_vs_A.csv) · [95% CI](cluster_bootstrap.csv) · [Coverage](coverage_overlap.csv) · [시간·메모리](latency_memory.csv)
