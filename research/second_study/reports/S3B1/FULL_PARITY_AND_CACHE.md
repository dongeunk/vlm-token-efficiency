> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s3b1/phase_s3b1_20261114_01/FULL_PARITY_AND_CACHE.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Full 동일성·Pruning·실제 KV 증거

각N에서 Native→Custom→No-op을 먼저 실행했다. Custom/No-op이 Native와 exact 일치한 뒤에만 L→U→C를 실행했다. tolerance는 처음부터0이며 변경하지 않았다. 최대18회 ledger를 write-ahead로 기록했다.

## Full 동일성

| N | 비교 | Layer0 H0 최대오차 | Layer0 H1 최대오차 | 첫logits 최대오차 | generated IDs / 전체 final KV |
|---:|---|---:|---:|---:|---|
|576|Custom vs Native|0|0|0|exact|
|576|No-op vs Native|0|0|0|exact|
|832|Custom vs Native|0|0|0|exact|
|832|No-op vs Native|0|0|0|exact|
|1088|Custom vs Native|0|0|0|exact|
|1088|No-op vs Native|0|0|0|exact|

오차가 있는 위치/최초 불일치는 없다. tensor_difference의 max_abs_index가0좌표로 기록된 경우에는 모든 오차가0인 상태에서 argmax가 고른 좌표이며 실패 지점이 아니다. 각 `<n>_custom_full_RESULT.json`, `<n>_noop_RESULT.json`의 exact_errors에는 dtype/shape·max_abs·좌표가 저장되어 있다.

hooks가 실제 vision/projector·rotary·30개 decoder layer·norm/head를 관측했다. 같은 output token별 실제 forward·cache tail/public length를 대조했고, 마지막에는30층 전체 key/value CPU snapshot도 비교했다. 이는 shape만의 동일성 검사가 아니다. No-op은 full Layer0 뒤 identity selection/gather/KV 코드까지 실제 실행했다.

## 실제 Pruning

같은N에서 세 방법 모두 같은 full Layer0/원본V를 사용했다. Native dense reference와 vision/projector/H0/H1/Layer0 KV가 exact 같고, L/U/C의 실제 GPU ERC도 exact 같았다. ERC는 FP16 차이를 먼저 구한 후 FP32 L2 norm이다. finite count=N, cell별 argmax64개, local48/global16을 확인했다.

`prune_visual_boundary`가 hidden/input_ids/mask/positions/Layer0 KV를 같은keep으로 gather했다. observer는 callback 값뿐 아니라 실제 gather된 다음 decoder 입력과 실제 cache.keys/values를 다시 관측했다. 원래 text/special complement를 전부 유지했고 selected visual ordinals도 unique/sorted/range를 만족했다.

| N | L | T | L'=T+64 | emitted m | compact prefill / final S | logical decode positions |
|---:|---:|---:|---:|---:|---|---|
|576|613|37|101|10|101 / 110|613..621|
|832|878|46|110|11|110 / 120|878..887|
|1088|1143|55|119|10|119 / 128|1143..1151|

각30층 physical/public S가 일치하고 KV는[1,3,S,64]다. Layer0 full KV를 먼저gather하고 upper29층은 short prefill을 수행했다. suffix mask는 full causal mask의 keep×keep gather, rotary는 full cos/sin의 keep gather와 exact 일치했다. 다음 token의 논리 위치는 원래L부터 증가하며 short physical S로 재번호화하지 않았다. 설치 Llama에 없는 cache_position 인자는 추가하지 않았다.

세 방법의 local allocation/cell 수는 달라도 같은 partition 함수와 ERC 수식·runtime을 사용했다. global16은 공통이다. 실제 token 선택/cell/ERC 원문은 `_RUNTIME_CPU.pt`와 observer `_CPU_TENSORS.pt`에 저장되어 있다. method 간 output의 동일 여부는 실패 조건이 아니다.

## 기존832 회귀

기존S2-B 각방법 repeat1의 JSON3개만 read-only로 확인했다. 새832 결과의 local budget, final64 indices,832개 ERC score, generated IDs, final cache lengths가 모두 exact 같다. 결과는 `N832_PREVIOUS_RUNTIME_REGRESSION.json`이다. 다른 과거 예측 전수 분석이나 정답 채점은 하지 않았다.
