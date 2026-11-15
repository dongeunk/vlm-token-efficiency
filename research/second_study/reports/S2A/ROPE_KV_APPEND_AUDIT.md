> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s2a/phase_s2a_20261114_01/ROPE_KV_APPEND_AUDIT.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# 실제 1D RoPE·KV-cache·Decode Append 감사

실행 대상은 cached Idefics3/Llama/Transformers5.15.1, K=N832다. Qwen M-RoPE나 구형 cache counter를 사용하지 않는다.

## 실제 관측값

| 항목 | 결과 |
|---|---|
| actual cache class | `transformers.cache_utils.DynamicCache` |
| per-layer type | DynamicLayer, `layers[i].keys/values` |
| layers | 30 |
| native prefix | 길이878, positions0..877 |
| prefill mask | [1,1,878,878], 허용 causal 위치0/차단 위치finfo min |
| decode forwards | 10, 매 step query1 |
| decode positions | 878..887 |
| 마지막 decode mask | [1,1,1,888] |
| final all-layer lengths | physical/public 전부888 |
| final all-layer K/V shapes | [1,3,888,64] |
| generated tokens | 11, 마지막EOS49279 |
| cache 계산 | 878+11−1=888 |
| cache_position keyword | 없음; 실제 설치 API에 공식 인자가 없고 주입하지 않음 |

실제 rotary module이 받은1D IDs와 output cos/sin을 관측하고 각 layer가 같은 값의 actual cos/sin을 받는지 검사했다. Native/Custom/No-op의 값은 exact 일치했다. cache physical/public length와 각 layer의 append 전후 길이, 새 key/value tail도 비교했다.

## No-op boundary

layer0은 전체878 sequence/832 visual을 먼저 처리했다. upper layers의 cache가 비어 있을 때 keep=0..877로 H1/IDs/mask/original positions/layer0KV를 실제 GPU gather했다. 모든 payload exact 보존, 모든 visual832 보존, 삭제0이다.

suffix causal mask는 upper cache가 empty인 기준으로 만들었다. layer0 cache length를 과거로 잘못 사용해2L mask를 만들지 않았으며 원래 mask와 exact 같았다. suffix RoPE 재계산의 cos/sin도 exact 같았다. No-op prefill rotary 호출은2회지만 값 일치가 검증된 재계산이며 동일성 실패로 처리하지 않았다.

## 남은 미검증

K<N의 gaps가 있는 position IDs, short physical cache, 선택 후 Layer1 입력, 실제 ERC 및 방법별 삭제는 실행하지 않았다. No-op의 identity equality를 이 경우의 정합성 증명으로 확장하지 않는다. CPU S1의 mock 계약이 있어도 actual short-cache binding은 S2-B에서 별도 검증해야 한다.

각 route의 `*_TRACE.json`, Custom/No-op `*_RUNTIME.json`, raw CPU tensors에 실제 경계/step 정보가 있다. 실제 HF cache를 호출한 이번 결과와 S1 fake cache test를 구분한다.
