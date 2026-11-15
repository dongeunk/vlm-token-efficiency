> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s2b/phase_s2b_20261114_01/ROPE_KV_APPEND_AUDIT.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Actual RoPE·KV-cache·Decode Append 검증

**6회 모두 PASS.** 실제 class는 `transformers.cache_utils.DynamicCache`, per-layer `DynamicLayer`이며 `layers[i].keys/values`와 `get_seq_length(i)`를 관측했다. HF cache와 실제 Llama layers를 실행한 결과이며 S1 mock 결과와 구분한다.

## full→compact prefill

| 단계 | query/physical 길이 | 원래 위치 의미 |
|---|---|---|
| full Layer0 | 878, visual832+nonvisual46 | original1D positions0..877 |
| Layer0 뒤 gather | 110, visual64+nonvisual46 | keep positions에 해당하는 original IDs, 중간 gap 유지 |
| Layer1 input | [1,110,576] | 실제 H1의 같은 keep gather와 exact 일치 |
| upper29 layers | 각110, 과거KV empty에서 채움 | gathered original position IDs/cos/sin |
| prefill 완료 | 모든30층 physical/public110 | 마지막 text position877 보존 |

실제 Layer0 KV도 sequence axis−2를 같은 keep으로 gather했다. 이미 original RoPE가 적용된 layer0 keys를 다시 회전하지 않았다. actual cache object가 바뀌지 않고 upper layers의 KV는 gather 시점에 비어 있는지 검사했다.

suffix causal mask는 explicit2D mask와 empty upper-past(cache=None)로 만들었다. actual shape[1,1,110,110]과 값이 full[1,1,878,878]의 두 축 keep gather와 exact 일치했다. gappy position을 packed-sequence 경계로 잘못 해석하지 않았고, suffix cos/sin도 full rotary의 keep gather와 exact 같았다.

## 실제 decode

생성11tokens 중 마지막 EOS는 forward하지 않는다. 나머지10개가 실제 append됐다.

- logical positions: **878..887**.
- physical cache before:110..119, after:111..120.
- 각 decode query는1, 2D mask는111..120.
- 마지막 actual causal mask:[1,1,1,120].
- final 모든30층 physical/public length:**120=110+11−1**.
- final K/V shape:[1,3,120,64].

설치 Llama API에 없는 `cache_position` 인자를 넣지 않았다. logical878을 physical slot110으로 재번호화하지 않고 서로 다른 값으로 유지했다. cache lengths, actual appended tails, actual positions와 layer에 전달된 cos/sin/mask를 모든 step에서 검사했다.

동일 방법의 두 실행은 전체 final KV tensors까지 exact 비교했다. 다른 방법의 KV나 답변이 같아야 한다는 조건은 없다. Global16·full prefix ERC는 공통 경로이므로 방법 간 exact 공통 조건으로 별도 확인했다.

## 근거와 한계

각 `*_TRACE.json` 및 `*_CPU_TENSORS.pt`, runtime/result의 `boundary`, `steps`, `final_cache`에 실제 기록이 있다. Root JSON은8백여 ERC 및 keep/selected ordinals/원래 sequence positions를 저장한다.

검증 범위는 N832/L878/T46/배치1/FP16/eager 한 입력이다. N1088·다른 질문 길이·padding·여러 이미지·다른 cache 종류는 actual GPU 미검증이다. 이를 자동 지원한다고 주장하지 않는다.
