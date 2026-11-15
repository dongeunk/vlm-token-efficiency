> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s1/phase_s1_20261114_01/MOCK_PREFILL_CACHE_CONTRACT.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Offline Prefill 및 Mock KV-cache 계약

실제 HF DynamicCache, LlamaDecoder, model, RoPE 계산을 실행하거나 구현하지 않았다. 아래는 공급된 CPU tensor의 선택·순서·shape·논리 위치를 검증하는 계약이다.

## 인터페이스 경계

| API | 입력·출력 |
|---|---|
| `map_prompt_positions` | CPU int64 IDs[1,L]→visual/nonvisual positions, expected N 대조 |
| `compute_erc_scores` | supplied H0/H1[1,L,D]→native residual/FP32 norm[N] |
| `build_pruned_prefill` | supplied post-Layer0 H1, IDs/mask/original1D positions, selected visual ordinals, layer0 keys/values→독립 compact payload |
| `MockKVCache.from_layer0` | supplied layer0KV→layer0 populated/upper empty cache snapshot |
| `gather_layer0` | sorted keep positions와 explicit next logical position→layer0 compact, upper empty |
| `fill_upper_layers` | supplied upper KV arrays→all layers 같은 compact length |
| `append_token` | 모든 layer의 supplied1-token KV와 exact logical position→physical length+1/logical position+1 |
| `build_decode_contract` | valid pruned prefill+complete mock cache→2D mask,1D position,explicit4D mask,physical slot |

prefill은 batch1·single-image·unpadded all-ones mask를 지원한다. 원래 positions는 nonnegative offset에서 시작하는 contiguous1D int64여야 한다. padding/multi-image/beam/gappy-original-input은 자동 지원하지 않는다.

selected ordinal은 sorted/unique/range를 검증한다. prompt의 모든 nonvisual positions와 합쳐 원래 순서로 gather한다. IDs/H1/mask/original position IDs 및 layer0KV의 sequence axis−2에 동일 keep을 적용한다. 마지막 prompt position은 first-logit 의미를 위해 반드시 보존한다. text/special hidden과 IDs가 exact 보존되는지 확인한다. 키는 supplied 값 그대로 gather하며 RoPE를 재적용하지 않는다.

## Physical slots와 original positions

L=N+T, K64이면 suffix length L'=T+64다. Layer0 입력은 L이고 gather 후 layer0KV도 L'다. upper layers는 빈 cache에서 L'을 한 번 채운다.

- suffix causal mask는[1,1,L',L']이며 upper past=0이다. 이미 채워진 layer0 길이를 upper past로 잘못 넣는2L' mask를 만들지 않는다.
- selected original1D positions에는 gap이 남고 재번호화하지 않는다. 명시적 mask를 사용해 gap이 packed-sequence 경계가 되지 않도록 runtime에 전달할 계약을 둔다.
- next logical RoPE position=원래 마지막 position+1.
- next physical slot=compact cache length.
- append가 j개면 모든 layers의 길이=L'+j, next logical=원래 next+j다.
- 실제 greedy에서 m개 생성한 뒤 보통 마지막 token은 아직 forward되지 않으므로 최종 cache=L'+m−1이다. S1은 token을 생성하지 않고1-token append payload만 검사했다.

## 소유권과 오류 처리

입력 payload를 복제하고 각 transition은 새 cache snapshot을 반환한다. 원본/full/compact/upper-filled/appended cache가 storage를 공유하지 않는다. 잘못된 append는 원본을 일부 변경하지 않는다.

decode 전 모든 layer의 shape/dtype/finite/physical length와 next logical offset을 확인한다. layer0만 채웠거나 mixed lengths, stale logical position, 잘못된 token shape, 재-gather는 거부한다. mutable PrunedPrefill은 decode 경계에서 IDs/visual ownership/counts/keep/mask/positions/explicit causal matrix/KV schema/metadata를 다시 검증한다. numpy boolean scalar도 정수 설정으로 받지 않는다.

## 결과

Prefill12 + MockKV8 + adversarial3 tests가 통과했다. synthetic diagnostics는 실제와 같은30개 layer 수를 사용하되 hidden D8, KV heads3/head_dim4인 작은 supplied payload다. 실제 hidden576/head_dim64의 decoder 연산이 아니다.

| synthetic case | full Layer0 length | compact suffix | 30층 append 후 | next logical |
|---|---:|---:|---:|---|
| local3×4, N832, T44 | 876 | 108 | 109 | 876→877 |
| local4×4, N1088, T56 | 1144 | 120 | 121 | 1144→1145 |
| no-local64 identity | 72 | 72 | 별도 no-op gather만 | 원래 길이 유지 |

K=N identity-gather는 모든 visual을 supplied identity selection으로 보존하는 runtime 계약 검증용이다. 고정 local48/global16 allocator의 일반 K 지원을 뜻하지 않는다. 실제 cache class/API·rotary·mask·decode 수치 검증은 다음 runtime 단계에 남는다.
