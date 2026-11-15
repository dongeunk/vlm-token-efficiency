> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s0/phase_s0_20261114_01/LAYER0_ERC_FEASIBILITY.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Phase S0 — Layer-0 ERC Pruning 가능성 및 실행 계약

작성: 2026-11-14. **정적 설계상 가능, 실제 동작은 미검증.** 이 문서는 향후 구현 명세이며 이번에 모델·CPU 테스트·GPU를 실행하지 않았다.

## 1. 가능성 판정

로컬 Idefics3/Llama는 merger output, 첫 decoder block, per-layer KV에 접근할 수 있다. 따라서 원본 S²Prune의 `full Layer0 → native residual ERC → Cell 대표 → sequence/KV gather → suffix layers` 흐름을 이식할 수 있다.

다만 **기존 pre-LLM pruning helper나 hidden만 자르는 layer hook으로 구현할 수 없다.** LlamaModel은 mask/RoPE를 모든 layer에 공유하고 native generate는 중간 pruning으로 달라진 prompt·mask·cache·position 상태를 자동으로 함께 갱신하지 않는다. 별도 prefill/decode coordination이 필요하다. Qwen 원본 함수 호출을 이름만 바꾸어 쓰지 않는다.

## 2. 상태·인덱스 계약

| 이름 | 의미 |
|---|---|
| N | 전체 original visual tokens, 64(RC+1) |
| T | image placeholder를 제외한 모든 text/special token 수 |
| L=N+T | 원래 unpadded prompt 길이, Layer0 입력 길이 |
| K | retained visual tokens, 주 설정64 |
| L'=T+K | pruning 후 suffix 길이 |
| visual ordinal | native projected V의0..N−1 순서 |
| visual_positions | prompt에서 image_token_id49190 위치의 실제 배열 |
| keep_positions | 모든 nonvisual positions와 visual_positions[selected]의 정렬된 합집합 |
| logical positions | original1D position_ids를 keep gather한 값; 중간 gap 허용 |
| physical cache slots | 0..L'−1 및 이후 append 슬롯; logical positions와 구분 |

separator, row/col tag, global tag, newline, BOS/EOS/chat-template token은 image placeholder가 아니면 모두 보존한다. 이미지 시작 offset 상수를 더하는 방식은 사용하지 않는다. 초기 구현 범위는 batch1·단일 이미지·unpadded prompt이며 padding/multi-image/beam은 별도 확장이다.

## 3. 권고 prefill 절차

다음은 명세이며 신규 알고리즘 구현 코드는 작성하지 않았다.

1. 고정 processor로 전체 prompt/pixels를 준비하고 모든 views의 projected V를 구한다. placeholder 수=N, V shape=[N,576], R/C 일치를 검사한다.
2. native merger와 같은 방식으로 H0=[1,L,576]을 만든다. L/U/C allocation은 동일 projector V/RGB에서 CPU로 계산한다. 임시 Coverage 선택으로 H0를 줄이지 않는다.
3. fresh DynamicCache와 full causal mask, original position_ids=[0..L−1]로 실제 decoder layer0을 한 번 실행한다.
4. visual H0/H1에서 FP16 subtraction 후 FP32 L2 norm으로 ERC[N]을 구한다. post-attention만의 residual이나 final norm 출력을 사용하지 않는다.
5. method별 budgets만 달리하고 동일 Cell/ERC 함수로 K개 native ordinals를 결정한다.
6. H1·2D attention·original position IDs를 keep_positions로 gather한다. **이미 생성된 layer0 keys와 values도 같은 sequence positions로 gather한다.** key는 이미 original RoPE가 적용되어 있으므로 재회전하지 않는다.
7. upper prefill용 explicit compact causal mask를 구성하고 gathered original positions로 RoPE cos/sin을 준비한다. upper layers의 과거 KV는 아직 비어 있다는 기준을 사용한다.
8. layers1..29를 각각 한 번 실행해 동일 cache object의 해당 layer를 L'로 채운다. 모든 layer의 actual KV/public length가 L'인지 확인한다.
9. 마지막 hidden에 Llama final norm과 lm_head를 적용한다. 첫 생성 logits는 마지막 prompt position에서 얻는다. 마지막 text/special 위치가 보존되었는지 검사한다.

Full branch는 allocation/ERC/선택/gather를 호출하지 않고 full30층을 실행한다. K=N no-op branch는 full Layer0와 identity gather를 통과하는 독립 검증 경로다. 이 경로는 고정 local48/global16 예산을 적용하지 않고 supplied identity selection으로 전체N개를 유지하는 runtime 검증용이며, 일반 K 지원이나 allocation 성능을 뜻하지 않는다. Full과 no-op가 같아야 pruning adapter 자체의 영향을 분리할 수 있다.

## 4. 실제 mask/cache 함정과 방지 조건

| 함정 | 필요한 계약 |
|---|---|
| layer0 full cache를 남기고 upper layers만 short | decode에서 공유 mask와 per-layer KV length가 충돌. layer0 KV도 compact하게 gather |
| upper mask 생성에 layer0 cache length를 과거 길이로 사용 | upper layers는 아직 empty인데 L' past로 오인. suffix mask는 empty upper prefix 기준으로 생성 |
| gappy positions + mask 생략 | packed-sequence boundary 오인 가능. explicit2D 또는 검증된4D causal mask 유지 |
| layer0 keys의 RoPE 재계산/재회전 | 이미 원래 positions로 회전된 key 값을 보존하고 gather만 수행 |
| cache tensor만 변경하고 public length 미확인 | 현재 DynamicLayer.get_seq_length()와 actual shape를 모두 검사. 임의 counter 수정 금지 |
| 구형 key_cache/value_cache·cache_position 적용 | 실제5.15.1의 layers[i].keys/values API만 사용. 없는 인자를 주입하지 않음 |
| cache를 방법/질문 간 공유 | 각 generation마다 fresh cache. 동일 projector 입력 공유와 KV 공유를 구분 |

실제 참고 지점은 `LlamaModel.forward:386–400`, `DynamicLayer.update/get_seq_length`, `masking_utils.find_packed_sequence_indices:728` 및 explicit mask/cache 분기:854다. suffix mask를 만들 때 사용하는 helper의 layer index와 past 계산을 CPU mock 계약으로 먼저 드러내야 한다.

## 5. Decode 및 generate 설계

초기 구현은 **Full/no-op/L/U/C 공통 explicit greedy decode**를 권고한다. 모든 방법에서 동일 EOS/PAD/BOS, max_new_tokens16, do_sample=False, num_beams1 정책을 사용한다. POPE 전용 closed-answer early stop, forced continuation, 반복제거를 새로 도입하지 않는다.

- Prefill logits에서 첫 token을 뽑는다. token이 EOS이면 즉시 종료한다.
- 다음 forward에는 실제 생성 token1개와 현재 short cache, 길이 L'+1의2D mask, **logical position L**을 보낸다.
- j개 decode tokens를 cache에 append한 뒤 모든30층 physical length는 L'+j다. 다음 logical position은 L+j다.
- 생성 m개가 정상 종료되면 일반적인 greedy loop의 최종 cache 길이는 L'+m−1이다. 마지막으로 뽑은 token은 아직 forward하지 않았기 때문이다.
- first-token-only/EOS로 append가 없으면 `unexercised`로 보고한다. 미리 잠근 description prompt까지 포함해도 pruning 방법의 append가 관측되지 않으면 decode gate를 통과시킨 것으로 처리하지 않는다.

Native `generate()`는 Full reference로 사용해 custom Full의 logits·실제 greedy IDs·종료를 비교한다. Pruned cache를 그대로 기본 generate에 넣는 것은 초기 지원 범위에서 제외한다. 향후 연결하려면 `prepare_inputs_for_generation:519`, `_prepare_position_ids_for_generation:751`, `_update_model_kwargs_for_generation:940`에서 input slicing·mask·position update가 original logical offset을 보존하는지 별도로 검증해야 한다. explicit position_ids는 마지막 supplied 값+1로 연장될 수 있다(:975–985). 문제는 logical position 보존이 불가능하다는 것이 아니라 모든 generation 상태의 동시 정합성이다. 캐시 길이만으로 position을 추정하면 안 된다.

## 6. CPU 단계와 GPU 단계의 검증 구분

### 다음 CPU 구현에서 검증할 계약

- view-major↔lattice↔prompt positions 양방향 bijection 및 rectangular layouts.
- nonvisual/text/special 보존, final64 unique/sorted, local48/global16, 지역 q와 cell 수 일치.
- synthetic native FP16 residual ERC와 FP32-first subtraction을 구분하는 fixture.
- ERC exact tie의 최소 native ordinal, Coverage tie/zero-gain/capacity 포화.
- mock30-layer cache에서 layer0 full→gather→upper empty fill→decode append.
- suffix mask가 L'×L'이고2L'가 되지 않는 조건, gappy positions를 segment로 쪼개지 않는 조건.
- source/cache 객체 aliasing 및 방법 간 contamination 방지.
- no-local64의 명시적 no-op 및 metadata 불일치 중단.

이 검증은 모델 연산이나 실제 cache/rotary의 수치 정합성 증명이 아니다.

### 후속 GPU에서만 판정할 항목

- Native Full vs custom Full vs K=N no-op: first logits, 실제 greedy token IDs, EOS/길이, layer input/output 및 KV parity.
- selected Layer1 input이 actual H1의 keep gather와 exact 일치.
- 실제 layer0 cache gather 및 모든30층 physical/public length, original RoPE coordinates, actual decode append.
- L/U/C가 동일 H0/H1/ERC와 공통 global selection을 사용하는지 확인. 단, 각 방법의 cache는 독립 생성한다.
- logits/hidden 수치 차이가 나면 원인을 기록하고 중단한다. 최종 indices나 IDs 차이를 tolerance로 숨기지 않는다.

구조·인덱스·같은 payload gather는 exact 검사다. native/custom no-op 수치 비교도 우선 exact로 시작하며 kernel/operation-order 차이가 나오면 별도 원인 분석과 사전 기준 수정 없이 자동 통과시키지 않는다. no-op가 통과해도 K64가 native Full과 같은 답을 내야 한다는 요구는 아니다.

## 7. 근거와 남은 blocker

`third_party/S2Prune/s2prune/qwen.py::s2prune_prefill:330`, ERC:365–367, gather:403–406, `greedy_decode:464`는 흐름의 참조다. Qwen loader/preparer/decoder helper는 Smol runtime으로 재사용하지 않는다.

`src/inference/pruned_vlm.py::build_pruned_inputs`의 text/special 보존 패턴과 `GenerationAudit`의 cache append 관측 원칙은 참고 가능하다. contiguous reindex 및 LLM 입력 전 삭제 전제는 새 adapter에서 교체해야 한다.

현재 blocker는 mapping-aware adapter, actual Idefics3/Llama 전용 prefill/decode coordination, 새 layer0 audit의 **미구현**, image backend의 runtime lock 및 GPU no-op/cache parity의 **미검증**이다. 설치/VRAM 부족 때문에 정적으로 불가능하다는 증거는 없다. 이 문서는 구현 완료나 GPU 성공을 주장하지 않는다.
