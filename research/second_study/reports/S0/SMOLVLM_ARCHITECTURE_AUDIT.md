> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s0/phase_s0_20261114_01/SMOLVLM_ARCHITECTURE_AUDIT.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Phase S0 — SmolVLM 아키텍처 감사

작성: 2026-11-14, Asia/Seoul. **소스·JSON·기존 보고서의 정적 분석 결과다. 모델 로드, inference, CUDA 초기화, 테스트를 수행하지 않았다.** 이번 산출물은 이 폴더의 설계 문서 6개뿐이다. 기존 코드·평가 결과는 수정하지 않으며 S²Prune 라이선스/publication HOLD를 유지한다.

## 1. 결론과 실제 대상

**SmolVLM에서 전체 Decoder Layer 0 이후 ERC를 계산하고 visual tokens를 제거하는 것은 구조상 가능하다. 실행 정합성은 미검증이다.** 기존 SmolVLM pruning은 LLM 진입 전에 삭제하므로 새 방법의 실행 근거가 될 수 없다.

모델 이름만 보고 최신 `models/smolvlm` 소스를 사용하면 안 된다. 로컬 snapshot은 **Idefics3 vision-language model + Llama decoder**로 구성되어 있다. Qwen M-RoPE, 576-token grid, Transformers 4.49 cache 구조를 적용하지 않는다.

| 항목 | 로컬 파일에서 확인한 값 |
|---|---|
| checkpoint | `HuggingFaceTB/SmolVLM-256M-Instruct` |
| cached revision | `7e3e67edbbed1bf9888184d9df282b700a323964` |
| architecture / model_type | `Idefics3ForConditionalGeneration` / `idefics3` |
| processor | `Idefics3Processor`, `Idefics3ImageProcessor` 계열 |
| 설치 환경 | `<HOME>/miniconda3/envs/vlm_project` |
| 설치 metadata | Transformers `5.15.1`, Torch `2.13.0+cu126` |
| 기존 실행 보고 | Python 3.11.15, FP16/eager, batch 1, GTX 1660 SUPER |
| decoder | Llama, 30 layers, hidden 576, MLP 1536 |
| attention | query heads 9, KV heads 3, head dimension 64 |
| position | 1D Llama RoPE, theta 100000, max positions 8192 |
| vocabulary | 49280, image placeholder ID 49190 |
| generation config | BOS 0, EOS 49279, PAD 2 |

위 표의 버전은 metadata/기존 보고 값이며 이번에 import해서 검사한 값이 아니다. snapshot `config.json`의 BF16 기본값과 과거 저장 버전 `4.46.0`은 현재 FP16/eager 실행 정책 및 실제 설치 버전과 구분한다. text_config 내부의 오래된 perceiver 관련 필드만으로 resampler가 있다고 판단하지 않는다.

로컬 cache 기준 경로는 `<HOME>/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM-256M-Instruct/snapshots/7e3e67edbbed1bf9888184d9df282b700a323964/`이다. `config.json`, `preprocessor_config.json`, `processor_config.json`, `generation_config.json`과 파일 이름·크기만 읽었으며 가중치 파일을 열거나 SHA를 다시 계산하지 않았다.

## 2. Vision Encoder와 Connector

Vision은 SigLIP 계열의 12-layer transformer다. 입력 view 하나가 512×512이고 patch size가 16이므로 32×32=1024 patch embeddings, hidden 768을 만든다. 모든 patch는 vision self-attention을 거친다.

`Idefics3Connector.pixel_shuffle`은 공간의 4×4 patch 묶음을 채널로 모아 8×8=64개로 줄인다. 이어 `Idefics3SimpleMLP`의 bias 없는 projection이 768×16=12288 차원에서 decoder hidden 576으로 보낸다. Qwen PatchMerger나 learned Perceiver latent sequence로 해석하지 않는다.

```text
RGB image → resize/split views [V,3,512,512]
          → vision [V,1024,768]
          → pixel shuffle + projection [V,64,576]
          → native view-major flatten [64V,576]
          → image placeholder에 병합 → [1,L,576]
```

`Idefics3Model.get_image_features`의 `pooler_output`이 decoder에 삽입되는 projected feature다. Encoder의 1024 patch states를 Coverage 후보나 최종 visual token으로 잘못 쓰면 안 된다.

## 3. 이미지별 원본 token 수와 Local/Global 순서

캐시 설정은 `size.longest_edge=2048`, `max_image_size.longest_edge=512`, splitting/resize/normalize/pad 활성화, `image_seq_len=64`다. **2×2 local crop 또는 N=320 고정을 가정하지 않는다.**

이미지는 종횡비를 고려해 resize된 뒤 vision encoder에 맞는 512 배수의 크기로 다시 resize된다. 실제 local grid가 R×C이면 local views가 row-major 순서로 나오고, 전체 이미지를 512×512로 줄인 Global view 하나가 마지막에 온다.

- `N_local = 64RC`, `N_global = 64`, `N = 64(RC+1)`.
- 기존 pilot: N=832가 26장(3×4 또는 방향에 따른 대응 layout), N=1088이 4장(4×4).
- 기존 smoke image 376959는 3×4 local + Global, 436127은 4×4 local + Global로 보고되어 있다.
- R/C는 원본 크기·종횡비·resize rounding·splitting 설정에 따라 달라진다. prompt 길이는 질문과 row/col/global marker 토큰 수도 반영한다.
- splitting 없는 single-view 경로는 R=C=0 metadata와 Global 64개일 수 있다. 이때 local이 있다고 해석하지 않고 별도 no-local 경로로 처리한다.

패딩된 dummy views와 실제 image views를 구분해야 한다. batch=1 단일 이미지의 미래 관측에서 실제 projected view 수, placeholder 수, processor R/C를 서로 대조한다. backend 선택이 실행 시 달라질 수 있으므로 PIL/torchvision 등 실제 image processor class도 후속 단계에서 잠근다. 이번에는 processor를 생성하지 않았다.

## 4. 각 visual token의 좌표

local crop `(r,c)`와 그 안의 8×8 token `(v,u)`에 대해 native visual ordinal은 다음과 같다.

```text
i_native = 64(rC+c) + 8v+u
y_lattice = 8r+v, x_lattice = 8c+u
i_lattice = y_lattice·(8C)+x_lattice
```

native view-major와 stitched lattice row-major는 다른 순열이다. 이 대응은 양방향으로 보존해야 한다. `spatial_floor_coverage.local_region_indices`가 기존 구현 근거다.

resize된 local mosaic는 원본 전체와 대응하므로 nominal normalized center와 box는 다음과 같다.

```text
x_center=(c+(u+0.5)/8)/C, y_center=(r+(v+0.5)/8)/R
x_box=[(c+u/8)/C, (c+(u+1)/8)/C]
y_box=[(r+v/8)/R, (r+(v+1)/8)/R]
```

원본 pixel 좌표는 x에 W0, y에 H0를 곱한다. Global의 `(v,u)`는 같은 식에서 crop offset 없이 `[u/8,(u+1)/8]×[v/8,(v+1)/8]`에 대응한다. local/grid resize에는 종횡비 rounding 변형이 있을 수 있다. **이는 padding 영역을 실제 원본 공간으로 매핑하는 규칙이 아니다.**

pixel shuffle의 최종 8×8 순서는 row-major다. 단, vision의 full self-attention 때문에 이 box는 **명목상 patch footprint**이며 해당 영역만의 정보가 들어 있다는 보장은 없다. Global과 Local은 같은 원본 공간을 다른 해상도와 문맥에서 중복 표현한다.

## 5. 실제 Layer 0과 ERC 위치

Decoder 경로는 `model.model.text_model.layers[0]`이다.

- 입력 H0: text embedding과 projected V가 merger를 거친 전체 multimodal `[1,L,576]`.
- 출력 H1: layer 0의 attention residual 및 MLP residual까지 더해진 hidden. 최종 `LlamaModel.norm` 이전이다.
- ERC: 각 visual placeholder 위치 p에서 `norm((H1[p]-H0[p]).float(), p=2)`. FP16 subtraction을 먼저 하고 FP32 norm을 적용한다.
- embedding norm, attention score, vision encoder residual, 최종 normalized decoder hidden은 ERC 대체물이 아니다.
- `output_hidden_states=True`로 30개 layer 전체를 저장할 필요가 없다. H0의 visual slice와 실제 layer0 출력만 관측하면 된다.

일반적인 image-then-question causal prompt에서는 visual position이 뒤의 질문을 보지 못한다. 따라서 여기서 ERC를 곧바로 query-conditioned score라고 부르지 않는다. allocation은 명시적으로 image-only이며 ERC/final index/KV는 질문별 새 실행으로 검사한다.

## 6. Position, RoPE, cache, generate의 이식 경계

Llama는 original sequence의 **1D position_ids**를 rotary에 사용한다. Qwen의 3축 M-RoPE는 없다. 설치된 LlamaModel은 causal mask와 RoPE를 forward당 한 번 만들고 모든 decoder layers에 공유한다. 따라서 hidden만 짧게 만드는 forward hook은 안전한 pruning 구현이 아니다.

실제 cache는 `DynamicCache`의 `cache.layers[i].keys/.values`이며 shape는 `[1,3,S_i,64]`다. `DynamicLayer.update`는 sequence axis -2로 concat하고 `get_seq_length()` 기본값은 layer0 physical length를 읽는다. 이 DynamicLayer에는 Qwen 4.49의 `_seen_tokens` 같은 별도 누적 counter 가정이 없다. 설치 Llama forward에는 공식 `cache_position` 인자가 없다.

권고는 full Layer0 실행 후 **H1·mask·original position IDs뿐 아니라 Layer0 KV도 동일 keep positions로 gather**하는 것이다. upper layers를 짧은 입력으로 채우면 모든30층 physical cache length가 L'=T+K로 같아진다. RoPE 좌표는 original positions를 유지하고 decode position은 original L, L+1,…를 명시한다. physical length L'와 logical position L은 다르다.

주의할 실제 코드 경계:

1. suffix prefill의 mask를 만들 때 layer0만 채워진 cache를 기본 layer0 기준 past로 넘기면, upper layer의 빈 cache와 어긋나는 2L' 길이가 생길 수 있다. cache 없는 suffix 기준 mask 또는 검증된 explicit 4D causal mask가 필요하다.
2. gappy position IDs와 `attention_mask=None`는 packed-sequence 탐지에서 position 차이를 경계로 오인할 수 있다. explicit mask를 생략하지 않는다.
3. native `generate()`의 기본 position 추론은 attention mask/cache 길이를 사용한다. explicit position_ids를 공급하면 마지막 값+1로 연장할 수 있지만, 중간 pruning으로 바뀐 prompt·mask·cache·position 상태를 자동으로 함께 갱신해 주지는 않는다. `prepare_inputs_for_generation`/`GenerationMixin`과의 별도 coordination 없이 pruned generate 지원을 주장하지 않는다.
4. 기존 `build_pruned_inputs`와 `GenerationAudit`는 pre-LLM deletion·contiguous reindex·fresh short prompt 전제다. 새 layer0/cache 감사에 직접 재사용할 수 없다.

상세 실행 계약은 [LAYER0_ERC_FEASIBILITY.md](LAYER0_ERC_FEASIBILITY.md)에 둔다.

## 7. 주요 근거 파일·함수

설치 소스의 공통 루트는 `<HOME>/miniconda3/envs/vlm_project/Lib/site-packages/transformers/`다. 줄 번호는 현재 로컬 파일 기준이며 실행 검증을 뜻하지 않는다.

| 파일 | 확인한 함수/지점 |
|---|---|
| `models/idefics3/modeling_idefics3.py` | `Idefics3Connector.pixel_shuffle:397`, `Idefics3Model.get_image_features`, `inputs_merger`, `forward:660`의 DynamicCache/text_model 경로 |
| `models/idefics3/image_processing_idefics3.py` | `resize:233`, `split_images:260`, `resize_for_vision_encoder:319` |
| `models/idefics3/processing_idefics3.py` | image prompt expansion, row/col/global marker 및 image_seq_len 사용 |
| `models/llama/modeling_llama.py` | `LlamaAttention.forward:259`, `LlamaModel.forward:386–400`, decoder residual/rotary/cache update |
| `cache_utils.py` | `DynamicLayer.update:143`, `DynamicCache`, per-layer `get_seq_length`/mask size |
| `masking_utils.py` | `find_packed_sequence_indices:728`, `create_causal_mask`의 position/mask/cache 조건:854 |
| `generation/utils.py` | `prepare_inputs_for_generation`, position/mask/cache length 갱신, legacy-only cache_position 처리:623 |
| `examples/run_small_vlm.py::main` | 기존 FP16/eager inference 및 표준 chat template |
| `examples/inspect_visual_tokens.py::main` | projected64/view와 R/C 관측 설계 |
| `examples/inspect_llm_input.py::main` | merger 및 placeholder→view 대응 관측 설계 |
| `src/inference/pruned_vlm.py::build_pruned_inputs` | 기존 pre-LLM pruning 및 compact positions |
| `src/inference/generation_audit.py::GenerationAudit` | 기존 contiguous-position/fresh-cache 감사 |

공식 [SmolVLM-256M 모델 카드](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)는 Idefics3 계열 및 512×512 view당64 visual tokens라는 설명을 뒷받침한다. 현재 배포의 주장이 로컬 실행의 메모리 보장을 대신하지는 않는다. 이 감사의 세부 API 판단은 최신 웹 예제가 아니라 위 설치 소스와 고정 snapshot에 근거한다.
