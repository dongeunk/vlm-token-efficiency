> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s3a/phase_s3a_20261114_01/RUN_RECORD_SCHEMA.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Phase S3 run record v1 — CPU 평가 입력 계약

구현: `src/smol_s3_eval/contracts.py`, `scoring.py`. 신규 GPU writer는 아직 없다. 아래 필드는 실제 실행에서 채워야 하며, CPU mock이 실제 측정을 대신하지 않는다.

## 문항·방법 단위 JSONL

기본 필드는 `schema_version="smol_s3_run_v1"`, split(pilot/main), image_id, question_id, method(full/s2_l/s2_u/s2_c), status다. 성공만 status=ok다. 오류는 별도 error_type/message/trace/attempt ledger에 보존한다. 하나라도 오류·누락·중복·외부 키가 있으면 scorer는 `IncompleteEvaluation`을 발생시키고 전체 통계 생성을 중단한다.

`protocol_sha256`, `selection_lock_sha256`, `model_revision`, `image_sha256`, `prompt_sha256`, `source_record_sha256`, label을 고정 manifest와 대조한다. `input_ids_sha256`와 `pixel_values_sha256`는 방법 간 같아야 한다. 실제 dtype/shape/attention backend와 processor provenance도 run header에 기록한다. tensor digest 규칙은 dtype+shape+contiguous raw bytes를 명시해 S3-B writer에서 고정한다. 현재 CPU validator는 기록된 hash의 형식·일치만 확인하며 GPU tensor 내용 자체를 관측하지 않는다.

`raw_text`는 원문 그대로 저장한다. `generated_token_ids`는 EOS 포함1~16개. tokenizer decode 설정은 skip_special_tokens=True, clean_up_tokenization_spaces=False로 S3-B에서 고정·검증하고 raw IDs를 기준 증거로 남긴다. 정답과 질문은 parser에 넘기지 않는다.

## sequence object

| 필드 | 내용·검증 |
|---|---|
| rows, cols | 실제 processor 출력 R/C; width/height 예상값 대입 금지 |
| local_tokens/global_tokens/visual_tokens | 64RC / 64 / 64(RC+1), placeholder·projector·view 수 교차 검증 |
| original_sequence_length | 실제 전체 L |
| input_visual_positions | 원본 input_ids에서 ID49190인 오름차순 위치 N개 |
| nonvisual_positions / nonvisual_tokens | 위 위치의 complement와 개수 T |
| selected_visual_indices | native visual ordinal; sorted/unique/range; Full=N, pruning=64 |
| kept_sequence_positions | 모든 nonvisual + 선택 visual의 원래 sequence 위치, 원래 순서 |
| original_position_ids_after_prune | keep positions 그대로; 0..L'−1로 재번호화 금지 |
| pruned_sequence_length | Full=L, pruning=T+64 |
| layer0_input_length / suffix_input_length | L / L' |
| suffix_attention_mask | batch1 unpadded all1, 길이L'; 실제 causal mask 검증은 runtime 증거 필요 |
| nonvisual_preserved / original_last_token_preserved | true, 실제 gather 결과도 점검 |

coarse capacities는 stitched(8R×8C)의 4×4 region에서 각4RC이다. q_local[16]은 sum48, 각1..capacity이고 Uniform은 모두3이다. 최종 local native ordinal이 속한 region별 count가 q와 일치해야 한다. global64 중16개를 공통 global recursive Cell/ERC로 뽑는다. 같은 input에서 L/U/C의 global16은 exact 동일해야 한다. Full에는 `local_budgets=null`, pruning에는16개 정수 배열을 기록한다.

## runtime object

- `prefill_cache_lengths`: 실제30층별 `[L']*30`. Layer0도 full 입력 후 KV를 gather한 길이여야 한다.
- `final_cache_lengths`: emitted token 수 m일 때 `[L'+m−1]*30`. 마지막 출력 token 자체는 다음 forward에 넣지 않는다.
- `append_counts`: `[m−1]*30`.
- `decode_position_ids`: 원래 `[L, L+1, ..., L+m−2]`; physical L' 기준 좌표 금지.
- `causal_mask_validated=true`: gappy1D RoPE를 segment 경계로 오인하지 않는 실제 mask 증거를 첨부한다.

`erc_finite=true`는 pruning에 필수다. 실제 S3-B에서는 layer0 입력/출력 shape·native FP16 subtraction→FP32 norm·finite count=N도 별도 evidence에 기록한다. CPU `mock_cache_prune_append`는 Python list로 sequence-axis gather/append 계약을 검사할 뿐 HF DynamicCache 검증이 아니다.

## 선택·Coverage·측정 evidence

기본 scorer가 검증하는 필드는 local_budgets, selected_visual_indices, 실제 위치/길이/캐시, erc_finite와 measurement다. 향후 GPU writer는 아래 추가 scientific evidence를 artifact SHA/reference로 보존하고 Gate A/C에서 완전성을 검사한다.

- 공통 full Layer0/ERC summary, Cell ID·region·rectangle·membership 및 cell별 selected native ordinal, tie rule.
- local/global 구분, native↔lattice 대응, nominal 원본 normalized box. feature receptive field의 독립적 위치라고 주장하지 않는다.
- C의 demand=all, candidates=local, 임시32개 greedy 선택·marginal gain·region histogram; L의 complexity/extra integer allocation, U의 q=3.
- 공통 FP32 calibration으로 최종 F_local/F_all 및 normalized coverage. 이 추가 feature 진단은 이번 S3-A에서 실제 feature 없이 측정하지 않았고 기본 scorer가 계산하지 않는다.
- `measurement`: ttft_e2e_ms, ttft_decoder_ms, selection_overhead_ms, peak_allocated_mib, peak_reserved_mib. finite/nonnegative, reserved>=allocated. 실제 event 경계·synchronize/peak reset 여부는 run header/trace로 보존한다.

`score_complete()`는 paired question rows, 이미지별8개 count vector(TP,FP,FN,TN,invalid_yes,invalid_no,strict_correct,strict_invalid), 방법별 지표·지연/메모리 요약·mean region budgets를 만든다. 4방법6쌍의 전체/local/global intersection/Jaccard도 계산한다. ERC-selected 위치와 token overlap은 최종 native ordinal을 기준으로 하며 query별로 기록한다.

## CPU 분석 사용법

모듈은 모델/runtime를 import하지 않는다. 아래는 **후속 완전한 저장 결과가 있을 때** 사용하는 CPU 명령이며 이번에는 실제 결과 파일에 실행하지 않았다.

```powershell
& '<HOME>/miniconda3/envs/vlm_project/python.exe' -B examples/analyze_smol_s3_cpu.py --split pilot --records <future_records.jsonl> --output results/smolvlm_s2_coverage_phase_s3a/phase_s3a_20261114_01/future_cpu_analysis_01
```

현재 CLI의 write boundary는 S3-A 결과 폴더 내부의 새 하위 폴더다. S3-B 전용 결과 폴더를 사용할 다음 runner는 별도로 작성한다. CPU 분석 라이브러리는 재사용할 수 있다. schema/runtime generalization과 실제 tensor 검증은 그 단계의 검토 대상이다.
