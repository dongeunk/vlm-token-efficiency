> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s3a/phase_s3a_20261114_01/PHASE_S3B_GPU_GATE_AND_PILOT_PLAN.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Phase S3-B GPU 기능 Gate 및 pilot 계획 — 이번에는 실행하지 않음

현재 승인된 S3-A 범위는 CPU 준비까지다. 아래 GPU 실행에는 후속 사용자 지시가 필요하다. main은 Gate C 검토 후 다시 별도 승인받는다. 로컬 GTX1660 SUPER6GB, FP16 모델1개, cached checkpoint만 순차 사용한다. 다운로드·원격/유료 GPU·병렬 모델은 사용하지 않는다.

## 선행 구현·진입 조건

S2-B `runtime.py::_validate`는 R3/C4/N832/L878/T46/Q1153에 고정되어 있고 ERC shape832, boundary N832/local768, 최종L110도 검사한다. 기존 코드를 수정하거나 제한 상수만 제거하지 않는다. 별도 S3 Runtime/runner에서 실제 metadata 기반 N/R/C/T/L/keep/mask/positions/cache를 처리해야 한다. S1 allocation/Cell/ERC와 S2-B native ERC 수식·cache gather 원칙은 그대로 재사용한다.

신규 runner는 모델 로드 전에 CPU manifest/protocol/source SHA, expected run keys, old110 disjoint, image SHA, annotation3yes/3no를 확인한다. 실제 processor output으로 R/C·view 수·placeholder count를 대조하고, 예상과 다르면 기록 후 중단한다. 새 image로 교체하지 않는다. weights 및 cached config는 변경하지 않는다. processor/attention/backend/FP16, decoding의 inactive config, tokenizer decode 옵션도 review한다.

## Gate A — 다양한 입력 기능 검증

고정 pilot anchors image554002(예상832), image134075(예상1088)를 사용한다. 각 이미지의 고정된 공식6문항을 **모델 생성 전에** processor로 입력화하고 실제 T가 가장 짧은 질문과 가장 긴 질문을 고른다. tie는 낮은qid다. 이것은 token 길이 기능 검사 선정이며 정답·모델 출력에 따라 고르는 절차가 아니다. 모든6문항은 이후 pilot에서도 유지한다. 이미지당 T가 다르지 않으면 길이 변화 조건은 미검증으로 표시하고 Gate B를 시작하지 않는다.

목표 2layouts×2질문길이=최대4입력, 입력당 Native Full / Custom Full / No-op / S2-L / S2-U / S2-C의 최대6 generation, **총24 generation 상한**을 제안한다. 같은 checkpoint를 한 번 로드해 순차 실행하고, 모든 route에서 greedy16·fresh KV·동일 inputs를 사용한다. 별도 재시도/warm-up generation은 포함하지 않는다. 이는 후속 승인 범위에 명시할 제안이며 현재 실행 권한이 아니다.

입력별 순서는 Native Full→Custom Full→No-op→L→U→C다. 앞 route 기능 실패 시 다음 route/입력을 실행하지 않는다.

1. 실제832는 local768+global64, 실제1088은 local1024+global64인지 확인한다. processor 관측이 해당 구조를 만들지 않으면 해당 layout은 미검증이다. 필요하다면 old80의436127/N1088 과거 CPU 관측을 기능 reference로만 참고할 수 있으나, 승인 없이 추가 generation하거나 새 평가 표본을 교체하지 않는다.
2. vision/projector·full Layer0 input/output·prefill logits·generated IDs의 Native/Custom/No-op parity를 입력별 확인한다. S2-A의 단일832 결과를 다른 layout의 근거로 대신하지 않는다. 임의 tolerance 확대 금지, 최초 차이의 위치/최대오차를 기록하고 중단한다.
3. L/U/C는 공통 full Layer0에서 모든N visual 처리, ERC finite N개, local48/global16, cell별1개 및 원래 sorted unique ordinal을 확인한다.
4. 실제 text/special complement 전체 보존 및 L'=L−N+64=T+64. 예: T46,N832일 때만878→110이며 다른 질문/이미지에110을 강제하지 않는다.
5. Layer0 KV도 keep positions gather, 30층 short prefill, gappy original1D RoPE·causal mask, physical length와 logical positions 분리, m emitted tokens에 m−1 append를 실제 cache에서 확인한다.
6. 같은 입력의 L/U/C global16·preselection visual features/full Layer0/ERC가 같은지 확인한다. Full과 pruned answer의 일치는 통과 요건이 아니다.
7. GPU 실행 전후 상태 및 각 route peak allocated/reserved를 기록한다. NaN·OOM·mask/mapping/cache/decode 오류·Full parity 불일치가 발생하면 즉시 중단한다. batch/해상도/cache를 임의로 바꾸어 재시도하지 않는다.

신규1088 구조와 다른T에 대한 실제 GPU 정합성은 **현재 미검증**이다. CPU에서는832/1088의 합성 input IDs/positions/list cache와 저장된 실제832 input metadata만 검증했다.

## Gate B — 고정 pilot10 × 6 × 4

Gate A 전부 통과 후 `pilot_questions.jsonl`과 `pilot_expected_run_keys.jsonl` 순서대로 Full/L/U/C를 실행한다. 최대240 generation, 오류 시 중단하고 partial ledger를 남긴다. Gate A output을 재사용하여 pilot 항목을 건너뛰거나 데이터를 바꾸지 않는다. fixed input·prompt·parser·K64·region16·local48/global16은 그대로다.

각 레코드는 `RUN_RECORD_SCHEMA.md`를 충족해야 한다. raw IDs/text/EOS, actual N/R/C/T/L/L', 30층cache/append/positions, budgets/selection/Cell/ERC, measurements와 input/config hash를 보존한다. 실제 GPU writer와 tensor hash 정의는 아직 구현하지 않았다. 이 phase에서 해당 구현·source lock review가 필요하다.

## Gate C — CPU 집계·운영 검토

240 expected keys 전부 존재하고 status=ok여야 한다. 누락/중복/오류를 제외한 paired subset으로 지표를 만들어 통과시키지 않는다. parser invalid는 남겨 전체 분모로 계산한다. invalid 비율, strict/semantic 차이, truncation/EOS, Accuracy/Precision/Recall/F1, C−L image bootstrap10000/seed42/95%CI를 함께 검토한다.

예산·선택 위치·global 공통성·token overlap·feature coverage artifacts·sequence/cache 검사, TTFT/selection·VRAM을 점검한다. F1이 기대보다 낮거나 S2-C가 L보다 낮다는 이유로 표본·parser·budget을 바꾸지 않는다. 기능 오류와 과학적 결과를 구분한다. parser/runtime를 고쳐야 하면 버전·이유·실패 evidence를 남기고 pilot을 새로운 실행으로 분리한다. main 결과를 보고 유리한 설정을 선택하지 않는다.

## Gate D — main 별도 승인 및 추가 layout

Gate C의 완전한 결과를 사용자에게 보고한 후 **200장 본 평가를 별도 승인받는다**. main1200문항×4=4800 generation이며 이번에는 시작하지 않는다.

main358255는 정적 예상576(local512+global64,2×4)이다. 데이터는 고정 상태로 유지한다. main 시작 전 실제 processor 확인과 별도 기능 Gate가 필요하다. 동일6routes×짧은/긴2질문=최대12 generation을 추가 기능 검사안으로 제시한다. CPU 계약은576도 검증했지만 실제576 GPU pruning·native parity는 미검증이다. 실제로 다른N/R/C가 나오거나 지원되지 않으면 main 전체를 대기하고 구현/승인 범위를 다시 검토한다. 조용한 제외나 대체는 금지한다.

Gate A/B/C/D의 완료 상태·중단 사유·generation ledger를 분리한다. 현 상태는 A 미실행/B 미실행/C 실험 결과 없음/D 미승인이다.
