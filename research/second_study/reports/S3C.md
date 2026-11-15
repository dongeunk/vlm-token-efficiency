> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s3c/phase_s3c_20261115_01/REPORT.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# SmolVLM S²Prune + Coverage — Phase S3-C

작성(KST): 2026-11-15T09:20:22.445136+09:00. **PASS: 고정200이미지·1,200문항·4방법, 4,800/4,800 Generation 완료.** 모델 로드 1회, 20개 구간(각10장·240개)을 안전하게 저장했다. 중복·누락·Runtime Error·NaN·OOM은0개다. 설정·표본 변경과 추가 Generation은 없으며, 전부 완성·무결성 확인 후에만 본 평가 지표를 계산했다. 후속 실험은 실행하지 않고 여기서 중단한다.

## 성능 및 사전 고정된 주요 비교

고정된 공식 POPE Random **200이미지 부분집합** 결과다. Frozen semantic parser/metric을 사용했으며, 공식 전체 POPE benchmark 점수와 같지 않다. 단위는%다.

|방법|Semantic Accuracy|Precision|Recall|Semantic F1|Strict Accuracy|Invalid|EOS|No-EOS at16|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|full|83.333|93.825|78.500|85.481|56.417|78|1162|38|
|s2_l|80.917|92.323|76.167|83.470|64.583|54|1148|52|
|s2_u|80.583|92.464|75.667|83.226|64.250|55|1148|52|
|s2_c|80.250|92.418|75.167|82.904|64.083|57|1147|53|

**S2-C − S2-L Semantic F1: -0.566%p; image-paired bootstrap95% CI [-1.575, +0.432]%p.**

The precommitted 95% CI includes zero; a robust C-vs-L Semantic F1 advantage is not established by this evaluation.

200개 이미지 단위로6문항을 함께 재표집했다. 10,000회, PCG64 seed42, percentile2.5/97.5(linear), confusion 합계로micro F1을 재계산했다. image F1 평균이 아니다. Accuracy·Invalid·Strict Accuracy의 paired CI도 `analysis_01/PAIRED_BOOTSTRAP_10000.json`에 있다. 보조 비교는 다중비교 보정 없는 exploratory 결과이며, 그 순위로 알고리즘·평가 조건을 변경하지 않았다.

|비교|Semantic F1 차이 %p|95%CI %p|
|---|---:|---|
|s2_c-s2_l|-0.566|[-1.575, +0.432]|
|s2_c-s2_u|-0.322|[-1.097, +0.442]|
|s2_l-s2_u|+0.244|[-0.634, +1.122]|
|s2_c-full|-2.577|[-5.049, -0.065]|
|s2_l-full|-2.011|[-4.382, +0.414]|
|s2_u-full|-2.255|[-4.671, +0.212]|

## Confusion, Invalid, 종료 진단

|방법|TP|FP|FN|TN|invalid_yes|invalid_no|Yes Accuracy %|No Accuracy %|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|full|471|31|91|529|38|40|78.500|88.167|
|s2_l|457|38|137|514|6|48|76.167|85.667|
|s2_u|454|37|141|513|5|50|75.667|85.500|
|s2_c|451|37|143|512|6|51|75.167|85.333|

Invalid는1,200문항 분모에 유지하고 Accuracy에서 오답으로 처리했다. Positive F1 분모는 `2TP+FP+FN+invalid_yes`, invalid_no는TN이 아니다. 따라서 F1만으로 전체 오류 개선을 판단하지 않는다. Strict와Semantic은 서로 다른 frozen parser 결과다. EOS 없이16토큰 생성된 경우는 **possible truncation**이며 실제 의미적 절단을 확정하지 않는다. 원시 응답·생성ID·EOS·invalid rule을 보존했고 재생성·후처리 규칙 변경·유리한 응답 제외는 없다.

`per_question.jsonl`, `per_image.jsonl`, `IMAGE_PAIRED_DISCORDANCE.jsonl`, `PAIRED_RESPONSE_DIFFERENCES.jsonl`은 모든6개 비교의 paired 결과, 유효 Yes/No 반전, Invalid↔Valid, EOS/Truncation, correctness 변화를 구분한다. `ERROR_MISSING_TRUNCATION.json`에 완전성과 방법별 종료 진단이 있다.

## Allocation → ERC → 답변 변화

C/L 예산이 다른 이미지는 200/200장, Local budget L1 차이는 평균 29.510, 중앙값 30.000다. 문항 단위 최종 selected token 변경은 1200/1,200개, 유효 Yes/No 반전은 21개다. 정답상태 delta 분포는 `{"0": 1170, "1": 11, "-1": 19}`, Invalid 전환은 `{"both_valid": 1139, "both_invalid": 50, "left_valid_right_invalid": 7, "left_invalid_right_valid": 4}`다. 이는 할당량·최종 토큰 변화가 답변 변화로 이어지는 빈도를 직접 보여주며 인과 기전을 입증하지는 않는다.

최종64개에 동일한 projector-space FP32 calibrated cosine, row mean-centering, nonnegative affinity를 적용해F(S)를 계산했다. C/L Coverage 증가 678문항, 감소 522문항이다. C의 임시 Coverage32개 중 최종 Local48에 남은 개수는 평균 2.625/32다. 최종 ERC64의F가 임시32보다 낮은 경우는 1200/1,200개다. 임시 Coverage 선택은 지역 예산 산정용이고 실제 보존 토큰은 셀별ERC이므로 두 집합은 같을 필요가 없다.

이미지200개 단위 ΔCoverage/N과ΔAccuracy의 Pearson=-0.07656949098849228, Spearman=-0.094474100958092이다. 관찰적 상관이며, Coverage 차이를 정확도 차이의 원인으로 단정하지 않는다. 동일 이미지6문항의 Allocation/임시32 질문 독립성 관측: True. 모든 Allocation·ERC는 문항마다 재계산했고 캐시 최적화는 없다.

`REGION_BUDGET_ERC_DIAGNOSTICS.jsonl`, `IMAGE_REGION_BUDGET_DIFFERENCES.jsonl`, `ERC_SELECTED_TOKEN_POSITIONS.jsonl`, `COVERAGE_TEMPORARY_VS_FINAL_ERC.jsonl`, `FEATURE_COVERAGE.csv`, `token_overlap.jsonl`, `MECHANISM_SUMMARY.json`에 원시 예산·Laplacian·greedy32/trace·ERC·cell·토큰 위치·overlap을 보존했다. `TOKEN_STRUCTURE_METRICS.json`은 실제576/832/1088별 분석이다. 이미지 수는1/172/27이며576은 단1장이므로 구조별 일반화를 하지 않는다.

## Runtime·시간·VRAM

실제 입력은576토큰1장,832토큰172장,1088토큰27장이다. 첫 전체Layer0 후 Local48+Global16, 모든 Nonvisual 보존, 원래1D RoPE position과 물리적으로 짧아진 cache를 분리했다. 모든 문항에서 동일한 input tensor hash, Native/projector/Layer0 hidden 및 실제Layer0 KV gather exact, finite ERC·공통global16·셀별 최대 선택·30층decode append를 확인했다. Full 경로는Native HF, Pruning은 기존S3-B1/S3-B2 연산을 그대로 호출했다.

GPU worker 총 32223.845초. Peak Allocated 2402.018MiB / Reserved 3264.000MiB. GPU 전후 상태와 작업 종료 후CLI를 저장했다. Reserved는 앞선 입력의allocator cache를 포함할 수 있다.

아래는 관측 방식별 중앙값이다. E2E TTFT는JPEG 열기/전처리 시작부터 첫greedy argmax까지, Decoder TTFT는merged H0 준비부터다. 동기화·CPU 증거 전송·audit가 포함되고 방법 순서도 고정되어 있으므로 안정적인 속도 우위나 실서비스TTFT benchmark로 해석하지 않는다. 출력 길이가 달라Generation wall time도 직접적인 계산량 비교가 아니다.

|Audit|방법|n|E2E TTFT ms|Decoder TTFT ms|Selection ms|Generation wall s|
|---|---|---:|---:|---:|---:|---:|
|full_gate_b_audit|full|2|6560.0|944.4|0.0|7.679|
|full_gate_b_audit|s2_l|2|5779.4|428.0|62.9|6.947|
|full_gate_b_audit|s2_u|2|5898.5|412.0|43.4|7.129|
|full_gate_b_audit|s2_c|2|5961.0|460.0|92.1|7.126|
|main_light_audit|full|1198|5813.6|529.6|0.0|6.519|
|main_light_audit|s2_l|1198|5399.8|212.6|47.8|6.146|
|main_light_audit|s2_u|1198|5396.1|208.4|43.6|6.142|
|main_light_audit|s2_c|1198|5442.7|255.4|90.3|6.187|

Layer0/suffix prefill/Allocation/Coverage calibration/greedy 시간과 Peak memory 전체 원문은 `GPU_LATENCY_VRAM.csv`, 집계는 `LATENCY_SUMMARY.json`에 있다.

## 보존·검증 및 한계

CPU 테스트 **98/98 PASS**: 모델/CUDA 없이 ledger·구간·재개·metadata adapter·frozen parser/metric/bootstrap·분석oracle을 검증했다. 그중 실제 모델 정확도 주장은 없으며 성능은 본 평가4,800개 GPU 결과에만 근거한다. 기존 bounded 66개 source/report/lock SHA 및checkpoint stat은 변하지 않았다. 모델weights 전체SHA나프로젝트전수검사는 하지 않았다. 기존S0/S1/S2/S3-A/B1/B2,1차연구,Qwen연구와frozen manifest/parser/runtime/allocator는 수정하지 않았다. 신규파일·함수·SHA는 `CODE_MANIFEST.json`, 최종artifacts는 `FILES_SHA256.json`에 있다.

이 결과는SmolVLM-256M 로컬이식 실험이며Qwen2.5-VL-7B 원본S²Prune 재현이 아니다. 단일 checkpoint/데이터 부분집합/greedy16과frozen semantic parser, query별 반복을 갖는 한 번의실행이다. Bootstrap은이미지표본불확실성을 다루지만실행간수치변동이나parser모호성을 모두 반영하지 않는다. 모델default config의기존pad-token 경고는 실제고정generation EOS49279/PAD2 변경없이 기록했으며, Runtime 실패와 구분했다.

S²Prune license HOLD 유지. 다운로드·설치/업그레이드·외부/유료GPU·Git commit/push 없음. **Phase S3-C에서 중단하며 후속 알고리즘 개선·Ablation은 실행하지 않는다.**
