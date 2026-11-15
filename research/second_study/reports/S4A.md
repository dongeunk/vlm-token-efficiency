> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s4a/phase_s4a_20261115_01/REPORT.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# SmolVLM S²Prune + Coverage — Phase S4-A CPU 원인 분석

**결론: Coverage 배분의 성능 우위는 확인되지 않았다.** C−L Semantic F1은 −0.566%p, 기존 image-paired bootstrap 95% CI는 [−1.575, +0.432]%p다. 점추정은 음수지만 CI가 0을 포함하므로 L 대비 열세를 확정하지도 않는다. 관측상 가장 우선할 검증 가설은 **임시 Coverage 선택의 목적이 지역 예산→셀별 ERC 선택으로 충분히 전달되지 않았을 가능성**이다. 아래 수치는 원인을 확정하는 반사실 실험이 아니다.

분석 범위는 고정 200이미지·1,200문항의 S3-C 저장 결과다. 기존 Semantic 판정과 통계를 그대로 사용했다. **새 Generation·모델 로드·GPU/CUDA 초기화·알고리즘/Parser 변경·Bootstrap 재실행은 0회**다. Python 표준 라이브러리로 선택된 결과 파일만 집계했고 SHA 재검사는 하지 않았다. 예산·토큰 집합·Coverage는 방법별로 같은 이미지의 6질문에서 동일했다(각 방법 200/200장). 따라서 1,200개의 반복된 선택 결과를 독립 표본으로 해석하지 않는다.

## 1. 지역 예산은 얼마나 다른가?

Local48, Global16, Local 4×4 조건은 동일하다. L1은 16개 지역 예산 차이의 절댓값 합이다.

|비교|예산이 다른 이미지|평균 L1|중앙값 L1|평균 변경 지역 / 16|
|---|---:|---:|---:|---:|
|C−L|200/200|29.51|30|13.10|
|C−U|200/200|15.86|16|11.15|
|L−U|200/200|24.23|24|13.13|

C/L의 L1 범위는 14–48, IQR은 26–34다. 동일 총예산에서 이동한 지역 예산량은 L1/2이므로 평균 **14.755개**다(Local48의 30.74%, 추가32의 46.11%). 이는 지역 예산량의 재배분이며 특정 객체 토큰의 보존율은 아니다.

이미지 200장 평균 예산의 4×4 배열(행·열은 0부터, 행 우선 순서):

|Coarse 행|S2-L: 열 0–3|S2-C: 열 0–3|
|---|---|---|
|0|[2.13, 2.51, 2.73, 2.39]|[2.89, 2.94, 3.01, 2.80]|
|1|[2.85, 3.63, 3.89, 3.04]|[2.90, 3.12, 3.04, 2.73]|
|2|[3.15, 3.89, 3.83, 2.88]|[3.04, 3.04, 3.06, 2.98]|
|3|[2.40, 2.98, 3.06, 2.63]|[3.00, 3.35, 3.24, 2.87]|

U는 모든 지역이 3이다. C는 예산 거리상 L보다 U에 가깝고, L의 평균적인 중앙 지역 집중은 C에서 약해진다. **어떤 지역이 정답 객체를 포함하는지는 미측정**이므로 이를 배경/객체 할당의 우열로 해석하지 않는다.

## 2. 최종 ERC-selected token overlap

아래는 1,200문항의 평균이다. 모든 pruning 비교에서 Global16의 교집합은 16으로 동일하므로 Local만의 overlap을 함께 제시한다.

|비교|전체 교집합 / 64|Local 교집합 / 48|Local 공통 비율|전체 Jaccard|Local Jaccard|
|---|---:|---:|---:|---:|---:|
|C vs L|49.245|33.245|69.26%|0.6274|0.5329|
|C vs U|56.070|40.070|83.48%|0.7805|0.7180|
|L vs U|51.885|35.885|74.76%|0.6845|0.6013|

C/L 전체 교집합 범위는 40–57개다. **모든 C/L 문항에서 전체 교집합 = 64 − L1/2, Local 교집합 = 48 − L1/2**가 정확히 성립했다. C/U와 L/U에서도 각 1,200/1,200건이 일치했다. 이번 저장 결과에서 토큰 교체 수는 지역 예산 이동량과 정확히 대응한다. 이는 관측된 관계이며 일반적인 Cell Partition의 수학적 성질을 새로 증명한 것은 아니다.

## 3. Coverage 임시32 → 최종 ERC64

임시32는 Local 후보에서 뽑아 지역 예산의 개수를 정하는 용도다. 최종 선택은 공통 셀별 ERC이며 Global16을 별도로 포함한다.

|관측량|결과|
|---|---:|
|임시32 ∩ 최종 Local48|평균 2.625개, 중앙값 3, 범위 0–7|
|임시32 중 최종 보존 비율|8.20%|
|임시32와 최종 Local48의 평균 Jaccard|0.0343|
|교집합 0|78/1,200문항 = 13/200이미지|
|평균 F(임시32) → F(최종 C64)|140.979 → 122.730|
|문항별 F(최종 C64)/F(임시32)의 평균|87.15%|
|최종 C64의 F가 임시32보다 낮음|1,200/1,200문항 = 200/200이미지|

대표 토큰의 직접 보존과 Coverage 목적값은 이 전달 단계에서 약해졌다. 두 집합은 포함 관계가 없으므로 토큰 수가 32→64로 증가해도 F가 반드시 증가하는 것은 아니다. **이 현상이 F1 저하를 일으켰는지는 미측정**이다.

## 4. 최종 Feature Coverage F(S)는 정확도와 맞는가?

같은 projector-space FP32 cosine, 행 평균 중심화, nonnegative clipping, Local+Global demand/단위 가중치를 사용한 기존 F(S)다. N은 원본 Visual Token 수다. 아래는 이미지별 동일 가중 평균이며 F/F(Full)은 각 이미지 비율의 평균이다.

|방법|평균 F(S)|평균 F(S)/N|평균 F(S)/F(Full)|
|---|---:|---:|---:|
|Full|219.833|0.254266|100.000%|
|S2-L|122.064|0.141386|55.607%|
|S2-U|121.951|0.141238|55.547%|
|S2-C|122.730|0.142149|55.910%|

C−L의 평균 ΔF/N은 **+0.000763**이다. C가 높은 경우는 113/200장(678문항), 낮은 경우는 87/200장(522문항)이다.

|C/L 정답 상태|문항 수|C의 Coverage 높음 / 낮음|평균 ΔF/N|
|---|---:|---:|---:|
|C만 정답|11|4 / 7|-0.000861|
|L만 정답|19|11 / 8|0.001320|
|정답 상태 동일|1170|663 / 507|0.000769|

C가 정답을 얻은 11문항 중 7문항은 Coverage가 낮아졌고, 정답을 잃은 19문항 중 11문항은 Coverage가 높아졌다. 이미지 단위 ΔCoverage/N–ΔAccuracy의 Pearson은 **−0.0766**, Spearman은 **−0.0945**다. 이번 결과에서 Feature Coverage 증가와 정답 개선의 양의 정렬은 관측되지 않았다. 상관으로 인과를 판단하거나 유리한 문항만 선정하지 않았다.

## 5. 정답 변화·Invalid·실제 Yes/No 반전

“C만/L만 정답”은 **C와 L 두 방법 사이의 비교**다.

|L → C 변화|문항|C가 정답 획득|C가 정답 상실|정답 상태 동일|
|---|---:|---:|---:|---:|
|유효 Yes/No 반전|21|9|12|0|
|Invalid → Valid|4|2|0|2|
|Valid → Invalid|7|0|7|0|
|유효 판정 동일 또는 둘 다 Invalid|1,168|0|0|1,168|
|합계|1,200|11|19|1,170|

둘 다 정답 952문항, 둘 다 오답/Invalid 218문항이다. 정답 변화 30문항은 29이미지에서 발생했다. 순손실 8문항으로 C−L Accuracy는 −0.667%p다. 유효 반전에서 GT=Yes는 No→Yes 6건 개선/Yes→No 10건 악화, GT=No는 Yes→No 3건 개선/No→Yes 2건 악화였다. Confusion 차이는 TP −6, FP −1, FN +6, TN −2, invalid_yes 0, invalid_no +3이다.

**유효 반전 21건은 양쪽 모두 EOS 종료**였다. Valid→Invalid 7건 중 5건은 C의 No-EOS at16, 2건은 EOS가 있었다. EOS/possible-truncation 플래그가 달라진 것은 각각 9건이다. 따라서 생성 길이 문제는 일부 경로의 후보 원인이며 유효 Yes/No 반전 전체를 설명하지는 못한다. 16 이후 답변이 회복되는지는 **미측정**이다. Invalid는 기존 기준대로 오답이며 재분류하지 않았다.

전체 discordant 문항 ID 목록(형식: 이미지 ID/질문 ID, 누락·사례 선별 없음):

|정답 방법|유형|수|ID|
|---|---|---:|---|
|C만 정답|유효 Yes/No 반전|9|31773/234, 93948/1860, 303652/1277, 203479/771, 505542/1361, 536073/2755, 12333/294, 379404/1963, 516916/133|
|C만 정답|Invalid→Valid|2|313034/1297, 249720/1042|
|L만 정답|유효 Yes/No 반전|12|222118/2619, 159969/2831, 560744/2859, 307262/2409, 217397/335, 328301/2403, 564336/437, 318204/485, 44801/1970, 551908/919, 246928/2513, 506483/996|
|L만 정답|Valid→Invalid|7|316700/1104, 203479/770, 429033/1322, 332025/477, 235541/2267, 500257/235, 283412/34|

GT·원시 응답·Semantic rule·종료 플래그는 원본 `per_question.jsonl`, 문항별 예산 L1·토큰 교집합·ΔF는 `PAIRED_RESPONSE_DIFFERENCES.jsonl`에서 위 ID로 확인할 수 있다.

## 6. Local Token 수별 차이

모든 구조에 Local48 + Global16을 유지했다. F1 단위는 %, Δ는 %p다.

|Local / 전체 Visual|이미지 / 문항|Full F1|L F1|U F1|C F1|C−L F1|C만 / L만 정답|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|512 / 576|1 / 6|85.71|75.00|75.00|75.00|0.000|0 / 0|
|768 / 832|172 / 1032|85.59|83.65|83.62|83.24|-0.407|10 / 16|
|1024 / 1088|27 / 162|84.72|82.76|81.12|81.12|-1.640|1 / 3|

|Local 원본 수|Local48 보존율|평균 C/L 예산 L1|Local 교집합|임시32 최종 보존|평균 C−L ΔF/N|
|---|---:|---:|---:|---:|---:|
|512|9.375%|24.00|36.00 / 48|4.00 / 32|0.005147|
|768|6.250%|29.59|33.20 / 48|2.72 / 32|0.000684|
|1024|4.688%|29.19|33.41 / 48|2.00 / 32|0.001101|

1024 구조에서 C−L F1 점추정의 감소 폭이 크지만 내용·구도와 구조가 함께 달라진 집단 비교다. **같은 이미지에서 토큰 구조만 바꾼 대응 효과는 미측정**이다. 512 구조는 단 1장이므로 구조별 일반화나 성능 우위 판단에 사용하지 않는다.

## 7. Coverage Allocation의 추가 계산 비용

기존 light-audit 1,198문항/방법만 사용했다(최초 full-audit 2문항/방법 제외). 단위는 ms, 각 열은 중앙값이다.

|방법|Allocation|Affinity calibration|Coverage greedy|Selection 전체|
|---|---:|---:|---:|---:|
|S2-L|6.582|0.000|0.000|47.777|
|S2-U|2.328|0.000|0.000|43.577|
|S2-C|49.297|21.764|24.459|90.334|

같은 문항의 C−L 차이를 먼저 계산한 중앙값: Allocation **+43.142 ms**, Selection **+42.934 ms**, Decoder TTFT **+43.411 ms**, E2E TTFT **+44.539 ms**다. Allocation 차이의 평균은 +46.422 ms다. 개별 중앙값들의 합/차이와 paired 중앙값은 같을 필요가 없다.

Allocation은 질문 독립적이었지만 기존 실행에서는 6질문마다 재계산했다. **캐시 재사용 후 실제 절감 시간은 미측정**이다. 기록에는 동기화·관측·CPU 전송이 포함되고 방법 순서가 고정되어 있어, 이 수치로 안정적인 속도 우위를 주장하지 않는다.

## 8. 가능한 원인과 후속 Ablation 후보 — 미실행

|검증 가설|현재 근거|아직 미측정인 값|별도 승인 후 후보|
|---|---|---|---|
|Coverage 목적이 최종 ERC 선택에 충분히 전달되지 않음|임시32 보존 8.20%; 모든 이미지에서 최종 F < 임시 F|선택 규칙을 바꿨을 때의 F1|C 예산·셀·Global16·K64를 고정하고 셀 내 ERC vs Coverage 선택을 대조|
|Feature Coverage가 POPE 객체 판별과 정렬되지 않음|Coverage 소폭 증가에도 정답 순손실 8; 상관이 양수가 아님|GT 객체 위치/마스크와 선택 토큰의 실제 교차, 객체별 중요도|독립 위치 주석으로 보존율 분석; 사전 고정된 projector vs Layer-0 feature 목적 비교|
|Local+Global demand 또는 공간 배분의 효과가 예상과 다름|C는 예산·최종 토큰 모두 L보다 U에 가까움|Local/Global demand 각각의 F 기여 및 성능 영향|동일 후보·예산·ERC에서 demand=Local vs Local+Global; Uniform과 예산 집중도를 맞춘 대조|
|고정 Local48의 구조별 압축률 영향|Local 보존율 9.375%→4.688%; 1024 집단의 C−L 감소 폭이 큼|구조/이미지 내용을 분리한 대응 효과|독립 이미지에서 구조·budget 조건을 사전 고정한 대응 실험|
|16토큰 종료와 Invalid가 일부 차이를 증폭|C Invalid +3; Valid→Invalid 7건 중 5건 No-EOS|max_new_tokens 증가 시 회복률·성능|동일 Parser로 16/32 길이 민감도 분석; 유효 반전과 Invalid 변화를 분리|

이 후보들은 가설 검증용이며 개선을 보장하지 않는다. 특히 첫 가설에서 F를 회복해도 F1이 회복되지 않으면 두 번째 가설을 더 검토해야 한다. **현재 본 평가 데이터는 사후 분석에 이미 사용했으므로 후속 성능 주장은 별도 사전 고정과 독립 holdout이 권장된다.** 기존 코호트·파라미터·Parser는 그대로 보존한다. 이번 결론은 SmolVLM 이식 경로의 결과이며 원본 Qwen S²Prune/CoverPruner 전체에 대한 실패 판정으로 확대하지 않는다.

## 근거 및 작업 범위

원본 [S3-C 보고서](S3C.md), 이하 파일은 모두 `S3-C/analysis_01/`:

- 예산/토큰: `IMAGE_REGION_BUDGET_DIFFERENCES.jsonl`, `token_overlap.jsonl`, `COVERAGE_TEMPORARY_VS_FINAL_ERC.jsonl`, `ALLOCATION_REUSE_CHECK.json`.
- Coverage/정답: `FEATURE_COVERAGE.csv`, `PAIRED_RESPONSE_DIFFERENCES.jsonl`, `per_question.jsonl`, `COVERAGE_QUALITY_RELATION.json`.
- 구조/시간/기준 통계: `TOKEN_STRUCTURE_METRICS.json`, `GPU_LATENCY_VRAM.csv`, `METRICS.json`, `PAIRED_BOOTSTRAP_10000.json`.

집계 시 200이미지·1,200 paired 문항·4방법 Coverage/latency 4,800행, 지역 예산 합48, 공통 Global16, C만 11/L만 19와 예산–overlap 관계를 검산했다. 새 테스트 suite나 새 성능 실험을 실행하지 않았다. 기존 코드·결과·manifest는 읽기만 했고 이 S4-A 보고서 1개만 새로 작성했다. 프로젝트 전수 조사·15,920파일 SHA 재검사·다운로드·설치·Git commit/push는 수행하지 않았다. **S²Prune 라이선스 HOLD 유지. S4-A 보고서 작성으로 중단.**
