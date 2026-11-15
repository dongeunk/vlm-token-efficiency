> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s3b1/phase_s3b1_20261114_01/ACTUAL_MAPPING_AND_576.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# 실제 Mapping과576 구조 결론

실제 Idefics3Processor를 CPU에서 실행했고, GPU generation에는 별도의 비평가576 샘플을 사용했다. `FUNCTIONAL_SCOPE_LOCK.json`의 샘플은 processor/모델 결과를 보기 전에 고정했다. old80의 저장 metadata에는832×70/1088×10, old30 크기 예상에는576이 없었다. 남은 비평가 후보에서 image553165를 찾았다. 새 파일럿/본 평가210장은 GPU 입력에 포함하지 않았다.

## 실제 processor 출력

| image_id | 사용 | R×C | views | Local / Global native ordinals | 실제N | L/T/L' |
|---:|---|---|---:|---|---:|---|
|553165|576 GPU 기능|2×4|9|0..511 / 512..575|576|613/37/101|
|376959|832 GPU 기능, old80|3×4|13|0..767 / 768..831|832|878/46/110|
|436127|1088 GPU 기능, old80|4×4|17|0..1023 / 1024..1087|1088|1143/55/119|
|358255|main 문제 이미지 CPU 확인만|2×4|9|0..511 / 512..575|576|613/37/101, 기능 질문 기준|

LLM sequence 내 모든 visual/nonvisual 위치와 원래input_ids는 각 `CPU_PROCESSOR_<ID>.json`에 저장되어 있다. global/local native ordinal은 sequence index와 다르다. image marker/row-col/global marker와 질문·special token을 visual로 잘못 처리하지 않았다.

## 위치와 coarse region

실제 crop(r,c)의 순서는 row-major이며, crop 안64 projected token은8×8 row-major다. 마지막 view가 global이다. S0/S1의 기존 이 대응을 사용했다. 설치된 Idefics3 `split_images`와 processor prompt expansion에 근거하며, 실제 view/placeholder/projector 수를 CPU/GPU에서 각각 대조했다.

`native=64(r*C+c)+8v+u`, `y=8r+v`, `x=8c+u`. nominal source crop box는 `[c/C,r/R,(c+1)/C,(r+1)/R]`다. token_info는 이 정규화 좌표를 보존한다. vision self-attention을 거치므로 box를 정보가 독점적으로 들어오는 receptive field로 주장하지 않는다.

coarse region `(a,b), a,b∈{0,1,2,3}`의 local-lattice 좌표:

| crop layout | local lattice | region rectangle `[y0,y1)×[x0,x1)` | 각capacity |
|---|---|---|---:|
|2×4|16×32|`[4a,4(a+1)) × [8b,8(b+1))`|32|
|3×4|24×32|`[6a,6(a+1)) × [8b,8(b+1))`|48|
|4×4|32×32|`[8a,8(a+1)) × [8b,8(b+1))`|64|

독립적인 crop/v/u scalar loop로 모든 local token의 region ownership과 capacity를 대조했다. S1 stitched-lattice mapping과 exact 일치했다. 각지역의 q는1 이상/capacity 이하, sum48이며 U는16지역 모두3이다. 최종 local48 cells가 localN 전체를 겹침/누락 없이 분할한다. 각 cell의 실제 ERC 최대1개를 선택하고 정확한 tie는 낮은 native ordinal로 해소한다.

## Global16과576 결론

Global은 기존 단일8×8 region에 **공통 recursive partition16 cells + 같은 Layer0 ERC argmax**를 적용했다. 고정된16개의 절대 index를 모든 이미지에 강제하는 규칙이 아니다. 같은 입력의 L/U/C에서 global geometry/ERC/선택16개가 exact 같다. 전체global64도 cells 간 중복/누락 없이 포함된다.

576에서도 local512=capacity32×16으로 local48을 충족한다. 실제 GPU9개 views→projector[9,64,576], full Layer0[1,613,576], upper layers[1,101,576], compact KV101→110(9 decode appends)을 확인했다. Mapping·예산·Global 규칙을 새로 만들지 않았다. 기존 runtime의832/878/110 고정값만 별도 metadata-driven runtime에서 일반화했다.

main358255는 실제 같은2×4/576이다. 본 평가 이미지 자체의 GPU generation과 공식6문항 sequence는 실행하지 않았지만 동일구조의 기능 지원 근거는 비평가553165에서 얻었다. manifest 및 해당이미지는 그대로 유지한다.
