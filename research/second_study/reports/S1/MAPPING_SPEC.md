> Historical report view. Commands, raw-evidence paths and earlier publication-preparation statements are archival. Current public scope is counts/aggregate-only; see docs/SECOND_STUDY_RELEASE_SCOPE.md. Original experimental records remain unchanged.

> Public archival view of `results/smolvlm_s2_coverage_phase_s1/phase_s1_20261114_01/MAPPING_SPEC.md`. Home paths redacted; links to omitted local artifacts rendered as text. The original local report is unchanged. Historical commands are records, not portable launch instructions.

# Local/Global Mapping 명세

## 근거 및 불일치 여부

S0 architecture/spatial 설계, 기존 `src/pruning/spatial_floor_coverage.py::local_region_indices`, 설치 Transformers5.15.1의 Idefics3 processor/image processor/connector 소스를 필요한 부분만 정적으로 읽었다. processor/model을 생성하지 않았다.

`processing_idefics3.replace_image_token`은 local crop의 row-major 반복마다64 image placeholders를 만들고 마지막 Global tag 뒤64개를 둔다. separator/newline/row-col/global tag는 image placeholder와 다르다. `split_images`는 resized512배수 mosaic를 row-major local views로 만들고 resized Global을 tail에 더한다. `pixel_shuffle`은32×32 patch grid의4×4 묶음을 최종8×8 row-major로 변환한다. **S0와의 구조 불일치는 발견하지 않았다.** 실제 tokenizer/backend/feature 값 일치는 runtime gate로 남는다.

## 구현 계약

`make_layout(R,C,n_tokens=None)`는 관측된 metadata를 받는다. 이미지 크기에서 crop 수를 새로 추론하거나 임의 resize하지 않는다. R/C는 양의 정수 또는 둘 다0이어야 하며 boolean/float/negative/mismatched count는 거부한다.

```text
Nlocal=64RC, Nglobal=64, Ntotal=64(RC+1)
i_native=64(rC+c)+8v+u
y=8r+v, x=8c+u, i_lattice=y·8C+x
```

native↔lattice 양방향 순열과 native local ordinal별 coarse region ownership을 보존한다. Local 영역의 후보 집합은 native ordinal 오름차순이다. Global native ordinals는[Nlocal,Ntotal), region_id=16이다.

| layout | local lattice | local capacities | Global |
|---|---|---|---|
| R3/C4, N832 | 24×32 | 16지역 각각48, 합768 | tail64 |
| R4/C4, N1088 | 32×32 | 16지역 각각64, 합1024 | tail64 |
| R4/C3, N832 | 32×24 | 16지역 각각48, 합768 | tail64 |
| R0/C0, N64 | local 없음 | local regions 없음 | single64 identity 경로 |

coarse row/col boundaries는 `round(i·grid_size/4)`다. 3×4에서는 row[0,6,12,18,24], col[0,8,16,24,32], region(r,c)=[6r,6(r+1))×[8c,8(c+1))다. 4×4는 양축[0,8,16,24,32]다. 전체 rectangle 목록은 최종 synthetic JSON에 기록했다.

## 좌표와 prompt 위치

`token_info`는 crop row/col, view index,8×8 내부 좌표, lattice 좌표, coarse region, normalized box/center를 반환한다.

```text
local center: ((c+(u+.5)/8)/C, (r+(v+.5)/8)/R)
global center: ((u+.5)/8, (v+.5)/8)
```

box를 원본 W/H에 곱하면 nominal pixel footprint다. resize rounding에 의한 종횡비 변형을 포함하며 full vision self-attention의 실제 receptive field를 뜻하지 않는다. Global과 Local은 중복 공간 표현이므로 한 disjoint lattice에 섞지 않는다.

`map_prompt_positions`는 `input_ids == image_token_id(49190)`로 실제 visual positions를 찾고 나머지를 모두 nonvisual로 분류한다. visual ordinal은 vocabulary ID나 연속 prompt offset이 아니다. 최종 selected ordinals를 `visual_positions[selected]`로 변환한 뒤 모든 nonvisual positions와 합쳐 정렬한다. image placeholder 수와 layout N의 불일치는 중단한다.

## CPU 근거

MappingTests10개, scalar_mapping, 기존 local_region_indices의 여러 rectangular layouts 비교가 통과했다. native와 lattice 순서가 다른 경우 및 crop 경계를 넘는 coarse/cell도 검사했다. 새 실제 Smol processor 입력에 연결한 mapping 검증은 아직 수행하지 않았다.
