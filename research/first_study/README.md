# 1차 연구 보존 및 추가 집계

기존 원격66bbfaf의 30이미지 파일럿 코드·보고서·집계·그림·LICENSE·NOTICE는 원래 경로에 그대로 보존한다. 루트 README의 과거 본문도 그대로 남아 있다.

추가 공개 view에는 후속80이미지·160문항·다섯 방법800결과의 보고서/집계와 동일80이미지 Floor vs No-Floor320결과의 보고서/집계를 담았다. 원본 결과는 기존 C 작업 폴더에 그대로 있으며 여기로 이동하지 않았다. 새 후보에는 원시 응답과 원본 사진·질문 전문·실행 manifest를 포함하지 않는다.

- [800결과 보고서](reports/pope80_five_methods.md), [metrics](pope80/metrics.json).
- [320결과 보고서](reports/floor_vs_no_floor.md), [paired CI](paired320/paired_bootstrap_CI.json).

두 후속 평가는 동일80이미지에 대한 기록이며 독립 holdout 두 개로 해석하지 않는다. Ours81.875% vs PureGC80.625%의 차이 CI는0을 포함했고, Local-only Floor/No-Floor는 둘 다81.875%였다.

권리 HOLD인 S2 reference, fixture/parity test와 dependency-HOLD 실행 코드는 계속 제외한다. 현재 공개 scope는 docs/SECOND_STUDY_RELEASE_SCOPE.md와 docs/publication/FILE_LIST.md를 따른다. 기존 원격의 포괄 저작권 고지를 제거하지 않는다.
