# 최종버전 2

- 기준 코드: `final-v1-20260823` (`e3e36285f478a407ef6d82faa803747f0a97c7bd`)
- 코드 동작 차이: 신호등 2단 분류기만 `yolo11s_cls_best.pt` 기반 OpenVINO로 교체
- 차량 모델 폴더: `/home/xytron/xycar_ws/models/yolo11s_cls_best_v2_openvino_model`
- 원본 PT SHA256: `87ac68859af279e6d153334b7c90fb27b3ed221481bda2fc68e11c7bfeab16e0`
- OpenVINO 계약: FP32, static batch 1, 입력 `[1,3,320,320]`, 출력 `[1,4]`
- 클래스 순서: `GREEN`, `LEFT`, `RED`, `YELLOW`

V1의 `/home/xytron/xycar_ws/models/traffic_light_cls_best_openvino_model`은
덮어쓰거나 지우지 않는다. 각 Git 버전의 YAML이 서로 다른 모델 폴더를
가리켜야 코드 전환만으로 정확한 분류기까지 되돌릴 수 있다.

로컬 실제 녹화 crop 11개 비교에서는 PT/OpenVINO top-1이 11/11 일치했고,
전체 평균 절대 확률 차이는 0.003485였다.
