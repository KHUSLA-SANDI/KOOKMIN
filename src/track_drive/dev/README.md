# dev/ — 캘리브 개발 산출물 (차 PC 로컬)

## calib/
- `h_pts.csv`       수동 클릭 대응점 14점 (RMS 38mm)
- `h_pts_clean.csv` 이상점 4개 제거 10점 (RMS 13mm)
- `h_pts_auto.csv`  줄눈 자동 서브픽셀 검출 (RMS 6.1mm) ← **현재 camera.yaml 의 H 근거**
- `H_auto.npy`      자동정밀화 H 백업
- `camera_yaml_before_auto.yaml` 자동정밀화 직전 백업

## shots/
`calib_raw.jpg` 원본 · `calib_und.jpg` 왜곡보정 · `calib_enh.jpg` 대비강화
`backproj*.jpg` 격자 역투영 검증 · `bev.jpg` BEV 결과

## scripts/
분석용 (필요할 때만 실행)
- `_grab_undist.py`  한 프레임 캡처 + 보정 저장
- `_backproj.py`     타일 격자를 사진에 역투영 (육안 검증)
- `_measure_err.py`  예측 vs 실제 줄눈 오차 정량 측정
- `_profile_col.py`  세로 스트립의 실제 줄눈 전부 검출 (오검출 배제용)
- `_auto_refine.py`  줄눈 자동검출로 H 정밀화
- `_save_auto.py`    자동검출 대응점 CSV 생성

## 캘리브 요약 (2026-07-30)
- 렌즈: **plumb_bob** (벤더 PDF 의 "170도 어안"은 오기. 실측 HFOV 87.2/VFOV 64.1)
- K/D: 차의 `~/.ros/camera_info/default_cam.yaml` 임포트
- H: 복도 타일 격자, 피치 0.453333 m (2.72m/6칸), A' = 줄눈 교차점
- `h_frame: lidar_frame` (라이다가 카메라보다 10cm 앞 → x 오프셋 적용됨)
- `h_rms_m: 0.0061`, 예측↔실제 줄눈 일치 1.7px 이내
