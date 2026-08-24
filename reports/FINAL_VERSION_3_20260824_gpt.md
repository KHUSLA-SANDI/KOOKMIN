# 최종버전 3 — 전 장애물 하드코딩 회피

기준: 최종버전 2 + `overtake_strategy: hardcoded_all`.

## 실행 계약

- LiDAR 장애물 ROI는 기존 `0 < x <= 1.40 m`, 흰선 안쪽 `0.10 m`, 2프레임 확인을 유지한다.
- yellow/mid 기준 lateral offset이 `+0.10 m` 이상이면 1차선(왼쪽) 장애물, `-0.10 m` 이하이면 2차선(오른쪽) 장애물이다.
- deadband 안쪽 또는 양쪽에 유효 cluster가 동시에 있으면 `UNKNOWN`으로 거부한다.
- 1차선 장애물: 차량 RIGHT `+60 x 6틱 -> -60 x 10틱 -> 직진 20틱`.
- 2차선 장애물: 차량 LEFT `-60 x 7틱 -> +60 x 10틱 -> 직진 20틱`.
- 요청 수락 다음 20 Hz 제어 틱에서 바로 SHIFT한다. 속도 도달 대기는 없으며 속도 slew와 동시에 16을 목표로 한다.
- 블록 동안에도 새 CNN 경로는 계속 cache하지만 조향에는 쓰지 않는다. `simple_motion`만 `/xycar_motor`를 발행한다.
- 명시적 STOP/drive false 또는 CONE 진입은 블록을 취소한다. 중복 event id와 실행 중 재진입은 무시한다.
- V3에서 장애물 CNN 모드로 전환하지 않고 CNN 모드는 GENERAL을 유지한다.

## 현장 수정 위치

- 인식/전략: `src/track_drive_cnn_gpt/config/perception_cnn.yaml`
- 틱/조향/속도: `src/track_drive_cnn_gpt/config/simple_motion.yaml`
- 전방거리 `overtake_x_max_m`은 V3부터 `0 < max <= 3.0 m` 범위에서 YAML로 바꿀 수 있다.

## 검증

- 핵심 Python 5개 파일 byte-compile 성공.
- focused tests: `36 passed, 2 skipped`.
- 두 방향의 SHIFT/COUNTER/PASS 정확한 틱 수, 마지막 PASS N/N, 다음 주기 IDLE, 방향 확인, 모호 cluster 거부, event 계약, V3 YAML 계약을 검사했다.

실차 모터 시험은 이 작업에서는 수행하지 않았다.
