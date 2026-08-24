# 최종버전 4 — CONE 직후 고정장애물만 하드코딩

기준: 최종버전 3과 같은 코드, 전략만
`overtake_strategy: hardcoded_post_cone`으로 변경한다.

## 동작

- CONE의 10초 타이머가 자연 종료된 정확한 시각부터 5초 동안만 차선별
  하드코딩 블록을 사용한다.
- 이 5초 안의 1차선 장애물은 RIGHT 블록, 2차선 장애물은 LEFT 블록으로
  즉시 회피한다. 블록 목표속도는 16이고 속도 도달을 기다리지 않는다.
- 5초 창 밖의 장애물은 기존 `OVERTAKE` CNN 모드로 처리한다. 따라서 주행
  차량 추월 로직은 V2 방식이 유지된다.
- 명시적 reset이나 노드 재시작은 5초 창을 만들지 않는다. 반복 START_R은
  기존 CONE 타이머를 연장하지 않는다.
- 창 종료 직전에 시작한 블록은 중간에 끊지 않고 마지막 직진 틱까지
  끝낸다. 동일 장애물은 clear-frame 재무장 전까지 다시 실행하지 않는다.

## 현장 조정

- 창 길이: `perception_cnn.yaml`의 `post_cone_hardcode_window_sec`
- CONE 유지시간: 같은 파일의 `cone_hold_sec`
- 좌/우/직진 틱 및 속도: `simple_motion.yaml`의 `hardcoded_block_*`

모두 YAML 파라미터이므로 수정 후 빌드 없이 주행 launch만 재시작하면 된다.
