"""track_drive.lib.bev 왜곡모델/BEV 테스트 (rclpy 불필요).

검증 범위:
  (a) plumb_bob / fisheye 모델 분기
  (b) model 미기재 시 D 길이 자동추론 (5→plumb_bob, 4→fisheye)
  (c) 미캘리브 시 no-op 패스스루
  (d) new_K 불일치 경고 (H 는 undistort 픽셀 기준 → new_K 바뀌면 무효)
  (e) mask_to_bev_points 가 기존과 동일 동작 유지 (골든값 회귀)
  (f) 캘리브 플래그 분리 — intrinsics_calibrated(K/D) 와 h_calibrated(H) 는 별개.
      "K/D 는 임포트했지만 지면 H 는 아직" 이 실차 정상 중간 상태라,
      이때 K/D 까지 폴백으로 버리면 undistort 가 통째로 죽는다.
  (g) 실제 config/camera.yaml 이 그 규약대로 채워져 있는지 (배포 사고 방지)
"""

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from track_drive.lib import bev  # noqa: E402

SIZE = (640, 480)   # (width, height)

# 실차 default_cam.yaml 실측 plumb_bob 캘리브 (테스트 고정 입력으로만 사용)
REAL_K = [[438.78, 0.0, 305.59],
          [0.0, 437.30, 243.74],
          [0.0, 0.0, 1.0]]
REAL_D = [-0.361976, 0.110510, 0.001014, 0.000505, 0.0]   # 5개 → plumb_bob

# 합성 어안 왜곡계수 (4개 → fisheye)
FISH_D = [-0.05, 0.01, -0.002, 0.0004]

IDENT_H = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def write_yaml(tmp_path, body):
    p = tmp_path / "camera.yaml"
    p.write_text(body, encoding="utf-8")
    return str(p)


def yaml_body(model=None, calibrated=True, K=REAL_K, D=REAL_D, new_K=None,
              intrinsics=None, h=None, size=None):
    """camera.yaml 텍스트 생성 (PyYAML dump 대신 손으로 — 스키마를 눈으로 확인).

    calibrated: 구버전 단일 키 (None 이면 아예 안 쓴다).
    intrinsics / h: 신형 분리 플래그 (None 이면 안 쓴다 → calibrated 로 폴백).
    """
    lines = []
    if model is not None:
        lines.append(f"model: {model}")
    if calibrated is not None:
        lines.append(f"calibrated: {'true' if calibrated else 'false'}")
    if intrinsics is not None:
        lines.append(f"intrinsics_calibrated: {'true' if intrinsics else 'false'}")
    if h is not None:
        lines.append(f"h_calibrated: {'true' if h else 'false'}")
    if size is not None:
        lines.append(f"image_width: {int(size[0])}")
        lines.append(f"image_height: {int(size[1])}")
    lines.append("K:")
    lines += [f"  - [{r[0]!r}, {r[1]!r}, {r[2]!r}]" for r in K]
    lines.append("D: [" + ", ".join(repr(float(v)) for v in D) + "]")
    if new_K is not None:
        lines.append("new_K:")
        lines += [f"  - [{r[0]!r}, {r[1]!r}, {r[2]!r}]"
                  for r in np.asarray(new_K, dtype=float).tolist()]
    lines.append("H:")
    lines += [f"  - [{r[0]!r}, {r[1]!r}, {r[2]!r}]" for r in IDENT_H]
    return "\n".join(lines) + "\n"


# ============================================================
# (a) 모델 분기 — plumb_bob / fisheye
# ============================================================

def test_plumb_bob_branch_uses_optimal_new_camera_matrix():
    u = bev.Undistorter(REAL_K, REAL_D, SIZE, model="plumb_bob")
    assert u.enabled
    assert u.model == bev.MODEL_PLUMB_BOB
    # getOptimalNewCameraMatrix 경로 흔적: roi 가 채워지고 new_K != K
    assert u.roi is not None and len(u.roi) == 4
    assert not np.allclose(u.new_K, np.asarray(REAL_K, dtype=float))
    # new_K 는 여전히 정상적인 카메라 행렬 꼴이어야 한다
    assert u.new_K.shape == (3, 3)
    assert u.new_K[2, 2] == pytest.approx(1.0)
    assert u.new_K[0, 0] > 0 and u.new_K[1, 1] > 0


def test_fisheye_branch_keeps_new_K_equal_to_K():
    u = bev.Undistorter(REAL_K, FISH_D, SIZE, model="fisheye")
    assert u.enabled
    assert u.model == bev.MODEL_FISHEYE
    # 어안 규약: new_K = K (H 좌표계 일치). roi 는 계산하지 않는다.
    np.testing.assert_allclose(u.new_K, np.asarray(REAL_K, dtype=float))
    assert u.roi is None


def test_two_models_produce_different_undistort_result():
    """같은 K/D 라도 모델이 다르면 실제 리맵 결과가 달라야 한다 (분기 확인)."""
    img = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    img[::17, :, :] = 255          # 가로 줄무늬 — 왜곡 차이가 드러나는 패턴
    img[:, ::23, :] = 255

    out_pb = bev.Undistorter(REAL_K, REAL_D, SIZE, model="plumb_bob").apply(img)
    out_fe = bev.Undistorter(REAL_K, REAL_D, SIZE, model="fisheye").apply(img)

    assert out_pb.shape == img.shape and out_fe.shape == img.shape
    assert not np.array_equal(out_pb, out_fe)
    assert not np.array_equal(out_pb, img)      # 실제로 보정이 일어남


def test_plumb_bob_alpha_changes_new_K():
    """alpha 는 new_K 를 바꾼다 → H 무효화 위험의 근원. 기본값은 0.0."""
    u0 = bev.Undistorter(REAL_K, REAL_D, SIZE, model="plumb_bob", alpha=0.0)
    u1 = bev.Undistorter(REAL_K, REAL_D, SIZE, model="plumb_bob", alpha=1.0)
    assert not np.allclose(u0.new_K, u1.new_K)
    assert bev.UNDISTORT_ALPHA_DEFAULT == 0.0
    u_def = bev.Undistorter(REAL_K, REAL_D, SIZE, model="plumb_bob")
    np.testing.assert_allclose(u_def.new_K, u0.new_K)


def test_explicit_new_K_overrides_computed_one():
    """yaml 의 new_K 가 있으면 그걸 그대로 쓴다 (캘리브 당시 값이 정답)."""
    forced = np.array([[400.0, 0.0, 320.0],
                       [0.0, 400.0, 240.0],
                       [0.0, 0.0, 1.0]])
    for model, D in (("plumb_bob", REAL_D), ("fisheye", FISH_D)):
        u = bev.Undistorter(REAL_K, D, SIZE, model=model, new_K=forced)
        assert u.enabled
        np.testing.assert_allclose(u.new_K, forced)


def test_map_is_built_once_and_cached():
    """리맵 테이블은 초기화 1회 생성 후 캐시 — apply 가 재계산하지 않는다."""
    u = bev.Undistorter(REAL_K, REAL_D, SIZE, model="plumb_bob")
    m1, m2 = u._map1, u._map2
    img = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    u.apply(img)
    u.apply(img)
    assert u._map1 is m1 and u._map2 is m2


@pytest.mark.parametrize("K, D, size, model", [
    (REAL_K, [0.1] * 6, SIZE, "plumb_bob"),        # cv2 가 안 받는 D 길이
    (REAL_K, ["a"] * 5, SIZE, "plumb_bob"),        # 숫자가 아닌 D
    (REAL_K, REAL_D, (-5, 480), "plumb_bob"),      # 말이 안 되는 이미지 크기
    (REAL_K, REAL_D, ("x", 480), "plumb_bob"),     # 숫자가 아닌 크기
    ([1, 2, 3], REAL_D, SIZE, "plumb_bob"),        # 3x3 이 아닌 K
    ([1, 2, 3], FISH_D, SIZE, "fisheye"),          # 어안 경로도 동일하게 방어
])
def test_invalid_KD_falls_back_to_passthrough(K, D, size, model):
    """잘못된 K/D/size 는 예외를 던지지 않고 패스스루로 떨어진다."""
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    u = bev.Undistorter(K, D, size, model=model)
    assert not u.enabled
    assert u.apply(img) is img          # 노드가 죽지 않고 원본으로 계속 간다


def test_undistorter_never_raises_on_junk_input():
    """초기화 실패는 enabled=False 로만 드러나야 한다 (노드 크래시 금지)."""
    for K in (None, [1, 2, 3], np.zeros((3, 3)), REAL_K):
        for D in (None, [], [0.1] * 6, REAL_D, FISH_D):
            for size in ((0, 0), (-1, -1), SIZE):
                u = bev.Undistorter(K, D, size)
                assert isinstance(u.enabled, bool)
                assert u.model in bev.VALID_MODELS


# ============================================================
# (b) D 길이 자동추론
# ============================================================

@pytest.mark.parametrize("D, expect", [
    (REAL_D, bev.MODEL_PLUMB_BOB),      # 5개
    (FISH_D, bev.MODEL_FISHEYE),        # 4개
])
def test_model_inferred_from_D_length(tmp_path, D, expect):
    cam = bev.load_camera_config(write_yaml(tmp_path, yaml_body(model=None, D=D)))
    assert cam["calibrated"]
    assert cam["model"] == expect
    assert cam["note"]                                   # 추론 사실이 로그 문자열로 남음
    assert expect in cam["note"] and "추론" in cam["note"]


def test_explicit_model_wins_over_D_length(tmp_path):
    """model 이 명시돼 있으면 D 길이와 달라도 명시값을 따르고 note 는 비어 있다."""
    cam = bev.load_camera_config(
        write_yaml(tmp_path, yaml_body(model="fisheye", D=REAL_D)))
    assert cam["model"] == bev.MODEL_FISHEYE
    assert cam["note"] == ""
    assert cam["D"].size == 4                            # fisheye 규격으로 절삭
    np.testing.assert_allclose(cam["D"], REAL_D[:4])


def test_unknown_model_string_falls_back_to_inference(tmp_path):
    cam = bev.load_camera_config(
        write_yaml(tmp_path, yaml_body(model="170deg_fisheye", D=REAL_D)))
    assert cam["model"] == bev.MODEL_PLUMB_BOB
    assert "알 수 없는 값" in cam["note"]


def test_plumb_bob_D_is_zero_padded_to_five(tmp_path):
    cam = bev.load_camera_config(
        write_yaml(tmp_path, yaml_body(model="plumb_bob", D=[0.1, 0.2, 0.0])))
    assert cam["D"].size == 5
    np.testing.assert_allclose(cam["D"], [0.1, 0.2, 0.0, 0.0, 0.0])


def test_resolve_distortion_model_direct():
    assert bev.resolve_distortion_model("plumb_bob") == (bev.MODEL_PLUMB_BOB, "")
    assert bev.resolve_distortion_model("FishEye ") == (bev.MODEL_FISHEYE, "")
    m, note = bev.resolve_distortion_model(None, np.zeros(4))
    assert m == bev.MODEL_FISHEYE and note
    m, note = bev.resolve_distortion_model(None, np.zeros(5))
    assert m == bev.MODEL_PLUMB_BOB and note
    # D 조차 없으면 기본 모델 (실차 렌즈가 일반 광각이므로 plumb_bob)
    m, note = bev.resolve_distortion_model(None, None)
    assert m == bev.DEFAULT_MODEL == bev.MODEL_PLUMB_BOB and note


def test_undistorter_infers_model_when_not_given():
    assert bev.Undistorter(REAL_K, REAL_D, SIZE).model == bev.MODEL_PLUMB_BOB
    assert bev.Undistorter(REAL_K, FISH_D, SIZE).model == bev.MODEL_FISHEYE


# ============================================================
# (c) 미캘리브(K/D 없음) → 패스스루
# ============================================================

def test_uncalibrated_yaml_falls_back_to_sim_H(tmp_path):
    cam = bev.load_camera_config(
        write_yaml(tmp_path, yaml_body(model="plumb_bob", calibrated=False)))
    assert cam["calibrated"] is False
    # 미캘리브 H 는 신뢰하지 않고 예선 시뮬 폴백 H 사용
    np.testing.assert_allclose(cam["H"], bev.H_PIX2LIDAR_SIM_FALLBACK)
    np.testing.assert_allclose(cam["K"], np.eye(3))
    assert cam["new_K"] is None
    assert cam["model"] == bev.MODEL_PLUMB_BOB      # model 정보는 살려서 넘긴다


def test_missing_file_returns_default_model_fallback():
    cam = bev.load_camera_config("/nonexistent/camera.yaml")
    assert cam["calibrated"] is False
    assert cam["model"] == bev.DEFAULT_MODEL
    assert cam["D"].size == 5
    assert cam["new_K"] is None
    np.testing.assert_allclose(cam["H"], bev.H_PIX2LIDAR_SIM_FALLBACK)


def test_uncalibrated_undistorter_is_noop_passthrough(tmp_path):
    cam = bev.load_camera_config(
        write_yaml(tmp_path, yaml_body(model="plumb_bob", calibrated=False)))
    u = bev.Undistorter.from_config(cam, SIZE)
    assert not u.enabled
    img = np.arange(SIZE[1] * SIZE[0] * 3, dtype=np.uint8).reshape(SIZE[1], SIZE[0], 3)
    assert u.apply(img) is img                       # 복사조차 하지 않는 진짜 no-op
    assert u.apply(None) is None


def test_passthrough_for_both_models():
    for model, D in (("plumb_bob", REAL_D), ("fisheye", FISH_D)):
        u = bev.Undistorter(REAL_K, D, SIZE, calibrated=False, model=model)
        assert not u.enabled
        np.testing.assert_allclose(u.new_K, np.asarray(REAL_K, dtype=float))


def test_from_config_passes_model_and_new_K(tmp_path):
    forced = [[401.0, 0.0, 321.0], [0.0, 402.0, 241.0], [0.0, 0.0, 1.0]]
    cam = bev.load_camera_config(
        write_yaml(tmp_path, yaml_body(model="plumb_bob", new_K=forced)))
    np.testing.assert_allclose(cam["new_K"], forced)
    u = bev.Undistorter.from_config(cam, SIZE)
    assert u.enabled and u.model == bev.MODEL_PLUMB_BOB
    np.testing.assert_allclose(u.new_K, forced)      # yaml new_K 가 그대로 쓰인다


# ============================================================
# (d) new_K 불일치 경고
# ============================================================

def test_check_new_K_match_ok_returns_none():
    K = np.asarray(REAL_K, dtype=float)
    assert bev.check_new_K_match(K, K.copy()) is None
    assert bev.check_new_K_match(None, None) is None
    # 허용오차 이내의 미세한 차이는 통과
    K2 = K.copy()
    K2[0, 0] += bev.NEW_K_MATCH_TOL / 10.0
    assert bev.check_new_K_match(K, K2) is None


def test_check_new_K_match_reports_mismatch():
    K = np.asarray(REAL_K, dtype=float)
    K2 = K.copy()
    K2[0, 0] += 12.5
    msg = bev.check_new_K_match(K, K2)
    assert isinstance(msg, str)
    assert "new_K 불일치" in msg
    assert "12.5" in msg                              # 최대 절대차를 보고


def test_check_new_K_match_reports_missing_side():
    K = np.asarray(REAL_K, dtype=float)
    msg_calib_missing = bev.check_new_K_match(None, K)
    msg_runtime_missing = bev.check_new_K_match(K, None)
    assert isinstance(msg_calib_missing, str) and "new_K" in msg_calib_missing
    assert isinstance(msg_runtime_missing, str) and "new_K" in msg_runtime_missing


def test_alpha_drift_is_detected_end_to_end(tmp_path):
    """캘리브는 alpha=0 로 했는데 런타임이 alpha=1 로 돌면 H 가 무효 → 경고."""
    calib_new_K = bev.Undistorter(
        REAL_K, REAL_D, SIZE, model="plumb_bob", alpha=0.0).new_K
    cam = bev.load_camera_config(
        write_yaml(tmp_path, yaml_body(model="plumb_bob", new_K=calib_new_K)))

    good = bev.Undistorter(REAL_K, REAL_D, SIZE, model="plumb_bob", alpha=0.0)
    assert bev.check_new_K_match(cam["new_K"], good.new_K) is None

    drifted = bev.Undistorter(REAL_K, REAL_D, SIZE, model="plumb_bob", alpha=1.0)
    assert "new_K 불일치" in bev.check_new_K_match(cam["new_K"], drifted.new_K)


def test_model_swap_invalidates_new_K(tmp_path):
    """모델을 fisheye 로 잘못 바꾸면 new_K 가 K 로 돌아가 H 가 무효 → 경고."""
    calib_new_K = bev.Undistorter(
        REAL_K, REAL_D, SIZE, model="plumb_bob", alpha=0.0).new_K
    wrong = bev.Undistorter(REAL_K, REAL_D, SIZE, model="fisheye")
    assert "new_K 불일치" in bev.check_new_K_match(calib_new_K, wrong.new_K)


# ============================================================
# (e) mask_to_bev_points — 기존 동작 유지 (골든값 회귀)
# ============================================================

def bev_identity_H():
    """H_pix2bev = M_GRID @ H 가 단위행렬이 되는 H — 마스크가 그대로 BEV 가 된다."""
    return np.linalg.inv(bev.M_GRID)


def test_mask_to_bev_points_golden_identity_warp():
    mask = np.zeros((bev.BEV_H, bev.BEV_W), dtype=np.uint8)
    mask[:, 100] = 255              # u=100 → y=0 m 인 세로 차선
    mask[40:80, 60] = 255           # u=60  → y=+2 m 인 두 번째 차선 조각

    xs, ys = bev.mask_to_bev_points(mask, bev_identity_H())

    assert xs.dtype == np.float32 and ys.dtype == np.float32
    assert xs.size == ys.size == 24
    np.testing.assert_allclose(
        xs[:10],
        [-1.525, -0.625, 0.275, 1.175, 2.075, 2.975, 3.875, 4.775, 5.675, 6.575],
        rtol=0, atol=1e-6)
    np.testing.assert_allclose(ys[:10], np.zeros(10), atol=1e-6)
    assert float(xs.sum()) == pytest.approx(155.425003, abs=1e-4)
    assert float(ys.sum()) == pytest.approx(10.0, abs=1e-6)
    assert float(xs.min()) == pytest.approx(-1.525, abs=1e-6)
    assert float(xs.max()) == pytest.approx(11.775, abs=1e-6)
    # y=0(u=100)과 y=+2(u=60), 두 차선이 같은 bin 에 섞인 곳의 중앙값 1.0
    assert sorted({round(float(v), 6) for v in ys}) == [0.0, 1.0, 2.0]
    assert np.all(np.diff(xs) > 0)                    # x 오름차순 정렬 유지
    assert bev.lane_quality(xs, ys) == pytest.approx(1.0)


def test_mask_to_bev_points_golden_sim_fallback_H():
    """실제 이미지 크기(480x640) + 예선 시뮬 폴백 H 골든값."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[300:470, 300:308] = 255

    xs, ys = bev.mask_to_bev_points(mask, bev.H_PIX2LIDAR_SIM_FALLBACK)

    np.testing.assert_allclose(xs, [
        1.175, 1.725, 1.875, 2.05, 2.25, 2.625, 2.825, 2.975, 3.125, 3.3,
        3.55, 3.725, 3.875, 4.025, 4.2, 4.45, 4.625, 4.8, 4.95, 5.125,
        5.3, 5.475, 5.675, 5.85, 6.025, 6.225, 6.4, 6.575], rtol=0, atol=1e-6)
    np.testing.assert_allclose(ys, [
        0.05, 0.075, 0.075, 0.05, 0.075, 0.1, 0.125, 0.125, 0.125, 0.1,
        0.15, 0.125, 0.15, 0.175, 0.15, 0.2, 0.175, 0.15, 0.15, 0.15,
        0.15, 0.15, 0.175, 0.3, 0.3, 0.3, 0.3, 0.3], rtol=0, atol=1e-6)
    assert bev.lane_quality(xs, ys) == pytest.approx(1.0)


def test_mask_to_bev_points_empty_inputs():
    empty = (np.array([], dtype=np.float32), np.array([], dtype=np.float32))
    for mask in (None, np.zeros((0, 0), dtype=np.uint8)):
        xs, ys = bev.mask_to_bev_points(mask, bev.H_PIX2LIDAR_SIM_FALLBACK)
        assert xs.size == 0 and ys.size == 0
    xs, ys = bev.mask_to_bev_points(
        np.zeros((480, 640), dtype=np.uint8), bev.H_PIX2LIDAR_SIM_FALLBACK)
    assert xs.size == 0 and ys.size == 0
    assert bev.lane_quality(*empty) == 0.0


def test_bev_constants_unchanged():
    """BEV 격자 상수는 예선 common.py 와 동일해야 한다 (H 호환의 전제)."""
    assert bev.BEV_RES == 0.05
    assert (bev.BEV_X_MIN, bev.BEV_X_MAX) == (-2.0, 12.0)
    assert bev.BEV_Y_HALF == 5.0
    assert (bev.BEV_W, bev.BEV_H) == (200, 280)


def test_lane_quality_grading():
    assert bev.lane_quality([], []) == 0.0
    assert bev.lane_quality([1.0], [0.0]) == 0.0        # 점 1개 → 스팬 0
    xs = np.linspace(0.0, 3.0, 20)
    assert bev.lane_quality(xs, np.zeros(20)) == pytest.approx(1.0)
    xs_short = np.linspace(0.0, 1.5, 10)
    assert bev.lane_quality(xs_short, np.zeros(10)) == pytest.approx(0.25)


# ============================================================
# (f) 캘리브 플래그 분리 — intrinsics_calibrated(K/D) vs h_calibrated(H)
# ============================================================

def test_intrinsics_only_keeps_KD_but_falls_back_H(tmp_path):
    """오늘의 실차 상태: K/D 임포트 완료 + 지면 H 는 아직.

    이때 K/D 까지 폴백(단위행렬/0)으로 버리면 undistort 가 무용지물이 된다.
    """
    cam = bev.load_camera_config(write_yaml(tmp_path, yaml_body(
        model="plumb_bob", calibrated=False, intrinsics=True, h=False)))

    assert cam["intrinsics_calibrated"] is True
    assert cam["h_calibrated"] is False
    assert cam["calibrated"] is False                 # 요약 키 = 둘 다 참일 때만
    np.testing.assert_allclose(cam["K"], REAL_K)      # ★ 실측 K/D 가 살아 있어야 한다
    np.testing.assert_allclose(cam["D"], REAL_D)
    # H 는 미보정이므로 검증된 예선 시뮬 폴백
    np.testing.assert_allclose(cam["H"], bev.H_PIX2LIDAR_SIM_FALLBACK)
    assert "h_calibrated=false" in cam["note"]        # 폴백 사유가 note 에 남는다


def test_intrinsics_only_still_undistorts(tmp_path):
    """H 미보정이어도 K/D 가 실측이면 undistort 는 켜져야 한다."""
    cam = bev.load_camera_config(write_yaml(tmp_path, yaml_body(
        model="plumb_bob", calibrated=False, intrinsics=True, h=False)))
    u = bev.Undistorter.from_config(cam, SIZE)
    assert u.enabled and u.model == bev.MODEL_PLUMB_BOB

    img = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
    img[::17, :, :] = 255
    img[:, ::23, :] = 255
    out = u.apply(img)
    assert out is not img and not np.array_equal(out, img)   # 실제로 보정이 일어남


def test_h_only_drops_KD_but_keeps_H(tmp_path):
    """반대 조합: H 는 잡혔는데 K/D 가 미보정 → K/D 만 폴백, H 는 yaml 값 유지."""
    cam = bev.load_camera_config(write_yaml(tmp_path, yaml_body(
        model="plumb_bob", calibrated=False, intrinsics=False, h=True)))

    assert cam["intrinsics_calibrated"] is False and cam["h_calibrated"] is True
    assert cam["calibrated"] is False
    np.testing.assert_allclose(cam["K"], np.eye(3))
    assert not np.any(cam["D"])
    np.testing.assert_allclose(cam["H"], IDENT_H)     # yaml 의 H 를 그대로 쓴다
    assert not bev.Undistorter.from_config(cam, SIZE).enabled   # 패스스루
    assert cam["note"]                                # 이상 조합이므로 안내가 남는다


@pytest.mark.parametrize("legacy", [True, False])
def test_legacy_calibrated_key_means_both(tmp_path, legacy):
    """하위호환: 구버전 단일 키 calibrated 는 '둘 다' 의 의미."""
    cam = bev.load_camera_config(write_yaml(tmp_path, yaml_body(
        model="plumb_bob", calibrated=legacy)))
    assert cam["intrinsics_calibrated"] is legacy
    assert cam["h_calibrated"] is legacy
    assert cam["calibrated"] is legacy


def test_split_flags_win_over_legacy_key(tmp_path):
    """분리 플래그가 있으면 구버전 calibrated 보다 우선한다."""
    cam = bev.load_camera_config(write_yaml(tmp_path, yaml_body(
        model="plumb_bob", calibrated=True, intrinsics=True, h=False)))
    assert cam["h_calibrated"] is False and cam["calibrated"] is False
    np.testing.assert_allclose(cam["H"], bev.H_PIX2LIDAR_SIM_FALLBACK)


def test_new_K_ignored_when_h_not_calibrated(tmp_path):
    """new_K 는 H 와 짝(같은 undistort 픽셀 좌표계) — H 가 무효면 같이 버린다."""
    forced = [[401.0, 0.0, 321.0], [0.0, 402.0, 241.0], [0.0, 0.0, 1.0]]
    cam = bev.load_camera_config(write_yaml(tmp_path, yaml_body(
        model="plumb_bob", calibrated=False, intrinsics=True, h=False,
        new_K=forced)))
    assert cam["new_K"] is None
    assert "new_K" in cam["note"]

    # 런타임은 모델 규약대로 스스로 new_K 를 계산한다 (alpha=0.0)
    u = bev.Undistorter.from_config(cam, SIZE)
    expect = bev.Undistorter(REAL_K, REAL_D, SIZE,
                             model="plumb_bob", alpha=0.0).new_K
    np.testing.assert_allclose(u.new_K, expect)


def test_resolve_calib_flags_direct():
    assert bev.resolve_calib_flags({}) == (False, False)
    assert bev.resolve_calib_flags(None) == (False, False)
    assert bev.resolve_calib_flags({"calibrated": True}) == (True, True)
    assert bev.resolve_calib_flags({"calibrated": False}) == (False, False)
    # 분리 플래그가 우선, 없는 쪽만 calibrated 로 채운다
    assert bev.resolve_calib_flags(
        {"calibrated": True, "h_calibrated": False}) == (True, False)
    assert bev.resolve_calib_flags(
        {"intrinsics_calibrated": True}) == (True, False)


def test_image_size_from_yaml(tmp_path):
    """image_width/height 는 new_K 계산 기준 해상도 — 노드가 기동 때 쓴다."""
    cam = bev.load_camera_config(write_yaml(tmp_path, yaml_body(
        model="plumb_bob", intrinsics=True, h=False, size=SIZE)))
    assert cam["image_size"] == SIZE
    cam_no_size = bev.load_camera_config(write_yaml(tmp_path, yaml_body(
        model="plumb_bob", intrinsics=True, h=False)))
    assert cam_no_size["image_size"] is None


# ============================================================
# (g) 저장소의 실제 config/camera.yaml 규약 검사
# ============================================================

CONFIG_YAML = pathlib.Path(__file__).resolve().parents[1] / "config" / "camera.yaml"


@pytest.mark.skipif(not CONFIG_YAML.is_file(), reason="config/camera.yaml 없음")
def test_repo_camera_yaml_carries_real_intrinsics():
    """배포 사고 방지: 저장소 camera.yaml 이 실측 K/D 를 담고 있어야 한다.

    로컬이 자리표시자(단위행렬)인 채로 rsync --delete 하면 차의 실측 캘리브가
    덮여 날아간다. 이 테스트가 그 상태를 배포 전에 잡는다.
    """
    cam = bev.load_camera_config(str(CONFIG_YAML))
    assert cam["intrinsics_calibrated"] is True
    assert cam["model"] == bev.MODEL_PLUMB_BOB
    assert not np.allclose(cam["K"], np.eye(3))       # 자리표시자 금지
    assert np.any(cam["D"])
    assert cam["K"][0, 0] == pytest.approx(438.783367, abs=1e-3)
    assert cam["K"][1, 1] == pytest.approx(437.302876, abs=1e-3)
    assert cam["image_size"] == (640, 480)
    assert bev.Undistorter.from_config(cam, cam["image_size"]).enabled
    if cam["h_calibrated"]:
        # H 를 잡은 뒤에는 new_K 가 반드시 짝으로 기록돼 있어야 한다
        assert cam["new_K"] is not None
