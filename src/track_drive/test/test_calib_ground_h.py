"""tools/calib_ground_h.py 의 --grid / --enhance / 확대클릭 경로 테스트 (GUI 불필요).

검증 범위:
  (a) --grid 왕복: 알려진 호모그래피로 타일 격자점을 픽셀에 투영 → 클릭/격자번호 입력을
      몽키패치로 주입 → 복원된 H 의 재투영 RMS < 5mm
  (b) --enhance 는 표시 전용 — CLAHE+언샤프를 걸어도 수집 좌표/H 가 비트 단위로 동일
  (c) 확대(4x) 클릭 경로: 창 픽셀 → 원본 픽셀 역변환이 정확한 상수배라
      1x 정수 클릭보다 양자화 오차가 작아진다 (서브픽셀 조준이 실제로 먹는지)
  (d) 격자번호 파싱 / 프레임 평행이동(_shift_frame_x) / CSV 인라인 주석 왕복

GUI 는 cv2 창 함수를 전부 몽키패치해 대체한다 — 헤드리스 CI 에서도 돈다.
"""

import builtins
import importlib.util
import pathlib

import cv2
import numpy as np
import pytest

TOOL_PATH = (pathlib.Path(__file__).resolve().parents[1]
             / "tools" / "calib_ground_h.py")


def _load_tool():
    spec = importlib.util.spec_from_file_location("_calib_ground_h", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cgh = _load_tool()

W, H_IMG = 640, 480
PITCH = 0.453333            # 본선 복도 테라조 실측 2.72m / 6칸
RMS_LIMIT_M = 0.005         # 요구 조건: 재투영 RMS < 5mm


# ======================== 합성 지면→픽셀 카메라 ========================

def _ground_to_pixel(f=700.0, h_cam=1.0, theta_deg=35.0, cx=320.0, cy=240.0):
    """지면 (x전방, y좌) → 픽셀 (u,v) 의 알려진 호모그래피.

    카메라는 높이 h_cam 에서 theta 만큼 아래로 기울어져 앞을 본다.
    실차(높이 0.2m·거의 수평)보다 내려다보는 각을 크게 잡았다 — 여기서 확인할 것은
    '클릭 → 격자번호 → H' 배관이지 실차 기하가 아니고, 시선이 스치듯 깔리면
    정수 픽셀 클릭의 양자화만으로 cm 급 오차가 나 테스트가 무의미해지기 때문.
    (그 민감도 자체는 test_zoom_click_beats_integer_click 이 따로 다룬다)
    """
    t = np.radians(theta_deg)
    # (x,y,0) → 카메라 좌표 (X=-y, Y=-x sin t + h cos t, Z=x cos t + h sin t)
    M = np.array([[0.0, -1.0, 0.0],
                  [-np.sin(t), 0.0, h_cam * np.cos(t)],
                  [np.cos(t), 0.0, h_cam * np.sin(t)]])
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    return K @ M


def _lattice_pixels(rows=range(1, 7), cols=(-2, -1, 0, 1, 2), pitch=PITCH,
                    origin=(0.0, 0.0), **cam):
    """격자번호 (i,j) → (지면 x,y, 픽셀 u,v). 화면 밖으로 나가는 점은 버린다."""
    G = _ground_to_pixel(**cam)
    out = []
    for i in rows:
        for j in cols:
            x = origin[0] + i * pitch
            y = origin[1] + j * pitch
            p = G @ np.array([x, y, 1.0])
            if p[2] <= 1e-9:
                continue
            u, v = p[0] / p[2], p[1] / p[2]
            if 2.0 <= u <= W - 3.0 and 2.0 <= v <= H_IMG - 3.0:
                out.append(((float(i), float(j)), (x, y), (u, v)))
    return out


def _tile_image():
    """줄눈이 있는 가짜 테라조 — 렌더/매그니파이어 경로를 실제로 돌리기 위한 배경."""
    img = np.full((H_IMG, W, 3), 150, np.uint8)
    rng = np.random.default_rng(0)
    img = np.clip(img.astype(np.int16)
                  + rng.integers(-12, 12, img.shape), 0, 255).astype(np.uint8)
    for xx in range(0, W, 37):
        img[:, xx:xx + 2] = 120
    for yy in range(0, H_IMG, 29):
        img[yy:yy + 2, :] = 120
    return img


# ======================== GUI 몽키패치 (클릭/키 주입) ========================

class FakeGui:
    """cv2 창 함수 대역 — 스크립트대로 마우스/키 이벤트를 흘려 넣는다.

    액션: ("at", u, v)   현재 뷰포트 기준으로 원본좌표 (u,v) 가 보이는 창 픽셀을
                         계산해 좌클릭 (실사용자가 화면에서 그 점을 찍는 것과 동일)
          ("pan", u, v)  같은 방식으로 우클릭 = 그 지점으로 뷰포트 이동
          ("key", "c")   키 입력
    스크립트가 바닥나면 'q' 를 돌려 수집 루프를 끝낸다 (무한루프 방지).
    """

    def __init__(self, script):
        self.script = list(script)
        self.cb = None
        self.viewer = None
        self.frames = []

    # --- cv2 대역 ---
    def namedWindow(self, *_a, **_k):
        pass

    def setMouseCallback(self, _win, cb, *_a):
        self.cb = cb
        self.viewer = getattr(cb, "__self__", None)   # 바인드 메서드 → 뷰어 객체

    def imshow(self, _win, img):
        self.frames.append(img)

    def destroyAllWindows(self):
        pass

    def waitKey(self, delay=0):
        if delay is not None and delay <= 1:
            return 255            # 클릭 강조용 즉시 갱신 — 스크립트를 소비하지 않는다
        if not self.script:
            return ord("q")
        kind, a, b = self.script.pop(0)
        if kind == "key":
            return ord(a)
        wx, wy = self.viewer.img_to_win(a, b)
        wx, wy = int(round(wx)), int(round(wy))
        event = (cv2.EVENT_LBUTTONDOWN if kind == "at" else cv2.EVENT_RBUTTONDOWN)
        self.cb(event, wx, wy, 0, None)
        return 255


def _install(monkeypatch, gui, answers):
    for name in ("namedWindow", "setMouseCallback", "imshow", "waitKey",
                 "destroyAllWindows"):
        monkeypatch.setattr(cv2, name, getattr(gui, name))
    queue = list(answers)

    def fake_input(_prompt=""):
        assert queue, "콘솔 입력 스크립트 소진 — 예상보다 많이 물었다"
        return queue.pop(0)

    monkeypatch.setattr(builtins, "input", fake_input)
    return queue


def _run_grid(monkeypatch, base, lattice, zoom4=False, origin=(0.0, 0.0)):
    """--grid 수집을 스크립트로 돌려 (new_pts, labels) 를 얻는다."""
    script, answers = [], []
    for (i, j), _xy, (u, v) in lattice:
        if zoom4:
            # 실사용 순서: 그 부근으로 패닝 → [+] 두 번(1x→2x→4x) → 확대 상태에서 클릭
            #             → [-] 두 번으로 1x 복귀(다음 점을 화면에서 찾기 위해)
            script += [("pan", u, v), ("key", "+", None), ("key", "+", None),
                       ("at", u, v), ("key", "-", None), ("key", "-", None)]
        else:
            script.append(("at", u, v, None)[:3])
        answers.append("%g %g" % (i, j))
    gui = FakeGui(script)
    _install(monkeypatch, gui, answers)
    return cgh._collect_grid(base, PITCH, origin, [])


# ======================== (a) --grid 왕복 정확도 ========================

def test_grid_roundtrip_rms_under_5mm(monkeypatch):
    lat = _lattice_pixels()
    assert len(lat) >= 8, "합성 격자점이 너무 적다"
    pts, labels = _run_grid(monkeypatch, _tile_image(), lat)

    assert len(pts) == len(lat)
    assert labels == [ij for ij, _xy, _uv in lat]
    # 격자번호 → 지면좌표 변환이 (origin + i*pitch, origin + j*pitch) 인지
    for (u, v, x, y), (ij, xy, _uv) in zip(pts, lat):
        assert x == pytest.approx(ij[0] * PITCH, abs=1e-12)
        assert y == pytest.approx(ij[1] * PITCH, abs=1e-12)
        assert (x, y) == pytest.approx(xy, abs=1e-12)

    H, rms = cgh._compute_h(pts)
    assert rms < RMS_LIMIT_M, "재투영 RMS %.4fm >= %.3fm" % (rms, RMS_LIMIT_M)

    # 복원된 H 가 원 지면좌표를 되살리는지 (점별 최대 오차도 확인)
    arr = np.asarray(pts, np.float64)
    proj = cv2.perspectiveTransform(
        arr[:, :2].reshape(-1, 1, 2), H).reshape(-1, 2)
    assert np.abs(proj - arr[:, 2:]).max() < 0.01


def test_grid_origin_offset_shifts_all_points(monkeypatch):
    """--origin-x/y 는 격자 전체를 그대로 평행이동시킨다."""
    lat = _lattice_pixels()[:6]
    org = (0.12, -0.34)
    pts, _lab = _run_grid(monkeypatch, _tile_image(), lat, origin=org)
    for (u, v, x, y), (ij, _xy, _uv) in zip(pts, lat):
        assert x == pytest.approx(org[0] + ij[0] * PITCH, abs=1e-12)
        assert y == pytest.approx(org[1] + ij[1] * PITCH, abs=1e-12)


# ======================== (b) --enhance 는 기하 불변 ========================

def test_enhance_keeps_shape_but_changes_pixels():
    base = _tile_image()
    out = cgh._enhance_for_display(base)
    assert out.shape == base.shape and out.dtype == base.dtype
    assert not np.array_equal(out, base), "강화가 아무 일도 안 했다"


def test_enhance_does_not_move_any_coordinate(monkeypatch):
    """같은 클릭 스크립트를 원본/강화 이미지로 돌려 수집 좌표가 비트 단위로 같은지."""
    lat = _lattice_pixels()
    plain, _l1 = _run_grid(monkeypatch, _tile_image(), lat)
    enhanced, _l2 = _run_grid(monkeypatch,
                              cgh._enhance_for_display(_tile_image()), lat)
    assert plain == enhanced, "--enhance 가 클릭 좌표를 움직였다"

    H_a, rms_a = cgh._compute_h(plain)
    H_b, rms_b = cgh._compute_h(enhanced)
    assert np.array_equal(H_a, H_b)
    assert rms_a == rms_b


# ======================== (c) 확대 클릭 = 서브픽셀 조준 ========================

def test_zoom_click_beats_integer_click(monkeypatch):
    """4x 확대 클릭이 1x 정수 클릭보다 양자화 오차가 작아야 한다.

    창 픽셀 → 원본 픽셀 역변환이 정확한 1/zoom 배라는 것의 실질 증명.
    """
    lat = _lattice_pixels()
    p1, _a = _run_grid(monkeypatch, _tile_image(), lat, zoom4=False)
    p4, _b = _run_grid(monkeypatch, _tile_image(), lat, zoom4=True)

    truth = np.array([uv for _ij, _xy, uv in lat])
    e1 = np.abs(np.array(p1)[:, :2] - truth).max()
    e4 = np.abs(np.array(p4)[:, :2] - truth).max()
    assert e1 <= 0.5001, "1x 클릭 오차는 반픽셀 이내여야 한다: %.3f" % e1
    assert e4 <= 0.1251, "4x 클릭 오차는 1/8 픽셀 이내여야 한다: %.3f" % e4
    assert e4 < e1

    _H4, rms4 = cgh._compute_h(p4)
    _H1, rms1 = cgh._compute_h(p1)
    assert rms4 < RMS_LIMIT_M and rms4 <= rms1


def test_viewer_window_to_image_mapping_is_exact(monkeypatch):
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda *a, **k: None)
    view = cgh._ZoomViewer(_tile_image())
    for zi, zoom in enumerate(cgh.ZOOM_LEVELS):
        view.zi = zi
        x0, y0, vw, vh = view.viewport()
        assert 0 <= x0 and x0 + vw <= W and 0 <= y0 and y0 + vh <= H_IMG
        for mx, my in ((0, 0), (13, 7), (W - 1, H_IMG - 1)):
            u, v = view.win_to_img(mx, my)
            assert u == pytest.approx(x0 + mx / zoom, abs=1e-12)
            assert v == pytest.approx(y0 + my / zoom, abs=1e-12)
            bx, by = view.img_to_win(u, v)
            assert (bx, by) == pytest.approx((mx, my), abs=1e-9)
        # 렌더가 창 크기를 유지해야 위 역변환이 성립한다 (매그니파이어 경로도 같이 탄다)
        img = view.render([(x0 + 5.0, y0 + 5.0, "(1,0)", cgh.COLOR_NEW)], "test")
        assert img.shape == (H_IMG, W, 3)


# ======================== (d) 파싱 / 프레임 / CSV ========================

@pytest.mark.parametrize("text,want", [
    ("2 1", (2.0, 1.0)),
    ("2,1", (2.0, 1.0)),
    ("  3   -2 ", (3.0, -2.0)),
    ("2 0.5", (2.0, 0.5)),        # 반칸 지점
    ("-1.5,-0.5", (-1.5, -0.5)),
])
def test_parse_grid_ij_ok(text, want):
    assert cgh._parse_grid_ij(text) == want


@pytest.mark.parametrize("text", ["", "2", "2 3 4", "a b", "2 x", ","])
def test_parse_grid_ij_rejects(text):
    assert cgh._parse_grid_ij(text) is None


def test_ask_grid_ij_reasks_until_valid(monkeypatch):
    answers = ["oops", "1 2 3", "2 1"]
    monkeypatch.setattr(builtins, "input", lambda *_a: answers.pop(0))
    assert cgh._ask_grid_ij(1, 10.0, 20.0, []) == (2.0, 1.0)
    assert not answers


def test_ask_grid_ij_blank_cancels(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda *_a: "")
    assert cgh._ask_grid_ij(1, 10.0, 20.0, []) is None


def test_shift_frame_x():
    pts = [[10.0, 20.0, 1.000, 0.500], [30.0, 40.0, 2.000, -0.250]]
    same = cgh._shift_frame_x(pts, 0.0)
    assert same == pts and same is not pts          # 0 이면 값 보존 + 복사본
    moved = cgh._shift_frame_x(pts, 0.117)
    assert moved[0] == pytest.approx([10.0, 20.0, 0.883, 0.5])
    assert moved[1] == pytest.approx([30.0, 40.0, 1.883, -0.25])
    assert pts[0][2] == 1.000                       # 원본 불변


def test_shift_frame_x_moves_h_output_by_offset():
    """오프셋을 준 대응점으로 만든 H 는 출력 x 가 정확히 그만큼 줄어든다."""
    lat = _lattice_pixels()
    pts = [[u, v, x, y] for _ij, (x, y), (u, v) in lat]
    H0, _r0 = cgh._compute_h(pts)
    H1, _r1 = cgh._compute_h(cgh._shift_frame_x(pts, 0.117))
    uv = np.asarray(pts, np.float64)[:, :2].reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(uv, H0).reshape(-1, 2)
    b = cv2.perspectiveTransform(uv, H1).reshape(-1, 2)
    assert np.allclose(a[:, 0] - b[:, 0], 0.117, atol=1e-6)
    assert np.allclose(a[:, 1], b[:, 1], atol=1e-6)


def test_points_csv_roundtrip_with_grid_labels(tmp_path):
    csv = tmp_path / "pts.csv"
    pts = [[100.0, 200.0, 0.4533, 0.0], [150.25, 210.5, 0.9067, 0.4533]]
    labels = [(1.0, 0.0), (2.0, 1.0)]
    cgh._append_points_csv(str(csv), pts, "unit test (frame=A_prime)", labels)
    text = csv.read_text(encoding="utf-8")
    assert "# grid 1,0" in text and "# grid 2,1" in text

    back = cgh._load_points_csv(str(csv))          # 인라인 주석을 무시해야 한다
    assert len(back) == 2
    assert back[0] == pytest.approx([100.0, 200.0, 0.4533, 0.0], abs=1e-4)
    assert back[1] == pytest.approx([150.25, 210.5, 0.9067, 0.4533], abs=1e-4)

    # 라벨 없이 이어 써도(다른 모드) 같은 파일에서 읽힌다
    cgh._append_points_csv(str(csv), [[1.0, 2.0, 3.0, 4.0]], "click")
    assert len(cgh._load_points_csv(str(csv))) == 3


def test_frame_names_cover_cli_choices():
    assert cgh.FRAME_NAMES["a_prime"] == "A_prime"
    assert cgh.FRAME_NAMES["lidar"] == "lidar_frame"
