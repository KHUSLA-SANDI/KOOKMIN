#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""체커보드 자동 촬영 — 커버리지를 실시간으로 채워가며 스스로 저장한다.

왜 이 도구가 필요한가
  체커보드 캘리브가 실패하는 이유는 거의 항상 "촬영 커버리지 부족" 하나다.
  화면 중앙에서 정면으로만 15장 찍으면
    · 화면 가장자리에 데이터가 없어 k2/k3 가 외삽이 되고,
    · 판이 기울지 않아 초점거리(fx/fy)가 원리적으로 결정되지 않는다.
  그런데 촬영자는 그걸 알 방법이 없다. 그래서 무엇이 남았는지 화면에 띄우고,
  조건을 만족하는 순간 **자동으로** 저장한다 (손으로 판 들고 키를 누를 수 없으니까).

자동 저장 조건 (4개 전부 만족해야 저장)
  1. 판의 내부 코너가 전부 검출됨      → 잘린 장은 애초에 캘리브에 못 쓴다
  2. 정지 상태 (코너 이동 < 1.2 px)     → 흔들리면 코너 정확도가 깎인다
  3. 아직 안 채운 칸/기울기/크기를 채움 → 같은 자세 중복 촬영 방지
  4. 직전 저장 후 최소 간격 경과        → 연사 방지

카메라가 차에 고정돼 낮게(약 18 cm) 있으므로, 판을 **바닥에 놓기만 해도**
카메라 기준으로는 크게 기울어진 상태다 — 손으로 들 필요가 없다.
화면 위쪽(지평선 위)만 판을 책/상자에 기대 세워 채운다.

실행
  # 차에서 (카메라 노드 켜진 상태)
  python3 tools/chess_capture.py --topic /image_raw --out ~/calib_chess
  # 노트북 USB 캠으로 시험
  python3 tools/chess_capture.py --device 0 --out /tmp/chess
  # 저장된 사진들로 커버리지만 채점 (촬영 없이)
  python3 tools/chess_capture.py --dir ~/calib_chess --report

키: [스페이스] 강제 저장   [d] 마지막 장 삭제   [r] 커버리지 초기화   [q] 종료
"""
import argparse
import os
import sys
import threading
import time

import numpy as np
import cv2

_HERE = os.path.dirname(os.path.realpath(__file__))
_PKG = os.path.dirname(_HERE)
sys.path.insert(0, _PKG)

def _load_text_cls():
    """한글 렌더러(tuner.Text)를 같은 폴더에서 직접 로드.

    tools/ 에 __init__.py 가 없어서 'from tools.tuner import Text' 는 실행 위치에
    따라 실패한다 — 그러면 조용히 cv2.putText 폴백으로 떨어져 화면 한글이 전부
    '?' 가 된다. 파일 경로로 직접 로드해 그 사고를 막는다.
    """
    import importlib.util
    p = os.path.join(_HERE, "tuner.py")
    if not os.path.isfile(p):
        return None
    try:
        spec = importlib.util.spec_from_file_location("_tuner_text", p)
        m = importlib.util.module_from_spec(spec)
        sys.modules["_tuner_text"] = m
        old = sys.argv
        sys.argv = ["tuner.py"]          # argparse 는 main() 안이라 안 돌지만 방어
        try:
            spec.loader.exec_module(m)
        finally:
            sys.argv = old
        return getattr(m, "Text", None)
    except Exception:
        return None


Text = _load_text_cls()

# ---- 커버리지 목표 ----
# ★ 크기 기준은 '화면 대비 비율' 이 아니라 **칸 하나가 화면에서 몇 px 인가** 로 잡는다.
#   코너 서브픽셀 검출 윈도우가 11x11 이라, 칸이 그보다 작으면 옆 칸까지 물어서
#   코너 위치가 통째로 어긋난다. 실측 실패 사례: 칸 6.7~11.5px 로 12장 찍었더니
#   사진 2장 빼는 것만으로 fx 가 410 → 526 → 1052 로 날뛰었다(=제약이 전혀 안 됨).
GRID = 3                                # 화면을 3x3 으로 나눠 위치 커버
NEED_PER_CELL = 1                       # 각 칸 최소 장수
TILT_BINS = ["위", "아래", "왼쪽", "오른쪽"]   # 판의 어느 변이 카메라에 가까운가
NEED_PER_TILT = 1
SIZE_BINS = ["큼", "중", "작음"]
NEED_BIG = 4                            # 칸 >= CELL_BIG 인 사진 최소 장수
NEED_TOTAL = 15                         # 총 최소 장수

CELL_MIN = 18.0                         # 이보다 작으면 아예 저장 안 함 (못 쓰는 사진)
CELL_MID = 25.0                         # 작음 / 중 경계
CELL_BIG = 35.0                         # 중 / 큼 경계

TILT_MIN_DEG = 15.0                     # 미만이면 "거의 평행" — 초점거리 제약을 못 줌
TILT_MAX_DEG = 55.0                     # 초과면 너무 눕혀서 코너가 뭉갬 — 저장 안 함

STABLE_PX = 1.2                         # 정지 판정 (코너 평균 이동량)
STABLE_N = 3                            # 연속 프레임 수
MIN_GAP_S = 0.8                         # 저장 최소 간격


# 기울기 판정용 대략적인 K (실차 벤더값). 정확할 필요 없다 — 자세를 '분류' 하는 데만
# 쓰고 캘리브 결과에는 전혀 안 들어간다. 판의 실제 기울기 각도를 도(deg)로 얻기 위한 것.
K_ROUGH = np.array([[438.783367, 0.0, 305.593336],
                    [0.0, 437.302876, 243.738352],
                    [0.0, 0.0, 1.0]])


def board_metrics(corners, bw, bh, shape, sq=0.02):
    """검출된 코너에서 (칸 인덱스, 기울기 방향, 기울기 각도(deg), 크기 등급, 화면비율).

    기울기는 solvePnP 로 판의 자세를 풀어 **판 법선과 시선 사이의 각도**로 재고
    실제 도(deg) 로 준다. 마주보는 변의 길이비로 재면 판이 작거나 멀 때 30도
    기울여도 0.05 밖에 안 나와 '평행' 으로 오판된다 (그래서 그 방식을 버렸다).
    여기 쓰는 K 는 대략값이어도 분류에는 충분하고, 캘리브 결과에는 안 들어간다.
    """
    objp = np.zeros((bw * bh, 3), np.float64)
    objp[:, :2] = np.mgrid[0:bw, 0:bh].T.reshape(-1, 2) * sq
    ip = corners.reshape(-1, 2).astype(np.float64)
    tilt, ang = None, 0.0
    try:
        ok, rvec, tvec = cv2.solvePnP(objp, ip, K_ROUGH, None,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    except cv2.error:
        ok = False
    if ok:
        R, _ = cv2.Rodrigues(rvec)
        n = R[:, 2]                                  # 판 법선 (카메라 좌표계)
        ctr_cam = (R @ objp.mean(axis=0)) + tvec.ravel()
        view = ctr_cam / max(1e-9, np.linalg.norm(ctr_cam))
        if float(n @ view) > 0:                      # 카메라를 향하게 부호 정리
            n = -n
        ang = float(np.degrees(np.arccos(np.clip(-(n @ view), -1.0, 1.0))))
        if ang >= TILT_MIN_DEG:
            # 기울기 방향: 법선을 화면축에 투영해 어느 변이 카메라에 가까운지
            ny, nx = float(n[1]), float(n[0])
            if abs(ny) >= abs(nx):
                tilt = "위" if ny > 0 else "아래"
            else:
                tilt = "왼쪽" if nx > 0 else "오른쪽"

    ctr = ip.mean(axis=0)
    H, W = shape[:2]
    gx = min(GRID - 1, max(0, int(ctr[0] / (W / GRID))))
    gy = min(GRID - 1, max(0, int(ctr[1] / (H / GRID))))

    # ★ 칸 하나가 화면에서 몇 px 인가 — 코너 정밀도의 직접 지표
    g = corners.reshape(bh, bw, 2)
    dx = float(np.linalg.norm(np.diff(g, axis=1), axis=2).mean())
    dy = float(np.linalg.norm(np.diff(g, axis=0), axis=2).mean())
    cell = 0.5 * (dx + dy)
    size = "큼" if cell >= CELL_BIG else ("중" if cell >= CELL_MID else "작음")
    return (gy, gx), tilt, ang, size, cell


class Coverage:
    def __init__(self):
        self.cells = np.zeros((GRID, GRID), int)
        self.tilts = {k: 0 for k in TILT_BINS}
        self.sizes = {k: 0 for k in SIZE_BINS}
        self.flat = 0
        self.shots = []          # [(path, cell, tilt, size)]

    def wants(self, cell, tilt, size):
        """이 자세가 아직 필요한가 (뭘 채우는지 문자열로 반환, 아니면 None)."""
        need = []
        if self.cells[cell] < NEED_PER_CELL:
            need.append("칸%d%d" % cell)
        if tilt is not None and self.tilts[tilt] < NEED_PER_TILT:
            need.append("기울기:" + tilt)
        if size == "큼" and self.sizes["큼"] < NEED_BIG:
            need.append("큼 %d/%d" % (self.sizes["큼"] + 1, NEED_BIG))
        elif size == "중" and self.sizes["중"] < 2:
            need.append("중 %d/2" % (self.sizes["중"] + 1))
        if self.total() < NEED_TOTAL and size != "작음":
            need.append("총 %d/%d" % (self.total() + 1, NEED_TOTAL))
        return " + ".join(need) if need else None

    def add(self, path, cell, tilt, size):
        self.cells[cell] += 1
        if tilt is None:
            self.flat += 1
        else:
            self.tilts[tilt] += 1
        self.sizes[size] += 1
        self.shots.append((path, cell, tilt, size))

    def remove_last(self):
        if not self.shots:
            return None
        path, cell, tilt, size = self.shots.pop()
        self.cells[cell] -= 1
        if tilt is None:
            self.flat = max(0, self.flat - 1)
        else:
            self.tilts[tilt] = max(0, self.tilts[tilt] - 1)
        self.sizes[size] = max(0, self.sizes[size] - 1)
        return path

    def missing(self):
        m = []
        NAME = [["좌상", "상중앙", "우상"], ["좌중", "정중앙", "우중"],
                ["좌하", "하중앙", "우하"]]
        for r in range(GRID):
            for c in range(GRID):
                if self.cells[r, c] < NEED_PER_CELL:
                    m.append(NAME[r][c])
        for k in TILT_BINS:
            if self.tilts[k] < NEED_PER_TILT:
                m.append("기울기(%s)" % k)
        if self.sizes["큼"] < NEED_BIG:
            m.append("가까이(칸%.0fpx↑) %d/%d" % (CELL_BIG, self.sizes["큼"], NEED_BIG))
        if self.sizes["중"] < 2:
            m.append("중간거리 %d/2" % self.sizes["중"])
        if self.total() < NEED_TOTAL:
            m.append("총장수 %d/%d" % (self.total(), NEED_TOTAL))
        return m

    def done(self):
        return not self.missing()

    def total(self):
        return len(self.shots)


class Source:
    """--device / --topic 중 하나에서 프레임을 준다."""

    def __init__(self, args):
        self.args = args
        self._lock = threading.Lock()
        self._frame = None
        self.kind = None
        if args.topic:
            self.kind = "ros"
            self._start_ros()
        elif args.device is not None:
            self.kind = "cam"
            self.cap = cv2.VideoCapture(int(args.device))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not self.cap.isOpened():
                print("USB 캠 %s 를 열 수 없다" % args.device)
                sys.exit(1)
        else:
            print("--topic 또는 --device 중 하나가 필요하다")
            sys.exit(1)

    def _start_ros(self):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image
        from track_drive.lib import preprocess as PP
        rclpy.init()
        node = Node("chess_capture")

        def cb(msg):
            bgr, _ = PP.image_msg_to_bgr(msg)
            if bgr is not None:
                with self._lock:
                    self._frame = bgr
        node.create_subscription(Image, self.args.topic, cb, qos_profile_sensor_data)
        self._node, self._rclpy = node, rclpy
        threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
        print("구독 %s — 첫 프레임 대기…" % self.args.topic)

    def read(self):
        if self.kind == "cam":
            ok, f = self.cap.read()
            return f if ok else None
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def close(self):
        if self.kind == "cam":
            self.cap.release()
        else:
            try:
                self._node.destroy_node()
                if self._rclpy.ok():
                    self._rclpy.shutdown()
            except Exception:
                pass


def find_board(gray, bw, bh):
    """코너 검출 + 서브픽셀 정밀화. 실패 시 None."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(gray, (bw, bh), flags=flags)
    if not ok:
        return None
    cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    return corners


def draw_panel(frame, cov, corners, bw, bh, status, info, TXT):
    vis = frame.copy()
    H, W = vis.shape[:2]
    # 3x3 안내선
    for i in range(1, GRID):
        cv2.line(vis, (W * i // GRID, 0), (W * i // GRID, H - 1), (70, 70, 70), 1)
        cv2.line(vis, (0, H * i // GRID), (W - 1, H * i // GRID), (70, 70, 70), 1)
    # 채운 칸은 초록 반투명
    ov = vis.copy()
    for r in range(GRID):
        for c in range(GRID):
            if cov.cells[r, c] >= NEED_PER_CELL:
                cv2.rectangle(ov, (W * c // GRID, H * r // GRID),
                              (W * (c + 1) // GRID - 1, H * (r + 1) // GRID - 1),
                              (0, 120, 0), -1)
    vis = cv2.addWeighted(vis, 0.85, ov, 0.15, 0)
    if corners is not None:
        cv2.drawChessboardCorners(vis, (bw, bh), corners, True)

    pad = np.full((150, W, 3), 22, np.uint8)
    items = []
    NAME = [["좌상", "상중앙", "우상"], ["좌중", "정중앙", "우중"],
            ["좌하", "하중앙", "우하"]]
    # 3x3 커버 표
    for r in range(GRID):
        row = "  ".join("%s%s" % ("O" if cov.cells[r, c] >= NEED_PER_CELL else ".",
                                  ("%d" % cov.cells[r, c]) if cov.cells[r, c] else " ")
                        for c in range(GRID))
        items.append((row, 10, 8 + r * 17, 14, (150, 230, 150)))
    items.append(("기울기 " + " ".join("%s%d" % (k, cov.tilts[k]) for k in TILT_BINS)
                  + "   평행%d" % cov.flat, 110, 8, 14, (180, 200, 255)))
    items.append(("칸px  큼(>=%.0f) %d/%d   중 %d   작음 %d"
                  % (CELL_BIG, cov.sizes["큼"], NEED_BIG, cov.sizes["중"],
                     cov.sizes["작음"]), 110, 25, 14, (180, 200, 255)))
    items.append(("저장 %d/%d장" % (cov.total(), NEED_TOTAL), 110, 42, 15,
                  (140, 255, 140) if cov.done() else (255, 255, 140)))
    items.append((status, 10, 66, 16, info), )
    miss = cov.missing()
    if miss:
        items.append(("남은 것: " + ", ".join(miss[:8])
                      + (" …외 %d" % (len(miss) - 8) if len(miss) > 8 else ""),
                      10, 90, 13, (255, 220, 140)))
    else:
        items.append(("커버리지 완료 — [q] 로 끝내고 calib_camera.py 실행",
                      10, 90, 15, (140, 255, 140)))
    items.append(("판을 카메라에서 25~45cm 에 두고 20~45도 기울일 것 (칸 0px 이상).  "
                  "[스페이스]강제저장 [d]마지막삭제 [r]초기화 [q]종료",
                  10, 112, 12, (150, 150, 150)))
    out = cv2.vconcat([vis, pad])
    if TXT is not None:
        # pad 영역만 좌표 보정해서 그린다
        sub = out[H:, :]
        TXT.draw(sub, [(t, x, y, s, c) for (t, x, y, s, c) in items])
        out[H:, :] = sub
    else:
        for (t, x, y, s, c) in items:
            cv2.putText(out, t, (x, H + y + s), cv2.FONT_HERSHEY_SIMPLEX,
                        s / 30.0, c, 1, cv2.LINE_AA)
    return out


def report_dir(args):
    """이미 찍어둔 폴더의 커버리지를 채점만 한다."""
    bw, bh = [int(t) for t in args.board.lower().split("x")]
    sq = float(args.square)
    files = sorted(f for f in os.listdir(args.dir)
                   if f.lower().endswith((".jpg", ".png")))
    cov = Coverage()
    fail = []
    for f in files:
        img = cv2.imread(os.path.join(args.dir, f))
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        c = find_board(g, bw, bh)
        if c is None:
            fail.append(f)
            continue
        cell, tilt, mag, size, cpx = board_metrics(c, bw, bh, img.shape, sq)
        cov.add(f, cell, tilt, size)
        bad = []
        if cpx < CELL_MIN: bad.append("칸작음")
        if mag > TILT_MAX_DEG: bad.append("과도기울기")
        print("  %-22s 칸%d%d  %5.1f도  칸 %5.1fpx (%s)%s"
              % (f, cell[0], cell[1], mag, cpx, size,
                 "  <- " + "/".join(bad) if bad else ""))
    print("\n검출 성공 %d / 전체 %d장" % (cov.total(), len(files)))
    if fail:
        print("검출 실패(캘리브에 못 씀): %s" % ", ".join(fail))
    print("3x3 커버:\n%s" % cov.cells)
    print("기울기: %s  평행 %d" % (cov.tilts, cov.flat))
    print("크기: %s" % cov.sizes)
    miss = cov.missing()
    print("남은 것: %s" % (", ".join(miss) if miss else "없음 — 충분함 ✅"))
    return 0 if not miss else 1


def main():
    ap = argparse.ArgumentParser(description="체커보드 자동 촬영 (커버리지 안내)")
    ap.add_argument("--topic", default=None, help="ROS2 이미지 토픽 (예: /image_raw)")
    ap.add_argument("--device", default=None, help="USB 캠 번호 (예: 0)")
    ap.add_argument("--out", default=os.path.expanduser("~/calib_chess"))
    ap.add_argument("--board", default="9x6", help="내부 코너 수 가로x세로")
    ap.add_argument("--square", default=0.02, type=float,
                    help="칸 길이 m — 기울기 각도 계산에만 쓰인다 (캘리브 결과와 무관)")
    ap.add_argument("--dir", default=None, help="이 폴더의 커버리지만 채점 (--report)")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    bw, bh = [int(t) for t in args.board.lower().split("x")]
    if args.report or args.dir:
        if not args.dir:
            args.dir = args.out
        sys.exit(report_dir(args))

    os.makedirs(args.out, exist_ok=True)
    TXT = Text() if Text is not None else None
    src = Source(args)
    cov = Coverage()
    prev = None
    stable = 0
    last_save = 0.0
    n = len(os.listdir(args.out))
    win = "chess capture"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    print("판을 바닥에 놓고 천천히 움직이면 자동 저장된다. 목표: 3x3 전부 + 기울기 4방향 + 크기 3단계")

    while True:
        frame = src.read()
        if frame is None:
            time.sleep(0.03)
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_board(gray, bw, bh)

        status, color = "판이 안 보임 — 화면에 판 전체가 들어오게", (150, 150, 150)
        force = False
        if corners is not None:
            cur = corners.reshape(-1, 2)
            if prev is not None and prev.shape == cur.shape:
                mv = float(np.linalg.norm(cur - prev, axis=1).mean())
                stable = stable + 1 if mv < STABLE_PX else 0
            else:
                stable = 0
            prev = cur
            cell, tilt, mag, size, cpx = board_metrics(corners, bw, bh,
                                                      frame.shape, float(args.square))
            # 못 쓰는 사진은 아예 저장하지 않는다 (섞이면 결과가 통째로 망가진다)
            if cpx < CELL_MIN:
                status = ("판이 너무 작다 — 칸 %.0fpx (최소 %.0f). 카메라 쪽으로 더 가까이"
                          % (cpx, CELL_MIN))
                color = (100, 160, 255)
                cv2.imshow(win, draw_panel(frame, cov, corners, bw, bh, status, color, TXT))
                if (cv2.waitKeyEx(20) & 0xFF) in (27, ord('q')):
                    break
                continue
            if mag > TILT_MAX_DEG:
                status = ("너무 눕혔다 — %.0f도 (최대 %.0f). 코너가 뭉개짐, 세워서"
                          % (mag, TILT_MAX_DEG))
                color = (100, 160, 255)
                cv2.imshow(win, draw_panel(frame, cov, corners, bw, bh, status, color, TXT))
                if (cv2.waitKeyEx(20) & 0xFF) in (27, ord('q')):
                    break
                continue
            want = cov.wants(cell, tilt, size)
            if want is None:
                status = ("검출됨 (칸%d%d / %.0f도 / 칸%.0fpx) — 이미 채운 자세, 옮기세요"
                          % (cell[0], cell[1], mag, cpx))
                color = (140, 200, 255)
            elif stable < STABLE_N:
                status = "검출됨 — 움직임 멈추는 중 (%d/%d)" % (stable, STABLE_N)
                color = (0, 220, 255)
            elif time.time() - last_save < MIN_GAP_S:
                status = "저장 간격 대기"
                color = (0, 220, 255)
            else:
                path = os.path.join(args.out, "chess_%04d.jpg" % n)
                cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
                cov.add(path, cell, tilt, size)
                n += 1
                last_save = time.time()
                stable = 0
                status = "저장! → %s  (%s)" % (os.path.basename(path), want)
                color = (140, 255, 140)
                print("  저장 %s  칸%d%d 기울기 %s %.0f도 칸%.0fpx(%s) — %s"
                      % (os.path.basename(path), cell[0], cell[1],
                         tilt or "평행", mag, cpx, size, want))
        else:
            prev, stable = None, 0

        cv2.imshow(win, draw_panel(frame, cov, corners, bw, bh, status, color, TXT))
        k = cv2.waitKeyEx(20) & 0xFF
        if k in (27, ord('q')):
            break
        elif k == ord(' ') and corners is not None:
            cell, tilt, mag, size, cpx = board_metrics(corners, bw, bh,
                                                      frame.shape, float(args.square))
            path = os.path.join(args.out, "chess_%04d.jpg" % n)
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 97])
            cov.add(path, cell, tilt, size)
            n += 1
            print("  강제저장 %s" % os.path.basename(path))
        elif k == ord('d'):
            p = cov.remove_last()
            if p:
                try:
                    os.remove(p)
                except OSError:
                    pass
                print("  삭제 %s" % os.path.basename(p))
        elif k == ord('r'):
            cov = Coverage()
            print("  커버리지 초기화 (파일은 그대로)")

    cv2.destroyAllWindows()
    src.close()
    print("\n=== 촬영 결과 ===")
    print("저장 %d장 → %s" % (cov.total(), args.out))
    print("3x3 커버:\n%s" % cov.cells)
    miss = cov.missing()
    if miss:
        print("▲ 아직 남은 것: %s" % ", ".join(miss))
        print("   이 상태로 캘리브하면 그 영역은 외삽이 된다 — 더 찍는 게 좋다.")
    else:
        print("커버리지 충분 ✅")
    print("\n다음:\n  python3 tools/calib_camera.py --dir %s --board %s "
          "--square <실측 칸길이 m> --save config/camera.yaml" % (args.out, args.board))


if __name__ == "__main__":
    main()
