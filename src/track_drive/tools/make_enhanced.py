#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""바닥 사진 선명화 — CLAHE + 언샤프 (클릭 보조용 초간단 도구).

역할: 테라조 바닥처럼 줄눈 대비가 약한 사진을 또렷하게 만들어
      calib_ground_h.py 의 클릭 정확도를 높인다.
입력: in.jpg (왜곡 있는 raw 든 undistort 된 것이든 상관없음)
출력: out.jpg
예선 대비 변경점: 신규 도구 (시뮬 바닥은 대비가 충분해 필요 없었음).

★ 기하는 절대 건드리지 않는다 — 밝기/대비만 손대므로 픽셀 위치가 불변이다.
  그래서 이 결과 이미지 위에서 클릭한 (u,v) 를 원본 이미지의 (u,v) 로 그대로 써도 된다.
  (리사이즈·크롭·워프를 여기에 추가하면 그 전제가 깨진다 — 넣지 말 것)

사용법:
  python3 make_enhanced.py in.jpg out.jpg
  # 예) 클릭용 이미지 만들고 그 위에서 H 캘리브
  python3 make_enhanced.py ~/calib_raw.jpg /tmp/calib_enh.jpg
  python3 calib_ground_h.py --image /tmp/calib_enh.jpg --camera ../config/camera.yaml
  # bev_check.py --enhance 는 이 파일의 enhance_floor() 를 그대로 불러 쓴다
  # (두 도구가 같은 그림을 보게 하려고 파라미터를 여기 한 곳에만 둔다)
"""

import os
import sys

import cv2

CLAHE_CLIP = 2.5        # 대비 제한. 너무 키우면 테라조 반점이 줄눈처럼 보인다
CLAHE_GRID = 8          # 타일 그리드 (8x8)
UNSHARP_SIGMA = 3.0     # 언샤프 가우시안 sigma (px)
UNSHARP_AMOUNT = 1.0    # 언샤프 강도


def enhance_floor(bgr, clip=CLAHE_CLIP, grid=CLAHE_GRID,
                  sigma=UNSHARP_SIGMA, amount=UNSHARP_AMOUNT):
    """CLAHE(LAB 의 L 채널) + 언샤프 마스크. 기하 불변.

    색(a,b)은 건드리지 않고 밝기(L)만 평활화해서 색 틀어짐 없이 줄눈 대비만 올린다.
    """
    if bgr is None:
        return bgr
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=float(clip),
                           tileGridSize=(int(grid), int(grid))).apply(l_ch)
    out = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    blur = cv2.GaussianBlur(out, (0, 0), float(sigma))
    return cv2.addWeighted(out, 1.0 + float(amount), blur, -float(amount), 0.0)


def main():
    if len(sys.argv) != 3:
        sys.exit("사용법: python3 make_enhanced.py in.jpg out.jpg")
    src, dst = sys.argv[1], sys.argv[2]
    img = cv2.imread(src)
    if img is None:
        sys.exit("이미지 읽기 실패: %s" % src)
    # JPEG 전용 옵션을 png 에 넘기면 OpenCV 가 "unsupported key" 경고를 뱉으므로
    # 확장자를 보고 고른다. 클릭용이면 png(무손실)를 권장.
    params = ([cv2.IMWRITE_JPEG_QUALITY, 97]
              if os.path.splitext(dst)[1].lower() in (".jpg", ".jpeg") else [])
    if not cv2.imwrite(dst, enhance_floor(img), params):
        sys.exit("저장 실패: %s (경로/확장자 확인)" % dst)
    print("저장: %s  (%dx%d, CLAHE clip=%.1f grid=%d + 언샤프 sigma=%.1f amount=%.1f)"
          % (dst, img.shape[1], img.shape[0],
             CLAHE_CLIP, CLAHE_GRID, UNSHARP_SIGMA, UNSHARP_AMOUNT))


if __name__ == "__main__":
    main()
