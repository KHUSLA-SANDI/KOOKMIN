#!/usr/bin/env python3
"""왜곡보정이 실제로 적용되는지 검증 — no-op 이면 지금까지 측정이 전부 무의미"""
import sys
import numpy as np, cv2

PKG = "/home/xytron/xycar_ws/src/track_drive"
sys.path.insert(0, PKG)
from track_drive.lib import bev as B

cam = B.load_camera_config(PKG + "/config/camera.yaml")
print("=== load_camera_config 반환 ===")
for k in ("model", "intrinsics_calibrated", "h_calibrated", "calibrated"):
    print(f"  {k} = {cam.get(k)}")
K = np.array(cam["K"], np.float64)
D = np.ravel(cam["D"])
print(f"  K[0,0]={K[0,0]:.3f}  K[0,2]={K[0,2]:.3f}  K[1,1]={K[1,1]:.3f}  K[1,2]={K[1,2]:.3f}")
print(f"  D = {np.round(D,5)}")
nk = cam.get("new_K")
print(f"  new_K = {None if nk is None else np.round(np.array(nk),2).tolist()}")
print(f"  note = {cam.get('note','')}")

raw = cv2.imread(PKG + "/dev/shots/tape2_raw.jpg")
u = B.Undistorter.from_config(cam, (raw.shape[1], raw.shape[0]))
print(f"\n=== Undistorter ===")
print(f"  enabled = {u.enabled}   model = {getattr(u,'model',None)}")
out = u.apply(raw)
same_obj = out is raw
diff = int((cv2.absdiff(out, raw) > 3).sum())
print(f"  apply() 가 원본 객체 그대로 반환? {same_obj}")
print(f"  원본과 다른 화소 수: {diff} / {raw.size}  ({100*diff/raw.size:.1f}%)")
if same_obj or diff < raw.size * 0.02:
    print("\n  ❌❌ 왜곡보정이 사실상 적용 안 되고 있다! 지금까지의 '직선성' 측정은 RAW 영상 측정이었음")
else:
    print("\n  ✅ 왜곡보정 적용됨")

# 직접 보정본과 비교 (yaml K/D 를 직접 써서)
nk2, _ = cv2.getOptimalNewCameraMatrix(K, D[:5], (raw.shape[1], raw.shape[0]), 0.0)
mx, my = cv2.initUndistortRectifyMap(K, D[:5], np.eye(3), nk2,
                                     (raw.shape[1], raw.shape[0]), cv2.CV_32FC1)
ref = cv2.remap(raw, mx, my, cv2.INTER_LINEAR)
d2 = int((cv2.absdiff(ref, out) > 3).sum())
print(f"  lib 결과 vs 직접계산 결과 차이 화소: {d2} ({100*d2/raw.size:.1f}%)")
cv2.imwrite("/home/xytron/chk_lib.jpg", out)
cv2.imwrite("/home/xytron/chk_ref.jpg", ref)
cv2.imwrite("/home/xytron/chk_raw.jpg", raw)
print("saved: chk_raw.jpg / chk_lib.jpg / chk_ref.jpg")
