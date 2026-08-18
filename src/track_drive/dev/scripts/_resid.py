#!/usr/bin/env python3
"""대응점 잔차 분석 + 이상점 제거 후 재적합 비교"""
import numpy as np, cv2, re, sys

CSV = "/home/xytron/h_pts.csv"
rows, labels = [], []
for ln in open(CSV, encoding="utf-8"):
    ln = ln.strip()
    if not ln or ln.startswith("#"):
        continue
    body = ln.split("#")[0].strip()
    lab = ln.split("#")[1].strip() if "#" in ln else ""
    p = [float(t) for t in body.split(",")]
    rows.append(p); labels.append(lab)
P = np.array(rows)
pix, gnd = P[:, :2].astype(np.float64), P[:, 2:4].astype(np.float64)
print(f"대응점 {len(P)}개\n")


def fit_and_report(idx, tag):
    H, _ = cv2.findHomography(pix[idx], gnd[idx], 0)
    if H is None:
        print(f"[{tag}] 적합 실패"); return None, None
    # 전체 점에 대한 잔차
    ph = np.hstack([pix, np.ones((len(pix), 1))])
    q = (H @ ph.T).T
    q = q[:, :2] / q[:, 2:3]
    e = np.linalg.norm(q - gnd, axis=1)
    sub = e[idx]
    print(f"[{tag}] 사용 {len(idx)}점 → RMS {np.sqrt((sub**2).mean())*1000:6.1f} mm, "
          f"최대 {sub.max()*1000:6.1f} mm")
    return H, e


idx_all = np.arange(len(P))
H0, e0 = fit_and_report(idx_all, "전체")
print("\n점별 잔차 (mm):")
order = np.argsort(-e0)
for k in order:
    flag = "  ← 이상점 의심" if e0[k] > 0.04 else ""
    print(f"  {labels[k]:14s} 픽셀({pix[k,0]:6.1f},{pix[k,1]:6.1f}) "
          f"지면({gnd[k,0]:+.3f},{gnd[k,1]:+.3f})  잔차 {e0[k]*1000:7.1f}{flag}")

# 이상점 반복 제거 (잔차 최대인 점을 하나씩 빼며 개선 확인)
keep = list(idx_all)
print("\n=== 이상점 제거 ===")
for it in range(4):
    H, e = fit_and_report(np.array(keep), f"{it}차")
    sub = e[np.array(keep)]
    if np.sqrt((sub**2).mean()) < 0.015 or len(keep) <= 8:
        break
    worst = keep[int(np.argmax(sub))]
    print(f"   제거: {labels[worst]} (잔차 {e[worst]*1000:.1f} mm)")
    keep.remove(worst)

Hf, ef = fit_and_report(np.array(keep), "최종")
np.save("/home/xytron/H_clean.npy", Hf)
print("\n남긴 점:", [labels[k] for k in keep])
print("저장: /home/xytron/H_clean.npy")

# 소실선(지평선) 위치 — 원거리 정확도의 핵심 지표
# 지면 x→무한 이면 픽셀은 소실선 위. H^-1 로 지면 무한점을 픽셀로
Hi = np.linalg.inv(Hf)
vp = Hi @ np.array([1.0, 0.0, 0.0])      # x방향 무한점
print(f"\n소실점(전방무한): u={vp[0]/vp[2]:.1f}, v={vp[1]/vp[2]:.1f}")
for d in (0.5, 1.0, 1.5, 2.0, 3.0):
    p = Hi @ np.array([d, 0.0, 1.0]); p = p[:2] / p[2]
    print(f"  x={d:.1f}m, y=0  →  픽셀 v={p[1]:.1f}")
