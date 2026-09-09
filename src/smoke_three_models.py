"""
smoke_three_models.py -- Smoke test: chay KF/EKF/UKF tren mot nhom nho video.

Muc dich KHONG phai do hieu nang (mau qua nho), ma la:
  1. Xac minh RUNTIME dung dung lop bo loc (khong am tham roi ve KalmanFilterXYAH)
  2. Xac minh code da sua chay het video khong loi, covariance khong sinh NaN
  3. Do hoat dong cua vong association thu hai o hai muc conf khac nhau

Ket qua ghi vao thu muc RIENG, khong ghi de ket qua thi nghiem cu.

CACH DUNG
    python src/smoke_three_models.py --videos MVI_20011 MVI_40171 --conf 0.25
    python src/smoke_three_models.py --videos MVI_20011 MVI_40171 --conf 0.10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assoc_stats  # noqa: E402
import config  # noqa: E402
import tracker_ctrv  # noqa: E402

MODELS = [("cv", "KalmanFilterXYAH"), ("ekf_ctrv", "EKFTrackerCTRV"),
          ("ukf_ctrv", "UKFTrackerCTRV")]


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke test 3 motion model")
    ap.add_argument("--videos", nargs="+", default=["MVI_20011", "MVI_40171"])
    ap.add_argument("--n-frames", type=int, default=120)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out-dir", default="results/eval_fixed/smoke")
    a = ap.parse_args()

    from ultralytics import YOLO
    tracker_ctrv.register()
    img_root = config.find_images_root()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(f"SMOKE TEST 3 MOTION MODEL -- conf = {a.conf}, "
          f"{len(a.videos)} video x {a.n_frames} frame")
    print("=" * 78)

    rows = []
    for mm, want_filter in MODELS:
        spec = config.MOTION_MODELS[mm]
        assoc_stats.disable()
        assoc_stats.reset()
        assoc_stats.enable()

        model = YOLO(str(config.MODELS_DIR / "yolov8n.pt"))
        seen_filter = set()
        n_det = n_trk = 0
        bad = 0
        for v in a.videos:
            imgs = sorted((img_root / v).glob("img*.jpg"))[:a.n_frames]
            for i, p in enumerate(imgs):
                r = model.track(str(p), persist=(i > 0), tracker=spec["cfg"],
                                conf=a.conf, classes=config.YOLO_VEHICLE_CLASSES,
                                device=a.device, verbose=False)[0]
                n_det += 0 if r.boxes is None else len(r.boxes)
                if r.boxes is not None and r.boxes.id is not None:
                    n_trk += len(r.boxes.id)
            # Doc lop bo loc THUC SU dang duoc dung trong runtime
            tr = model.predictor.trackers[0]
            seen_filter.add(type(tr.kalman_filter).__name__)
            for st in list(tr.tracked_stracks) + list(tr.lost_stracks):
                if st.mean is not None and not np.all(np.isfinite(st.mean)):
                    bad += 1
                if st.covariance is not None:
                    P = np.asarray(st.covariance)
                    if not np.all(np.isfinite(P)):
                        bad += 1
                    elif P.shape[0] == P.shape[1]:
                        if float(np.linalg.eigvalsh(0.5 * (P + P.T)).min()) < -1e-6:
                            bad += 1
            model.predictor.trackers = []      # tracker moi cho video sau

        got = ",".join(sorted(seen_filter))
        ok = want_filter in seen_filter
        print(f"\n[{mm}]  lop bo loc runtime = {got}  "
              f"{'OK' if ok else 'SAI (mong ' + want_filter + ')'}")
        print(f"  detection = {n_det:,}, track output = {n_trk:,}, "
              f"track co mean/P hong = {bad}")
        print("  Vong association thu hai:")
        print(assoc_stats.summary())
        c = assoc_stats.counts()
        rows.append(dict(motion_model=mm, conf=a.conf, filter_runtime=got,
                         filter_ok=ok, n_det=n_det, n_track_out=n_trk,
                         n_bad_state=bad, **c))
        assert ok, f"{mm}: runtime dung {got}, mong {want_filter}"
        assert bad == 0, f"{mm}: {bad} track co mean/covariance hong"

    assoc_stats.disable()
    df = pd.DataFrame(rows)
    f = out / f"smoke_conf{a.conf}.csv"
    df.to_csv(f, index=False)
    print(f"\n  Da luu -> {f}")
    print("\n  LUU Y: mau nay QUA NHO de so sanh hieu nang giua 3 model.")
    print("  Muc dich chi la xac minh runtime dung lop bo loc va code chay sach.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
