"""
benchmark_fps.py -- Stage B: do lai FPS cua 3 motion model mot cach dang tin.

VI SAO CAN FILE NAY?
So FPS bao cao o Giai doan 2-4 (KF 54.6, EKF 80.8, UKF 46.3) co mot diem VO LY:
EKF nhanh hon KF baseline, trong khi EKF phai lam THEM viec tinh Jacobian 9x9.
Con so do lay tu MOT lan chay duy nhat moi model, tren 5 video khac nhau, khong
kiem soat nhieu do tai GPU -> khong du tin cay de ket luan.

File nay do lai theo 2 tang:

  TANG 1 (micro): chi phi THUAN TUY cua motion model, khong dinh GPU/detector.
                  Do truc tiep predict / update / multi_predict.
  TANG 2 (end-to-end): chay ca pipeline nhieu lan tren CUNG video, bao cao
                  trung binh +- do lech chuan, dong thoi DEM so lan goi
                  predict/update va so track -> kiem chung gia thuyet
                  "model nao it track hon thi ton it phep tinh hon".

CACH DUNG
    python src/benchmark_fps.py --micro                       # chi tang 1
    python src/benchmark_fps.py --videos MVI_20011 --reps 5   # tang 1 + 2
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402
from ukf_ctrv import UKFTrackerCTRV  # noqa: E402


# ===========================================================================
# TANG 1 - micro benchmark
# ===========================================================================
def micro(reps: int = 5, n_calls: int = 3000, n_tracks: int = 20) -> pd.DataFrame:
    from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH

    def make(f, kind):
        m, c = f.initiate(np.array([500., 300., 0.5, 40.]))
        if kind != "kf":
            m, c = f.initiate_from_motion(m, c, np.array([508., 304., .5, 40.]), 1.0)
        return m, c

    z = np.array([505., 302., 0.5, 40.])
    rows = []
    for name, f, kind in [("KF + CV", KalmanFilterXYAH(), "kf"),
                          ("EKF + CTRV", EKFTrackerCTRV(), "ekf"),
                          ("UKF + CTRV", UKFTrackerCTRV(), "ukf")]:
        for op in ["predict", "update", "multi_predict"]:
            times = []
            for _ in range(reps):
                if op == "multi_predict":
                    ms, cs = zip(*[make(f, kind) for _ in range(n_tracks)])
                    M, C = np.array(ms), np.array(cs)
                    k = max(1, n_calls // 10)
                    t0 = time.perf_counter()
                    for _ in range(k):
                        M, C = f.multi_predict(M, C)
                    times.append((time.perf_counter() - t0) / k * 1e6)
                else:
                    m, c = make(f, kind)
                    t0 = time.perf_counter()
                    for _ in range(n_calls):
                        if op == "predict":
                            m, c = f.predict(m, c)
                        else:
                            f.update(m, c, z)
                    times.append((time.perf_counter() - t0) / n_calls * 1e6)
            rows.append(dict(model=name, op=op, us_mean=round(float(np.mean(times)), 2),
                             us_std=round(float(np.std(times)), 2)))
    return pd.DataFrame(rows)


# ===========================================================================
# TANG 2 - end-to-end, co dem so lan goi
# ===========================================================================
class Counter:
    """Boc method cua bo loc de dem so lan goi va cong don thoi gian."""

    def __init__(self):
        self.n = {}
        self.t = {}
        self._orig = []

    def wrap(self, cls, meth: str, tag: str):
        orig = getattr(cls, meth)
        self.n.setdefault(tag, 0)
        self.t.setdefault(tag, 0.0)

        def wrapper(*a, **k):
            t0 = time.perf_counter()
            r = orig(*a, **k)
            self.t[tag] += time.perf_counter() - t0
            self.n[tag] += 1
            return r

        setattr(cls, meth, wrapper)
        self._orig.append((cls, meth, orig))

    def restore(self):
        for cls, meth, orig in self._orig:
            setattr(cls, meth, orig)
        self._orig = []


def run_once(video: str, motion_model: str, split_name: str, counter: Counter):
    """Chay tracking 1 video, tra ve (so_anh, giay, so_track)."""
    import cv2
    from ultralytics import YOLO
    import baseline_track as bt

    spec = config.MOTION_MODELS[motion_model]
    tracker_cfg = spec["cfg"]
    if motion_model != "cv":
        import tracker_ctrv
        tracker_ctrv.register()

    img_root = config.find_images_root()
    imgs = sorted((img_root / video).glob("img*.jpg"))
    regions = bt.load_ignored_regions(video, split_name)
    model = YOLO(config.YOLO_DEFAULT_MODEL)

    ids = set()
    t0 = time.perf_counter()
    for p in imgs:
        frame = cv2.imread(str(p))
        if regions:
            frame = bt.mask_ignored_regions(frame, regions)
        res = model.track(frame, persist=True, tracker=tracker_cfg,
                          conf=config.YOLO_DEFAULT_CONF,
                          classes=config.YOLO_VEHICLE_CLASSES,
                          device="0", verbose=False)
        b = res[0].boxes
        if b is not None and b.id is not None:
            ids.update(b.id.int().cpu().tolist())
    dt = time.perf_counter() - t0
    return len(imgs), dt, len(ids)


def end_to_end(videos: list[str], reps: int, split_name: str) -> pd.DataFrame:
    from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH

    rows = []
    for mm in ["cv", "ekf_ctrv", "ukf_ctrv"]:
        label = {"cv": "KF + CV", "ekf_ctrv": "EKF + CTRV", "ukf_ctrv": "UKF + CTRV"}[mm]
        for video in videos:
            for rep in range(reps):
                c = Counter()
                # Boc dung lop bo loc ma motion model nay su dung
                if mm == "cv":
                    c.wrap(KalmanFilterXYAH, "multi_predict", "multi_predict")
                    c.wrap(KalmanFilterXYAH, "update", "update")
                elif mm == "ekf_ctrv":
                    c.wrap(EKFTrackerCTRV, "multi_predict", "multi_predict")
                    c.wrap(EKFTrackerCTRV, "update", "update")
                else:
                    # UKF ke thua multi_predict tu EKF nhung override update
                    c.wrap(EKFTrackerCTRV, "multi_predict", "multi_predict")
                    c.wrap(UKFTrackerCTRV, "update", "update")
                try:
                    n_img, dt, n_tracks = run_once(video, mm, split_name, c)
                finally:
                    c.restore()
                rows.append(dict(
                    model=label, video=video, rep=rep, n_images=n_img,
                    seconds=round(dt, 2), fps=round(n_img / dt, 2), n_tracks=n_tracks,
                    n_multi_predict=c.n.get("multi_predict", 0),
                    n_update=c.n.get("update", 0),
                    t_motion_s=round(c.t.get("multi_predict", 0.0) + c.t.get("update", 0.0), 3),
                ))
                print(f"  {label:<12} {video} rep{rep}: {rows[-1]['fps']:6.2f} fps, "
                      f"{n_tracks:3d} track, motion {rows[-1]['t_motion_s']:6.2f}s")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage B: do lai FPS")
    ap.add_argument("--micro", action="store_true", help="Chi chay micro benchmark")
    ap.add_argument("--videos", nargs="*", default=["MVI_20011", "MVI_39811"])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--split-name", default="DETRAC-all")
    a = ap.parse_args()

    out_dir = config.RESULTS_DIR / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("TANG 1 - MICRO BENCHMARK (motion model thuan tuy, khong GPU)")
    print("=" * 78)
    m = micro()
    piv = m.pivot(index="model", columns="op", values="us_mean")
    err = m.pivot(index="model", columns="op", values="us_std")
    order = ["KF + CV", "EKF + CTRV", "UKF + CTRV"]
    print("\n  micro giay / lan goi (trung binh +- do lech chuan):")
    for mdl in order:
        s = "  ".join(f"{o}={piv.loc[mdl, o]:8.2f}+-{err.loc[mdl, o]:6.2f}"
                      for o in ["predict", "update", "multi_predict"])
        print(f"    {mdl:<12} {s}")
    print("\n  ti le so voi KF + CV:")
    for mdl in order:
        s = "  ".join(f"{o}={piv.loc[mdl, o]/piv.loc['KF + CV', o]:6.2f}x"
                      for o in ["predict", "update", "multi_predict"])
        print(f"    {mdl:<12} {s}")
    m.to_csv(out_dir / "micro_benchmark.csv", index=False)
    print(f"\n  -> {out_dir / 'micro_benchmark.csv'}")

    if a.micro:
        return 0

    print("\n" + "=" * 78)
    print(f"TANG 2 - END-TO-END ({len(a.videos)} video x {a.reps} lan lap x 3 model)")
    print("=" * 78)
    df = end_to_end(a.videos, a.reps, a.split_name)
    df.to_csv(out_dir / "end_to_end_benchmark.csv", index=False)

    print("\n  FPS trung binh +- do lech chuan:")
    g = df.groupby(["model", "video"])["fps"].agg(["mean", "std", "count"])
    print(g.round(2).to_string())
    print("\n  So lan goi + thoi gian motion model:")
    g2 = df.groupby("model").agg(
        n_tracks=("n_tracks", "mean"), n_multi_predict=("n_multi_predict", "mean"),
        n_update=("n_update", "mean"), t_motion_s=("t_motion_s", "mean"),
        seconds=("seconds", "mean"))
    print(g2.round(2).to_string())
    print(f"\n  -> {out_dir / 'end_to_end_benchmark.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
