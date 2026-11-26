"""
probe_conf_bands.py -- Nguong conf cua detector co lam TE LIET vong association
                       thu hai cua ByteTrack khong?

VAN DE. ByteTrack chia detection lam hai nhom (ultralytics 8.4.141,
byte_tracker.py dong 316-317):

    remain_inds = valid & (scores >= track_high_thresh)                    # vong 1
    inds_low    = valid & (scores >  track_low_thresh)
                        & (scores <  track_high_thresh)                    # vong 2

Cau hinh baseline dang dung: YOLO `conf=0.25`, tracker `track_high_thresh=0.25`,
`track_low_thresh=0.1`. YOLO da LOC BO moi detection duoi 0.25 TRUOC KHI dua vao
tracker, nen dai [0.1, 0.25) khong con gi -> `detections_second` luon rong ->
vong association thu hai KHONG BAO GIO chay.

Script nay DO truc tiep: chay detector o nguong rat thap roi dem so detection
roi vao tung dai, thay vi suy luan tu code.

Ghi chu: vong 2 con mot dieu kien nua (byte_tracker.py dong 427) - chi cac track
dang o trang thai `Tracked` va chua ghep duoc o vong 1 moi tham gia. Nen so
"detection diem thap dua vao" va so "cap ghep thanh cong" la hai con so khac
nhau, phai bao cao rieng.

CACH DUNG
    python src/probe_conf_bands.py --videos MVI_20011 MVI_40171 --n-frames 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

LOW = 0.10      # track_low_thresh
HIGH = 0.25     # track_high_thresh = conf cua baseline


def main() -> int:
    ap = argparse.ArgumentParser(description="Do phan bo diem tin cay theo dai")
    ap.add_argument("--videos", nargs="+", default=["MVI_20011", "MVI_40171"])
    ap.add_argument("--n-frames", type=int, default=60)
    ap.add_argument("--probe-conf", type=float, default=0.01,
                    help="Nguong THAP dung de do; phai thap hon track_low_thresh")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(config.MODELS_DIR / "yolov8n.pt"))
    classes = config.YOLO_VEHICLE_CLASSES
    img_root = config.find_images_root()

    rows = []
    for v in a.videos:
        vdir = img_root / v
        imgs = sorted(vdir.glob("img*.jpg"))[:a.n_frames]
        if not imgs:
            print(f"  [BO QUA] khong co anh cho {v}")
            continue
        for p in imgs:
            r = model.predict(str(p), conf=a.probe_conf, classes=classes,
                              device=a.device, verbose=False)[0]
            for s in r.boxes.conf.cpu().numpy():
                rows.append(dict(video=v, frame=p.stem, conf=float(s)))
        print(f"  {v}: {len(imgs)} frame")

    d = pd.DataFrame(rows)
    if not len(d):
        print("Khong do duoc detection nao.")
        return 1

    n = len(d)
    n_high = int((d.conf >= HIGH).sum())
    n_band = int(((d.conf > LOW) & (d.conf < HIGH)).sum())
    n_below = int((d.conf <= LOW).sum())

    print("\n" + "=" * 76)
    print(f"PHAN BO DIEM TIN CAY  ({n:,} detection, nguong do = {a.probe_conf})")
    print("=" * 76)
    print(f"  {'dai diem':<34}{'so detection':>16}{'ti le':>10}")
    print("  " + "-" * 60)
    print(f"  {'>= 0.25  (vao vong 1)':<34}{n_high:>16,}{n_high / n * 100:>9.1f}%")
    print(f"  {'(0.10, 0.25)  (vao vong 2)':<34}{n_band:>16,}{n_band / n * 100:>9.1f}%")
    print(f"  {'<= 0.10  (bi bo hoan toan)':<34}{n_below:>16,}{n_below / n * 100:>9.1f}%")

    print("\n" + "=" * 76)
    print("HE QUA VOI HAI CAU HINH")
    print("=" * 76)
    print(f"  Lượt A -- YOLO conf = 0.25 (baseline hien tai):")
    print(f"     detection vao vong 1 : {n_high:,}")
    print(f"     detection vao vong 2 : 0   <-- YOLO da loc bo dai (0.10, 0.25)")
    print(f"     => vong association thu hai KHONG BAO GIO CHAY.")
    print(f"        Do la co che dac trung cua ByteTrack, dang bi vo hieu hoa.")
    print(f"\n  Lượt B -- YOLO conf = 0.10:")
    print(f"     detection vao vong 1 : {n_high:,}")
    print(f"     detection vao vong 2 : {n_band:,}  "
          f"(= {n_band / max(n_high, 1) * 100:.1f}% so voi vong 1)")
    print(f"     => vong hai co du lieu de chay.")
    print("\n  LUU Y: day moi la so DETECTION DUA VAO vong 2. So cap GHEP THANH CONG")
    print("  se thap hon, vi vong 2 chi nhan cac track dang `Tracked` va chua ghep")
    print("  duoc o vong 1 (byte_tracker.py dong 427). Chua do duoc con so do o day,")
    print("  va KHONG duoc suy ra conf=0.10 tot hon khi chua chay do day du.")

    out = config.RESULTS_DIR / "eval_fixed"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "conf_bands_probe.csv", index=False)
    pd.DataFrame([dict(n_total=n, n_high=n_high, n_band=n_band, n_below=n_below,
                       probe_conf=a.probe_conf, low=LOW, high=HIGH,
                       videos=",".join(a.videos), n_frames=a.n_frames)]
                 ).to_csv(out / "conf_bands_summary.csv", index=False)
    print(f"\n  Da luu -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
