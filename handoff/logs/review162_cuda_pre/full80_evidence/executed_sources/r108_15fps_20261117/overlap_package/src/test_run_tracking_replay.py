"""
test_run_tracking_replay.py -- CAPTURE-AND-REPLAY: runner doc cache phai cho output
BIT-IDENTICAL voi duong model.track() cua baseline_track.py khi nhan cung detection.

Cach lam
    1. Chay dung vong lap cua baseline_track.track_one_video (cv2.imread -> mask ->
       model.track(frame, persist=True, tracker=yaml, classes, conf)) tren N frame
       dau cua mot video UA-DETRAC, dong thoi BOC `BYTETracker.update` o muc LOP de
       ghi lai mang detection (`results.data`) ma tracker nhan o TUNG frame.
    2. Dong goi cac mang do thanh mot cache npz (dung save_video) va chay
       run_tracking.track_video tren cache voi cung file .yaml.
    3. So sanh tung dong (frame, id, x, y, w, h, conf): phai bang nhau TUYET DOI.
    Them: doi chieu voi ket qua da luu cua Giai doan 1 (testA-cv) tren cung N frame
    bang ti le box khop IoU >= 0.9 - chi de THONG TIN (GPU khong bit-reproducible).

Can anh UA-DETRAC + models/yolov8n.pt. ~30-60 s tren GPU.

    python src/test_run_tracking_replay.py [--video MVI_39031] [--n-frames 150] [--device 0]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import run_tracking  # noqa: E402
import tracker_ctrv  # noqa: E402
from baseline_track import load_ignored_regions, mask_ignored_regions  # noqa: E402

from ultralytics.trackers.byte_tracker import BYTETracker  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def capture_run(video, split, tracker_yaml, conf, n_frames, device):
    """Chay model.track y het baseline_track.py, ghi lai detection tung frame + output."""
    from ultralytics import YOLO
    img_root = config.find_images_root()
    paths = sorted((img_root / video).glob("img*.jpg"))[:n_frames]
    regions = load_ignored_regions(video, split)
    model = YOLO(config.YOLO_DEFAULT_MODEL)

    captured = []          # (frame_num, det (n,6) float32, (H, W))
    orig_update = BYTETracker.update

    def wrapped(self, results, img=None, feats=None, **kw):
        captured.append((results.data.astype(np.float32).copy(), img.shape[:2]))
        return orig_update(self, results, img, feats, **kw)
    BYTETracker.update = wrapped
    rows = []
    try:
        for frame_num, fp in enumerate(paths, start=1):
            frame = cv2.imread(str(fp))
            frame = mask_ignored_regions(frame, regions)
            r = model.track(frame, persist=True, tracker=tracker_yaml, classes=config.YOLO_VEHICLE_CLASSES,
                            conf=conf, device=device, verbose=False)[0]
            if r.boxes.id is None:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy(); ids = r.boxes.id.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), tid, cf in zip(xyxy, ids, confs):
                rows.append((frame_num, int(tid), float(x1), float(y1), float(x2 - x1), float(y2 - y1), float(cf)))
    finally:
        BYTETracker.update = orig_update
    assert len(captured) == len(paths), f"capture {len(captured)} != {len(paths)} frame"
    return captured, pd.DataFrame(rows, columns=run_tracking.TRK_COLS)


def replay(captured, tracker_yaml, conf):
    """Dong goi detection da ghi thanh cache tam roi chay runner."""
    F = len(captured)
    arrays = dict(frame_idx=np.arange(1, F + 1, dtype=np.int32),
                  n_per_frame=np.array([len(d) for d, _ in captured], np.int32),
                  unreadable=np.zeros(F, bool), truncated=np.zeros(F, bool),
                  time_s=np.arange(F) / config.FPS,
                  data=np.concatenate([d for d, _ in captured]) if F else np.zeros((0, 6), np.float32),
                  orig_shape=np.array(captured[0][1], np.int32))
    with tempfile.TemporaryDirectory() as td:
        det_cache.save_video(Path(td), "v", arrays)
        vd = det_cache.VideoDetections(Path(td) / "v.npz", conf_min=conf, classes=config.YOLO_VEHICLE_CLASSES,
                                       max_det=300)
        cfg = run_tracking.load_tracker_cfg(tracker_yaml)
        df, _, _ = run_tracking.track_video(vd, cfg, 1.0 / config.FPS, elapsed_mode="skip")
    return df


def iou_agreement(a: pd.DataFrame, b: pd.DataFrame, thr=0.9) -> float:
    """Ti le box cua a co box cung frame trong b voi IoU >= thr (khong xet ID)."""
    if len(a) == 0:
        return float("nan")
    hit = 0
    bb = {f: g[["bb_left", "bb_top", "bb_width", "bb_height"]].to_numpy() for f, g in b.groupby("frame")}
    for f, g in a.groupby("frame"):
        B = bb.get(f)
        if B is None:
            continue
        A = g[["bb_left", "bb_top", "bb_width", "bb_height"]].to_numpy()
        ax2, ay2 = A[:, 0] + A[:, 2], A[:, 1] + A[:, 3]
        bx2, by2 = B[:, 0] + B[:, 2], B[:, 1] + B[:, 3]
        iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(A[:, 0][:, None], B[:, 0][None]), 0, None)
        ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(A[:, 1][:, None], B[:, 1][None]), 0, None)
        inter = iw * ih
        union = (A[:, 2] * A[:, 3])[:, None] + (B[:, 2] * B[:, 3])[None] - inter
        hit += int((inter / np.maximum(union, 1e-9)).max(1).__ge__(thr).sum())
    return hit / len(a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="MVI_39031")
    ap.add_argument("--split", default="DETRAC-test")
    ap.add_argument("--n-frames", type=int, default=150)
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()
    tracker_ctrv.register()

    yamls = {"cv": "bytetrack.yaml", "ekf": str(ROOT / "configs/bytetrack_ekf_ctrv.yaml"),
             "ukf": str(ROOT / "configs/bytetrack_ukf_ctrv.yaml")}
    failed = 0
    for name, y in yamls.items():
        print(f"\n--- {name}: capture ({a.n_frames} frame {a.video}) -> replay ---")
        try:
            captured, base = capture_run(a.video, a.split, y, a.conf, a.n_frames, a.device)
            rep = replay(captured, y, a.conf)
            same = len(base) == len(rep) and base.equals(rep)
            detail = f"baseline {len(base)} dong, replay {len(rep)} dong"
            if not same and len(base) == len(rep):
                d = (base.drop(columns=["frame", "id"]) - rep.drop(columns=["frame", "id"])).abs().max().max()
                detail += f", lech toa do lon nhat {d:.3e}, id giong: {(base['id'] == rep['id']).all()}"
            _check(f"[{name}] runner replay == model.track (bit-identical)", same, detail)
            saved = config.PROCESSED_DIR / "trackers" / a.split / f"testA-{'cv' if name == 'cv' else name + '_ctrv'}" / "data" / f"{a.video}.txt"
            if saved.exists() and a.conf == 0.25:
                s = pd.read_csv(saved, header=None, names=run_tracking.TRK_COLS)
                s = s[s.frame <= a.n_frames]
                print(f"         [thong tin] vs testA-{name} (GD1, GPU khac lan chay): {len(s)} dong da luu, "
                      f"{len(base)} dong hom nay; ti le box khop IoU>=0.9: {iou_agreement(base, s):.4f}")
        except AssertionError as e:
            failed += 1
            print(f"  !! {e}")

    # --- Test 2: DETECTOR -> CACHE. Chay truc tiep (model.track) vs cache xay bang model.predict
    # (conf 0.05, 80 lop, max_det 3000) roi loc >conf / [2,3,5,7] / 300 va chay runner, CUNG tien trinh.
    # Khac test 1 (capture-and-replay: cung mang detection -> kiem tra duong TRACKER), test nay kiem
    # tra duong detector -> NMS -> cache -> loc. GPU co the khong bit-reproducible giua hai lan forward
    # nen khang dinh o muc cau truc (cap (frame, id) trung >= 99 %), va bao cao ti le bit-identical.
    print(f"\n--- test 2: model.track truc tiep vs cache model.predict (conf 0.05) -> runner, {a.n_frames} frame ---")
    try:
        from mot_datasets import get_dataset
        ds = get_dataset("detrac")
        part = "test" if a.split == "DETRAC-test" else "train"
        frames = ds.frames(a.video, part)[: a.n_frames]
        regions = load_ignored_regions(a.video, a.split)
        from ultralytics import YOLO
        model = YOLO(config.YOLO_DEFAULT_MODEL)
        spec = det_cache.DetectorSpec(conf=0.05, classes=None, max_det=3000, device=a.device,
                                      dataset="detrac", part=part).fill_env()
        arrays = det_cache.build_video_yolo(model, frames, regions, spec, ds.orig_shape)
        with tempfile.TemporaryDirectory() as td:
            det_cache.save_video(Path(td), "v", arrays)
            vd = det_cache.VideoDetections(Path(td) / "v.npz", conf_min=a.conf, classes=config.YOLO_VEHICLE_CLASSES, max_det=300)
            _check("cache khong frame nao cham tran max_det", not vd.truncated.any())
            for name, y in yamls.items():
                _, direct = capture_run(a.video, a.split, y, a.conf, a.n_frames, a.device)   # model.track truc tiep
                cfg = run_tracking.load_tracker_cfg(y)
                via_cache, _, _ = run_tracking.track_video(vd, cfg, 1.0 / config.FPS, elapsed_mode="skip")
                key = ["frame", "id"]
                m = direct.merge(via_cache, on=key, suffixes=("_d", "_c"))
                agree = len(m) / max(len(direct), 1)
                exact = len(direct) == len(via_cache) and direct.equals(via_cache)
                dmax = float((m[["bb_left_d", "bb_top_d", "bb_width_d", "bb_height_d"]].to_numpy()
                              - m[["bb_left_c", "bb_top_c", "bb_width_c", "bb_height_c"]].to_numpy()).__abs__().max()) if len(m) else 0.0
                _check(f"[{name}] direct vs cache: cap (frame,id) trung >= 99% "
                       f"({agree:.4f}; direct {len(direct)} / cache {len(via_cache)} dong; bit-identical={exact}; lech toa do max {dmax:.3g} px)",
                       agree >= 0.99 and abs(len(direct) - len(via_cache)) <= 0.01 * len(direct))
                # MOC 2: lien ket ID xuyen suot sequence - anh xa ID 1-1 CO DINH toan sequence (khong chi
                # tung frame): moi track cua ban direct ung voi dung mot track cua ban cache va nguoc lai.
                # tol theo px vi GPU khong bit-reproducible giua hai lan forward (lech <= ~1e-4 px).
                from track_compare import id_bijection
                r = id_bijection(direct, via_cache, tol=max(1e-3, 10 * dmax))
                _check(f"[{name}] direct vs cache: SONG ANH ID toan sequence ({r.summary()})",
                       r.is_bijection and r.is_identity)
    except AssertionError as e:
        failed += 1
        print(f"  !! {e}")
    n_ok = sum(ok for _, ok in _RESULTS)
    print("\n" + "=" * 78)
    print(f"KET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
