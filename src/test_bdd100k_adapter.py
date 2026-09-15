"""
test_bdd100k_adapter.py -- Kiem tra adapter BDD100K-MOT tren FIXTURE (khong anh) va di
HET duong chay: JSON -> interim -> MOTChallenge theo lop -> detection tong hop tu GT ->
runner (CV, EKF) -> tach lop -> TrackEval (crowd-ignore) -> occlusion_segments/eval (gap).

Split fixture dung tien to BDDFX- va tag bddfx_val de KHONG lan voi du lieu that (BDD-val).

    python src/test_bdd100k_adapter.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import run_tracking  # noqa: E402
import run_trackeval  # noqa: E402
from mot_datasets.bdd100k_mot import (BDD100KMOTDataset, COCO_TO_BDD, MOT_IGNORE_CLASS, SchemaError,  # noqa: E402
                                      check_schema, export_motchallenge, parse_video,
                                      split_tracker_output_by_class, stats_report, track_gaps)

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "data" / "fixtures" / "bdd100k_mini"
BENCH, PART, TAG = "BDDFX", "val", "bddfx_val"
PY = sys.executable
_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def test_1_schema():
    ds = BDD100KMOTDataset(FIX)
    vids = ds.list_videos(PART)
    _check("fixture co 3 video", len(vids) == 3, str(vids))
    good = ds.load_json(vids[0], PART)
    st = check_schema(good, vids[0])
    _check("schema hop le: 12 frame, co categories", st["n_frames"] == 12 and "car" in st["categories"])
    old = json.loads(json.dumps(good))
    for f in old:
        f["index"] = f.pop("frameIndex")
    try:
        check_schema(old, "old"); _check("schema CU (index) bi tu choi", False)
    except SchemaError as e:
        _check("schema CU (index) bi tu choi", "schema CU" in str(e), str(e))
    old2 = json.loads(json.dumps(good))
    old2[0]["labels"][0]["attributes"] = {"Crowd": False, "Occluded": False, "Truncated": False}
    try:
        check_schema(old2, "old2"); _check("attributes schema CU (Crowd) bi tu choi", False)
    except SchemaError:
        _check("attributes schema CU (Crowd) bi tu choi", True)
    bad = json.loads(json.dumps(good)); bad[3]["frameIndex"] = 9
    try:
        check_schema(bad, "bad"); _check("frameIndex khong lien tuc bi tu choi", False)
    except SchemaError:
        _check("frameIndex khong lien tuc bi tu choi", True)
    bad = json.loads(json.dumps(good)); bad[0]["labels"][0]["category"] = "tuk-tuk"
    try:
        check_schema(bad, "bad"); _check("category la bi tu choi", False)
    except SchemaError:
        _check("category la bi tu choi", True)


def test_2_parse_and_stats():
    ds = BDD100KMOTDataset(FIX)
    df = ds.build_interim(PART)
    _check("interim parquet mang tag thu muc fixture",
           ds.interim_parquet(PART).name == "bdd100k_mini_val_annotations.parquet" and ds.interim_parquet(PART).exists())
    b1 = df[df.video.str.startswith("b1")]
    _check("frame 1-based lien tuc", sorted(b1.frame.unique().tolist()) == list(range(1, 13)))
    truck = b1[b1.category == "truck"]
    _check("occluded -> occlusion_ratio 0.5 (frame 6-8), khac 0",
           set(truck[truck.occluded].frame) == {6, 7, 8} and (truck[truck.occluded].occlusion_ratio == 0.5).all()
           and (truck[~truck.occluded].occlusion_ratio == 0).all())
    _check("crowd, rider, pedestrian, other vehicle deu duoc giu trong interim",
           b1.crowd.sum() == 12 and (b1.category == "rider").sum() == 12
           and (df.category == "other vehicle").sum() == 12)
    gaps = track_gaps(df).sort_values("track_id").reset_index(drop=True)
    _check("track_gaps: xe G (id 11) gap frame 5-7, xe M (id 15) gap frame 7-8",
           len(gaps) == 2 and (gaps.track_id.tolist() == [11, 15]) and gaps.iloc[0].gap_start == 5
           and gaps.iloc[0].gap_end == 7 and gaps.iloc[1].gap_start == 7 and gaps.iloc[1].gap_end == 8, gaps.to_string())
    st = stats_report(df)
    _check("stats_report co 8 lop, rider evaluated=True, other vehicle evaluated=False",
           len(st) == 8 and bool(st[st.category == "rider"].evaluated.iloc[0])
           and not bool(st[st.category == "other vehicle"].evaluated.iloc[0]))
    fr3 = ds.frames("b3e7f1c0-9a2b4d6e", PART)
    t = np.array([f.time_s for f in fr3])
    _check("time_s tu timestamp (b3): co khoang 0.4 s (bo 1 frame)", np.isclose(np.diff(t).max(), 0.4)
           and np.isclose(t[1], 0.2), str(np.round(t, 2)))
    fr1 = ds.frames("b1c66a42-6f7d68ca", PART)
    _check("time_s danh nghia (b1): 0.2 s/buoc", np.isclose(fr1[5].time_s, 1.0))
    _check("path anh None khi khong co thu muc images", fr1[0].path is None)
    return df


def test_3_export_mot(df):
    out_root = config.PROCESSED_DIR
    for split in list(out_root.glob(f"{BENCH}-*")) + list((out_root / "trackers").glob(f"{BENCH}-*")):
        shutil.rmtree(split)
    r = export_motchallenge(df, PART, "vehicle", out_root=out_root, benchmark=BENCH)
    rc = export_motchallenge(df, PART, "car", out_root=out_root, benchmark=BENCH)
    rm = export_motchallenge(df, PART, "motorcycle", out_root=out_root, benchmark=BENCH)
    _check("split vehicle: BDDFX-val", r["split_name"] == f"{BENCH}-val" and r["n_videos"] == 3)
    gt = pd.read_csv(out_root / f"{BENCH}-val" / "b1c66a42-6f7d68ca" / "gt" / "gt.txt", header=None,
                     names=["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"])
    _check("vehicle: crowd -> class 13 conf 0; car/truck/motorcycle -> class 1; khong rider/pedestrian",
           set(gt.cls.unique()) == {1, MOT_IGNORE_CLASS} and (gt[gt.cls == 13].conf == 0).all()
           and gt[gt.cls == 1].id.nunique() == 4 and 5 not in gt.id.values and 4 not in gt.id.values,
           gt.groupby("cls").id.nunique().to_string())
    _check("vehicle: vis = 0.5 khi occluded (truck frame 6-8)",
           np.isclose(gt[(gt.id == 2) & (gt.frame == 6)].vis.iloc[0], 0.5))
    gt2 = pd.read_csv(out_root / f"{BENCH}-val" / "b2d0a5e2-3c1f9a7b" / "gt" / "gt.txt", header=None,
                      names=["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"])
    _check("vehicle: 'other vehicle' -> class 13", (gt2[gt2.id == 13].cls == 13).all())
    gtc = pd.read_csv(out_root / f"{BENCH}-val_car" / "b1c66a42-6f7d68ca" / "gt" / "gt.txt", header=None,
                      names=["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"])
    _check("split car: chi car class 1 (id 1, 8), crowd van class 13",
           set(gtc[gtc.cls == 1].id.unique()) == {1, 8} and (gtc.cls == 13).sum() == 12)
    seq = (out_root / f"{BENCH}-val" / "b3e7f1c0-9a2b4d6e" / "seqinfo.ini").read_text()
    _check("seqinfo: frameRate=5, seqLength=8, 1280x720", "frameRate=5" in seq and "seqLength=8" in seq
           and "imWidth=1280" in seq)
    _check("seqmap 3 video", (out_root / "seqmaps" / f"{BENCH}-val.txt").read_text().strip().split("\n")[1:] == sorted(df.video.unique()))
    return rc, rm


def test_4_end_to_end():
    """gt-synthetic cache -> runner CV/EKF -> tach lop -> TrackEval crowd-ignore -> occlusion_eval gap."""
    cache_id = "fixture_bddfx_gt"
    cdir = det_cache.CACHE_ROOT / cache_id
    if cdir.exists():
        shutil.rmtree(cdir)
    cmd = [PY, "src/det_cache.py", "--dataset", "bdd100k", "--data-root", str(FIX), "--part", PART, "--all",
           "--cache-id", cache_id, "--source", "gt", "--gt-jitter", "0.02", "--gt-dropout", "0.0", "--gt-seed", "0"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    _check("det_cache --source gt tren fixture chay xong", r.returncode == 0, r.stdout[-800:] + r.stderr[-800:])
    meta = json.loads((cdir / "meta.json").read_text())
    _check("meta backend gt-synthetic, dataset bdd100k", meta["backend"] == "gt-synthetic" and meta["dataset"] == "bdd100k")
    z = np.load(cdir / "b1c66a42-6f7d68ca.npz")
    _check("cache b1: 12 frame, lop COCO {1,2,3,5,7} (khong rider/pedestrian/crowd)",
           len(z["frame_idx"]) == 12 and set(z["data"][:, 5].astype(int).tolist()) <= {1, 2, 3, 5, 7}
           and len(z["data"]) == 12 * 5,   # car A, truck B, moto C, crowd(car, co box) , F1 -> 5 box/frame
           f"{len(z['data'])} box, lop {set(z['data'][:, 5].astype(int).tolist())}")

    rows = {}
    for name, yml in [("cv", "bytetrack.yaml"), ("ekf", "configs/bytetrack_ekf_ctrv.yaml")]:
        cmd = [PY, "src/run_tracking.py", "--cache-id", cache_id, "--split-name", f"{BENCH}-val",
               "--tracker-yaml", yml, "--tracker-name", f"fx-{name}", "--all-cached", "--conf-min", "0.25",
               "--classes", "1", "2", "3", "5", "7", "--track-buffer-sec", "1.2", "--elapsed-mode", "steps"]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        _check(f"run_tracking {name} tren cache fixture", r.returncode == 0, r.stdout[-600:] + r.stderr[-600:])
        meta_run = json.loads((config.PROCESSED_DIR / "trackers" / f"{BENCH}-val" / f"fx-{name}" / "run_meta.json").read_text())
        _check(f"{name}: track_buffer 1.2 s @5fps = 6 buoc, fps=5 trong run_meta",
               meta_run["track_buffer_frames"] == 6 and meta_run["fps"] == 5.0)
        tm = pd.read_csv(config.PROCESSED_DIR / "trackers" / f"{BENCH}-val" / f"fx-{name}" / "timing" / "b3e7f1c0-9a2b4d6e.csv")
        _check(f"{name}: b3 co 1 frame n_steps=2 (timestamp nhay 0.4 s)", (tm.n_steps == 2).sum() == 1 and (tm.n_steps == 1).sum() == 7)
        w = split_tracker_output_by_class(config.PROCESSED_DIR / "trackers" / f"{BENCH}-val" / f"fx-{name}",
                                          config.PROCESSED_DIR, PART, ["car", "motorcycle"], benchmark=BENCH)
        rows[name] = w
        _check(f"{name}: tach lop car / motorcycle co dong", w.get(f"{BENCH}-val_car", 0) > 0 and w.get(f"{BENCH}-val_motorcycle", 0) > 0, str(w))

    res = {}
    for split in (f"{BENCH}-val", f"{BENCH}-val_car", f"{BENCH}-val_motorcycle"):
        for name in ("cv", "ekf"):
            results, _ = run_trackeval.run(split, f"fx-{name}", None, "mot_crowd")
            df = run_trackeval.flatten_results(results, f"fx-{name}")
            c = df[df.video == "COMBINED_SEQ"].iloc[0]
            res[(split, name)] = c
    v = res[(f"{BENCH}-val", "cv")]
    _check("TrackEval crowd-ignore tren BDDFX-val chay: HOTA huu han, FP=0 (det trong crowd bi loai, det khop GT)",
           np.isfinite(v.HOTA) and v.FP == 0, f"HOTA={v.HOTA:.3f} MOTA={v.MOTA:.3f} FP={v.FP} FN={v.FN} IDSW={v.IDSW}")
    _check("theo lop: split _car va _motorcycle co GT (GT_dets > 0)",
           res[(f"{BENCH}-val_car", "cv")].GT_dets > 0 and res[(f"{BENCH}-val_motorcycle", "cv")].GT_dets > 0)
    _check("EKF cung chay xong tren 3 split", all(np.isfinite(res[(s, "ekf")].HOTA) for s in
                                                  (f"{BENCH}-val", f"{BENCH}-val_car", f"{BENCH}-val_motorcycle")))

    # occlusion_segments + occlusion_eval voi gap
    ds = BDD100KMOTDataset(FIX)
    cmd = [PY, "src/occlusion_segments.py", "--parquet", str(ds.interim_parquet(PART)), "--fps", "5",
           "--out-tag", TAG, "--min-len", "1"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    _check("occlusion_segments --parquet --fps 5 --out-tag", r.returncode == 0, r.stdout[-500:] + r.stderr[-500:])
    seg = pd.read_csv(config.INTERIM_DIR / f"{TAG}_occlusion_segments.csv")
    full = pd.read_csv(config.INTERIM_DIR / f"{TAG}_full_occlusion_segments.csv")
    gaps = pd.read_csv(config.INTERIM_DIR / f"{TAG}_track_gaps.csv")
    _check("segments: co doan partial (truck 6-8), KHONG co doan full (>=0.90), 2 gap; duration theo 5 fps",
           len(seg) > 0 and len(full) == 0 and len(gaps) == 2 and np.isclose(gaps.sort_values("track_id").duration_s.iloc[0], 0.6),
           f"partial {len(seg)} full {len(full)} gaps {len(gaps)}")
    out_dir = ROOT / "results" / "phase2" / "fixture_occl"
    cmd = [PY, "src/occlusion_eval.py", "--split-name", f"{BENCH}-val", "--trackers", "fx-cv", "fx-ekf",
           "--tag", TAG, "--fps", "5", "--context-s", "1.2", "--gaps", "--n-boot", "200", "--out-dir", str(out_dir)]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    _check("occlusion_eval --gaps --context-s 1.2 --fps 5", r.returncode == 0, r.stdout[-800:] + r.stderr[-800:])
    ev = pd.read_csv(out_dir / f"events_{BENCH}-val.csv")
    _check("events co cot has_gap va it nhat 1 su kien gap (xe G) + su kien partial (truck B)",
           "has_gap" in ev.columns and ev.has_gap.sum() >= 1 and (~ev.has_gap).sum() >= 1, ev.to_string())
    g11 = ev[(ev.track_id == 11) & ev.has_gap]; g15 = ev[(ev.track_id == 15) & ev.has_gap]
    _check("gap KHONG suy dien la che hoan toan: has_full=False; xe G -> gap_evidence 'unknown' "
           "(frame 4, 8 khong occluded); xe M -> 'occluded_adjacent' (frame 6, 9 occluded)",
           len(g11) == 1 and len(g15) == 1 and not g11.has_full.iloc[0] and not g15.has_full.iloc[0]
           and g11.gap_evidence.iloc[0] == "unknown" and g15.gap_evidence.iloc[0] == "occluded_adjacent",
           ev[ev.has_gap][["track_id", "start_frame", "end_frame", "has_full", "gap_evidence"]].to_string())
    outc = pd.read_csv(out_dir / f"event_outcomes_{BENCH}-val.csv")
    _check("outcome cua su kien gap duoc cham (khong NaN) cho ca 2 tracker",
           outc[outc.has_gap & outc.eligible].outcome.notna().all() and set(outc.tracker) == {"fx-cv", "fx-ekf"},
           outc[["tracker", "track_id", "has_gap", "eligible", "outcome"]].to_string())


def main() -> int:
    failed = 0
    try:
        test_1_schema()
        df = test_2_parse_and_stats()
        test_3_export_mot(df)
        test_4_end_to_end()
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
