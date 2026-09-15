"""
test_trackeval_ext.py -- Kiem tra MotChallengeCrowdIgnore (vung bo qua theo IoA, giao thuc BDD100K).

    1. Ba truong hop bat buoc (fixture 3 frame, GT: 1 xe class 1 + 1 box crowd LON class 13 conf 0):
         (a) det CHUA KHOP nam gon trong crowd (IoA = 1, IoU voi crowd << 0.5)
                MotChallenge2DBox goc : tinh FP (3)         <- ly do can subclass
                MotChallengeCrowdIgnore: bi loai -> FP = 0
         (b) det KHOP GT hop le nam TRONG vung crowd (IoU >= 0.5 voi GT, IoA = 1 voi crowd)
                -> phai la TP, KHONG bi loai (ghep GT truoc, chi xet ignore cho det chua khop)
         (c) FP thuong: det chi 40% trong crowd (IoA 0.4 <= 0.5) -> van FP o ca hai lop
    2. GT khong co class 13 (UA-DETRAC that, 1 video, tracker testA-cv): hai lop cho ket qua
       TRUNG KHIT (HOTA/MOTA/IDF1/FP/FN/IDSW) - bao toan danh gia Giai doan 1.
    3. DOI CHIEU VOI ADAPTER GOC `trackeval.datasets.BDD100K`: cung fixture BDD100K-mini, cung
       output tracker (CV tren cache gt-synthetic, lop theo da so), doi sang schema CU cua adapter
       goc (`index`, `attributes.Crowd`, tracker JSON labels[{id, category, box2d}]) -> chay lop goc
       theo tung lop (car / truck / motorcycle) va MotChallengeCrowdIgnore tren split
       BDDFX-val_<lop> -> HOTA / MOTA / IDF1 / FP / FN / IDSW / TP phai TRUNG NHAU.

    python src/test_trackeval_ext.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import run_tracking  # noqa: E402
import run_trackeval  # noqa: E402
from trackeval_ext import MotChallengeCrowdIgnore  # noqa: E402

import trackeval  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_RESULTS: list[tuple[str, bool]] = []
GT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def _metrics(r) -> dict:
    return {"HOTA": float(np.mean(r["HOTA"]["HOTA"])), "MOTA": float(r["CLEAR"]["MOTA"]),
            "IDF1": float(r["Identity"]["IDF1"]), "FP": int(r["CLEAR"]["CLR_FP"]),
            "FN": int(r["CLEAR"]["CLR_FN"]), "IDSW": int(r["CLEAR"]["IDSW"]), "TP": int(r["CLEAR"]["CLR_TP"])}


def _eval_cfg():
    ecfg = trackeval.Evaluator.get_default_eval_config()
    ecfg.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False, "PRINT_ONLY_COMBINED": True,
                 "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "TIME_PROGRESS": False,
                 "LOG_ON_ERROR": None})
    mcfg = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5, "PRINT_CONFIG": False}
    return ecfg, [trackeval.metrics.HOTA(mcfg), trackeval.metrics.CLEAR(mcfg), trackeval.metrics.Identity(mcfg)]


def evaluate(gt_root: Path, trackers_root: Path, benchmark: str, split: str, tracker: str, cls):
    """MotChallenge2DBox hoac MotChallengeCrowdIgnore tren thu muc tuy y (class 'pedestrian' = id 1)."""
    dcfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update({"GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(trackers_root), "BENCHMARK": benchmark,
                 "SPLIT_TO_EVAL": split, "TRACKERS_TO_EVAL": [tracker], "CLASSES_TO_EVAL": ["pedestrian"],
                 "SEQMAP_FOLDER": str(gt_root / "seqmaps"), "SKIP_SPLIT_FOL": False, "PRINT_CONFIG": False,
                 "OUTPUT_FOLDER": str(trackers_root / "_out")})
    ecfg, metrics = _eval_cfg()
    res, _ = trackeval.Evaluator(ecfg).evaluate([cls(dcfg)], metrics)
    return _metrics(next(iter(res.values()))[tracker]["COMBINED_SEQ"]["pedestrian"])


def evaluate_native_bdd(gt_dir: Path, trk_root: Path, tracker: str, category: str):
    """Adapter goc trackeval.datasets.BDD100K (schema cu), mot lop."""
    dcfg = trackeval.datasets.BDD100K.get_default_dataset_config()
    dcfg.update({"GT_FOLDER": str(gt_dir), "TRACKERS_FOLDER": str(trk_root), "TRACKERS_TO_EVAL": [tracker],
                 "CLASSES_TO_EVAL": [category], "TRACKER_SUB_FOLDER": "data", "PRINT_CONFIG": False,
                 "OUTPUT_FOLDER": str(trk_root / "_out")})
    ecfg, metrics = _eval_cfg()
    ds = trackeval.datasets.BDD100K(dcfg)
    # Chi can ket qua theo LOP. Buoc gop super-category cua TrackEval 1.3.0 loi (IndexError trong
    # HOTA.combine_classes_det_averaged) khi chi danh gia 1 lop -> tat gop, khong anh huong so theo lop.
    ds.should_classes_combine = False
    ds.use_super_categories = False
    res, _ = trackeval.Evaluator(ecfg).evaluate([ds], metrics)
    return _metrics(res["BDD100K"][tracker]["COMBINED_SEQ"][category])


def make_fixture(root: Path, case: str):
    """1 seq 'fx1', 3 frame. GT: xe id1 (class 1) + crowd 300x200 tai (400,100) (class 13, conf 0).
    case 'inside'   : det 2 nam gon trong crowd, khong khop GT nao        -> (a)
    case 'match'    : them GT xe id 3 NAM TRONG crowd; det 2 khop xe id 3 -> (b)
    case 'partial'  : det 2 chi 40% trong crowd                          -> (c)
    """
    seq = root / "FX-val" / "fx1"
    (seq / "gt").mkdir(parents=True)
    gt, trk = [], []
    for f in (1, 2, 3):
        gt.append(f"{f},1,{100 + 10 * f},100,50,40,1,1,1.0")
        gt.append(f"{f},9,400,100,300,200,0,13,0.0")
        trk.append(f"{f},1,{100 + 10 * f},100,50,40,0.9")
        if case == "inside":
            trk.append(f"{f},2,450,150,50,40,0.8")
        elif case == "match":
            gt.append(f"{f},3,{450 + 5 * f},150,50,40,1,1,1.0")            # xe hop le BEN TRONG crowd
            trk.append(f"{f},2,{452 + 5 * f},151,50,40,0.8")               # khop (IoU ~0.9), IoA crowd = 1
        elif case == "partial":
            trk.append(f"{f},2,370,150,50,40,0.8")                         # [370,420] giao [400,700] = 20/50
    (seq / "gt" / "gt.txt").write_text("\n".join(gt) + "\n")
    (seq / "seqinfo.ini").write_text("[Sequence]\nname=fx1\nimDir=img1\nframeRate=5\nseqLength=3\n"
                                     "imWidth=1280\nimHeight=720\nimExt=.jpg\n")
    (root / "seqmaps").mkdir(exist_ok=True)
    (root / "seqmaps" / "FX-val.txt").write_text("name\nfx1\n")
    td = root / "trackers" / "FX-val" / "trk" / "data"
    td.mkdir(parents=True)
    (td / "fx1.txt").write_text("\n".join(trk) + "\n")


def test_1_three_ignore_cases():
    for case in ("inside", "match", "partial"):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_fixture(root, case)
            base = evaluate(root, root / "trackers", "FX", "val", "trk", trackeval.datasets.MotChallenge2DBox)
            ext = evaluate(root, root / "trackers", "FX", "val", "trk", MotChallengeCrowdIgnore)
            if case == "inside":
                _check("(a) goc: det chua khop trong box crowd bi tinh FP (3)", base["FP"] == 3 and base["TP"] == 3, str(base))
                _check("(a) crowd-ignore: det trong crowd (IoA=1) bi loai -> FP 0, TP 3, MOTA 1",
                       ext["FP"] == 0 and ext["TP"] == 3 and abs(ext["MOTA"] - 1.0) < 1e-9, str(ext))
                _check("(a) crowd-ignore: GT crowd khong bi tinh FN", ext["FN"] == 0)
            elif case == "match":
                _check("(b) det KHOP GT hop le nam trong crowd: van TP (TP 6, FP 0, FN 0) o crowd-ignore",
                       ext["TP"] == 6 and ext["FP"] == 0 and ext["FN"] == 0 and ext["IDSW"] == 0, str(ext))
                _check("(b) lop goc cung TP 6 (ghep GT truoc; crowd khong lam mat det da khop)",
                       base["TP"] == 6 and base["FP"] == 0, str(base))
            else:
                _check("(c) det chi 40% trong crowd (IoA 0.4 <= 0.5): van FP o ca hai lop",
                       base["FP"] == 3 and ext["FP"] == 3, f"{base} / {ext}")


def test_2_identical_on_detrac_without_class13():
    split = config.PROCESSED_DIR / "DETRAC-test"
    trk = config.PROCESSED_DIR / "trackers" / "DETRAC-test" / "testA-cv" / "data" / "MVI_39031.txt"
    if not (split / "MVI_39031" / "gt" / "gt.txt").exists() or not trk.exists():
        print("  [BO QUA] khong co DETRAC-test / testA-cv tren may")
        return
    eval_cfg, dcfg, mcfg = run_trackeval.build_configs("DETRAC-test", "testA-cv", ["MVI_39031"])
    eval_cfg.update({"PRINT_RESULTS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False})
    dcfg["OUTPUT_FOLDER"] = tempfile.mkdtemp()
    out = {}
    for name, cls in [("mot", trackeval.datasets.MotChallenge2DBox), ("mot_crowd", MotChallengeCrowdIgnore)]:
        ev = trackeval.Evaluator(eval_cfg)
        res, _ = ev.evaluate([cls(dcfg)], [trackeval.metrics.HOTA(mcfg), trackeval.metrics.CLEAR(mcfg),
                                           trackeval.metrics.Identity(mcfg)])
        out[name] = _metrics(next(iter(res.values()))["testA-cv"]["MVI_39031"]["pedestrian"])
    _check("UA-DETRAC (khong class 13): mot_crowd == mot (HOTA, MOTA, IDF1, FP, FN, IDSW)",
           out["mot"] == out["mot_crowd"], f"mot={out['mot']}\nmot_crowd={out['mot_crowd']}")


def _to_old_schema_gt(frames: list) -> list:
    """box_track_20 -> schema cu cua trackeval/datasets/bdd100k.py: `index`, attributes.Crowd."""
    out = []
    for f in frames:
        labels = [{"id": lb["id"], "category": lb["category"], "box2d": lb["box2d"],
                   "attributes": {"Crowd": bool(lb["attributes"]["crowd"]), "Occluded": bool(lb["attributes"]["occluded"]),
                                  "Truncated": bool(lb["attributes"]["truncated"])}} for lb in f["labels"]]
        out.append({"name": f["name"], "videoName": f["videoName"], "index": int(f["frameIndex"]), "labels": labels})
    return out


def test_3_cross_check_with_native_bdd100k_adapter():
    from mot_datasets.bdd100k_mot import (BDD100KMOTDataset, COCO_TO_BDD, export_motchallenge,
                                          split_tracker_output_by_class)
    fix = ROOT / "data" / "fixtures" / "bdd100k_mini"
    ds = BDD100KMOTDataset(fix)
    df = ds.build_interim("val")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        # --- GT MOT theo lop (duong cua ta) ---
        for target in ("car", "truck", "motorcycle"):
            export_motchallenge(df, "val", target, out_root=root, benchmark="BDDX")
        # --- cache gt-synthetic + tracker CV (in-process), tach lop theo da so ---
        spec = det_cache.DetectorSpec(backend="gt-synthetic", conf=0.05, classes=None, max_det=3000,
                                      gt_jitter=0.02, gt_dropout=0.1, gt_seed=0, dataset="bdd100k", part="val")
        cfg = run_tracking.load_tracker_cfg("bytetrack.yaml", {"track_buffer": 6})
        trk_dir = root / "trackers_raw" / "fx-cv"
        seq_len = {}
        for v in ds.list_videos("val"):
            frames = ds.frames(v, "val")
            seq_len[v] = len(frames)
            arrays = det_cache.build_video_gt(df[df.video == v], frames, spec, ds.orig_shape, ds.category_to_coco)
            det_cache.save_video(root / "cache", v, arrays)
            vd = det_cache.VideoDetections(root / "cache" / f"{v}.npz", 0.25, [1, 2, 3, 5, 7], 300)
            run_tracking.run_video_to_dir(trk_dir, v, vd, cfg, 0.2, "steps")
        split_tracker_output_by_class(trk_dir, root, "val", ["car", "truck", "motorcycle"], benchmark="BDDX")
        # --- schema CU cho adapter goc: GT + tracker JSON (cung lop theo da so) ---
        gt_old = root / "native" / "gt"; gt_old.mkdir(parents=True)
        trk_old = root / "native" / "trackers" / "fx-cv" / "data"; trk_old.mkdir(parents=True)
        for v in ds.list_videos("val"):
            (gt_old / f"{v}.json").write_text(json.dumps(_to_old_schema_gt(ds.load_json(v, "val"))), encoding="utf-8")
            det = pd.read_csv(trk_dir / "data" / f"{v}.txt", header=None, names=["frame", "id", "x", "y", "w", "h", "conf"])
            cl = pd.read_csv(trk_dir / "data_cls" / f"{v}.csv")
            cl["bdd"] = cl["cls"].map(COCO_TO_BDD)
            maj = cl.groupby("id")["bdd"].agg(lambda s: s.value_counts().idxmax())
            det["category"] = det["id"].map(maj)
            frames_json = []
            for fi in range(1, seq_len[v] + 1):
                g = det[det.frame == fi]
                frames_json.append({"name": f"{v}-{fi:07d}.jpg", "index": fi - 1,
                                    "labels": [{"id": int(r.id), "category": r.category,
                                                "box2d": {"x1": float(r.x), "y1": float(r.y), "x2": float(r.x + r.w), "y2": float(r.y + r.h)}}
                                               for r in g.itertuples() if isinstance(r.category, str)]})
            (trk_old / f"{v}.json").write_text(json.dumps(frames_json), encoding="utf-8")
        # --- so sanh tung lop ---
        for target in ("car", "truck", "motorcycle"):
            ours = evaluate(root, root / "trackers", "BDDX", f"val_{target}", "fx-cv", MotChallengeCrowdIgnore)
            native = evaluate_native_bdd(gt_old, root / "native" / "trackers", "fx-cv", target)
            same = all(abs(ours[k] - native[k]) < 1e-9 if isinstance(ours[k], float) else ours[k] == native[k] for k in ours)
            _check(f"[{target}] MotChallengeCrowdIgnore == trackeval.datasets.BDD100K goc "
                   f"(HOTA {ours['HOTA']:.4f}, MOTA {ours['MOTA']:.4f}, TP {ours['TP']}, FP {ours['FP']}, FN {ours['FN']}, IDSW {ours['IDSW']})",
                   same, f"ta={ours}\ngoc={native}")


def main() -> int:
    failed = 0
    for t in [test_1_three_ignore_cases, test_2_identical_on_detrac_without_class13,
              test_3_cross_check_with_native_bdd100k_adapter]:
        print(f"\n--- {t.__name__} ---")
        try:
            t()
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
