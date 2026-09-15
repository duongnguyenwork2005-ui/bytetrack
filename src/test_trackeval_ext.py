"""
test_trackeval_ext.py -- Kiem tra MotChallengeCrowdIgnore (vung bo qua theo IoA, giao thuc BDD100K).

    1. Fixture 3 frame: GT co 1 xe (class 1) + 1 box crowd LON (class 13, conf 0). Tracker
       ra 2 box/frame: mot khop xe, mot nam GON trong box crowd (IoA = 1, IoU voi crowd << 0.5).
         - MotChallenge2DBox goc      : box trong crowd bi tinh FP  (3 FP)   <- ly do can subclass
         - MotChallengeCrowdIgnore    : bi loai                     (0 FP)
       Det nam MOT PHAN (IoA <= 0.5) trong crowd van la FP o ca hai (dung nguong 0.5).
    2. GT khong co class 13 (UA-DETRAC that, 1 video, tracker testA-cv): hai lop cho ket qua
       TRUNG KHIT (HOTA/MOTA/IDF1/FP/FN/IDSW) - bao toan danh gia Giai doan 1.

    python src/test_trackeval_ext.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import run_trackeval  # noqa: E402
from trackeval_ext import MotChallengeCrowdIgnore  # noqa: E402

import trackeval  # noqa: E402

_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def evaluate(gt_root: Path, trackers_root: Path, benchmark: str, split: str, tracker: str, cls):
    dcfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dcfg.update({"GT_FOLDER": str(gt_root), "TRACKERS_FOLDER": str(trackers_root), "BENCHMARK": benchmark,
                 "SPLIT_TO_EVAL": split, "TRACKERS_TO_EVAL": [tracker], "CLASSES_TO_EVAL": ["pedestrian"],
                 "SEQMAP_FOLDER": str(gt_root / "seqmaps"), "SKIP_SPLIT_FOL": False, "PRINT_CONFIG": False,
                 "OUTPUT_FOLDER": str(trackers_root / "_out")})
    ecfg = trackeval.Evaluator.get_default_eval_config()
    ecfg.update({"USE_PARALLEL": False, "PRINT_RESULTS": False, "PRINT_CONFIG": False, "PRINT_ONLY_COMBINED": True,
                 "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False, "PLOT_CURVES": False, "TIME_PROGRESS": False,
                 "LOG_ON_ERROR": None})
    mcfg = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5, "PRINT_CONFIG": False}
    ev = trackeval.Evaluator(ecfg)
    ds = cls(dcfg)
    res, _ = ev.evaluate([ds], [trackeval.metrics.HOTA(mcfg), trackeval.metrics.CLEAR(mcfg),
                               trackeval.metrics.Identity(mcfg)])
    r = next(iter(res.values()))[tracker]["COMBINED_SEQ"]["pedestrian"]
    return {"HOTA": float(np.mean(r["HOTA"]["HOTA"])), "MOTA": float(r["CLEAR"]["MOTA"]),
            "IDF1": float(r["Identity"]["IDF1"]), "FP": int(r["CLEAR"]["CLR_FP"]),
            "FN": int(r["CLEAR"]["CLR_FN"]), "IDSW": int(r["CLEAR"]["IDSW"]), "TP": int(r["CLEAR"]["CLR_TP"])}


def make_fixture(root: Path, partial: bool):
    """1 seq 'fx1', 3 frame. partial=True: det thu 2 chi nam 40% trong crowd (IoA 0.4)."""
    seq = root / "FX-val" / "fx1"
    (seq / "gt").mkdir(parents=True)
    gt, trk = [], []
    for f in (1, 2, 3):
        gt.append(f"{f},1,{100 + 10 * f},100,50,40,1,1,1.0")          # xe (class 1)
        gt.append(f"{f},9,400,100,300,200,0,13,0.0")                    # crowd 300x200 (class 13, conf 0)
        trk.append(f"{f},1,{100 + 10 * f},100,50,40,0.9")              # khop xe
        if partial:
            trk.append(f"{f},2,370,150,50,40,0.8")                     # [370,420] giao [400,700] = 20/50 -> IoA 0.4
        else:
            trk.append(f"{f},2,450,150,50,40,0.8")                     # gon trong crowd -> IoA 1.0, IoU 0.033
    (seq / "gt" / "gt.txt").write_text("\n".join(gt) + "\n")
    (seq / "seqinfo.ini").write_text("[Sequence]\nname=fx1\nimDir=img1\nframeRate=5\nseqLength=3\n"
                                     "imWidth=1280\nimHeight=720\nimExt=.jpg\n")
    (root / "seqmaps").mkdir(exist_ok=True)
    (root / "seqmaps" / "FX-val.txt").write_text("name\nfx1\n")
    td = root / "trackers" / "FX-val" / "trk" / "data"
    td.mkdir(parents=True)
    (td / "fx1.txt").write_text("\n".join(trk) + "\n")


def test_1_crowd_ignore_fixture():
    for partial in (False, True):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            make_fixture(root, partial)
            base = evaluate(root, root / "trackers", "FX", "val", "trk", trackeval.datasets.MotChallenge2DBox)
            ext = evaluate(root, root / "trackers", "FX", "val", "trk", MotChallengeCrowdIgnore)
            if not partial:
                _check("goc: det trong box crowd bi tinh FP (3)", base["FP"] == 3 and base["TP"] == 3, str(base))
                _check("crowd-ignore: det trong box crowd (IoA=1) bi loai -> FP = 0, TP = 3, MOTA = 1",
                       ext["FP"] == 0 and ext["TP"] == 3 and abs(ext["MOTA"] - 1.0) < 1e-9, str(ext))
                _check("crowd-ignore: GT crowd khong bi tinh FN", ext["FN"] == 0)
            else:
                _check("det chi 40% trong crowd (IoA 0.4 <= 0.5): van FP o ca hai lop",
                       base["FP"] == 3 and ext["FP"] == 3, f"{base} / {ext}")


def test_2_identical_on_detrac_without_class13():
    split = config.PROCESSED_DIR / "DETRAC-test"
    trk = config.PROCESSED_DIR / "trackers" / "DETRAC-test" / "testA-cv" / "data" / "MVI_39031.txt"
    if not (split / "MVI_39031" / "gt" / "gt.txt").exists() or not trk.exists():
        print("  [BO QUA] khong co DETRAC-test / testA-cv tren may")
        return
    cfgs = run_trackeval.build_configs("DETRAC-test", "testA-cv", ["MVI_39031"])
    eval_cfg, dcfg, mcfg = cfgs
    eval_cfg.update({"PRINT_RESULTS": False, "OUTPUT_SUMMARY": False, "OUTPUT_DETAILED": False})
    dcfg["OUTPUT_FOLDER"] = tempfile.mkdtemp()
    out = {}
    for name, cls in [("mot", trackeval.datasets.MotChallenge2DBox), ("mot_crowd", MotChallengeCrowdIgnore)]:
        ev = trackeval.Evaluator(eval_cfg)
        res, _ = ev.evaluate([cls(dcfg)], [trackeval.metrics.HOTA(mcfg), trackeval.metrics.CLEAR(mcfg),
                                           trackeval.metrics.Identity(mcfg)])
        r = next(iter(res.values()))["testA-cv"]["MVI_39031"]["pedestrian"]
        out[name] = (float(np.mean(r["HOTA"]["HOTA"])), float(r["CLEAR"]["MOTA"]), float(r["Identity"]["IDF1"]),
                     int(r["CLEAR"]["CLR_FP"]), int(r["CLEAR"]["CLR_FN"]), int(r["CLEAR"]["IDSW"]))
    _check("UA-DETRAC (khong class 13): mot_crowd == mot (HOTA, MOTA, IDF1, FP, FN, IDSW)",
           out["mot"] == out["mot_crowd"], f"mot={out['mot']}\nmot_crowd={out['mot_crowd']}")


def main() -> int:
    failed = 0
    for t in [test_1_crowd_ignore_fixture, test_2_identical_on_detrac_without_class13]:
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
