"""
test_evaluator_choice.py -- Lop evaluator phai duoc chon theo LOAI DATASET, khong theo ten thu muc
output (review vong 6, finding 1).

Loi da xay ra: quy tac cu `split.startswith("BDD-")` nen split smoke dat ten `BDDSMOKE-val` bi cham
bang `mot` va MAT xu ly crowd-ignore (GT co vung bo qua class 13 conf 0).

  1. `dataset_class_for()` chon dung theo dataset, KHONG theo ten split; dataset la -> LOI RO RANG.
  2. Ten split KHONG con quyet dinh chinh sach: `BDDSMOKE-val` + dataset bdd -> `mot_crowd`.
  3. Lua chon evaluator nam trong PROVENANCE va DAU VAN TAY -> doi lop => phai danh gia lai.
  4. HANH VI thuc te tren fixture mot frame (GT: 1 muc tieu + 1 vung bo qua; tracking: 1 khop dung,
     1 roi vao vung bo qua, 1 FP thuong):
        - `mot`       : FP = 2  (KHONG loai detection trong vung bo qua)
        - `mot_crowd` : FP = 1  (loai dung mot cai)
        - ca hai      : detection khop GT hop le VAN duoc giu (TP = 1)
  5. Chay qua `run_experiment.stage_eval()` THAT (review ca93c1e, P3 -- truoc day phep kiem nay
     chi TINH LAI bieu thuc chon lop roi kiem `rex is not None`, khong he goi runner, nen van PASS
     ke ca khi duong goi evaluator trong runner bi sua sai):
       5a. runner chon `mot_crowd` va GHI vao `eval_meta.json`; FP = 1 (da loai detection trong
           vung bo qua), TP = 1 (detection khop GT hop le VAN duoc giu).
       5b. DO NHAY: ep runner dung `mot` tren cung fixture -> FP = 2. Neu khong doi thi phep kiem
           nay vo nghia, nen chenh lech FP chinh la bang chung test bat duoc loi chon evaluator.
       5c. GT hop le CHONG LEN vung bo qua: detection khop no VAN la TP, khong bi vung ignore nuot.

    python src/test_evaluator_choice.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines()[-8:]:
            print(f"         {line}")
    assert ok, name


def make_fixture(root: Path, split: str, tracker: str) -> None:
    """Mot video, MOT frame.

    GT      : muc tieu (class 1, conf 1) o (100,100,50,50); VUNG BO QUA (class 13, conf 0) o (400,100,80,80)
    Tracking: (a) khop muc tieu, (b) nam GON trong vung bo qua, (c) FP thuong o goc khac
    """
    seq = root / split / "v1"
    (seq / "gt").mkdir(parents=True, exist_ok=True)
    (seq / "gt" / "gt.txt").write_text(
        "1,1,100,100,50,50,1,1,1.000\n"
        "1,2,400,100,80,80,0,13,0.000\n", encoding="utf-8")
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=v1\nimDir=img1\nframeRate=5\nseqLength=1\nimWidth=1280\nimHeight=720\n"
        "imExt=.jpg\n", encoding="utf-8")
    (root / "seqmaps").mkdir(parents=True, exist_ok=True)
    (root / "seqmaps" / f"{split}.txt").write_text("name\nv1\n", encoding="utf-8")
    td = root / "trackers" / split / tracker / "data"
    td.mkdir(parents=True, exist_ok=True)
    (td / "v1.txt").write_text(
        "1,1,100,100,50,50,0.9,-1,-1,-1\n"       # (a) khop GT muc tieu
        "1,2,410,110,60,60,0.9,-1,-1,-1\n"       # (b) nam trong vung bo qua
        "1,3,800,400,50,50,0.9,-1,-1,-1\n",      # (c) FP thuong
        encoding="utf-8")


def make_fixture_chong_lan(root: Path, split: str, tracker: str) -> None:
    """GT hop le NAM CHONG LEN vung bo qua -> detection khop no phai VAN duoc tinh la TP.

    GT      : muc tieu A (100,100,50,50); VUNG BO QUA (400,100,80,80);
              muc tieu B (405,105,70,70) -- chong len vung bo qua
    Tracking: khop A, khop B, va mot detection nam GON trong vung bo qua nhung khong khop GT nao
    """
    seq = root / split / "v1"
    (seq / "gt").mkdir(parents=True, exist_ok=True)
    (seq / "gt" / "gt.txt").write_text(
        "1,1,100,100,50,50,1,1,1.000\n"
        "1,2,400,100,80,80,0,13,0.000\n"
        "1,3,405,105,70,70,1,1,1.000\n", encoding="utf-8")
    (seq / "seqinfo.ini").write_text(
        "[Sequence]\nname=v1\nimDir=img1\nframeRate=5\nseqLength=1\nimWidth=1280\nimHeight=720\n"
        "imExt=.jpg\n", encoding="utf-8")
    (root / "seqmaps").mkdir(parents=True, exist_ok=True)
    (root / "seqmaps" / f"{split}.txt").write_text("name\nv1\n", encoding="utf-8")
    td = root / "trackers" / split / tracker / "data"
    td.mkdir(parents=True, exist_ok=True)
    (td / "v1.txt").write_text(
        "1,1,100,100,50,50,0.9,-1,-1,-1\n"        # khop muc tieu A
        "1,2,405,105,70,70,0.9,-1,-1,-1\n"        # khop muc tieu B (chong vung bo qua)
        "1,3,470,160,12,12,0.9,-1,-1,-1\n",       # trong vung bo qua, khong khop GT nao
        encoding="utf-8")


def _stage_eval_that(rex, config, proc: Path, res: Path, split: str, tracker: str,
                     dataset_cls: str | None = None) -> tuple[dict, dict]:
    """Goi `run_experiment.stage_eval()` THAT; tra ve (metric COMBINED, eval_meta.json)."""
    import contextlib
    import io
    import pandas as pd
    exp = {"name": "evalchoice", "split_name": split, "dataset": "bdd100k_coco",
           "cache_id": "khong-co", "classes": [2], "max_det": 300, "fps": 5,
           "elapsed_mode": "steps", "conditions": {"c25": {"conf_min": 0.25}}, "occlusion": {}}
    if dataset_cls:
        exp["dataset_class"] = dataset_cls
    rows = [{"cond": "c25", "tracker": "cv", "yaml": "bytetrack.yaml", "set": {}, "seed": None,
             "conf_min": 0.25, "dir": tracker, "tracker_type": "bytetrack", "track_buffer_sec": None}]
    old_p, old_r = config.PROCESSED_DIR, config.RESULTS_DIR
    try:
        config.PROCESSED_DIR, config.RESULTS_DIR = proc, res
        with contextlib.redirect_stdout(io.StringIO()):
            rex.stage_eval(exp, rows, ["v1"], force=True)
        d = pd.read_csv(res / "trackeval" / split / tracker / "combined_metrics.csv").iloc[0]
        meta = json.loads((res / "trackeval" / split / tracker / "eval_meta.json").read_text())
    finally:
        config.PROCESSED_DIR, config.RESULTS_DIR = old_p, old_r
    return {"FP": int(d["FP"]), "FN": int(d["FN"]), "GT_dets": int(d["GT_dets"]),
            "HOTA": round(float(d["HOTA"]), 7)}, meta


def main() -> int:
    failed = 0
    tmp = Path(tempfile.mkdtemp(prefix="evalchoice_"))
    try:
        import config
        import run_trackeval as rt

        # --- 1 + 2. chon theo dataset, khong theo ten thu muc ---
        cases = [("bdd100k_coco", "BDDSMOKE-val", "mot_crowd"),
                 ("bdd100k_coco", "TEN-BAT-KY", "mot_crowd"),
                 ("bdd100k", "BDD-val", "mot_crowd"),
                 ("detrac", "DETRAC-test", "mot"),
                 ("detrac", "BDD-nham-ten", "mot")]
        sai = [(ds, sp, rt.dataset_class_for(ds, sp), want) for ds, sp, want in cases
               if rt.dataset_class_for(ds, sp) != want]
        _check("lop evaluator chon theo LOAI DATASET, ten thu muc output KHONG quyet dinh chinh sach",
               not sai, f"sai: {sai}")

        err = ""
        try:
            rt.dataset_class_for("dataset_la", "X")
        except ValueError as e:
            err = str(e)
        _check("dataset khong khai bao -> LOI RO RANG (khong am tham chon 'mot')",
               "khong biet dung lop evaluator" in err, err[:120] or "(khong bao loi!)")

        # --- 3. nam trong dau van tay ---
        import run_fingerprint as fp
        a = fp.evaluator_fingerprint("mot")
        b = fp.evaluator_fingerprint("mot_crowd")
        _check("lua chon evaluator nam trong provenance/dau van tay -> doi lop thi phai danh gia lai",
               a != b and a["dataset_class"] == "mot" and b["dataset_class"] == "mot_crowd",
               f"{a} vs {b}")

        # --- 4. HANH VI: mot vs mot_crowd tren cung fixture ---
        split, tracker = "BDDSMOKE-fixture", "trk"
        make_fixture(tmp, split, tracker)
        old_proc = config.PROCESSED_DIR
        ket = {}
        try:
            config.PROCESSED_DIR = tmp
            rt.config.PROCESSED_DIR = tmp
            for cls in ("mot", "mot_crowd"):
                res, _ = rt.run(split, tracker, ["v1"], cls)
                # khoa cap cao la ten lop dataset cua TrackEval (khac nhau giua mot / mot_crowd)
                ds_key = next(iter(res))
                r = res[ds_key][tracker]["COMBINED_SEQ"]["pedestrian"]
                ket[cls] = {"FP": int(r["CLEAR"]["CLR_FP"]), "TP": int(r["CLEAR"]["CLR_TP"]),
                            "FN": int(r["CLEAR"]["CLR_FN"]), "Dets": int(r["Count"]["Dets"])}
        finally:
            config.PROCESSED_DIR = old_proc
            rt.config.PROCESSED_DIR = old_proc
        _check("HANH VI: detection roi vao vung bo qua bi loai voi `mot_crowd` (FP 2 -> 1), "
               "detection khop GT hop le VAN duoc giu (TP = 1), FP thuong VAN bi tinh",
               ket["mot"]["FP"] == 2 and ket["mot_crowd"]["FP"] == 1
               and ket["mot"]["TP"] == 1 and ket["mot_crowd"]["TP"] == 1,
               f"mot={ket['mot']}  mot_crowd={ket['mot_crowd']}")

        # --- 5. qua stage_eval() THAT (P3) ---
        import run_experiment as rex

        proc5, res5 = tmp / "p5", tmp / "r5"
        split5, trk5 = "BDDSMOKE-val", "p2-evalchoice-c25-cv"
        make_fixture(proc5, split5, trk5)

        m_auto, meta_auto = _stage_eval_that(rex, config, proc5, res5, split5, trk5)
        _check("5a. `stage_eval()` THAT voi split 'BDDSMOKE-*' + dataset bdd100k_coco: runner dung "
               "`mot_crowd`, ghi vao `eval_meta.json`; FP = 1 (loai detection trong vung bo qua), "
               "GT_dets = 1",
               meta_auto.get("evaluator", {}).get("dataset_class") == "mot_crowd"
               and m_auto["FP"] == 1 and m_auto["GT_dets"] == 1,
               f"eval_meta.evaluator={meta_auto.get('evaluator')}  metric={m_auto}")

        m_mot, meta_mot = _stage_eval_that(rex, config, proc5, res5, split5, trk5, dataset_cls="mot")
        _check("5b. DO NHAY: ep runner dung `mot` tren CUNG fixture -> FP = 2. Chenh lech nay la "
               "bang chung phep kiem bat duoc loi chon evaluator (truoc day khong bat duoc)",
               meta_mot.get("evaluator", {}).get("dataset_class") == "mot"
               and m_mot["FP"] == 2 and m_mot["FP"] != m_auto["FP"],
               f"mot: FP={m_mot['FP']}  mot_crowd: FP={m_auto['FP']}")

        proc5b, res5b = tmp / "p5b", tmp / "r5b"
        split5b = "BDDSMOKE-chonglan"
        make_fixture_chong_lan(proc5b, split5b, trk5)
        m_ov, meta_ov = _stage_eval_that(rex, config, proc5b, res5b, split5b, trk5)
        _check("5c. GT hop le CHONG LEN vung bo qua: detection khop no VAN la TP (2 GT duoc cham, "
               "khong bi vung ignore nuot); detection trong vung bo qua khong khop GT thi bi loai",
               meta_ov.get("evaluator", {}).get("dataset_class") == "mot_crowd"
               and m_ov["GT_dets"] == 2 and m_ov["FN"] == 0 and m_ov["FP"] == 0,
               f"metric={m_ov}")

    except AssertionError:
        failed += 1
    except Exception as e:  # noqa: BLE001
        failed += 1
        print(f"  [FAIL] loi khong mong doi: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
