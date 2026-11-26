"""
test_eval_guard.py -- Guard cua buoc DANH GIA: dau vao danh gia doi thi metric cu phai mat hieu luc.
(review `handoff/CODEX_REVIEW_ca93c1e.md`, finding P1)

LOI DA XAY RA: `run_fingerprint()` co tracking/cau hinh/cache/code/evaluator nhung KHONG co noi dung
ground truth. Doi `gt.txt` tai CUNG duong dan -> `stage_eval()` in "TAI SU DUNG" va giu nguyen metric
cu; `eval_meta.json` khong doi du hash GT da doi. Tai hien truoc khi sua:
`handoff/logs/review7/repro_F1_truoc_khi_sua.log`.

Moi phep kiem duoi day di qua `run_experiment.stage_eval()` THAT voi TrackEval THAT (khong mock metric):

  1. Dau vao khong doi                  -> TAI SU DUNG (khong chay lai vo co).
  2. Doi `gt.txt` tai cung duong dan     -> CHAY LAI va metric doi DUNG gia tri cua GT moi.
  3. Doi `seqLength` trong seqinfo.ini   -> KHONG tai su dung (do dai timeline la dau vao danh gia).
  4. Xoa `gt.txt` bat buoc               -> BAO LOI, KHONG tra metric cu.
  5. GT RONG HOP LE                      -> KHAC han GT THIEU (khong bi lan thanh mot).
  6. Doi nguon annotation che khuat      -> `occl_fingerprint` doi.
  7. `run_state.py` nam trong CODE_FILES -> sua chinh sach resume lam mat hieu luc ket qua.

Kich ban GT dung dung tinh huong cua bao cao review: mot dong crowd (`conf=0, class=13`) doi thanh
GT hop le (`conf=1, class=1`), de doi chieu truc tiep voi bang so trong bao cao do.

    python src/test_eval_guard.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_RESULTS: list[tuple[str, bool]] = []

#: GT goc: 1 muc tieu hop le + 1 VUNG BO QUA (crowd)
GT_CROWD = ("1,1,100,100,50,50,1,1,1.000\n"
            "1,2,400,100,80,80,0,13,0.000\n")
#: Cung file, dong 2 doi thanh GT HOP LE -> so muc tieu 1 -> 2
GT_HOPLE = ("1,1,100,100,50,50,1,1,1.000\n"
            "1,2,400,100,80,80,1,1,1.000\n")
#: Tracking: (a) khop muc tieu, (b) nam trong vung bo qua, (c) FP thuong
TRK = ("1,1,100,100,50,50,0.9,-1,-1,-1\n"
       "1,2,410,110,60,60,0.9,-1,-1,-1\n"
       "1,3,800,400,50,50,0.9,-1,-1,-1\n")


def _check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines()[-6:]:
            print(f"         {line}")
    assert ok, name


def _seqinfo(seq_len: int) -> str:
    return (f"[Sequence]\nname=v1\nimDir=img1\nframeRate=5\nseqLength={seq_len}\n"
            f"imWidth=1280\nimHeight=720\nimExt=.jpg\n")


def make_fixture(proc: Path, split: str, tracker: str, gt: str = GT_CROWD, seq_len: int = 1) -> None:
    seq = proc / split / "v1"
    (seq / "gt").mkdir(parents=True, exist_ok=True)
    (seq / "gt" / "gt.txt").write_text(gt, encoding="utf-8")
    (seq / "seqinfo.ini").write_text(_seqinfo(seq_len), encoding="utf-8")
    (proc / "seqmaps").mkdir(parents=True, exist_ok=True)
    (proc / "seqmaps" / f"{split}.txt").write_text("name\nv1\n", encoding="utf-8")
    td = proc / "trackers" / split / tracker / "data"
    td.mkdir(parents=True, exist_ok=True)
    (td / "v1.txt").write_text(TRK, encoding="utf-8")


def _exp(split: str) -> dict:
    return {"name": "guard", "split_name": split, "dataset": "bdd100k_coco", "cache_id": "khong-co",
            "classes": [2], "max_det": 300, "fps": 5, "elapsed_mode": "steps",
            "conditions": {"c25": {"conf_min": 0.25}}, "occlusion": {}}


def _rows(tracker: str) -> list[dict]:
    return [{"cond": "c25", "tracker": "cv", "yaml": "bytetrack.yaml", "set": {}, "seed": None,
             "conf_min": 0.25, "dir": tracker, "tracker_type": "bytetrack", "track_buffer_sec": None}]


def _metric(res: Path, split: str, tracker: str) -> dict:
    import pandas as pd
    d = pd.read_csv(res / "trackeval" / split / tracker / "combined_metrics.csv").iloc[0]
    return {"GT_dets": int(d["GT_dets"]), "IDF1": round(float(d["IDF1"]), 7),
            "HOTA": round(float(d["HOTA"]), 7)}


def _run(rex, exp, rows, videos, force=False) -> str:
    """Chay stage_eval that, bat stdout de biet no TAI SU DUNG hay CHAY LAI."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rex.stage_eval(exp, rows, videos, force=force)
    return buf.getvalue()


def main() -> int:  # noqa: C901
    failed = 0
    tmp = Path(tempfile.mkdtemp(prefix="evalguard_"))
    try:
        import config
        import run_experiment as rex
        import run_fingerprint as fpm

        proc, res = tmp / "processed", tmp / "results"
        split, tracker = "BDDGUARD-val", "p2-guard-c25-cv"
        old_p, old_r = config.PROCESSED_DIR, config.RESULTS_DIR
        try:
            config.PROCESSED_DIR, config.RESULTS_DIR = proc, res
            make_fixture(proc, split, tracker)
            exp, rows = _exp(split), _rows(tracker)

            # --- 1. lan dau + khong doi gi -> tai su dung ---
            _run(rex, exp, rows, ["v1"])
            m0 = _metric(res, split, tracker)
            meta0 = json.loads((res / "trackeval" / split / tracker / "eval_meta.json").read_text())
            out2 = _run(rex, exp, rows, ["v1"])
            _check("dau vao KHONG doi -> TAI SU DUNG (guard khong chay lai vo co)",
                   "TAI SU DUNG" in out2 and _metric(res, split, tracker) == m0,
                   f"m0={m0}  out={out2.strip()[-90:]}")

            # --- 2. doi gt.txt tai CUNG duong dan -> chay lai, metric doi dung ---
            (proc / split / "v1" / "gt" / "gt.txt").write_text(GT_HOPLE, encoding="utf-8")
            out3 = _run(rex, exp, rows, ["v1"])
            m1 = _metric(res, split, tracker)
            meta1 = json.loads((res / "trackeval" / split / tracker / "eval_meta.json").read_text())
            _check("doi `gt.txt` tai cung duong dan -> CHAY LAI, metric doi dung GT moi "
                   "(GT_dets 1->2, IDF1 0.6666667->0.8, HOTA 0.7071068->0.6832349)",
                   "CHAY LAI" in out3 and "gt" in out3
                   and m0 == {"GT_dets": 1, "IDF1": 0.6666667, "HOTA": 0.7071068}
                   and m1 == {"GT_dets": 2, "IDF1": 0.8, "HOTA": 0.6832349},
                   f"truoc={m0}  sau={m1}")
            _check("`eval_meta.json` phan anh dau van tay GT moi (truoc day khong doi)",
                   meta0.get("gt") != meta1.get("gt") and meta1.get("gt", {}).get("gt_sha256"),
                   f"gt_sha256: {str(meta0.get('gt', {}).get('gt_sha256'))[:16]} -> "
                   f"{str(meta1.get('gt', {}).get('gt_sha256'))[:16]}")

            # --- 3. doi seqLength -> khong duoc tai su dung ---
            (proc / split / "v1" / "seqinfo.ini").write_text(_seqinfo(50), encoding="utf-8")
            out4 = _run(rex, exp, rows, ["v1"])
            _check("doi `seqLength` (do dai timeline) -> KHONG tai su dung ket qua cu",
                   "CHAY LAI" in out4, out4.strip()[-120:])
            (proc / split / "v1" / "seqinfo.ini").write_text(_seqinfo(1), encoding="utf-8")
            _run(rex, exp, rows, ["v1"])

            # --- 4. xoa gt.txt bat buoc -> BAO LOI, khong tra metric cu ---
            (proc / split / "v1" / "gt" / "gt.txt").unlink()
            loi = ""
            try:
                _run(rex, exp, rows, ["v1"])
            except fpm.ThieuDauVaoDanhGia as e:
                loi = str(e)
            _check("xoa `gt.txt` bat buoc -> BAO LOI (ThieuDauVaoDanhGia), KHONG tra metric cu",
                   "thieu" in loi and "gt.txt" in loi, loi[:160] or "(KHONG bao loi!)")

            loi_ini = ""
            (proc / split / "v1" / "gt" / "gt.txt").write_text(GT_HOPLE, encoding="utf-8")
            (proc / split / "v1" / "seqinfo.ini").unlink()
            try:
                _run(rex, exp, rows, ["v1"])
            except fpm.ThieuDauVaoDanhGia as e:
                loi_ini = str(e)
            _check("xoa `seqinfo.ini` -> cung BAO LOI (do dai timeline la dau vao bat buoc)",
                   "seqinfo.ini" in loi_ini, loi_ini[:160] or "(KHONG bao loi!)")
            (proc / split / "v1" / "seqinfo.ini").write_text(_seqinfo(1), encoding="utf-8")

            # --- 5. GT RONG HOP LE khac GT THIEU ---
            g = proc / split / "v1" / "gt" / "gt.txt"
            g.write_text("", encoding="utf-8")
            fp_rong = fpm.gt_fingerprint(split, ["v1"])
            g.unlink()
            fp_thieu = fpm.gt_fingerprint(split, ["v1"])
            g.write_text(GT_CROWD, encoding="utf-8")
            _check("GT RONG HOP LE khac han GT THIEU (khong bi lan thanh mot)",
                   fp_rong["gt_sha256"] != fp_thieu["gt_sha256"]
                   and fp_rong["n_missing_gt"] == 0 and fp_thieu["n_missing_gt"] == 1,
                   f"rong: n_missing={fp_rong['n_missing_gt']}  thieu: n_missing={fp_thieu['n_missing_gt']}")

            # --- 6. nguon annotation che khuat nam trong dau van tay ---
            old_i = config.INTERIM_DIR
            try:
                config.INTERIM_DIR = tmp / "interim"
                config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
                occ = {"tag": "fx", "n_boot": 10}
                for n in ("occlusion_segments.csv", "full_occlusion_segments.csv"):
                    (config.INTERIM_DIR / f"fx_{n}").write_text("video,track,f0,f1\n", encoding="utf-8")
                a = fpm.occl_fingerprint(exp, "c25", [tracker], ["v1"], occ)
                (config.INTERIM_DIR / "fx_full_occlusion_segments.csv").write_text(
                    "video,track,f0,f1\nv1,1,2,3\n", encoding="utf-8")
                b = fpm.occl_fingerprint(exp, "c25", [tracker], ["v1"], occ)
                _check("doi NGUON ANNOTATION che khuat -> `occl_fingerprint` doi",
                       a["annotation_source"] != b["annotation_source"] and fpm.compare(a, b) != [],
                       f"truong lech: {fpm.compare(a, b)}")
                (proc / split / "v1" / "gt" / "gt.txt").write_text(GT_HOPLE, encoding="utf-8")
                c = fpm.occl_fingerprint(exp, "c25", [tracker], ["v1"], occ)
                _check("doi GT -> `occl_fingerprint` cung doi (occlusion_eval doc GT cua split)",
                       "gt" in fpm.compare(b, c), f"truong lech: {fpm.compare(b, c)}")
            finally:
                config.INTERIM_DIR = old_i
        finally:
            config.PROCESSED_DIR, config.RESULTS_DIR = old_p, old_r

        # --- 7. run_state.py nam trong CODE_FILES ---
        _check("`run_state.py` nam trong CODE_FILES (chinh sach resume dung chung CLI + runner)",
               "run_state.py" in fpm.CODE_FILES, f"CODE_FILES co {len(fpm.CODE_FILES)} module")
        _check("quy uoc hash duoc cong bo trong module (`HASH_CONVENTION`)",
               "LF" in fpm.HASH_CONVENTION and "MISSING" in fpm.HASH_CONVENTION, fpm.HASH_CONVENTION)
    except AssertionError:
        failed += 1
    except Exception as e:  # noqa: BLE001
        failed += 1
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] loi khong mong doi: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
