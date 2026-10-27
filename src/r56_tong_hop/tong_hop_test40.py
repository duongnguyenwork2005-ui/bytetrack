"""
tong_hop_test40.py -- CHI DOC: tong hop metric TrackEval cua 36 run tracking test40 (phien thuc thi R56, 21/10/2026).

Doc lan cham hoan tat (co HOAN_TAT.json, moi run dung 1 lan) trong
results/phase2/r55_tracking/thuc_thi/<phien>/cham/<run>/lan_*/trackeval/<run>/pedestrian_summary.txt
va ghi bang tong hop vao results/phase2/r56_tong_hop/<phien>/ (thu muc moi, tu choi ghi de):
  tong_hop_36_run.csv   moi run mot dong (detector, dieu kien, bo loc, seed, HOTA, DetA, AssA, MOTA, IDF1, IDSW, Frag, ...)
  pf500_trung_binh.csv  PF500: trung binh / min / max qua 3 seed theo (detector, dieu kien)
  nguon.json            duong dan + sha256 tung file summary da doc
Khong goi tracker/evaluator, khong sua file nao trong phien. So lieu chi de mo ta; khong bootstrap, khong kiem dinh.

    C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -B src/r56_tong_hop/tong_hop_test40.py --phien <ten phien>
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

GOC = Path(__file__).resolve().parents[2]
DET = ["yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5"]
COND = ["c25", "c10"]
LOC = ["cv", "ekf", "ukf", "pf500-s0", "pf500-s1", "pf500-s2"]
COT = ["HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "IDF1", "IDSW", "Frag", "CLR_FP", "CLR_FN", "IDs", "GT_IDs"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phien", required=True)
    a = ap.parse_args()
    if "/" in a.phien or "\\" in a.phien or ".." in a.phien:
        print("--phien chi la ten phien"); return 2
    cham = GOC / "results/phase2/r55_tracking/thuc_thi" / a.phien / "cham"
    out = GOC / "results/phase2/r56_tong_hop" / a.phien
    if out.exists():
        print(f"{out} da ton tai, khong ghi de"); return 1
    rows, nguon = [], {}
    for d in DET:
        for c in COND:
            for loc in LOC:
                run = f"r54-{d}-{c}-{loc}"
                lans = sorted(x for x in (cham / run).glob("lan_*") if (x / "HOAN_TAT.json").is_file())
                if len(lans) != 1:
                    print(f"{run}: {len(lans)} lan cham hoan tat (can dung 1)"); return 2
                s = lans[0] / "trackeval" / run / "pedestrian_summary.txt"
                h, v = s.read_text(encoding="utf-8").splitlines()[:2]
                m = dict(zip(h.split(), v.split()))
                bo_loc, seed = (loc.split("-s") + [""])[:2] if loc.startswith("pf500") else (loc, "")
                rows.append({"run": run, "detector": d, "dieu_kien": c, "bo_loc": bo_loc, "seed": seed,
                             **{k: m[k] for k in COT}})
                nguon[run] = {"file": s.relative_to(GOC).as_posix(), "sha256": sha(s)}
    out.mkdir(parents=True)
    with open(out / "tong_hop_36_run.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    pf = []
    for d in DET:
        for c in COND:
            g = [r for r in rows if r["detector"] == d and r["dieu_kien"] == c and r["bo_loc"] == "pf500"]
            rec = {"detector": d, "dieu_kien": c, "so_seed": len(g)}
            for k in ["HOTA", "AssA", "MOTA", "IDF1", "IDSW"]:
                x = [float(r[k]) for r in g]
                rec[f"{k}_tb"] = round(sum(x) / len(x), 4); rec[f"{k}_min"] = min(x); rec[f"{k}_max"] = max(x)
            pf.append(rec)
    with open(out / "pf500_trung_binh.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pf[0]))
        w.writeheader(); w.writerows(pf)
    (out / "nguon.json").write_text(json.dumps({"phien": a.phien, "so_run": len(rows), "summary": nguon},
                                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[tong-hop] {len(rows)} run -> {out.relative_to(GOC).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
