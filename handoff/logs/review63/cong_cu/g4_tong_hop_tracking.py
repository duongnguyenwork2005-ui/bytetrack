"""R63 G4: so cap tracking ung vien rect640 (384x640) vs doi chung vuong640 (640x640) bang boot_stats.paired_bootstrap
(COMBINED_SEQ, bootstrap ghep cap theo video, 5.000 lan, seed 0) va ap cong Q2 (c25; c10 = do nhay, khong cong):
  dHOTA >= -0,5 diem, dIDF1 >= -0,5 diem (thang 100), IDSW tang <= 5 % (doi chung = 0 -> ung vien phai = 0).

    python handoff/logs/review63/cong_cu/g4_tong_hop_tracking.py --mien dev12|bdddev --ra <json>
"""
import argparse
import json
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GOC / "src"))
import boot_stats  # noqa: E402

MIEN = {"dev12": ("DETRAC-all", "r63g4"), "bdddev": ("BDDDEV-train", "r63g4b"),
        # bao cao cuoi (g4_bao_cao_cuoi.sh): chi bao cao sau khi chot C1, doc cong o day khong doi quyet dinh
        "test40": ("DETRAC-test", "r63bc"), "bddval": ("BDDCOCOVID-val", "r63bcb")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mien", choices=sorted(MIEN), required=True)
    ap.add_argument("--ra", required=True)
    a = ap.parse_args()
    if Path(a.ra).exists():
        print("[LOI] da ton tai", a.ra); return 2
    split, tien_to = MIEN[a.mien]
    goc = GOC / "results" / "trackeval" / split
    out = {"mien": a.mien, "split": split, "phuong_phap": "boot_stats.paired_bootstrap(ung_vien, doi_chung), 5000, seed 0",
           "cong_q2": "c25: dHOTA>=-0.5, dIDF1>=-0.5 (diem), IDSW tang<=5% (doi chung 0 -> 0)", "ket_qua": {}}
    for cond in ("c25", "c10"):
        for loc in ("cv", "ekf"):
            ra_ = boot_stats.load_raw_fields(goc / f"{tien_to}-rect640-{cond}-{loc}" / "raw_fields.json")
            rb_ = boot_stats.load_raw_fields(goc / f"{tien_to}-vuong640-{cond}-{loc}" / "raw_fields.json")
            if set(ra_) != set(rb_):
                print("[LOI] tap video khac nhau", cond, loc); return 2
            vids = sorted(ra_)
            r = boot_stats.paired_bootstrap(ra_, rb_, vids)
            idsw_a, idsw_b = r["IDSW"]["a"], r["IDSW"]["b"]
            tang_idsw = (idsw_a - idsw_b) / idsw_b if idsw_b else (0.0 if idsw_a == 0 else float("inf"))
            cong = {"HOTA": 100 * r["HOTA"]["diff"] >= -0.5, "IDF1": 100 * r["IDF1"]["diff"] >= -0.5,
                    "IDSW": (idsw_a == 0) if idsw_b == 0 else tang_idsw <= 0.05}
            out["ket_qua"][f"{cond}-{loc}"] = {"n_video": len(vids), "so_cap": r, "idsw_tang_tuong_doi": tang_idsw,
                                               "cong": cong if cond == "c25" else "do nhay (khong ap cong)",
                                               "dat": all(cong.values()) if cond == "c25" else None}
            print("%s %s-%s: HOTA %.4f->%.4f (%+.2f, CI %+.2f..%+.2f) IDF1 %.4f->%.4f (%+.2f, CI %+.2f..%+.2f) "
                  "IDSW %d->%d (%+.1f%%) %s" % (
                      a.mien, cond, loc, r["HOTA"]["b"], r["HOTA"]["a"], 100 * r["HOTA"]["diff"],
                      100 * r["HOTA"]["ci_low"], 100 * r["HOTA"]["ci_high"], r["IDF1"]["b"], r["IDF1"]["a"],
                      100 * r["IDF1"]["diff"], 100 * r["IDF1"]["ci_low"], 100 * r["IDF1"]["ci_high"],
                      idsw_b, idsw_a, 100 * tang_idsw, ("DAT" if all(cong.values()) else "TRUOT") if cond == "c25" else "(do nhay)"))
    Path(a.ra).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
