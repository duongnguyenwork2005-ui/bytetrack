"""
test_boot_stats.py -- (MOC 2) Tai to hop thong ke du per-sequence (boot_stats.combine) phai TRUNG
COMBINED_SEQ cua TrackEval (1e-9) tren du lieu that (tracker dev da co); bootstrap ghep cap on dinh
theo seed; trung binh seed (PF) tuyen tinh.

    python src/test_boot_stats.py            # can data/processed/trackers/DETRAC-all/p2-groupA_detrac_dev-A-{cv,ekf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import boot_stats  # noqa: E402
import run_trackeval  # noqa: E402

_RESULTS = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")
    assert ok, name


def main() -> int:
    import tempfile
    videos = ["MVI_20052", "MVI_20062", "MVI_39851", "MVI_40213"]
    raws = {}
    for tr in ("p2-groupA_detrac_dev-A-cv", "p2-groupA_detrac_dev-A-ekf"):
        res, _ = run_trackeval.run("DETRAC-all", tr, videos, "mot")
        d = next(iter(res.values()))[tr]["COMBINED_SEQ"]["pedestrian"]
        ref = {"HOTA": float(np.mean(d["HOTA"]["HOTA"])), "DetA": float(np.mean(d["HOTA"]["DetA"])),
               "AssA": float(np.mean(d["HOTA"]["AssA"])), "IDF1": float(d["Identity"]["IDF1"]),
               "MOTA": float(d["CLEAR"]["MOTA"]), "IDSW": float(d["CLEAR"]["IDSW"]),
               "FP": float(d["CLEAR"]["CLR_FP"]), "FN": float(d["CLEAR"]["CLR_FN"])}
        with tempfile.TemporaryDirectory() as td:
            raw = boot_stats.save_raw_fields(res, tr, Path(td) / "raw.json")
            raw2 = boot_stats.load_raw_fields(Path(td) / "raw.json")
        comb = boot_stats.combine(raw2, videos)
        ok = all(abs(comb[m] - ref[m]) < 1e-9 for m in ref)
        _check(f"[{tr[-3:]}] combine(raw_fields) == COMBINED_SEQ TrackEval (8 chi so, 1e-9)", ok,
               "  ".join(f"{m}: {comb[m]:.6f}/{ref[m]:.6f}" for m in ref))
        raws[tr] = raw2
    a, b = raws["p2-groupA_detrac_dev-A-ekf"], raws["p2-groupA_detrac_dev-A-cv"]
    r1 = boot_stats.paired_bootstrap(a, b, videos, n_boot=500, seed=0)
    r2 = boot_stats.paired_bootstrap(a, b, videos, n_boot=500, seed=0)
    _check("bootstrap xac dinh theo seed (2 lan cung ket qua)",
           all(r1[m] == r2[m] for m in r1))
    _check("diff = COMBINED(a) - COMBINED(b) va CI chua diff",
           all(abs(r1[m]["diff"] - (boot_stats.combine(a, videos)[m] - boot_stats.combine(b, videos)[m])) < 1e-12
               and r1[m]["ci_low"] <= r1[m]["diff"] <= r1[m]["ci_high"] for m in r1))
    # mau bootstrap toan bo = chinh no khi chi 1 video (CI suy bien = diff)
    r3 = boot_stats.paired_bootstrap(a, b, videos[:1], n_boot=50, seed=0)
    _check("1 video: CI suy bien = diff", all(abs(r3[m]["ci_low"] - r3[m]["diff"]) < 1e-12 for m in r3))
    # trung binh seed tuyen tinh: average_seeds([x, x]) == x
    avg = boot_stats.average_seeds([a, a])
    _check("average_seeds([x, x]) == x", all(np.allclose(avg[v][f], a[v][f]) for v in a for f in boot_stats.ARRAY_FIELDS))
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 0 if n_ok == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
