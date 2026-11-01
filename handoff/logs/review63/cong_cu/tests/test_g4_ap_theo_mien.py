"""Regression cho g4_ap_theo_mien.py ban 2 (khong GPU, khong dataset that ngoai manifest).
    python handoff/logs/review63/cong_cu/tests/test_g4_ap_theo_mien.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import g4_ap_theo_mien as g  # noqa: E402

KQ = []


def kiem(ten, ok):
    KQ.append(bool(ok)); print("  [%s] %s" % ("PASS" if ok else "FAIL", ten))


def main():
    a2v = g.anh_sang_video()
    kiem("manifest: anh UA-DETRAC 'MVI_20052_img00010.jpg' -> video MVI_20052 (loi ban 1: rsplit('-') giu nguyen ten)",
         a2v.get("MVI_20052_img00010.jpg") == ("MVI_20052", "detrac"))
    kiem("manifest: anh BDD '0000f77c-cb820c98-0000001.jpg' -> video 0000f77c-cb820c98",
         a2v.get("0000f77c-cb820c98-0000001.jpg") == ("0000f77c-cb820c98", "bdd"))
    det = sorted({v for v, n in a2v.values() if n == "detrac"})
    bdd = {v for v, n in a2v.values() if n == "bdd"}
    kiem("dev manifest: 12 video UA-DETRAC, 139 video BDD", len(det) == 12 and len(bdd) == 139)

    # noi theo khoi video == noi theo anh (cung thu tu mau, co lap lai)
    rng = np.random.default_rng(1)
    T = len(g.ev.NGUONG_IOU)
    gh = []
    for _ in range(9):
        hang = []
        for c in range(len(g.ev.LOP_5)):
            n = int(rng.integers(0, 5))
            hang.append((np.sort(rng.random(n))[::-1], rng.random((T, n)) < 0.6, rng.random((T, n)) < 0.1,
                         int(rng.integers(0, 4))))
        gh.append(hang)
    theo_v = {"v0": [0, 1, 2], "v1": [3, 4], "v2": [5, 6, 7, 8]}
    kh = g.khoi_video(gh, theo_v)
    ok = True
    for mau in (["v0", "v1", "v2"], ["v2", "v2", "v0"], ["v1", "v0", "v1"]):
        sel = [i for v in mau for i in theo_v[v]]
        a50, a95 = [], []
        for c in range(len(g.ev.LOP_5)):
            n_gt = sum(gh[i][c][3] for i in sel)
            if n_gt == 0:
                continue
            conf = np.concatenate([gh[i][c][0] for i in sel])
            tp = np.concatenate([gh[i][c][1] for i in sel], axis=1)
            bq = np.concatenate([gh[i][c][2] for i in sel], axis=1)
            aps = [g.ev.ap_coco(conf, tp[t], bq[t], n_gt) for t in range(T)]
            a50.append(aps[0]); a95.append(float(np.mean(aps)))
        ok &= np.allclose(g.map_tu_khoi(kh, mau), (np.mean(a50), np.mean(a95)), atol=0, rtol=0)
    kiem("map_tu_khoi (noi khoi video) trung tung bit voi noi theo anh, 3 mau co lap lai", ok)
    print("KET QUA: %d/%d PASS" % (sum(KQ), len(KQ)))
    return 0 if all(KQ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
