"""
boot_stats.py -- Bootstrap GHEP CAP THEO VIDEO tren THONG KE DU cua TrackEval, tai to hop DUNG CONG
THUC COMBINED_SEQ (Moc 2, muc 4) -> khoang tin cay cho chenh lech cua chi so trong BANG CHINH.

Vi sao: bang chinh (summary.csv) dung COMBINED_SEQ cua TrackEval = gop thong ke moi sequence roi tinh
metric (KHONG phai trung binh cua metric tung video). Bootstrap cu (aggregate_phase2.py) lay mau lai
video roi tinh TRUNG BINH metric theo video -> uoc luong mot dai luong KHAC bang chinh. O day, moi lan
lay mau lai (co hoan lai) tap video, ta CONG thong ke du cua cac video duoc chon va tinh lai metric y
het TrackEval:
    HOTA  (19 alpha): TP/FN/FP cong; AssA/AssRe/AssPr = trung binh trong so HOTA_TP;
                      DetA = TP/(TP+FN+FP); HOTA_alpha = sqrt(DetA*AssA); HOTA = mean_alpha
    IDF1            : IDTP/(IDTP + 0.5*IDFN + 0.5*IDFP)         (IDTP/IDFN/IDFP cong)
    MOTA            : (CLR_TP - CLR_FP - IDSW)/(CLR_TP + CLR_FN)
    IDSW, FP, FN    : cong
Ghep cap: cung mau video cho ca hai tracker (PF: thong ke du cua tung seed duoc TRUNG BINH theo video
truoc khi cong - tuong ung "trung binh giua seed cua chi so COMBINED" chi xap xi; ghi ro trong bao cao).

    raw_fields.json  : {video: {HOTA_TP: [19], HOTA_FN: [19], HOTA_FP: [19], AssA: [19], AssRe: [19],
                        AssPr: [19], IDTP, IDFN, IDFP, CLR_TP, CLR_FN, CLR_FP, IDSW}}
    save_raw_fields(results, tracker, path)   goi trong run_experiment.stage_eval
    combine(raw, videos) -> dict metric        == combined_metrics.csv (test_boot_stats.py, 1e-9)
    paired_bootstrap(raw_a, raw_b, videos, n_boot=5000, seed=0) -> {metric: (diff, lo, hi)}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARRAY_FIELDS = ["HOTA_TP", "HOTA_FN", "HOTA_FP", "AssA", "AssRe", "AssPr"]
SCALAR_FIELDS = ["IDTP", "IDFN", "IDFP", "CLR_TP", "CLR_FN", "CLR_FP", "IDSW"]
METRICS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN"]


def save_raw_fields(results: dict, tracker_name: str, path: str | Path) -> dict:
    d = next(iter(results.values()))[tracker_name]
    out = {}
    for seq, cls_dict in d.items():
        if seq == "COMBINED_SEQ":
            continue
        m = cls_dict.get("pedestrian")
        if m is None:
            continue
        rec = {f: [float(x) for x in np.asarray(m["HOTA"][f]).ravel()] for f in ARRAY_FIELDS}
        rec.update({"IDTP": int(m["Identity"]["IDTP"]), "IDFN": int(m["Identity"]["IDFN"]), "IDFP": int(m["Identity"]["IDFP"]),
                    "CLR_TP": int(m["CLEAR"]["CLR_TP"]), "CLR_FN": int(m["CLEAR"]["CLR_FN"]), "CLR_FP": int(m["CLEAR"]["CLR_FP"]),
                    "IDSW": int(m["CLEAR"]["IDSW"])})
        out[seq] = rec
    Path(path).write_text(json.dumps(out), encoding="utf-8")
    return out


def load_raw_fields(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _stack(raw: dict, videos: list[str]) -> dict:
    """Xep thong ke cua cac video (theo thu tu, co the lap) thanh mang: array (n, 19), scalar (n,)."""
    st = {f: np.array([raw[v][f] for v in videos], dtype=float) for f in ARRAY_FIELDS}
    for f in SCALAR_FIELDS:
        st[f] = np.array([raw[v][f] for v in videos], dtype=float)
    return st


def combine_stacked(st: dict) -> dict:
    """Tai to hop y het trackeval.metrics.{HOTA,CLEAR,Identity}.combine_sequences."""
    tp, fn, fp = st["HOTA_TP"].sum(0), st["HOTA_FN"].sum(0), st["HOTA_FP"].sum(0)
    w = st["HOTA_TP"]
    ass = {f: np.maximum(1e-10, (st[f] * w).sum(0)) / np.maximum(1e-10, w.sum(0)) for f in ("AssA", "AssRe", "AssPr")}
    det_a = tp / np.maximum(1, tp + fn + fp)
    hota = np.sqrt(det_a * ass["AssA"])
    idtp, idfn, idfp = st["IDTP"].sum(), st["IDFN"].sum(), st["IDFP"].sum()
    ctp, cfn, cfp, idsw = st["CLR_TP"].sum(), st["CLR_FN"].sum(), st["CLR_FP"].sum(), st["IDSW"].sum()
    return {"HOTA": float(hota.mean()), "DetA": float(det_a.mean()), "AssA": float(ass["AssA"].mean()),
            "IDF1": float(idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn)),
            "MOTA": float((ctp - cfp - idsw) / max(1.0, ctp + cfn)),
            "IDSW": float(idsw), "FP": float(cfp), "FN": float(cfn)}


def combine(raw: dict, videos: list[str] | None = None) -> dict:
    videos = list(videos) if videos is not None else sorted(raw)
    return combine_stacked(_stack(raw, videos))


def average_seeds(raws: list[dict]) -> dict:
    """PF: trung binh thong ke du cua cac seed TRONG TUNG VIDEO (tuyen tinh) -> mot 'raw' dai dien."""
    if len(raws) == 1:
        return raws[0]
    out = {}
    for v in raws[0]:
        rec = {}
        for f in ARRAY_FIELDS:
            rec[f] = list(np.mean([np.asarray(r[v][f], float) for r in raws], axis=0))
        for f in SCALAR_FIELDS:
            rec[f] = float(np.mean([r[v][f] for r in raws]))
        out[v] = rec
    return out


def paired_bootstrap(raw_a: dict, raw_b: dict, videos: list[str], n_boot: int = 5000, seed: int = 0,
                     metrics: list[str] = METRICS) -> dict:
    """Chenh lech COMBINED(a) - COMBINED(b) + CI 95 % bootstrap theo video (cung mau cho ca hai)."""
    videos = list(videos)
    sa, sb = _stack(raw_a, videos), _stack(raw_b, videos)
    obs_a, obs_b = combine_stacked(sa), combine_stacked(sb)
    rng = np.random.default_rng(seed)
    n = len(videos)
    boots = {m: np.empty(n_boot) for m in metrics}
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ca = combine_stacked({f: sa[f][idx] for f in sa})
        cb = combine_stacked({f: sb[f][idx] for f in sb})
        for m in metrics:
            boots[m][i] = ca[m] - cb[m]
    out = {}
    for m in metrics:
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        out[m] = {"a": obs_a[m], "b": obs_b[m], "diff": obs_a[m] - obs_b[m], "ci_low": float(lo), "ci_high": float(hi)}
    return out


if __name__ == "__main__":
    import argparse
    import pandas as pd
    ap = argparse.ArgumentParser(description="Bootstrap ghep cap theo video tren thong ke du (COMBINED)")
    ap.add_argument("a", help="raw_fields.json cua tracker A")
    ap.add_argument("b", help="raw_fields.json cua tracker B (baseline)")
    ap.add_argument("--n-boot", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    ns = ap.parse_args()
    ra, rb = load_raw_fields(ns.a), load_raw_fields(ns.b)
    vids = sorted(set(ra) & set(rb))
    r = paired_bootstrap(ra, rb, vids, ns.n_boot, ns.seed)
    print(pd.DataFrame(r).T.to_string(float_format=lambda v: f"{v:+.4f}"))
