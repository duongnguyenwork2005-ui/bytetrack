"""
thong_ke_bdd.py -- Tong hop COMBINED / PF500 va bootstrap ghep cap theo video cho R61 (giao_thuc.json muc thong_ke).

Chi goi helper hien co: boot_stats._stack / combine_stacked / combine (cong thuc TrackEval). Khong doi cong thuc.

  - Bang chinh: COMBINED_SEQ cua evaluator (doc combined_metrics.csv); PF500 = trung binh metric COMBINED 3 seed.
  - Bootstrap: MOT ma tran chi so (so_mau x so_video) sinh bang default_rng(0), moi mau rng.integers(0, n, size=n)
    (cung day voi boot_stats.paired_bootstrap), dung chung cho MOI run va MOI seed. Moi mau: combine_stacked tung run;
    PF: COMBINED tung seed roi trung binh metric; sau do hieu so. Phan vi 2,5/97,5 method=linear.
  - Cam: trung binh metric video, gop thong ke du giua seed, bootstrap seed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import boot_stats as bs  # noqa: E402

METRIC = ("HOTA", "IDF1", "IDSW")
HE_SO = {"HOTA": 100.0, "IDF1": 100.0, "IDSW": 1.0}
DON_VI = {"HOTA": "pp", "IDF1": "pp", "IDSW": "count"}
CAP = (("ekf", "cv"), ("ukf", "cv"), ("ukf", "ekf"), ("pf500", "cv"), ("pf500", "ekf"), ("pf500", "ukf"))


def tao_chi_so(n_video: int, so_mau: int = 5000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([rng.integers(0, n_video, size=n_video) for _ in range(so_mau)]).astype(np.int64)


def xep(raw: dict, videos: list[str]) -> dict:
    thieu = [v for v in videos if v not in raw]
    if thieu or len(raw) != len(videos):
        raise ValueError(f"raw_fields thieu/thua video: thieu {thieu[:5]}, so {len(raw)} != {len(videos)}")
    return bs._stack(raw, videos)


def combined(st: dict, idx: np.ndarray | None = None) -> dict:
    return bs.combine_stacked(st if idx is None else {f: a[idx] for f, a in st.items()})


def metric_bo_loc(st_seeds: list[dict], idx: np.ndarray | None = None) -> dict:
    """Mot bo loc: list thong ke da xep cua cac seed (1 phan tu voi cv/ekf/ukf). Trung binh METRIC COMBINED."""
    cs = [combined(s, idx) for s in st_seeds]
    return {m: float(np.mean([c[m] for c in cs])) for m in METRIC}


def bootstrap_nhom(raws: dict[str, list[dict]], videos: list[str], chi_so: np.ndarray) -> dict:
    """raws: {cv: [raw], ekf: [raw], ukf: [raw], pf500: [raw_s0, raw_s1, raw_s2]} cua MOT nhom detector x conf.

    Tra ve {(a, b): {metric: {diff, ci_low, ci_high, mau (so_mau,)}}} tren thang da nhan he so (pp/count).
    """
    if chi_so.shape[1] != len(videos):
        raise ValueError("chi_so khong khop so video")
    st = {k: [xep(r, videos) for r in rs] for k, rs in raws.items()}
    obs = {k: metric_bo_loc(s) for k, s in st.items()}
    mau = {(a, b): {m: np.empty(len(chi_so)) for m in METRIC} for a, b in CAP}
    for i, idx in enumerate(chi_so):
        g = {k: metric_bo_loc(s, idx) for k, s in st.items()}
        for a, b in CAP:
            for m in METRIC:
                mau[(a, b)][m][i] = (g[a][m] - g[b][m]) * HE_SO[m]
    out = {}
    for a, b in CAP:
        out[(a, b)] = {}
        for m in METRIC:
            lo, hi = np.percentile(mau[(a, b)][m], [2.5, 97.5], method="linear")
            out[(a, b)][m] = {"a": obs[a][m], "b": obs[b][m], "diff": (obs[a][m] - obs[b][m]) * HE_SO[m],
                              "ci_low": float(lo), "ci_high": float(hi), "don_vi": DON_VI[m], "mau": mau[(a, b)][m]}
    return out


def bang_ci(nhom: dict[tuple[str, str], dict[str, list[dict]]], videos: list[str], chi_so: np.ndarray) -> list[dict]:
    """nhom: {(detector, dk): raws}. Tra ve list dong CI (6 cap x 3 metric moi nhom)."""
    rows = []
    for (det, dk), raws in nhom.items():
        r = bootstrap_nhom(raws, videos, chi_so)
        for (a, b), mm in r.items():
            for m in METRIC:
                x = mm[m]
                rows.append({"detector": det, "dieu_kien": dk, "cap": f"{a}-{b}", "metric": m, "don_vi": x["don_vi"],
                             "diff": x["diff"], "ci_low": x["ci_low"], "ci_high": x["ci_high"],
                             "ci_chua_0": bool(x["ci_low"] <= 0.0 <= x["ci_high"]),
                             "dau": "+" if x["diff"] > 0 else ("-" if x["diff"] < 0 else "0")})
    return rows


def bang_chinh(combined_evaluator: dict[str, dict]) -> list[dict]:
    """combined_evaluator: {run: {metric: gia tri COMBINED_SEQ}}; PF500 = trung binh 3 seed + min/max mo ta."""
    rows, nhom = [], {}
    for run, c in combined_evaluator.items():
        p = run.split("-")                      # r61-<det>-<dk>-<bo_loc>[-s<seed>]
        det, dk, bl = p[1], p[2], p[3]
        nhom.setdefault((det, dk, bl), []).append((run, c))
    for (det, dk, bl), ds in sorted(nhom.items()):
        row = {"detector": det, "dieu_kien": dk, "bo_loc": bl, "so_seed": len(ds), "run": ";".join(r for r, _ in ds)}
        for m in METRIC + ("DetA", "AssA", "MOTA"):
            v = [float(c[m]) for _, c in ds]
            row[m] = float(np.mean(v))
            if len(ds) > 1:
                row[f"{m}_min"], row[f"{m}_max"] = min(v), max(v)
                for (r, _), x in zip(ds, v):
                    row[f"{m}_{r.rsplit('-', 1)[1]}"] = x
        rows.append(row)
    return rows


def sai_so_tai_to_hop(raw: dict, videos: list[str], combined_eval: dict) -> dict:
    """|boot_stats.combine(raw) - COMBINED_SEQ evaluator| cho HOTA/IDF1/IDSW."""
    c = bs.combine(raw, videos)
    return {m: abs(float(c[m]) - float(combined_eval[m])) for m in METRIC}


def tb_metric_video(per_video: dict[str, dict], m: str) -> float:
    """CHI dung trong fixture de chung minh KHAC voi COMBINED (cam dung trong bang chinh)."""
    return float(np.mean([r[m] for r in per_video.values()]))
