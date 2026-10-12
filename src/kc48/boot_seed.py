"""
boot_seed.py -- R48D vong sua (Codex review_003 F3): bootstrap ghep cap theo video, PF nhieu seed dung thu tu cua
giao thuc 04 muc 5.2: trong MOI mau bootstrap, tinh COMBINED rieng tung seed (src/boot_stats.combine_stacked, cong
thuc TrackEval) roi lay TRUNG BINH cac seed. Khong sua src/boot_stats.py.

Khac ban cu (phan_tich_dev truoc vong sua): ban cu trung binh truong tho giua seed (boot_stats.average_seeds) roi
moi tinh COMBINED -- HOTA / IDF1 / MOTA / DetA / AssA la ham phi tuyen cua truong tho, nen hai thu tu cho so khac nhau.
IDSW, FP, FN la tong tuyen tinh -> hai thu tu trung nhau.

Dung cung day so ngau nhien voi boot_stats.paired_bootstrap (default_rng(seed), integers(0, n, size=n) moi mau):
mot seed moi ben -> trung khit boot_stats.paired_bootstrap (test).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import boot_stats as bs  # noqa: E402


def _tb(cs: list[dict], metrics) -> dict:
    return {m: float(np.mean([c[m] for c in cs])) for m in metrics}


def combine_tb_seed(raws: list[dict], videos: list[str], metrics=bs.METRICS) -> dict:
    """Trung binh giua seed cua chi so COMBINED tung seed."""
    return _tb([bs.combine(r, videos) for r in raws], metrics)


def paired_bootstrap_seed(raws_a: list[dict], raws_b: list[dict], videos: list[str], n_boot: int = 5000,
                          seed: int = 0, metrics=bs.METRICS, tra_mau: bool = False) -> dict:
    """mean_s COMBINED(a_s) - mean_s COMBINED(b_s) + CI 95 % bootstrap phan vi theo video (cung mau video cho moi seed
    cua ca hai ben)."""
    videos = list(videos)
    sa = [bs._stack(r, videos) for r in raws_a]
    sb = [bs._stack(r, videos) for r in raws_b]
    obs_a = _tb([bs.combine_stacked(s) for s in sa], metrics)
    obs_b = _tb([bs.combine_stacked(s) for s in sb], metrics)
    rng = np.random.default_rng(seed)
    n = len(videos)
    boots = {m: np.empty(n_boot) for m in metrics}
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ca = _tb([bs.combine_stacked({f: s[f][idx] for f in s}) for s in sa], metrics)
        cb = _tb([bs.combine_stacked({f: s[f][idx] for f in s}) for s in sb], metrics)
        for m in metrics:
            boots[m][i] = ca[m] - cb[m]
    out = {}
    for m in metrics:
        lo, hi = np.percentile(boots[m], [2.5, 97.5])
        out[m] = {"a": obs_a[m], "b": obs_b[m], "diff": obs_a[m] - obs_b[m], "ci_low": float(lo), "ci_high": float(hi)}
    if tra_mau:
        out["_mau"] = boots
    return out
