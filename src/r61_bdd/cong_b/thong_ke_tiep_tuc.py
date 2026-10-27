"""
thong_ke_tiep_tuc.py -- Bootstrap ghep cap R61 chia KHOI CHI SO xac dinh de tiep tuc / tai tinh hau kiem theo don vi
(plan chuan bi B0). KHONG doi cong thuc: moi mau dung dung thong_ke_bdd.xep / metric_bo_loc (COMBINED tung run; PF
trung binh metric COMBINED 3 seed), hieu so * HE_SO, cung ma tran chi so default_rng(0) (so_mau x so_video) dung chung.

  - Khoi k = hang [i0, i1) cua ma tran chi so da LUU; mau cua khoi luu nguyen (float64, shape (G, 6 cap, 3 metric, i1-i0)).
  - CI: ghep MOI khoi theo thu tu hang -> mang day du so_mau; np.percentile([2.5, 97.5], method="linear") tren TOAN mau.
    Khong gop CI tung khoi. Thu tu dong / gia tri trung thong_ke_bdd.bang_ci (fixture doi chieu byte CSV).
Chia khoi chi phuc vu van hanh (tiep tuc sau exit 10 / crash, han thoi gian); khong phai thay doi thong ke.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import thong_ke_bdd as T  # noqa: E402


def khoi(so_mau: int, kt: int) -> list[list[int]]:
    if kt < 1:
        raise ValueError("kich thuoc khoi phai >= 1")
    return [[i, min(i + kt, so_mau)] for i in range(0, so_mau, kt)]


def nhom_raws(runs: list[dict], raw: dict) -> dict:
    """{(detector, dieu_kien): {bo_loc: [raw ...]}} theo DUNG thu tu dieu_phoi_bdd.tinh_ci (thu tu ctx.runs())."""
    nhom = {}
    for r in runs:
        nhom.setdefault((r["detector"], r["dieu_kien"]), {}).setdefault(r["bo_loc"], []).append(raw[r["run"]])
    return nhom


def chuan_bi(nhom: dict, videos: list[str]) -> tuple[dict, dict]:
    """Thong ke da xep (T.xep) + gia tri quan sat (T.metric_bo_loc, idx None) moi nhom."""
    st = {g: {k: [T.xep(r, videos) for r in rs] for k, rs in raws.items()} for g, raws in nhom.items()}
    obs = {g: {k: T.metric_bo_loc(s) for k, s in sg.items()} for g, sg in st.items()}
    return st, obs


def mau_khoi(st: dict, chi_so: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """Mau bootstrap hang [i0, i1): out[g, c, m, i - i0] = (metric a - metric b) * HE_SO (giong T.bootstrap_nhom)."""
    if chi_so.ndim != 2 or not 0 <= i0 < i1 <= chi_so.shape[0]:
        raise ValueError(f"khoi [{i0}, {i1}) ngoai chi so {chi_so.shape}")
    out = np.empty((len(st), len(T.CAP), len(T.METRIC), i1 - i0))
    for gi, sg in enumerate(st.values()):
        for i in range(i0, i1):
            idx = chi_so[i]
            g = {k: T.metric_bo_loc(s, idx) for k, s in sg.items()}
            for ci, (a, b) in enumerate(T.CAP):
                for mi, m in enumerate(T.METRIC):
                    out[gi, ci, mi, i - i0] = (g[a][m] - g[b][m]) * T.HE_SO[m]
    return out


def npy_bytes(a: np.ndarray) -> bytes:
    b = io.BytesIO()
    np.save(b, a)
    return b.getvalue()


def doc_npy(p: Path) -> np.ndarray:
    return np.load(p, allow_pickle=False)


def ghep(cac_khoi: list[np.ndarray], so_mau: int) -> np.ndarray:
    m = np.concatenate(cac_khoi, axis=3)
    if m.shape[3] != so_mau:
        raise ValueError(f"ghep khoi {m.shape[3]} mau != {so_mau}")
    return m


def hang_ci(nhom: dict, obs: dict, mau: np.ndarray) -> list[dict]:
    """Dong CI giong T.bang_ci: phan vi tren TOAN mau ghep (khong gop CI tung khoi)."""
    rows = []
    for gi, (det, dk) in enumerate(nhom):
        o = obs[(det, dk)]
        for ci, (a, b) in enumerate(T.CAP):
            for mi, m in enumerate(T.METRIC):
                lo, hi = np.percentile(mau[gi, ci, mi], [2.5, 97.5], method="linear")
                diff = (o[a][m] - o[b][m]) * T.HE_SO[m]
                rows.append({"detector": det, "dieu_kien": dk, "cap": f"{a}-{b}", "metric": m, "don_vi": T.DON_VI[m],
                             "diff": diff, "ci_low": float(lo), "ci_high": float(hi),
                             "ci_chua_0": bool(float(lo) <= 0.0 <= float(hi)),
                             "dau": "+" if diff > 0 else ("-" if diff < 0 else "0")})
    return rows


def doi_chieu_nguyen_khoi(nhom: dict, videos: list[str], chi_so: np.ndarray, mau: np.ndarray, rows: list[dict]) -> dict:
    """FIXTURE / kiem chung: tinh NGUYEN KHOI bang T.bang_ci + T.bootstrap_nhom tren cung du lieu; so dong, thu tu, mau."""
    ref_rows = T.bang_ci(nhom, videos, chi_so)
    lech_mau = []
    for gi, (key, raws) in enumerate(nhom.items()):
        r = T.bootstrap_nhom(raws, videos, chi_so)
        for ci, cap in enumerate(T.CAP):
            for mi, m in enumerate(T.METRIC):
                if not np.array_equal(r[cap][m]["mau"], mau[gi, ci, mi]):
                    lech_mau.append(f"{key}/{cap}/{m}")
    return {"so_dong": [len(rows), len(ref_rows)], "dong_trung": rows == ref_rows, "mau_lech": lech_mau}
