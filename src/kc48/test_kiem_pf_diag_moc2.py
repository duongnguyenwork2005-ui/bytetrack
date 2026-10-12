"""
test_kiem_pf_diag_moc2.py -- Test cho src/kc48/kiem_pf_diag_moc2.py tren API that cua src/pf_ctrv.py.

    1. Dong chan doan do PFCTRV.update_particles ghi o nhanh suy bien co dung quy uoc ma kiem_pf_diag_moc2 dua vao
       (kind update, ess_ratio 0, resampled True, n_unique N, degenerate True, maha2_best > nguong); dong hop le
       co ESS/N >= 1/N va maha2_best <= nguong.
    2. Cong thuc cu (moi dong update) TRON tai khoi tao: ESS/N trung binh thap hon, ti le resample cao hon cong thuc
       tach; cong thuc tach dung bang thong ke tinh tay tren dong hop le.
    3. Khong co tai khoi tao -> hai cong thuc trung khit (dieu kien de so moc 2 cu khong doi).
    4. Bin n_lost trung pd.cut([-1, 0, 2, 5, 10, 30, 10000]) (dong ben phai) o cac bien.
    5. Phan nhom dev/test (sua F1): regex va thu muc split tach dung, run test khong lot vao dev.

    python src/kc48/test_kiem_pf_diag_moc2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pf_ctrv import PFCTRV  # noqa: E402
import kiem_pf_diag_moc2 as K  # noqa: E402

COLS = ["track_id", "frame", "kind", "ess_ratio", "resampled", "n_unique", "spread_xy", "spread_theta",
        "spread_v", "degenerate", "maha2_best", "n_lost"]
_KQ: list[bool] = []


def _check(name, ok, detail=""):
    _KQ.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def chuoi_diag(n_nhay: int, N: int = 100, T: int = 30, ratio: float = 0.5):
    """Mot track: T update bam quanh duong thang; n_nhay lan quan sat nhay xa 1e4 px (ep nhanh suy bien)."""
    pf = PFCTRV(num_particles=N, degenerate_maha2=1e4, resample_ess_ratio=ratio)
    pf.enable_diag()
    rng = pf.make_rng(7)
    z = np.array([300.0, 200.0, 0.5, 40.0])
    m, P = pf.initiate(z)
    X = pf.sample_initial(m, P, rng)
    logw = np.full(N, -np.log(N))
    nhay = set(range(5, 5 + 3 * n_nhay, 3))
    off = 0.0
    for t in range(1, T + 1):
        X = pf.propagate(X, rng)
        if t in nhay:
            off += 10_000.0
        zt = z + np.array([4.0 * t + off, 0.0, 0.0, 0.0])
        X, logw, _, _ = pf.update_particles(X, logw, zt, 0.9, rng, diag_key=(1, t, 0))
    rows = [{c: str(v) for c, v in zip(COLS, r)} for r in pf.diag_rows]
    return pf, K.tu_dong(rows)


def test_1_quy_uoc():
    N = 100
    pf, a = chuoi_diag(3, N)
    deg = a["deg"]
    _check("bo dem n_degenerate == so dong degenerate", pf.n_degenerate == int(deg.sum()) == 3,
           f"n_degenerate {pf.n_degenerate}, dong {int(deg.sum())}")
    _check("dong suy bien: kind update, ess 0, resampled True, n_unique N, maha2 > 1e4",
           bool(np.all(a["kind"][deg] == "update") and np.all(a["ess"][deg] == 0.0) and np.all(a["rs"][deg])
                and np.all(a["nu"][deg] == N) and np.all(a["maha"][deg] > 1e4)))
    hop = ~deg
    _check("dong hop le: ESS/N >= 1/N, maha2 <= 1e4",
           bool(np.all(a["ess"][hop] >= 1.0 / N) and np.all(a["maha"][hop] <= 1e4)),
           f"min ESS/N {a['ess'][hop].min():.4f}")


def test_2_cu_tron_moi_tach():
    # ratio 0: update hop le khong bao gio resample -> moi "resampled" trong cong thuc cu deu den tu tai khoi tao
    N = 100
    _, a = chuoi_diag(3, N, ratio=0.0)
    upd = a["kind"] == "update"
    hop = upd & ~a["deg"]
    cu = K.tk(a["ess"][upd], a["rs"][upd], a["nu"][upd], N)
    moi = K.tk(a["ess"][hop], a["rs"][hop], a["nu"][hop], N)
    e = a["ess"][hop]
    _check("ratio 0: tach -> ti le resample 0; cu -> 3/30", moi["frac_resampled"] == 0.0
           and np.isclose(cu["frac_resampled"], 3 / 30), f"cu {cu['frac_resampled']:.4f}")
    _check("cong thuc tach == tinh tay tren dong hop le",
           np.isclose(moi["ess_ratio_mean"], e.sum() / len(e)) and
           np.isclose(moi["frac_resampled"], a["rs"][hop].sum() / len(e)))
    _check("cong thuc cu tron: ESS/N thap hon, resample cao hon",
           cu["ess_ratio_mean"] < moi["ess_ratio_mean"] and cu["frac_resampled"] > moi["frac_resampled"],
           f"ESS/N cu {cu['ess_ratio_mean']:.4f} vs tach {moi['ess_ratio_mean']:.4f}; "
           f"resample cu {cu['frac_resampled']:.4f} vs tach {moi['frac_resampled']:.4f}")


def test_3_khong_suy_bien_trung_khit():
    N = 100
    pf, a = chuoi_diag(0, N)
    upd = a["kind"] == "update"
    hop = upd & ~a["deg"]
    cu = K.tk(a["ess"][upd], a["rs"][upd], a["nu"][upd], N)
    moi = K.tk(a["ess"][hop], a["rs"][hop], a["nu"][hop], N)
    _check("0 tai khoi tao -> cong thuc cu == tach (moi chi so)",
           pf.n_degenerate == 0 and all(cu[k] == moi[k] for k in cu))


def test_4_bin():
    nl = np.array([0, 1, 2, 3, 5, 6, 10, 11, 30, 31, 500])
    nhan = [next(lab for lab in K.BIN_LABELS if K.mat_na_bin(np.array([v]), lab)[0]) for v in nl]
    ky_vong = ["0", "1-2", "1-2", "3-5", "3-5", "6-10", "6-10", "11-30", "11-30", ">30", ">30"]
    _check("bin n_lost dong ben phai nhu pd.cut", nhan == ky_vong, str(nhan))


def test_5_phan_nhom_dev_test():
    # sua F1: run test phai vao nhom test, khong lot vao dev (va nguoc lai); ten khac khong khop nhom nao
    dev, test = "p2-expA_detrac_dev-c10-pf300-s1", "p2-expA_detrac_test-c25-pf500-s2"
    _check("regex dev/test tach dung",
           bool(K.RE_RUN.match(dev)) and not K.RE_RUN.match(test) and bool(K.RE_RUN_TEST.match(test))
           and not K.RE_RUN_TEST.match(dev) and not K.RE_RUN.match("p2-expA_detrac_test-c25-ekf")
           and K.RE_RUN_TEST.match(test).groups() == ("c25", "pf500", "500", "2"))
    _check("nhom dev/test: thu muc split va hau to file khac nhau",
           [(n[0], n[1].name, n[5]) for n in K.NHOM] == [("dev", "DETRAC-all", ""), ("test", "DETRAC-test", "_test")])


def main() -> int:
    for t in (test_1_quy_uoc, test_2_cu_tron_moi_tach, test_3_khong_suy_bien_trung_khit, test_4_bin,
              test_5_phan_nhom_dev_test):
        print(f"--- {t.__name__} ---")
        t()
    print(f"KET QUA: {sum(_KQ)}/{len(_KQ)} PASS")
    return 0 if all(_KQ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
