"""
test_k1_k2.py -- R48D: test cho hai ky thuat PF cua giao thuc 04 muc 6 (K1 proposal toi uu - tuy chon MOI;
K2 roughening - tuy chon DA CO) va hook chan doan chi so resample (`PFCTRV.last_resample_idx`).

Ky vong lay tu nguon DOC LAP voi cai dat:
    1. Mac dinh (K1 tat) bit-identical voi ma GOC tai eaf2e87 (nap nguyen van bang `git show`, khong checkout):
       chuoi bo loc 60 buoc co resample / suy bien / mat do, va pipeline PFByteTracker 30 frame (particle, logw,
       tlwh moi frame moi track). Kem ca roughening 0.5 (duong resample_if_needed co hook moi).
    2. K1: trong so = p(z | x_{k-1}^i) tinh bang Gauss DA BIEN day du N(z; H f(x), H Q_i H^T + R) (ma tran 4x4,
       slogdet + solve) - khong qua cong thuc tung toa do cua ma.
    3. K1: mo hinh tuyen tinh mot buoc (v = 0, Q gan hang) -> trung binh / phuong sai co trong so khop hau nghiem
       Kalman tinh tay; ESS cua K1 > ESS bootstrap tren CUNG particle; toa do khong do giu prior.
    4. K1: prior cua predict chi dung cho DUNG mot update; update khong co predict truoc -> bootstrap (dem fallback).
    5. K1 trong pipeline: bat qua khoa yaml `pf_optimal_proposal`, trong so huu han / tong 1 moi frame, co update K1,
       chay lai cung seed bit-identical, khac bootstrap.
    6. K2: roughening 0 -> hang moi la BAN SAO dung X[idx] (idx tu hook); roughening c > 0 -> X_r - X[idx] co do lech
       chuan ~ c * sigma_Q tung toa do (N lon); khoa yaml `pf_roughening` toi duoc bo loc.

    python src/kc48/test_k1_k2.py
"""
from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "kc48"))
import tracker_ctrv  # noqa: E402
from pf_ctrv import PFCTRV, PFTrackView  # noqa: E402
from run_tracking import load_tracker_cfg, make_tracker  # noqa: E402
import test_pf_dung_dan as tpd  # noqa: E402  (dung lai canh tong hop _canh, khong chay test cua no)

BASE = "eaf2e87"
CX, CY, V, TH, OM, A, H, VA, VH = range(9)
IDX = [CX, CY, A, H]
_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def _nap_cu(ten_file: str, ten_mod: str):
    src = subprocess.run(["git", "show", f"{BASE}:src/{ten_file}"], cwd=ROOT, capture_output=True,
                         check=True).stdout.decode("utf-8")
    mod = types.ModuleType(ten_mod)
    mod.__file__ = str(ROOT / "src" / ten_file)
    exec(compile(src, f"{BASE}:src/{ten_file}", "exec"), mod.__dict__)
    return mod


_CU = {}


def _modul_cu():
    if not _CU:
        pf_cu = _nap_cu("pf_ctrv.py", "pf_ctrv_cu")
        saved = sys.modules.get("pf_ctrv")
        sys.modules["pf_ctrv"] = pf_cu
        try:
            tr_cu = _nap_cu("tracker_ctrv.py", "tracker_ctrv_cu")
        finally:
            sys.modules["pf_ctrv"] = saved
        _CU.update(pf=pf_cu, tr=tr_cu)
    return _CU["pf"], _CU["tr"]


# --------------------------------------------------------------------------------------------------------------
def _chuoi_bo_loc(pf, n_buoc=60):
    """Chuoi predict / update co mat do (buoc 20-27), nhay xa (suy bien, buoc 40), tra ve danh sach trang thai."""
    rng = pf.make_rng(3)
    m0, P0 = pf.initiate(np.array([200.0, 150.0, 1.6, 40.0]))
    X = pf.sample_initial(m0, P0, rng)
    lw = np.full(pf.num_particles, -np.log(pf.num_particles))
    out = []
    for k in range(1, n_buoc):
        X = pf.propagate(X, rng)
        z = np.array([200.0 + 3 * k, 150.0 + 1.5 * k, 1.6, 40.0 + 0.1 * k])
        if k == 40:
            z[:2] += 900.0
        if not (20 <= k < 28):
            X, lw, m, P = pf.update_particles(X, lw, z, 0.8, rng)
            out.append((X.copy(), lw.copy(), m.copy(), P.copy()))
    return out, (pf.n_resample, pf.n_degenerate)


def _pipeline(make, overrides=None, seed=0, n_frames=30):
    tr = make(load_tracker_cfg(tpd.PF_YAML, overrides))
    tr.set_seed(seed, "kc48")
    snap = []
    for f in range(1, n_frames + 1):
        out = tr.update(tpd._canh(f, b_dich=1.5, b_mat=(10, 11, 12), co_c=True).cpu().numpy(), None)
        ts = sorted(list(tr.tracked_stracks) + list(tr.lost_stracks), key=lambda t: t.track_id)
        snap.append(([(t.track_id, t.particles.copy(), t.logw.copy(), np.asarray(t.tlwh).copy()) for t in ts],
                     np.asarray(out).copy()))
    return snap, tr


def _bang_nhau(s1, s2) -> bool:
    if len(s1) != len(s2):
        return False
    for (a, oa), (b, ob) in zip(s1, s2):
        if not np.array_equal(oa, ob) or len(a) != len(b):
            return False
        for (i1, x1, l1, t1), (i2, x2, l2, t2) in zip(a, b):
            if i1 != i2 or not (np.array_equal(x1, x2) and np.array_equal(l1, l2) and np.array_equal(t1, t2)):
                return False
    return True


def test_1_mac_dinh_bit_identical_ma_goc():
    pf_cu, tr_cu = _modul_cu()
    for rough in (0.0, 0.5):
        a, ca = _chuoi_bo_loc(pf_cu.PFCTRV(num_particles=200, roughening=rough))
        b, cb = _chuoi_bo_loc(PFCTRV(num_particles=200, roughening=rough))
        same = len(a) == len(b) and all(all(np.array_equal(u, v) for u, v in zip(p, q)) for p, q in zip(a, b))
        _check(f"bo loc 60 buoc (roughening {rough}): ma moi mac dinh == ma {BASE} tung bit", same and ca == cb,
               f"bo dem (n_resample, n_degenerate): goc {ca}, moi {cb}")
    tracker_ctrv.register()
    reg_cu = {"bytetrack_pf_ctrv": tr_cu.PFByteTracker}
    for ov in (None, {"pf_roughening": 0.5}):
        s_cu, _ = _pipeline(lambda c: make_tracker(c, registry=reg_cu), ov)
        s_moi, t_moi = _pipeline(make_tracker, ov)
        _check(f"pipeline PFByteTracker 30 frame {ov or 'mac dinh'}: particle / logw / tlwh / output == ma {BASE}",
               _bang_nhau(s_cu, s_moi) and not t_moi.kalman_filter.optimal_proposal,
               f"so track frame cuoi {len(s_moi[-1][0])}; n_resample {t_moi.kalman_filter.n_resample}")


def test_2_trong_so_k1_cong_thuc_da_bien():
    pf = PFCTRV(num_particles=5, optimal_proposal=True, degenerate_maha2=1e30, resample_ess_ratio=0.0)
    rng = np.random.default_rng(2)
    X0 = np.array([300.0, 200.0, 5.0, 0.3, 0.02, 1.6, 40.0, 0.0, 0.5]) + rng.standard_normal((5, 9)) * \
        np.array([4, 4, 1, 0.3, 0.02, 0.1, 6, 1e-4, 0.3])
    lw0 = np.log(np.array([0.1, 0.3, 0.2, 0.25, 0.15]))
    Xp, prior = pf.propagate(X0, pf.make_rng(1), return_prior=True)
    z = np.array([306.0, 201.0, 1.55, 43.0])
    R = pf._measurement_noise(z[3], 0.7)
    Hm = np.zeros((4, 9)); Hm[range(4), IDX] = 1.0
    ky_vong = []
    for i in range(5):
        Qi = np.diag(pf._process_noise(max(X0[i, H], 1e-3)).diagonal())     # Q cua EKF tai h particle (truoc predict)
        S = Hm @ Qi @ Hm.T + R
        d = z - Hm @ pf.f(X0[i])
        ky_vong.append(lw0[i] - 0.5 * (d @ np.linalg.solve(S, d)) - 0.5 * np.linalg.slogdet(S)[1])
    ky_vong = np.array(ky_vong)
    ky_vong = np.exp(ky_vong - ky_vong.max()); ky_vong /= ky_vong.sum()
    _, lw1, _, _ = pf.update_particles(Xp.copy(), lw0.copy(), z, 0.7, pf.make_rng(5), prior=prior)
    w1 = np.exp(lw1 - lw1.max()); w1 /= w1.sum()
    _check("K1: trong so == w_{k-1} N(z; H f(x), H Q_i H^T + R) da bien (sai < 1e-10)",
           np.max(np.abs(w1 - ky_vong)) < 1e-10 and pf.n_k1_update == 1,
           f"K1 {np.round(w1, 6)}\nky vong {np.round(ky_vong, 6)}")


def test_3_k1_hoi_tu_kalman():
    N = 40000
    rng = np.random.default_rng(13)
    m0 = np.array([300.0, 200.0, 0.0, 0.3, 0.0, 1.6, 40.0, 0.0, 0.0])
    sd0 = np.array([3.0, 3.0, 0.0, 0.2, 0.0, 0.08, 1e-6, 0.0, 0.0])        # v = 0 -> f tuyen tinh (dong nhat)
    X0 = m0 + rng.standard_normal((N, 9)) * sd0
    z = np.array([303.5, 197.0, 1.66, 40.6])
    kq = {}
    for k1 in (True, False):
        pf = PFCTRV(num_particles=N, optimal_proposal=k1, resample_ess_ratio=0.0)
        Xp, prior = pf.propagate(X0.copy(), pf.make_rng(0), return_prior=True)
        q = prior[1][0]
        _, lw1, mean, cov = pf.update_particles(Xp, np.full(N, -np.log(N)), z, None, pf.make_rng(1), prior=prior)
        kq[k1] = (pf.ess(lw1), mean, cov, q)
    ess, mean, cov, q = kq[True]
    r = np.sqrt(np.diag(PFCTRV()._measurement_noise(z[3], None)))
    ok, lines = True, [f"ESS K1 {ess:.0f}, bootstrap {kq[False][0]:.0f} / {N}"]
    for j, k in enumerate(IDX):
        p = sd0[k] ** 2 + q[k] ** 2                                       # prior du doan = P0 + Q
        mu = (m0[k] / p + z[j] / r[j] ** 2) / (1 / p + 1 / r[j] ** 2)
        var = 1 / (1 / p + 1 / r[j] ** 2)
        ok_k = abs(mean[k] - mu) < 5 * np.sqrt(var / ess) and abs(cov[k, k] / var - 1) < 0.05
        ok &= ok_k
        lines.append(f"toa do {k}: mean {mean[k]:.4f} vs KF {mu:.4f}; var {cov[k, k]:.5f} vs KF {var:.5f}")
    ok_u = abs(mean[TH] - m0[TH]) < 5 * np.sqrt(sd0[TH] ** 2 + q[TH] ** 2) / np.sqrt(ess)
    _check("K1 (N = 40000, tuyen tinh 1 buoc): hau nghiem khop Kalman tinh tay; theta (khong do) giu prior",
           ok and ok_u, "\n".join(lines))
    _check("K1: ESS > ESS bootstrap tren cung particle prior", ess > kq[False][0])


def test_4_prior_dung_mot_lan():
    pf = PFCTRV(num_particles=50, optimal_proposal=True)
    rng = pf.make_rng(1)
    m0, P0 = pf.initiate(np.array([100.0, 100.0, 1.6, 30.0]))
    trk = types.SimpleNamespace(particles=pf.sample_initial(m0, P0, rng), logw=np.full(50, -np.log(50)), rng=rng,
                                track_id=1, frame_id=1)
    view = PFTrackView(pf, trk)
    view.predict(None, None, lost=False)
    co_prior = trk._k1_prior is not None
    view.update(None, None, np.array([101.0, 100.5, 1.6, 30.2]))
    view.update(None, None, np.array([101.5, 100.7, 1.6, 30.2]))       # khong predict truoc -> bootstrap
    _check("K1: predict luu prior; prior dung dung 1 update; update khong predict -> bootstrap (fallback 1)",
           co_prior and trk._k1_prior is None and pf.n_k1_update == 1 and pf.n_k1_fallback == 1,
           f"n_k1_update {pf.n_k1_update}, n_k1_fallback {pf.n_k1_fallback}")
    pf0 = PFCTRV(num_particles=50)
    trk0 = types.SimpleNamespace(particles=trk.particles.copy(), logw=np.full(50, -np.log(50)), rng=pf0.make_rng(1),
                                 track_id=1, frame_id=1)
    PFTrackView(pf0, trk0).predict(None, None, lost=False)
    _check("K1 tat: predict khong luu prior", getattr(trk0, "_k1_prior", None) is None)


def test_5_k1_pipeline():
    tracker_ctrv.register()
    ov = {"pf_optimal_proposal": True}
    s1, t1 = _pipeline(make_tracker, ov)
    s2, _ = _pipeline(make_tracker, ov)
    s0, _ = _pipeline(make_tracker, None)
    ok = True
    for tracks, _ in s1:
        for _, X, lw, _ in tracks:
            ok &= bool(np.all(np.isfinite(X)) and np.isfinite(lw.max()) and abs(float(np.sum(np.exp(lw))) - 1.0) < 1e-9)
    kf = t1.kalman_filter
    _check("K1 pipeline (yaml pf_optimal_proposal true): particle huu han, tong trong so 1 moi frame, co update K1",
           ok and kf.optimal_proposal and kf.n_k1_update > 0,
           f"n_update {kf.n_update}, n_k1_update {kf.n_k1_update}, n_k1_fallback {kf.n_k1_fallback}")
    _check("K1 pipeline: chay lai cung seed bit-identical; khac bootstrap", _bang_nhau(s1, s2) and not _bang_nhau(s0, s1))
    cfg = load_tracker_cfg(tpd.PF_YAML, {"pf_optimal_proposal": "false"})
    _check("khoa yaml: 'false' -> tat; mac dinh yaml cu -> tat",
           not make_tracker(cfg).kalman_filter.optimal_proposal
           and not make_tracker(load_tracker_cfg(tpd.PF_YAML)).kalman_filter.optimal_proposal)


def test_6_k2_roughening_va_hook():
    n = 20000
    rng = np.random.default_rng(4)
    X = np.array([300.0, 200.0, 5.0, 0.3, 0.02, 1.6, 40.0, 0.0, 0.5]) + rng.standard_normal((n, 9))
    lw = rng.standard_normal(n) * 3.0
    lw -= np.log(np.sum(np.exp(lw - lw.max()))) + lw.max()
    pf0 = PFCTRV(num_particles=n, roughening=0.0)
    Xr0, _, rs0 = pf0.resample_if_needed(X, lw.copy(), np.random.default_rng(7))
    idx0 = pf0.last_resample_idx
    _check("hook: roughening 0 -> hang moi == X[last_resample_idx] dung tung bit; so to tien < N",
           rs0 and np.array_equal(Xr0, X[idx0]) and len(np.unique(idx0)) < n, f"to tien {len(np.unique(idx0))}/{n}")
    c = 0.5
    pf1 = PFCTRV(num_particles=n, roughening=c)
    Xr1, _, _ = pf1.resample_if_needed(X, lw.copy(), np.random.default_rng(7))
    idx1 = pf1.last_resample_idx
    sig = pf1.process_noise_std_batch(np.maximum(X[idx1, H], 1e-3))
    jit = (Xr1 - X[idx1]) / sig
    jit[:, TH] = ((Xr1[:, TH] - X[idx1, TH] + np.pi) % (2 * np.pi) - np.pi) / sig[:, TH]
    sd = jit.std(axis=0)
    _check(f"K2: roughening {c} -> (X_r - X[idx]) / sigma_Q co SD ~ {c} moi toa do (sai < 5 %)",
           np.array_equal(idx0, idx1) and np.all(np.abs(sd / c - 1) < 0.05), f"SD chuan hoa {np.round(sd, 4)}")
    tracker_ctrv.register()
    kf = make_tracker(load_tracker_cfg(tpd.PF_YAML, {"pf_roughening": 0.25})).kalman_filter
    _check("khoa yaml pf_roughening toi bo loc", kf.roughening == 0.25)
    pf2 = PFCTRV(num_particles=10)
    pf2.resample_if_needed(np.zeros((10, 9)), np.full(10, -np.log(10)), np.random.default_rng(0))
    _check("hook: khong resample (ESS = N) -> last_resample_idx None", pf2.last_resample_idx is None)


def main() -> int:
    failed = 0
    for t in [test_1_mac_dinh_bit_identical_ma_goc, test_2_trong_so_k1_cong_thuc_da_bien, test_3_k1_hoi_tu_kalman,
              test_4_prior_dung_mot_lan, test_5_k1_pipeline, test_6_k2_roughening_va_hook]:
        print(f"\n--- {t.__name__} ---")
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  !! {e}")
    n_ok = sum(ok for _, ok in _RESULTS)
    print("\n" + "=" * 78)
    print(f"KET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS, {failed} test co loi")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
