"""
test_pf_dung_dan.py -- R48C (M4, phan tinh dung): kiem TINH CHAT cua PF tren API that cua src/pf_ctrv.py va
PFByteTracker, so voi ky vong DOC LAP (ly thuyet / phan thuc), khong chep lai cai dat. Bo sung src/test_pf_ctrv.py
(khong lap lai: f_batch == f, trung binh vong, trigger ESS, seed qua tracker, multi_predict == predict).

     1. Cap nhat mot buoc tren mo hinh tuyen tinh-Gauss: trung binh / phuong sai co trong so ~ hau nghiem Kalman
     2. ESS = 1 / sum w^2 dung tren trong so biet truoc (deu, mot diem, hai khoi)
     3. systematic_resample: chi so trong mien, dem tong = N, moi dem trong {floor, ceil}(N w_i),
        trung binh dem qua nhieu u -> N w_i (khong lech), particle trong so 0 khong bao gio duoc chon
     4. resample_if_needed: duoi nguong -> hang la BAN SAO hang cu, logw deu, bo dem tang; tren nguong -> khong doi
     5. On dinh so: log-likelihood cuc tri (khoang cach 1e5 px, nguong suy bien noi rong) -> trong so huu han, tong 1,
        particle gan nhat nhan trong so lon nhat; logw co -inf (particle trong so 0) van hop le
     6. dt: f_batch(dt = 2) == f_batch(dt = 1) hai lan (CTRV hang so, nhanh re va thang); khoa yaml ctrv_dt toi bo loc
     7. Chuan hoa goc: 200 buoc omega = 0.3 -> theta luon trong [-pi, pi), huong vat ly va vi tri khop cong thuc dong
     8. Khong co do (LOST): PFTrackView.predict(lost = True) dat vh = 0 truoc khi du doan, logw khong doi
     9. Pipeline PFByteTracker co mat do + resample: moi frame moi track logw huu han, sum exp = 1, particle huu han, h > 0;
        track LOST giu nguyen logw qua cac frame mat do
    10. Doc lap trang thai giua track (phan thuc): doi / bo detection cua B -> particle, logw cua A bit-identical;
        them track C sinh sau -> A, B bit-identical (khong dung chung rng / trang thai); khong cap mang nao chung bo nho
    11. Tai lap theo seed o cap bo loc qua chuoi co resample + suy bien: cung (seed, video, track_id) -> bit-identical;
        khac track_id / video -> khac

    python src/kc48/test_pf_dung_dan.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402
import tracker_ctrv  # noqa: E402
from pf_ctrv import PFCTRV, PFTrackView, systematic_resample  # noqa: E402
from run_tracking import load_tracker_cfg, make_tracker  # noqa: E402

from ultralytics.engine.results import Boxes  # noqa: E402

PF_YAML = ROOT / "configs/phase2/trackers/bytetrack_pf_ctrv_n100.yaml"
CX, CY, V, TH, OM, A, H, VA, VH = range(9)
_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def _w(logw):
    return np.exp(logw - np.max(logw)) / np.sum(np.exp(logw - np.max(logw)))


# --------------------------------------------------------------------------------------------------------------
def test_1_mot_buoc_vs_kalman():
    """Prior Gauss cheo cho (cx, cy, a, h), do tuyen tinh z = Hx + v: hau nghiem dung la Kalman tung toa do."""
    N = 40000
    pf = PFCTRV(num_particles=N, resample_ess_ratio=0.0)        # ratio 0: khong resample -> doc dung uoc luong IS
    rng = np.random.default_rng(11)
    m0 = np.array([300.0, 200.0, 5.0, 0.3, 0.0, 1.6, 40.0, 0.0, 0.0])
    sd0 = np.array([3.0, 3.0, 1.0, 0.2, 0.05, 0.08, 2.0, 1e-3, 0.2])
    X = m0 + rng.standard_normal((N, 9)) * sd0
    logw = np.full(N, -np.log(N))
    z = np.array([302.5, 198.0, 1.65, 41.0])
    r = np.sqrt(np.diag(pf._measurement_noise(z[3], None)))    # R cua ma, tinh tai h quan sat
    X1, logw1, mean, cov = pf.update_particles(X.copy(), logw, z, None, pf.make_rng(0))
    ess = pf.ess(logw1)
    ok, lines = True, [f"ESS = {ess:.0f} / {N}"]
    for j, k in enumerate((CX, CY, A, H)):
        p, rr = sd0[k] ** 2, r[j] ** 2
        mu_kf = (m0[k] / p + z[j] / rr) / (1 / p + 1 / rr)      # gia tri ky vong theo Kalman, khong qua ma PF
        var_kf = 1 / (1 / p + 1 / rr)
        tol_m = 5 * np.sqrt(var_kf / ess)
        ok_m = abs(mean[k] - mu_kf) < tol_m
        ok_v = abs(cov[k, k] / var_kf - 1) < 0.10
        ok &= ok_m and ok_v
        lines.append(f"toa do {k}: mean {mean[k]:.4f} vs KF {mu_kf:.4f} (tol {tol_m:.4f}); var {cov[k, k]:.5f} vs KF {var_kf:.5f}")
    # toa do khong do: phan bo bien duyen khong doi (prior doc lap voi z)
    ok_u = abs(mean[V] - m0[V]) < 5 * sd0[V] / np.sqrt(ess)
    _check("1 buoc update (N = 40000) khop hau nghiem Kalman tung toa do do; v (khong do) giu prior", ok and ok_u,
           "\n".join(lines))
    _check("X khong bi doi khi khong suy bien (update chi doi trong so)", np.array_equal(X1, X))


def test_2_ess():
    pf = PFCTRV(num_particles=10)
    n = 10
    _check("ESS: trong so deu -> N", abs(pf.ess(np.full(n, -np.log(n))) - n) < 1e-9)
    lw = np.full(n, -np.inf); lw[3] = 0.0
    _check("ESS: mot particle mang toan bo trong so -> 1", abs(pf.ess(lw) - 1) < 1e-12)
    lw = np.full(n, -np.inf); lw[[2, 7]] = np.log(0.5)
    _check("ESS: hai khoi 0.5 / 0.5 -> 2; khong doi khi cong hang so vao logw", abs(pf.ess(lw) - 2) < 1e-12
           and abs(pf.ess(lw + 123.4) - 2) < 1e-9)


def test_3_systematic_resample():
    rng = np.random.default_rng(5)
    ok_mien, ok_floor = True, True
    for _ in range(200):
        n = int(rng.integers(2, 60))
        w = rng.random(n) ** 3
        w[rng.random(n) < 0.3] = 0.0
        if w.sum() == 0:
            w[0] = 1.0
        w /= w.sum()
        idx = systematic_resample(w, float(rng.random()))
        cnt = np.bincount(idx, minlength=n)
        ok_mien &= len(idx) == n and idx.min() >= 0 and idx.max() < n and cnt.sum() == n
        ok_floor &= bool(np.all(cnt >= np.floor(n * w) - 1e-9) and np.all(cnt <= np.ceil(n * w) + 1e-9))
        ok_floor &= bool(np.all(cnt[w == 0] == 0))
    _check("200 bo trong so ngau nhien (co 0): chi so trong [0, N), tong dem = N", ok_mien)
    _check("moi dem nam trong {floor, ceil}(N w_i) (tinh chat systematic); trong so 0 -> dem 0", ok_floor)
    n, M = 7, 20000
    w = np.array([0.31, 0.02, 0.2, 0.0, 0.17, 0.25, 0.05])
    tot = np.zeros(n)
    for u in rng.random(M):
        tot += np.bincount(systematic_resample(w, float(u)), minlength=n)
    mean_cnt = tot / M
    # sai so chuan cua trung binh dem <= 0.5 / sqrt(M) (dem chi nhan floor/ceil)
    _check("khong lech: trung binh dem qua 20000 u ~ N w_i (|lech| < 5 * 0.5 / sqrt(M))",
           np.all(np.abs(mean_cnt - n * w) < 5 * 0.5 / np.sqrt(M)),
           f"dem TB {np.round(mean_cnt, 4)}\nN w    {np.round(n * w, 4)}")


def test_4_resample_if_needed():
    n = 50
    pf = PFCTRV(num_particles=n, resample_ess_ratio=0.5)
    rng = np.random.default_rng(3)
    X = rng.standard_normal((n, 9)) * 10
    lw = np.log(np.r_[np.full(5, 0.18), np.full(n - 5, 0.1 / (n - 5))])   # ESS ~ 7.6 < 25
    Xr, lwr, did = pf.resample_if_needed(X, lw - np.log(np.exp(lw).sum()), pf.make_rng(1))
    rows_ok = all(any(np.array_equal(Xr[i], X[j]) for j in range(n)) for i in range(n))
    _check("ESS < 0.5 N: co resample, moi hang moi la ban sao mot hang cu, logw deu, n_resample = 1",
           did and rows_ok and np.allclose(lwr, -np.log(n)) and pf.n_resample == 1)
    lw2 = np.full(n, -np.log(n))
    Xs, lws, did2 = pf.resample_if_needed(X, lw2, pf.make_rng(1))
    _check("ESS = N >= 0.5 N: khong resample, X va logw giu nguyen", (not did2) and Xs is X and lws is lw2
           and pf.n_resample == 1)


def test_5_on_dinh_so():
    n = 200
    pf = PFCTRV(num_particles=n, degenerate_maha2=1e30)       # noi nguong suy bien de di qua nhanh log-sum-exp
    rng = pf.make_rng(4)
    m, P = pf.initiate(np.array([300.0, 200.0, 1.6, 40.0]))
    X = pf.sample_initial(m, P, rng)
    z = np.array([300.0 + 1e5, 200.0, 1.6, 40.0])               # Mahalanobis^2 ~ 1e10: exp(loglik) = 0 neu tinh thang
    ll = pf.log_likelihood(X, z, np.diag(pf._measurement_noise(z[3], None)))
    naive = np.exp(ll)
    X1, lw1, mean, _ = pf.update_particles(X.copy(), np.full(n, -np.log(n)), z, None, rng)
    w = np.exp(lw1)
    _check("loglik ~ -1e10: exp truc tiep = 0 (de doi chieu) nhung logw sau update huu han, sum exp = 1",
           naive.max() == 0.0 and np.isfinite(lw1).all() and abs(w.sum() - 1) < 1e-12 and np.isfinite(mean).all(),
           f"max loglik {ll.max():.3e}, sum exp {w.sum():.15f}, n_degenerate {pf.n_degenerate}")
    # ESS ~ 1 nen co resample -> trong so deu; kiem particle gan z nhat chiem trong so lon nhat TRUOC resample:
    pf0 = PFCTRV(num_particles=n, degenerate_maha2=1e30, resample_ess_ratio=0.0)
    _, lw0, _, _ = pf0.update_particles(X.copy(), np.full(n, -np.log(n)), z, None, pf0.make_rng(4))
    _check("particle co cx lon nhat (gan z nhat) nhan trong so lon nhat; chi so trong so = phan thuc",
           int(np.argmax(lw0)) == int(np.argmax(X[:, CX])) and np.isfinite(lw0).all())
    lw = np.full(n, -np.log(n - 3)); lw[:3] = -np.inf                     # 3 particle trong so 0
    z2 = np.array([301.0, 199.0, 1.6, 40.0])
    _, lw2, mean2, cov2 = pf0.update_particles(X.copy(), lw, z2, None, pf0.make_rng(5))
    _check("logw co -inf (particle trong so 0): sau update van -inf o do, phan con lai huu han, sum exp = 1",
           np.isneginf(lw2[:3]).all() and np.isfinite(lw2[3:]).all() and abs(np.exp(lw2).sum() - 1) < 1e-12
           and np.isfinite(mean2).all() and np.isfinite(cov2).all())


def test_6_dt():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(40, 9)) * [50, 50, 3, 2, 0.1, 0.1, 10, 0.01, 0.5] + [300, 200, 5, 0, 0, 1.6, 40, 0, 0]
    X[:10, OM] = 0.0                                             # nhanh thang (omega = 0 chinh xac)
    X[10:, OM] = np.sign(X[10:, OM]) * (np.abs(X[10:, OM]) + 1e-3)   # nhanh re (|omega| > eps)
    p1, p2 = PFCTRV(num_particles=2, dt=1.0), PFCTRV(num_particles=2, dt=2.0)
    one_twice = p1.f_batch(p1.f_batch(X))
    two = p2.f_batch(X)
    d = np.abs(one_twice - two)
    d[:, TH] = np.abs(((one_twice[:, TH] - two[:, TH]) + np.pi) % (2 * np.pi) - np.pi)
    _check("CTRV hang so: f(dt = 2) == f(dt = 1) o f(dt = 1) (ca nhanh thang va re), sai so < 1e-9",
           d.max() < 1e-9, f"max |d| = {d.max():.2e}")
    _check("don vi mac dinh: PFCTRV().dt == config.CTRV_DT == 1 buoc", PFCTRV(num_particles=2).dt == config.CTRV_DT == 1.0)
    tracker_ctrv.register()
    tr = make_tracker(load_tracker_cfg(PF_YAML, {"ctrv_dt": 2.0}))
    _check("khoa yaml ctrv_dt = 2.0 duoc truyen toi PFCTRV cua tracker (khong bi bo qua)", tr.kalman_filter.dt == 2.0,
           f"dt = {tr.kalman_filter.dt}")


def test_7_chuan_hoa_goc():
    pf = PFCTRV(num_particles=2)
    th0, om, v = np.pi - 0.05, 0.3, 6.0
    X = np.array([[100.0, 100.0, v, th0, om, 1.6, 40.0, 0.0, 0.0]])
    ok_range, ok_dir, ok_pos = True, True, True
    for k in range(1, 201):
        X = pf.f_batch(X)
        th_true = th0 + k * om
        ok_range &= bool(-np.pi <= X[0, TH] < np.pi)
        ok_dir &= abs(np.cos(X[0, TH]) - np.cos(th_true)) < 1e-9 and abs(np.sin(X[0, TH]) - np.sin(th_true)) < 1e-9
        ok_pos &= abs(X[0, CX] - (100 + v / om * (np.sin(th_true) - np.sin(th0)))) < 1e-6 \
            and abs(X[0, CY] - (100 + v / om * (np.cos(th0) - np.cos(th_true)))) < 1e-6
    _check("200 buoc omega = 0.3 (qua bien +-pi nhieu lan): theta luon trong [-pi, pi)", ok_range)
    _check("huong vat ly (cos, sin) va vi tri khop cong thuc dong cua cung tron", ok_dir and ok_pos)


def test_8_khong_do():
    n = 4000
    pf = PFCTRV(num_particles=n)
    rng = pf.make_rng(9)
    X = np.zeros((n, 9)); X[:, CX] = 300; X[:, CY] = 200; X[:, V] = 4; X[:, A] = 1.6; X[:, H] = 40; X[:, VH] = 5.0
    lw = np.log(np.random.default_rng(1).dirichlet(np.ones(n)))
    trk = types.SimpleNamespace(particles=X.copy(), logw=lw.copy(), rng=rng)
    view = PFTrackView(pf, trk)
    m, P = view.predict(None, None, lost=True)
    q_vh = pf._std_weight_velocity * 40
    vh = trk.particles[:, VH]
    _check("predict(lost = True): vh = 0 truoc du doan -> vh moi ~ N(0, q_vh^2) (khong ~ 5)",
           abs(vh.mean()) < 5 * q_vh / np.sqrt(n) and abs(vh.std() / q_vh - 1) < 0.1,
           f"vh TB {vh.mean():.4f}, sd {vh.std():.4f}, q_vh {q_vh:.4f}")
    _check("predict khong do: logw giu nguyen tung bit, mean / cov huu han", np.array_equal(trk.logw, lw)
           and np.isfinite(m).all() and np.isfinite(P).all())
    trk2 = types.SimpleNamespace(particles=X.copy(), logw=lw.copy(), rng=pf.make_rng(9))
    PFTrackView(pf, trk2).predict(None, None, lost=False)
    _check("doi chung lost = False: vh giu ~ 5", abs(trk2.particles[:, VH].mean() - 5) < 5 * q_vh / np.sqrt(n))


# --------------------------------------------------------------------------------------------------------------
def _boxes(rows):
    return Boxes(np.array(rows, dtype=np.float32).reshape(-1, 6), (540, 960))


def _canh(f, b_dich=0.0, b_mat=(), co_c=False):
    rows = []
    cx, cy = 100 + 3 * f, 100 + 1 * f                                    # A: goc trai tren
    rows.append([cx - 20, cy - 15, cx + 20, cy + 15, 0.9, 2.0])
    if f not in b_mat:                                                   # B: goc phai duoi
        cx, cy = 700 - 2 * f + (b_dich if f % 3 == 0 else 0.0), 400 + 0.5 * f
        rows.append([cx - 30, cy - 20, cx + 30, cy + 20, 0.85, 2.0])
    if co_c and f >= 5:                                                  # C: giua duoi, sinh o frame 5
        cx, cy = 420, 500 - 2 * f
        rows.append([cx - 15, cy - 10, cx + 15, cy + 10, 0.8, 2.0])
    return _boxes(rows)


def _chay(n_frames=30, seed=0, overrides=None, **kw):
    tr = make_tracker(load_tracker_cfg(PF_YAML, overrides))
    tr.set_seed(seed, "kc48")
    snap = []
    for f in range(1, n_frames + 1):
        tr.update(_canh(f, **kw).cpu().numpy(), None)
        ts = list(tr.tracked_stracks) + list(tr.lost_stracks)
        snap.append({t.track_id: (t.particles.copy(), t.logw.copy(), t.state, t) for t in ts})
    return snap, tr


def test_9_pipeline_trong_so_va_lost():
    tracker_ctrv.register()
    snap, tr = _chay(b_dich=1.5, b_mat=(10, 11, 12))
    ok, n_chk = True, 0
    for fr in snap:
        for tid, (X, lw, _, _) in fr.items():
            n_chk += 1
            ok &= bool(np.isfinite(lw).all() and abs(np.exp(lw).sum() - 1) < 1e-9 and np.isfinite(X).all()
                       and (X[:, H] > 0).all())
    st = tr.pf_stats()
    _check(f"pipeline 30 frame co mat do: {n_chk} lan kiem (track x frame) logw huu han, sum exp = 1, X huu han, h > 0; "
           f"co resample (n_resample = {st['n_resample']})", ok and st["n_resample"] > 0)
    # Voi ratio 0.5, gan nhu moi update deu resample -> logw sau update la deu 1/N, nen "logw khong doi khi LOST" se
    # dung mot cach tam thuong. Phan kiem LOST tat resample (ratio 0) de logw truoc khi mat do KHONG deu.
    snap0, _ = _chay(b_dich=1.5, b_mat=(10, 11, 12), overrides={"pf_resample_ess_ratio": 0.0})
    b9, b10, b11, b12, b13 = (snap0[i][2] for i in (8, 9, 10, 11, 12))
    khong_deu = float(np.ptp(b9[1])) > 1.0
    ok_lost = all(np.array_equal(b[1], b9[1]) for b in (b10, b11, b12)) \
        and not np.array_equal(b10[0], b9[0]) and not np.array_equal(b11[0], b10[0])
    _check("ratio 0: logw cua B sau frame 9 KHONG deu; B mat detection frame 10-12 (trang thai Lost): logw giu nguyen "
           "tung bit, particle van du doan", khong_deu and ok_lost and b11[2] != b9[2],
           f"ptp(logw) frame 9 = {np.ptp(b9[1]):.2f}; trang thai frame 9 / 11 = {b9[2]} / {b11[2]}")
    _check("B ghep lai o frame 13 cung ID 2, trang thai Tracked, logw doi (co nhan likelihood)",
           (not np.array_equal(b13[1], b12[1])) and b13[2] == b9[2])


def test_10_doc_lap_track():
    tracker_ctrv.register()
    ref, _ = _chay()
    alt, _ = _chay(b_dich=1.5, b_mat=(10, 11, 12))
    them, _ = _chay(co_c=True)
    same_a = all(np.array_equal(r[1][0], a[1][0]) and np.array_equal(r[1][1], a[1][1]) for r, a in zip(ref, alt))
    diff_b = any(not np.array_equal(r[2][0], a[2][0]) for r, a in zip(ref, alt))
    _check("doi / bo detection cua B (ID 2): particle + logw cua A (ID 1) bit-identical o MOI frame; B thi khac",
           same_a and diff_b)
    same_ab = all(np.array_equal(r[k][0], t[k][0]) and np.array_equal(r[k][1], t[k][1])
                  for r, t in zip(ref, them) for k in (1, 2))
    has_c = all(3 in t for t in them[5:])
    _check("them track C (ID 3, sinh frame 5): A, B bit-identical o moi frame (khong dung chung rng / trang thai)",
           same_ab and has_c)
    shared = 0
    for fr in them:
        objs = [(tid, v[3]) for tid, v in fr.items()]
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                ti, tj = objs[i][1], objs[j][1]
                shared += int(np.shares_memory(ti.particles, tj.particles) or np.shares_memory(ti.logw, tj.logw)
                              or ti.rng is tj.rng)
    _check("khong cap track nao dung chung bo nho particle / logw hay chung doi tuong rng (moi frame)", shared == 0,
           f"so cap chung = {shared}")


def test_11_tai_lap_seed_bo_loc():
    def chuoi(seed, video, tid):
        pf = PFCTRV(num_particles=100, seed=seed)
        pf.set_seed(seed, video)
        rng = pf.make_rng(tid)
        m, P = pf.initiate(np.array([300.0, 200.0, 1.6, 40.0]))
        X = pf.sample_initial(m, P, rng); lw = np.full(100, -np.log(100))
        for k in range(1, 40):
            X = pf.propagate(X, rng)
            if k == 25:
                z = np.array([5000.0, 5000.0, 1.6, 40.0])             # nhay xa -> nhanh suy bien (reseed vi tri)
            else:
                z = np.array([300.0 + 4 * k, 200.0 + 1.0 * k, 1.6, 40.0])
            if k % 7:
                X, lw, _, _ = pf.update_particles(X, lw, z, None, rng)
        return X, lw, pf.n_resample, pf.n_degenerate

    a, b = chuoi(0, "v", 1), chuoi(0, "v", 1)
    # nhay toi (5000, 5000) roi quay ve -> hai lan suy bien (di va ve), deu di qua nhanh reseed ngau nhien
    _check("cung (seed, video, track_id): particle + logw bit-identical sau 40 buoc; chuoi co di qua resample va "
           "nhanh suy bien", np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[2] > 0 and a[3] >= 1
           and (a[2], a[3]) == (b[2], b[3]), f"n_resample {a[2]}, n_degenerate {a[3]}")
    c, d = chuoi(0, "v", 2), chuoi(0, "w", 1)
    _check("khac track_id hoac khac video -> dong ngau nhien khac", not np.array_equal(a[0], c[0])
           and not np.array_equal(a[0], d[0]))


def main() -> int:
    failed = 0
    for t in [test_1_mot_buoc_vs_kalman, test_2_ess, test_3_systematic_resample, test_4_resample_if_needed,
              test_5_on_dinh_so, test_6_dt, test_7_chuan_hoa_goc, test_8_khong_do, test_9_pipeline_trong_so_va_lost,
              test_10_doc_lap_track, test_11_tai_lap_seed_bo_loc]:
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
