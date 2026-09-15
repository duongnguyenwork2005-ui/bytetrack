"""
test_pf_ctrv.py -- Kiem tra Particle Filter CTRV (src/pf_ctrv.py + PFSTrack/PFByteTracker).

    1. f_batch (vector hoa) == EKFTrackerCTRV.f tung particle (re va di thang, |omega| ~ 0)
    2. omega ~ 0: khong NaN, dich chuyen == CV
    3. Trung binh vong theta quanh bien +-pi
    4. Thieu measurement 30 buoc: bbox huu han, trace(cov) tang
    5. Trong so suy bien (z cach 1e4 px): reseed vi tri, khong NaN, bo dem tang
    6. Cung seed -> bit-identical qua PFByteTracker; seed khac -> khac
    7. multi_predict khong phu thuoc thu tu track trong pool; == predict() tung track
    8. ESS / resampling: kich hoat khi ratio = 0.5, khong bao gio khi ratio = 0
    9. Nhat quan: duong thang nhieu Gaussian, N = 3000 -> PF ~ EKF (RMSE, v, theta);
       N = 100 suy bien (in ra de doi chieu, khong khang dinh)
   10. Vong doi qua PFByteTracker: activate -> update -> lost -> re_activate, bookkeeping
       trung CTRVSTrack, box > 0, mean huu han
   11. Runner: --seeds 0 1 tao 2 thu muc -s0/-s1, cung seed chay lai bit-identical (file txt)

    python src/test_pf_ctrv.py
"""
from __future__ import annotations

import copy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import tracker_ctrv  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV, normalize_angle  # noqa: E402
from pf_ctrv import PFCTRV, systematic_resample  # noqa: E402
from run_tracking import load_tracker_cfg, make_tracker  # noqa: E402
from tracker_ctrv import CTRVByteTracker, PFByteTracker, PFSTrack  # noqa: E402

from ultralytics.engine.results import Boxes  # noqa: E402
from ultralytics.trackers.basetrack import TrackState  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PF_YAML = ROOT / "configs/phase2/trackers/bytetrack_pf_ctrv_n100.yaml"
_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def boxes(rows):
    return Boxes(np.array(rows, dtype=np.float32).reshape(-1, 6), (540, 960))


def scene(f, gap_every=5):
    if gap_every and f % gap_every == 0:
        return boxes(np.zeros((0, 6)))
    rows = []
    for k in range(3):
        cx, cy = 100 + 80 * k + (4 + k) * f, 120 + 100 * k + 2 * f
        rows.append([cx - 20, cy - 15, cx + 20, cy + 15, 0.85 - 0.1 * k, 2.0])
    return boxes(rows)


def test_1_f_batch_matches_f():
    pf = PFCTRV(num_particles=10)
    ekf = EKFTrackerCTRV()
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 9)) * [200, 200, 5, 2, 0.05, 0.1, 10, 0.01, 0.5] + [300, 200, 5, 0, 0, 0.6, 40, 0, 0]
    X[:10, 4] = 0.0                       # dung thang
    X[10:15, 4] = 5e-5                    # duoi eps
    X[:, 3] = normalize_angle(X[:, 3])
    ref = np.array([ekf.f(x) for x in X])
    out = pf.f_batch(X)
    _check("f_batch == f tung particle (re + thang, 50 mau), sai so < 1e-12", np.allclose(out, ref, atol=1e-12),
           f"max |d| = {np.abs(out - ref).max():.2e}")


def test_2_omega_near_zero():
    pf = PFCTRV(num_particles=10)
    X = np.array([[100, 100, 6.0, np.pi / 4, 0.0, 0.5, 40, 0, 0], [100, 100, 6.0, np.pi / 4, 1e-6, 0.5, 40, 0, 0]])
    out = pf.f_batch(X)
    _check("omega = 0 va 1e-6: khong NaN, dich chuyen = v*cos/sin(theta)",
           np.isfinite(out).all() and np.allclose(out[:, 0], 100 + 6 * np.cos(np.pi / 4))
           and np.allclose(out[:, 1], 100 + 6 * np.sin(np.pi / 4)))


def test_3_circular_mean():
    pf = PFCTRV(num_particles=4)
    X = np.zeros((4, 9)); X[:, 6] = 40
    X[:, 3] = [np.pi - 0.05, -np.pi + 0.05, np.pi - 0.02, -np.pi + 0.02]
    logw = np.full(4, -np.log(4))
    m, P = pf.estimate_state(X, logw)
    _check("trung binh vong: 4 goc sat +-pi -> mean ~ +-pi (khong phai ~0)", abs(abs(m[3]) - np.pi) < 0.05, f"theta={m[3]:.4f}")
    _check("phuong sai theta nho (residual da wrap)", P[3, 3] < 0.01, f"P[theta]={P[3, 3]:.5f}")


def test_4_missing_measurements():
    pf = PFCTRV(num_particles=200)
    rng = pf.make_rng(1)
    m, P = pf.initiate(np.array([300.0, 200.0, 0.5, 40.0]))
    X = pf.sample_initial(m, P, rng)
    X = pf.reseed_motion(X, 5.0, 0.3, 40.0, rng)
    logw = np.full(200, -np.log(200))
    traces = []
    for _ in range(30):
        X = pf.propagate(X, rng)
        m, P = pf.estimate_state(X, logw)
        traces.append(np.trace(P[:2, :2]))
    _check("30 buoc predict khong measurement: mean/cov huu han, h > 0",
           np.isfinite(m).all() and np.isfinite(P).all() and m[6] > 0)
    _check("do bat dinh vi tri tang theo thoi gian (trace tang tong the)",
           traces[-1] > traces[0] * 3, f"trace: {traces[0]:.1f} -> {traces[-1]:.1f}")


def test_5_degenerate_weights():
    pf = PFCTRV(num_particles=100, degenerate_maha2=1e4)
    rng = pf.make_rng(2)
    m, P = pf.initiate(np.array([300.0, 200.0, 0.5, 40.0]))
    X = pf.sample_initial(m, P, rng); logw = np.full(100, -np.log(100))
    X, logw, mean, cov = pf.update_particles(X, logw, np.array([9000.0, 9000.0, 0.5, 40.0]), 0.9, rng)
    _check("z cach 1e4 px: bo dem suy bien tang, khong NaN, particle da doi ve quanh z",
           pf.n_degenerate == 1 and np.isfinite(X).all() and np.isfinite(mean).all()
           and abs(mean[0] - 9000) < 20 and np.isfinite(logw).all(), f"mean={mean[:2]}")
    X, logw, mean2, _ = pf.update_particles(X, logw, np.array([9002.0, 9001.0, 0.5, 40.0]), 0.9, rng)
    _check("update binh thuong sau do: khong suy bien them", pf.n_degenerate == 1 and abs(mean2[0] - 9002) < 5)
    idx = systematic_resample(np.ones(3) / 3, 0.5)
    _check("systematic_resample hop le voi trong so deu", len(idx) == 3 and set(idx.tolist()) <= {0, 1, 2})


def _run_pf(seed, yaml=PF_YAML, n_frames=40, video="vid", overrides=None):
    cfg = load_tracker_cfg(yaml, overrides)
    tr = make_tracker(cfg); tr.set_seed(seed, video)
    outs = [tr.update(scene(f).cpu().numpy(), None).copy() for f in range(1, n_frames + 1)]
    return outs, tr


def test_6_seed_reproducibility():
    a, _ = _run_pf(0); b, _ = _run_pf(0); c, _ = _run_pf(1); d, _ = _run_pf(0, video="other")
    _check("cung seed + video -> bit-identical (40 frame)", all(np.array_equal(x, y) for x, y in zip(a, b)))
    _check("seed khac -> khac", not all(np.array_equal(x, y) for x, y in zip(a, c)))
    _check("video khac (salt) -> khac", not all(np.array_equal(x, y) for x, y in zip(a, d)))


def test_7_pool_order_independence():
    cfg = load_tracker_cfg(PF_YAML)
    tr = make_tracker(cfg); tr.set_seed(0, "vid")
    for f in (1, 2, 3):
        tr.update(scene(f, gap_every=0).cpu().numpy(), None)
    tracks = list(tr.tracked_stracks)
    _check("co 3 track de kiem tra", len(tracks) == 3)
    # deepcopy CA DANH SACH mot lan de cac ban sao van dung chung MOT PFCTRV (guard cua Moc 2:
    # moi track trong pool phai cung instance bo loc; copy tung track rieng le se nhan ban PFCTRV).
    A = copy.deepcopy(tracks)
    B = copy.deepcopy(tracks)[::-1]
    PFSTrack.multi_predict(A)
    PFSTrack.multi_predict(B)
    same = all(np.array_equal(a.particles, b.particles) and np.array_equal(a.mean, b.mean)
               for a, b in zip(A, B[::-1]))
    _check("multi_predict: dao thu tu pool -> particle/mean tung track khong doi", same)
    C = copy.deepcopy(tracks)
    for t in C:
        t.predict()
    _check("multi_predict == predict() tung track (cung rng)",
           all(np.array_equal(a.particles, c.particles) for a, c in zip(A, C)))


def test_8_resampling():
    _, tr = _run_pf(0, overrides={"pf_resample_ess_ratio": 0.5})
    st = tr.pf_stats()
    _check("ratio 0.5: co resample", st["n_resample"] > 0, str(st))
    _, tr0 = _run_pf(0, overrides={"pf_resample_ess_ratio": 0.0})
    _check("ratio 0: khong bao gio resample", tr0.pf_stats()["n_resample"] == 0, str(tr0.pf_stats()))


def test_9_consistency_with_ekf():
    rng0 = np.random.default_rng(123)
    T, v_true, th_true = 60, 5.0, np.deg2rad(30)
    truth = np.array([[100 + v_true * np.cos(th_true) * t, 100 + v_true * np.sin(th_true) * t, 0.5, 40.0] for t in range(T)])
    meas = truth + rng0.normal(0, [1.0, 1.0, 0.01, 1.0], size=truth.shape)
    ekf = EKFTrackerCTRV()
    res = {}
    for N in (100, 3000):
        pf = PFCTRV(num_particles=N, seed=0); rng = pf.make_rng(1)
        m, P = ekf.initiate(meas[0]); X = pf.sample_initial(m, P, rng); logw = np.full(N, -np.log(N))
        me, Pe = ekf.initiate(meas[0]); ie = ip = False
        ee, ep, vv = [], [], []
        for t in range(1, T):
            me, Pe = ekf.predict(me, Pe)
            if not ie:
                me, Pe, ie = ekf.try_initiate_from_motion(me, Pe, meas[t], 1, (meas[t - 1][0], meas[t - 1][1]))
            me, Pe = ekf.update(me, Pe, meas[t])
            X = pf.propagate(X, rng); mp, Pp = pf.estimate_state(X, logw)
            if not ip:
                m2, _, ok = ekf.try_initiate_from_motion(mp, Pp, meas[t], 1, (meas[t - 1][0], meas[t - 1][1]))
                if ok:
                    X = pf.reseed_motion(X, m2[2], m2[3], 40.0, rng); logw = np.full(N, -np.log(N)); ip = True
            X, logw, mp, Pp = pf.update_particles(X, logw, meas[t], None, rng)
            if t > 10:
                ee.append(np.hypot(*(me[:2] - truth[t][:2]))); ep.append(np.hypot(*(mp[:2] - truth[t][:2])))
                vv.append((mp[2], np.degrees(mp[3])))
        vv = np.array(vv)
        res[N] = dict(rmse_ekf=float(np.sqrt(np.mean(np.square(ee)))), rmse_pf=float(np.sqrt(np.mean(np.square(ep)))),
                      v=float(vv[:, 0].mean()), th=float(vv[:, 1].mean()))
    r = res[3000]
    _check("N=3000 (duong thang, nhieu Gaussian): RMSE PF trong 1.3x EKF, v ~ 5, theta ~ 30 deg",
           r["rmse_pf"] < 1.3 * r["rmse_ekf"] + 0.2 and abs(r["v"] - 5) < 0.3 and abs(r["th"] - 30) < 3,
           f"EKF {r['rmse_ekf']:.2f} px | PF {r['rmse_pf']:.2f} px, v={r['v']:.2f}, theta={r['th']:.1f}")
    print(f"         [thong tin] N=100: RMSE PF {res[100]['rmse_pf']:.2f} px (EKF {res[100]['rmse_ekf']:.2f}), "
          f"v={res[100]['v']:.2f}, theta={res[100]['th']:.1f} -> N=100 suy bien (sample impoverishment), "
          f"khong phai loi cai dat")


def test_10_lifecycle_bookkeeping():
    def run(T):
        tr = make_tracker(load_tracker_cfg(PF_YAML if T is PFByteTracker else ROOT / "configs/bytetrack_ekf_ctrv.yaml"))
        if hasattr(tr, "set_seed"):
            tr.set_seed(0, "v")
        book = []
        seq = [boxes([[100 + 5 * f, 100, 150 + 5 * f, 140, 0.9, 2.0]]) if f not in (4, 5) else boxes(np.zeros((0, 6)))
               for f in range(1, 10)]
        for f, b in enumerate(seq, start=1):
            out = tr.update(b.cpu().numpy(), None)
            ts = tr.tracked_stracks + tr.lost_stracks
            for t in ts:
                book.append((f, t.track_id, t.state, t.is_activated, t.frame_id, t.start_frame, t.tracklet_len,
                             float(t.score), int(t.cls), int(t.idx)))
            if len(out):
                assert np.isfinite(out).all() and (out[:, 2] > out[:, 0]).all() and (out[:, 3] > out[:, 1]).all()
                assert np.isfinite(ts[0].mean).all() and np.isfinite(ts[0].covariance).all()
        return book, tr
    b_pf, tr_pf = run(PFByteTracker)
    b_ekf, _ = run(CTRVByteTracker)
    _check("bookkeeping (id, state, is_activated, frame_id, start_frame, tracklet_len, score, cls, idx) "
           "PF == EKF qua activate -> update -> lost (2 frame) -> re_activate", b_pf == b_ekf,
           f"pf={b_pf[-1]}\nekf={b_ekf[-1]}")
    t = (tr_pf.tracked_stracks + tr_pf.lost_stracks)[0]
    _check("track PF co particles (N,9), logw (N,), rng rieng, kalman_filter la PFTrackView",
           t.particles.shape == (100, 9) and t.logw.shape == (100,) and t.rng is not None
           and type(t.kalman_filter).__name__ == "PFTrackView")
    _check("track da qua re_activate va tracklet_len dung", t.state == TrackState.Tracked and t.tracklet_len >= 3)


def test_11_runner_seeds():
    """Cache tong hop nho -> run_tracking --seeds 0 1 -> 2 thu muc; chay lai seed 0 bit-identical."""
    cid = "fixture_pf_seed_test"
    cdir = det_cache.CACHE_ROOT / cid
    if cdir.exists():
        shutil.rmtree(cdir)
    det_cache.write_meta(cdir, det_cache.DetectorSpec(backend="gt-synthetic", conf=0.05, classes=None,
                                                      max_det=3000, dataset="detrac", part="sample"))
    F = 40
    data = np.concatenate([scene(f).data for f in range(1, F + 1)])
    arrays = dict(frame_idx=np.arange(1, F + 1, dtype=np.int32),
                  n_per_frame=np.array([len(scene(f)) for f in range(1, F + 1)], np.int32),
                  unreadable=np.zeros(F, bool), truncated=np.zeros(F, bool), time_s=np.arange(F) / 25.0,
                  data=data.astype(np.float32), orig_shape=np.array([540, 960], np.int32))
    det_cache.save_video(cdir, "synth", arrays)
    out_root = config.PROCESSED_DIR / "trackers" / "DETRAC-sample"
    cmd = [sys.executable, "src/run_tracking.py", "--cache-id", cid, "--split-name", "DETRAC-sample",
           "--tracker-yaml", str(PF_YAML), "--tracker-name", "p2-pftest", "--videos", "synth", "--seeds", "0", "1"]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    _check("run_tracking PF --seeds 0 1 chay xong", r.returncode == 0, r.stdout[-500:] + r.stderr[-500:])
    f0 = out_root / "p2-pftest-s0" / "data" / "synth.txt"; f1 = out_root / "p2-pftest-s1" / "data" / "synth.txt"
    _check("tao 2 thu muc -s0 va -s1", f0.exists() and f1.exists())
    t0 = f0.read_text()
    r = subprocess.run(cmd[:-2] + ["0"], cwd=ROOT, capture_output=True, text=True)
    _check("chay lai seed 0: file txt bit-identical", r.returncode == 0 and f0.read_text() == t0)
    _check("seed 1 khac seed 0", f1.read_text() != t0)
    meta = pd.read_json(out_root / "p2-pftest-s1" / "run_meta.json", typ="series")
    _check("run_meta ghi seed", int(meta["seed"]) == 1)


def main() -> int:
    tracker_ctrv.register()
    failed = 0
    for t in [test_1_f_batch_matches_f, test_2_omega_near_zero, test_3_circular_mean, test_4_missing_measurements,
              test_5_degenerate_weights, test_6_seed_reproducibility, test_7_pool_order_independence,
              test_8_resampling, test_9_consistency_with_ekf, test_10_lifecycle_bookkeeping, test_11_runner_seeds]:
        print(f"\n--- {t.__name__} ---")
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"  !! {e}")
    n_ok = sum(ok for _, ok in _RESULTS)
    print("\n" + "=" * 78)
    print(f"KET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
