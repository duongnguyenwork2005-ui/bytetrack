"""
test_quan_sat.py -- (R65) fixture cho cong "log/quan sat khong doi hanh vi" o cac nhanh du lieu that co the khong
kich hoat du: suy bien (reset), resample, Lost/refind. Khong co pytest trong env -> bo chay rieng (nhu R64).

    python -B src/r65_pf_roundabout/test_quan_sat.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pf_ctrv  # noqa: E402
import quan_sat  # noqa: E402
import run_tracking as rt  # noqa: E402

PF_YAML = "configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml"


def _pf(diag: bool) -> pf_ctrv.PFCTRV:
    pf = pf_ctrv.PFCTRV(num_particles=500, seed=0)
    pf.set_seed(0, "fixture")
    if diag:
        pf.enable_diag()
    return pf


def _chuoi_cap_nhat(diag: bool, observer: bool):
    """predict/update tren mot track gia: quan sat binh thuong, roi nhay xa (suy bien), roi binh thuong."""
    if observer:
        quan_sat.cai_dat(None)
        quan_sat._bat_dau_video("fixture", 0)
    try:
        pf = _pf(diag)
        rng = pf.make_rng(1)
        mean = np.array([100.0, 100.0, 2.0, 0.0, 0.0, 1.5, 40.0, 0.0, 0.0])   # cx cy v theta omega a h va vh
        cov = np.eye(pf.NDIM) * 4.0
        X = pf.sample_initial(mean, cov, rng)
        logw = np.full(len(X), -np.log(len(X)))
        zs = [(102, 100), (104, 100), (106, 100), (2000, 1500), (2002, 1500), (2004, 1500)]
        ket = []
        for k, (cx, cy) in enumerate(zs):
            X = pf.propagate(X, rng)
            z = np.array([cx, cy, 1.5, 40.0])
            X, logw, m, P = pf.update_particles(X, logw, z, 0.9, rng, diag_key=(1, k + 1, 1))
            ket.append((X.copy(), logw.copy(), m.copy(), P.copy()))
        dem = dict(quan_sat._CUR.get("dem", {}))
        return ket, rng.bit_generator.state, pf.n_degenerate, pf.n_resample, dem
    finally:
        if observer:
            quan_sat._CUR.clear()
            quan_sat.go_bo()


def test_suy_bien_resample_khong_doi():
    goc = _chuoi_cap_nhat(False, False)
    for diag, obs in ((True, False), (False, True), (True, True)):
        kq = _chuoi_cap_nhat(diag, obs)
        for a, b in zip(goc[0], kq[0]):
            for x, y in zip(a, b):
                assert np.array_equal(x, y), (diag, obs)
        assert goc[1] == kq[1], "trang thai rng lech"
        assert goc[2:4] == kq[2:4]
    assert goc[2] >= 1, "fixture phai kich hoat suy bien"
    assert goc[3] >= 1, "fixture phai kich hoat resample"
    dem = _chuoi_cap_nhat(False, True)[4]
    assert dem.get("pf_degenerate", 0) == goc[2] and dem.get("pf_resample", 0) == goc[3], dem


class _VD:
    """VideoDetections gia: 2 xe, xe 1 mat detection frame 8-12 (Lost -> refind)."""
    orig_shape = (1080, 1920)

    def __len__(self):
        return 30

    def iter_frames(self):
        for f in range(1, 31):
            rows = []
            if not 8 <= f <= 12:
                x = 100 + 6 * f
                rows.append([x, 300, x + 60, 340, 0.9, 2])
            x2 = 900 - 4 * f
            rows.append([x2, 500, x2 + 80, 550, 0.85, 2])
            yield f, (f - 1) / 15.0, False, np.array(rows, dtype=np.float32)


def _chay(yaml: str, seed, observer: bool, assoc: bool):
    if observer:
        quan_sat.cai_dat(None)
    try:
        cfg = rt.load_tracker_cfg(yaml, {"track_buffer": 18})
        sink = {"assoc_log": assoc, "pf_diag": assoc}
        df, dfc, _ = rt.track_video(_VD(), cfg, 1 / 15.0, "steps", seed, "fixture", log_sink=sink)
        return df, dfc, sink
    finally:
        if observer:
            quan_sat.go_bo()


def test_tracker_lost_refind_khong_doi():
    for yaml, seed in (("bytetrack.yaml", None), (PF_YAML, 0), (PF_YAML, 1)):
        goc = _chay(yaml, seed, False, False)
        for obs, assoc in ((True, False), (False, True), (True, True)):
            kq = _chay(yaml, seed, obs, assoc)
            assert goc[0].equals(kq[0]) and goc[1].equals(kq[1]), (yaml, seed, obs, assoc)
        # observer co chay toi ranh gioi sau du doan (multi_predict) va ghi do trai prior cho PF
        quan_sat.cai_dat(None)
        try:
            ghi = {}
            goc_bd = quan_sat._bat_dau_video

            def bd(video, seed_):
                goc_bd(video, seed_)
                ghi["cur"] = quan_sat._CUR
            quan_sat._bat_dau_video = bd
            cfg = rt.load_tracker_cfg(yaml, {"track_buffer": 18})
            rt.track_video(_VD(), cfg, 1 / 15.0, "steps", seed, "fixture")
        finally:
            quan_sat._bat_dau_video = goc_bd
            quan_sat.go_bo()
        ev = _chay(yaml, seed, False, True)[2]["assoc_log"].frames()[2]
        assert (ev["kind"] == "lost").any() and (ev["kind"] == "refind").any(), (yaml, ev["kind"].tolist())
    assert rt.track_video.__name__ == "track_video" and not hasattr(rt.track_video, "__wrapped__"), "go_bo loi"


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    loi = 0
    for t in tests:
        try:
            t(); print("PASS", t.__name__)
        except Exception:
            loi += 1; print("FAIL", t.__name__); traceback.print_exc()
    print(f"{len(tests) - loi}/{len(tests)} dat")
    return 1 if loi else 0


if __name__ == "__main__":
    raise SystemExit(main())
