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


def _anh_chup(tr) -> str:
    """Bam trang thai tracker DOC LAP voi observer (de so cac to hop observer/log): moi track Tracked/Lost theo id:
    state, mean, covariance; PF them particles, logw, trang thai RNG. Kiem huu han."""
    import hashlib
    import json
    h = hashlib.sha256()
    for nhan, pool in (("T", tr.tracked_stracks), ("L", tr.lost_stracks)):
        for st in sorted(pool, key=lambda x: int(x.track_id)):
            h.update(f"{nhan}{int(st.track_id)}:{int(st.state)}".encode())
            mang = [st.mean, st.covariance]
            if getattr(st, "particles", None) is not None:
                mang += [st.particles, st.logw]
                h.update(json.dumps(st.rng.bit_generator.state, sort_keys=True, default=int).encode())
            for a in mang:
                a = np.ascontiguousarray(np.asarray(a, dtype=float))
                assert np.isfinite(a).all(), f"track {st.track_id} khong huu han"
                h.update(a.tobytes())
    return h.hexdigest()


def _chay(yaml: str, seed, observer: bool, assoc: bool, trace_dir: Path | None = None):
    """Chay fixture; tra ve (df, df_cls, sink, danh sach hash trang thai TUNG FRAME)."""
    if observer:
        quan_sat._TEN_RUN["ten"] = "fixture"
        quan_sat.cai_dat(trace_dir)
    goc_make = rt.make_tracker
    bat = []

    def make_tracker(*a, **k):            # boc NGOAI CUNG (sau observer) chi de lay tham chieu tracker
        tr = goc_make(*a, **k)
        bat.append(tr)
        return tr
    rt.make_tracker = make_tracker
    anh = []
    try:
        cfg = rt.load_tracker_cfg(yaml, {"track_buffer": 18})
        sink = {"assoc_log": assoc, "pf_diag": assoc}
        df, dfc, _ = rt.track_video(_VD(), cfg, 1 / 15.0, "steps", seed, "fixture", log_sink=sink,
                                    on_frame=lambda f, out: anh.append(_anh_chup(bat[-1])))
        return df, dfc, sink, anh
    finally:
        rt.make_tracker = goc_make
        if observer:
            quan_sat.go_bo()


def test_tracker_lost_refind_khong_doi():
    """Output VA trang thai tung frame (particles/logw/mean/cov/RNG) trung giua moi to hop observer x log."""
    for yaml, seed in (("bytetrack.yaml", None), (PF_YAML, 0), (PF_YAML, 1)):
        goc = _chay(yaml, seed, False, False)
        assert len(goc[3]) == 30
        for obs, assoc in ((True, False), (False, True), (True, True)):
            kq = _chay(yaml, seed, obs, assoc)
            assert goc[0].equals(kq[0]) and goc[1].equals(kq[1]), (yaml, seed, obs, assoc)
            assert goc[3] == kq[3], ("trang thai tung frame lech", yaml, seed, obs, assoc)
        ev = _chay(yaml, seed, False, True)[2]["assoc_log"].frames()[2]
        assert (ev["kind"] == "lost").any() and (ev["kind"] == "refind").any(), (yaml, ev["kind"].tolist())
    assert rt.track_video.__name__ == "track_video" and not hasattr(rt.track_video, "__wrapped__"), "go_bo loi"
    assert not hasattr(rt.make_tracker, "__wrapped__"), "go_bo loi make_tracker"


def test_trace_ghi_ra_file_va_cong():
    """Observer ghi trace ra file: du frame, hook du doan chay, PF co prior/update khop bo dem -> cong kiem_trace dat."""
    import json
    import tempfile
    import chan_doan as cd
    for yaml, seed, loc in (("bytetrack.yaml", None, "cv"), (PF_YAML, 0, "pf500")):
        with tempfile.TemporaryDirectory() as td:
            _chay(yaml, seed, True, False, Path(td))
            d = Path(td) / ("fixture" + (f"-s{seed}" if seed is not None else ""))
            tr = json.loads((d / "fixture.json").read_text(encoding="utf-8"))
            assert tr["dem"]["frame"] == 30 and tr["dem"]["du_doan_track"] > 0, tr["dem"]
            n_pr = n_up = None
            if loc == "pf500":
                pr = pd.read_csv(d / "fixture_pf_prior.csv"); up = pd.read_csv(d / "fixture_pf_update.csv")
                n_pr, n_up = len(pr), len(up)
                assert n_pr > 0 and n_up > 0
                assert (up["state"] == 2).any(), "fixture phai co update tu Lost"
            assert cd.kiem_trace(tr, loc, n_pr, n_up, True, n_frame=30) == [], cd.kiem_trace(tr, loc, n_pr, n_up, True, 30)


def test_cong_bat_loi():
    """Test am: cong kiem_trace phai bat trace thieu frame, hook du doan khong chay, thieu chuoi hash, so dong lech,
    gia tri khong huu han."""
    import chan_doan as cd
    tot = {"dem": {"frame": 30, "du_doan_track": 50, "pf_update_particles": 40}, "chuoi_frame_sha256": "a",
           "chuoi_pf_sha256": "b", "chuoi_du_doan_sha256": "c"}
    assert cd.kiem_trace(tot, "pf500", 50, 40, True, n_frame=30) == []
    hong = [({**tot, "dem": {**tot["dem"], "frame": 29}}, 50, 40, True),
            ({**tot, "dem": {**tot["dem"], "du_doan_track": 0}}, 0, 40, True),
            ({k: v for k, v in tot.items() if k != "chuoi_du_doan_sha256"}, 50, 40, True),
            (tot, 49, 40, True), (tot, 50, 41, True), (tot, 50, 40, False)]
    for t, a, b, hh in hong:
        assert cd.kiem_trace(t, "pf500", a, b, hh, n_frame=30), (t, a, b, hh)


def test_cong_hoan_tat_bat_loi():
    """Test am (review GPT lan 3): kiem_hoan_tat_run / xong phai tu choi run_meta dang chay, thieu fingerprint (ca hai
    phia cung thieu), video_meta chua hoan tat, fingerprint lech, trace thieu/hong; dat khi day du."""
    import json
    import tempfile
    import chan_doan as cd
    vids = ["v1", "v2"]

    def dung(goc: Path, status="complete", fp_run="abc", vm=None, trace=True):
        d = goc / "trackers" / cd.SPLIT / cd.ten("off", "cv")
        (d / "data").mkdir(parents=True, exist_ok=True)
        (d / "video_meta").mkdir(exist_ok=True)
        rm = {"status": status}
        if fp_run is not None:
            rm["input_fingerprint"] = fp_run
        (d / "run_meta.json").write_text(json.dumps(rm), encoding="utf-8")
        for v in vids:
            (d / "data" / f"{v}.txt").write_text("", encoding="utf-8")
            m = {"video": v, "complete": True, "input_fingerprint": fp_run} if vm is None else vm
            (d / "video_meta" / f"{v}.json").write_text(json.dumps(m), encoding="utf-8")
            if trace:
                t = goc / "trace" / cd.ten("off", "cv")
                t.mkdir(parents=True, exist_ok=True)
                (t / f"{v}.json").write_text(json.dumps({"video": v}), encoding="utf-8")
        return d

    goc_proc, goc_trace = cd.config.PROCESSED_DIR, cd.TRACE
    try:
        for ca, kw, ky_vong_dat in (
                ("day du", {}, True),
                ("run_meta dang chay", {"status": "running"}, False),
                ("hai phia thieu fingerprint", {"fp_run": None, "vm": {"complete": True}}, False),
                ("fingerprint rong", {"fp_run": ""}, False),
                ("video_meta chua hoan tat", {"vm": {"complete": False, "input_fingerprint": "abc"}}, False),
                ("fingerprint lech", {"vm": {"complete": True, "input_fingerprint": "khac"}}, False),
                ("thieu trace", {"trace": False}, False)):
            with tempfile.TemporaryDirectory() as td:
                goc = Path(td)
                cd.config.PROCESSED_DIR, cd.TRACE = goc, goc / "trace"
                d = dung(goc, **kw)
                assert (cd.kiem_hoan_tat_run(d, vids) == []) == (ky_vong_dat or ca == "thieu trace"), ca
                assert cd.xong("off", "cv", vids) is ky_vong_dat, ca
        with tempfile.TemporaryDirectory() as td:          # trace hong (khong phai JSON)
            goc = Path(td)
            cd.config.PROCESSED_DIR, cd.TRACE = goc, goc / "trace"
            dung(goc)
            (goc / "trace" / cd.ten("off", "cv") / "v1.json").write_text("{hong", encoding="utf-8")
            assert cd.xong("off", "cv", vids) is False
    finally:
        cd.config.PROCESSED_DIR, cd.TRACE = goc_proc, goc_trace


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
