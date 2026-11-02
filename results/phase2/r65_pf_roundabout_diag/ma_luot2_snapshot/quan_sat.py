"""
quan_sat.py -- (R65, chan doan PF RoundaboutHD) chay `src/run_tracking.py` voi BO QUAN SAT CHI DOC.

Quyet dinh GPT 02/11 (`D:/LABOTARY/automation/runs_local/hoi_codex_R64_viec_tiep_20261102/answer.json` Q2): cong
"log khong doi hanh vi" phai bam (hash) trang thai RNG tung track va particles/logw/mean/covariance tai cac ranh gioi
predict/update/reseed/resample bang observer CHI DOC o CA HAI nhanh (log tat / log bat), khong rut so ngau nhien.

Bo quan sat (giong nhau o hai nhanh; nhanh chi khac co --assoc-log --pf-diag cua run_tracking):
    - moi frame (sau tracker.update): chuoi sha256 noi tiep tren MOI track tracked/lost/removed-moi:
      track_id, state, frame_id, mean, covariance; track PF them particles, logw, rng.bit_generator.state;
    - PF tung buoc: propagate / update_particles / reseed_motion / sample_initial -> chuoi sha256 rieng
      (ten buoc + trang thai rng sau buoc + mang dau ra);
    - PF tung update (PFTrackView.update): track_id, frame hien tai, khoang cach toi lan update truoc (gap),
      resampled, so to tien khac nhau tu `last_resample_idx` (hook chi doc R48D), suy bien (n_degenerate tang).
Doc `rng.bit_generator.state` va `ndarray.tobytes()` khong doi trang thai. Dau ra ghi NGUYEN TU theo video sau khi
`track_video` tra ve: <trace-dir>/<tracker-name>[-sN]/<video>.json + <video>_pf_update.csv (chi PF).

    python -B src/r65_pf_roundabout/quan_sat.py --trace-dir <thu muc> -- <tham so run_tracking.py>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "src"))

_CUR: dict = {}            # trang thai quan sat cua video dang chay


def _bam_mang(h, *arrs) -> None:
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(f"{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())


def _bam_rng(h, rng) -> None:
    h.update(json.dumps(rng.bit_generator.state, sort_keys=True, default=int).encode())


def _bat_dau_video(video: str, seed) -> None:
    _CUR.clear()
    _CUR.update(video=video, seed=seed, h_frame=hashlib.sha256(), h_pf=hashlib.sha256(), h_du_doan=hashlib.sha256(),
                dem=Counter(), pf_update=[], pf_prior=[])


def _ghi_video(trace_dir: Path, ten_run: str) -> None:
    d = trace_dir / ten_run
    d.mkdir(parents=True, exist_ok=True)
    v = _CUR["video"]
    out = {"video": v, "seed": _CUR["seed"], "chuoi_frame_sha256": _CUR["h_frame"].hexdigest(),
           "chuoi_pf_sha256": _CUR["h_pf"].hexdigest(), "chuoi_du_doan_sha256": _CUR["h_du_doan"].hexdigest(),
           "dem": dict(sorted(_CUR["dem"].items()))}
    import pandas as pd
    for khoa, cols in (("pf_update", ["track_id", "frame", "frame_id_truoc", "gap", "resampled", "n_to_tien",
                                      "degenerate", "state"]),
                       ("pf_prior", ["track_id", "frame", "state", "h_du_doan", "spread_xy", "spread_theta",
                                     "spread_v"])):
        if _CUR[khoa]:
            tmp = d / f"{v}_{khoa}.csv.tmp"
            pd.DataFrame(_CUR[khoa], columns=cols).to_csv(tmp, index=False)
            os.replace(tmp, d / f"{v}_{khoa}.csv")
    tmp = d / f"{v}.json.tmp"
    tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
    os.replace(tmp, d / f"{v}.json")


def cai_dat(trace_dir: Path | None) -> None:
    """Gan cac hook chi doc. trace_dir None: chi tinh (dung cho test), khong ghi file."""
    import pf_ctrv
    import run_tracking as rt

    pf_cls, view_cls = pf_ctrv.PFCTRV, pf_ctrv.PFTrackView

    def boc_pf(ten):
        goc = getattr(pf_cls, ten)

        def f(self, *a, **k):
            rng = next((x for x in list(a) + list(k.values()) if isinstance(x, np.random.Generator)), None)
            n_deg = self.n_degenerate
            kq = goc(self, *a, **k)
            if _CUR:
                h = _CUR["h_pf"]
                h.update(ten.encode())
                if rng is not None:
                    _bam_rng(h, rng)
                for x in (kq if isinstance(kq, tuple) else (kq,)):
                    if isinstance(x, np.ndarray):
                        _bam_mang(h, x)
                    elif isinstance(x, tuple):
                        _bam_mang(h, *[y for y in x if isinstance(y, np.ndarray)])
                _CUR["dem"][f"pf_{ten}"] += 1
                if ten == "update_particles":
                    if self.n_degenerate > n_deg:
                        _CUR["dem"]["pf_degenerate"] += 1
                    if self.last_resample_idx is not None:
                        _CUR["dem"]["pf_resample"] += 1
            return kq
        f.__wrapped__ = goc
        setattr(pf_cls, ten, f)

    for ten in ("propagate", "update_particles", "reseed_motion", "sample_initial"):
        boc_pf(ten)

    goc_view_update = view_cls.update

    def view_update(self, mean, covariance, measurement, confidence=None):
        t = self.track
        fid = int(getattr(t, "_pending_frame_id", getattr(t, "frame_id", -1)))
        fid_truoc = int(t.frame_id) if t.frame_id is not None else -1
        state = int(t.state)
        n_deg = self.pf.n_degenerate
        kq = goc_view_update(self, mean, covariance, measurement, confidence)
        if _CUR:
            idx = self.pf.last_resample_idx
            _CUR["pf_update"].append([int(t.track_id), fid, fid_truoc, fid - fid_truoc, idx is not None,
                                      int(len(np.unique(idx))) if idx is not None else -1,
                                      self.pf.n_degenerate > n_deg, state])
        return kq
    view_update.__wrapped__ = goc_view_update
    view_cls.update = view_update

    goc_make = rt.make_tracker

    def make_tracker(*a, **k):
        tr = goc_make(*a, **k)
        goc_update = tr.update
        goc_mp = tr.multi_predict

        def multi_predict(tracks):
            # ranh gioi SAU du doan, TRUOC association (moi bo loc; PF du doan qua PFSTrack.multi_predict vector hoa,
            # khong qua PFCTRV.propagate)
            kq = goc_mp(tracks)
            if _CUR:
                h = _CUR["h_du_doan"]
                h.update(f"P{int(tr.frame_id)}".encode())
                for s in sorted(tracks, key=lambda s: int(s.track_id)):
                    h.update(f"{int(s.track_id)}:{int(s.state)}".encode())
                    _bam_mang(h, s.mean, s.covariance)
                    if getattr(s, "particles", None) is not None:
                        _bam_mang(h, s.particles, s.logw)
                        _bam_rng(h, s.rng)
                        pf = s.kalman_filter.pf
                        sxy, sth, sv = pf._diag_spread(s.particles, s.logw)
                        _CUR["pf_prior"].append([int(s.track_id), int(tr.log_frame), int(s.state),
                                                 float(np.asarray(s.tlwh)[3]), sxy, sth, sv])
                _CUR["dem"]["du_doan_track"] += len(tracks)
            return kq
        tr.multi_predict = multi_predict

        def update(*ua, **uk):
            out = goc_update(*ua, **uk)
            if _CUR:
                h = _CUR["h_frame"]
                h.update(str(int(tr.frame_id)).encode())
                pools = (("T", tr.tracked_stracks), ("L", tr.lost_stracks))
                for nhan, pool in pools:
                    for s in sorted(pool, key=lambda s: int(s.track_id)):
                        h.update(f"{nhan}{int(s.track_id)}:{int(s.state)}:{s.frame_id}".encode())
                        _bam_mang(h, s.mean, s.covariance)
                        if getattr(s, "particles", None) is not None:
                            _bam_mang(h, s.particles, s.logw)
                            _bam_rng(h, s.rng)
                _CUR["dem"]["frame"] += 1
                _CUR["dem"]["track_frame_T"] += len(tr.tracked_stracks)
                _CUR["dem"]["track_frame_L"] += len(tr.lost_stracks)
            return out
        tr.update = update
        return tr
    make_tracker.__wrapped__ = goc_make
    rt.make_tracker = make_tracker

    goc_track_video = rt.track_video

    def track_video(vd, cfg, dt_nominal, elapsed_mode="skip", seed=None, video="", *a, **k):
        _bat_dau_video(video, seed)
        try:
            kq = goc_track_video(vd, cfg, dt_nominal, elapsed_mode, seed, video, *a, **k)
            if trace_dir is not None:
                _ghi_video(trace_dir, _TEN_RUN["ten"] + (f"-s{seed}" if seed is not None else ""))
            return kq
        finally:
            _CUR.clear()
    track_video.__wrapped__ = goc_track_video
    rt.track_video = track_video


def go_bo() -> None:
    """Go hook (test)."""
    import pf_ctrv
    import run_tracking as rt
    for ten in ("propagate", "update_particles", "reseed_motion", "sample_initial"):
        f = getattr(pf_ctrv.PFCTRV, ten)
        if hasattr(f, "__wrapped__"):
            setattr(pf_ctrv.PFCTRV, ten, f.__wrapped__)
    for obj, ten in ((pf_ctrv.PFTrackView, "update"), (rt, "make_tracker"), (rt, "track_video")):
        f = getattr(obj, ten)
        if hasattr(f, "__wrapped__"):
            setattr(obj, ten, f.__wrapped__)


_TEN_RUN = {"ten": ""}


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv or argv[0] != "--trace-dir":
        print(__doc__); return 2
    trace_dir = Path(argv[1])
    rt_argv = argv[argv.index("--") + 1:]
    _TEN_RUN["ten"] = rt_argv[rt_argv.index("--tracker-name") + 1]
    cai_dat(trace_dir)
    import run_tracking as rt
    sys.argv = [str(GOC / "src" / "run_tracking.py"), *rt_argv]
    return rt.main()


if __name__ == "__main__":
    raise SystemExit(main())
