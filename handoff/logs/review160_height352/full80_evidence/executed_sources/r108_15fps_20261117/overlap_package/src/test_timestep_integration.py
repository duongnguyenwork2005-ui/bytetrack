"""
test_timestep_integration.py -- (MOC 2, muc 1) Chung minh TIMESTEP di tu TIMESTAMP cua cache toi
phep DU DOAN va VONG DOI track tren DUONG CHAY DAY DU (det_cache -> run_tracking.track_video ->
ElapsedStepsMixin.update(n_steps) -> multi_predict k lan -> _remove_stale_lost), cho CA BON bo loc
CV / EKF / UKF / PF. Kiem thu rieng time_units.py (test_time_units.py) chi kiem tra mixin; test nay
kiem tra TICH HOP: so buoc tinh tu `time_s`, box du doan sau khoang nhay == predict k lan thu cong,
track LOST bi xoa theo TONG SO BUOC (khac che do skip), frame rong (0 detection) va frame bi bo qua
(unreadable) duoc phan biet.

Cache tong hop 20 frame @ 25 fps:
    - time_s nhay 3 buoc giua frame 8 va 9 (timestamp thuc, KHONG phai unreadable)  -> n_steps = 3
    - frame 12 unreadable (bo qua)                                                   -> frame 13: n_steps = 2
    - frame 15 rong (0 detection)                                                    -> n_steps = 1, track -> LOST
    - xe A di thang 5 px/buoc (toa do theo THOI GIAN, nen du doan k buoc phai khop)
    - xe B chi co frame 1-5 -> LOST tu frame 6; track_buffer = 6 buoc:
          steps: xoa khi frame_id - 5 > 6, frame_id(f10) = 12  -> xoa tai frame 10
          skip : frame_id(f10) = 10, (f13) = 12                 -> xoa tai frame 13

    python src/test_timestep_integration.py
"""
from __future__ import annotations

import copy
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_cache  # noqa: E402
import run_tracking  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_RESULTS: list[tuple[str, bool]] = []
DT = 0.04


def _check(name, ok, detail=""):
    _RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")
    assert ok, name


def make_cache(tmpdir: str):
    F = 20
    t = np.zeros(F)
    for i in range(1, F):
        t[i] = t[i - 1] + (3 * DT if i == 8 else DT)          # frame 9 (index 8) cach frame 8 ba buoc
    unreadable = np.zeros(F, bool); unreadable[11] = True     # frame 12
    empty = {15}
    data, n_per = [], []
    for f in range(1, F + 1):
        rows = []
        if not unreadable[f - 1] and f not in empty:
            step = t[f - 1] / DT
            xa = 100 + 5.0 * step
            rows.append([xa, 100, xa + 50, 140, 0.9, 2.0])                  # A
            if f <= 5:
                rows.append([500 + 2.0 * step, 300, 560 + 2.0 * step, 340, 0.8, 2.0])   # B
        data.extend(rows); n_per.append(len(rows))
    arrays = dict(frame_idx=np.arange(1, F + 1, dtype=np.int32), n_per_frame=np.array(n_per, np.int32),
                  unreadable=unreadable, truncated=np.zeros(F, bool), time_s=t,
                  data=np.array(data, np.float32).reshape(-1, 6), orig_shape=np.array([540, 960], np.int32))
    det_cache.save_video(Path(tmpdir), "v", arrays)
    return det_cache.VideoDetections(Path(tmpdir) / "v.npz", 0.25, [2], 300)


FILTERS = [("cv", "bytetrack.yaml", "bytetrack"),
           ("ekf", "configs/bytetrack_ekf_ctrv.yaml", None),
           ("ukf", "configs/bytetrack_ukf_ctrv.yaml", None),
           ("pf", "configs/phase2/trackers/bytetrack_pf_ctrv_n100.yaml", None)]
EXPECTED_STEPS = {f: 1 for f in range(1, 21)}
EXPECTED_STEPS[9] = 3; EXPECTED_STEPS[13] = 2
del EXPECTED_STEPS[12]


def run_one(name, yaml, ttype):
    from assoc_log import with_assoc_log  # noqa: F401  (kiem tra import)
    cfg = run_tracking.load_tracker_cfg(yaml if ttype else ROOT / yaml, {"track_buffer": 6})
    if ttype:
        cfg.tracker_type = ttype
    with tempfile.TemporaryDirectory() as td:
        vd = make_cache(td)
        sink = {"assoc_log": True}
        df, _, tm = run_tracking.track_video(vd, cfg, DT, elapsed_mode="steps", seed=0, video="v", log_sink=sink)
        tracks, dets, events = sink["assoc_log"].frames()
        # (1) n_steps tinh tu time_s cua cache, frame unreadable khong co dong
        got = dict(zip(tm.frame.astype(int), tm.n_steps.astype(int)))
        _check(f"[{name}] n_steps theo timestamp: f9 = 3 (nhay 3 buoc), f13 = 2 (f12 unreadable), con lai 1",
               got == EXPECTED_STEPS, f"got={got}")
        # tracker.frame_id cuoi = tong so buoc = 22
        _check(f"[{name}] tracker.frame_id cuoi = 22 (tong so buoc), khong phai 19 (so frame xu ly)",
               int(sink["tracker"].frame_id) == 22, f"frame_id={sink['tracker'].frame_id}")
        # (2) box DU DOAN tai frame 9 == predict 3 lan thu cong tren cung trang thai sau frame 8
        ida = int(df[df.frame == 1].sort_values("bb_left").id.iloc[0])            # A = xe o trai (x=100)
        idb = int(df[df.frame == 1].sort_values("bb_left").id.iloc[1])
        logged = tracks[(tracks.frame == 9) & (tracks.track_id == ida)][["x", "y", "w", "h"]].to_numpy()[0]
        tr2 = run_tracking.make_tracker(cfg)
        if hasattr(tr2, "set_seed"):
            tr2.set_seed(0, "v")
        for f, _t, unr, arr in vd.iter_frames():
            if f > 8:
                break
            tr2.update(det_cache.to_boxes(arr, vd.orig_shape).cpu().numpy(), None, n_steps=1)
        pool = copy.deepcopy(tr2.tracked_stracks + tr2.lost_stracks)   # deepcopy CA danh sach (chung bo loc)
        for _ in range(3):
            tr2.multi_predict(pool)                                     # ElapsedSteps ngoai update: 1 buoc/lan
        manual = [np.asarray(t.tlwh, float) for t in pool if t.track_id == ida][0]
        tol = 0.0 if name == "pf" else 1e-9
        _check(f"[{name}] box du doan tai f9 (log) == predict 3 lan thu cong sau f8 (tol {tol})",
               np.allclose(logged, manual, atol=tol, rtol=0), f"log={logged}\nmanual={manual}")
        one = [np.asarray(t.tlwh, float) for t in copy.deepcopy(tr2.tracked_stracks + tr2.lost_stracks)
               if t.track_id == ida][0]
        _check(f"[{name}] du doan 3 buoc khac trang thai truoc du doan (thoi gian da troi qua)",
               not np.allclose(logged, one, atol=1e-6))
        # (3) vong doi: B bi xoa theo TONG SO BUOC -> tai frame 10 (frame_id 12 > 5 + 6)
        rm = events[(events.kind == "removed") & (events.track_id == idb)]
        _check(f"[{name}] track B LOST bi xoa tai frame 10 (steps: frame_id 12 - end_frame 5 > track_buffer 6)",
               len(rm) == 1 and int(rm.frame.iloc[0]) == 10 and int(rm.n_lost.iloc[0]) == 7,
               rm.to_string())
        # (4) frame rong 15: n_steps = 1, A -> LOST roi khop lai cung ID o frame 16
        lost15 = events[(events.kind == "lost") & (events.frame == 15) & (events.track_id == ida)]
        refind16 = events[(events.kind == "refind") & (events.frame == 16) & (events.track_id == ida)]
        _check(f"[{name}] frame rong 15: A mark_lost tai 15, refind tai 16 cung ID, khong co output frame 15",
               len(lost15) == 1 and len(refind16) == 1 and 15 not in set(df.frame)
               and set(df[df.frame.isin([14, 16])].id) == {ida})
        # (5) che do skip: B bi xoa MUON hon (frame 13), du doan tai f9 CHI 1 buoc
        sink2 = {"assoc_log": True}
        df_s, _, tm_s = run_tracking.track_video(vd, cfg, DT, elapsed_mode="skip", seed=0, video="v", log_sink=sink2)
        tr_s, _, ev_s = sink2["assoc_log"].frames()
        idb_s = int(df_s[df_s.frame == 1].sort_values("bb_left").id.iloc[1])
        ida_s = int(df_s[df_s.frame == 1].sort_values("bb_left").id.iloc[0])
        rm_s = ev_s[(ev_s.kind == "removed") & (ev_s.track_id == idb_s)]
        logged_s = tr_s[(tr_s.frame == 9) & (tr_s.track_id == ida_s)][["x", "y", "w", "h"]].to_numpy()[0]
        _check(f"[{name}] skip: moi n_steps = 1, B xoa tai frame 13, du doan f9 khac che do steps",
               (tm_s.n_steps == 1).all() and len(rm_s) == 1 and int(rm_s.frame.iloc[0]) == 13
               and not np.allclose(logged_s, logged, atol=1e-6), rm_s.to_string())
        # (6) log khong doi hanh vi: output co log == khong log (bit-identical)
        df_nolog, _, _ = run_tracking.track_video(vd, cfg, DT, elapsed_mode="steps", seed=0, video="v")
        _check(f"[{name}] assoc_log khong doi ket qua tracking (bit-identical)",
               df_nolog.shape == df.shape and np.array_equal(df_nolog.to_numpy(), df.to_numpy()))


def main() -> int:
    import tracker_ctrv
    tracker_ctrv.register()
    failed = 0
    for name, yaml, ttype in FILTERS:
        print(f"\n--- {name} ---")
        try:
            run_one(name, yaml, ttype)
        except AssertionError as e:
            failed += 1; print(f"  !! {e}")
    n_ok = sum(ok for _, ok in _RESULTS)
    print("\n" + "=" * 78 + f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS\n" + "=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
