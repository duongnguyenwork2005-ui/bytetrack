"""
test_lost_recovery.py -- (MOC 2, C2) LostRecoveryMixin: tat -> BIT-IDENTICAL tracker goc tren cache that
(MVI_39031, 300 frame, CV + EKF); bat -> co tac dung tren kich ban tong hop (track LOST duoc noi lai o vong 3
voi IoU < 0.2 hoac trong gate Mahalanobis; lost_no_fuse cuu det conf thap); assoc_log ghi stage 4.

    python src/test_lost_recovery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_cache  # noqa: E402
import run_tracking  # noqa: E402
import tracker_ctrv  # noqa: E402
from ultralytics.engine.results import Boxes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_RESULTS = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")
    assert ok, name


def run_cfg(yaml, overrides, vd):
    cfg = run_tracking.load_tracker_cfg(yaml, overrides)
    sink = {"assoc_log": True}
    df, _, _ = run_tracking.track_video(vd, cfg, 1 / 25.0, elapsed_mode="skip", log_sink=sink)
    return df, sink


def scene(frame, gap=(10, 16), jump=60.0, conf_after=0.9):
    """Xe di thang 8 px/frame; bien mat trong `gap`; khi xuat hien lai bi lech `jump` px (du doan lech)."""
    rows = []
    if not (gap[0] <= frame <= gap[1]):
        x = 100 + 8.0 * frame + (jump if frame > gap[1] else 0.0)
        c = conf_after if frame > gap[1] else 0.9
        rows.append([x, 200, x + 60, 240, c, 2.0])
    rows.append([700, 400, 760, 440, 0.9, 2.0])            # xe dung yen (giu pool khong rong)
    return Boxes(np.array(rows, dtype=np.float32), (540, 960))


def synth_ids(yaml, overrides, n=30, **kw):
    cfg = run_tracking.load_tracker_cfg(yaml, overrides)
    tr = run_tracking.make_tracker(cfg, assoc_log=True)
    ids_before, ids_after, stages = set(), set(), []
    for f in range(1, n + 1):
        out = tr.update(scene(f, **kw).cpu().numpy(), None).reshape(-1, 8)
        for r in out:
            if r[1] < 300:                       # xe chuyen dong (y = 200)
                (ids_before if f < 10 else ids_after).add(int(r[4]))
    tracks, dets, events = tr.assoc_log.frames()
    return ids_before, ids_after, tracks, events


def main() -> int:
    tracker_ctrv.register()
    failed = 0
    try:
        # --- 1. tat -> bit-identical tren cache that ---
        vd = det_cache.open_cache("yolov8n-coco_conf0.05_all", "MVI_39031", 0.25, [2, 3, 5, 7], 300)
        vd.frame_idx = vd.frame_idx[:300]; vd.time_s = vd.time_s[:300]; vd.unreadable = vd.unreadable[:300]
        for name, y in (("cv", "bytetrack.yaml"), ("ekf", str(ROOT / "configs/bytetrack_ekf_ctrv.yaml"))):
            base, _ = run_cfg(y, None, vd)
            off, s_off = run_cfg(y, {"lost_no_fuse": False, "lost_recover": "none"}, vd)
            _check(f"[{name}] lost_recovery TAT == goc (bit-identical, 300 frame MVI_39031, {len(base)} dong)",
                   base.shape == off.shape and np.array_equal(base.to_numpy(), off.to_numpy()))
            _check(f"[{name}] tracker khong bi boc khi tat", "Recover" not in type(s_off["tracker"]).__name__)
            on, s_on = run_cfg(y, {"lost_recover": "iou", "lost_recover_iou": 0.05}, vd)
            _check(f"[{name}] bat lost_recover=iou -> lop Recover* va output KHAC goc ({len(on)} dong)",
                   "Recover" in type(s_on["tracker"]).__name__ and not (on.shape == base.shape and np.array_equal(on.to_numpy(), base.to_numpy())))
        # --- 2. kich ban tong hop: du doan lech -> IoU < 0.2 -> goc sinh ID moi; vong 3 noi lai ---
        y = str(ROOT / "configs/bytetrack_ekf_ctrv.yaml")
        b0, a0, _, _ = synth_ids(y, None)
        _check("goc: xe lech 60 px sau gap -> ID moi (switched)", b0 != a0 and len(a0) == 1, f"{b0} -> {a0}")
        b0b, a0b, _, _ = synth_ids(y, None, jump=45.0)
        b1, a1, t1, e1 = synth_ids(y, {"lost_recover": "iou", "lost_recover_iou": 0.02}, jump=45.0)
        _check("lech 45 px (IoU ~0.14 < 0.2): goc doi ID; lost_recover=iou 0.02 noi lai cung ID, refind stage 4",
               b0b != a0b and
               b1 == a1 and (e1.kind == "refind").any() and (t1.stage == 4).any(), f"{b1} -> {a1}")
        b2, a2, t2, e2 = synth_ids(y, {"lost_recover": "maha", "lost_recover_maha2": 5.991}, jump=200.0)
        _check("lost_recover=maha: lech 200 px NGOAI gate -> khong noi (ID moi)", b2 != a2, f"{b2} -> {a2}")
        # gate Mahalanobis theo P cua bo loc: sau 3 buoc mat chi ~15 px (chat hon IoU 0.2 voi box 60 px);
        # sau 15 buoc ~48 px -> lech 42 px (IoU 0.176 < 0.2) nam TRONG gate
        b3, a3, t3, e3 = synth_ids(y, {"lost_recover": "maha", "lost_recover_maha2": 5.991}, jump=50.0, gap=(10, 24),
                                   n=40)
        _check("lost_recover=maha: lech ~46 px (IoU 0.13 < 0.2) sau 15 frame mat, trong gate -> noi lai cung ID",
               b3 == a3 and (t3.stage == 4).any(), f"{b3} -> {a3}")
        # --- 3. lost_no_fuse: det tai xuat hien conf 0.3, IoU ~0.5 -> fused cost 0.85 > 0.8 bi loai; khong fuse -> nhan ---
        b4, a4, _, _ = synth_ids(y, None, jump=22.0, conf_after=0.3)
        b5, a5, t5, _ = synth_ids(y, {"lost_no_fuse": True}, jump=22.0, conf_after=0.3)
        _check("lost_no_fuse: goc doi ID (fused cost > 0.8), khong fuse -> giu ID (stage 1)",
               b4 != a4 and b5 == a5, f"goc {b4}->{a4}; no_fuse {b5}->{a5}")
    except AssertionError as e:
        failed += 1; print(f"  !! {e}")
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
