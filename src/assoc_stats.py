"""
assoc_stats.py -- Do hoat dong cua VONG ASSOCIATION THU HAI cua ByteTrack.

Vong hai la co che dac trung cua ByteTrack: ghep cac detection DIEM THAP voi
cac track chua ghep duoc o vong mot. Voi cau hinh baseline hien tai (YOLO
conf = 0.25 = track_high_thresh) thi dai diem (track_low_thresh, conf) rong,
nen vong hai khong bao gio co du lieu -- xem src/probe_conf_bands.py.

Module nay dem BA con so KHAC NHAU, khong duoc gop lam mot:

    n_low_dets    so detection diem thap DUA VAO vong hai
    n_cand_tracks so track DU DIEU KIEN tham gia (dang `Tracked` va chua ghep
                  duoc o vong mot -- byte_tracker.py dong 427)
    n_matched     so cap GHEP THANH CONG

Mot detection dua vao khong co nghia la ghep duoc: vong hai con phai co track
du dieu kien, va con phai vuot nguong IoU 0.5 cua chinh no.

CACH DUNG
    import assoc_stats
    assoc_stats.enable()          # phai goi TRUOC khi tao tracker
    ... chay tracking ...
    print(assoc_stats.summary())
    assoc_stats.disable()         # tra lai ham goc
"""
from __future__ import annotations

from collections import Counter

from ultralytics.trackers.basetrack import TrackState
from ultralytics.trackers.byte_tracker import BYTETracker

_COUNTS: Counter = Counter()
_ORIGINAL = None


def reset() -> None:
    _COUNTS.clear()


def enable() -> None:
    """Boc `BYTETracker._second_association` de dem, khong doi hanh vi."""
    global _ORIGINAL
    if _ORIGINAL is not None:
        return
    _ORIGINAL = BYTETracker._second_association

    def wrapped(self, strack_pool, u_track, detections_second,
                activated, refind, lost, *args, **kwargs):
        # Dem TRUOC khi goi ham goc, vi ham goc ghi de bien u_track
        n_low = len(detections_second)
        n_cand = sum(1 for i in u_track
                     if strack_pool[i].state == TrackState.Tracked)
        n_act_before, n_ref_before = len(activated), len(refind)
        out = _ORIGINAL(self, strack_pool, u_track, detections_second,
                        activated, refind, lost, *args, **kwargs)
        # Cap ghep thanh cong o vong hai = so track duoc them vao activated/refind
        n_matched = (len(activated) - n_act_before) + (len(refind) - n_ref_before)
        _COUNTS["n_frames"] += 1
        _COUNTS["n_low_dets"] += n_low
        _COUNTS["n_cand_tracks"] += n_cand
        _COUNTS["n_matched"] += n_matched
        if n_low == 0:
            _COUNTS["n_frames_no_low_det"] += 1
        return out

    BYTETracker._second_association = wrapped


def disable() -> None:
    global _ORIGINAL
    if _ORIGINAL is not None:
        BYTETracker._second_association = _ORIGINAL
        _ORIGINAL = None


def counts() -> dict:
    return dict(_COUNTS)


def summary() -> str:
    c = _COUNTS
    n_f = c.get("n_frames", 0)
    if not n_f:
        return "  (chua do duoc frame nao -- da goi enable() truoc khi chay chua?)"
    no_low = c.get("n_frames_no_low_det", 0)
    lines = [
        f"  {'so frame da chay':<44}{n_f:>12,}",
        f"  {'frame KHONG co detection diem thap nao':<44}{no_low:>12,}"
        f"  ({no_low / n_f * 100:.1f}%)",
        f"  {'detection diem thap dua vao vong 2':<44}{c.get('n_low_dets', 0):>12,}",
        f"  {'track du dieu kien tham gia vong 2':<44}{c.get('n_cand_tracks', 0):>12,}",
        f"  {'cap ghep THANH CONG o vong 2':<44}{c.get('n_matched', 0):>12,}",
    ]
    nl = c.get("n_low_dets", 0)
    if nl:
        lines.append(f"  {'ti le ghep duoc / detection dua vao':<44}"
                     f"{c.get('n_matched', 0) / nl * 100:>11.1f}%")
    return "\n".join(lines)
