"""
test_occlusion_eval.py -- Test cho phep ghep IoU va dinh nghia su kien che khuat.

Bao ve cac loi da tung xay ra trong `stratified_analysis.py` (xem FIX_REPORT.md):

  LOI A  Chay Hungarian tren toan bo ma tran IoU roi MOI loc nguong -> cac cap
         duoi nguong keo phep ghep di sai va lam mat cap hop le.
  LOI B  Doan che "full" nam long trong doan "partial" cua cung mot lan bi che
         bi dem thanh hai mau doc lap.

    python src/test_occlusion_eval.py        # in bao cao tung phep kiem tra
    pytest src/test_occlusion_eval.py -q     # neu co pytest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from occlusion_eval import (  # noqa: E402
    IOU_THRESHOLD, add_gt_eligibility, build_events, evaluate_events,
    iou_matrix, match_frame,
)

_RESULTS: list[tuple[str, bool]] = []


def _check(name: str, ok: bool, detail: str = ""):
    _RESULTS.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


# ---------------------------------------------------------------------------
# A. GHEP IoU
# ---------------------------------------------------------------------------
def test_example_from_bug_report():
    """Vi du chinh xac trong bao cao loi: cap hop le 0.6384 khong duoc bi mat."""
    iou = np.array([[0.1281, 0.3739],
                    [0.3963, 0.6384]])
    pairs = match_frame(iou, 0.5)
    ok = pairs == [(1, 1)]
    # Doi chieu voi cach CU de thay ro khac biet
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-iou)
    old = [(int(i), int(j)) for i, j in zip(r, c) if iou[i, j] >= 0.5]
    _check("A1. Vi du trong bao cao loi: giu duoc cap (1,1) IoU=0.6384", ok,
           f"cach cu   -> {old}  (mat cap hop le)\n"
           f"cach moi  -> {pairs}")


def test_rectangular_matrices():
    """Ma tran chu nhat ca hai chieu: khong duoc loi, khong ep ghep."""
    # 1 GT, 3 tracker: chi cap thu 2 hop le
    a = np.array([[0.2, 0.7, 0.4]])
    pa = match_frame(a, 0.5)
    # 3 GT, 1 tracker: chi hang thu 3 hop le
    b = np.array([[0.1], [0.3], [0.9]])
    pb = match_frame(b, 0.5)
    _check("A2. Ma tran chu nhat (1x3 va 3x1)",
           pa == [(0, 1)] and pb == [(2, 0)],
           f"1x3 -> {pa} (mong [(0,1)])\n3x1 -> {pb} (mong [(2,0)])")


def test_empty_matrices():
    """Ma tran rong o moi chieu: tra ve danh sach rong, khong nem loi."""
    cases = [np.zeros((0, 0)), np.zeros((0, 3)), np.zeros((3, 0))]
    got = [match_frame(m, 0.5) for m in cases]
    _check("A3. Ma tran rong (0x0, 0x3, 3x0)", all(g == [] for g in got),
           f"ket qua: {got}")


def test_no_valid_pair():
    """Khong co cap nao dat nguong -> khong ghep gi ca, khong ep ghep."""
    iou = np.array([[0.10, 0.20],
                    [0.30, 0.45]])
    pairs = match_frame(iou, 0.5)
    _check("A4. Khong cap nao dat nguong -> khong ghep", pairs == [],
           f"ket qua: {pairs} (mong [])")


def test_threshold_is_inclusive():
    """Dung DUNG nguong: IoU bang dung nguong phai duoc chap nhan."""
    iou = np.array([[0.5]])
    p_at = match_frame(iou, 0.5)
    p_below = match_frame(np.array([[0.5 - 1e-12]]), 0.5)
    _check("A5. So sanh nguong dung '>=' (IoU = nguong thi hop le)",
           p_at == [(0, 0)] and p_below == [],
           f"IoU = 0.5      -> {p_at} (mong [(0,0)])\n"
           f"IoU = 0.5-1e-12 -> {p_below} (mong [])")


def test_maximizes_valid_count_then_iou():
    """Muc tieu tu dien: uu tien SO cap hop le, roi moi den tong IoU."""
    # Duong cheo chinh: 2 cap hop le, tong = 0.55 + 0.55 = 1.10
    # Duong cheo phu : 1 cap hop le 0.99 + 1 cap khong hop le 0.10
    iou = np.array([[0.55, 0.99],
                    [0.10, 0.55]])
    pairs = sorted(match_frame(iou, 0.5))
    _check("A6. Uu tien so cap hop le truoc, roi moi toi tong IoU",
           pairs == [(0, 0), (1, 1)],
           f"ket qua {pairs} (mong [(0,0),(1,1)] = 2 cap hop le, "
           f"thay vi 1 cap 0.99)")


def test_iou_matrix_basic():
    """IoU tinh dung tren vai truong hop de kiem tay."""
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[0.0, 0.0, 10.0, 10.0],      # trung khit -> 1.0
                  [10.0, 0.0, 10.0, 10.0],     # ke nhau, khong giao -> 0.0
                  [5.0, 0.0, 10.0, 10.0]])     # giao nua -> 50/150
    m = iou_matrix(a, b)
    _check("A7. iou_matrix dung tren truong hop kiem tay",
           abs(m[0, 0] - 1.0) < 1e-12 and abs(m[0, 1]) < 1e-12
           and abs(m[0, 2] - 50.0 / 150.0) < 1e-12,
           f"{np.round(m, 6)}")


# ---------------------------------------------------------------------------
# B. DINH NGHIA SU KIEN
# ---------------------------------------------------------------------------
def test_full_nested_in_partial_not_double_counted():
    """Doan full nam long trong partial phai gop thanh MOT su kien."""
    partial = pd.DataFrame([dict(video="V", track_id=1, start_frame=100, end_frame=160)])
    full = pd.DataFrame([dict(video="V", track_id=1, start_frame=120, end_frame=140)])
    ev = build_events(partial, full)
    _check("B1. full nam long trong partial -> 1 su kien, khong phai 2",
           len(ev) == 1 and int(ev.iloc[0].start_frame) == 100
           and int(ev.iloc[0].end_frame) == 160 and bool(ev.iloc[0].has_full),
           f"so su kien = {len(ev)} (mong 1), "
           f"khoang = {int(ev.iloc[0].start_frame)}-{int(ev.iloc[0].end_frame)}, "
           f"has_full = {bool(ev.iloc[0].has_full)}")


def test_separate_events_stay_separate():
    """Hai lan bi che cach xa nhau van la HAI su kien."""
    partial = pd.DataFrame([
        dict(video="V", track_id=1, start_frame=100, end_frame=120),
        dict(video="V", track_id=1, start_frame=200, end_frame=220),
    ])
    full = pd.DataFrame(columns=["video", "track_id", "start_frame", "end_frame"])
    ev = build_events(partial, full)
    _check("B2. Hai lan che cach xa nhau -> 2 su kien rieng",
           len(ev) == 2 and not ev.has_full.any(),
           f"so su kien = {len(ev)} (mong 2)")


def test_adjacent_segments_merge():
    """Hai doan ke nhau (end + 1 == start) la cung mot lan bi che."""
    partial = pd.DataFrame([
        dict(video="V", track_id=1, start_frame=100, end_frame=120),
        dict(video="V", track_id=1, start_frame=121, end_frame=130),
    ])
    ev = build_events(partial, pd.DataFrame(
        columns=["video", "track_id", "start_frame", "end_frame"]))
    _check("B3. Doan ke nhau (121 ngay sau 120) -> gop lam 1",
           len(ev) == 1 and int(ev.iloc[0].end_frame) == 130,
           f"so su kien = {len(ev)}, khoang = "
           f"{int(ev.iloc[0].start_frame)}-{int(ev.iloc[0].end_frame)}")


def _mk_gt(frames, video="V", tid=1):
    return pd.DataFrame([dict(video=video, id=tid, frame=f, x=0.0, y=0.0,
                              w=10.0, h=10.0) for f in frames])


def test_eligibility_from_gt_only():
    """Du dieu kien hay khong CHI phu thuoc GT, khong phu thuoc tracker."""
    partial = pd.DataFrame([dict(video="V", track_id=1, start_frame=100, end_frame=120)])
    empty_full = pd.DataFrame(columns=["video", "track_id", "start_frame", "end_frame"])
    ev = build_events(partial, empty_full)

    gt_both = _mk_gt(list(range(90, 131)))          # co GT ca truoc lan sau
    gt_no_after = _mk_gt(list(range(90, 121)))      # GT het ngay sau su kien
    gt_no_before = _mk_gt(list(range(100, 131)))    # track sinh ra khi da bi che

    e1 = add_gt_eligibility(ev, gt_both)
    e2 = add_gt_eligibility(ev, gt_no_after)
    e3 = add_gt_eligibility(ev, gt_no_before)
    _check("B4. Dieu kien du xac dinh tu GT, phan biet dung ly do loai",
           bool(e1.iloc[0].eligible)
           and (not bool(e2.iloc[0].eligible))
           and e2.iloc[0].exclusion_reason == "gt_missing_after"
           and (not bool(e3.iloc[0].eligible))
           and e3.iloc[0].exclusion_reason == "gt_missing_before",
           f"GT ca hai phia   -> eligible={bool(e1.iloc[0].eligible)}\n"
           f"GT het sau su kien -> {e2.iloc[0].exclusion_reason}\n"
           f"GT thieu truoc     -> {e3.iloc[0].exclusion_reason}")


def test_empty_tracker_still_evaluated():
    """File tracking rong KHONG duoc bo qua am tham: van phai bi cham diem."""
    partial = pd.DataFrame([dict(video="V", track_id=1, start_frame=100, end_frame=120)])
    ev = build_events(partial, pd.DataFrame(
        columns=["video", "track_id", "start_frame", "end_frame"]))
    gt = _mk_gt(list(range(90, 131)))
    ev = add_gt_eligibility(ev, gt)
    empty_match = pd.DataFrame(columns=["frame", "gt_id", "tracker_id", "iou", "video"])
    res = evaluate_events(ev, empty_match, gt)
    _check("B5. Tracker khong ra gi -> van duoc cham la no_match_before",
           len(res) == 1 and bool(res.iloc[0].eligible)
           and res.iloc[0].outcome == "no_match_before",
           f"outcome = {res.iloc[0].outcome} (mong no_match_before, "
           f"KHONG phai bi bo qua)")


def test_lost_after_vs_gt_missing():
    """Phan biet: GT con ma tracker mat (that bai) vs GT het (khong danh gia duoc)."""
    partial = pd.DataFrame([dict(video="V", track_id=1, start_frame=100, end_frame=120)])
    empty_full = pd.DataFrame(columns=["video", "track_id", "start_frame", "end_frame"])
    ev0 = build_events(partial, empty_full)

    # Truong hop 1: GT co ca hai phia, tracker ghep duoc TRUOC nhung khong ghep duoc SAU
    gt1 = _mk_gt(list(range(90, 131)))
    ev1 = add_gt_eligibility(ev0, gt1)
    m1 = pd.DataFrame([dict(video="V", gt_id=1, frame=f, tracker_id=7, iou=0.9)
                       for f in range(90, 100)])
    r1 = evaluate_events(ev1, m1, gt1)

    # Truong hop 2: GT het sau su kien -> phai bi LOAI, khong tinh la that bai
    gt2 = _mk_gt(list(range(90, 121)))
    ev2 = add_gt_eligibility(ev0, gt2)
    r2 = evaluate_events(ev2, m1, gt2)

    _check("B6. GT con ma mat = lost_after; GT het = loai khoi mau",
           r1.iloc[0].outcome == "lost_after"
           and r2.iloc[0].outcome is None
           and r2.iloc[0].exclusion_reason == "gt_missing_after",
           f"GT con  -> outcome = {r1.iloc[0].outcome}\n"
           f"GT het   -> outcome = {r2.iloc[0].outcome}, "
           f"ly do loai = {r2.iloc[0].exclusion_reason}")


def test_id_match_vs_id_continuous():
    """Hai khai niem 'giu ID' phai KHAC nhau va do rieng."""
    partial = pd.DataFrame([dict(video="V", track_id=1, start_frame=100, end_frame=104)])
    ev = build_events(partial, pd.DataFrame(
        columns=["video", "track_id", "start_frame", "end_frame"]))
    gt = _mk_gt(list(range(90, 121)))
    ev = add_gt_eligibility(ev, gt)

    # Truong hop A: ID trung truoc/sau NHUNG mat dau giua chung
    mA = pd.DataFrame(
        [dict(video="V", gt_id=1, frame=f, tracker_id=7, iou=0.9) for f in range(90, 100)]
        + [dict(video="V", gt_id=1, frame=f, tracker_id=7, iou=0.9) for f in range(105, 115)])
    rA = evaluate_events(ev, mA, gt)
    # Truong hop B: bam lien tuc suot su kien
    mB = pd.DataFrame(
        [dict(video="V", gt_id=1, frame=f, tracker_id=7, iou=0.9) for f in range(90, 115)])
    rB = evaluate_events(ev, mB, gt)

    _check("B7. id_match va id_continuous la hai khai niem khac nhau",
           rA.iloc[0].outcome == "preserved" and not bool(rA.iloc[0].id_continuous)
           and rB.iloc[0].outcome == "preserved" and bool(rB.iloc[0].id_continuous),
           f"mat dau giua chung : outcome={rA.iloc[0].outcome}, "
           f"id_continuous={bool(rA.iloc[0].id_continuous)}\n"
           f"bam lien tuc       : outcome={rB.iloc[0].outcome}, "
           f"id_continuous={bool(rB.iloc[0].id_continuous)}")


def main() -> int:
    print("=" * 78)
    print("TEST -- ghep IoU va dinh nghia su kien che khuat")
    print("=" * 78)
    print("\n--- A. Ghep IoU ---")
    for fn in (test_example_from_bug_report, test_rectangular_matrices,
               test_empty_matrices, test_no_valid_pair, test_threshold_is_inclusive,
               test_maximizes_valid_count_then_iou, test_iou_matrix_basic):
        fn()
    print("\n--- B. Dinh nghia su kien ---")
    for fn in (test_full_nested_in_partial_not_double_counted,
               test_separate_events_stay_separate, test_adjacent_segments_merge,
               test_eligibility_from_gt_only, test_empty_tracker_still_evaluated,
               test_lost_after_vs_gt_missing, test_id_match_vs_id_continuous):
        fn()
    n_ok = sum(1 for _, ok in _RESULTS if ok)
    print("\n" + "=" * 78)
    print(f"KET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    print("=" * 78)
    return 0 if n_ok == len(_RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
