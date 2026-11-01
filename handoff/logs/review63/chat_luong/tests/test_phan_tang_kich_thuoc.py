"""Test phan_tang_kich_thuoc tren du lieu TONG HOP nho (khong can cache that).
Chay: python handoff/logs/review63/chat_luong/tests/test_phan_tang_kich_thuoc.py"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phan_tang_kich_thuoc as pk  # noqa: E402


def test_ghep_1_1_va_nguong():
    gt = np.array([[0, 0, 10, 10], [100, 100, 140, 140]], float)
    det = np.array([[1, 1, 11, 11], [0, 0, 10, 10], [300, 300, 310, 310]], float)
    g, d = pk.ghep(gt, det)
    assert g.tolist() == [True, False]
    assert d.sum() == 1 and not d[2]                         # 1-1: chi mot det ghep voi gt 0


def test_ghep_uu_tien_so_cap_truoc_tong_iou():
    # Regression GPT R63 Q7: 5 box 3x1, GT x=i, det x=j-1 (i,j=0..4). Ghep i<->i: 4 cap IoU=1 + 1 cap loai;
    # ghep det dich 1: 5 cap IoU=0,5. Phai chon 5 cap.
    gt = np.array([[i, 0, i + 3, 1] for i in range(5)], float)
    det = np.array([[j - 1, 0, j + 2, 1] for j in range(5)], float)
    g, d = pk.ghep(gt, det)
    assert g.sum() == 5 and d.sum() == 5


def test_so_cap_tu_choi_tap_video_khac():
    a = {"v1": {}, "v2": {}}; b = {"v1": {}}
    try:
        pk.so_cap(a, b, n_boot=10)
    except ValueError:
        return
    raise AssertionError("phai bao loi khi hai cache khac tap video")


def test_dem_theo_nhom_kich_thuoc_va_che():
    # frame 1: xe nho (20x20) bi che nang, detect duoc; xe lon (100x100) khong che, bo sot.
    gt = {1: np.array([[0, 0, 20, 20, 0.3], [200, 200, 300, 300, 1.0]], float)}
    det = {1: np.array([[0, 0, 20, 20, 0.9, 2], [500, 500, 530, 530, 0.8, 2]], float)}
    dem = pk.dem_video(gt, det)
    assert dem[("gt", "nho", "nang")] == 1 and dem[("tp", "nho", "nang")] == 1
    assert dem[("gt", "rat_lon", "khong_che")] == 1 and ("tp", "rat_lon", "khong_che") not in dem
    assert dem[("det", "*")] == 2 and dem[("dtp", "*")] == 1 and dem[("det", "vua")] == 1
    hang = {(h["kich_thuoc"], h["che"]): h for h in pk.bang_tong({"v": dem})}
    assert hang[("*", "*")]["recall"] == 0.5 and hang[("*", "*")]["precision"] == 0.5
    assert hang[("nho", "*")]["recall"] == 1.0 and hang[("rat_lon", "*")]["recall"] == 0.0


def test_so_cap_bootstrap_dau_va_gt_khac():
    a, b = {}, {}
    for i in range(6):
        gt = {1: np.array([[0, 0, 20, 20, 1.0], [50, 50, 70, 70, 1.0]], float)}
        a["v%d" % i] = pk.dem_video(gt, {1: np.array([[0, 0, 20, 20, .9, 2], [50, 50, 70, 70, .9, 2]], float)})
        b["v%d" % i] = pk.dem_video(gt, {1: np.array([[0, 0, 20, 20, .9, 2]], float)})
    kq = {(r["kich_thuoc"], r["che"]): r for r in pk.so_cap(a, b, n_boot=200)}
    r = kq[("*", "*")]
    assert abs(r["hieu_recall"] - 0.5) < 1e-12 and r["ci95"] == [0.5, 0.5] and not r["ci_chua_0"]
    assert kq[("lon", "*")]["hieu_recall"] is None


if __name__ == "__main__":
    n = 0
    for ten, f in sorted(globals().items()):
        if ten.startswith("test_") and callable(f):
            f(); n += 1; print("PASS", ten)
    print("%d/%d PASS" % (n, n))
