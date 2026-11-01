"""Test nho cho danh_gia_ngoai (converter GT, cap trung, doi ID mixed5 -> COCO). Chay: python -B -m pytest -q <file>"""
import sys
from pathlib import Path

import numpy as np
try:
    import pytest
except ImportError:                              # khong co pytest: raises toi gian
    import contextlib

    class pytest:  # noqa: N801
        @staticmethod
        @contextlib.contextmanager
        def raises(exc):
            try:
                yield
            except exc:
                return
            raise AssertionError(f"khong nem {exc.__name__}")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import danh_gia_ngoai as dg  # noqa: E402


def test_roundabout_xyxy_sang_xywh(tmp_path):
    p = tmp_path / "SCT_GT.txt"
    p.write_text("1 7 10.00 20.00 110.00 70.00 2\n2 7 12.50 21.00 113.00 72.00 2\n")
    a = dg.doc_gt_roundabout(p)
    assert a.shape == (2, 9)
    np.testing.assert_allclose(a[0], [1, 7, 10, 20, 100, 50, 1, 2, 1.0])
    np.testing.assert_allclose(a[1, 4:6], [100.5, 51])


def test_roundabout_tu_choi_khong_phai_xyxy(tmp_path):
    p = tmp_path / "SCT_GT.txt"
    p.write_text("1 7 10 20 5 70 2\n")
    with pytest.raises(dg.LoiHopDong):
        dg.doc_gt_roundabout(p)


def test_dong_trung_va_kiem_gt():
    a = np.array([[1, 5, 0, 0, 10, 10, 1, 1, 1], [1, 5, 50, 0, 10, 10, 1, 1, 1], [1, 6, 0, 0, 10, 10, 1, 1, 1],
                  [2, 5, 0, 0, 10, 10, 1, 1, 1]], float)
    assert dg.dong_trung(a).tolist() == [True, True, False, False]
    with pytest.raises(dg.LoiHopDong):
        dg.kiem_gt(a, 2, "x")
    assert dg.kiem_gt(a, 2, "x", cho_phep_trung=True)["n_box"] == 4
    with pytest.raises(dg.LoiHopDong):
        dg.kiem_gt(a, 1, "x", cho_phep_trung=True)          # frame 2 > n_frame


# --- cong cache (review GPT 02/11, B_sua): fixture cache gia 4 frame, detector mixed5, dataset svmot smoke ---------
NAMES5 = {"0": "car", "1": "truck", "2": "bus", "3": "motorcycle", "4": "bicycle"}
BANG5 = {"0": 2, "1": 7, "2": 5, "3": 3, "4": 1}
DET = "yolov8n_mixed5"


def _lam_cache(root, n=4, sua=None):
    """Tao cache r64smoke_svmot_<DET> hop le trong root (gia CACHE_ROOT) va TAP svmot voi n frame; sua(z, info, meta)."""
    import json as js
    dg.det_cache.CACHE_ROOT = root
    t = dict(dg.TAP["svmot"]); t["n_goc"] = dict(t["n_goc"]); t["n_goc"]["seq_22_morning"] = n
    dg.TAP["svmot"] = t
    dg.SMOKE_N = n
    sha = dg.detector_r54(DET)["sha256"]
    meta = {"backend": "ultralytics", "weights_sha256": sha, "imgsz": 640, "conf": 0.05, "iou": 0.7, "classes": None,
            "max_det": 3000, "half": False, "agnostic_nms": False, "augment": False, "rect": True, "batch": 1,
            "mask_ignored_regions": False, "dataset": "svmot", "spec_sha256": "abc"}
    info = {"detector": DET, "dataset": "svmot", "checkpoint_sha256": sha, "names": NAMES5, "anh_xa_coco": BANG5,
            "spec_sha256": "abc"}
    data = np.array([[10, 10, 50, 40, 0.9, 2], [100, 100, 150, 140, 0.3, 3], [5, 5, 9, 9, 0.06, 1]], np.float32)
    z = {"frame_idx": np.arange(1, n + 1, dtype=np.int32), "n_per_frame": np.array([2, 0, 1] + [0] * (n - 3), np.int32),
         "unreadable": np.zeros(n, bool), "truncated": np.zeros(n, bool),
         "time_s": np.arange(n, dtype=np.float64) / 30.0, "data": data, "orig_shape": np.array([1080, 1920], np.int32),
         "cls_goc": np.array([0, 3, 4], np.int16)}
    if sua:
        sua(z, info, meta)
    cdir = root / dg.ten_cache("svmot", DET, True)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "meta.json").write_text(js.dumps(meta))
    (cdir / "seq_22_morning.json").write_text(js.dumps(info))
    np.savez_compressed(cdir / "seq_22_morning.npz", **z)
    (cdir / "seq_22_morning.xong.json").write_text(js.dumps({
        "detector": DET, "video": "seq_22_morning", "dataset": "svmot", "split": "SVMOTSYS-test-smoke128",
        "sha256": {f: dg.sha256_file(cdir / f) for f in ("seq_22_morning.npz", "seq_22_morning.json", "meta.json")}}))
    return cdir


def _phai_loi(tmp_path, sua=None, hong=None):
    cdir = _lam_cache(tmp_path, sua=sua)
    if hong:
        hong(cdir)
    with pytest.raises(dg.LoiHopDong):
        dg.kiem_don_vi_cache("svmot", DET, "seq_22_morning", True)


def test_cong_cache_hop_le(tmp_path):
    _lam_cache(tmp_path)
    assert dg.kiem_don_vi_cache("svmot", DET, "seq_22_morning", True)["n_boxes"] == 3


def test_cong_cache_marker_thieu_hoac_hong(tmp_path):
    _phai_loi(tmp_path, hong=lambda c: (c / "seq_22_morning.xong.json").unlink())
    _phai_loi(tmp_path / "b", hong=lambda c: (c / "seq_22_morning.xong.json").write_text("{khong json"))


def test_cong_cache_hash_sai(tmp_path):
    _phai_loi(tmp_path, hong=lambda c: (c / "seq_22_morning.json").write_text((c / "seq_22_morning.json").read_text() + " "))


def test_cong_cache_timeline_sai(tmp_path):
    def sua(z, i, m):
        z["time_s"] = z["time_s"] * 2
    _phai_loi(tmp_path, sua=sua)

    def sua2(z, i, m):
        z["frame_idx"] = z["frame_idx"] + 1
    _phai_loi(tmp_path / "b", sua=sua2)


def test_cong_cache_lop_sai_va_nan(tmp_path):
    def sua(z, i, m):
        z["data"] = z["data"].copy(); z["data"][0, 5] = 7         # cls khong khop anh xa(cls_goc=0 -> 2)
    _phai_loi(tmp_path, sua=sua)

    def sua2(z, i, m):
        z["data"] = z["data"].copy(); z["data"][1, 2] = np.nan
    _phai_loi(tmp_path / "b", sua=sua2)

    def sua3(z, i, m):
        i["anh_xa_coco"] = {**BANG5, "0": 7}
    _phai_loi(tmp_path / "c", sua=sua3)


def test_cong_cache_meta_lech(tmp_path):
    def sua(z, i, m):
        m["mask_ignored_regions"] = True
    _phai_loi(tmp_path, sua=sua)


def test_don_vi_do_khong_ghi_de(tmp_path):
    cdir = _lam_cache(tmp_path)
    (cdir / "seq_22_morning.xong.json").unlink()                     # artifact con, marker mat -> don vi do
    with pytest.raises(dg.LoiHopDong):
        dg.buoc_cache("svmot", DET, True, None, None)


class _FakeCap:
    def __init__(self, frames):
        self.f = list(frames)

    def isOpened(self):
        return True

    def read(self):
        return (True, self.f.pop(0)) if self.f else (False, None)

    def release(self):
        pass


def test_mp4_thieu_frame_sai_kich_thuoc_va_goi_detect_frame():
    import cv2
    goi = []
    goc_cap, goc_det = cv2.VideoCapture, dg.det_cache.detect_frame
    dg.det_cache.detect_frame = lambda m, img, s: goi.append(img.shape) or np.zeros((1, 6), np.float32)

    class S:
        max_det = 3000
    try:
        cv2.VideoCapture = lambda p: _FakeCap([np.zeros((4, 6, 3), np.uint8)] * 3)
        a = dg.build_tu_video(None, Path("x.mp4"), 3, 15.0, S, (4, 6))
        assert goi == [(4, 6, 3)] * 3 and a["n_per_frame"].tolist() == [1, 1, 1]
        np.testing.assert_allclose(a["time_s"], [0, 1 / 15, 2 / 15])
        cv2.VideoCapture = lambda p: _FakeCap([np.zeros((4, 6, 3), np.uint8)] * 2)
        with pytest.raises(dg.LoiHopDong):
            dg.build_tu_video(None, Path("x.mp4"), 3, 15.0, S, (4, 6))
        cv2.VideoCapture = lambda p: _FakeCap([np.zeros((5, 6, 3), np.uint8)] * 3)
        with pytest.raises(dg.LoiHopDong):
            dg.build_tu_video(None, Path("x.mp4"), 3, 15.0, S, (4, 6))
    finally:
        cv2.VideoCapture, dg.det_cache.detect_frame = goc_cap, goc_det


def test_doi_id_giu_box_conf_thu_tu():
    goc = np.array([[0, 0, 1, 1, 0.9, 3], [5, 5, 6, 6, 0.5, 0]], np.float32)
    moi = dg.doi_id_sang_coco(goc, {0: 2, 3: 3})
    np.testing.assert_array_equal(moi[:, :5], goc[:, :5])
    assert moi[:, 5].tolist() == [3, 2]
    assert dg.doi_id_sang_coco(goc, None) is not goc
    with pytest.raises(dg.LoiHopDong):
        dg.doi_id_sang_coco(goc, {0: 2})


def test_bang_anh_xa_mixed5_va_coco():
    import detector_classes as dc  # noqa: F401  (qua sys.path cua danh_gia_ngoai)
    coco = {i: n for i, n in enumerate(["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck"]
                                       + [f"x{i}" for i in range(8, 80)])}
    with pytest.raises(dg.LoiHopDong):
        dg.bang_anh_xa_coco(coco, "lop5")


if __name__ == "__main__":                       # env khoaluan-mot khong co pytest: chay tung test_* (tmp_path tu tao)
    import inspect
    import tempfile
    ok = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        try:
            if "tmp_path" in inspect.signature(f).parameters:
                with tempfile.TemporaryDirectory() as d:
                    f(Path(d))
            else:
                f()
            ok += 1
            print("PASS", n)
        except Exception as e:  # noqa: BLE001
            print("FAIL", n, repr(e))
    print(f"KET QUA: {ok}/{len(tests)} PASS")
    sys.exit(0 if ok == len(tests) else 1)
