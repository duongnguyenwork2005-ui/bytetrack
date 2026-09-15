"""
test_det_cache.py -- Kiem tra detection cache (Giai doan 2, buoc 3).

Diem quan trong nhat: CHUNG MINH BANG SO rang loc cache (conf thap, tat ca lop,
max_det lon) len dieu kien chat hon cho DUNG tap detection nhu chay NMS voi
dieu kien chat do (tinh chat tien to cua NMS tham lam, xem det_cache.py). Test
nay goi thang `ultralytics.utils.nms.non_max_suppression` tren tensor du doan
tho tong hop (khong can GPU / anh).

    python src/test_det_cache.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import det_cache  # noqa: E402
from det_cache import (DetectorSpec, VideoDetections, build_video_gt, check_compatible,  # noqa: E402
                       save_video, to_boxes, write_meta)
from mot_datasets import FrameRef  # noqa: E402

from ultralytics.engine.results import Boxes  # noqa: E402
from ultralytics.utils.nms import non_max_suppression  # noqa: E402

_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def raw_predictions(n_boxes: int, nc: int = 80, seed: int = 0) -> torch.Tensor:
    """Tensor du doan tho (1, 4 + nc, N) dang YOLOv8: xywh + diem tung lop.

    Box tap trung thanh cum de NMS thuc su loai bo nhieu; diem lop chinh ~ U(0,1),
    lop phu ~ U(0, 0.3) de doi khi lop phu vuot nguong thap (thu thach `best class only`).
    """
    g = torch.Generator().manual_seed(seed)
    centers = torch.rand(n_boxes, 2, generator=g) * torch.tensor([960.0, 540.0])
    centers += torch.randn(n_boxes, 2, generator=g) * 15          # cum
    wh = 20 + torch.rand(n_boxes, 2, generator=g) * 60
    scores = torch.rand(n_boxes, nc, generator=g) * 0.3
    main = torch.randint(0, nc, (n_boxes,), generator=g)
    scores[torch.arange(n_boxes), main] = torch.rand(n_boxes, generator=g)
    pred = torch.cat([centers, wh, scores], dim=1).T.unsqueeze(0)   # (1, 4+nc, N)
    return pred


def run_nms(pred, conf, classes, max_det):
    out = non_max_suppression(pred.clone(), conf_thres=conf, iou_thres=0.7, classes=classes,
                              agnostic=False, max_det=max_det)[0]
    return out.cpu().numpy().astype(np.float32)


def filter_like_cache(arr, conf, classes, max_det):
    keep = arr[:, 4] > conf
    if classes is not None:
        keep &= np.isin(arr[:, 5].astype(int), np.array(classes))
    return arr[keep][:max_det]


def test_1_nms_prefix_property():
    for n_boxes, seed in [(400, 0), (3000, 1)]:
        pred = raw_predictions(n_boxes, seed=seed)
        cache = run_nms(pred, 0.05, None, 3000)
        for conf, classes, max_det in [(0.25, [2, 3, 5, 7], 300), (0.10, [2, 3, 5, 7], 300),
                                       (0.25, None, 300), (0.10, [1, 2, 3, 5, 7], 300), (0.5, [2], 50)]:
            direct = run_nms(pred, conf, classes, max_det)
            via_cache = filter_like_cache(cache, conf, classes, max_det)
            same = direct.shape == via_cache.shape and np.array_equal(direct, via_cache)
            _check(f"N={n_boxes} conf>{conf} classes={classes} max_det={max_det}: "
                   f"loc cache == NMS truc tiep ({len(direct)} box)", same,
                   "" if same else f"direct {direct.shape} vs cache {via_cache.shape}")
        # tinh chat sap xep giam dan (dieu kien cua phep cat tien to)
        _check(f"N={n_boxes}: cache theo thu tu diem giam dan", np.all(np.diff(cache[:, 4]) <= 0))
        _check(f"N={n_boxes}: cache khong cham tran max_det=3000", len(cache) < 3000, f"{len(cache)} box")

    # TRUONG HOP CACHE BI CAT (truncated): tinh chat tien to KHONG con dung - dung nhu
    # tai lieu da canh bao. Day la ly do npz co co `truncated` va VideoDetections tu
    # choi frame bi cat o che do strict.
    pred = raw_predictions(6000, seed=2)
    capped = run_nms(pred, 0.05, None, 3000)
    uncapped = run_nms(pred, 0.05, None, 10**6)
    direct = run_nms(pred, 0.25, [2, 3, 5, 7], 300)
    _check("N=6000: cache cham tran 3000 (bi cat) -> co the phat hien qua len == max_det",
           len(capped) == 3000 and len(uncapped) > 3000)
    _check("N=6000: cache BI CAT -> loc KHONG con bang NMS truc tiep (canh bao trong tai lieu la dung)",
           not np.array_equal(direct, filter_like_cache(capped, 0.25, [2, 3, 5, 7], 300)))
    _check("N=6000: cache KHONG bi cat -> loc == NMS truc tiep",
           np.array_equal(direct, filter_like_cache(uncapped, 0.25, [2, 3, 5, 7], 300)))


def test_2_boxes_dtype_matches_ultralytics_path():
    arr = np.array([[10.5, 20.25, 110.75, 220.125, 0.9, 2], [0, 0, 5, 5, 0.3, 7]], dtype=np.float32)
    ours = to_boxes(arr, (540, 960))
    theirs = Boxes(torch.from_numpy(arr), (540, 960)).cpu().numpy()   # duong cua ultralytics
    _check("Boxes.data float32, bang nhau", ours.data.dtype == np.float32 and np.array_equal(ours.data, theirs.data))
    _check("Boxes.xywh float32, bang nhau", ours.xywh.dtype == np.float32 and np.array_equal(ours.xywh, theirs.xywh))
    _check("Boxes.conf / cls bang nhau", np.array_equal(ours.conf, theirs.conf) and np.array_equal(ours.cls, theirs.cls))
    empty = to_boxes(np.zeros((0, 6), np.float32), (540, 960))
    _check("Boxes rong hop le (0 box)", len(empty) == 0 and empty.xywh.shape == (0, 4))


def test_3_roundtrip_and_filtering():
    with tempfile.TemporaryDirectory() as td:
        cd = Path(td) / "cache_x"
        spec = DetectorSpec(backend="gt-synthetic", conf=0.05, classes=None, max_det=3000,
                            dataset="fixture", part="val")
        write_meta(cd, spec)
        frames = np.array([1, 2, 3, 4], dtype=np.int32)
        data = np.array([[0, 0, 10, 10, 0.9, 2], [5, 5, 15, 15, 0.2, 7], [1, 1, 9, 9, 0.12, 3],   # frame 1
                         [20, 20, 30, 30, 0.6, 5]],                                                # frame 3
                        dtype=np.float32)
        arrays = dict(frame_idx=frames, n_per_frame=np.array([3, 0, 1, 0], np.int32),
                      unreadable=np.array([0, 0, 0, 1], bool), truncated=np.zeros(4, bool),
                      time_s=np.arange(4) * 0.04, data=data, orig_shape=np.array([540, 960], np.int32))
        save_video(cd, "v1", arrays)
        vd = VideoDetections(cd / "v1.npz", conf_min=0.10, classes=[2, 3, 5, 7], max_det=300)
        got = list(vd.iter_frames())
        _check("4 frame ke ca frame rong / unreadable", [g[0] for g in got] == [1, 2, 3, 4] and got[3][2] is True)
        _check("frame 1: loc conf>0.10 -> 3 box, tat ca lop cho phep", len(got[0][3]) == 3)
        vd2 = VideoDetections(cd / "v1.npz", conf_min=0.25, classes=[2, 5], max_det=300)
        f1 = vd2.frame(0)
        _check("frame 1: loc conf>0.25 & lop {2,5} -> 1 box", len(f1) == 1 and f1[0, 5] == 2)
        vd3 = VideoDetections(cd / "v1.npz", conf_min=0.10, classes=None, max_det=2)
        _check("max_det cat tien to", len(vd3.frame(0)) == 2 and vd3.frame(0)[0, 4] == np.float32(0.9))
        vd4 = VideoDetections(cd / "v1.npz", conf_min=0.10, classes=None, max_det=300, class_map={7: 2, 3: 2})
        _check("class_map gop lop", set(vd4.frame(0)[:, 5].tolist()) == {2.0})
        _check("frame rong -> mang (0,6) float32", vd.frame(1).shape == (0, 6) and vd.frame(1).dtype == np.float32)
        meta = json.loads((cd / "meta.json").read_text())
        _check("check_compatible: conf thap hon cache -> loi", bool(check_compatible(meta, 0.01, None, 300)))
        _check("check_compatible: max_det lon hon cache -> loi", bool(check_compatible(meta, 0.25, None, 5000)))
        _check("check_compatible: hop le -> khong loi", check_compatible(meta, 0.25, [2, 3], 300) == [])
        meta2 = dict(meta, classes=[2, 3])
        _check("check_compatible: lop ngoai cache -> loi", bool(check_compatible(meta2, 0.25, [2, 5], 300)))
        try:
            write_meta(cd, DetectorSpec(backend="ultralytics", dataset="fixture", part="val"))
            _check("write_meta tu choi thong so STRICT khac", False)
        except RuntimeError:
            _check("write_meta tu choi thong so STRICT khac", True)


def test_4_gt_synthetic_deterministic():
    gt = pd.DataFrame({"video": ["v"] * 6, "frame": [1, 1, 2, 2, 3, 3], "track_id": [1, 2] * 3,
                       "bb_left": [10, 100, 12, 102, 14, 104], "bb_top": [10, 50, 11, 51, 12, 52],
                       "bb_width": [40] * 6, "bb_height": [30] * 6, "category": ["car", "motorcycle"] * 3})
    frames = [FrameRef(i, None, (i - 1) * 0.2) for i in (1, 2, 3)]
    c2c = {"car": 2, "motorcycle": 3}
    a = build_video_gt(gt, frames, DetectorSpec(backend="gt-synthetic", gt_jitter=0.05, gt_dropout=0.0, gt_seed=0), (720, 1280), c2c)
    b = build_video_gt(gt, frames, DetectorSpec(backend="gt-synthetic", gt_jitter=0.05, gt_dropout=0.0, gt_seed=0), (720, 1280), c2c)
    c = build_video_gt(gt, frames, DetectorSpec(backend="gt-synthetic", gt_jitter=0.05, gt_dropout=0.0, gt_seed=1), (720, 1280), c2c)
    _check("gt-synthetic: cung seed -> giong het", np.array_equal(a["data"], b["data"]))
    _check("gt-synthetic: seed khac -> khac", not np.array_equal(a["data"], c["data"]))
    _check("gt-synthetic: du 6 box, lop COCO dung", len(a["data"]) == 6 and set(a["data"][:, 5].tolist()) == {2.0, 3.0})
    d = build_video_gt(gt, frames, DetectorSpec(backend="gt-synthetic", gt_jitter=0.0, gt_dropout=0.9, gt_seed=0), (720, 1280), c2c)
    _check("gt-synthetic: dropout 0.9 bo sot", len(d["data"]) < 6)
    _check("gt-synthetic: box hop le trong anh", np.all(a["data"][:, 2] > a["data"][:, 0]) and np.all(a["data"][:, :4] >= 0))


def main() -> int:
    failed = 0
    for t in [test_1_nms_prefix_property, test_2_boxes_dtype_matches_ultralytics_path,
              test_3_roundtrip_and_filtering, test_4_gt_synthetic_deterministic]:
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
