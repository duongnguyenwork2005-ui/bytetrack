"""
test_nano_detector.py -- Kiem HOP DONG tien/hau xu ly cua `src/nano_detector.py` tren PC (co torch + ultralytics),
truoc khi chay TensorRT tren Nano. Phan TensorRT KHONG kiem duoc o day (chi co tren thiet bi).

PHAM VI GIA LAP (ghi ro): suy luan tho lay tu model .pt cua ultralytics bang torch FP32 (`TorchPtInfer`) thay cho
engine TensorRT; letterbox / decode / NMS / scale / ghi cache la MA THAT cua nano_detector.py (cu phap 3.6).

Cac phep kiem:
  1. letterbox(640) PIXEL-IDENTICAL voi ultralytics LetterBox((640,640), auto=False) tren 3 frame that (960x540).
  2. preprocess == torch permute/flip/div 255 cua BasePredictor (bang nhau chinh xac).
  3. decode_nms + scale_boxes tren CUNG dau ra tho == nms.non_max_suppression + ops.scale_boxes (5 frame that):
     cung so box, cung thu tu, bbox |d| <= 1e-4, conf |d| <= 1e-6, cls bang nhau.
  4. NMS tong hop: khac lop khong loai nhau (offset lop); cung lop IoU > thres bi loai; IoU == thres GIU (torchvision '>');
     max_det cat dung; classes loc dung; frame khong box -> (0, 6).
  5. Pipeline: NanoDetector (TorchPtInfer) tren 40 frame smoke (co che vung bo qua nhu cache PC) -> ghi cache dung
     dinh dang -> det_cache.open_cache(strict=True) doc duoc (frame rong giu, time_s dung) -> run_tracking.py replay
     exit 0 -> nano_det_compare voi cache PC (.pt rect 640x384): ti le ghep >= 0,8 (KHONG ky vong bit-identical:
     letterbox 640x640 vs rect 640x384 -- day la sai khac detector duoc do, khong phai loi).
  6. Cache ghi lai voi meta khac (conf khac) -> RuntimeError (khong tron cache).

    python src/test_nano_detector.py
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
for _l in (sys.stdout, sys.stderr):
    try:
        _l.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
ROOT = Path(__file__).resolve().parents[1]
_RESULTS: list[tuple[str, bool]] = []
SMOKE = ROOT / "data" / "extracted" / "DETRAC-Images" / "MVI_39031"
PC_CACHE = "yolov8n-coco_conf0.05_all"
REGIONS = ROOT / "data" / "processed" / "DETRAC-test" / "MVI_39031" / "gt" / "ignored_regions.txt"


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for ln in str(detail).strip().splitlines()[-6:]:
            print(f"         {ln}")
    assert ok, name


def main() -> int:  # noqa: C901
    failed = 0
    tmp = Path(tempfile.mkdtemp(prefix="nano_det_"))
    try:
        import cv2
        import torch
        import nano_detector as nd
        from ultralytics.data.augment import LetterBox
        from ultralytics.utils import nms as ul_nms, ops
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        frames = [SMOKE / f"img{i:05d}.jpg" for i in (1, 2, 3, 150, 300)]
        imgs = [cv2.imread(str(p)) for p in frames]
        assert all(im is not None for im in imgs), "thieu anh MVI_39031"

        # --- 1. letterbox ---
        lb_ul = LetterBox((640, 640), auto=False)
        ok1, det1 = True, []
        for im in imgs[:3]:
            a, r, pad = nd.letterbox(im, 640)
            b = lb_ul(image=im)
            ok1 &= a.shape == b.shape and np.array_equal(a, b)
            det1.append(f"shape {a.shape} r={r:.4f} pad={pad} identical={np.array_equal(a, b)}")
        _check("1. letterbox(640) pixel-identical voi ultralytics LetterBox(auto=False) tren 3 frame 960x540", ok1, "\n".join(det1))

        # --- 2. preprocess ---
        lb, _, _ = nd.letterbox(imgs[0], 640)
        x = nd.preprocess(lb)
        t = torch.from_numpy(lb).permute(2, 0, 1).flip(0).float().div(255).numpy()[None]
        _check("2. preprocess == permute/flip(BGR->RGB)/div255 cua BasePredictor (bang nhau chinh xac, float32 (1,3,640,640))",
               x.shape == (1, 3, 640, 640) and x.dtype == np.float32 and np.array_equal(x, t), f"max|d|={np.abs(x - t).max()}")

        ramp = np.arange(16 * 16 * 3).astype(np.uint8).reshape(16, 16, 3)
        cases = [ramp, ramp[::-1, ::-1, ::-1], ramp[::2, ::2], np.asfortranarray(ramp),
                 ramp[:1, :1], ramp.astype(np.float32), ramp.astype(np.uint16), ramp[:0]]
        for im in cases:
            before = im.copy()
            got = nd.preprocess(im)
            expected = torch.from_numpy(im.copy()).float().permute(2, 0, 1).flip(0).div(255).numpy()[None]
            assert got.dtype == np.float32 and got.shape == expected.shape and got.flags.c_contiguous
            assert got.tobytes() == expected.tobytes() and np.array_equal(im, before)
        retained = nd.preprocess(ramp)
        saved = retained.tobytes()
        nd.preprocess(np.zeros_like(ramp))
        _check("2b. preprocess: all uint8 values/layouts, fallback dtype, empty input, independent output",
               retained.tobytes() == saved)

        # --- 3. decode+NMS+scale vs ultralytics tren CUNG dau ra tho ---
        infer = nd.TorchPtInfer(str(ROOT / "models" / "yolov8n.pt"), device=dev)
        ok3, det3 = True, []
        for im in imgs:
            lb, _, _ = nd.letterbox(im, 640)
            raw = infer(nd.preprocess(lb))                              # (1, 84, 8400) float32
            mine = nd.decode_nms(raw, 0.05, 0.7, 3000)
            mine = nd.scale_boxes(mine, lb.shape[:2], im.shape[:2])
            ref = ul_nms.non_max_suppression(torch.from_numpy(raw.copy()), 0.05, 0.7, max_det=3000)[0]
            ref[:, :4] = ops.scale_boxes(lb.shape[:2], ref[:, :4], im.shape[:2])
            ref = ref.numpy()
            same_n = len(mine) == len(ref)
            if same_n and len(ref):
                d_box = float(np.abs(mine[:, :4] - ref[:, :4]).max()); d_conf = float(np.abs(mine[:, 4] - ref[:, 4]).max())
                same_cls = bool(np.array_equal(mine[:, 5], ref[:, 5]))
            else:
                d_box = d_conf = float("nan"); same_cls = False
            ok_i = same_n and (len(ref) == 0 or (d_box <= 1e-4 and d_conf <= 1e-6 and same_cls))
            ok3 &= ok_i
            det3.append(f"n={len(mine)}/{len(ref)} max|d bbox|={d_box:.2e} max|d conf|={d_conf:.2e} cls={same_cls} -> {ok_i}")
        _check("3. decode_nms + scale_boxes == nms.non_max_suppression + ops.scale_boxes tren CUNG dau ra tho (5 frame): "
               "cung so box/thu tu, bbox<=1e-4, conf<=1e-6, cls bang", ok3, "\n".join(det3))

        # --- 4. NMS tong hop ---
        def raw_tu_box(boxes_xyxy, confs, clss, nc=80):
            n = len(boxes_xyxy)
            p = np.zeros((4 + nc, n), np.float32)
            for i, ((x1, y1, x2, y2), c, k) in enumerate(zip(boxes_xyxy, confs, clss)):
                p[0, i], p[1, i], p[2, i], p[3, i] = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
                p[4 + k, i] = c
            return p[None]
        # a) cung box, khac lop -> ca hai giu
        r = nd.decode_nms(raw_tu_box([(10, 10, 50, 50), (10, 10, 50, 50)], [0.9, 0.8], [2, 7]), 0.05, 0.7, 3000)
        a_ok = len(r) == 2 and set(r[:, 5].tolist()) == {2.0, 7.0}
        # b) cung lop IoU > 0.7 -> giu 1 (diem cao hon)
        r = nd.decode_nms(raw_tu_box([(10, 10, 50, 50), (12, 12, 52, 52)], [0.8, 0.9], [2, 2]), 0.05, 0.7, 3000)
        b_ok = len(r) == 1 and abs(r[0, 4] - 0.9) < 1e-6
        # c) IoU == thres -> GIU (torchvision loai khi IoU > thres)
        # box A (0,0,10,10) dien tich 100; B (0,0,10,x): IoU = 10x/100 = 0.7 khi x = 7 -> inter 70, union 100 -> 0.7
        r = nd.decode_nms(raw_tu_box([(0, 0, 10, 10), (0, 0, 10, 7)], [0.9, 0.8], [2, 2]), 0.05, 0.7, 3000)
        c_ok = len(r) == 2
        # d) max_det
        r = nd.decode_nms(raw_tu_box([(i * 100, 0, i * 100 + 50, 50) for i in range(6)], [0.9 - i * 0.05 for i in range(6)], [2] * 6), 0.05, 0.7, 4)
        d_ok = len(r) == 4 and abs(r[0, 4] - 0.9) < 1e-6
        # e) classes loc
        r = nd.decode_nms(raw_tu_box([(0, 0, 50, 50), (100, 0, 150, 50)], [0.9, 0.9], [2, 0]), 0.05, 0.7, 3000, classes=[2])
        e_ok = len(r) == 1 and r[0, 5] == 2.0
        # f) rong
        r = nd.decode_nms(np.zeros((1, 84, 8400), np.float32), 0.05, 0.7, 3000)
        f_ok = r.shape == (0, 6) and r.dtype == np.float32
        _check("4. NMS tong hop: khac lop giu ca hai; cung lop IoU>0.7 giu 1; IoU==0.7 GIU; max_det cat; classes loc; rong -> (0,6)",
               a_ok and b_ok and c_ok and d_ok and e_ok and f_ok, f"a={a_ok} b={b_ok} c={c_ok} d={d_ok} e={e_ok} f={f_ok}")

        # --- 5. pipeline -> cache -> open_cache -> replay -> det compare ---
        import det_cache
        N = 40
        anh = [str(SMOKE / f"img{i:05d}.jpg") for i in range(1, N + 1)]
        regions = nd.doc_regions(str(REGIONS))
        det = nd.NanoDetector(infer, 640, 0.05, 0.7, 3000)
        arr, timing, tong = nd.chay_nguon(det, anh, 25.0, regions, None, 5, log_every=0)
        cid = "nano_test_torchpt"
        cdir = tmp / "det_cache" / cid
        meta = {"backend": "torch-pt-test", "weights": "models/yolov8n.pt", "weights_sha256": nd.sha256_file(str(ROOT / "models/yolov8n.pt")),
                "imgsz": 640, "conf": 0.05, "iou": 0.7, "classes": None, "max_det": 3000, "half": False, "agnostic_nms": False,
                "augment": False, "batch": 1, "device": dev, "mask_ignored_regions": True, "dataset": "video_file", "part": "images",
                "letterbox": "auto=False 640x640", "versions": {"numpy": np.__version__}, "git_commit": None, "created": "test"}
        info = nd.ghi_cache(str(cdir), "MVI_39031", arr["frame_idx"], arr["time_s"], arr["n_per"], arr["unreadable"], arr["truncated"],
                            arr["data"], arr["orig_shape"], meta, {"fps": 25.0})
        # doc bang det_cache (goc CACHE_ROOT tam)
        goc_cu = det_cache.CACHE_ROOT
        det_cache.CACHE_ROOT = tmp / "det_cache"
        try:
            vd = det_cache.open_cache(cid, "MVI_39031", conf_min=0.25, classes=[1, 2, 3, 5, 7], max_det=300, strict=True)
            n_fr = len(vd); ts_ok = np.allclose(vd.time_s, np.arange(N) / 25.0); shape_ok = vd.orig_shape == (540, 960)
            n_raw = int(vd.n_per_frame.sum())
        finally:
            det_cache.CACHE_ROOT = goc_cu
        # replay bang run_tracking (subprocess) voi CACHE_ROOT tam qua bien moi truong? -> chep cache vao repo tam thoi
        cdir_repo = det_cache.CACHE_ROOT / cid
        shutil.rmtree(cdir_repo, ignore_errors=True); shutil.copytree(cdir, cdir_repo)
        try:
            r = subprocess.run([sys.executable, str(ROOT / "src/run_tracking.py"), "--cache-id", cid, "--split-name", "NANO-test", "--tracker-yaml",
                                "bytetrack.yaml", "--tracker-name", "test-cv", "--videos", "MVI_39031", "--conf-min", "0.25", "--classes", "1", "2", "3", "5", "7",
                                "--max-det", "300", "--fps", "25", "--track-buffer-sec", "1.2", "--elapsed-mode", "steps"],
                               cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
                               env={**__import__("os").environ, "BYTETRACK_BACKEND": "np"})
            mot = ROOT / "data/processed/trackers/NANO-test/test-cv/data/MVI_39031.txt"
            n_mot = sum(1 for _ in open(mot, encoding="utf-8")) if mot.is_file() else 0
            pj = tmp / "det_compare.json"
            r2 = subprocess.run([sys.executable, str(ROOT / "src/nano_det_compare.py"), "--a", PC_CACHE, "--b", cid, "--video", "MVI_39031",
                                 "--max-frames", str(N), "--conf-min", "0.25", "--classes", "1", "2", "3", "5", "7", "--json", str(pj)],
                                cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            cmp_ = json.loads(pj.read_text(encoding="utf-8")) if pj.is_file() else {}
        finally:
            shutil.rmtree(cdir_repo, ignore_errors=True)
            shutil.rmtree(ROOT / "data/processed/trackers/NANO-test", ignore_errors=True)
        _check(f"5. NanoDetector(torch FP32, che vung bo qua) {N} frame -> ghi cache dung dinh dang -> open_cache strict OK "
               f"(frame/time_s/orig_shape) -> run_tracking exit 0 -> det_compare voi cache PC (.pt rect 640x384): ti le ghep >= 0,8 "
               f"(sai khac letterbox 640x640 la DUOC DO, khong ky vong bit-identical)",
               info["n_frames"] == N and n_fr == N and ts_ok and shape_ok and n_raw == info["n_boxes"] and r.returncode == 0 and n_mot > 0
               and r2.returncode == 0 and (cmp_.get("ti_le_ghep_a") or 0) >= 0.8 and (cmp_.get("ti_le_ghep_b") or 0) >= 0.8,
               f"cache: {info['n_frames']} frame {info['n_boxes']} box (rong {int(sum(1 for n in arr['n_per'] if n == 0))}) | open_cache n={n_fr} ts={ts_ok} shape={shape_ok} "
               f"| replay exit={r.returncode} mot={n_mot} hang | det_compare: cap={cmp_.get('n_cap')} ghep a={cmp_.get('ti_le_ghep_a')} b={cmp_.get('ti_le_ghep_b')} "
               f"IoU tb={cmp_.get('iou_trung_binh')} |dconf| tb={cmp_.get('d_conf_trung_binh')} max={cmp_.get('d_conf_lon_nhat')} trung lop={cmp_.get('ti_le_trung_lop')}"
               f"\ntiming tb sau warm-up (PC {dev}, torch FP32 -- KHONG phai so Nano): {json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in tong.items()})}")

        # --- 6. meta khac -> tu choi ghi them ---
        try:
            meta2 = dict(meta); meta2["conf"] = 0.1
            nd.ghi_cache(str(cdir), "MVI_39031", arr["frame_idx"], arr["time_s"], arr["n_per"], arr["unreadable"], arr["truncated"],
                         arr["data"], arr["orig_shape"], meta2, {})
            tu_choi = False
        except RuntimeError as e:
            tu_choi = "khac" in str(e)
        _check("6. ghi vao cache da co voi meta KHAC (conf) -> RuntimeError, khong tron cache", tu_choi)
    except AssertionError:
        failed += 1
    except Exception as e:  # noqa: BLE001
        failed += 1
        import traceback; traceback.print_exc()
        print(f"  [FAIL] loi khong mong doi: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
