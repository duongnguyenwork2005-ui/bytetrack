"""
test_track_video_file.py -- Kiem he thong tracking tren PC: video -> detector -> tracker -> ket qua.

TANG 1 (khong can GPU, nhanh): adapter `LiveVideoDetections` va nguon frame.
  1. Bo loc cua adapter TRUNG KHIT `VideoDetections.frame()` (cung `filter_dets`) tren cung mang tho.
  2. Frame khong doc duoc -> yield unreadable=True, mang rong, VAN duoc ghi vao cache arrays.
  3. `cache_arrays()` ghi ra npz -> `VideoDetections` doc lai cho DUNG cac mang da loc (vong tron).
  4. Nguon khong hop le: khong ton tai / duoi la / thu muc khong anh / thu muc thieu fps -> loi ro rang.
  5. `on_frame` tra False -> `track_video` dung som nhung VAN giu hang cua frame cuoi.

TANG 1b (review 8abb368 -- fixture: detector/imread gia lap, tracker THAT, chay `main()` that):
 F1. Bo doc anh mat ~20 ms/frame -> `decode_ms` ~20 (khong phai ~0), `frame_total_ms` bao phu no,
     overhead ghi rieng; CSV pipeline co du cot de tinh lai tong.
 F2. live -> save-cache -> replay:
     (a) thong thuong: ca hai exit 0, parity DAT;
     (b) `--det-conf 0.5 --conf-min 0.1`: live TU CHOI SOM (exit 2) bang cung guard check_compatible;
     (c) `--det-max-det 1 --max-det 1`, 1 det/frame: mac dinh live exit 3 (cham tran, khong ghi ket qua);
         voi `--allow-truncated`: live exit 0, replay (lenh sinh ra CO --allow-truncated) exit 0, parity DAT;
     (d) `max_det` consumer > producer: live exit 2.
 F4. Thu muc anh: 1,2,10 -> theo SO; 001,002,010 -> nhu nhau; tien to khac nhau / khong co so -> loi ro;
     --frames-manifest -> dung thu tu manifest; frames_manifest.json duoc ghi ra.
 Q1/Q2 (review 923305d). Bam q (GUI gia lap) o frame 1 / frame 2, warm-up 0: moi dong pipeline huu han
     (frame_total + overhead, show_ms frame cuoi > 0), n_frame_tong_ket == so dong, sum(stage)+overhead ==
     frame_total, frames_manifest tach `nguon` (3) / `da_xu_ly`, tracking + cache du va replay tuong duong.
 S1/S2 (review 923305d). Nguon 'traffic clip' + YAML 'cfg dir/ekf ctrv.yaml': `replay_argv` giu moi gia
     tri la MOT phan tu, chuoi `replay` quote theo shell nen tang; replay CHAY TU replay_argv (khong
     .split()) exit 0, parity DAT; ca --allow-truncated cung vay.

TANG 2 (can detector that; dung GPU neu co, khong thi CPU voi it frame hon):
  6. LIVE tren sequence DETRAC + --save-cache, roi REPLAY bang run_tracking.py tu cache do
     -> parity DAT (bit-identical): cung detector, cung loc, cung vong lap.
  7. Detection tho cua live so voi cache GOC cua thi nghiem: cung so detection tung frame va cung
     lop (bao cao muc lech bbox/conf -- nhieu GPU, khong phai loi).
  8. File .mp4 THAT (ma hoa tu anh) + --save-video: exit 0, video ra doc duoc va du frame,
     run_meta.json du truong tai hien, ket qua duoc ghi.
  9. Video toan frame DEN (khong co xe): exit 0, 0 hang tracking, timing co DU moi frame
     (vong doi van duoc cap nhat o frame rong).
 10. Checkpoint khong ton tai -> exit 2, thong bao ro; nguon khong ton tai -> exit != 0.

Ket qua tang 2 KHONG dung de tuning; chi kiem tich hop.

    python src/test_track_video_file.py
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
_RESULTS: list[tuple[str, bool]] = []
PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]


def _check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines()[-6:]:
            print(f"         {line}")
    assert ok, name


class _NguonGia:
    """Nguon frame gia: 4 frame, frame 3 khong doc duoc."""
    fps, orig_shape, regions = 5.0, (720, 1280), []
    ten, loai, duong_dan, n_frames_khai_bao = "gia", "fake", "gia", 4

    def __iter__(self):
        for i in range(1, 5):
            img = None if i == 3 else np.zeros((720, 1280, 3), dtype=np.uint8)
            yield i, (i - 1) / self.fps, img, 0.5          # (idx, t_s, img, decode_ms)

    def dong(self):
        pass


class _DetectorGia:
    """Tra ve detection co dinh theo frame, DA theo thu tu NMS (giam dan conf)."""
    def __init__(self):
        self.goi = 0
        self.theo_frame = {
            1: np.array([[10, 10, 50, 50, .9, 2], [60, 60, 90, 90, .3, 3], [100, 100, 120, 120, .2, 2],
                         [5, 5, 8, 8, .8, 0]], dtype=np.float32),
            2: np.array([[12, 10, 52, 50, .85, 2]], dtype=np.float32),
            4: np.zeros((0, 6), dtype=np.float32),
        }

    def predict(self, img, **kw):
        self.goi += 1
        return [_R(self.theo_frame[self.goi if self.goi < 3 else 4])]


class _R:
    def __init__(self, det):
        self.boxes = _B(det)


class _B:
    def __init__(self, det):
        import torch
        self.data = torch.from_numpy(det)


def main() -> int:  # noqa: C901
    failed = 0
    tmp = Path(tempfile.mkdtemp(prefix="tvf_"))
    try:
        import det_cache
        import run_tracking
        import track_video_file as tvf
        import pandas as pd

        # ================= TANG 1 =================
        spec = det_cache.DetectorSpec(conf=0.05, classes=None, max_det=3000)
        det = _DetectorGia()
        vd = tvf.LiveVideoDetections(_NguonGia(), det, spec, conf_min=0.25, classes=[2, 3], max_det=1,
                                     class_map=None)
        ra = list(vd.iter_frames())
        # frame 1: 4 det tho -> loc conf>0.25 & lop {2,3} -> [.9 cls2, .3 cls3] -> [:1] -> 1 det
        f1 = ra[0][3]
        mong = det_cache.filter_dets(det.theo_frame[1], 0.25, np.array([2, 3]), 1, None)
        _check("adapter loc y het `filter_dets` / `VideoDetections.frame` (conf, lop, [:max_det])",
               np.array_equal(f1, mong) and len(f1) == 1 and f1[0, 5] == 2, f"f1={f1.tolist()}")
        _check("frame khong doc duoc -> unreadable=True, mang rong, VAN co trong cache arrays",
               ra[2][2] is True and len(ra[2][3]) == 0 and vd.raw["unreadable"][2] is True
               and len(vd.raw["frame_idx"]) == 4, f"raw.unreadable={vd.raw['unreadable']}")

        arrs = vd.cache_arrays()
        cdir = tmp / "cache"
        cdir.mkdir()
        np.savez_compressed(cdir / "gia.npz", **arrs)
        vd2 = det_cache.VideoDetections(cdir / "gia.npz", 0.25, [2, 3], 1)
        lai = [(fi, u, a) for fi, _, u, a in vd2.iter_frames()]
        _check("cache_arrays() -> npz -> VideoDetections doc lai: cung frame, cung unreadable, cung mang loc",
               [x[0] for x in lai] == [1, 2, 3, 4] and lai[2][1] is True
               and all(np.array_equal(lai[i][2], ra[i][3]) for i in range(4)),
               f"frame={[x[0] for x in lai]}")

        loi = {}
        for ten, kw in (("khong ton tai", dict(source=str(tmp / "khong_co.mp4"))),
                        ("duoi la", dict(source=str(tmp / "cache" / "gia.npz"))),
                        ("thu muc khong anh", dict(source=str(cdir))),):
            try:
                tvf.NguonFrame(kw["source"], None, "test", None, None, None)
                loi[ten] = None
            except Exception as e:  # noqa: BLE001
                loi[ten] = type(e).__name__
        d_anh = tmp / "anh"
        d_anh.mkdir()
        import cv2
        cv2.imwrite(str(d_anh / "a.jpg"), np.zeros((64, 64, 3), np.uint8))
        try:
            tvf.NguonFrame(str(d_anh), None, "test", None, None, None)
            loi["thu muc thieu fps"] = None
        except ValueError as e:
            loi["thu muc thieu fps"] = "ValueError" if "fps" in str(e) else str(e)
        _check("nguon khong hop le -> loi ro rang (khong ton tai / duoi la / khong anh / thieu fps)",
               loi == {"khong ton tai": "FileNotFoundError", "duoi la": "ValueError",
                       "thu muc khong anh": "FileNotFoundError", "thu muc thieu fps": "ValueError"}, f"{loi}")

        # 5. on_frame tra False -> dung som, giu hang frame cuoi
        cfg = run_tracking.load_tracker_cfg("bytetrack.yaml", {"track_buffer": 6})
        vd3 = det_cache.VideoDetections(cdir / "gia.npz", 0.25, [2, 3], 5)
        goi = []

        def stop_at_2(fi, out):
            goi.append(fi)
            return fi < 2
        df, _, df_t = run_tracking.track_video(vd3, cfg, 0.2, "steps", None, "gia", on_frame=stop_at_2)
        _check("on_frame tra False o frame 2 -> dung som; timing co dung 2 frame, hang cua frame 2 VAN duoc giu",
               goi == [1, 2] and list(df_t.frame) == [1, 2] and (len(df) == 0 or df.frame.max() <= 2)
               and (2 in set(df.frame) or len(df) == 0),
               f"goi={goi} timing={list(df_t.frame)} rows_frame={sorted(set(df.frame))}")

        # ================= TANG 1b: fixture qua main() that (review 8abb368) =================
        import contextlib
        import io
        import time
        import types
        import ultralytics
        import nano_parity as npar

        DELAY = {"ms": 0.0}
        DETS = {"queue": None, "mac_dinh": np.array([[10, 10, 50, 50, 0.9, 2]], dtype=np.float32)}
        _imread0, _detect0, _yolo0 = cv2.imread, det_cache.detect_frame, ultralytics.YOLO

        def _imread_gia(pth, *aa, **kk):
            time.sleep(DELAY["ms"] / 1000.0)
            return np.zeros((240, 320, 3), np.uint8)

        def _detect_gia(model, img, spec):
            q = DETS["queue"]
            return (q.pop(0) if q else DETS["mac_dinh"]).copy()

        def _main(mod, argv):
            old = sys.argv; sys.argv = ["x"] + argv
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    try:
                        rc = mod.main()
                    except SystemExit as e:
                        rc = int(e.code or 0)
                    except Exception as e:  # noqa: BLE001
                        rc = -1; buf.write(f"[EXC] {type(e).__name__}: {e}")
            finally:
                sys.argv = old
            return rc, buf.getvalue()

        ck_gia = tmp / "gia.pt"; ck_gia.write_bytes(b"fake")
        anh_gia = tmp / "anh_gia"; anh_gia.mkdir()
        for i in (1, 2, 3):
            (anh_gia / f"{i:03d}.jpg").write_bytes(b"x")
        try:
            cv2.imread, det_cache.detect_frame = _imread_gia, _detect_gia
            ultralytics.YOLO = lambda w: types.SimpleNamespace(names={2: "car"})
            base = ["--source", str(anh_gia), "--fps", "5", "--model", str(ck_gia), "--device", "cpu",
                    "--classes", "2", "--warmup-frames", "0"]

            # --- F1 ---
            DELAY["ms"] = 20.0
            o1 = tmp / "f1"
            rc, log = _main(tvf, base + ["--out", str(o1)])
            m = json.loads((o1 / "run_meta.json").read_text(encoding="utf-8"))["thoi_gian"]
            tb_ = m["trung_binh_ms"]
            csv1 = pd.read_csv(o1 / "pipeline" / "anh_gia.csv")
            cot = {"decode_ms", "mask_ms", "detect_ms", "t_update_ms", "render_ms", "write_ms", "show_ms",
                   "frame_total_ms", "overhead_ms"}
            tong_lai = (csv1[["decode_ms", "mask_ms", "detect_ms", "t_update_ms", "render_ms", "write_ms",
                              "show_ms", "overhead_ms"]].sum(axis=1) - csv1["frame_total_ms"]).abs().max()
            _check("F1. bo doc ~20 ms/frame -> decode_ms ~20 (truoc: ~0), frame_total bao phu, "
                   "CSV du cot va tong stage + overhead == frame_total",
                   rc == 0 and 18.0 <= tb_["decode_ms"] <= 30.0 and tb_["frame_total_ms"] >= tb_["decode_ms"]
                   and cot <= set(csv1.columns) and tong_lai < 1e-6
                   and m["wall_vong_lap_s"] * 1000 >= 3 * 18.0,
                   f"decode={tb_['decode_ms']:.2f} total={tb_['frame_total_ms']:.2f} "
                   f"wall_vong_lap={m['wall_vong_lap_s']*1000:.1f}ms tong_lai={tong_lai:.2e} rc={rc}")
            DELAY["ms"] = 0.0

            # --- F2 ---
            def live_replay(tag, live_extra, replay_extra, src_dir=None):
                cid = f"tvf_f2_{tag}"
                shutil.rmtree(det_cache.CACHE_ROOT / cid, ignore_errors=True)
                o = tmp / f"f2_{tag}"
                src = src_dir or anh_gia
                argv_live = [x for x in base]
                argv_live[argv_live.index("--source") + 1] = str(src)
                rc1, l1 = _main(tvf, argv_live + ["--out", str(o), "--save-cache", cid] + live_extra)
                meta = json.loads((o / "run_meta.json").read_text(encoding="utf-8")) if (o / "run_meta.json").exists() else {}
                split = f"TVF_F2_{tag}"
                shutil.rmtree(ROOT / "data/processed/trackers" / split, ignore_errors=True)
                rc2, l2, pr = None, "", None
                if rc1 == 0 and meta.get("replay_argv"):
                    # (review 923305d) dung `replay_argv` CO CAU TRUC: thay hai phan tu placeholder, KHONG .split()
                    argv = [split if x == "<split>" else ("cv" if x == "<ten>" else x) for x in meta["replay_argv"]][2:]
                    rc2, l2 = _main(run_tracking, argv)
                    ten_src = src.name
                    f_live, f_rep = o / "data" / f"{ten_src}.txt", ROOT / "data/processed/trackers" / split / "cv" / "data" / f"{ten_src}.txt"
                    if f_rep.exists():
                        pr = npar.so_mot_video(f_live, f_rep)
                shutil.rmtree(ROOT / "data/processed/trackers" / split, ignore_errors=True)
                shutil.rmtree(det_cache.CACHE_ROOT / cid, ignore_errors=True)
                return rc1, rc2, pr, meta, (l1 + l2)[-300:]

            DETS["queue"] = None
            a_ = live_replay("thuong", ["--conf-min", "0.25"], [])
            _check("F2a. thong thuong: live exit 0, replay (lenh tu run_meta) exit 0, parity DAT lech 0",
                   a_[0] == 0 and a_[1] == 0 and a_[2] and a_[2]["dat"] and a_[2]["max_bbox"] == 0.0,
                   f"live={a_[0]} replay={a_[1]} parity={a_[2] and a_[2].get('ly_do','DAT')}")
            b_ = live_replay("conf", ["--det-conf", "0.5", "--conf-min", "0.1"], [])
            _check("F2b. --det-conf 0.5 --conf-min 0.1 -> live TU CHOI SOM exit 2 (cung guard check_compatible), "
                   "khong tao cache",
                   b_[0] == 2 and b_[1] is None and not b_[3], f"live={b_[0]} log={b_[4][-160:]}")
            c1 = live_replay("tran", ["--det-max-det", "1", "--max-det", "1"], [])
            _check("F2c. --det-max-det 1, 1 det/frame, mac dinh -> live exit 3 (cham tran, KHONG ghi ket qua), "
                   "y het VideoDetections tu choi",
                   c1[0] == 3 and c1[1] is None and not c1[3], f"live={c1[0]} log={c1[4][-160:]}")
            c2 = live_replay("tran_ok", ["--det-max-det", "1", "--max-det", "1", "--allow-truncated"], [])
            _check("F2c'. cung ca voi --allow-truncated -> live exit 0, meta ghi allow_truncated + "
                   "replay_trang_thai, lenh replay CO --allow-truncated, replay exit 0, parity DAT",
                   c2[0] == 0 and c2[3].get("loc", {}).get("allow_truncated") is True
                   and "--allow-truncated" in c2[3].get("replay", "") and "cham tran" in c2[3].get("replay_trang_thai", "")
                   and c2[1] == 0 and c2[2] and c2[2]["dat"],
                   f"live={c2[0]} replay={c2[1]} parity={c2[2] and c2[2].get('ly_do','DAT')} "
                   f"replay_cmd_ok={'--allow-truncated' in c2[3].get('replay','')}")
            d_ = live_replay("maxdet", ["--det-max-det", "5", "--max-det", "10"], [])
            _check("F2d. max_det consumer (10) > producer (5) -> live exit 2",
                   d_[0] == 2 and d_[1] is None, f"live={d_[0]} log={d_[4][-160:]}")

            # --- 923305d ca 1: bam q o frame 1 / frame 2, warm-up 0 (GUI gia lap) ---
            _named0, _imshow0, _wait0, _destroy0 = cv2.namedWindow, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows
            GUI = {"n": 0, "q_tai": 0}

            def _waitkey_gia(ms=1):
                GUI["n"] += 1
                return ord("q") if GUI["n"] == GUI["q_tai"] else 255
            try:
                cv2.namedWindow = lambda *aa, **kk: None
                cv2.imshow = lambda *aa, **kk: None
                cv2.destroyAllWindows = lambda *aa, **kk: None
                cv2.waitKey = _waitkey_gia
                DELAY["ms"] = 20.0
                for q_tai in (1, 2):
                    GUI["n"], GUI["q_tai"] = 0, q_tai
                    cid_q = f"tvf_q{q_tai}"
                    shutil.rmtree(det_cache.CACHE_ROOT / cid_q, ignore_errors=True)
                    oq = tmp / f"q{q_tai}"
                    rc_q, log_q = _main(tvf, base + ["--out", str(oq), "--show", "--save-cache", cid_q])
                    mq = json.loads((oq / "run_meta.json").read_text(encoding="utf-8"))
                    pq = pd.read_csv(oq / "pipeline" / "anh_gia.csv")
                    fmq = json.loads((oq / "frames_manifest.json").read_text(encoding="utf-8"))
                    tq = mq["thoi_gian"]
                    huu_han = pq.frame_total_ms.notna() & pq.overhead_ms.notna()
                    tong_lai_q = (pq[["decode_ms", "mask_ms", "detect_ms", "t_update_ms", "render_ms", "write_ms",
                                     "show_ms", "overhead_ms"]].sum(axis=1) - pq["frame_total_ms"]).abs().max()
                    # replay tu cache 2 frame bang replay_argv
                    split_q = f"TVF_Q{q_tai}"
                    shutil.rmtree(ROOT / "data/processed/trackers" / split_q, ignore_errors=True)
                    argv_q = [split_q if x == "<split>" else ("cv" if x == "<ten>" else x) for x in mq["replay_argv"]][2:]
                    rc_rq, _ = _main(run_tracking, argv_q)
                    pr_q = npar.so_mot_video(oq / "data" / "anh_gia.txt",
                                             ROOT / "data/processed/trackers" / split_q / "cv" / "data" / "anh_gia.txt")
                    _check(f"Q{q_tai}. bam q o frame {q_tai} (warm-up 0): exit 0, dung_som, {q_tai} dong pipeline DEU huu han "
                           "(frame_total + overhead), show_ms cua frame cuoi > 0, n_frame_tong_ket == so dong huu han, "
                           "sum(stage)+overhead == frame_total, manifest tach nguon 3 / da_xu_ly, replay cache tuong duong",
                           rc_q == 0 and mq["ket_qua"]["dung_som"] is True and len(pq) == q_tai and huu_han.all()
                           and float(pq.show_ms.iloc[-1]) > 0 and tq["n_frame_tong_ket"] == int(huu_han.sum()) == q_tai
                           and tong_lai_q < 1e-6 and pq.frame_total_ms.iloc[-1] >= pq.decode_ms.iloc[-1]
                           and fmq["n_nguon"] == 3 and fmq["n_da_xu_ly"] == q_tai and fmq["da_xu_ly"] == list(range(1, q_tai + 1))
                           and rc_rq == 0 and pr_q["dat"] and pr_q["n_ref"] == q_tai,
                           f"rc={rc_q} n={len(pq)} huu_han={int(huu_han.sum())} tong_ket={tq['n_frame_tong_ket']} "
                           f"show_cuoi={float(pq.show_ms.iloc[-1]):.3f} tong_lai={tong_lai_q:.1e} "
                           f"manifest={fmq.get('n_nguon')}/{fmq.get('n_da_xu_ly')} replay={rc_rq} parity={pr_q.get('ly_do','DAT')}")
                    shutil.rmtree(ROOT / "data/processed/trackers" / split_q, ignore_errors=True)
                    shutil.rmtree(det_cache.CACHE_ROOT / cid_q, ignore_errors=True)
                DELAY["ms"] = 0.0
            finally:
                cv2.namedWindow, cv2.imshow, cv2.waitKey, cv2.destroyAllWindows = _named0, _imshow0, _wait0, _destroy0

            # --- 923305d ca 2: ten nguon va duong dan YAML co DAU CACH, replay tu replay_argv ---
            src_sp = tmp / "traffic clip"; src_sp.mkdir()
            for i in (1, 2, 3):
                (src_sp / f"{i:03d}.jpg").write_bytes(b"x")
            yaml_sp = tmp / "cfg dir" / "ekf ctrv.yaml"
            yaml_sp.parent.mkdir()
            shutil.copy(ROOT / "configs" / "bytetrack_ekf_ctrv.yaml", yaml_sp)
            sp1 = live_replay("space", ["--conf-min", "0.25", "--tracker-yaml", str(yaml_sp)], [], src_dir=src_sp)
            m_sp = sp1[3]
            _check("S1. nguon 'traffic clip' + YAML 'cfg dir/ekf ctrv.yaml': live exit 0; replay_argv giu MOI gia tri "
                   "la MOT phan tu; chuoi `replay` quote theo shell (Windows: list2cmdline); replay tu replay_argv exit 0, parity DAT",
                   sp1[0] == 0 and "traffic clip" in m_sp.get("replay_argv", []) and str(yaml_sp) in m_sp.get("replay_argv", [])
                   and '"traffic clip"' in m_sp.get("replay", "") and m_sp.get("replay_quote")
                   and sp1[1] == 0 and sp1[2] and sp1[2]["dat"],
                   f"live={sp1[0]} replay={sp1[1]} parity={sp1[2] and sp1[2].get('ly_do','DAT')} quote={m_sp.get('replay_quote')}")
            sp2 = live_replay("space_tran", ["--det-max-det", "1", "--max-det", "1", "--allow-truncated",
                                             "--tracker-yaml", str(yaml_sp)], [], src_dir=src_sp)
            _check("S2. cung nguon/YAML co dau cach + --allow-truncated: replay_argv co '--allow-truncated', replay exit 0, parity DAT",
                   sp2[0] == 0 and "--allow-truncated" in sp2[3].get("replay_argv", []) and sp2[1] == 0 and sp2[2] and sp2[2]["dat"],
                   f"live={sp2[0]} replay={sp2[1]} parity={sp2[2] and sp2[2].get('ly_do','DAT')}")

            # --- F4 ---
            def thu_tu(ten_files, manifest=None):
                d = tmp / ("f4_" + str(abs(hash(tuple(ten_files))) % 10**6))
                d.mkdir(exist_ok=True)
                for n in ten_files:
                    (d / n).write_bytes(b"x")
                ng = tvf.NguonFrame(str(d), None, "x", None, 5.0, None, manifest)
                return [Path(q).name for _, _, q in ng.paths]
            _check("F4a. 1.jpg 2.jpg 10.jpg -> theo SO: 1, 2, 10 (truoc: 1, 10, 2)",
                   thu_tu(["1.jpg", "2.jpg", "10.jpg"]) == ["1.jpg", "2.jpg", "10.jpg"])
            _check("F4b. 001 002 010 (zero-padding) -> nhu nhau",
                   thu_tu(["001.jpg", "002.jpg", "010.jpg"]) == ["001.jpg", "002.jpg", "010.jpg"])
            _check("F4c. img_7.jpg img_08.jpg img_10.jpg (cung tien to, padding lan lon) -> theo SO",
                   thu_tu(["img_10.jpg", "img_7.jpg", "img_08.jpg"]) == ["img_7.jpg", "img_08.jpg", "img_10.jpg"])
            loi4 = {}
            for ten, files in (("tien to khac", ["a1.jpg", "b2.jpg"]), ("khong co so", ["x.jpg", "y.jpg"]),
                               ("trung so", ["a01.jpg", "a1.jpg"])):
                try:
                    thu_tu(files); loi4[ten] = None
                except ValueError as e:
                    loi4[ten] = "frames-manifest" in str(e)
            _check("F4d. tien to khac nhau / khong co so / trung so -> ValueError co huong dan --frames-manifest",
                   all(v is True for v in loi4.values()), f"{loi4}")
            mf = tmp / "thu_tu.json"
            mf.write_text(json.dumps(["b2.jpg", "a1.jpg"]), encoding="utf-8")
            _check("F4e. --frames-manifest -> dung DUNG thu tu manifest (b2 truoc a1) du ten mo ho",
                   thu_tu(["a1.jpg", "b2.jpg"], str(mf)) == ["b2.jpg", "a1.jpg"])
            fm = json.loads((o1 / "frames_manifest.json").read_text(encoding="utf-8"))
            _check("F4f. run ghi frames_manifest.json: quy uoc + `nguon` (file, frame, time_s) + `da_xu_ly` (chay het: 3/3)",
                   fm["quy_uoc"] and [x["file"] for x in fm["nguon"]] == ["001.jpg", "002.jpg", "003.jpg"]
                   and [x["frame"] for x in fm["nguon"]] == [1, 2, 3] and abs(fm["nguon"][2]["time_s"] - 0.4) < 1e-9
                   and fm["n_nguon"] == 3 and fm["n_da_xu_ly"] == 3 and fm["da_xu_ly"] == [1, 2, 3] and fm["dung_som"] is False,
                   f"{fm.get('nguon')} da_xu_ly={fm.get('da_xu_ly')}")
        finally:
            cv2.imread, det_cache.detect_frame, ultralytics.YOLO = _imread0, _detect0, _yolo0

        # ================= TANG 2 =================
        import torch
        device = "0" if torch.cuda.is_available() else "cpu"
        n = 120 if device == "0" else 40
        ck = ROOT / "models" / "yolov8n.pt"
        if not ck.exists():
            print(f"  [NOT_RUN] tang 2: khong co checkpoint {ck}")
        else:
            out_live = tmp / "live"
            cache_id = "tvf_test_cache"
            cache_dir = det_cache.CACHE_ROOT / cache_id
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cmd = [PY, "src/track_video_file.py", "--dataset", "detrac", "--part", "test", "--video", "MVI_39031",
                   "--max-frames", str(n), "--tracker-yaml", "configs/bytetrack_ekf_ctrv.yaml",
                   "--classes", "2", "3", "5", "7", "--conf-min", "0.25", "--track-buffer-sec", "1.2",
                   "--elapsed-mode", "steps", "--device", device, "--save-cache", cache_id, "--out", str(out_live)]
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            _check(f"6a. LIVE tren DETRAC MVI_39031 ({n} frame, device={device}) exit 0, co data/timing/pipeline/run_meta",
                   r.returncode == 0 and (out_live / "data" / "MVI_39031.txt").exists()
                   and (out_live / "timing" / "MVI_39031.csv").exists()
                   and (out_live / "pipeline" / "MVI_39031.csv").exists()
                   and (out_live / "run_meta.json").exists() and (cache_dir / "MVI_39031.npz").exists(),
                   (r.stdout + r.stderr)[-500:])
            split = "TVFTEST"
            trk = ROOT / "data" / "processed" / "trackers" / split
            if trk.exists():
                shutil.rmtree(trk)
            cmd2 = [PY, "src/run_tracking.py", "--cache-id", cache_id, "--split-name", split,
                    "--tracker-yaml", "configs/bytetrack_ekf_ctrv.yaml", "--tracker-name", "ekf",
                    "--videos", "MVI_39031", "--conf-min", "0.25", "--classes", "2", "3", "5", "7",
                    "--max-det", "300", "--fps", "25", "--track-buffer-sec", "1.2", "--elapsed-mode", "steps"]
            r2 = subprocess.run(cmd2, cwd=ROOT, capture_output=True, text=True)
            import nano_parity as npar
            pr = npar.so_mot_video(out_live / "data" / "MVI_39031.txt", trk / "ekf" / "data" / "MVI_39031.txt")
            _check("6b. REPLAY tu cache vua luu bang run_tracking.py -> parity DAT, lech 0 (bit-identical)",
                   r2.returncode == 0 and pr["dat"] and pr["max_bbox"] == 0.0 and pr["max_conf"] == 0.0,
                   f"exit={r2.returncode} parity={pr.get('ly_do', 'DAT')} n={pr.get('n_ref')}")
            shutil.rmtree(trk, ignore_errors=True)

            # 7. so detection tho voi cache GOC (neu co)
            goc = det_cache.CACHE_ROOT / "yolov8n-coco_conf0.05_all" / "MVI_39031.npz"
            if goc.exists():
                # `with`: NpzFile giu file mo -> tren Windows rmtree(cache_dir) se that bai am tham
                with np.load(cache_dir / "MVI_39031.npz") as za, np.load(goc) as zb:
                    a = {k: za[k] for k in ("frame_idx", "n_per_frame", "data")}
                    b = {k: zb[k] for k in ("n_per_frame", "data")}
                k = len(a["frame_idx"])
                cung_n = np.array_equal(a["n_per_frame"], b["n_per_frame"][:k])
                if cung_n:
                    tot = int(a["n_per_frame"].sum())
                    d = np.abs(a["data"][:tot] - b["data"][:tot])
                    cls_ok = bool(np.array_equal(a["data"][:tot, 5], b["data"][:tot, 5]))
                    _check("7. detection tho live vs cache GOC thi nghiem: cung so det tung frame, cung lop; "
                           f"lech bbox {d[:, :4].max():.1e} px / conf {d[:, 4].max():.1e} (nhieu GPU, bao cao)",
                           cls_ok and d[:, :4].max() < 1e-2, f"cung_n={cung_n} cls_ok={cls_ok}")
                else:
                    _check("7. detection tho live vs cache GOC: cung so det tung frame", False,
                           "so detection tung frame KHAC -> detector khong tai lap duoc tren may nay")
            shutil.rmtree(cache_dir, ignore_errors=True)

            # 8. file .mp4 THAT + --save-video
            from mot_datasets import get_dataset
            frames = get_dataset("detrac").frames("MVI_39031", "test")[:60]
            mp4 = tmp / "xe.mp4"
            w = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 25, (960, 540))
            for f in frames:
                w.write(cv2.imread(str(f.path)))
            w.release()
            out_mp4 = tmp / "mp4"
            r3 = subprocess.run([PY, "src/track_video_file.py", "--source", str(mp4), "--tracker-yaml", "bytetrack.yaml",
                                 "--classes", "2", "3", "5", "7", "--device", device, "--save-video",
                                 "--out", str(out_mp4)], cwd=ROOT, capture_output=True, text=True)
            vout = out_mp4 / "xe_tracked.mp4"
            cap = cv2.VideoCapture(str(vout)) if vout.exists() else None
            n_out = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap and cap.isOpened() else -1
            if cap:
                cap.release()
            meta = json.loads((out_mp4 / "run_meta.json").read_text(encoding="utf-8")) if (out_mp4 / "run_meta.json").exists() else {}
            du = all(k in meta for k in ("nguon", "detector_spec", "loc", "tracker", "code", "ket_qua", "thoi_gian"))
            _check("8. file .mp4 THAT + --save-video: exit 0, video ra 60 frame, run_meta du truong "
                   "(nguon/detector_spec/loc/tracker/code/ket_qua/thoi_gian), fps doc tu metadata = 25",
                   r3.returncode == 0 and n_out == 60 and du and meta.get("nguon", {}).get("fps") == 25.0
                   and meta["nguon"]["sha256"] and meta["detector_spec"]["weights_sha256"]
                   and meta["tracker"]["yaml_sha256"] and meta["thoi_gian"]["n_frame_xu_ly"] == 60,
                   (r3.stdout + r3.stderr)[-400:] + f"\nn_out={n_out} du={du}")

            # 9. video toan frame DEN
            den = tmp / "den.mp4"
            w = cv2.VideoWriter(str(den), cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
            for _ in range(15):
                w.write(np.zeros((240, 320, 3), np.uint8))
            w.release()
            out_den = tmp / "den"
            r4 = subprocess.run([PY, "src/track_video_file.py", "--source", str(den), "--device", device,
                                 "--out", str(out_den)], cwd=ROOT, capture_output=True, text=True)
            import pandas as pd
            t_den = pd.read_csv(out_den / "timing" / "den.csv") if (out_den / "timing" / "den.csv").exists() else pd.DataFrame()
            mot_den = (out_den / "data" / "den.txt").read_text(encoding="utf-8") if (out_den / "data" / "den.txt").exists() else "x"
            _check("9. video toan frame DEN: exit 0, 0 hang tracking, timing van co DU 15 frame (vong doi cap nhat o frame rong)",
                   r4.returncode == 0 and mot_den.strip() == "" and len(t_den) == 15,
                   (r4.stdout + r4.stderr)[-300:] + f"\nn_timing={len(t_den)}")

            # 10. checkpoint khong ton tai / nguon khong ton tai
            r5 = subprocess.run([PY, "src/track_video_file.py", "--source", str(den), "--model", str(tmp / "khong_co.pt"),
                                 "--out", str(tmp / "x")], cwd=ROOT, capture_output=True, text=True)
            r6 = subprocess.run([PY, "src/track_video_file.py", "--source", str(tmp / "khong_co.mp4"),
                                 "--out", str(tmp / "y")], cwd=ROOT, capture_output=True, text=True)
            _check("10. checkpoint khong ton tai -> exit 2 + thong bao; nguon khong ton tai -> exit != 0",
                   r5.returncode == 2 and "checkpoint" in (r5.stdout + r5.stderr) and r6.returncode != 0,
                   f"r5 exit={r5.returncode} r6 exit={r6.returncode}")
    except AssertionError:
        failed += 1
    except Exception as e:  # noqa: BLE001
        failed += 1
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] loi khong mong doi: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
