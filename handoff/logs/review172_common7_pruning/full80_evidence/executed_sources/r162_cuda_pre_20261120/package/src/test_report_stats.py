"""
test_report_stats.py -- Thong ke detection / thoi gian cua cong cu bao cao phai khop DAU VAO
THAT cua tracker. (review `handoff/CODEX_REVIEW_ca93c1e.md`, finding P2)

LOI DA XAY RA trong `src/report_bdd_transfer.py`:
  - `_det_counts()` quet MOI `*.npz` trong thu muc cache va chi loc lop/confidence. No KHONG ap
    `max_det` theo tung frame nhu `VideoDetections.frame()`, va KHONG gioi han theo manifest cua
    thi nghiem. Cache dung chung nhieu thi nghiem nen quet thu muc la sai nguon.
  - `_detector_seconds()` cung quet moi `*.json`, va sidecar THIEU truong `seconds` bi coi nhu 0 giay.

Tai hien truoc khi sua: `handoff/logs/review7/repro_P2_truoc_khi_sua.log`
(1 detection that -> bao cao 5; 1 giay that -> bao cao 21 giay / 2 frame).

Cac phep kiem:
  1. Frame co nhieu detection hon `max_det` -> dem DUNG bang so tracker nhan (cat `[:max_det]`).
  2. Video ngoai manifest KHONG duoc dem.
  3. Thoi gian detector chi cong sidecar cua sequence trong manifest.
  4. Sidecar THIEU truong `seconds` -> bao thieu, KHONG cong 0.
  5. Sidecar thieu HAN -> bao thieu rieng.
  6. Du lieu day du -> `du_lieu_du` True va tong khop.
  7. So dem khop CHINH XAC voi `det_cache.open_cache(...).frame(i)` (duong doc cua runner).
  8. Thi nghiem khong co manifest -> bao loi ro rang, khong am tham quet thu muc.

    python src/test_report_stats.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_RESULTS: list[tuple[str, bool]] = []

SPEC = "fingerprint-cau-hinh-sinh-du-lieu"


def _check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines()[-6:]:
            print(f"         {line}")
    assert ok, name


def viet_cache(cdir: Path, ten: str, frames: list[list[list[float]]], giay: float | None,
               co_sidecar: bool = True) -> None:
    """Mot video nhieu frame. `frames[i]` = danh sach detection [x1,y1,x2,y2,conf,cls]."""
    n_per = [len(f) for f in frames]
    data = [d for f in frames for d in f]
    np.savez(cdir / f"{ten}.npz",
             frame_idx=np.arange(len(frames), dtype=np.int32),
             n_per_frame=np.array(n_per, dtype=np.int32),
             unreadable=np.zeros(len(frames), dtype=bool),
             truncated=np.zeros(len(frames), dtype=bool),
             time_s=np.zeros(len(frames), dtype=np.float64),
             data=np.array(data, dtype=np.float32).reshape(-1, 6),
             orig_shape=np.array([720, 1280], dtype=np.int32))
    if not co_sidecar:
        return
    info = {"video": ten, "n_frames": len(frames), "n_boxes": len(data),
            "spec_sha256": SPEC, "part": "val"}
    if giay is not None:
        info["seconds"] = giay
    (cdir / f"{ten}.json").write_text(json.dumps(info), encoding="utf-8")


def main() -> int:  # noqa: C901
    failed = 0
    tmp = Path(tempfile.mkdtemp(prefix="repstats_"))
    try:
        import config
        import det_cache
        import report_bdd_transfer as rep

        old_i, old_root = config.INTERIM_DIR, det_cache.CACHE_ROOT
        try:
            config.INTERIM_DIR = tmp / "interim"
            det_cache.CACHE_ROOT = config.INTERIM_DIR / "det_cache"
            cdir = det_cache.CACHE_ROOT / "fxcache"
            cdir.mkdir(parents=True)
            (cdir / "meta.json").write_text(json.dumps(
                {"conf": 0.05, "classes": [1, 2, 3], "max_det": 3000, "spec_sha256": SPEC}),
                encoding="utf-8")

            # video TRONG manifest: frame0 co 3 detection lop 2 (vuot max_det=1),
            #                       frame1 co 1 detection lop 3 va 1 detection lop 2 duoi nguong conf
            viet_cache(cdir, "trong_manifest",
                       [[[0, 0, 10, 10, 0.9, 2], [20, 20, 30, 30, 0.8, 2], [40, 40, 50, 50, 0.7, 2]],
                        [[0, 0, 10, 10, 0.9, 3], [20, 20, 30, 30, 0.10, 2]]],
                       giay=1.0)
            # video NGOAI manifest
            viet_cache(cdir, "ngoai_manifest",
                       [[[0, 0, 10, 10, 0.9, 2], [20, 20, 30, 30, 0.8, 2]]], giay=20.0)

            man = tmp / "man.json"
            man.write_text(json.dumps({"videos": ["trong_manifest"]}), encoding="utf-8")
            exp = {"name": "fx", "cache_id": "fxcache", "classes": [2, 3], "max_det": 1, "part": "val",
                   "manifest": str(man), "conditions": {"c25": {"conf_min": 0.25}}}

            # --- duong doc THAT cua runner ---
            vd = det_cache.open_cache("fxcache", "trong_manifest", 0.25, [2, 3], 1)
            that = [arr for arr in (vd.frame(i) for i in range(len(vd)))]
            that_tong = sum(len(a) for a in that)
            that_theo_lop: dict[int, int] = {}
            for a in that:
                for c in a[:, 5].astype(int):
                    that_theo_lop[int(c)] = that_theo_lop.get(int(c), 0) + 1

            dem = rep._det_counts(exp)["c25"]
            dem_tong = sum(dem.values())

            _check("frame vuot `max_det` -> dem DUNG bang so tracker nhan (cat [:max_det])",
                   dem_tong == that_tong == 2,
                   f"bao cao={dem_tong}  tracker that={that_tong}  (frame0 3->1, frame1 2->1)")
            _check("video NGOAI manifest khong duoc dem (truoc day cong ca vao)",
                   dem_tong == 2, f"neu gom ca video ngoai manifest se la 4; thuc te {dem_tong}")
            _check("so dem khop CHINH XAC voi duong doc cua runner, theo tung lop",
                   dict(dem) == that_theo_lop, f"bao cao={dict(dem)}  runner={that_theo_lop}")

            # --- thoi gian ---
            t = rep._detector_seconds(exp)
            _check("thoi gian detector chi cong sidecar cua sequence trong manifest",
                   t["du_lieu_du"] and abs(t["tong_giay"] - 1.0) < 1e-9 and t["n_frame"] == 2,
                   f"tong={t['tong_giay']} n_frame={t['n_frame']} (neu quet ca thu muc se la 21,0 / 3)")

            # --- sidecar thieu truong seconds ---
            viet_cache(cdir, "trong_manifest",
                       [[[0, 0, 10, 10, 0.9, 2]]], giay=None)
            t2 = rep._detector_seconds(exp)
            _check("sidecar THIEU `seconds` -> bao thieu, KHONG coi la 0 giay",
                   (not t2["du_lieu_du"]) and t2["thieu_seconds"] == ["trong_manifest"]
                   and t2["tong_giay"] == 0.0 and t2["n_video_co_so_do"] == 0,
                   f"du_lieu_du={t2['du_lieu_du']} thieu_seconds={t2['thieu_seconds']}")

            # --- sidecar thieu han ---
            (cdir / "trong_manifest.json").unlink()
            t3 = rep._detector_seconds(exp)
            _check("thieu HAN sidecar -> bao rieng o `thieu_sidecar`",
                   (not t3["du_lieu_du"]) and t3["thieu_sidecar"] == ["trong_manifest"],
                   f"thieu_sidecar={t3['thieu_sidecar']}")

            # --- bao cao TU CHOI dua ra tong khi thieu so do ---
            out = tmp / "toc_do.md"
            out.write_text("", encoding="utf-8")
            rep.toan_pipeline(exp, out)
            txt = out.read_text(encoding="utf-8")
            _check("thieu so do -> bao cao TU CHOI dua ra tong (khong in con so nua voi)",
                   "KHÔNG đưa ra tổng" in txt and "không** được coi là 0" in txt,
                   txt.strip()[:200])

            # --- khong co manifest -> loi ro rang ---
            loi = ""
            try:
                rep.videos_of({"name": "khong_manifest", "cache_id": "fxcache"})
            except ValueError as e:
                loi = str(e)
            _check("thi nghiem khong khai bao `manifest` -> LOI RO RANG (khong am tham quet thu muc)",
                   "manifest" in loi, loi[:140] or "(KHONG bao loi!)")
        finally:
            config.INTERIM_DIR, det_cache.CACHE_ROOT = old_i, old_root
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
