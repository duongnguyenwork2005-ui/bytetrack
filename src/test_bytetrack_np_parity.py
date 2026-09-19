"""
test_bytetrack_np_parity.py -- Backend NumPy (`BYTETRACK_BACKEND=np`, ban vendor
`third_party/ultralytics_trackers/`) phai cho ra KET QUA Y HET backend hien tai (ultralytics da cai),
va phai chay duoc trong moi truong KHONG co torch/Ultralytics.

SEQUENCE CHON TRUOC (ghi cung o day, chon TRUOC khi xem bat ky ket qua nao): 3 video dau cua manifest
dev `results/phase2/manifests/detrac_dev.json` theo dung thu tu trong file - khong chon theo ket qua.

TOLERANCE CONG BO TRUOC: **bit-identical**. Hai backend chay CUNG mot ban ma CTRV/CV (chi khac lop nen
`STrack`/`BYTETracker` lay tu dau), cung NumPy, cung thu tu phep toan => khong co ly do gi de lech.
Lech du mot bit la dau hieu ban vendor khong tuong duong => GHI LAI FRAME SAI DAU TIEN va tim nguyen
nhan, KHONG noi tolerance de dat PASS.

  1. Ban vendor GIONG HET ban ultralytics da cai, tru dung 5 dong import da ghi trong PROVENANCE.json,
     VA moi truong hash trong PROVENANCE.json dung theo quy uoc da cong bo (sha256 cua noi dung chuan
     hoa ve LF) - review vong 5 muc 5: truoc day test chi so `read_text()` nen bo qua khac biet xuong
     dong va khong he kiem cac truong hash.
  2. Backend np import duoc trong tien trinh CHAN san `torch` va `ultralytics` (chan o sys.meta_path).
  3. Parity CV  tren cac sequence da chon: moi hang (frame, id, bbox, conf, cls) bit-identical.
  4. Parity EKF tren cac sequence da chon: nhu tren.
  5. Song anh ID 1-1 xuyen suot sequence (khong anh xa lai tung frame).
  6. Tinh huong bien: frame khong detection, frame bi bo (n_steps > 1), track tai xuat hien,
     het track_buffer -> ca hai backend xu ly giong nhau (kiem tren chinh chuoi da chay).
  7. Hai tracker chay xen ke trong CUNG tien trinh: bo dem track_id khong ro ri giua cac lan chay.
  8. Hop dong Detections: xywh/xyxy/conf/cls dung dtype, dung cong thuc, khop `Boxes` cua ultralytics.

    python src/test_bytetrack_np_parity.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
CACHE_ID = "yolov8n-coco_conf0.05_all"
CONF_MIN, CLASSES = 0.25, [2, 3, 5, 7]
N_SEQ = 3
_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines()[-8:]:
            print(f"         {line}")
    assert ok, name


def chosen_videos() -> list[str]:
    """3 video dau cua manifest dev, theo dung thu tu trong file (chon truoc, khong theo ket qua)."""
    m = json.loads((config.RESULTS_DIR / "phase2" / "manifests" / "detrac_dev.json")
                   .read_text(encoding="utf-8"))
    return list(m["videos"])[:N_SEQ]


# --------------------------------------------------------------------- kich ban chay trong subprocess
RUNNER = r'''
import json, sys
BLOCK = %(block)r
if BLOCK:
    class _Block:
        def find_module(self, name, path=None):
            if name.split(".")[0] in ("torch", "ultralytics"):
                return self
        def load_module(self, name):
            raise ImportError("CHAN (test parity): " + name)
    sys.meta_path.insert(0, _Block())

sys.path.insert(0, %(src)r)
import numpy as np
import det_cache, run_tracking, ul_backend

videos = %(videos)r
out = {"backend": ul_backend.backend_name(), "blocked": BLOCK, "videos": {}}
for tracker_type, yaml in %(trackers)r:
    for v in videos:
        vd = det_cache.open_cache(%(cache)r, v, conf_min=%(conf)r, classes=%(classes)r, strict=False)
        cfg = run_tracking.load_tracker_cfg(yaml)   # tu chon duong nap theo backend
        df, df_cls, df_t = run_tracking.track_video(vd, cfg, dt_nominal=%(dt)r,
                                                    elapsed_mode="steps", video=v)
        # frame, id, x, y, w, h, conf, cls -- so TOAN BO dau ra, khong chi mot cot dai dien
        m = df.merge(df_cls, on=["frame", "id"], how="left")
        cols = list(df.columns) + ["cls"]
        a = m[cols].to_numpy(dtype=np.float64) if len(m) else np.zeros((0, len(cols)))
        a = np.ascontiguousarray(a)
        out["videos"].setdefault(tracker_type, {})[v] = {
            "n_rows": int(len(a)), "cols": cols,
            "sha": __import__("hashlib").sha256(a.tobytes()).hexdigest(),
            "rows": a.tolist()}
out["torch_loaded"] = "torch" in sys.modules
out["ultralytics_loaded"] = "ultralytics" in sys.modules
print("@@JSON@@" + json.dumps(out))
'''


def run_side(videos, trackers, block: bool, dt: float) -> dict:
    code = RUNNER % {"block": block, "src": str(ROOT / "src"), "videos": videos, "trackers": trackers,
                     "cache": CACHE_ID, "conf": CONF_MIN, "classes": CLASSES, "dt": dt}
    env = {**os.environ, "BYTETRACK_BACKEND": "np" if block else "ultralytics"}
    r = subprocess.run([PY, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    if "@@JSON@@" not in r.stdout:
        raise RuntimeError(f"subprocess (block={block}) that bai:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
    return json.loads(r.stdout.split("@@JSON@@", 1)[1].splitlines()[0])


def first_diff(a: list, b: list) -> str:
    A, B = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if A.shape != B.shape:
        return f"so hang khac: {A.shape} vs {B.shape}"
    bad = np.argwhere(A != B)
    if not len(bad):
        return "(khong lech)"
    i = int(bad[0][0])
    return (f"hang lech dau tien #{i} (frame {A[i][0]:.0f}): \n"
            f"           ultralytics = {A[i].tolist()}\n           np          = {B[i].tolist()}")


def main() -> int:
    failed = 0
    try:
        # --- 1. ban vendor giong ban da cai tru 5 dong import ---
        prov = json.loads((ROOT / "third_party/ultralytics_trackers/PROVENANCE.json")
                          .read_text(encoding="utf-8"))
        import importlib.util
        spec = importlib.util.find_spec("ultralytics")
        ul = Path(spec.origin).parent
        rw = {}
        for r in prov["import_rewrites"]:
            rw.setdefault(r["file"], []).append((r["from"], r["to"]))
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "src"))
        from ul_backend import HASH_CONVENTION, sha256_lf

        khac, hash_sai = [], []
        for rel, meta in prov["files"].items():
            up_p = ul / meta["upstream_path"].split("ultralytics/", 1)[1]
            ve_p = ROOT / "third_party/ultralytics_trackers" / rel
            # so NOI DUNG sau khi ap dung dung cac dong import da khai bao
            up = up_p.read_text(encoding="utf-8")
            for old, new in rw.get(rel, []):
                up = up.replace(old, new, 1)
            if up != ve_p.read_text(encoding="utf-8"):
                khac.append(rel)
            # va kiem CA HAI truong hash trong PROVENANCE theo dung quy uoc da cong bo
            if sha256_lf(up_p) != meta["sha256_upstream"]:
                hash_sai.append(f"{rel}:sha256_upstream")
            if sha256_lf(ve_p) != meta["sha256_vendor"]:
                hash_sai.append(f"{rel}:sha256_vendor")
        _check(f"ban vendor == ultralytics {prov['version']} sau khi ap dung {len(prov['import_rewrites'])} "
               f"dong import da khai bao", not khac, f"file khac: {khac}")
        _check(f"moi truong hash trong PROVENANCE dung theo quy uoc da cong bo "
               f"({prov.get('hash_convention', '(THIEU)')})",
               not hash_sai and prov.get("hash_convention") == HASH_CONVENTION,
               f"hash sai: {hash_sai}; quy uoc trong file = {prov.get('hash_convention')!r}")

        # hash phai ON DINH giua CRLF va LF (review vong 5: fingerprint tung phu thuoc nen tang)
        import hashlib
        import tempfile as _tf
        mau = ROOT / "third_party/ultralytics_trackers/basetrack.py"
        lf = mau.read_bytes().replace(b"\r\n", b"\n")
        with _tf.TemporaryDirectory() as d:
            a, b = Path(d) / "lf.py", Path(d) / "crlf.py"
            a.write_bytes(lf)
            b.write_bytes(lf.replace(b"\n", b"\r\n"))
            _check("sha256 theo quy uoc LF cho KET QUA GIONG NHAU voi ban LF va ban CRLF "
                   "(fingerprint khong phu thuoc nen tang)",
                   sha256_lf(a) == sha256_lf(b)
                   and hashlib.sha256(a.read_bytes()).hexdigest()
                   != hashlib.sha256(b.read_bytes()).hexdigest(),
                   f"LF {sha256_lf(a)[:12]} vs CRLF {sha256_lf(b)[:12]}")

        videos = chosen_videos()
        print(f"         sequence da chon (3 dau cua detrac_dev.json): {videos}")
        dt = 1.0 / 25.0
        trackers = [("bytetrack", "bytetrack.yaml"),
                    ("bytetrack_ekf_ctrv", "configs/bytetrack_ekf_ctrv.yaml")]

        # --- 2 + 3 + 4. chay hai ben ---
        ref = run_side(videos, trackers, block=False, dt=dt)
        new = run_side(videos, trackers, block=True, dt=dt)
        _check("backend np chay trong tien trinh CHAN torch va ultralytics (khong nap ca hai)",
               new["backend"] == "np" and not new["torch_loaded"] and not new["ultralytics_loaded"],
               f"backend={new['backend']} torch={new['torch_loaded']} ul={new['ultralytics_loaded']}")

        for tt, ten in (("bytetrack", "CV"), ("bytetrack_ekf_ctrv", "EKF-CTRV")):
            lech = []
            for v in videos:
                a, b = ref["videos"][tt][v], new["videos"][tt][v]
                if a["sha"] != b["sha"]:
                    lech.append((v, first_diff(a["rows"], b["rows"])))
            n_rows = sum(ref["videos"][tt][v]["n_rows"] for v in videos)
            _check(f"parity {ten}: {len(videos)} sequence / {n_rows} hang bit-identical "
                   f"(tolerance cong bo truoc = 0)",
                   not lech, "\n".join(f"{v}: {d}" for v, d in lech) if lech else
                   f"sha khop cho moi video: {[ref['videos'][tt][v]['sha'][:12] for v in videos]}")

        # --- 5. song anh ID 1-1 xuyen suot sequence ---
        import pandas as pd

        import track_compare
        xau, n_id = [], 0
        for tt in ("bytetrack", "bytetrack_ekf_ctrv"):
            for v in videos:
                cols = ref["videos"][tt][v]["cols"]
                A = pd.DataFrame(ref["videos"][tt][v]["rows"], columns=cols)
                B = pd.DataFrame(new["videos"][tt][v]["rows"], columns=cols)
                if not len(A):
                    continue
                res = track_compare.id_bijection(A, B)
                n_id += res.n_ids_a
                # song anh 1-1 TOAN SEQUENCE, va o day phai la ANH XA DONG NHAT (cung bo dem track_id)
                if not (res.is_bijection and res.is_identity):
                    xau.append((tt, v, f"bijection={res.is_bijection} identity={res.is_identity} "
                                       f"n_pairs={res.n_pairs} a_multi={list(res.a_multi)[:3]}"))
        _check("song anh ID 1-1 xuyen suot sequence, va la anh xa DONG NHAT "
               f"({n_id} track qua 6 lan chay)", not xau, xau[:3])

        # --- 6. tinh huong bien co that trong chuoi da chay ---
        bien = {}
        for v in videos:
            A = np.asarray(ref["videos"]["bytetrack_ekf_ctrv"][v]["rows"])
            if not len(A):
                continue
            fr = np.unique(A[:, 0]).astype(int)
            bien.setdefault("frame_khong_track", 0)
            bien["frame_khong_track"] += int(fr.max() - fr.min() + 1 - len(fr))   # frame khong co track nao
            ids, first = {}, {}
            for f, i in A[:, :2]:
                first.setdefault(int(i), int(f))
                ids.setdefault(int(i), []).append(int(f))
            bien["track_tai_xuat_hien"] = bien.get("track_tai_xuat_hien", 0) + sum(
                1 for i, fs in ids.items() if max(fs) - min(fs) + 1 > len(set(fs)))
        _check("chuoi kiem chua tinh huong bien (frame khong track, track tai xuat hien sau gap)",
               bien.get("frame_khong_track", 0) > 0 and bien.get("track_tai_xuat_hien", 0) > 0, bien)

        # --- 7. hai tracker xen ke trong cung tien trinh: bo dem track_id khong ro ri ---
        code = RUNNER % {"block": True, "src": str(ROOT / "src"),
                         "videos": videos[:1], "trackers": trackers + [("bytetrack", "bytetrack.yaml")],
                         "cache": CACHE_ID, "conf": CONF_MIN, "classes": CLASSES, "dt": dt}
        env = {**os.environ, "BYTETRACK_BACKEND": "np"}
        r = subprocess.run([PY, "-c", code], cwd=ROOT, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", env=env)
        xen = json.loads(r.stdout.split("@@JSON@@", 1)[1].splitlines()[0])
        v0 = videos[0]
        _check("chay xen ke CV -> EKF -> CV trong cung tien trinh: lan CV thu hai giong lan dau "
               "(bo dem track_id duoc reset)",
               xen["videos"]["bytetrack"][v0]["sha"] == new["videos"]["bytetrack"][v0]["sha"],
               f"{xen['videos']['bytetrack'][v0]['sha'][:16]} vs {new['videos']['bytetrack'][v0]['sha'][:16]}")

        # --- 8. hop dong Detections khop Boxes cua ultralytics ---
        import det_cache
        from ultralytics.engine.results import Boxes
        sys.path.insert(0, str(ROOT / "src"))
        import bytetrack_np
        rng = np.random.default_rng(0)
        arr = np.column_stack([rng.uniform(0, 500, 7), rng.uniform(0, 300, 7),
                               rng.uniform(501, 900, 7), rng.uniform(301, 530, 7),
                               rng.uniform(0.3, 0.99, 7), rng.integers(0, 8, 7)]).astype(np.float32)
        b_ul = Boxes(arr.copy(), (540, 960))
        b_np = bytetrack_np.Detections.from_cache_array(arr, (540, 960))
        same = (np.array_equal(np.asarray(b_ul.xywh), b_np.xywh)
                and np.array_equal(np.asarray(b_ul.conf), b_np.conf)
                and np.array_equal(np.asarray(b_ul.cls), b_np.cls)
                and np.array_equal(np.asarray(b_ul.xyxy), b_np.xyxy)
                and len(b_ul) == len(b_np) and b_np.xywh.dtype == np.float32)
        _check("hop dong Detections khop `Boxes` cua ultralytics (xywh/xyxy/conf/cls, dtype float32)",
               same, f"xywh dtype={b_np.xywh.dtype}, n={len(b_np)}")
    except AssertionError:
        failed += 1
    except Exception as e:  # noqa: BLE001
        failed += 1
        print(f"  [FAIL] loi khong mong doi: {type(e).__name__}: {e}")
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
