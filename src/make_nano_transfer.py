"""
make_nano_transfer.py -- Dung goi CHUYEN SANG JETSON NANO (doc lap, KHONG phai review delta) va kiem goi.

    python src/make_nano_transfer.py build   [--onnx-check <json>]    -> handoff/nano_transfer/ + handoff/nano_transfer_latest.zip
    python src/make_nano_transfer.py verify  [--zip handoff/nano_transfer_latest.zip] [--python <interp>]
        -> giai nen vao thu muc TAM SACH tren PC, chay THEO HUONG DAN TRONG GOI (duong dan tuong doi, cwd = goc goi):
           manifest -> nano_env_check -> import backend np (CHAN torch/ultralytics) -> test_nano_parity ->
           replay cache CV/EKF (CHAN) -> nano_parity voi pc-ref (+ bit-identical) -> doc ONNX -> doc JPG/mp4.
           --python: interpreter khac (vd venv stack cu) de kiem phu thuoc thap hon.
           CHI xac minh goi du file/du phu thuoc; KHONG chung minh tuong thich Nano.

Goi gom (giu cau truc thu muc de `config.PROJECT_ROOT` = goc goi):
    src/ (tru src/demo/**), configs/, third_party/ultralytics_trackers/ (+LICENSE, PROVENANCE), scripts/nano/,
    models/export/yolov8n_640_op12.onnx + .export.json (hash checkpoint/ONNX, lenh + tham so export, doi chieu
    trong so voi ban re-export), det cache cua MVI_39031 (meta.json + npz + sidecar), doan smoke frame 1-300
    (JPG y nguyen + mp4 lossy + JSON nguon/khoang frame/fps) + seqinfo/ignored_regions, ket qua tham chieu PC
    CV/EKF chay tu chinh cache do bang backend NumPy (+ doi chieu voi backend ultralytics), NANO_START_HERE.md,
    PACKAGE_INFO.json (commit/HEAD/working tree/phien ban moi truong PC), MANIFEST_SHA256.txt (tung file).
Co them: models/yolov8n.pt (checkpoint dang dung -- goi TRIEN KHAI duoc phep, khac chinh sach goi review),
requirements_nano.txt, src/nano_env_check.py, cache CAT dung doan smoke + tham chieu (NANO-smoke) va cache goc + tham
chieu toan sequence (NANO-full).
KHONG gom: env Conda/venv, .git, dataset day du, video dai, lich su ket qua, engine TensorRT, GT, ZIP cu, .gitignore,
src/demo/**, live.bat, outputs/**.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "handoff" / "nano_transfer"
ZIP = ROOT / "handoff" / "nano_transfer_latest.zip"
PY = sys.executable

VIDEO = "MVI_39031"
CACHE_ID = "yolov8n-coco_conf0.05_all"            # cache goc PC (toan bo 1470 frame)
SMOKE_CACHE_ID = "nano_smoke_MVI_39031_f0001-0300" # cache CAT dung doan smoke (frame 1-300), cung DetectorSpec
SPLIT = "NANO-smoke"                               # tham chieu tren doan smoke (cache cat)
SPLIT_FULL = "NANO-full"                           # tham chieu tren toan bo sequence (cache goc)
SMOKE_FRAMES = (1, 300)                       # img00001..img00300 (12,0 s @ 25 fps)
FPS = 25
ONNX_NAME = "yolov8n_640_op12.onnx"
CHECKPOINT = ROOT / "models" / "yolov8n.pt"
TRACKERS = {"cv": "bytetrack.yaml", "ekf": "configs/bytetrack_ekf_ctrv.yaml"}
TRACK_ARGS = ["--videos", VIDEO, "--conf-min", "0.25", "--classes", "1", "2", "3", "5", "7", "--max-det", "300",
              "--fps", str(FPS), "--track-buffer-sec", "1.2", "--elapsed-mode", "steps"]
PRIVATE = (".gitignore", "live.bat")
PRIVATE_PREFIX = ("src/demo/", "outputs/")

BLOCK = '''
import sys
class _Block:
    def find_module(self, name, path=None):
        if name.split(".")[0] in ("torch", "ultralytics"):
            return self
    def load_module(self, name):
        raise ImportError("CHAN (kiem goi nano): " + name)
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("torch", "ultralytics"):
            raise ImportError("CHAN (kiem goi nano): " + name)
sys.meta_path.insert(0, _Block())
'''


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          errors="replace").stdout.strip()


def _skip(rel: str) -> bool:
    rel = rel.replace("\\", "/")
    return (rel in PRIVATE or rel.startswith(PRIVATE_PREFIX) or "__pycache__" in rel
            or rel.endswith((".pyc", ".pyo")))


def copy_tree(src: Path, dst: Path, rel_root: Path) -> list[str]:
    out = []
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(rel_root).as_posix()
        if _skip(rel):
            continue
        d = dst / rel
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d)
        out.append(rel)
    return out


def env_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "scipy", "lap", "pandas", "yaml", "cv2", "onnx", "ultralytics", "torch"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            out[m] = f"KHONG CO ({type(e).__name__})"
    return out


def run(cmd: list[str], cwd: Path, env: dict | None = None, log: list[str] | None = None) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       env=env, shell=False)
    if log is not None:
        log.append(f"$ {subprocess.list2cmdline(cmd)}\n  (cwd {cwd})\n  exit={r.returncode}")
        tail = (r.stdout + r.stderr).strip().splitlines()[-6:]
        log.extend("  | " + ln for ln in tail)
    return r


def run_blocked(script_rel: str, args: list[str], cwd: Path, backend: str = "np", log: list[str] | None = None,
                py: str = PY):
    """Chay <script_rel> (tuong doi goc goi, cwd = goc goi) trong tien trinh CHAN torch/ultralytics."""
    code = (BLOCK + "import sys, runpy\nsys.path.insert(0, 'src')\n"
            f"sys.argv = [{script_rel!r}] + {args!r}\n"
            f"runpy.run_path({script_rel!r}, run_name='__main__')\n")
    env = {**os.environ, "BYTETRACK_BACKEND": backend, "PYTHONIOENCODING": "utf-8"}
    return run([py, "-c", code], cwd, env, log)


# ------------------------------------------------------------------------------------------ build
def build(onnx_check: Path | None) -> int:
    t0 = time.time()
    log: list[str] = []
    if PKG.exists():
        shutil.rmtree(PKG)
    PKG.mkdir(parents=True)

    # ---- 1. ma nguon, cau hinh, vendor, script (giu cau truc) ----
    files = []
    files += copy_tree(ROOT / "src", PKG, ROOT)
    files += copy_tree(ROOT / "configs", PKG, ROOT)
    files += copy_tree(ROOT / "third_party" / "ultralytics_trackers", PKG, ROOT)
    files += copy_tree(ROOT / "scripts" / "nano", PKG, ROOT)
    (PKG / "scripts" / "nano" / "NANO_START_HERE.md").unlink()          # ban chinh nam o goc goi
    files.remove("scripts/nano/NANO_START_HERE.md")
    assert not any(_skip(f) for f in files)
    assert not any(f.startswith("src/demo/") for f in files)
    # LICENSE vendor phai co
    assert (PKG / "third_party/ultralytics_trackers/LICENSE").is_file()

    # ---- 2. ONNX + thong tin checkpoint/export ----
    onnx_src = ROOT / "models" / "export" / ONNX_NAME
    onnx_ok = onnx_src.is_file()
    (PKG / "models" / "export").mkdir(parents=True)
    exp_info = {"checkpoint": "models/yolov8n.pt", "checkpoint_sha256": sha256_file(CHECKPOINT) if CHECKPOINT.is_file() else None,
                "checkpoint_bytes": CHECKPOINT.stat().st_size if CHECKPOINT.is_file() else None,
                "checkpoint_trong_goi": False,
                "checkpoint_ghi_chu": "Nano khong chay .pt (khong torch); hash de doi chieu voi cache meta.json/weights_sha256",
                "onnx": f"models/export/{ONNX_NAME}" if onnx_ok else None,
                "onnx_sha256": sha256_file(onnx_src) if onnx_ok else None,
                "onnx_bytes": onnx_src.stat().st_size if onnx_ok else None,
                "lenh_export": "python src/export_detector.py --weights models/yolov8n.pt --opset 12 --imgsz 640",
                "tham_so_export": {"format": "onnx", "opset": 12, "imgsz": 640, "dynamic": False, "simplify": False,
                                   "half": False, "nms": False, "batch": 1},
                "dau_vao": "images 1x3x640x640", "dau_ra": "output0 1x84x8400 (khong NMS trong graph)",
                "doi_chieu_trong_so": None,
                "trang_thai": ("SAN SANG de build engine tren Nano (ket luan cuoi cung phai qua trtexec tren thiet bi)"
                               if onnx_ok else "THIEU ONNX -- chua san sang")}
    src_json = ROOT / "results" / "phase2" / "export" / "yolov8n_op12.json"
    if src_json.is_file():
        exp_info["kiem_op_trt82"] = json.loads(src_json.read_text(encoding="utf-8"))
    if onnx_check and onnx_check.is_file():
        chk = json.loads(onnx_check.read_text(encoding="utf-8"))
        exp_info["doi_chieu_trong_so"] = chk
        if chk.get("onnx_sha256") != exp_info["onnx_sha256"] or chk.get("checkpoint_sha256") != exp_info["checkpoint_sha256"]:
            sys.exit("DUNG: --onnx-check khong khop hash ONNX/checkpoint hien tai")
        if not (chk.get("initializer_giong_het") and chk.get("graph_node_giong_het")):
            exp_info["trang_thai"] = "ONNX KHONG khop ban re-export tu checkpoint -> KHONG coi la san sang"
    if onnx_ok:
        shutil.copy2(onnx_src, PKG / "models" / "export" / ONNX_NAME)
        files.append(f"models/export/{ONNX_NAME}")
    (PKG / "models" / "export" / f"{Path(ONNX_NAME).stem}.export.json").write_text(
        json.dumps(exp_info, indent=2, ensure_ascii=False), encoding="utf-8")
    files.append(f"models/export/{Path(ONNX_NAME).stem}.export.json")

    # ---- 2b. checkpoint .pt (goi TRIEN KHAI duoc phep chua; Nano can torch + loader rieng moi dung duoc) ----
    if CHECKPOINT.is_file():
        (PKG / "models").mkdir(exist_ok=True)
        shutil.copy2(CHECKPOINT, PKG / "models" / CHECKPOINT.name)
        files.append(f"models/{CHECKPOINT.name}")
        exp_info["checkpoint_trong_goi"] = True
        exp_info["checkpoint_ghi_chu"] = ("co trong goi de doi chieu/khi Nano co torch; ultralytics 8.4 can Python >= 3.8, "
                                          "KHONG cai duoc tren Python 3.6 he thong -> duong detector chinh la ONNX -> TensorRT")
        (PKG / "models" / "export" / f"{Path(ONNX_NAME).stem}.export.json").write_text(
            json.dumps(exp_info, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- 3. det cache: (a) cache goc toan sequence, (b) cache CAT dung doan smoke (giu frame rong + time_s) ----
    cdir = ROOT / "data" / "interim" / "det_cache" / CACHE_ID
    for name in ("meta.json", f"{VIDEO}.npz", f"{VIDEO}.json"):
        p = cdir / name
        if not p.is_file():
            sys.exit(f"DUNG: thieu cache {p}")
        d = PKG / "data" / "interim" / "det_cache" / CACHE_ID / name
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, d)
        files.append(d.relative_to(PKG).as_posix())
    meta = json.loads((cdir / "meta.json").read_text(encoding="utf-8"))
    if exp_info["checkpoint_sha256"] and meta.get("weights_sha256") != exp_info["checkpoint_sha256"]:
        sys.exit("DUNG: cache meta weights_sha256 khac checkpoint hien tai")
    import numpy as np
    z = np.load(cdir / f"{VIDEO}.npz")
    a0, b0 = SMOKE_FRAMES
    sel = (z["frame_idx"] >= a0) & (z["frame_idx"] <= b0)
    idx = np.flatnonzero(sel)
    assert len(idx) == b0 - a0 + 1 and idx[0] == 0 and np.all(np.diff(idx) == 1), "doan smoke phai la tien to lien tuc cua cache"
    n_per = z["n_per_frame"][idx]
    off = int(np.concatenate([[0], np.cumsum(z["n_per_frame"])])[idx[-1] + 1])
    arrs = {"frame_idx": z["frame_idx"][idx], "n_per_frame": n_per, "unreadable": z["unreadable"][idx],
            "truncated": z["truncated"][idx], "time_s": z["time_s"][idx], "data": z["data"][:off], "orig_shape": z["orig_shape"]}
    assert int(n_per.sum()) == len(arrs["data"])
    sdir = PKG / "data" / "interim" / "det_cache" / SMOKE_CACHE_ID
    sdir.mkdir(parents=True)
    shutil.copy2(cdir / "meta.json", sdir / "meta.json")            # CUNG DetectorSpec (copy nguyen van)
    side_goc = json.loads((cdir / f"{VIDEO}.json").read_text(encoding="utf-8"))
    np.savez_compressed(sdir / f"{VIDEO}.npz", **arrs)
    side = {"video": VIDEO, "n_frames": int(len(idx)), "n_boxes": int(len(arrs["data"])),
            "n_unreadable": int(arrs["unreadable"].sum()), "n_truncated": int(arrs["truncated"].sum()),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cat_tu": {"cache_id": CACHE_ID, "sidecar_goc": side_goc, "khoang_frame": [a0, b0],
                       "n_frame_rong_giu_nguyen": int((n_per == 0).sum()),
                       "time_s_dau_cuoi": [float(arrs["time_s"][0]), float(arrs["time_s"][-1])],
                       "ghi_chu": "cat TIEN TO frame 1-300 tu cache goc; frame khong co detection van co mat (n_per_frame=0); "
                                  "meta.json copy nguyen van (cung DetectorSpec/weights_sha256)"}}
    for k in ("regions_hash", "n_regions", "part"):
        if k in side_goc:
            side[k] = side_goc[k]
    (sdir / f"{VIDEO}.json").write_text(json.dumps(side, indent=2, ensure_ascii=False), encoding="utf-8")
    for name in ("meta.json", f"{VIDEO}.npz", f"{VIDEO}.json"):
        files.append((sdir / name).relative_to(PKG).as_posix())

    # ---- 4. doan smoke: JPG y nguyen + mp4 + seqinfo/regions ----
    img_dir = ROOT / "data" / "extracted" / "DETRAC-Images" / VIDEO
    a, b = SMOKE_FRAMES
    sm = PKG / "data" / "smoke" / f"{VIDEO}_f{a:04d}-{b:04d}"
    sm.mkdir(parents=True)
    frame_files = []
    for i in range(a, b + 1):
        p = img_dir / f"img{i:05d}.jpg"
        if not p.is_file():
            sys.exit(f"DUNG: thieu frame {p}")
        shutil.copy2(p, sm / p.name)
        frame_files.append(p.name)
        files.append((sm / p.name).relative_to(PKG).as_posix())
    import cv2
    first = cv2.imread(str(sm / frame_files[0]))
    h, w = first.shape[:2]
    mp4 = PKG / "data" / "smoke" / f"{VIDEO}_f{a:04d}-{b:04d}.mp4"
    wr = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    assert wr.isOpened()
    for fn in frame_files:
        wr.write(cv2.imread(str(sm / fn)))
    wr.release()
    files.append(mp4.relative_to(PKG).as_posix())
    cap = cv2.VideoCapture(str(mp4))
    n_mp4 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps_mp4 = cap.get(cv2.CAP_PROP_FPS); cap.release()
    for name, src in (("seqinfo.ini", ROOT / "data/processed/DETRAC-test" / VIDEO / "seqinfo.ini"),
                      ("ignored_regions.txt", ROOT / "data/processed/DETRAC-test" / VIDEO / "gt" / "ignored_regions.txt")):
        d = PKG / "data" / "smoke" / VIDEO / name
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, d)
        files.append(d.relative_to(PKG).as_posix())
    smoke_info = {
        "video": VIDEO, "nguon": f"UA-DETRAC test, data/extracted/DETRAC-Images/{VIDEO}/ (1470 frame)",
        "khoang_frame": [a, b], "n_frame": b - a + 1, "fps": FPS, "thoi_luong_s": (b - a + 1) / FPS,
        "kich_thuoc": [w, h],
        "jpg": {"thu_muc": sm.relative_to(PKG).as_posix(), "ghi_chu": "ban Y NGUYEN byte cua dataset; thu tu theo SO trong ten file",
                "sha256_frame_dau": sha256_file(sm / frame_files[0]), "sha256_frame_cuoi": sha256_file(sm / frame_files[-1])},
        "mp4": {"file": mp4.relative_to(PKG).as_posix(), "codec": "mp4v (OpenCV VideoWriter, LOSSY)", "n_frame_doc_lai": n_mp4,
                "fps_doc_lai": fps_mp4, "sha256": sha256_file(mp4),
                "ghi_chu": "chi de tien --source; detection tu mp4 KHONG so duoc voi cache (nen lai + khong che vung bo qua)"},
        "cache_lien_quan": {"cache_doan_smoke": SMOKE_CACHE_ID, "cache_goc_toan_sequence": CACHE_ID,
                            "pham_vi": f"cache doan smoke = frame {a}-{b} (cat tien to tu cache goc 1470 frame, giu frame rong + time_s)",
                            "che_vung_bo_qua": bool(meta.get("mask_ignored_regions")),
                            "regions_file": f"data/smoke/{VIDEO}/ignored_regions.txt"},
    }
    (PKG / "data" / "smoke" / f"{VIDEO}_smoke.json").write_text(json.dumps(smoke_info, indent=2, ensure_ascii=False), encoding="utf-8")
    files.append(f"data/smoke/{VIDEO}_smoke.json")

    # ---- 5. ket qua tham chieu PC: backend np (trong goi) + doi chieu voi backend ultralytics ----
    (PKG / "reference").mkdir()
    lenh = ["#!/usr/bin/env bash", "# Lenh da tao ban tham chieu PC (chay tai goc goi). Tren Nano: doi --tracker-name thanh nano-cv / nano-ekf.",
            "set -e", "cd \"$(dirname \"$0\")/..\""]
    parity_pc = {}
    for split, cid in ((SPLIT, SMOKE_CACHE_ID), (SPLIT_FULL, CACHE_ID)):
        lenh.append(f"# --- {split}: cache {cid} ---")
        for ten, yaml in TRACKERS.items():
            args = ["--cache-id", cid, "--split-name", split, "--tracker-yaml", yaml, "--tracker-name", f"pc-ref-{ten}", *TRACK_ARGS]
            r = run_blocked("src/run_tracking.py", args, PKG, "np", log)
            if r.returncode != 0:
                sys.exit(f"DUNG: run_tracking np ({split}/{ten}) exit {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
            lenh.append("BYTETRACK_BACKEND=np python3 src/run_tracking.py " + subprocess.list2cmdline(args))
            # doi chieu: backend ultralytics (co torch) cung dau vao -> phai y het
            args_ul = ["--cache-id", cid, "--split-name", split, "--tracker-yaml", yaml, "--tracker-name", f"pc-ul-{ten}", *TRACK_ARGS]
            env = {**os.environ, "BYTETRACK_BACKEND": "ultralytics", "PYTHONIOENCODING": "utf-8"}
            r2 = run([PY, str(PKG / "src" / "run_tracking.py"), *args_ul], PKG, env, log)
            if r2.returncode != 0:
                sys.exit(f"DUNG: run_tracking ultralytics ({split}/{ten}) exit {r2.returncode}\n{r2.stderr[-2000:]}")
            pj = PKG / "reference" / f"parity_pc_np_vs_ul_{split}_{ten}.json"
            r3 = run([PY, str(PKG / "src" / "nano_parity.py"), "--ref", str(PKG / "data/processed/trackers" / split / f"pc-ul-{ten}"),
                      "--test", str(PKG / "data/processed/trackers" / split / f"pc-ref-{ten}"), "--json", str(pj)], PKG, None, log)
            bit = (sha256_file(PKG / "data/processed/trackers" / split / f"pc-ul-{ten}" / "data" / f"{VIDEO}.txt")
                   == sha256_file(PKG / "data/processed/trackers" / split / f"pc-ref-{ten}" / "data" / f"{VIDEO}.txt"))
            parity_pc[f"{split}/{ten}"] = {"exit": r3.returncode, "bit_identical_data_txt": bit, "json": pj.relative_to(PKG).as_posix()}
            if r3.returncode != 0:
                sys.exit(f"DUNG: backend np KHAC ultralytics ({split}/{ten}) -> khong dong goi tham chieu")
            shutil.rmtree(PKG / "data/processed/trackers" / split / f"pc-ul-{ten}")
    lenh += ["", "# Parity (tren Nano, sau khi chay cac lenh tren voi --tracker-name nano-cv / nano-ekf):",
             f"python3 src/nano_parity.py --ref data/processed/trackers/{SPLIT}/pc-ref-cv  --test data/processed/trackers/{SPLIT}/nano-cv  --json reference/parity_nano_cv.json",
             f"python3 src/nano_parity.py --ref data/processed/trackers/{SPLIT}/pc-ref-ekf --test data/processed/trackers/{SPLIT}/nano-ekf --json reference/parity_nano_ekf.json",
             f"python3 src/nano_parity.py --ref data/processed/trackers/{SPLIT_FULL}/pc-ref-cv  --test data/processed/trackers/{SPLIT_FULL}/nano-cv  --json reference/parity_nano_full_cv.json",
             f"python3 src/nano_parity.py --ref data/processed/trackers/{SPLIT_FULL}/pc-ref-ekf --test data/processed/trackers/{SPLIT_FULL}/nano-ekf --json reference/parity_nano_full_ekf.json"]
    # requirements toi thieu cho duong tracker-tu-cache (tu chuoi import THAT, khong phai requirements PC)
    (PKG / "requirements_nano.txt").write_text(
        "# Phu thuoc TOI THIEU cho duong tracker-tu-cache (chuoi import that, torch/ultralytics BI CHAN khi kiem).\n"
        "# Python >= 3.8 (vermin tinh). KHONG phai requirements cua PC; phien ban PC chi de tham khao trong PACKAGE_INFO.json.\n"
        "# Da kiem tren PC voi stack cu hon: numpy 1.24.4, pandas 1.5.3, pyyaml 6.0.1, lap 0.5.12 (khong scipy) -> parity DAT, bit-identical.\n"
        "numpy\nlap>=0.4\npandas\npyyaml\n# tuy chon:\n# scipy        # chi khi use_lap=False (khong dung mac dinh)\n"
        "# opencv-python  # buoc chay video/anh (khong can cho replay cache)\n", encoding="utf-8", newline="\n")
    files.append("requirements_nano.txt")
    lenh.insert(1, "# CHI DE TAI TAO THAM CHIEU TREN PC. TREN NANO: dung `python src/nano_run_cache.py` (chi ghi nano-*, khong dung pc-ref-*).")
    (PKG / "reference" / "lenh_tai_hien.sh").write_text("\n".join(lenh) + "\n", encoding="utf-8", newline="\n")
    # hash cay tham chieu PC (bat bien) -> nano_run_cache doi chieu truoc/sau khi chay tren thiet bi
    pc_dirs = [PKG / "data/processed/trackers" / split / f"pc-ref-{t}" for split in (SPLIT, SPLIT_FULL) for t in TRACKERS]
    lines = [f"{sha256_file(q)}  {q.relative_to(PKG).as_posix()}" for d in pc_dirs for q in sorted(d.rglob('*')) if q.is_file()]
    (PKG / "reference" / "pc_ref_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    for p in sorted((PKG / "data/processed").rglob("*")):
        if p.is_file():
            files.append(p.relative_to(PKG).as_posix())
    for p in sorted((PKG / "reference").rglob("*")):
        if p.is_file():
            files.append(p.relative_to(PKG).as_posix())
    # __pycache__ sinh ra khi chay -> xoa
    for p in list(PKG.rglob("__pycache__")):
        shutil.rmtree(p, ignore_errors=True)

    # ---- 6. tai lieu + PACKAGE_INFO + MANIFEST ----
    shutil.copy2(ROOT / "scripts" / "nano" / "NANO_START_HERE.md", PKG / "NANO_START_HERE.md")
    files.append("NANO_START_HERE.md")
    st = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True,
                        encoding="utf-8", errors="replace").stdout          # KHONG strip: cot trang thai o 2 ky tu dau
    dirty = [ln.rstrip() for ln in st.splitlines()
             if len(ln) > 3 and ln[:2].strip() in ("M", "A", "D", "R") and not _skip(ln[3:].strip())]
    info = {
        "goi": "nano_transfer (doc lap, KHONG phai review delta)", "tao_luc": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nguon": {"repo": "D:/LABOTARY/bytetrack", "HEAD": git("rev-parse", "HEAD"), "HEAD_short": git("rev-parse", "--short", "HEAD"),
                  "branch": git("branch", "--show-current"), "subject": git("log", "-1", "--format=%s"),
                  "commit_dong_goi": git("rev-parse", "--short", "HEAD"),
                  "commit_code_cuoi": git("log", "-1", "--format=%h %s", "--", "src", "configs", "third_party"),
                  "commit_code_cuoi_khong_tinh_builder": git("log", "-1", "--format=%h %s", "--", "src", "configs", "third_party",
                                                             ":(exclude)src/make_nano_transfer.py", ":(exclude)src/nano_env_check.py"),
                  "tracked_da_sua_ngoai_file_rieng": dirty,
                  "ghi_chu_launcher": "L-F1..L-F4 (review 2d1a8df) da sua o 25c11b6/2e96bc7, CHUA duoc Codex nghiem thu"},
        "moi_truong_pc": env_versions(),
        "python_toi_thieu": "3.8 (vermin 1.8.0 tinh tren chuoi import that: from __future__ import annotations x21, dataclasses, importlib.metadata)",
        "phu_thuoc_python_toi_thieu_cho_replay": ["numpy", "lap", "pandas", "pyyaml"],
        "phu_thuoc_tuy_chon": {"scipy": "chi khi use_lap=False (khong phai mac dinh)", "opencv-python": "buoc chay video/anh"},
        "khong_can_cho_replay_parity": ["torch", "ultralytics", "scipy"],
        "requirements": "requirements_nano.txt",
        "checkpoint": {k: exp_info[k] for k in ("checkpoint", "checkpoint_sha256", "checkpoint_bytes", "checkpoint_trong_goi")},
        "onnx": {k: exp_info[k] for k in ("onnx", "onnx_sha256", "onnx_bytes", "trang_thai")},
        "det_cache": {"cache_doan_smoke": {"cache_id": SMOKE_CACHE_ID, "video": VIDEO, "khoang_frame": [a0, b0],
                                           "npz_sha256": sha256_file(sdir / f"{VIDEO}.npz"), "sidecar": side},
                      "cache_goc_toan_sequence": {"cache_id": CACHE_ID, "video": VIDEO, "meta": meta,
                                                  "npz_sha256": sha256_file(cdir / f"{VIDEO}.npz"), "sidecar": side_goc}},
        "tham_chieu_pc": {"splits": {SPLIT: SMOKE_CACHE_ID, SPLIT_FULL: CACHE_ID}, "ten": [f"pc-ref-{t}" for t in TRACKERS],
                          "backend": "np (torch/ultralytics BI CHAN khi tao)",
                          "cau_hinh_hieu_luc": "conf-min 0.25, classes 1 2 3 5 7, max-det 300, fps 25, track-buffer 1.2 s (30 buoc), elapsed steps; "
                                               "chi tiet trong run_meta.json (tracker_cfg, overrides, input_fingerprint)",
                          "doi_chieu_np_vs_ultralytics": parity_pc},
        "smoke": smoke_info,
        "trang_thai_nano": {"backend_np": "co, chay that tren PC", "engine_trt": "chua (build tren Nano)",
                            "detector_trt_noi_vong_lap": "CHUA CO MA (xem NANO_START_HERE.md muc 5)",
                            "chay_video_tren_nano": "chua", "so_do_toc_do_nano": "chua co"},
        "kiem_goi_sach": "ket qua kiem goi trong thu muc sach tren PC nam NGOAI goi (de hash zip khong doi sau khi kiem): "
                         "repo PC handoff/logs/nano_transfer/kiem_goi_sach.{log,json}; CHI xac minh du file/phu thuoc, KHONG chung minh tuong thich Nano",
    }
    (PKG / "PACKAGE_INFO.json").write_text(json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    manifest_lines = []
    all_files = sorted(p for p in PKG.rglob("*") if p.is_file() and p.name != "MANIFEST_SHA256.txt")
    for p in all_files:
        manifest_lines.append(f"{sha256_file(p)}  {p.relative_to(PKG).as_posix()}")
    (PKG / "MANIFEST_SHA256.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8", newline="\n")
    # file cam
    # goi TRIEN KHAI duoc phep chua checkpoint .pt (khac chinh sach goi review); van cam engine PC, GT, file rieng
    xau = [p.relative_to(PKG).as_posix() for p in all_files if _skip(p.relative_to(PKG).as_posix())
           or p.suffix in (".engine", ".plan", ".trt") or p.name == "gt.txt"
           or (p.suffix == ".pt" and p.name != CHECKPOINT.name)]
    assert not xau, xau

    # ---- 7. zip ----
    if ZIP.exists():
        ZIP.unlink()
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(PKG.rglob("*")):
            if p.is_file():
                z.write(p, f"nano_transfer/{p.relative_to(PKG).as_posix()}")
    with zipfile.ZipFile(ZIP) as z:
        assert z.testzip() is None
        n = len(z.namelist())
    tong = sum(p.stat().st_size for p in PKG.rglob("*") if p.is_file())
    print(f"goi: {PKG} ({tong/1e6:.1f} MB, {len(all_files) + 1} file)")
    print(f"ZIP: {ZIP} ({ZIP.stat().st_size:,} B, {n} thanh vien) sha256 {sha256_file(ZIP)}  [{time.time()-t0:.0f}s]")
    (ROOT / "handoff" / "logs" / "nano_transfer").mkdir(parents=True, exist_ok=True)
    (ROOT / "handoff" / "logs" / "nano_transfer" / "build.log").write_text(
        f"# build nano_transfer {time.strftime('%Y-%m-%dT%H:%M:%S')} HEAD {git('rev-parse', '--short', 'HEAD')}\n"
        + "\n".join(log) + f"\n\nZIP {ZIP.stat().st_size} B sha256 {sha256_file(ZIP)}\n", encoding="utf-8")
    return 0


# ------------------------------------------------------------------------------------------ verify
def verify(zip_path: Path, py: str = PY, suffix: str = "") -> int:
    """Giai nen vao thu muc TAM SACH (khong phai repo), chay chuoi kiem. Ket qua ghi NGOAI goi
    (handoff/logs/nano_transfer/kiem_goi_sach.{log,json}) de hash cua zip da kiem = hash cua zip ban giao."""
    tmp = Path(tempfile.mkdtemp(prefix="nano_pkg_"))
    log: list[str] = [f"# kiem goi sach {time.strftime('%Y-%m-%dT%H:%M:%S')} zip={zip_path} sha256={sha256_file(zip_path)}",
                      f"# giai nen vao {tmp} (thu muc tam, KHONG phai repo); cwd = goc goi, duong dan TUONG DOI nhu NANO_START_HERE.md",
                      f"# interpreter: {py}; moi buoc replay/parity CHAN torch va ultralytics",
                      "# Y NGHIA: chi xac minh goi DU FILE + DU PHU THUOC Python tren PC; KHONG chung minh tuong thich Nano"]
    kq = {"zip_sha256": sha256_file(zip_path), "zip_bytes": zip_path.stat().st_size, "interpreter": py, "buoc": {}}
    ver = subprocess.run([py, "-c", "import sys, json\nout={'python': sys.version.split()[0]}\n"
                          "for m in ('numpy','scipy','lap','pandas','yaml','cv2'):\n"
                          "    try: out[m]=__import__(m).__version__\n    except Exception as e: out[m]='KHONG CO'\n"
                          "print(json.dumps(out))"], capture_output=True, text=True).stdout.strip()
    kq["phien_ban_dung_de_kiem"] = json.loads(ver) if ver.startswith("{") else ver
    log.append(f"# phien ban: {ver}")
    ok_all = True
    try:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        g = tmp / "nano_transfer"
        # 0. manifest
        bad = []
        for ln in (g / "MANIFEST_SHA256.txt").read_text(encoding="utf-8").splitlines():
            h, rel = ln.split("  ", 1)
            p = g / rel
            if not p.is_file() or sha256_file(p) != h:
                bad.append(rel)
        n_manifest = len((g / "MANIFEST_SHA256.txt").read_text(encoding="utf-8").splitlines())
        kq["buoc"]["0_manifest"] = {"n_file": n_manifest, "lech_hoac_thieu": bad}
        ok_all &= not bad
        log.append(f"0 manifest: {n_manifest} file, lech/thieu = {bad}")
        # 1. nano_env_check (theo huong dan trong goi) + import backend np, CHAN torch/ultralytics
        r = run([py, "src/nano_env_check.py", "--json", "reference/env_check_pc.json"], g, {**os.environ, "PYTHONIOENCODING": "utf-8"}, log)
        kq["buoc"]["1a_nano_env_check"] = {"exit": r.returncode, "ket_luan": next((ln for ln in r.stdout.splitlines() if ln.startswith("KET LUAN")), "")}
        ok_all &= r.returncode == 0
        code = (BLOCK + "import sys, json\nsys.path.insert(0, 'src')\nimport ul_backend\n"
                "fp = ul_backend.fingerprint()\nprint(json.dumps({'backend': ul_backend.backend_name(), 'torch': 'torch' in sys.modules, "
                "'ul': 'ultralytics' in sys.modules, 'fp_keys': sorted(fp)[:6]}))\n")
        r = run([py, "-c", code], g, {**os.environ, "BYTETRACK_BACKEND": "np", "PYTHONIOENCODING": "utf-8"}, log)
        kq["buoc"]["1b_import_np"] = {"exit": r.returncode, "out": r.stdout.strip()[-300:], "err": r.stderr.strip()[-300:]}
        ok_all &= r.returncode == 0 and '"torch": false' in r.stdout
        # 2. test_nano_parity
        r = run_blocked("src/test_nano_parity.py", [], g, "np", log, py=py)
        kq["buoc"]["2_test_nano_parity"] = {"exit": r.returncode, "ket_qua": next((ln for ln in r.stdout.splitlines() if "KET QUA" in ln), "")}
        ok_all &= r.returncode == 0
        # 3-4. CHINH script chay tren thiet bi: nano_run_cache.py (kiem dau vao, hash pc-ref truoc/sau, chi ghi kiem-*,
        #      replay CV/EKF smoke -> full, parity + bit-identical; tu CHAN torch/ultralytics)
        r = run([py, "src/nano_run_cache.py", "--tag", "kiem"], g, {**os.environ, "PYTHONIOENCODING": "utf-8"}, log)
        runs = sorted((g / "reference").glob("nano_run_*.json"))
        rj = json.loads(runs[-1].read_text(encoding="utf-8")) if runs else {}
        kq["buoc"]["3_4_nano_run_cache"] = {"exit": r.returncode, "ket_luan": rj.get("ket_luan"),
                                            "pc_ref_khop_ban_dong_goi": rj.get("buoc", {}).get("1_pc_ref_khop_ban_dong_goi"),
                                            "pc_ref_khong_doi_sau_chay": rj.get("buoc", {}).get("4_pc_ref_khong_doi_sau_chay"),
                                            "chay": {k: {kk: v.get(kk) for kk in ("replay_exit", "parity_exit", "dat", "n_ref", "n_test", "bit_identical_data_txt", "wall_s")}
                                                     for k, v in rj.get("buoc", {}).items() if k.startswith("2_")}}
        ok_all &= r.returncode == 0 and rj.get("ket_luan") == "DAT"
        # 4b. NEGATIVE (review 20207ec): moi truong hop -> nano_run_cache exit 2, KHONG tao <tag>-*, khong chay tracker;
        #     nano_env_check nonzero khi asset khong dung duoc. Cay pc-ref khong doi sau tat ca.
        def _out(tag):
            return sorted(str(q.relative_to(g)) for q in (g / "data/processed/trackers").glob(f"*/{tag}-*"))
        def _khong_buoc2(tag):
            fs = sorted((g / "reference").glob("nano_run_*.json"))
            jj = json.loads(fs[-1].read_text(encoding="utf-8")) if fs else {}
            return not any(k.startswith("2_") for k in jj.get("buoc", {}))
        env_pkg = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        neg = {}
        # (i) thieu 1 tham chieu
        pref = g / "data/processed/trackers" / SPLIT_FULL / "pc-ref-ekf" / "data" / f"{VIDEO}.txt"
        pref.rename(pref.with_suffix(".txt.tam"))
        try:
            r = run([py, "src/nano_run_cache.py", "--tag", "neg1"], g, env_pkg, log)
        finally:
            pref.with_suffix(".txt.tam").rename(pref)
        neg["i_thieu_pc_ref"] = {"exit": r.returncode, "output": _out("neg1"), "khong_buoc_2": _khong_buoc2("neg1")}
        # (ii) sidecar hong (N15-P1)
        side = g / "data/interim/det_cache" / SMOKE_CACHE_ID / f"{VIDEO}.json"; goc_side = side.read_bytes()
        side.write_text("{ bad json", encoding="utf-8")
        try:
            r = run([py, "src/nano_run_cache.py", "--tag", "neg2"], g, env_pkg, log)
        finally:
            side.write_bytes(goc_side)
        neg["ii_sidecar_hong"] = {"exit": r.returncode, "output": _out("neg2"), "khong_buoc_2": _khong_buoc2("neg2")}
        # (iii) thieu lap (N15-P2): goi gia `lap` nem ImportError dat truoc PYTHONPATH
        fake = tmp / "fake_lap"; fake.mkdir(exist_ok=True)
        (fake / "lap.py").write_text("raise ImportError('lap bi chan (kiem goi)')\n", encoding="utf-8")
        env_nolap = {**env_pkg, "PYTHONPATH": str(fake) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        r0 = run([py, "src/nano_env_check.py"], g, env_nolap, log)
        r = run([py, "src/nano_run_cache.py", "--tag", "neg3"], g, env_nolap, log)
        neg["iii_thieu_lap"] = {"env_check_exit": r0.returncode, "exit": r.returncode, "output": _out("neg3"), "khong_buoc_2": _khong_buoc2("neg3")}
        # (iv) nano_env_check doc lap: xoa NPZ + file tham chieu (N15-P3)
        mv = []
        for q in list((g / "data/interim/det_cache").glob(f"*/{VIDEO}.npz")) + list((g / "data/processed/trackers").glob(f"*/pc-ref-*/data/{VIDEO}.txt")):
            q2 = q.with_suffix(q.suffix + ".tam"); q.rename(q2); mv.append((q, q2))
        try:
            r = run([py, "src/nano_env_check.py"], g, env_pkg, log)
        finally:
            for q, q2 in mv:
                q2.rename(q)
        neg["iv_env_check_thieu_npz_va_pc_ref"] = {"exit": r.returncode, "in_DAT_1_4": "DAT (1-4)" in r.stdout}
        kq["buoc"]["4b_negative"] = neg
        ok_all &= (neg["i_thieu_pc_ref"]["exit"] == 2 and neg["i_thieu_pc_ref"]["output"] == [] and neg["i_thieu_pc_ref"]["khong_buoc_2"]
                   and neg["ii_sidecar_hong"]["exit"] == 2 and neg["ii_sidecar_hong"]["output"] == [] and neg["ii_sidecar_hong"]["khong_buoc_2"]
                   and neg["iii_thieu_lap"]["env_check_exit"] != 0 and neg["iii_thieu_lap"]["exit"] == 2 and neg["iii_thieu_lap"]["output"] == []
                   and neg["iii_thieu_lap"]["khong_buoc_2"]
                   and neg["iv_env_check_thieu_npz_va_pc_ref"]["exit"] != 0 and not neg["iv_env_check_thieu_npz_va_pc_ref"]["in_DAT_1_4"])
        # cay pc-ref sau tat ca negative == ban dong goi
        chuan = set(ln.strip() for ln in (g / "reference/pc_ref_sha256.txt").read_text(encoding="utf-8").splitlines() if ln.strip())
        hien = set(f"{sha256_file(q)}  {q.relative_to(g).as_posix()}" for split in (SPLIT, SPLIT_FULL) for t in TRACKERS
                   for q in sorted((g / "data/processed/trackers" / split / f"pc-ref-{t}").rglob("*")) if q.is_file())
        kq["buoc"]["4d_pc_ref_khong_doi_sau_negative"] = (hien == chuan)
        ok_all &= hien == chuan
        log.append(f"4b negative: {json.dumps(neg, ensure_ascii=False)} | pc-ref khong doi: {hien == chuan}")
        # 4c. nano_det_compare: cache cat vs cache goc (tien to) -> phai ghep 100%, IoU 1, dconf 0
        pj = g / "reference" / "det_compare_kiem.json"
        r = run([py, "src/nano_det_compare.py", "--a", SMOKE_CACHE_ID, "--b", CACHE_ID, "--video", VIDEO, "--max-frames", str(SMOKE_FRAMES[1]),
                 "--json", str(pj)], g, {**os.environ, "PYTHONIOENCODING": "utf-8"}, log)
        dc = json.loads(pj.read_text(encoding="utf-8")) if pj.is_file() else {}
        kq["buoc"]["4c_det_compare_tu_kiem"] = {"exit": r.returncode, "n_cap": dc.get("n_cap"), "ti_le_ghep_a": dc.get("ti_le_ghep_a"),
                                                "iou_nho_nhat": dc.get("iou_nho_nhat"), "d_conf_lon_nhat": dc.get("d_conf_lon_nhat")}
        ok_all &= r.returncode == 0 and dc.get("ti_le_ghep_a") == 1.0 and dc.get("ti_le_ghep_b") == 1.0 and (dc.get("d_conf_lon_nhat") or 0) == 0
        # 5. ONNX doc duoc (onnx co tren PC; tren Nano se la trtexec)
        code = (f"import onnx, hashlib, json\np = r{str(g / 'models/export' / ONNX_NAME)!r}\nm = onnx.load(p)\nonnx.checker.check_model(m)\n"
                "print(json.dumps({'ir': m.ir_version, 'opset': [(o.domain, o.version) for o in m.opset_import], 'n_node': len(m.graph.node), "
                "'inputs': [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input], "
                "'sha256': hashlib.sha256(open(p, 'rb').read()).hexdigest()}))\n")
        r = run([PY, "-c", code], g, None, log)            # onnx: goi cua env PC (tren Nano la trtexec)
        kq["buoc"]["5_onnx_doc_duoc"] = {"exit": r.returncode, "out": r.stdout.strip()[-400:]}
        ok_all &= r.returncode == 0
        # 6. doan smoke doc duoc bang cv2 (JPG + mp4)
        code = (f"import cv2, json, glob\nd = r{str(g / 'data/smoke')!r}\nfs = sorted(glob.glob(d + '/{VIDEO}_f0001-0300/*.jpg'))\n"
                f"im = cv2.imread(fs[0]); cap = cv2.VideoCapture(d + '/{VIDEO}_f0001-0300.mp4'); n = 0\n"
                "while True:\n    ok, _ = cap.read()\n    if not ok: break\n    n += 1\n"
                "print(json.dumps({'n_jpg': len(fs), 'jpg_shape': list(im.shape), 'mp4_frames_doc': n}))\n")
        r = run([PY, "-c", code], g, None, log)            # cv2: goi cua env PC
        kq["buoc"]["6_smoke_doc_duoc"] = {"exit": r.returncode, "out": r.stdout.strip()[-200:]}
        ok_all &= r.returncode == 0 and '"n_jpg": 300' in r.stdout and '"mp4_frames_doc": 300' in r.stdout
        r = run([PY, "src/render_mot_video.py", "--images", f"data/smoke/{VIDEO}_f0001-0300", "--mot", f"data/processed/trackers/{SPLIT}/pc-ref-cv/data/{VIDEO}.txt",
                 "--cls", f"data/processed/trackers/{SPLIT}/pc-ref-cv/data_cls/{VIDEO}.csv", "--fps", "25", "--out", "reference/render_kiem.mp4"], g, None, log)
        kq["buoc"]["6b_render_video"] = {"exit": r.returncode, "out": r.stdout.strip()[-120:], "mp4_bytes": ((g / "reference/render_kiem.mp4").stat().st_size if (g / "reference/render_kiem.mp4").is_file() else 0)}
        ok_all &= r.returncode == 0 and kq["buoc"]["6b_render_video"]["mp4_bytes"] > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    kq["ket_luan"] = "DAT: goi du file/du phu thuoc de replay + parity tren PC (KHONG chung minh tuong thich Nano)" if ok_all else "KHONG DAT"
    log.append(f"KET LUAN: {kq['ket_luan']}")
    out = ROOT / "handoff" / "logs" / "nano_transfer" / f"kiem_goi_sach{suffix}.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(log) + "\n", encoding="utf-8")
    (ROOT / "handoff" / "logs" / "nano_transfer" / f"kiem_goi_sach{suffix}.json").write_text(json.dumps(kq, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(log[-12:]))
    print(f"-> {out}")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "verify"])
    ap.add_argument("--onnx-check", default=None, help="JSON doi chieu trong so ONNX vs re-export (build)")
    ap.add_argument("--zip", default=str(ZIP))
    ap.add_argument("--python", default=PY, help="interpreter dung de kiem (vd venv stack cu)")
    ap.add_argument("--log-suffix", default="", help="hau to ten file log kiem (vd _oldstack)")
    a = ap.parse_args()
    if a.mode == "build":
        return build(Path(a.onnx_check) if a.onnx_check else None)
    return verify(Path(a.zip), a.python, a.log_suffix)


if __name__ == "__main__":
    raise SystemExit(main())
