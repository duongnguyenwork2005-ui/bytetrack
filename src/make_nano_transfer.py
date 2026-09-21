"""
make_nano_transfer.py -- Dung goi CHUYEN SANG JETSON NANO (doc lap, KHONG phai review delta) va kiem goi.

    python src/make_nano_transfer.py build   [--onnx-check <json>]    -> handoff/nano_transfer/ + handoff/nano_transfer.zip
    python src/make_nano_transfer.py verify  [--zip handoff/nano_transfer.zip]
        -> giai nen vao thu muc TAM SACH tren PC, chay: import backend np (CHAN torch/ultralytics) ->
           test_nano_parity -> replay cache CV/EKF (CHAN torch/ultralytics) -> nano_parity voi pc-ref -> hash manifest.
           CHI xac minh goi du file/du phu thuoc; KHONG chung minh tuong thich Nano.

Goi gom (giu cau truc thu muc de `config.PROJECT_ROOT` = goc goi):
    src/ (tru src/demo/**), configs/, third_party/ultralytics_trackers/ (+LICENSE, PROVENANCE), scripts/nano/,
    models/export/yolov8n_640_op12.onnx + .export.json (hash checkpoint/ONNX, lenh + tham so export, doi chieu
    trong so voi ban re-export), det cache cua MVI_39031 (meta.json + npz + sidecar), doan smoke frame 1-300
    (JPG y nguyen + mp4 lossy + JSON nguon/khoang frame/fps) + seqinfo/ignored_regions, ket qua tham chieu PC
    CV/EKF chay tu chinh cache do bang backend NumPy (+ doi chieu voi backend ultralytics), NANO_START_HERE.md,
    PACKAGE_INFO.json (commit/HEAD/working tree/phien ban moi truong PC), MANIFEST_SHA256.txt (tung file).
KHONG gom: env Conda/venv, dataset day du, lich su ket qua, engine TensorRT, checkpoint .pt, GT, .gitignore,
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
ZIP = ROOT / "handoff" / "nano_transfer.zip"
PY = sys.executable

VIDEO = "MVI_39031"
CACHE_ID = "yolov8n-coco_conf0.05_all"
SPLIT = "NANO-smoke"
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


def run_blocked(script_rel: str, args: list[str], cwd: Path, backend: str = "np", log: list[str] | None = None):
    """Chay src/<script> trong tien trinh CHAN torch/ultralytics (bang chung khong can hai goi nay)."""
    code = (BLOCK + f"import sys, runpy\nsys.path.insert(0, r{str(cwd / 'src')!r})\n"
            f"sys.argv = [r{str(cwd / script_rel)!r}] + {args!r}\n"
            f"runpy.run_path(r{str(cwd / script_rel)!r}, run_name='__main__')\n")
    env = {**os.environ, "BYTETRACK_BACKEND": backend, "PYTHONIOENCODING": "utf-8"}
    return run([PY, "-c", code], cwd, env, log)


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

    # ---- 3. det cache (meta + npz + sidecar) ----
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
        "cache_lien_quan": {"cache_id": CACHE_ID, "pham_vi": "TOAN BO 1470 frame cua sequence (khong chi 1-300)",
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
    for ten, yaml in TRACKERS.items():
        args = ["--cache-id", CACHE_ID, "--split-name", SPLIT, "--tracker-yaml", yaml, "--tracker-name", f"pc-ref-{ten}", *TRACK_ARGS]
        r = run_blocked("src/run_tracking.py", args, PKG, "np", log)
        if r.returncode != 0:
            sys.exit(f"DUNG: run_tracking np ({ten}) exit {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
        lenh.append("BYTETRACK_BACKEND=np python3 src/run_tracking.py " + subprocess.list2cmdline(args))
        # doi chieu: backend ultralytics (co torch) cung dau vao -> phai y het
        args_ul = ["--cache-id", CACHE_ID, "--split-name", SPLIT, "--tracker-yaml", yaml, "--tracker-name", f"pc-ul-{ten}", *TRACK_ARGS]
        env = {**os.environ, "BYTETRACK_BACKEND": "ultralytics", "PYTHONIOENCODING": "utf-8"}
        r2 = run([PY, str(PKG / "src" / "run_tracking.py"), *args_ul], PKG, env, log)
        if r2.returncode != 0:
            sys.exit(f"DUNG: run_tracking ultralytics ({ten}) exit {r2.returncode}\n{r2.stderr[-2000:]}")
        pj = PKG / "reference" / f"parity_pc_np_vs_ul_{ten}.json"
        r3 = run([PY, str(PKG / "src" / "nano_parity.py"), "--ref", str(PKG / "data/processed/trackers" / SPLIT / f"pc-ul-{ten}"),
                  "--test", str(PKG / "data/processed/trackers" / SPLIT / f"pc-ref-{ten}"), "--json", str(pj)], PKG, None, log)
        parity_pc[ten] = {"exit": r3.returncode, "json": pj.relative_to(PKG).as_posix()}
        if r3.returncode != 0:
            sys.exit(f"DUNG: backend np KHAC ultralytics ({ten}) -> khong dong goi tham chieu")
        shutil.rmtree(PKG / "data/processed/trackers" / SPLIT / f"pc-ul-{ten}")
    lenh += ["", "# Parity (tren Nano, sau khi chay 2 lenh tren voi --tracker-name nano-cv / nano-ekf):",
             f"python3 src/nano_parity.py --ref data/processed/trackers/{SPLIT}/pc-ref-cv  --test data/processed/trackers/{SPLIT}/nano-cv  --json reference/parity_nano_cv.json",
             f"python3 src/nano_parity.py --ref data/processed/trackers/{SPLIT}/pc-ref-ekf --test data/processed/trackers/{SPLIT}/nano-ekf --json reference/parity_nano_ekf.json"]
    (PKG / "reference" / "lenh_tai_hien.sh").write_text("\n".join(lenh) + "\n", encoding="utf-8", newline="\n")
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
                  "tracked_da_sua_ngoai_file_rieng": dirty,
                  "ghi_chu_launcher": "L-F1..L-F4 (review 2d1a8df) da sua o 25c11b6/2e96bc7, CHUA duoc Codex nghiem thu"},
        "moi_truong_pc": env_versions(),
        "phu_thuoc_python_toi_thieu_cho_replay": ["numpy", "scipy", "lap", "pandas", "pyyaml"],
        "phu_thuoc_them_cho_video": ["opencv-python"],
        "khong_can_cho_buoc_2_4": ["torch", "ultralytics"],
        "checkpoint": {k: exp_info[k] for k in ("checkpoint", "checkpoint_sha256", "checkpoint_bytes", "checkpoint_trong_goi")},
        "onnx": {k: exp_info[k] for k in ("onnx", "onnx_sha256", "onnx_bytes", "trang_thai")},
        "det_cache": {"cache_id": CACHE_ID, "video": VIDEO, "meta": meta,
                      "npz_sha256": sha256_file(cdir / f"{VIDEO}.npz"), "sidecar": json.loads((cdir / f"{VIDEO}.json").read_text(encoding="utf-8"))},
        "tham_chieu_pc": {"split": SPLIT, "ten": [f"pc-ref-{t}" for t in TRACKERS], "backend": "np",
                          "cau_hinh": "conf-min 0.25, classes 1 2 3 5 7, max-det 300, fps 25, track-buffer 1.2 s (30 buoc), elapsed steps",
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
    xau = [p.relative_to(PKG).as_posix() for p in all_files if _skip(p.relative_to(PKG).as_posix())
           or p.suffix in (".pt", ".engine") or p.name == "gt.txt"]
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
def verify(zip_path: Path) -> int:
    """Giai nen vao thu muc TAM SACH (khong phai repo), chay chuoi kiem. Ket qua ghi NGOAI goi
    (handoff/logs/nano_transfer/kiem_goi_sach.{log,json}) de hash cua zip da kiem = hash cua zip ban giao."""
    tmp = Path(tempfile.mkdtemp(prefix="nano_pkg_"))
    log: list[str] = [f"# kiem goi sach {time.strftime('%Y-%m-%dT%H:%M:%S')} zip={zip_path} sha256={sha256_file(zip_path)}",
                      f"# giai nen vao {tmp} (thu muc tam, KHONG phai repo); moi buoc replay/parity CHAN torch va ultralytics",
                      "# Y NGHIA: chi xac minh goi DU FILE + DU PHU THUOC Python tren PC; KHONG chung minh tuong thich Nano"]
    kq = {"zip_sha256": sha256_file(zip_path), "zip_bytes": zip_path.stat().st_size, "buoc": {}}
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
        # 1. import backend np, CHAN torch/ultralytics
        code = (BLOCK + f"import sys, json\nsys.path.insert(0, r{str(g / 'src')!r})\nimport ul_backend\n"
                "fp = ul_backend.fingerprint()\nprint(json.dumps({'backend': ul_backend.backend_name(), 'torch': 'torch' in sys.modules, "
                "'ul': 'ultralytics' in sys.modules, 'fp_keys': sorted(fp)[:6]}))\n")
        r = run([PY, "-c", code], g, {**os.environ, "BYTETRACK_BACKEND": "np", "PYTHONIOENCODING": "utf-8"}, log)
        kq["buoc"]["1_import_np"] = {"exit": r.returncode, "out": r.stdout.strip()[-300:], "err": r.stderr.strip()[-300:]}
        ok_all &= r.returncode == 0 and '"torch": false' in r.stdout
        # 2. test_nano_parity
        r = run_blocked("src/test_nano_parity.py", [], g, "np", log)
        kq["buoc"]["2_test_nano_parity"] = {"exit": r.returncode, "ket_qua": next((ln for ln in r.stdout.splitlines() if "KET QUA" in ln), "")}
        ok_all &= r.returncode == 0
        # 3. replay cache CV/EKF (CHAN torch/ultralytics) -> 4. parity voi pc-ref
        for ten, yaml in TRACKERS.items():
            args = ["--cache-id", CACHE_ID, "--split-name", SPLIT, "--tracker-yaml", yaml, "--tracker-name", f"kiem-{ten}", *TRACK_ARGS]
            r = run_blocked("src/run_tracking.py", args, g, "np", log)
            pj = g / "reference" / f"parity_kiem_goi_sach_{ten}.json"
            r2 = run_blocked("src/nano_parity.py", ["--ref", f"data/processed/trackers/{SPLIT}/pc-ref-{ten}",
                                                   "--test", f"data/processed/trackers/{SPLIT}/kiem-{ten}", "--json", str(pj)], g, "np", log)
            pjd = json.loads(pj.read_text(encoding="utf-8")) if pj.is_file() else {}
            kq["buoc"][f"3_4_replay_parity_{ten}"] = {"replay_exit": r.returncode, "parity_exit": r2.returncode,
                                                     "n_dat": pjd.get("n_dat"), "n_video": pjd.get("n_video"),
                                                     "video": [{k: v.get(k) for k in ("video", "n_ref", "n_test", "dat", "max_d_bbox", "max_d_conf")}
                                                               for v in pjd.get("ket_qua", [])],
                                                     "parity_json": pjd}
            ok_all &= r.returncode == 0 and r2.returncode == 0
        # 5. ONNX doc duoc (onnx co tren PC; tren Nano se la trtexec)
        code = (f"import onnx, hashlib, json\np = r{str(g / 'models/export' / ONNX_NAME)!r}\nm = onnx.load(p)\nonnx.checker.check_model(m)\n"
                "print(json.dumps({'ir': m.ir_version, 'opset': [(o.domain, o.version) for o in m.opset_import], 'n_node': len(m.graph.node), "
                "'inputs': [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input], "
                "'sha256': hashlib.sha256(open(p, 'rb').read()).hexdigest()}))\n")
        r = run([PY, "-c", code], g, None, log)
        kq["buoc"]["5_onnx_doc_duoc"] = {"exit": r.returncode, "out": r.stdout.strip()[-400:]}
        ok_all &= r.returncode == 0
        # 6. doan smoke doc duoc bang cv2 (JPG + mp4)
        code = (f"import cv2, json, glob\nd = r{str(g / 'data/smoke')!r}\nfs = sorted(glob.glob(d + '/{VIDEO}_f0001-0300/*.jpg'))\n"
                f"im = cv2.imread(fs[0]); cap = cv2.VideoCapture(d + '/{VIDEO}_f0001-0300.mp4'); n = 0\n"
                "while True:\n    ok, _ = cap.read()\n    if not ok: break\n    n += 1\n"
                "print(json.dumps({'n_jpg': len(fs), 'jpg_shape': list(im.shape), 'mp4_frames_doc': n}))\n")
        r = run([PY, "-c", code], g, None, log)
        kq["buoc"]["6_smoke_doc_duoc"] = {"exit": r.returncode, "out": r.stdout.strip()[-200:]}
        ok_all &= r.returncode == 0 and '"n_jpg": 300' in r.stdout and '"mp4_frames_doc": 300' in r.stdout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    kq["ket_luan"] = "DAT: goi du file/du phu thuoc de replay + parity tren PC (KHONG chung minh tuong thich Nano)" if ok_all else "KHONG DAT"
    log.append(f"KET LUAN: {kq['ket_luan']}")
    out = ROOT / "handoff" / "logs" / "nano_transfer" / "kiem_goi_sach.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(log) + "\n", encoding="utf-8")
    (ROOT / "handoff" / "logs" / "nano_transfer" / "kiem_goi_sach.json").write_text(json.dumps(kq, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n".join(log[-12:]))
    print(f"-> {out}")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "verify"])
    ap.add_argument("--onnx-check", default=None, help="JSON doi chieu trong so ONNX vs re-export (build)")
    ap.add_argument("--zip", default=str(ZIP))
    a = ap.parse_args()
    if a.mode == "build":
        return build(Path(a.onnx_check) if a.onnx_check else None)
    return verify(Path(a.zip))


if __name__ == "__main__":
    raise SystemExit(main())
