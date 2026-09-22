"""
test_nano_run_cache.py -- Regression cho `src/nano_run_cache.py` + `src/nano_env_check.py` + `src/nano_preflight.py`
(review 20207ec: N15-P1 sidecar hong, N15-P2 env check that bai, N15-P3 env check chap nhan asset khong dung duoc).

Moi phep kiem goi ENTRY POINT THAT (subprocess) trong mot THU MUC DANG GOI tam (package-shaped): src/ + configs/ +
third_party/ultralytics_trackers/ chep tu repo; hai cache TONG HOP nho (dung ten hop dong: nano_smoke_MVI_39031_f0001-0300
3 frame, yolov8n-coco_conf0.05_all 5 frame; video MVI_39031); bon tham chieu pc-ref-{cv,ekf} cho NANO-smoke/NANO-full
duoc tao BANG CHINH run_tracking.py (backend np, chan torch/ultralytics) trong thu muc tam; reference/pc_ref_sha256.txt.
Khong dung dataset/GT/cache that; khong dung cham repo.

Cac phep kiem:
  1. Dau vao hop le, --tag kiem: exit 0, KET LUAN DAT, 4 parity DAT (smoke + full, CV + EKF), 4 thu muc kiem-* co,
     sha256 TOAN BO cay pc-ref-* truoc == sau (tinh doc lap trong test).
  2. Thieu 1 file pc-ref (NANO-full/pc-ref-ekf/data/MVI_39031.txt): exit 2, KHONG co thu muc bad1-*, log neu duong dan.
  3. Sidecar smoke `{ bad json`: exit 2, KHONG chay tracker (JSON khong co buoc 2_*), KHONG co bad2-*, log neu 'sidecar'.
  4. Thieu `lap` (goi gia `lap` nem ImportError dat truoc tren PYTHONPATH -> import lap that bai trong moi tien trinh con):
     nano_env_check.py nonzero; nano_run_cache.py exit 2, khong chay tracker, khong co bad3-*, JSON co tail env_check.
  5. nano_env_check.py doc lap: xoa 2 NPZ + 4 file tham chieu -> nonzero, KHONG in 'DAT (1-4)'; chi xoa NPZ -> nonzero;
     khoi phuc -> exit 0.
  6. Sidecar sai schema (n_frames lech): nano_env_check nonzero; nano_run_cache exit 2, khong co bad4-*.
  7. --tag pc-ref-x bi tu choi (exit 2), khong tao gi.
Tren ma 20207ec (truoc sua): 3, 4, 5, 6 FAIL (xem handoff/logs/review17/regression_bat_ma_cu.log).

    python src/test_nano_run_cache.py
"""
from __future__ import annotations

import hashlib
import json
import os
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
VIDEO = "MVI_39031"
SMOKE, FULL = "nano_smoke_MVI_39031_f0001-0300", "yolov8n-coco_conf0.05_all"
TRACK_ARGS = ["--videos", VIDEO, "--conf-min", "0.25", "--classes", "1", "2", "3", "5", "7", "--max-det", "300",
              "--fps", "25", "--track-buffer-sec", "1.2", "--elapsed-mode", "steps"]
BLOCK = ("import sys\nclass _B:\n    def find_spec(self, n, p=None, t=None):\n"
         "        if n.split('.')[0] in ('torch', 'ultralytics'): raise ImportError('CHAN ' + n)\nsys.meta_path.insert(0, _B())\n")


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for ln in str(detail).strip().splitlines()[-6:]:
            print(f"         {ln}")
    assert ok, name


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def hash_cay(g: Path) -> list[str]:
    return sorted(f"{sha256_file(p)}  {p.relative_to(g).as_posix()}" for p in (g / "data/processed/trackers").rglob("*")
                  if p.is_file() and "pc-ref-" in p.as_posix())


def lam_cache(cdir: Path, n_frame: int, seed: int) -> None:
    """Cache tong hop: moi frame 2-3 box xe (lop 2) di chuyen deu + 1 box conf thap; frame cuoi rong neu n_frame >= 4."""
    rng = np.random.default_rng(seed)
    cdir.mkdir(parents=True)
    fi, npf, chunks = [], [], []
    for i in range(n_frame):
        boxes = []
        if not (n_frame >= 4 and i == n_frame - 2):                       # mot frame rong o giua
            for k in range(2):
                x = 100 + 40 * i + 300 * k; y = 200 + 10 * k
                boxes.append([x, y, x + 120, y + 80, 0.9 - 0.05 * k, 2.0])
            boxes.append([600.0, 50.0, 640.0, 90.0, 0.08, 7.0])            # conf thap: bi loc o c25
        arr = np.asarray(boxes, np.float32).reshape(-1, 6) if boxes else np.zeros((0, 6), np.float32)
        fi.append(i + 1); npf.append(len(arr)); chunks.append(arr)
    data = np.concatenate(chunks) if chunks else np.zeros((0, 6), np.float32)
    np.savez_compressed(cdir / f"{VIDEO}.npz", frame_idx=np.asarray(fi, np.int32), n_per_frame=np.asarray(npf, np.int32),
                        unreadable=np.zeros(n_frame, bool), truncated=np.zeros(n_frame, bool),
                        time_s=np.arange(n_frame, dtype=np.float64) / 25.0, data=data, orig_shape=np.asarray([540, 960], np.int32))
    (cdir / "meta.json").write_text(json.dumps({"backend": "gt-synthetic", "weights": "x", "weights_sha256": None, "imgsz": 640,
                                                "conf": 0.05, "iou": 0.7, "classes": None, "max_det": 3000, "half": False,
                                                "agnostic_nms": False, "augment": False, "batch": 1, "device": "cpu",
                                                "mask_ignored_regions": True, "dataset": "video_file", "part": "images",
                                                "gt_jitter": 0.0, "gt_dropout": 0.0, "gt_seed": 0, "gt_conf_range": [0.3, 0.95],
                                                "versions": {}, "git_commit": None, "created": "test"}, indent=1), encoding="utf-8")
    (cdir / f"{VIDEO}.json").write_text(json.dumps({"video": VIDEO, "n_frames": n_frame, "n_boxes": int(len(data)),
                                                    "n_unreadable": 0, "n_truncated": 0, "created": "test"}, indent=1), encoding="utf-8")


def dung_goi(tmp: Path) -> Path:
    g = tmp / "nano_transfer"
    for rel in ("src", "configs", "third_party/ultralytics_trackers"):
        shutil.copytree(ROOT / rel, g / rel, ignore=shutil.ignore_patterns("__pycache__", "demo", "*.pyc"))
    lam_cache(g / "data/interim/det_cache" / SMOKE, 3, 1)
    lam_cache(g / "data/interim/det_cache" / FULL, 5, 2)
    # tham chieu PC THAT bang chinh run_tracking.py (np, chan torch/ul)
    for cid, split in ((SMOKE, "NANO-smoke"), (FULL, "NANO-full")):
        for t, yaml in (("cv", "bytetrack.yaml"), ("ekf", "configs/bytetrack_ekf_ctrv.yaml")):
            r = chay(g, "src/run_tracking.py", ["--cache-id", cid, "--split-name", split, "--tracker-yaml", yaml,
                                                "--tracker-name", f"pc-ref-{t}", *TRACK_ARGS])
            assert r.returncode == 0, r.stdout[-800:] + r.stderr[-800:]
    (g / "reference").mkdir()
    (g / "reference/pc_ref_sha256.txt").write_text("\n".join(hash_cay(g)) + "\n", encoding="utf-8")
    return g


def chay(g: Path, script: str, args: list[str], py: str = sys.executable, env_extra: dict | None = None):
    code = (BLOCK + "import sys, runpy\nsys.path.insert(0, 'src')\n"
            f"sys.argv = [{script!r}] + {args!r}\nrunpy.run_path({script!r}, run_name='__main__')\n")
    env = {**os.environ, "BYTETRACK_BACKEND": "np", "PYTHONIOENCODING": "utf-8", **(env_extra or {})}
    return subprocess.run([py, "-c", code], cwd=str(g), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)


def goi_script(g: Path, script: str, args: list[str], env_extra: dict | None = None):
    """Goi entry point THAT nhu nguoi dung: python src/<script> ... (khong bao BLOCK; script tu chan khi can)."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **(env_extra or {})}
    return subprocess.run([sys.executable, script, *args], cwd=str(g), capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)


def outputs(g: Path, tag: str) -> list[str]:
    return sorted(p.relative_to(g).as_posix() for p in (g / "data/processed/trackers").glob(f"*/{tag}-*"))


def json_moi(g: Path) -> dict:
    fs = sorted((g / "reference").glob("nano_run_*.json"))
    return json.loads(fs[-1].read_text(encoding="utf-8")) if fs else {}


def main() -> int:  # noqa: C901
    failed = 0
    tmp = Path(tempfile.mkdtemp(prefix="nrc_"))
    try:
        g = dung_goi(tmp)
        truoc = hash_cay(g)
        _check("0. goi tam dung xong: 2 cache tong hop (3/5 frame, co frame rong) + 4 pc-ref that tu run_tracking (np) + pc_ref_sha256.txt",
               len(truoc) >= 4 and all((g / "data/processed/trackers" / s / f"pc-ref-{t}" / "data" / f"{VIDEO}.txt").is_file()
                                       for s in ("NANO-smoke", "NANO-full") for t in ("cv", "ekf")), f"{len(truoc)} file pc-ref")

        # --- 1. hop le ---
        r = goi_script(g, "src/nano_run_cache.py", ["--tag", "kiem"])
        j = json_moi(g); sau = hash_cay(g)
        chay_ok = {k: (v.get("dat"), v.get("replay_exit"), v.get("parity_exit")) for k, v in j.get("buoc", {}).items() if k.startswith("2_")}
        _check("1. dau vao hop le --tag kiem: exit 0, KET LUAN DAT, 4 parity DAT (smoke/full x CV/EKF), 4 thu muc kiem-*, "
               "hash cay pc-ref-* truoc == sau (tinh doc lap), khop ban dong goi",
               r.returncode == 0 and j.get("ket_luan") == "DAT" and len(chay_ok) == 4 and all(v == (True, 0, 0) for v in chay_ok.values())
               and len(outputs(g, "kiem")) == 4 and sau == truoc and j["buoc"].get("1_pc_ref_khop_ban_dong_goi") is True
               and j["buoc"].get("4_pc_ref_khong_doi_sau_chay") is True,
               f"exit={r.returncode} ket_luan={j.get('ket_luan')} chay={chay_ok} out={outputs(g, 'kiem')}\n{(r.stdout + r.stderr)[-300:]}")

        # --- 2. thieu 1 file pc-ref ---
        p_ref = g / "data/processed/trackers/NANO-full/pc-ref-ekf/data" / f"{VIDEO}.txt"
        p_ref.rename(p_ref.with_suffix(".txt.tam"))
        try:
            r = goi_script(g, "src/nano_run_cache.py", ["--tag", "bad1"]); j = json_moi(g)
        finally:
            p_ref.with_suffix(".txt.tam").rename(p_ref)
        _check("2. thieu NANO-full/pc-ref-ekf/data/MVI_39031.txt -> exit 2, KHONG co bad1-*, khong buoc 2_*, log neu duong dan",
               r.returncode == 2 and outputs(g, "bad1") == [] and not any(k.startswith("2_") for k in j.get("buoc", {}))
               and "pc-ref-ekf" in r.stdout + r.stderr, f"exit={r.returncode} out={outputs(g, 'bad1')}\n{(r.stdout + r.stderr)[-300:]}")

        # --- 3. sidecar hong (N15-P1) ---
        side = g / "data/interim/det_cache" / SMOKE / f"{VIDEO}.json"; goc_side = side.read_bytes()
        side.write_text("{ bad json", encoding="utf-8")
        try:
            r = goi_script(g, "src/nano_run_cache.py", ["--tag", "bad2"]); j = json_moi(g)
        finally:
            side.write_bytes(goc_side)
        _check("3. (N15-P1) sidecar smoke '{ bad json' -> exit 2 TRUOC khi chay tracker (khong buoc 2_*), KHONG co bad2-*, log neu 'sidecar'",
               r.returncode == 2 and outputs(g, "bad2") == [] and not any(k.startswith("2_") for k in j.get("buoc", {}))
               and "sidecar" in (r.stdout + r.stderr), f"exit={r.returncode} out={outputs(g, 'bad2')}\n{(r.stdout + r.stderr)[-300:]}")

        # --- 4. thieu lap (N15-P2): goi gia `lap` nem ImportError, dat truoc tren PYTHONPATH ---
        fake = tmp / "fake_lap"; fake.mkdir()
        (fake / "lap.py").write_text("raise ImportError('lap bi chan boi test (mo phong thieu lap)')\n", encoding="utf-8")
        env = {"PYTHONPATH": str(fake) + os.pathsep + os.environ.get("PYTHONPATH", "")}
        r0 = goi_script(g, "src/nano_env_check.py", [], env)
        r = goi_script(g, "src/nano_run_cache.py", ["--tag", "bad3"], env); j = json_moi(g)
        tail = " ".join(j.get("buoc", {}).get("0_env_check", {}).get("tail", []))
        _check("4. (N15-P2) thieu `lap` (goi gia nem ImportError): nano_env_check nonzero; nano_run_cache exit 2, khong chay tracker, "
               "KHONG co bad3-*, JSON co tail cua env_check neu 'lap'",
               r0.returncode != 0 and r.returncode == 2 and outputs(g, "bad3") == [] and not any(k.startswith("2_") for k in j.get("buoc", {}))
               and "lap" in tail, f"env_check exit={r0.returncode} run exit={r.returncode} out={outputs(g, 'bad3')} tail={tail[:160]}")

        # --- 5. nano_env_check doc lap (N15-P3) ---
        mv = []
        for p in list((g / "data/interim/det_cache").glob(f"*/{VIDEO}.npz")) + list((g / "data/processed/trackers").glob(f"*/pc-ref-*/data/{VIDEO}.txt")):
            q = p.with_suffix(p.suffix + ".tam"); p.rename(q); mv.append((p, q))
        try:
            r5a = goi_script(g, "src/nano_env_check.py", [])
        finally:
            for p, q in mv:
                if q.name.endswith(".txt.tam"):
                    q.rename(p)
        # con thieu NPZ
        r5b = goi_script(g, "src/nano_env_check.py", [])
        for p, q in mv:
            if q.exists():
                q.rename(p)
        r5c = goi_script(g, "src/nano_env_check.py", [])
        _check("5. (N15-P3) nano_env_check: xoa 2 NPZ + 4 file tham chieu -> nonzero, KHONG in 'DAT (1-4)'; chi thieu NPZ -> nonzero; "
               "khoi phuc -> exit 0 + 'DAT (1-4)'",
               r5a.returncode != 0 and "DAT (1-4)" not in r5a.stdout and r5b.returncode != 0 and "DAT (1-4)" not in r5b.stdout
               and r5c.returncode == 0 and "DAT (1-4)" in r5c.stdout,
               f"a={r5a.returncode} b={r5b.returncode} c={r5c.returncode}\n" + "\n".join(l for l in r5a.stdout.splitlines() if "KET LUAN" in l))

        # --- 6. sidecar sai schema ---
        d = json.loads(goc_side.decode("utf-8")); d["n_frames"] = 99
        side.write_text(json.dumps(d), encoding="utf-8")
        try:
            r6a = goi_script(g, "src/nano_env_check.py", [])
            r6b = goi_script(g, "src/nano_run_cache.py", ["--tag", "bad4"])
        finally:
            side.write_bytes(goc_side)
        _check("6. sidecar sai schema (n_frames=99 != 3): nano_env_check nonzero; nano_run_cache exit 2, KHONG co bad4-*",
               r6a.returncode != 0 and r6b.returncode == 2 and outputs(g, "bad4") == [], f"a={r6a.returncode} b={r6b.returncode}")

        # --- 7. tag pc-ref ---
        r = goi_script(g, "src/nano_run_cache.py", ["--tag", "pc-ref-x"])
        _check("7. --tag pc-ref-x -> exit 2, khong tao gi", r.returncode == 2 and outputs(g, "pc-ref-x") == [], f"exit={r.returncode}")

        _check("   cay pc-ref-* sau TOAN BO cac phep kiem y het ban dau", hash_cay(g) == truoc)
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
