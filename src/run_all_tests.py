"""
run_all_tests.py -- Chay TUNG bo test (script thuan) bang subprocess, ghi LENH, EXIT CODE, ket qua
(PASS / FAIL / SKIPPED / NOT_RUN) va toan bo stdout+stderr vao handoff/logs/, de nguoi review doi chieu.

    PASS     exit 0
    FAIL     exit != 0 (loi test hoac loi chay)
    SKIPPED  dieu kien tien quyet thieu (GPU / dataset / ket qua tracker) - ghi ly do, KHONG chay
    NOT_RUN  bi loai bang --only/--skip

    python src/run_all_tests.py                      # tat ca
    python src/run_all_tests.py --only test_filter_config test_time_units
    python src/run_all_tests.py --skip test_run_tracking_replay
Ket qua: handoff/logs/tests_<YYYYmmdd-HHMM>.log (day du), handoff/logs/tests_summary.csv (1 dong / test),
handoff/logs/tests_latest.log (ban sao cua log day du gan nhat).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

#: (ten, args, dieu kien tien quyet: None | callable -> str ly do skip hoac "")
def _need_gpu() -> str:
    try:
        import torch
        return "" if torch.cuda.is_available() else "khong co GPU CUDA"
    except Exception as e:  # noqa: BLE001
        return f"torch: {e}"


def _need_detrac_images() -> str:
    d = ROOT / "data/extracted/DETRAC-Images/MVI_39031"
    return "" if d.is_dir() else "thieu anh UA-DETRAC (data/extracted/DETRAC-Images)"


def _need_dev_runs() -> str:
    d = ROOT / "data/processed/trackers/DETRAC-all/p2-groupA_detrac_dev-A-cv/data"
    return "" if d.is_dir() else "thieu ket qua tracker dev (p2-groupA_detrac_dev-A-cv)"


def _need_filterpy() -> str:
    """Ma upstream SORT can filterpy (cai rieng vao third_party/_testdeps, khong vao moi truong chinh)."""
    import importlib.util
    import sys as _s
    p = ROOT / "third_party" / "_testdeps"
    if p.is_dir():
        _s.path.insert(0, str(p))
    ok = importlib.util.find_spec("filterpy") is not None
    return "" if ok else ("thieu filterpy: python -m pip install --no-deps --target third_party/_testdeps "
                          "filterpy==1.4.5")


def _need_cache() -> str:
    d = ROOT / "data/interim/det_cache/yolov8n-coco_conf0.05_all/MVI_39031.npz"
    return "" if d.exists() else "thieu detection cache MVI_39031"


TESTS = [
    ("test_ekf_ctrv", [], None),
    ("test_ukf_ctrv", [], None),
    ("test_init_regression", [], None),
    ("test_tracker_lifecycle", [], None),
    ("test_occlusion_eval", [], None),
    ("test_filter_config", [], None),
    ("test_time_units", [], None),
    ("test_timestep_integration", [], None),
    ("test_det_cache", [], None),
    ("test_cache_provenance", [], None),
    ("test_trackeval_ext", [], None),
    ("test_bdd100k_adapter", [], None),
    ("test_pf_ctrv", [], None),
    ("test_sort_tracker", [], None),
    ("test_smoke_phase2", [], None),
    ("test_lost_recovery", [], _need_cache),
    ("test_sort_upstream_parity", [], _need_filterpy),
    ("test_run_experiment_guard", [], None),
    ("test_run_resume_guard", [], None),
    ("test_review_delta", [], None),
    ("test_boot_stats", [], _need_dev_runs),
    ("test_run_tracking_replay", ["--n-frames", "150"], lambda: _need_gpu() or _need_detrac_images()),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--out-dir", default="handoff/logs")
    a = ap.parse_args()
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M")
    log_path = out / f"tests_{stamp}.log"
    rows = []
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# run_all_tests {time.strftime('%Y-%m-%d %H:%M:%S')}  python={sys.version.split()[0]}  cwd={ROOT}\n")
        try:
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        except Exception:  # noqa: BLE001
            commit = "?"
        log.write(f"# git HEAD {commit}\n\n")
        for name, args, pre in TESTS:
            cmd = [PY, f"src/{name}.py", *args]
            cmd_str = " ".join(["python", f"src/{name}.py", *args])
            if (a.only and name not in a.only) or name in a.skip:
                rows.append({"test": name, "command": cmd_str, "status": "NOT_RUN", "exit_code": "", "seconds": 0, "note": "loai boi --only/--skip"})
                continue
            reason = pre() if pre else ""
            if reason:
                rows.append({"test": name, "command": cmd_str, "status": "SKIPPED", "exit_code": "", "seconds": 0, "note": reason})
                log.write(f"=== {name}: SKIPPED ({reason})\n$ {cmd_str}\n\n")
                print(f"  {name:<28} SKIPPED  {reason}")
                continue
            t0 = time.time()
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
            sec = time.time() - t0
            status = "PASS" if r.returncode == 0 else "FAIL"
            tail = [ln for ln in r.stdout.splitlines() if "KET QUA" in ln or "phep kiem tra" in ln]
            note = tail[-1].strip() if tail else ""
            rows.append({"test": name, "command": cmd_str, "status": status, "exit_code": r.returncode,
                         "seconds": round(sec, 1), "note": note})
            log.write(f"=== {name}: {status} (exit {r.returncode}, {sec:.1f}s)\n$ {cmd_str}\n--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}\n\n")
            print(f"  {name:<28} {status:<7} exit={r.returncode} {sec:6.1f}s  {note}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "tests_summary.csv", index=False)
    shutil.copyfile(log_path, out / "tests_latest.log")
    n_pass = int((df.status == "PASS").sum()); n_fail = int((df.status == "FAIL").sum())
    n_skip = int((df.status == "SKIPPED").sum())
    print(f"\nPASS {n_pass}  FAIL {n_fail}  SKIPPED {n_skip}  NOT_RUN {int((df.status == 'NOT_RUN').sum())}  -> {log_path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
