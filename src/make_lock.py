"""
make_lock.py -- (MOC 2, muc 5) Ghi BAN KHOA cau hinh truoc khi chay tap TEST: commit, thoi diem, sha256 cua
tung file cau hinh (experiment + tracker yaml), tieu chi lua chon, danh sach sequence, cache meta, moi truong.
Sau khi ghi, MOI thay doi cau hinh cho tap test deu phai them muc vao `configs/phase2/LOCK_moc2.md`.

    python src/make_lock.py --exps configs/phase2/experiments/exp{A,B,C}_detrac_test.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import det_cache  # noqa: E402
import run_tracking  # noqa: E402
from run_experiment import load_exp, plan  # noqa: E402

ROOT = config.PROJECT_ROOT


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--out", default="results/phase2/lock/lock_moc2.json")
    a = ap.parse_args()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_branch": subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_dirty_files": [ln for ln in subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).splitlines() if ln],
            "selection_rule": "Tieu chi dang ky truoc (configs/phase2/PREREG_moc2.md): HOTA c25 tren dev, chi coi la"
                              " cai thien khi CI 95% bootstrap (COMBINED, theo video) KHONG chua 0. Khong dung ket qua test.",
            "platform": platform.platform(), "python": sys.version.split()[0], "experiments": {}}
    for m in ("ultralytics", "torch", "trackeval", "numpy", "scipy", "pandas"):
        try:
            mod = __import__(m); lock[f"version_{m}"] = getattr(mod, "__version__", "?")
        except Exception as e:  # noqa: BLE001
            lock[f"version_{m}"] = f"? ({e})"
    for f in a.exps:
        exp = load_exp(f)
        rows = plan(exp)
        videos = json.loads((ROOT / exp["manifest"]).read_text(encoding="utf-8"))["videos"]
        meta = det_cache.load_meta(det_cache.CACHE_ROOT / exp["cache_id"])
        yamls = sorted({r["yaml"] for r in rows})
        lock["experiments"][exp["name"]] = {
            "file": f, "file_sha256": sha256(ROOT / f), "split_name": exp["split_name"],
            "n_videos": len(videos), "videos": videos, "cache_id": exp["cache_id"],
            "cache_weights_sha256": meta.get("weights_sha256"), "cache_conf": meta.get("conf"),
            "cache_max_det": meta.get("max_det"), "cache_versions": meta.get("versions"),
            "conditions": exp["conditions"], "classes": exp["classes"], "max_det": exp["max_det"],
            "fps": exp["fps"], "track_buffer_sec": exp.get("track_buffer_sec"), "elapsed_mode": exp["elapsed_mode"],
            "tracker_yaml_sha256": {y: sha256(Path(run_tracking.resolve_tracker_yaml(y))) for y in yamls},
            "runs": [{"cond": r["cond"], "tracker": r["tracker"], "yaml": r["yaml"], "set": r["set"],
                      "seed": r["seed"], "dir": r["dir"]} for r in rows]}
    out.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}  ({len(lock['experiments'])} thi nghiem, "
          f"{sum(len(v['runs']) for v in lock['experiments'].values())} lan chay, commit {lock['git_commit'][:8]}, "
          f"file tracked dang sua: {lock['git_dirty_files'] or 'khong'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
