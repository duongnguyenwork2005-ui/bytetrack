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
import run_trackeval  # noqa: E402
import ul_backend  # noqa: E402
from run_experiment import load_exp, plan  # noqa: E402

ROOT = config.PROJECT_ROOT


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha256_raw(p: Path) -> str:
    """Hash BYTE THO - dung cho file du lieu lon (khong chuan hoa xuong dong)."""
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _label_source(exp: dict) -> dict:
    """Nguon nhan (neu manifest ghi lai) - de biet ban khoa gan voi file nhan nao."""
    try:
        m = json.loads((ROOT / exp["manifest"]).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return {k: m[k] for k in ("nguon_nhan", "nguon_nhan_sha256", "thieu_trong_nguon",
                              "sequence_da_xem_truoc") if k in m}


def _seqmap_sha(split: str) -> str | None:
    p = config.PROCESSED_DIR / "seqmaps" / f"{split}.txt"
    return sha256(p) if p.exists() else None


def _gt_sha(split: str, videos: list[str]) -> dict:
    out = {}
    for v in videos:
        g = config.PROCESSED_DIR / split / v / "gt" / "gt.txt"
        if g.exists():
            out[v] = sha256(g)[:16]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", required=True)
    ap.add_argument("--out", default="results/phase2/lock/lock_moc2.json")
    ap.add_argument("--selection-rule", default=None, help="ghi de tieu chi lua chon trong ban khoa")
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
            "platform": platform.platform(), "python": sys.version.split()[0],
            # BACKEND TRACKER co hieu luc khi ghi ban khoa (khac backend DETECTOR cua cache)
            "tracker_backend": ul_backend.fingerprint(),
            "experiments": {}}
    if a.selection_rule:
        lock["selection_rule"] = a.selection_rule
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
                      "seed": r["seed"], "dir": r["dir"]} for r in rows],
            # --- provenance dau vao va cach cham diem (khoa TRUOC khi xem ket qua) ---
            "dataset": exp.get("dataset"),
            "evaluator_dataset_class": run_trackeval.dataset_class_for(exp.get("dataset"),
                                                                       exp["split_name"]),
            "manifest_file": exp["manifest"], "manifest_sha256": sha256(ROOT / exp["manifest"]),
            "label_source": _label_source(exp),
            "gt_seqmap_sha256": _seqmap_sha(exp["split_name"]),
            "gt_sha256_per_video": _gt_sha(exp["split_name"], videos),
            "pairs": exp.get("pairs"), "baseline": exp.get("baseline"),
            "occlusion_enabled": bool(exp.get("occlusion")),
            "aggregation": {
                "combined": "TrackEval COMBINED_SEQ (tai to hop tu thong ke du - src/boot_stats.py)",
                "paired_bootstrap": "ghep cap THEO VIDEO, 5000 lan, seed 0",
                "pf_multi_seed": ("trung binh METRIC theo seed cho bang chinh; bootstrap trung binh "
                                  "THONG KE DU roi tai to hop (hai dai luong khac nhau voi metric phi "
                                  "tuyen - xem PHASE2_REPORT muc 15.2)"),
            }}
    out.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {out}  ({len(lock['experiments'])} thi nghiem, "
          f"{sum(len(v['runs']) for v in lock['experiments'].values())} lan chay, commit {lock['git_commit'][:8]}, "
          f"file tracked dang sua: {lock['git_dirty_files'] or 'khong'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
