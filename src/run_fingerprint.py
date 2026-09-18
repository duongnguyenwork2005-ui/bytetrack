"""
run_fingerprint.py -- (sau review Codex) DAU VAN TAY cua mot lan chay de quyet dinh co duoc TAI SU DUNG
ket qua danh gia da co hay khong.

VAN DE DUOC SUA: `run_experiment.stage_eval` truoc day bo qua TrackEval chi vi ba file CSV/JSON da ton tai
-> neu tracking duoc chay lai (mã khac, cau hinh khac, cache khac) thi metric CU van duoc giu, tao ra bang
ket qua "tracking moi + metric cu" ma khong co canh bao nao.

DAU VAN TAY gom (tat ca deu anh huong ket qua danh gia):
    tracking  : sha256 cua TUNG file `data/<video>.txt` cua lan chay (theo dung danh sach sequence)
    config    : tracker_cfg + overrides + conf_min + classes + max_det + elapsed_mode + fps + seed
    cache     : meta.json cua cache + sha256 cua tung <video>.npz duoc dung
    sequences : danh sach video (co thu tu)
    code      : git commit + sha256 cua cac module anh huong ket qua
                (run_tracking, tracker_ctrv, sort_tracker, ekf_ctrv, ukf_ctrv, pf_ctrv, lost_recovery,
                 lost_association, time_units, det_cache, occlusion_eval, boot_stats, run_trackeval,
                 trackeval_ext)
    evaluator : phien ban trackeval + lop dataset dung de danh gia
Ghi ra `eval_meta.json` (canh ket qua TrackEval) va `occl_meta.json` (canh ket qua occlusion_eval).
`compare(old, new)` tra ve danh sach truong LECH -> runner in ly do chay lai.

Luu y: sha256 file .txt/.npz duoc tinh tren BYTE nguyen ban (khong chuan hoa xuong dong) vi day la file do
chinh pipeline sinh ra, khong qua git.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

#: Module anh huong truc tiep ket qua tracking/danh gia (doi -> phai danh gia lai)
CODE_FILES = ("run_tracking.py", "run_experiment.py", "tracker_ctrv.py", "sort_tracker.py", "ekf_ctrv.py",
              "ekf_ctrv_clamped.py", "ukf_ctrv.py", "pf_ctrv.py", "lost_recovery.py", "lost_association.py",
              "time_units.py", "det_cache.py", "occlusion_eval.py", "boot_stats.py", "run_trackeval.py",
              "trackeval_ext.py", "assoc_log.py",
              # backend ByteTrack (ultralytics da cai / ban vendor khong phu thuoc torch)
              "ul_backend.py", "bytetrack_np.py")


def sha256_file(p: Path, missing: str = "MISSING") -> str:
    if not p.exists():
        return missing
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def code_fingerprint() -> dict:
    """Dau van tay cua MA sinh ra ket qua.

    Gom ca DANH TINH BACKEND: doi `BYTETRACK_BACKEND` (ultralytics <-> np), doi ban vendor, hay doi
    phien ban numpy/scipy/lap deu lam fingerprint doi => runner KHONG duoc tai su dung ket qua cu.
    """
    import ul_backend                                       # noqa: PLC0415

    src = Path(__file__).resolve().parent
    return {"git_commit": _git_commit(),
            "modules": {f: sha256_file(src / f)[:16] for f in CODE_FILES},
            "backend": ul_backend.fingerprint()}


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.PROJECT_ROOT,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return "?"


def evaluator_fingerprint(dataset_cls: str) -> dict:
    try:
        import trackeval
        ver = getattr(trackeval, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        ver = f"? ({e})"
    return {"trackeval": ver, "dataset_class": dataset_cls}


def tracking_fingerprint(tracker_dir: Path, videos: list[str]) -> dict:
    """sha256 cua tung file ket qua tracking + so file thieu."""
    files = {v: sha256_file(tracker_dir / "data" / f"{v}.txt") for v in videos}
    return {"n_videos": len(videos), "n_missing": sum(1 for h in files.values() if h == "MISSING"),
            "sha256": sha256_obj(files), "per_video": {v: h[:16] for v, h in files.items()}}


def cache_fingerprint(cache_id: str, videos: list[str]) -> dict:
    import det_cache
    cdir = det_cache.CACHE_ROOT / cache_id
    try:
        meta = det_cache.load_meta(cdir)
    except FileNotFoundError:
        meta = {"MISSING": str(cdir)}
    npz = {v: sha256_file(cdir / f"{v}.npz")[:16] for v in videos}
    return {"cache_id": cache_id, "meta_sha256": sha256_obj(meta), "npz_sha256": sha256_obj(npz),
            "weights_sha256": meta.get("weights_sha256"), "conf": meta.get("conf"),
            "classes": meta.get("classes"), "max_det": meta.get("max_det")}


def run_fingerprint(exp: dict, r: dict, videos: list[str], cfg, dataset_cls: str = "mot") -> dict:
    """Dau van tay day du cua mot lan chay (dung cho buoc danh gia TrackEval)."""
    tracker_dir = config.PROCESSED_DIR / "trackers" / exp["split_name"] / r["dir"]
    cfg_d = vars(cfg) if not isinstance(cfg, dict) else cfg
    return {
        "dir": r["dir"], "split_name": exp["split_name"],
        "tracking": tracking_fingerprint(tracker_dir, videos),
        "config": sha256_obj({"tracker_cfg": cfg_d, "overrides": r.get("set"), "conf_min": r.get("conf_min"),
                              "classes": exp["classes"], "max_det": exp["max_det"],
                              "elapsed_mode": exp["elapsed_mode"], "fps": exp["fps"], "seed": r.get("seed")}),
        "cache": cache_fingerprint(exp["cache_id"], videos),
        "sequences": sha256_obj(list(videos)),
        "code": code_fingerprint(),
        "evaluator": evaluator_fingerprint(dataset_cls),
    }


def occl_fingerprint(exp: dict, cond: str, dirs: list[str], videos: list[str], occ: dict) -> dict:
    """Dau van tay cho buoc occlusion_eval cua mot dieu kien (phu thuoc MOI tracker tham gia)."""
    split = exp["split_name"]
    tr = {d: tracking_fingerprint(config.PROCESSED_DIR / "trackers" / split / d, videos)["sha256"] for d in dirs}
    return {"cond": cond, "split_name": split, "trackers": list(dirs), "tracking": sha256_obj(tr),
            "sequences": sha256_obj(list(videos)), "settings": sha256_obj(occ), "code": code_fingerprint()}


def compare(old: dict | None, new: dict) -> list[str]:
    """Danh sach truong LECH giua dau van tay cu va moi ('' = khong co ban cu)."""
    if not old:
        return ["(chua co dau van tay)"]
    diff = []
    for k, v in new.items():
        if old.get(k) != v:
            diff.append(k)
    return diff


def load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def save(path: Path, fp: dict, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = dict(fp)
    if extra:
        d.update(extra)
    path.write_text(json.dumps(d, indent=2, default=str), encoding="utf-8")
