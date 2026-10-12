"""
kiem_cache_dev.py -- R48C (giao thuc muc 2): xac thuc cache detection co dinh cho so sanh bo loc.

Tinh `run_fingerprint.cache_fingerprint(cache_id, 12 video dev)` (cung ham runner dung khi ghi eval_meta.json) va so
voi dau van tay DA COMMIT cua run moc 2 dev `p2-expB_detrac_dev_fix1-A-bt-ekf`. Chi doc cache, chi doc file tham chieu;
ghi results/phase2/kc48_pf_code/cache_dev.json. Exit 0 khi trung khit, 1 khi lech.

    python src/kc48/kiem_cache_dev.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import run_fingerprint as fp  # noqa: E402

CACHE_ID = "yolov8n-coco_conf0.05_all"
MANIFEST = ROOT / "results/phase2/manifests/detrac_dev.json"
THAM_CHIEU = "results/trackeval/DETRAC-all/p2-expB_detrac_dev_fix1-A-bt-ekf/eval_meta.json"
KHOA = ("cache_id", "meta_sha256", "npz_sha256", "weights_sha256", "conf", "classes", "max_det")
OUT = ROOT / "results/phase2/kc48_pf_code/cache_dev.json"


def main() -> int:
    videos = json.loads(MANIFEST.read_text(encoding="utf-8"))["videos"]
    now = fp.cache_fingerprint(CACHE_ID, videos)
    ref_disk = json.loads((ROOT / THAM_CHIEU).read_text(encoding="utf-8"))["cache"]
    # ban da commit (khong phu thuoc file tren dia co bi sua sau commit hay khong)
    blob = subprocess.run(["git", "show", f"HEAD:{THAM_CHIEU}"], cwd=ROOT, capture_output=True, text=True)
    ref_git = json.loads(blob.stdout)["cache"] if blob.returncode == 0 else None
    so = {k: {"hien_tai": now.get(k), "tham_chieu_dia": ref_disk.get(k),
              "tham_chieu_HEAD": None if ref_git is None else ref_git.get(k)} for k in KHOA}
    ok = all(v["hien_tai"] == v["tham_chieu_dia"] == v["tham_chieu_HEAD"] for v in so.values())
    out = {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cache_id": CACHE_ID, "videos": videos,
           "tham_chieu": THAM_CHIEU, "so_sanh": so, "trung_khit": ok,
           "ghi_chu": "cache_fingerprint: sha256 meta + sha256 cua danh sach sha256[:16] tung npz (src/run_fingerprint.py)"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in so.items():
        print(f"  {k:<15} {'OK ' if v['hien_tai'] == v['tham_chieu_dia'] == v['tham_chieu_HEAD'] else 'LECH'} {v['hien_tai']}")
    print(f"-> {OUT.relative_to(ROOT).as_posix()}  trung_khit={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
