"""Chay tuan tu 3 luot danh gia UA-DETRAC test theo ke hoach khoa
handoff/logs/review47/03_ke_hoach_khoa_danh_gia_detrac_test.md (muc 5). Mot model mot luot, thu muc moi, khong ghi de.
Luot loi khong chan luot sau; trang thai ghi o trang_thai.json canh file nay.

    <python env khoaluan-mot> -X utf8 chay_3_model.py <sha256 ke hoach khoa>
"""
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(r"D:\LABOTARY\bytetrack")
PY = r"C:\Users\Mikuno\anaconda3\envs\khoaluan-mot\python.exe"
LOG = Path(__file__).resolve().parent
MODELS = [
    ("yolov8n_coco", "models/yolov8n.pt", "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"),
    ("yolov8n_ft", "models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/weights/best.pt",
     "5f66776d3202389ca484f3d9e2f107dddb354d21a9f1762d19401d690a4a0d8a"),
    ("yolo11n_ft", "models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/weights/best.pt",
     "d7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923"),
]


def now():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


trang_thai = {"ke_hoach": "handoff/logs/review47/03_ke_hoach_khoa_danh_gia_detrac_test.md",
              "ke_hoach_sha256": sys.argv[1] if len(sys.argv) > 1 else None, "bat_dau": now(), "pid": os.getpid(), "luot": {}}


def ghi():
    (LOG / "trang_thai.json").write_text(json.dumps(trang_thai, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


for ten, w, sha in MODELS:
    out = f"results/phase2/detector_eval/detrac_test/20261010_{ten}"
    cmd = [PY, "-u", "src/danh_gia_detector_detrac_test.py", "--model", f"{ten}={w}", "--sha256", f"{ten}={sha}", "--out", out]
    trang_thai["luot"][ten] = {"lenh": cmd, "out": out, "bat_dau": now(), "trang_thai": "dang_chay"}
    ghi()
    with open(LOG / f"{ten}.log", "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUTF8="1"))
    trang_thai["luot"][ten].update(ket_thuc=now(), exit=p.returncode, trang_thai="xong" if p.returncode == 0 else "loi")
    ghi()
trang_thai["ket_thuc"] = now()
ghi()
