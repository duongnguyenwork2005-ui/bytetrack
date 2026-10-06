# -*- coding: utf-8 -*-
"""Vong 44: kiem ket qua BDD vong 43, checkpoint va file rieng cua nguoi dung KHONG bi thay doi (CHI DOC).
    python handoff/logs/review44/kiem_khong_doi.py --moc "2026-10-07 01:19:00"
1. Moi file lon da khai bao (logs/review43/ket_qua/tep_lon_khong_ban_giao.csv: 3 NPZ du doan, 3 log, 3 CSV RAM):
   bytes + sha256 hien tai == gia tri da ghi.
2. ket_qua.json / theo_lop.csv / tom_tat.json / trang_thai.json trong results/ == ban da commit trong
   logs/review43/ket_qua/<ten>/ (blob HEAD, tung byte).
3. Ba checkpoint: sha256 == ke hoach khoa vong 43 muc 2.
4. File rieng (.gitignore, src/demo/**, live.bat, outputs/**, CODEX_REVIEW_c4d64a1_EVIDENCE/): khong co file nao sua sau
   --moc (dau vong 44); khong co gi trong index (khong stage)."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CHECKPOINT = {"models/yolov8n.pt": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
              "models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/weights/best.pt":
                  "5f66776d3202389ca484f3d9e2f107dddb354d21a9f1762d19401d690a4a0d8a",
              "models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/weights/best.pt":
                  "d7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923"}
RIENG = [".gitignore", "src/demo", "live.bat", "outputs", "CODEX_REVIEW_c4d64a1_EVIDENCE"]


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for k in iter(lambda: f.read(1 << 20), b""):
            h.update(k)
    return h.hexdigest()


def git(*a: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--moc", required=True, help="thoi diem dau vong 44 (gio may)")
    a = ap.parse_args()
    moc = dt.datetime.strptime(a.moc, "%Y-%m-%d %H:%M:%S").timestamp()
    loi = 0
    with open(REPO / "handoff/logs/review43/ket_qua/tep_lon_khong_ban_giao.csv", newline="", encoding="utf-8") as f:
        hang = list(csv.DictReader(f))
    for r in hang:
        p = REPO / r["duong_dan"]
        ok = p.is_file() and p.stat().st_size == int(r["bytes"]) and sha_file(p) == r["sha256"]
        loi += not ok
        print(f"[{'DAT' if ok else 'KHONG DAT'}] file lon vong 43 {r['duong_dan']} ({r['bytes']} B, {r['sha256'][:12]})")
    for ten in ("yolov8n_coco", "yolov8n_ft", "yolo11n_ft"):
        for goc, ban in ((f"results/phase2/detector_eval/bdd_val_cocovid/r43_{ten}/ket_qua.json", "ket_qua.json"),
                         (f"results/phase2/detector_eval/bdd_val_cocovid/r43_{ten}/theo_lop.csv", "theo_lop.csv"),
                         (f"results/phase2/detector_eval/giam_sat/r43_{ten}/tom_tat.json", "giam_sat/tom_tat.json"),
                         (f"results/phase2/detector_eval/giam_sat/r43_{ten}/trang_thai.json", "giam_sat/trang_thai.json")):
            b = git("show", f"HEAD:handoff/logs/review43/ket_qua/{ten}/{ban}")
            ok = (REPO / goc).read_bytes() == b
            loi += not ok
            print(f"[{'DAT' if ok else 'KHONG DAT'}] {goc} == ban commit logs/review43/ket_qua/{ten}/{ban} "
                  f"(sha256 {hashlib.sha256(b).hexdigest()[:12]})")
    for p, s in CHECKPOINT.items():
        ok = sha_file(REPO / p) == s
        loi += not ok
        print(f"[{'DAT' if ok else 'KHONG DAT'}] checkpoint {p} sha256 {s[:12]}")
    sua_sau_moc, dem = [], 0
    for r in RIENG:
        p = REPO / r
        ds = [p] if p.is_file() else [x for x in p.rglob("*") if x.is_file()] if p.is_dir() else []
        dem += len(ds)
        sua_sau_moc += [str(x.relative_to(REPO)) for x in ds if x.stat().st_mtime >= moc]
    staged = git("diff", "--cached", "--name-only").decode().split()
    staged_rieng = [x for x in staged if x == ".gitignore" or x.startswith(("src/demo/", "outputs/")) or x == "live.bat"]
    ok = not sua_sau_moc and not staged_rieng
    loi += not ok
    print(f"[{'DAT' if ok else 'KHONG DAT'}] file rieng: {dem} file trong {RIENG}; sua sau {a.moc}: {sua_sau_moc[:5]}; "
          f"dang stage: {staged_rieng}")
    print(f"KET LUAN: {'DAT' if loi == 0 else f'{loi} muc KHONG DAT'}")
    return 0 if loi == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
