# -*- coding: utf-8 -*-
"""Vong 42: dong goi BANG CHUNG lot training dai vao handoff (CHI DOC nguon, chi ghi vao --out).

    python handoff/logs/review42/dong_goi_ket_qua.py --out handoff/logs/review42/ket_qua [--model yolov8n yolo11n]

Moi model <M>:
  <out>/<M>/runner/       ket qua runner (train_bdd5_v1_<M>_day_du*.json) -- nho
  <out>/<M>/luot/         nhat_ky_luot.jsonl, results.csv, args.yaml cua thu muc luot -- nho
  <out>/<M>/giam_sat/<thu muc>/  tom_tat.json, trang_thai.json, log_loc.txt
        log_loc.txt = log.txt cua bo giam sat BO cac dong tien do tung batch (giu dong 100% cuoi moi pha train / val
        va moi dong khac: canh bao, bang val, [DAT]/[BI NGAT], # exit=...)
  <out>/tep_lon_khong_ban_giao.csv   duong dan, bytes, sha256 cua file LON giu tren may (khong dong goi): log.txt day du,
        ram_mau_tho.csv, theo_batch.csv, CSV nvidia-smi, weights/last.pt + best.pt, khoi_phuc/**/*.pt
Khong chep checkpoint / weights / dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path

GOC = Path(os.environ.get("GOC_REPO") or Path(__file__).resolve().parents[3])


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(GOC)).replace("\\", "/")
    except ValueError:
        return str(p)


def loc_log(src: Path, dst: Path) -> tuple[int, int]:
    giu = bo = 0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        for dong in fi:
            s = dong.decode("utf-8", "replace")
            tien_do = ("it/s" in s or "s/it" in s) and "100%" not in s
            if tien_do:
                bo += 1
                continue
            fo.write(dong)
            giu += 1
    return giu, bo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", nargs="+", default=["yolov8n", "yolo11n"])
    a = ap.parse_args()
    out = Path(a.out)
    if not out.is_absolute():
        out = GOC / out
    lon = []
    for m in a.model:
        ten = f"train_bdd5_v1_{m}_day_du"
        R = GOC / "models/finetune_phase2" / ten
        kq_dir = GOC / "results/phase2/detector_train"
        (out / m / "runner").mkdir(parents=True, exist_ok=True)
        (out / m / "luot").mkdir(parents=True, exist_ok=True)
        for f in sorted(kq_dir.glob(f"{ten}*.json")):
            shutil.copy2(f, out / m / "runner" / f.name)
        for f in sorted(kq_dir.glob(f"{ten}*_tai_nguyen_nvidia_smi.csv")):
            lon.append(f)
        for n in ("nhat_ky_luot.jsonl", "results.csv", "args.yaml"):
            if (R / n).is_file():
                shutil.copy2(R / n, out / m / "luot" / n)
        lon.append(R / "theo_batch.csv")
        for n in ("last", "best"):
            if (R / "weights" / f"{n}.pt").is_file():
                lon.append(R / "weights" / f"{n}.pt")
        lon += sorted((R / "khoi_phuc").rglob("*.pt"))
        for gs in sorted((kq_dir / "giam_sat").glob(f"{m}_*")):
            d = out / m / "giam_sat" / gs.name
            d.mkdir(parents=True, exist_ok=True)
            for n in ("tom_tat.json", "trang_thai.json"):
                if (gs / n).is_file():
                    shutil.copy2(gs / n, d / n)
            if (gs / "log.txt").is_file():
                giu, bo = loc_log(gs / "log.txt", d / "log_loc.txt")
                print(f"{rel(gs)}/log.txt -> log_loc.txt: giu {giu} dong, bo {bo} dong tien do tung batch")
                lon.append(gs / "log.txt")
            if (gs / "ram_mau_tho.csv").is_file():
                lon.append(gs / "ram_mau_tho.csv")
    with open(out / "tep_lon_khong_ban_giao.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["duong_dan", "bytes", "sha256", "ghi_chu"])
        for p in lon:
            ghi_chu = ("checkpoint (khong commit / khong dong goi)" if p.suffix == ".pt" else
                       "log / CSV lon giu tren may (ban loc hoac tom tat nam trong goi)")
            w.writerow([rel(p), p.stat().st_size, sha(p), ghi_chu])
    print(f"{len(lon)} file lon -> {rel(out / 'tep_lon_khong_ban_giao.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
