#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 37 -- so khop cau hinh pilot NGAY TRUOC khi chay voi file KHOA da commit (01_khoa_cau_hinh_pilot_<model>.json).

    python handoff/logs/review37/so_khop_khoa.py <file khoa .json> <log kiem-pilot moi>
Lay JSON trong log kiem-pilot (tu '{' dau toi '}' cuoi), so cac muc quyet dinh luot chay: tham so hieu luc + resolve,
khac biet voi smoke, cau hinh (sha256 + ke thua), checkpoint khoi tao, dataset (manifest, data.yaml, split, so anh),
warmup du doan theo ma, sha256 trainer.py ultralytics, phien ban thu vien, lenh, run_dir.
Bo qua muc thay doi tu nhien (tao_luc, VRAM dang dung truoc khi chay, thoi diem / thoi gian kiem hop dong).
Exit 0 khop; 1 khac (khong duoc chay pilot); 2 loi dau vao.
"""
import json
import sys
from pathlib import Path

MUC = ["loai", "che_do", "ten", "model", "cau_hinh", "pilot_khac_smoke", "checkpoint_khoi_tao", "tham_so_khai_bao",
       "batch_ghi_de", "workers_ghi_de", "tham_so_resolve_tinh", "suy_ra", "run_dir", "warmup_du_doan_theo_ma",
       "ultralytics_trainer_py_sha256", "lenh_pilot"]
MUC_DATASET = ["hop_dong", "phien_ban_manifest", "loi", "ten", "tap_con", "data_yaml", "sha256_data_yaml",
               "sha256_tom_tat", "manifest_sha256", "sha256_video_json", "split_da_khoa", "duong_dan_resolve", "so_anh",
               "so_video", "so_nhan_rong_hop_le", "so_anh_co_file_ignore", "cache_anh", "noi_dung_da_xac_minh",
               "so_file_da_bam"]
MUC_MOI_TRUONG = ["python", "os", "ultralytics", "torch", "cuda", "cudnn", "numpy", "opencv"]


def doc(p: Path) -> dict:
    s = p.read_text(encoding="utf-8")
    return json.loads(s[s.index("{"): s.rindex("}") + 1])


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    try:
        a, b = doc(Path(sys.argv[1])), doc(Path(sys.argv[2]))
    except (OSError, ValueError) as e:
        print(f"[LOI DAU VAO] {e}")
        return 2
    khac = [k for k in MUC if a.get(k) != b.get(k)]
    khac += [f"dataset.{k}" for k in MUC_DATASET if a.get("dataset", {}).get(k) != b.get("dataset", {}).get(k)]
    khac += [f"moi_truong.{k}" for k in MUC_MOI_TRUONG if a.get("moi_truong", {}).get(k) != b.get("moi_truong", {}).get(k)]
    ga, gb = a.get("moi_truong", {}).get("gpu", {}), b.get("moi_truong", {}).get("gpu", {})
    khac += [f"moi_truong.gpu.{k}" for k in ("ten", "vram_mib", "driver") if ga.get(k) != gb.get(k)]
    so = len(MUC) + len(MUC_DATASET) + len(MUC_MOI_TRUONG) + 3
    print(f"so khop khoa {sys.argv[1]} vs {sys.argv[2]}: {so - len(khac)}/{so} muc khop")
    if khac:
        print("KHAC:", khac)
        print("KET LUAN: KHAC KHOA -- KHONG chay pilot")
        return 1
    print("KET LUAN: KHOP KHOA")
    return 0


if __name__ == "__main__":
    sys.exit(main())
