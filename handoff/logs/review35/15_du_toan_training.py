#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 35 -- DU TOAN training local (thoi gian, RAM/VRAM, dia) tu SO DO smoke tren du lieu that + so anh dataset day du.

Chay tu goc repo:  python handoff/logs/review35/15_du_toan_training.py --ra handoff/logs/review35/15_du_toan_training.json
Chi doc: 12_ket_qua_runner_<model>_lan2.json, 12_smoke_<model>_lan2.log (moc gio, dong "Speed:"),
12_ram_tien_trinh_<model>_lan2.json, 14_dem_anh_day_du.json. Moi gia dinh ghi trong khoa "gia_dinh". KHONG train.
Day la UOC TINH; so epoch / buoc frame / ngan sach CHUA duoc duyet -- bang phuong an chi de nguoi dung quyet dinh.
Moi muc ("trung tam" va "cao") la KICH BAN DU TOAN theo cac gia dinh trong GIA_DINH, KHONG phai can tren hay bao dam:
thoi gian, RAM va VRAM thuc te co the vuot ca kich ban "cao" (dinh chinh vong 36, R35-F2; cong thuc va so lieu khong doi).

Mo hinh thoi gian (moi model, chay tuan tu, cung cau hinh train_bdd5.yaml: batch 8, imgsz 640, 2 worker, amp tat):
  tong(E) = khoi_tao + cho_batch_dau + E * (train_epoch + val_epoch) + val_cuoi
  train_epoch = ceil(n_train / 8) * s_moi_batch (trung vi on dinh do o smoke, bo 5 batch dau)
  val_epoch   = val_s_smoke (gom phan co dinh: khoi dong worker val, luu checkpoint, don dep) + (n_dev - 84) * ms_anh_val
  kich ban "cao" (cham hon) = kich ban "trung tam" * (1 + bien_cham), gia dinh cho rui ro chua do: GPU nong/giam xung khi
  chay nhieu gio, phan trang RAM. Day la mot kich ban, khong bao trum moi truong hop xau.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

L = Path(__file__).resolve().parent
MODEL = ("yolov8n", "yolo11n")
GIA_DINH = {
    "batch": 8,
    "bien_cham_cao": 0.30,
    "ms_hau_xu_ly_them_khi_model_da_hoc": 5.0,
    "toc_do_bam_hop_dong_MB_s": 42.0,
    "toc_do_quet_nhan_ultralytics_anh_s": 1100.0,
    "ti_le_byte_anh_to_xam_xuat_tren_nguon": 1.551,
    "byte_nhan_ignore_manifest_moi_anh": 4096 * 1.4 + 600,
    "dung_luong_luot_train_moi_model_GB": 0.05,
    "ram_them_cho_dataset_day_du_MiB": (100, 500),
    "vram_them_cho_lo_nhieu_box_MiB": (0, 300),
    "ghi_chu": {
        "bien_cham_cao": "smoke chi do 36 khoang batch (~1 phut); chua do duoc giam xung GPU/phan trang RAM khi chay nhieu gio",
        "ms_hau_xu_ly_them_khi_model_da_hoc": "model smoke gan nhu khong co box tren nguong 0,001 nen NMS rat nhanh; model da "
                                              "hoc co the co nhieu box/anh hon (max_det 300) -> kich ban cham hon GIA DINH "
                                              "cong them 5 ms/anh; chua do, thuc te co the cham hon nua",
        "toc_do_bam_hop_dong_MB_s": "do o smoke: hop dong bam 418 anh + 418 nhan + 152 ignore (~55 MB) trong 1,31 s",
        "toc_do_quet_nhan_ultralytics_anh_s": "do o smoke: 334 nhan quet o ~1,1K it/s khi tao labels/train.cache",
        "ti_le_byte_anh_to_xam_xuat_tren_nguon": "do tren dataset smoke: 152 anh to xam ghi lai JPEG q95 = 1,551 x byte nguon; "
                                                  "anh khong to xam chep nguyen byte (1,000)",
        "byte_nhan_ignore_manifest_moi_anh": "NTFS cap phat >= 4 KiB moi file nho: 1 file nhan + ~0,4 file ignore moi anh, "
                                              "cong ~600 B manifest/cache -> gia dinh du, chua do tren dataset day du",
        "dung_luong_luot_train_moi_model_GB": "last.pt + best.pt (5-6 MB sau khi bo optimizer; ~2-3 lan khi con optimizer) + "
                                                 "results.csv + args.yaml; save_period = -1 (khong luu checkpoint tung epoch)",
        "ram_them_cho_dataset_day_du_MiB": "nhan cua ~56K anh nam trong dataset cua tien trinh chinh + moi worker (spawn tren "
                                            "Windows -> moi worker mot ban); UOC, chua do",
        "vram_them_cho_lo_nhieu_box_MiB": "bo gan nhan TAL cap phat theo so box lon nhat trong lo; lo nhieu box nhat quan sat "
                                           "o smoke ~180 box; UOC, chua do tren du lieu day du, lo thuc te co the nhieu box "
                                           "hon va VRAM co the vuot kich ban"}}


def doc_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def moc(log, mau):
    for s in Path(log).read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^\[\+\s*([0-9.]+)s\] (.*)$", s)
        if m and mau in m.group(2):
            return float(m.group(1)), m.group(2)
    return None, None


def gio(s):
    return round(s / 3600, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ra", required=True)
    a = ap.parse_args()
    g = GIA_DINH
    dem = doc_json(L / "14_dem_anh_day_du.json")
    kq = {"loai": "UOC TINH tu smoke 1 epoch tren du lieu that; KHONG phai cam ket ngan sach; epoch/buoc frame CHUA duyet; "
                  "moi muc (ke ca 'cao' va khoang RAM/VRAM) la KICH BAN DU TOAN theo gia_dinh, KHONG phai can tren hay "
                  "bao dam -- thoi gian va bo nho thuc te co the vuot",
          "gia_dinh": g, "do_o_smoke": {}, "dataset_day_du": {}, "thoi_gian_theo_phuong_an": [], "ram_vram": {}, "dia": {}}
    for b, r in dem["theo_buoc"].items():
        kq["dataset_day_du"][f"buoc_{b}"] = {"anh_train": r["train"]["so_anh"], "anh_dev": r["dev"]["so_anh"],
                                             "byte_nguon": r["train"]["byte_anh_nguon"] + r["dev"]["byte_anh_nguon"],
                                             "anh_to_xam_xap_xi": r["train"]["anh_to_xam_xap_xi"] + r["dev"]["anh_to_xam_xap_xi"]}
    for m in MODEL:
        r = doc_json(L / f"12_ket_qua_runner_{m}_lan2.json")
        e = r["epoch"][0]
        log = L / f"12_smoke_{m}_lan2.log"
        t_start, _ = moc(log, "Starting training")
        _, speed = moc(log, "Speed:")
        ms = [float(x) for x in re.findall(r"([0-9.]+)ms", speed)]          # preprocess, inference, loss, postprocess
        ram = doc_json(L / f"12_ram_tien_trinh_{m}_lan2.json")
        s_batch = e["on_dinh"]["s_moi_batch_trung_vi"]
        kq["do_o_smoke"][m] = {
            "khoi_tao_s": t_start, "train_s": e["train_s"], "so_batch": e["so_batch"], "s_moi_batch_trung_vi": s_batch,
            "anh_moi_s_trung_vi": e["on_dinh"]["anh_moi_s_trung_vi"],
            "cho_batch_dau_s": round(e["train_s"] - e["so_batch"] * s_batch, 1), "val_s": e["val_s"],
            "ms_anh_val_speed": {"preprocess": ms[0], "inference": ms[1], "postprocess": ms[3]},
            "model_train_s": r["thoi_gian_s"], "torch_max_allocated_mib": e["torch_max_allocated_mib"],
            "torch_max_reserved_mib": e["torch_max_reserved_mib"], "nvidia_smi_max_ca_gpu_mib": r["vram_nvidia_smi"]["max_mib"],
            "rss_tien_trinh_va_con_max_mib": ram["rss_tien_trinh_va_con_max_mib"],
            "ram_he_thong_con_trong_min_mib": min(ram["ram_he_thong_con_trong_min_mib"],
                                                  r["vram_nvidia_smi"]["ram_con_trong_min_mib"])}
    for buoc in ("5", "10"):
        d = kq["dataset_day_du"][f"buoc_{buoc}"]
        n_tr, n_dv = d["anh_train"], d["anh_dev"]
        byte_xuat = d["byte_nguon"] * (1 + (g["ti_le_byte_anh_to_xam_xuat_tren_nguon"] - 1) *
                                       d["anh_to_xam_xap_xi"] / (n_tr + n_dv)) + (n_tr + n_dv) * g["byte_nhan_ignore_manifest_moi_anh"]
        hop_dong_s = byte_xuat / 1e6 / g["toc_do_bam_hop_dong_MB_s"]
        quet_s = (n_tr + n_dv) / g["toc_do_quet_nhan_ultralytics_anh_s"]
        for E in (15, 30):
            pa = {"buoc_frame": int(buoc), "epoch": E, "anh_train": n_tr, "anh_dev": n_dv, "theo_model": {}}
            tong_tt = tong_cao = 0.0
            for m in MODEL:
                s = kq["do_o_smoke"][m]
                nb = math.ceil(n_tr / g["batch"])
                train_ep = nb * s["s_moi_batch_trung_vi"]
                ms_val = s["ms_anh_val_speed"]["preprocess"] + s["ms_anh_val_speed"]["inference"] + \
                    s["ms_anh_val_speed"]["postprocess"]
                val_ep = s["val_s"] + (n_dv - 84) * ms_val / 1000
                val_ep_cao = s["val_s"] + (n_dv - 84) * (ms_val + g["ms_hau_xu_ly_them_khi_model_da_hoc"]) / 1000
                khoi_tao = s["khoi_tao_s"] + hop_dong_s + quet_s
                tt = khoi_tao + s["cho_batch_dau_s"] + E * (train_ep + val_ep) + val_ep
                cao = (khoi_tao + s["cho_batch_dau_s"] + E * (train_ep + val_ep_cao) + val_ep_cao) * (1 + g["bien_cham_cao"])
                pa["theo_model"][m] = {"batch_moi_epoch": nb, "train_moi_epoch_h": gio(train_ep),
                                       "val_moi_epoch_phut": round(val_ep / 60, 1),
                                       "val_moi_epoch_phut_cao": round(val_ep_cao / 60, 1),
                                       "khoi_tao_phut": round(khoi_tao / 60, 1), "tong_h_trung_tam": gio(tt),
                                       "tong_h_cao": gio(cao)}
                tong_tt += tt
                tong_cao += cao
            pa["tong_hai_model_h_trung_tam"] = gio(tong_tt)
            pa["tong_hai_model_h_cao"] = gio(tong_cao)
            pa["tong_hai_model_ngay_trung_tam"] = round(tong_tt / 86400, 2)
            kq["thoi_gian_theo_phuong_an"].append(pa)
        kq["dia"][f"buoc_{buoc}"] = {"dataset_xuat_GB_uoc": round(byte_xuat / 1e9, 2),
                                    "hai_luot_train_GB_uoc": 2 * g["dung_luong_luot_train_moi_model_GB"]}
    for m in MODEL:
        s = kq["do_o_smoke"][m]
        kq["ram_vram"][m] = {
            "ram_rss_tien_trinh_va_con_mib_do": s["rss_tien_trinh_va_con_max_mib"],
            "ram_rss_uoc_day_du_mib": [s["rss_tien_trinh_va_con_max_mib"] + x for x in g["ram_them_cho_dataset_day_du_MiB"]],
            "ram_he_thong_con_trong_min_mib_do": s["ram_he_thong_con_trong_min_mib"],
            "torch_reserved_mib_do": s["torch_max_reserved_mib"],
            "torch_reserved_uoc_day_du_mib": [s["torch_max_reserved_mib"] + x for x in g["vram_them_cho_lo_nhieu_box_MiB"]],
            "nvidia_smi_ca_gpu_mib_do": s["nvidia_smi_max_ca_gpu_mib"],
            "nvidia_smi_ca_gpu_uoc_day_du_mib": [s["nvidia_smi_max_ca_gpu_mib"] + x for x in g["vram_them_cho_lo_nhieu_box_MiB"]],
            "vram_card_mib": 4096}
    Path(a.ra).write_text(json.dumps(kq, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    for pa in kq["thoi_gian_theo_phuong_an"]:
        mm = pa["theo_model"]
        print(f"buoc {pa['buoc_frame']:2d}, {pa['epoch']:2d} epoch ({pa['anh_train']} anh train): " + " | ".join(
            f"{m} {v['train_moi_epoch_h']} h/epoch train + {v['val_moi_epoch_phut']}-{v['val_moi_epoch_phut_cao']} ph val"
            f" -> {v['tong_h_trung_tam']} h (kich ban cham hon {v['tong_h_cao']} h)" for m, v in mm.items())
            + f" || tong {pa['tong_hai_model_h_trung_tam']} h (kich ban cham hon {pa['tong_hai_model_h_cao']} h)", flush=True)
    for b, v in kq["dia"].items():
        print(f"dia {b}: dataset ~{v['dataset_xuat_GB_uoc']} GB + 2 luot train ~{v['hai_luot_train_GB_uoc']} GB")
    for m, v in kq["ram_vram"].items():
        print(f"{m} (khoang uoc = kich ban theo gia dinh, thuc te co the vuot): "
              f"RAM RSS do {v['ram_rss_tien_trinh_va_con_mib_do']} MiB -> uoc {v['ram_rss_uoc_day_du_mib']} MiB; RAM he thong "
              f"con trong min {v['ram_he_thong_con_trong_min_mib_do']} MiB; torch reserved {v['torch_reserved_mib_do']} -> "
              f"{v['torch_reserved_uoc_day_du_mib']} MiB; ca GPU {v['nvidia_smi_ca_gpu_mib_do']} -> "
              f"{v['nvidia_smi_ca_gpu_uoc_day_du_mib']} MiB / 4096")
    print("# moi so tren la KICH BAN DU TOAN theo gia dinh (xem khoa gia_dinh), KHONG phai can tren hay bao dam")
    print(f"[ra] {a.ra}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
