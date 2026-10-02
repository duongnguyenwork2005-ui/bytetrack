#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 37 -- TOM TAT pilot (CHI DOC ket qua da ghi; khong nap model, khong train, khong inference).

    python handoff/logs/review37/07_tom_tat_pilot.py --ra handoff/logs/review37/07_tom_tat_pilot.json
Doc results/phase2/detector_pilot/train_bdd5_v1_<model>_pilot{.json,_batch.csv,_tai_nguyen_nvidia_smi.csv} va
logs/review37/06_ram_ngoai_<model>{.json,_mau_tho.csv}. Ghi:
  - bang do theo diem (truoc hoc / sau epoch 1 / sau epoch 2: model hien tai va EMA / checkpoint last, best) -- ten nguon ro;
  - warmup + LR + accumulate + so lan cap nhat THUC CHAY, doi chieu cong thuc trong BaseTrainer._do_train (cung phep doi
    chieu cua test P4e);
  - thoi gian: khoi tao, train, val, luu, do chan doan, final_eval (tu moc callback);
  - RAM/VRAM: dinh va trung vi tinh lai TU MAU THO (khong dung so tong hop).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

GOC = Path(__file__).resolve().parents[3]
KQ = GOC / "results/phase2/detector_pilot"
L = Path(__file__).resolve().parent
NGUON = [("truoc_hoc_model", "truoc hoc -- model trainer da dung (= EMA)"),
         ("sau_epoch1_model", "sau epoch 1 -- model hien tai"), ("sau_epoch1_ema", "sau epoch 1 -- EMA"),
         ("sau_epoch2_model", "sau epoch 2 -- model hien tai"), ("sau_epoch2_ema", "sau epoch 2 -- EMA"),
         ("checkpoint_last", "checkpoint last.pt (EMA FP16 nap tu dia)"),
         ("checkpoint_best", "checkpoint best.pt (EMA FP16 nap tu dia)")]


def doi_chieu_warmup(rows: list[dict], w: dict) -> dict:
    a = w["args"]
    E, nw = int(a["epochs"]), int(w["so_buoc_warmup_nw"])
    lf = lambda x: max(1 - x / E, 0) * (1.0 - a["lrf"]) + a["lrf"]  # noqa: E731
    loi, cap, last, acc = [], [], -1, None
    for r in rows:
        ni, ep = int(r["ni"]), int(r["epoch"]) - 1
        if ni < nw:
            acc = max(1, int(np.interp(ni, [0, nw], [1, a["nbs"] / a["batch"]]).round()))
            for g in ("weight", "bn", "bias"):
                lr = float(np.interp(ni, [0, nw], [a["warmup_bias_lr"] if g == "bias" else 0.0, a["lr0"] * lf(ep)]))
                mo = float(np.interp(ni, [0, nw], [a["warmup_momentum"], a["momentum"]]))
                if abs(float(r[f"lr_{g}"]) - lr) > 1e-12 or abs(float(r[f"momentum_{g}"]) - mo) > 1e-12:
                    loi.append(ni)
        else:
            for g in ("weight", "bn", "bias"):
                if abs(float(r[f"lr_{g}"]) - a["lr0"] * lf(ep)) > 1e-12:
                    loi.append(ni)
        acc = acc if acc is not None else max(round(a["nbs"] / a["batch"]), 1)
        if int(r["accumulate"]) != acc:
            loi.append(ni)
        if ni - last >= acc:
            cap.append(ni)
            last = ni
    return {"so_batch_lech_cong_thuc": len(set(loi)), "ni_cap_nhat_mo_phong": cap}


def thong_ke(x):
    x = [v for v in x if v is not None]
    if not x:
        return None
    s = sorted(x)
    return {"n": len(s), "min": s[0], "trung_vi": s[len(s) // 2], "max": s[-1]}


def tom_tat_model(m: str) -> dict:
    ten = f"train_bdd5_v1_{m}_pilot"
    rec = json.loads((KQ / f"{ten}.json").read_text(encoding="utf-8"))
    p = rec["pilot"]
    rows = list(csv.DictReader(open(KQ / f"{ten}_batch.csv", encoding="utf-8")))
    bc = p["truoc_hoc"]["bang_chung_khoi_tao"]
    out = {"ten": ten, "loi_runner": rec.get("loi"), "loi_pilot": p.get("loi"), "tom_tat_dat": p.get("tom_tat"),
           "checkpoint_khoi_tao": rec["checkpoint_khoi_tao"]["sha256"],
           "dau_vao_da_nap": rec.get("dau_vao_da_nap"),
           "truoc_hoc": {"ten_lop": bc["ten_lop_model"], "nap_trong_so": {k: bc["nap_trong_so"][k] for k in (
               "tong_tensor_model", "chep_nguyen", "remap_hang_theo_ten", "khoi_tao_moi", "cung_shape_khac_gia_tri",
               "tensor_remap", "chi_so_lop_checkpoint_theo_ten")},
               "tensor_khoi_tao_moi": [x["ten"] for x in bc["nap_trong_so"]["tensor_khoi_tao_moi"]],
               "optimizer": {k: bc["optimizer"][k] for k in ("so_trang_thai", "ema_updates", "accumulate",
                                                             "scheduler_last_epoch")},
               "so_tham_so_co_gradient": bc["so_tham_so_co_gradient"], "model_va_ema_giong_het":
                   bc["model_va_ema_giong_het"], "model_che_do_train": bc["model_che_do_train"]},
           "log_ultralytics": [x["dong"] for x in p.get("log_ultralytics", [])], "do": {}}
    for k, mo_ta in NGUON:
        d = p.get("do", {}).get(k)
        if not d:
            out["do"][k] = None
            continue
        dc = d["danh_gia_chinh"]
        out["do"][k] = {
            "nguon": mo_ta, "file_sha256": d["sha256"], "so_detection_conf_ge_0_001": d["detection"]["tong"],
            "detection_theo_lop": d["detection"]["theo_lop"], "so_anh_co_detection": d["detection"]["so_anh_co_detection"],
            "confidence_detection": d["detection"]["confidence"], "histogram": d["detection"]["histogram"],
            "AP50_theo_lop": {c: v["AP50"] for c, v in dc["theo_lop"].items()},
            "AP50_95_theo_lop": {c: v["AP50_95"] for c, v in dc["theo_lop"].items()},
            "mAP50_3_lop_co_gt": dc["macro"]["mAP50"], "mAP50_95_3_lop_co_gt": dc["macro"]["mAP50_95"],
            "diem_truoc_nguong_max_moi_anh": d["diem_truoc_nguong"]["max_moi_anh"],
            "diem_truoc_nguong_theo_lop_trung_vi": d["diem_truoc_nguong"]["theo_lop_trung_vi_max_moi_anh"],
            "so_anh_diem_max_ge_0_001": d["diem_truoc_nguong"]["so_anh_max_ge_0_001"],
            "bat_bien": d.get("bat_bien", {}).get("khong_doi_sau_khoi_phuc"),
            "thay_doi_truoc_khoi_phuc": d.get("bat_bien", {}).get("thay_doi_truoc_khi_khoi_phuc_rng_co"),
            "thoi_gian_s": d.get("thoi_gian_tong_s")}
    w = p["warmup_theo_ma"]
    dc = doi_chieu_warmup(rows, w)
    cap_log = [ni for e in p["epoch"] for ni in e["ni_co_cap_nhat"]]
    out["warmup_va_cap_nhat"] = {
        "nw_theo_ma": w["so_buoc_warmup_nw"], "so_batch_moi_epoch": w["so_batch_moi_epoch"], "ma_nguon_sha256":
            w["ma_nguon_sha256"], "so_batch_ghi": len(rows), **dc, "ni_cap_nhat_thuc_chay": cap_log,
        "khop_mo_phong": dc["ni_cap_nhat_mo_phong"] == cap_log,
        "theo_epoch": [{"epoch": e["epoch"], "mosaic_thuc_te": e.get("mosaic_thuc_te"), "so_batch": e["so_batch"],
                        "so_lan_cap_nhat": e["so_lan_cap_nhat"], "lr_cuoi": e["lr_cuoi"],
                        "torch_max_allocated_mib": e["torch_max_allocated_mib"],
                        "torch_max_reserved_mib": e["torch_max_reserved_mib"], "fitness": e.get("fitness"),
                        "metrics_val_ultralytics": e.get("metrics_val_ultralytics")} for e in p["epoch"]],
        "lr_dau_cuoi_epoch": {f"epoch{ep}": {g: [float(rr[f"lr_{g}"]) for rr in (r[0], r[-1])] for g in ("weight", "bn",
                                                                                                         "bias")}
                              for ep in (1, 2) if (r := [x for x in rows if int(x["epoch"]) == ep])},
        "accumulate_dau_cuoi": [int(rows[0]["accumulate"]), int(rows[-1]["accumulate"])] if rows else None,
        "loss_huu_han_moi_batch": all(r["loss_huu_han"] == "True" for r in rows),
        "loss_trung_binh_theo_epoch": {f"epoch{ep}": {c: float(np.mean([float(x[c]) for x in rows if int(x["epoch"]) == ep]))
                                                      for c in rows[0] if c.endswith("_loss")} for ep in (1, 2)}}
    moc = p["moc"]

    def t(ten_moc, ep=None, lan=0):
        ds = [x["t"] for x in moc if x["ten"] == ten_moc and (ep is None or x.get("epoch") == ep)]
        return ds[lan] if len(ds) > lan else None
    val_s = [x["t"] for x in moc if x["ten"] == "on_val_start"]
    val_e = [x["t"] for x in moc if x["ten"] == "on_val_end"]
    out["thoi_gian_s"] = {
        "model_train_tong": rec.get("thoi_gian_s"),
        "do_truoc_hoc": round(t("het_do_truoc_hoc") - t("on_train_start"), 1),
        "theo_epoch": [{"epoch": ep,
                        "train_tu_bat_dau_epoch": round(t("on_train_epoch_end", ep) - t("on_train_epoch_start", ep), 1),
                        "train_s_runner": rec["epoch"][ep - 1]["train_s"],
                        "val_trong_epoch": round(val_e[ep - 1] - val_s[ep - 1], 1) if len(val_e) >= ep else None,
                        "het_val_toi_luu_model": round(t("on_model_save", ep) - val_e[ep - 1], 1) if len(val_e) >= ep else None,
                        "val_s_runner_tu_het_train_toi_fit": rec["epoch"][ep - 1].get("val_s"),
                        "do_sau_epoch": round(t("het_do_sau_epoch", ep) - t("on_fit_epoch_end", ep), 1)}
                       for ep in (1, 2)],
        "final_eval_val": round(val_e[-1] - val_s[-1], 1) if len(val_s) >= 3 else None,
        "danh_gia_cuoi_ultralytics": rec.get("danh_gia_cuoi_ultralytics", {}).get("thoi_gian_tu_het_train_epoch_cuoi_s"),
        "do_checkpoint": {k: p["do"][k]["thoi_gian_tong_s"] for k in ("checkpoint_last", "checkpoint_best") if k in p["do"]}}
    tn = list(csv.DictReader(open(KQ / f"{ten}_tai_nguyen_nvidia_smi.csv", encoding="utf-8")))
    out["tai_nguyen"] = {
        "nvidia_smi_ca_gpu_mib": thong_ke([int(x["nvidia_smi_memory_used_mib_ca_gpu"]) for x in tn
                                           if x["nvidia_smi_memory_used_mib_ca_gpu"]]),
        "ram_he_thong_con_trong_mib_runner": thong_ke([int(x["ram_he_thong_con_trong_mib"]) for x in tn
                                                      if x["ram_he_thong_con_trong_mib"]]),
        "gpu_util_pct": thong_ke([int(x["gpu_util_pct"]) for x in tn if x["gpu_util_pct"]]),
        "cach": "luong DoVram trong runner, ~1 s/mau: nvidia-smi memory.used (CA GPU) + utilization.gpu; psutil available"}
    fr = L / f"06_ram_ngoai_{m}_mau_tho.csv"
    if fr.is_file():
        rr = list(csv.DictReader(open(fr, encoding="utf-8")))
        out["tai_nguyen"]["ram_ngoai_mau_tho"] = {
            "rss_tien_trinh_chinh_mib": thong_ke([int(x["rss_tien_trinh_chinh_mib"]) for x in rr if x["rss_tien_trinh_chinh_mib"]]),
            "rss_chinh_va_con_mib": thong_ke([int(x["rss_chinh_va_con_mib"]) for x in rr]),
            "so_tien_trinh_con": thong_ke([int(x["so_tien_trinh_con"]) for x in rr]),
            "ram_he_thong_con_trong_mib": thong_ke([int(x["ram_he_thong_con_trong_mib"]) for x in rr]),
            "cach": "bo do ngoai psutil ~1 s/mau: RSS (working set) tien trinh runner va tong voi moi tien trinh con "
                    "(co the dem trung trang chia se); RAM he thong con trong = virtual_memory().available"}
    out["checkpoint"] = {k: {x: v[x] for x in ("sha256", "bytes", "dung_5_lop", "train_args_lech_cau_hinh")}
                         for k, v in rec.get("checkpoint", {}).items()}
    out["danh_gia_dev_best_pt_runner"] = (rec.get("danh_gia_dev_smoke") or {}).get("macro")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ra", required=True)
    a = ap.parse_args()
    kq = {"ghi_chu": "tom tat CHI DOC tu ket qua pilot; khong so sanh / xep hang model; dev smoke khong co motorcycle / "
                     "bicycle", "model": {}}
    for m in ("yolov8n", "yolo11n"):
        try:
            kq["model"][m] = tom_tat_model(m)
        except FileNotFoundError as e:
            kq["model"][m] = {"loi": f"thieu ket qua: {e}"}
    Path(a.ra).write_text(json.dumps(kq, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
    for m, v in kq["model"].items():
        if "loi" in v and "do" not in v:
            print(m, v["loi"])
            continue
        print(f"===== {m}: tom tat {v['tom_tat_dat']['dat'] if v['tom_tat_dat'] else None} | loi runner {v['loi_runner']}")
        n = v["truoc_hoc"]["nap_trong_so"]
        print(f"  nap: chep {n['chep_nguyen']} remap {n['remap_hang_theo_ten']} moi {n['khoi_tao_moi']} / {n['tong_tensor_model']}")
        for k, d in v["do"].items():
            if d is None:
                print(f"  {k:18s} (khong co)")
                continue
            print(f"  {k:18s} det {d['so_detection_conf_ge_0_001']:6d} | car AP50 {d['AP50_theo_lop']['car']} | mAP50 "
                  f"{d['mAP50_3_lop_co_gt']} | diem max/anh trung vi {d['diem_truoc_nguong_max_moi_anh'].get('trung_vi')} "
                  f"| bat bien {d['bat_bien']}")
        w = v["warmup_va_cap_nhat"]
        print(f"  warmup nw {w['nw_theo_ma']} | batch {w['so_batch_ghi']} | lech cong thuc {w['so_batch_lech_cong_thuc']} | "
              f"cap nhat {w['ni_cap_nhat_thuc_chay']} khop mo phong {w['khop_mo_phong']}")
    print(f"[ra] {a.ra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
