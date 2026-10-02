"""Independent, read-only audit of round 37 receipts; does NOT train/infer.

Usage: python check_round37.py --tree TREE --out evidence/independent.json
Needs only numpy and the supplied target files/logs. AP/confidence values are
cross-checked as receipts, NOT recomputed from images or checkpoints.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def load_json(p):
    return json.loads(p.read_text(encoding="utf-8"))


def rows(p):
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    tree = args.tree.resolve()
    log = tree / "handoff/logs/review37"
    report = {"scope": "Independent arithmetic/hash/receipt checks; no image, checkpoint, GPU or training rerun.",
              "models": {}, "checks": []}

    def check(name, ok):
        report["checks"].append({"name": name, "pass": bool(ok)})
        if not ok:
            print("FAIL", name)

    summary = load_json(log / "07_tom_tat_pilot.json")
    original_summary = load_module("original_summary37", log / "07_tom_tat_pilot.py")
    # Rerun the supplied summarizer using exact handed-over receipts under its
    # expected filenames, exclusively in a reviewer-created scratch directory.
    stage = args.out.parent / "summary_inputs"
    stage.mkdir(parents=True, exist_ok=True)
    original_summary.KQ = stage
    original_summary.L = log
    expected_names = ["car", "truck", "bus", "motorcycle", "bicycle"]
    initial_sha = {
        "yolov8n": "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36",
        "yolo11n": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    }
    recomputed_summary = {"ghi_chu": summary["ghi_chu"], "model": {}}
    for model in initial_sha:
        rec = load_json(log / f"06_ket_qua_runner_{model}.json")
        lock = load_json(log / f"01_khoa_cau_hinh_pilot_{model}.json")
        batch = rows(log / f"06_batch_{model}.csv")
        p = rec["pilot"]
        a = p["warmup_theo_ma"]["args"]
        w = p["warmup_theo_ma"]
        bc = p["truoc_hoc"]["bang_chung_khoi_tao"]
        out = {}
        check(f"{model}: declared/runtime params equal", all(rec["hieu_luc"]["args"].get(k) == v
                                                             for k, v in rec["tham_so_khai_bao"].items()))
        for k in ("cau_hinh", "checkpoint_khoi_tao", "tham_so_khai_bao", "tham_so_resolve_tinh", "pilot_khac_smoke"):
            check(f"{model}: locked {k}", rec[k] == lock[k])
        check(f"{model}: original checkpoint", rec["checkpoint_khoi_tao"]["sha256"] == initial_sha[model])
        check(f"{model}: epochs only changed 1->2", rec["pilot_khac_smoke"] == {"epochs": [1, 2]})
        check(f"{model}: 334/84", rec["dataset"]["so_anh"] == {"train": 334, "dev": 84})
        ds_sha = {
            "sha256_data_yaml": "9744ba88ce2f10438b4d9dfcf3ffeade018f319c2c5bde8eeb397d199e458661",
            "sha256_tom_tat": "1c4e7521e397160f80e74c29e7f077d3aa29a9de29e161363dac65acecf8c64c",
            "manifest_sha256": "c48998266b5490531df6834d715b9d159fb6555e063cc53520e39b41c9b074c1",
            "sha256_video_json": "a07d12f893055f83ac504a185864ee031c8a3dfb8d2ccebca792e28938da9912",
        }
        check(f"{model}: round35 dataset hashes", all(rec["dataset"][k] == h for k, h in ds_sha.items()))
        check(f"{model}: loaded dataset counts and labels", all(
            rec["dau_vao_da_nap"][s]["so_anh_nap"] == n
            and all(rec["dau_vao_da_nap"][s][k] == 0 for k in ("thieu", "thua", "nhan_khac", "npy_canh_anh"))
            for s, n in (("train", 334), ("dev", 84))))
        check(f"{model}: two complete epochs and success", [e["epoch"] for e in p["epoch"]] == [1, 2]
              and rec["loi"] is None and p["loi"] is None and p["tom_tat"]["dat"])
        check(f"{model}: before first optimizer update", bc["optimizer"]["so_trang_thai"] == 0
              and bc["optimizer"]["ema_updates"] == 0 and bc["so_tham_so_co_gradient"] == 0
              and bc["model_va_ema_giong_het"] and list(bc["ten_lop_model"].values()) == expected_names)
        check(f"{model}: recorded warmup source hash", hashlib.sha256(w["ma_nguon"].encode()).hexdigest()
              == w["ma_nguon_sha256"])
        check(f"{model}: exactly 84 ordered batches", [int(r["ni"]) for r in batch] == list(range(84))
              and [int(r["epoch"]) for r in batch] == [1] * 42 + [2] * 42)
        check(f"{model}: numeric finite losses", all(math.isfinite(float(v))
              for r in batch for k, v in r.items() if k == "loss_tong" or k.endswith("_loss")))
        nw = round(min(a["warmup_epochs"], max(a["epochs"] - 1, 0)) * 42)
        check(f"{model}: nw 42", nw == w["so_buoc_warmup_nw"] == 42)
        max_lr_error = 0.0
        max_momentum_error = 0.0
        expected_updates, observed_updates = [], []
        last = -1
        previous = 0
        for r in batch:
            i, ep = int(r["ni"]), int(r["epoch"]) - 1
            factor = max(1 - ep / a["epochs"], 0) * (1 - a["lrf"]) + a["lrf"]
            acc = max(1, round(1 + (a["nbs"] / a["batch"] - 1) * min(i, nw - 1) / nw))
            check(f"{model}: accumulate batch {i}", int(r["accumulate"]) == acc)
            for group in ("weight", "bn", "bias"):
                end = a["lr0"] * factor
                start = a["warmup_bias_lr"] if group == "bias" else 0.0
                expected_lr = start + (end - start) * i / nw if i < nw else end
                max_lr_error = max(max_lr_error, abs(float(r[f"lr_{group}"]) - expected_lr))
                expected_m = a["warmup_momentum"] + (a["momentum"] - a["warmup_momentum"]) * min(i, nw - 1) / nw
                max_momentum_error = max(max_momentum_error, abs(float(r[f"momentum_{group}"]) - expected_m))
            if i - last >= acc:
                expected_updates.append(i)
                last = i
            current = int(r["ema_updates"])
            check(f"{model}: EMA increment batch {i}", current - previous in (0, 1))
            if current > previous:
                observed_updates.append(i)
            previous = current
        check(f"{model}: LR formula", max_lr_error < 1e-12)
        check(f"{model}: momentum formula", max_momentum_error < 1e-12)
        check(f"{model}: independent update schedule", expected_updates == observed_updates
              == [x for e in p["epoch"] for x in e["ni_co_cap_nhat"]]
              and previous == 17 and [e["so_lan_cap_nhat"] for e in p["epoch"]] == [12, 5])
        check(f"{model}: no mosaic", all(e["mosaic_thuc_te"] is False for e in p["epoch"]))
        out["warmup"] = {"nw": nw, "max_lr_abs_error": max_lr_error,
                         "max_momentum_abs_error": max_momentum_error, "optimizer_update_batches": observed_updates}
        out["measurements"] = {}
        expected_points = ["truoc_hoc_model", "sau_epoch1_model", "sau_epoch1_ema", "sau_epoch2_model", "sau_epoch2_ema",
                           "checkpoint_last", "checkpoint_best"]
        check(f"{model}: seven measurements", set(p["do"]) == set(expected_points))
        for key in expected_points:
            d = p["do"][key]
            check(f"{model}: {key} protocol", d["so_anh"] == 84 and d["quy_tac_du_doan"]["conf"] == 0.001
                  and d["quy_tac_du_doan"]["iou_nms"] == 0.7 and d["quy_tac_du_doan"]["max_det"] == 300)
            counts = d["detection"]
            check(f"{model}: {key} count sums", counts["tong"] == sum(counts["theo_lop"].values())
                  == sum(counts["histogram"].values()))
            dg = d["danh_gia_chinh"]
            check(f"{model}: {key} class support", [dg["theo_lop"][c]["n_gt"] for c in expected_names]
                  == [542, 6, 4, 0, 0])
            check(f"{model}: {key} mAP averages supported classes", abs(dg["macro"]["mAP50"] -
                  sum(dg["theo_lop"][c]["AP50"] for c in expected_names[:3]) / 3) < 1e-14)
            if "bat_bien" in d:
                check(f"{model}: {key} restoration receipt", d["bat_bien"]["khong_doi_sau_khoi_phuc"]
                      and d["bat_bien"]["khac_sau_khoi_phuc"] == [])
            out["measurements"][key] = {"detections": counts["tong"], "car_AP50_receipt": dg["theo_lop"]["car"]["AP50"],
                                         "median_max_score_receipt": d["diem_truoc_nguong"]["max_moi_anh"]["trung_vi"]}
        check(f"{model}: zero detections before learning", p["do"]["truoc_hoc_model"]["detection"]["tong"] == 0)
        bcw = bc["nap_trong_so"]
        check(f"{model}: weight accounting", bcw["chep_nguyen"] + bcw["remap_hang_theo_ten"] + bcw["khoi_tao_moi"]
              == bcw["tong_tensor_model"] and bcw["remap_hang_theo_ten"] == 3
              and bcw["khoi_tao_moi"] == (33 if model == "yolov8n" else 48) and not bcw["cung_shape_khac_gia_tri"])
        for name in ("last", "best"):
            check(f"{model}: checkpoint {name} receipt", rec["checkpoint"][name]["dung_5_lop"]
                  and rec["checkpoint"][name]["train_args_lech_cau_hinh"] == {}
                  and rec["checkpoint"][name]["sha256"] == p["do"][f"checkpoint_{name}"]["sha256"])
        marks = p["moc"]
        def t(name, epoch=None):
            return next(r["t"] for r in marks if r["ten"] == name and r.get("epoch") == epoch)
        check(f"{model}: before learning ordered timestamps", t("on_train_start") < t("het_do_truoc_hoc")
              <= t("on_train_epoch_start", 1) <= float(batch[0]["t"]))
        durations = []
        for ep in (1, 2):
            start = t("het_do_truoc_hoc") if ep == 1 else t("het_do_sau_epoch", ep - 1)
            train = t("on_train_epoch_end", ep) - start
            val = t("on_fit_epoch_end", ep) - t("on_train_epoch_end", ep)
            check(f"{model}: epoch{ep} train timing", abs(train - rec["epoch"][ep - 1]["train_s"]) < 0.02)
            check(f"{model}: epoch{ep} val timing", abs(val - rec["epoch"][ep - 1]["val_s"]) < 0.02)
            durations.append({"train_s": train, "val_plus_save_s": val})
        out["epoch_durations"] = durations
        nr = rows(log / f"06_tai_nguyen_nvidia_smi_{model}.csv")
        ram = rows(log / f"06_ram_ngoai_{model}_mau_tho.csv")
        out["resources"] = {
            "gpu_mib_max": max(int(r["nvidia_smi_memory_used_mib_ca_gpu"]) for r in nr if r["nvidia_smi_memory_used_mib_ca_gpu"]),
            "available_ram_min_runner_mib": min(int(r["ram_he_thong_con_trong_mib"]) for r in nr if r["ram_he_thong_con_trong_mib"]),
            "available_ram_min_external_mib": min(int(r["ram_he_thong_con_trong_mib"]) for r in ram),
            "rss_main_children_max_mib": max(int(r["rss_chinh_va_con_mib"]) for r in ram),
            "gpu_samples": len(nr), "ram_samples": len(ram),
        }
        check(f"{model}: peak resource summaries", out["resources"]["gpu_mib_max"] == rec["vram_nvidia_smi"]["max_mib"]
              and out["resources"]["available_ram_min_external_mib"] ==
              load_json(log / f"06_ram_ngoai_{model}.json")["ram_he_thong_con_trong_min_mib"])
        prefix = f"train_bdd5_v1_{model}_pilot"
        for dst, src in ((f"{prefix}.json", f"06_ket_qua_runner_{model}.json"),
                         (f"{prefix}_batch.csv", f"06_batch_{model}.csv"),
                         (f"{prefix}_tai_nguyen_nvidia_smi.csv", f"06_tai_nguyen_nvidia_smi_{model}.csv")):
            (stage / dst).write_bytes((log / src).read_bytes())
        recomputed_summary["model"][model] = original_summary.tom_tat_model(model)
        check(f"{model}: supplied summary reproducible", recomputed_summary["model"][model] == summary["model"][model])
        report["models"][model] = out
    report["pass"] = sum(c["pass"] for c in report["checks"])
    report["fail"] = sum(not c["pass"] for c in report["checks"])
    args.out.parent.mkdir(exist_ok=True, parents=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.out.parent / "summary_recomputed.json").write_text(json.dumps(recomputed_summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "fail": report["fail"], "models": report["models"]}, ensure_ascii=False, indent=2))
    return 0 if not report["fail"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
