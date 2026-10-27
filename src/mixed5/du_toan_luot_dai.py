"""
du_toan_luot_dai.py -- tom tat smoke R52B + du toan ngan sach luot dai 15 epoch (KICH BAN theo gia dinh, KHONG phai can
tren / bao dam).

    python -X utf8 src/mixed5/du_toan_luot_dai.py

Chi doc: results/phase2/mixed5/smoke/train_mixed5_v1_<M>_smoke.json (smoke 1 epoch R52B) va
results/phase2/detector_train/train_bdd5_v1_<M>_day_du*.json (luot dai BDD-only da chay, de doi chieu s/batch thuc te
nhieu gio). Ghi results/phase2/mixed5/smoke/tom_tat_smoke.json + du_toan_luot_dai.json.

GIA DINH (ghi vao file ket qua)
- batch/epoch = ceil(anh train / 8) (dataset day du mixed5_v1: 57.043 anh -> 7.131 batch); khong tinh dung som.
- Kich ban A (smoke): train/epoch = batch/epoch x s/batch trung vi cua smoke (bo 5 batch dau).
- Kich ban B (luot dai BDD-only thuc do): s/batch hieu dung = trung vi (train_s / so_batch) qua cac epoch cua luot dai
  BDD-only cung model, cung may, cung batch/worker.
- val/epoch (ca hai kich ban) = trung vi val_s / anh dev cua luot dai BDD-only x 7.190 anh dev mixed5_v1.
- Khoi tao moi lan chay runner: 2 x thoi gian bam hop dong dataset day du (kiem-cau-hinh R52B do ~155 s; runner bam truoc
  khi nap model va callback doi chieu nhan) + 5 phut quet nhan / tao cache ultralytics.
- Kich ban cham hon: x1,3 cho rui ro chua do (giam xung khi chay nhieu gio, RAM thap / paging, resume mat epoch dang chay).
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mixed5_lib as m  # noqa: E402
from mixed5_lib import GOC, bx  # noqa: E402

MODELS = ("yolov8n", "yolo11n")
SMOKE = GOC / "results/phase2/mixed5/smoke"
BDD_DAI = GOC / "results/phase2/detector_train"
ANH_TRAIN, ANH_DEV, BATCH, EPOCH = 57043, 7190, 8, 15
BAM_HOP_DONG_S = 155.0
QUET_NHAN_S = 300.0
HE_SO_CHAM = 1.3


def doc(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def tom_tat_smoke(md: str) -> dict:
    p = SMOKE / f"train_mixed5_v1_{md}_smoke.json"
    r = doc(p)
    e = r["epoch"][0]
    dg = r.get("danh_gia_dev_smoke", {})
    return {
        "file": bx.rel(p, GOC), "sha256_file": bx.sha256_file(p), "loi": r["loi"], "loi_callback": r.get("loi_callback"),
        "loss_huu_han": r["loss_huu_han"], "loss_epoch1": e["loss"], "so_batch": e["so_batch"],
        "s_moi_batch_trung_vi": e["on_dinh"]["s_moi_batch_trung_vi"], "anh_moi_s_trung_vi": e["on_dinh"]["anh_moi_s_trung_vi"],
        "train_s": e["train_s"], "val_s": e["val_s"], "thoi_gian_tong_s": r["thoi_gian_s"],
        "torch_max_allocated_mib": e.get("torch_max_allocated_mib"), "torch_max_reserved_mib": e.get("torch_max_reserved_mib"),
        "nvidia_smi_max_mib_ca_gpu": r["vram_nvidia_smi"]["max_mib"],
        "ram_con_trong_min_mib": r["vram_nvidia_smi"].get("ram_con_trong_min_mib"),
        "gpu_util_trung_vi_pct": r["vram_nvidia_smi"].get("gpu_util_trung_vi_pct"),
        "checkpoint": {n: {"sha256": c["sha256"], "nap_lai": c["nap_lai"], "dung_5_lop": c["dung_5_lop"],
                           "names": c["names"], "train_args_lech_cau_hinh": c["train_args_lech_cau_hinh"]}
                       for n, c in r["checkpoint"].items()},
        "danh_gia_dev_smoke": {"n_anh": dg.get("n_anh"), "macro": dg.get("macro"),
                               "AP50_theo_lop": {c: v["AP50"] for c, v in dg.get("theo_lop", {}).items()},
                               "ghi_chu": "chi kiem duong chay; KHONG so sanh model"},
        "dataset": {k: r["dataset"][k] for k in ("ten", "loai", "manifest_sha256", "so_anh", "loi")},
        "cau_hinh_sha256": r["cau_hinh"]["sha256"],
    }


def bdd_dai(md: str) -> dict:
    s_batch, val_anh, files = [], [], []
    for p in sorted(BDD_DAI.glob(f"train_bdd5_v1_{md}_day_du*.json")):
        r = doc(p)
        dev = r["dataset"]["so_anh"]["dev"]
        files.append(bx.rel(p, GOC))
        for e in r.get("epoch", []):
            if e.get("train_s") and e.get("so_batch"):
                s_batch.append(e["train_s"] / e["so_batch"])
            if e.get("val_s"):
                val_anh.append(e["val_s"] / dev)
    return {"file": files, "so_epoch_do": len(s_batch), "s_moi_batch_hieu_dung_trung_vi": round(st.median(s_batch), 4),
            "s_moi_batch_hieu_dung_max": round(max(s_batch), 4), "val_s_moi_anh_dev_trung_vi": round(st.median(val_anh), 5)}


def du_toan(s_batch: float, val_anh: float) -> dict:
    nb = math.ceil(ANH_TRAIN / BATCH)
    tr, va = nb * s_batch, ANH_DEV * val_anh
    kd = 2 * BAM_HOP_DONG_S + QUET_NHAN_S
    tong = EPOCH * (tr + va) + kd
    return {"batch_moi_epoch": nb, "train_s_moi_epoch": round(tr), "val_s_moi_epoch": round(va),
            "h_moi_epoch": round((tr + va) / 3600, 2), "khoi_tao_s": round(kd), "tong_h_15_epoch": round(tong / 3600, 2),
            "tong_h_cham_hon": round(tong * HE_SO_CHAM / 3600, 2)}


def main() -> int:
    sm = {md: tom_tat_smoke(md) for md in MODELS}
    loi = [f"{md}: {k}" for md, s in sm.items() for k, ok in (
        ("loi", s["loi"] is None), ("loi_callback", not s["loi_callback"]), ("loss_huu_han", s["loss_huu_han"]),
        ("checkpoint last+best", {"last", "best"} <= set(s["checkpoint"])),
        ("dung_5_lop", all(c["dung_5_lop"] and c["nap_lai"] for c in s["checkpoint"].values())),
        ("danh_gia_dev", s["danh_gia_dev_smoke"]["n_anh"] == 295)) if not ok]
    m.ghi_json(SMOKE / "tom_tat_smoke.json", {"smoke": sm, "loi": loi, "dat": not loi})
    bd = {md: bdd_dai(md) for md in MODELS}
    kb = {}
    for md in MODELS:
        kb[md] = {"A_theo_smoke": du_toan(sm[md]["s_moi_batch_trung_vi"], bd[md]["val_s_moi_anh_dev_trung_vi"]),
                  "B_theo_luot_dai_bdd_only": du_toan(bd[md]["s_moi_batch_hieu_dung_trung_vi"],
                                                      bd[md]["val_s_moi_anh_dev_trung_vi"])}
    tong = {k: {"tong_h": round(sum(kb[md][k]["tong_h_15_epoch"] for md in MODELS), 2),
                "tong_h_cham_hon": round(sum(kb[md][k]["tong_h_cham_hon"] for md in MODELS), 2)} for k in kb[MODELS[0]]}
    kq = {"loai": "KICH BAN DU TOAN theo gia dinh -- khong phai can tren hay bao dam",
          "gia_dinh": {"anh_train": ANH_TRAIN, "anh_dev": ANH_DEV, "batch": BATCH, "epoch": EPOCH,
                       "bam_hop_dong_s": BAM_HOP_DONG_S, "quet_nhan_s": QUET_NHAN_S, "he_so_cham_hon": HE_SO_CHAM,
                       "khong_tinh": ["dung som patience 5", "thoi gian cho / gian doan giua cac lan resume",
                                      "epoch dang chay bi mat khi ngat (resume chi tu epoch da xong)"]},
          "smoke": {md: {k: sm[md][k] for k in ("s_moi_batch_trung_vi", "val_s", "thoi_gian_tong_s")} for md in MODELS},
          "luot_dai_bdd_only": bd, "kich_ban": kb, "tong_hai_model_tuan_tu": tong}
    m.ghi_json(SMOKE / "du_toan_luot_dai.json", kq)
    for md in MODELS:
        s = sm[md]
        print(f"[smoke {md}] s/batch {s['s_moi_batch_trung_vi']}, train {s['train_s']} s, val {s['val_s']} s, tong "
              f"{s['thoi_gian_tong_s']} s, torch reserved {s['torch_max_reserved_mib']} MiB, nvidia-smi max "
              f"{s['nvidia_smi_max_mib_ca_gpu']} MiB, RAM trong min {s['ram_con_trong_min_mib']} MiB, mAP50 dev smoke "
              f"{s['danh_gia_dev_smoke']['macro']['mAP50']:.4f}")
        for k, v in kb[md].items():
            print(f"  {k}: {v}")
    print(f"tong hai model: {tong}")
    for e in loi:
        print(f"[LOI] {e}", file=sys.stderr)
    return 0 if not loi else 3


if __name__ == "__main__":
    sys.exit(main())
