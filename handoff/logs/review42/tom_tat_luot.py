# -*- coding: utf-8 -*-
"""Vong 42: tom tat MOT model sau training dai -- CHI DOC ket qua da ghi (khong nap model, khong chay danh gia moi).

    python handoff/logs/review42/tom_tat_luot.py --model yolov8n --out handoff/logs/review42/<file>.json

Nguon (tat ca do runner / bo giam sat ghi trong luc chay):
  - results/phase2/detector_train/train_bdd5_v1_<M>_day_du.json (+ __tiep_tuc_<N>.json neu co resume);
  - models/finetune_phase2/train_bdd5_v1_<M>_day_du/nhat_ky_luot.jsonl (bat_dau / epoch_da_luu / ket_thuc);
  - results/phase2/detector_train/giam_sat/<M>_*/tom_tat.json + trang_thai.json (exit code THAT, RAM dung PID + con);
  - weights/last.pt, weights/best.pt, khoi_phuc/last_eNNN.pt (sha256 tinh lai tu file, doi chieu nhat ky / ket qua).
Ket qua dev = validation ultralytics tren DEV NOI BO moi epoch (metrics trong nhat ky) + danh gia cuoi best.pt cua
ultralytics (`danh_gia_cuoi_ultralytics`). KHONG phai test (BDD val / UA-DETRAC)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GOC = Path(__file__).resolve().parents[3]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def doc_jsonl(p: Path) -> list[dict]:
    out = []
    for dong in p.read_bytes().split(b"\n"):
        if dong.strip():
            out.append(json.loads(dong))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["yolov8n", "yolo11n"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ten = f"train_bdd5_v1_{a.model}_day_du"
    kq_dir = GOC / "results/phase2/detector_train"
    run_dir = GOC / "models/finetune_phase2" / ten
    kq_files = sorted([kq_dir / f"{ten}.json"] + sorted(kq_dir.glob(f"{ten}__tiep_tuc_*.json")))
    kq = [json.loads(p.read_text(encoding="utf-8")) for p in kq_files if p.is_file()]
    nk = doc_jsonl(run_dir / "nhat_ky_luot.jsonl")
    gs = []
    for d in sorted((kq_dir / "giam_sat").glob(f"{a.model}_*")):
        tom = json.loads((d / "tom_tat.json").read_text(encoding="utf-8")) if (d / "tom_tat.json").is_file() else None
        tt = json.loads((d / "trang_thai.json").read_text(encoding="utf-8")) if (d / "trang_thai.json").is_file() else None
        gs.append({"thu_muc": str(d.relative_to(GOC)).replace("\\", "/"), "trang_thai": (tt or {}).get("trang_thai"),
                   "exit_code": (tom or {}).get("exit_code", (tt or {}).get("exit_code")),
                   "bat_dau": (tt or {}).get("bat_dau"), "ket_thuc": (tom or {}).get("ket_thuc"),
                   "thoi_gian_s": (tom or {}).get("thoi_gian_s"), "ram": (tom or {}).get("ram"),
                   "ly_do_dung_lay_mau": (tom or {}).get("ly_do_dung_lay_mau"), "ctrl_c": (tom or {}).get("ctrl_c")})
    ep_nk = [e for e in nk if e.get("loai") == "epoch_da_luu"]
    epoch = []
    for e in ep_nk:
        m = e.get("metrics") or {}
        bs = run_dir / e["ban_sao"]
        epoch.append({"lan": e.get("lan"), "epoch": e["epoch"], "t": e.get("t_iso"), "fitness": e.get("fitness"),
                      "best_fitness": e.get("best_fitness"), "best_ghi_epoch_nay": e.get("best_ghi_epoch_nay"),
                      "dung": e.get("dung"), "mosaic_thuc_te": e.get("mosaic_thuc_te"),
                      "precision": m.get("metrics/precision(B)"), "recall": m.get("metrics/recall(B)"),
                      "mAP50": m.get("metrics/mAP50(B)"), "mAP50-95": m.get("metrics/mAP50-95(B)"),
                      "ban_sao": e["ban_sao"], "ban_sao_sha256_nhat_ky": e.get("ban_sao_sha256"),
                      "ban_sao_sha256_file": sha(bs) if bs.is_file() else None})
    thoi_gian_epoch = []
    for r in kq:
        for e in r.get("epoch", []):
            thoi_gian_epoch.append({"epoch": e.get("epoch"), "train_s": e.get("train_s"), "val_s": e.get("val_s"),
                                    "s_moi_batch_trung_vi": (e.get("on_dinh") or {}).get("s_moi_batch_trung_vi"),
                                    "loss": e.get("loss"), "torch_max_reserved_mib": e.get("torch_max_reserved_mib")})
    ck = {}
    for n in ("last", "best"):
        p = run_dir / "weights" / f"{n}.pt"
        if p.is_file():
            ck[n] = {"duong_dan": str(p.relative_to(GOC)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha(p),
                     "sha256_ket_qua_runner": ((kq[-1].get("checkpoint") or {}).get(n) or {}).get("sha256") if kq else None}
    cuoi = kq[-1] if kq else {}
    out = {
        "model": a.model, "ten_luot": ten, "pham_vi": "training + validation DEV noi bo; KHONG phai test",
        "file_ket_qua": [str(p.relative_to(GOC)).replace("\\", "/") for p in kq_files if p.is_file()],
        "lan_chay": [{"lan": e.get("lan"), "che_do_luot": e.get("che_do_luot"), "start_epoch": e.get("start_epoch"),
                      "tong_epoch": e.get("tong_epoch"), "pid": e.get("pid"), "t": e.get("t_iso"),
                      "git_head": e.get("git_head")} for e in nk if e.get("loai") == "bat_dau"],
        "ket_thuc_nhat_ky": [{k: e.get(k) for k in ("lan", "trang_thai", "exit_code", "kiem_sau_train_dat", "loi",
                                                    "epoch_da_luu_lan_nay", "t_iso")}
                             for e in nk if e.get("loai") == "ket_thuc"],
        "giam_sat": gs,
        "epoch_hoan_tat": [e["epoch"] for e in epoch], "epoch": epoch, "thoi_gian_epoch": thoi_gian_epoch,
        "danh_gia_cuoi_best_ultralytics": cuoi.get("danh_gia_cuoi_ultralytics"),
        "checkpoint": ck, "thoi_gian_runner_s": [r.get("thoi_gian_s") for r in kq],
        "vram_nvidia_smi": [r.get("vram_nvidia_smi") for r in kq], "loi": [r.get("loi") for r in kq],
        "loi_callback": [r.get("loi_callback") for r in kq], "loss_huu_han": [r.get("loss_huu_han") for r in kq],
        "dau_vao_da_nap": [r.get("dau_vao_da_nap") for r in kq],
        "kiem_khoi_phuc": [r.get("kiem_khoi_phuc") for r in kq if r.get("kiem_khoi_phuc")],
    }
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("model", "epoch_hoan_tat", "file_ket_qua", "loi")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
