"""
so_cau_hinh_train.py -- bang chung train_mixed5.yaml chi khac train_bdd5.yaml o cac khoa duoc phep (R52B).

    python -X utf8 src/mixed5/so_cau_hinh_train.py

So khoa-gia-tri tren YAML da phan tich (lam phang thanh duong dan khoa; so ca KIEU gia tri, vd 1 != 1.0 != True).
Khoa duoc phep khac: ten, dataset.day_du, dataset.smoke -- va gia tri moi phai DUNG nhu ke hoach. Kiem them cac gia tri
chot (khoi tao COCO + sha256 file, 15 epoch, batch 8, SGD, lr, patience 5, seed 0, deterministic, amp/cache false).
Ghi results/phase2/mixed5/so_cau_hinh_train.json. Ma thoat: 0 dat; 3 khac ngoai khoa duoc phep / gia tri chot sai.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mixed5_lib as m  # noqa: E402
from mixed5_lib import GOC, bx  # noqa: E402

GOC_CFG = GOC / "configs/phase2/detector/train_bdd5.yaml"
MOI_CFG = GOC / "configs/phase2/detector/train_mixed5.yaml"
DUOC_DOI = {"ten": "train_mixed5_v1", "dataset.day_du": "data/yolo_mixed5_v1/data.yaml",
            "dataset.smoke": "data/yolo_mixed5_v1_smoke/data.yaml"}
GIA_TRI_CHOT = {"huan_luyen.epochs": 15, "huan_luyen.batch": 8, "huan_luyen.optimizer": "SGD", "huan_luyen.lr0": 0.01,
                "huan_luyen.lrf": 0.01, "huan_luyen.momentum": 0.937, "huan_luyen.weight_decay": 0.0005,
                "huan_luyen.patience": 5, "huan_luyen.seed": 0, "huan_luyen.deterministic": True,
                "huan_luyen.amp": False, "huan_luyen.cache": False, "huan_luyen.imgsz": 640,
                "huan_luyen.pretrained": True, "smoke.epochs": 1}


def lam_phang(d, tien_to: str = "") -> dict:
    out = {}
    for k, v in d.items():
        p = f"{tien_to}{k}"
        if isinstance(v, dict):
            out |= lam_phang(v, p + ".")
        else:
            out[p] = v
    return out


def cung(a, b) -> bool:
    return type(a) is type(b) and a == b


def so(goc: dict, moi: dict) -> dict:
    fa, fb = lam_phang(goc), lam_phang(moi)
    khac = {k: [fa.get(k, "<thieu>"), fb.get(k, "<thieu>")] for k in sorted(set(fa) | set(fb))
            if k not in fa or k not in fb or not cung(fa[k], fb[k])}
    loi = [f"khac ngoai khoa duoc phep: {k} {v}" for k, v in khac.items() if k not in DUOC_DOI]
    loi += [f"{k} = {fb.get(k)!r}, ke hoach {v!r}" for k, v in DUOC_DOI.items() if not cung(fb.get(k), v)]
    loi += [f"{k} = {fb.get(k)!r}, gia tri chot {v!r}" for k, v in GIA_TRI_CHOT.items() if not cung(fb.get(k), v)]
    return {"so_khoa": {"goc": len(fa), "moi": len(fb)}, "khoa_khac": khac,
            "khoa_trung": sorted(k for k in fa if k in fb and cung(fa[k], fb[k])), "loi": loi}


def main() -> int:
    import yaml
    goc = yaml.safe_load(GOC_CFG.read_text(encoding="utf-8"))
    moi = yaml.safe_load(MOI_CFG.read_text(encoding="utf-8"))
    kq = so(goc, moi)
    wt = {}
    for k, v in moi["model"].items():
        p = GOC / v["weights"]
        sha = bx.sha256_file(p) if p.is_file() else None
        wt[k] = {"weights": v["weights"], "sha256_khai_bao": v["sha256"], "sha256_file": sha, "khop": sha == v["sha256"]}
        if sha != v["sha256"]:
            kq["loi"].append(f"checkpoint khoi tao {k}: sha256 file {sha} khac khai bao")
    kq["checkpoint_khoi_tao"] = wt
    kq["file"] = {"goc": {"duong_dan": bx.rel(GOC_CFG, GOC), "sha256": bx.sha256_file(GOC_CFG)},
                  "moi": {"duong_dan": bx.rel(MOI_CFG, GOC), "sha256": bx.sha256_file(MOI_CFG)}}
    kq["khoa_duoc_phep_doi"] = DUOC_DOI
    kq["gia_tri_chot_da_kiem"] = GIA_TRI_CHOT
    kq["dat"] = not kq["loi"]
    m.ghi_json(GOC / "results/phase2/mixed5/so_cau_hinh_train.json", kq)
    print(f"so khoa goc {kq['so_khoa']['goc']}, moi {kq['so_khoa']['moi']}, trung {len(kq['khoa_trung'])}; "
          f"khac: {kq['khoa_khac']}")
    print(f"checkpoint khoi tao: { {k: v['khop'] for k, v in wt.items()} }")
    for e in kq["loi"]:
        print(f"[LOI] {e}", file=sys.stderr)
    print("[OK] chi khac o khoa duoc phep" if kq["dat"] else "[KHONG DAT]")
    return 0 if kq["dat"] else 3


if __name__ == "__main__":
    sys.exit(main())
