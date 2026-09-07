"""
phaseD_confirm.py -- Giai doan D: kiem dinh XAC NHAN (confirmatory) tren tap TEST.

===========================================================================
VAN DE THONG KE PHAI XU LY: OPTIONAL STOPPING
===========================================================================
Gop them du lieu roi kiem dinh lai la mot dang "optional stopping" - neu cu
them mau roi thu lai cho den khi thay p < 0.05 thi xac suat duong tinh gia
tang manh, khong con la 5% nua.

CACH XU LY O DAY (3 lop bao ve):

1. GIA THUYET VA PHAN TANG DUOC DINH NGHIA TRUOC tren tap TRAIN, o cac giai
   doan da hoan thanh. File nay chi kiem dinh LAI dung cac o do tren tap TEST -
   day la vai tro XAC NHAN DOC LAP (confirmatory), khong phai kham pha.

2. KHONG DI TIM O MOI. Danh sach o kiem dinh duoc ghi cung trong PRE_REGISTERED
   duoi day. Neu tap test lo ra mot o khac co p nho, o do KHONG duoc bao cao
   nhu phat hien - chi duoc coi la giả thuyet moi can du lieu khac de kiem chung.

3. BAO CAO TRAIN VA TEST RIENG BIET, khong chi bao cao ban gop. Bao cao ban gop
   ma khong kem hai ban rieng se che mat viec hai tap co nhat quan hay khong.

===========================================================================
CAC O DA DINH NGHIA TRUOC (tu Giai doan 5 va Stage C tren tap TRAIN)
===========================================================================
O trong tam cua de tai, theo dung thu tu uu tien da neu trong bao cao:
  1. che >= 0.90 x medium  - noi EKF tung cho 11.54% so voi KF 6.41% (p=0.2188)
  2. che >= 0.90 x long    - noi ca 3 model deu 0% o buffer=30 (floor effect)
  3. che >= 0.10 x long    - noi tung co 1 cap KTC khong chua 0 (KF > EKF)

CACH DUNG
    python src/phaseD_confirm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from stratified_analysis import mcnemar, bootstrap_effect  # noqa: E402

#: Cac o DA DINH NGHIA TRUOC tren tap train. KHONG duoc them o moi sau khi
#: nhin thay ket qua tap test.
PRE_REGISTERED = [
    ("full", "medium"),
    ("full", "long"),
    ("partial", "long"),
]

LBL_FT = {"yolov8n-ft-bytetrack": "KF + CV",
          "yolov8n-ft-ekf-ctrv": "EKF + CTRV",
          "yolov8n-ft-ukf-ctrv": "UKF + CTRV"}
LBL_TEST = {"yolov8n-test-bytetrack": "KF + CV",
            "yolov8n-test-ekf-ctrv": "EKF + CTRV",
            "yolov8n-test-ukf-ctrv": "UKF + CTRV"}


def load(path: Path, lbl: dict) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [MISS] {path.name}")
        return None
    d = pd.read_csv(path)
    d = d[d.in_common].copy()
    if d.motion_model.isin(lbl).any():
        d["motion_model"] = d.motion_model.map(lbl).fillna(d.motion_model)
    return d


def summarise(d: pd.DataFrame, name: str) -> None:
    print(f"\n--- {name} ---")
    pv = d.pivot_table(index=["occlusion_level", "video", "track_id", "seg_id"],
                       columns="motion_model", values="status", aggfunc="first")
    same = pv.nunique(axis=1) == 1
    print(f"  Tong doan chung: {len(pv):,} | 3 model giong het nhau "
          f"{same.sum():,} ({same.mean() * 100:.1f}%) | khac nhau {(~same).sum()}")
    g = d.groupby("status").size()
    print("  Phan bo:  " + "  ".join(f"{k}={v / g.sum() * 100:.1f}%" for k, v in g.items()))

    for lvl, bkt in PRE_REGISTERED:
        s = d[(d.occlusion_level == lvl) & (d.bucket == bkt)]
        if not len(s):
            continue
        r = s.groupby("motion_model").status.apply(lambda x: (x == "preserved").mean() * 100)
        n = len(s) // s.motion_model.nunique()
        vals = "  ".join(f"{m}={r.get(m, float('nan')):.2f}%"
                         for m in ["KF + CV", "EKF + CTRV", "UKF + CTRV"])
        print(f"  [{lvl} x {bkt}] n={n:<5} {vals}")


def test_cells(d: pd.DataFrame, name: str) -> pd.DataFrame:
    """Kiem dinh CHI tren cac o da dinh nghia truoc."""
    keep = pd.Series(False, index=d.index)
    for lvl, bkt in PRE_REGISTERED:
        keep |= (d.occlusion_level == lvl) & (d.bucket == bkt)
    sub = d[keep]
    if not len(sub):
        return pd.DataFrame()
    mc = mcnemar(sub, ["occlusion_level", "bucket"], "KF + CV")
    bs = bootstrap_effect(sub, ["occlusion_level", "bucket"], n_boot=5000)
    mc["tap"] = name
    bs["tap"] = name
    return mc, bs


def main() -> int:
    R = config.RESULTS_DIR / "stratified"
    tr = load(R / "segment_outcomes_DETRAC-all_ft.csv", LBL_FT)
    te = load(R / "segment_outcomes_DETRAC-test.csv", LBL_TEST)

    print("=" * 84)
    print("GIAI DOAN D - KIEM DINH XAC NHAN TREN TAP TEST DOC LAP")
    print("=" * 84)
    print("  Cac o duoc kiem dinh DA DINH NGHIA TRUOC tren tap train:")
    for lvl, b in PRE_REGISTERED:
        print(f"    - {lvl} x {b}")
    print("  KHONG di tim o moi tren tap test (tranh optional stopping).")

    if tr is not None:
        summarise(tr, "TAP TRAIN (60 video, detector 3-fold)")
    if te is not None:
        summarise(te, "TAP TEST (40 video, detector huan luyen tren 60 video train)")

    out = config.RESULTS_DIR / "phaseD"
    out.mkdir(parents=True, exist_ok=True)
    mcs, bss = [], []
    for d, nm in ((tr, "train"), (te, "test")):
        if d is None:
            continue
        r = test_cells(d, nm)
        if isinstance(r, tuple):
            mcs.append(r[0]); bss.append(r[1])

    if mcs:
        mc = pd.concat(mcs, ignore_index=True)
        bs = pd.concat(bss, ignore_index=True)
        mc.to_csv(out / "mcnemar_prereg.csv", index=False)
        bs.to_csv(out / "bootstrap_prereg.csv", index=False)

        print("\n" + "=" * 84)
        print("McNEMAR tren cac o DA DINH NGHIA TRUOC - train va test RIENG BIET")
        print("=" * 84)
        show = mc[(mc.baseline_only + mc.model_only) > 0]
        cols = ["tap", "occlusion_level", "bucket", "model", "n_segments",
                "baseline_only", "model_only", "delta_preserved", "p_value"]
        print(show[cols].to_string(index=False) if len(show)
              else "  (khong co cap bat dong nao)")
        n_sig = int(mc.significant_5pct.sum())
        print(f"\n  So o dat p < 0.05: {n_sig}/{len(mc)}")

        print("\n" + "=" * 84)
        print("BOOTSTRAP effect size + KTC 95% (cluster theo track)")
        print("=" * 84)
        c2 = ["tap", "occlusion_level", "bucket", "model_a", "model_b",
              "n_segments", "effect_pp", "trk_lo", "trk_hi", "CI_chua_0"]
        print(bs[c2].to_string(index=False))
        print(f"\n  Cap co KTC khong chua 0: {int((~bs.CI_chua_0).sum())}/{len(bs)}")

    print(f"\n  Da luu -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
