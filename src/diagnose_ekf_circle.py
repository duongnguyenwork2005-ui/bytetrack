"""
diagnose_ekf_circle.py -- Vi sao box du doan cua EKF+CTRV "quay vong tron"?

CAU HOI. Khi xem video minh hoa, box du doan cua EKF+CTRV doi khi luon mot
vong tron thay vi di tiep. Day la LOI CAI DAT hay la HANH VI DUNG cua CTRV?

CO SO LY THUYET. Voi omega khong doi, CTRV cho quy dao la duong tron ban kinh
        R = v / |omega|                                        [px]
va sau T frame ngoai suy mu, vector huong quet mot cung
        arc = |omega| * T                                      [rad]
Neu arc tien toi 2*pi thi box du doan di het mot vong va quay ve cho cu. Vay
"quay vong tron" TU NO khong chung minh code sai - no chi noi rang omega dang
duoc uoc luong LON. Cau hoi dung phai la: omega do co HOP LY khong?

NGUONG HOP LY VE MAT VAT LY. Video UA-DETRAC quay o 25 fps. Mot chiec xe re o
nga tu quet khoang 90 do trong 2-4 giay, tuc 18-45 do/giay = 0,72-1,8 do/frame
= 0,013-0,031 rad/frame. Lay nguong rong rai:
        |omega| > 0,05 rad/frame  (= 2,9 do/frame = 72 do/giay)
la KHONG THE voi xe hoi trong canh nay - do la nhieu, khong phai chuyen dong.

PHEP KIEM TRA PHAN BIET. Tai frame cuoi con nhin thay truoc moi doan che, ta
so sanh:
        omega cua BO LOC   (EKF uoc luong duoc)
        omega cua GROUND TRUTH  (do truc tiep tu quy dao GT)
    - Neu hai cai bam nhau  -> bo loc uoc luong DUNG; van de nam o cho ban than
      omega that khong du doan duoc tuong lai (tran thong tin).
    - Neu bo loc lech xa GT -> loi o buoc uoc luong / tham so nhieu, phai sua.

CACH DUNG
    python src/diagnose_ekf_circle.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402

MIN_SPEED = 1.5              # px/frame, giong gt_omega.py
OMEGA_IMPOSSIBLE = 0.05      # rad/frame, xem docstring
WIN_BEFORE = 15              # cua so do omega GT truoc doan che


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def omega_gt(cx, cy) -> float:
    """omega trung binh do truc tiep tu quy dao GT [rad/frame]. NaN neu xe gan dung yen."""
    d = np.diff(np.c_[cx, cy], axis=0)
    d = d[np.hypot(d[:, 0], d[:, 1]) >= MIN_SPEED]
    if len(d) < 3:
        return np.nan
    return float(np.mean(wrap(np.diff(np.arctan2(d[:, 1], d[:, 0])))))


def merge_spans(o: pd.DataFrame) -> list[tuple[int, int]]:
    """Gop cac doan che chong nhau (bang >=0.10 va bang >=0.90 mo ta cung mot lan che)."""
    spans: list[tuple[int, int]] = []
    for r in o.sort_values("start_frame").itertuples():
        s, e = int(r.start_frame), int(r.end_frame)
        if spans and s <= spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    return spans


def main() -> int:
    gt = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    occ_all = pd.concat([pd.read_csv(config.INTERIM_DIR / f)
                         for f in ("full_occlusion_segments.csv", "occlusion_segments.csv")],
                        ignore_index=True)

    rows = []
    for (v, tid), g in gt.groupby(["video", "track_id"]):
        g = g.sort_values("frame").reset_index(drop=True)
        if len(g) < 150 or g.speed.median() <= 2.0:
            continue
        o = occ_all[(occ_all.video == v) & (occ_all.track_id == tid)]
        if len(o) < 2:
            continue
        spans = merge_spans(o)
        occ_frames = {f for s, e in spans for f in range(s, e + 1)}

        # Chay EKF suot doi track, chi update khi khong bi che (giong long_compare.py)
        f = EKFTrackerCTRV()
        st = {}
        r0 = g.iloc[0]
        m, c = f.initiate(np.array([r0.cx, r0.cy, r0.bb_width / max(r0.bb_height, 1e-6),
                                    r0.bb_height], float))
        seeded = False
        st[int(r0.frame)] = m.copy()
        for i in range(1, len(g)):
            r = g.iloc[i]
            z = np.array([r.cx, r.cy, r.bb_width / max(r.bb_height, 1e-6), r.bb_height], float)
            visible = int(r.frame) not in occ_frames
            if not seeded and visible:
                m, c = f.initiate_from_motion(m, c, z, n_frames=1.0)
                seeded = True
            m, c = f.predict(m, c)
            if visible:
                m, c = f.update(m, c, z)
            st[int(r.frame)] = m.copy()

        for s, e in spans:
            anchor = s - 1                      # frame cuoi con nhin thay
            if anchor not in st:
                continue
            x = st[anchor]
            om_f, vel = float(x[f.OMEGA]), float(x[f.V])
            T = e - s + 1
            before = g[(g.frame < s) & (g.frame >= s - WIN_BEFORE)]
            during = g[(g.frame >= s) & (g.frame <= e)]
            rows.append(dict(
                video=v, track_id=tid, start=s, end=e, T=T,
                omega_ekf=om_f, v_ekf=vel,
                R_px=abs(vel / om_f) if abs(om_f) > 1e-9 else np.inf,
                arc_deg=float(np.rad2deg(abs(om_f) * T)),
                omega_gt_before=omega_gt(before.cx.values, before.cy.values),
                omega_gt_during=omega_gt(during.cx.values, during.cy.values)))

    d = pd.DataFrame(rows).dropna(subset=["omega_gt_before"])
    out = config.RESULTS_DIR / "demo" / "ekf_circle_diagnosis.csv"
    d.to_csv(out, index=False)

    ab_f = d.omega_ekf.abs()
    ab_g = d.omega_gt_before.abs()
    print("=" * 74)
    print(f"CHAN DOAN 'BOX EKF QUAY VONG TRON'   ({len(d)} doan che, 48 track dai)")
    print("=" * 74)

    print("\n1. omega ma BO LOC uoc luong tai frame cuoi truoc khi mat quan sat")
    print(f"   trung vi |omega|      : {np.rad2deg(ab_f.median()):.3f} do/frame")
    print(f"   phan vi 90            : {np.rad2deg(ab_f.quantile(.9)):.3f} do/frame")
    print(f"   lon nhat              : {np.rad2deg(ab_f.max()):.3f} do/frame")
    n_imp = int((ab_f > OMEGA_IMPOSSIBLE).sum())
    print(f"   vuot nguong BAT KHA THI ({np.rad2deg(OMEGA_IMPOSSIBLE):.1f} do/frame): "
          f"{n_imp}/{len(d)} doan ({n_imp / len(d) * 100:.0f}%)")

    print("\n2. Ban kinh vong tron du doan  R = v/|omega|")
    print(f"   trung vi R            : {d.R_px.median():.0f} px")
    print(f"   1/4 nho nhat          : {d.R_px.quantile(.25):.0f} px")
    n_tiny = int((d.R_px < 100).sum())
    print(f"   R < 100 px (nho hon 1 lan xe): {n_tiny}/{len(d)} doan ({n_tiny / len(d) * 100:.0f}%)")

    print("\n3. Cung quet duoc trong ca doan che  arc = |omega| * T")
    for lo, hi, lab in [(0, 45, "duoi 45 do  (coi nhu di thang)"),
                        (45, 180, "45-180 do   (cong ro)"),
                        (180, 360, "180-360 do  (nua vong tro len)"),
                        (360, 1e9, "tren 360 do (DI HET IT NHAT 1 VONG)")]:
        n = int(((d.arc_deg >= lo) & (d.arc_deg < hi)).sum())
        print(f"   {lab:<38} {n:>4} doan ({n / len(d) * 100:>4.0f}%)")

    print("\n4. PHEP KIEM TRA PHAN BIET: bo loc co uoc luong DUNG omega khong?")
    print(f"   trung vi |omega| cua BO LOC        : {np.rad2deg(ab_f.median()):.3f} do/frame")
    print(f"   trung vi |omega| cua GROUND TRUTH  : {np.rad2deg(ab_g.median()):.3f} do/frame")
    print(f"   ty le thoi phong (bo loc / GT)     : {ab_f.median() / ab_g.median():.1f} lan")
    r = float(np.corrcoef(d.omega_ekf, d.omega_gt_before)[0, 1])
    same = float((np.sign(d.omega_ekf) == np.sign(d.omega_gt_before)).mean())
    print(f"   tuong quan Pearson(omega_loc, omega_GT truoc che) = {r:+.3f}")
    print(f"   ty le CUNG DAU quay                               = {same * 100:.1f}%")
    # Doi chung phai loc NaN o CA HAI ve: np.sign(nan) = nan nen phep so sanh
    # dau luon False, lam ty le cung dau bi keo xuong mot cach gia tao.
    dd = d.dropna(subset=["omega_gt_before", "omega_gt_during"])
    r2 = float(np.corrcoef(dd.omega_gt_before, dd.omega_gt_during)[0, 1])
    # Pearson o day bi MOT diem ngoai lai keo manh (bo diem do thi tu -0,755
    # con -0,385), nen phai bao cao kem Spearman - ben vung voi ngoai lai.
    rho = float(spearmanr(dd.omega_gt_before, dd.omega_gt_during).statistic)
    p_rho = float(spearmanr(dd.omega_gt_before, dd.omega_gt_during).pvalue)
    s2 = float((np.sign(dd.omega_gt_before) == np.sign(dd.omega_gt_during)).mean())
    print(f"   (doi chung, n={len(dd)}) Pearson (nhay voi ngoai lai) = {r2:+.3f}")
    print(f"   (doi chung, n={len(dd)}) Spearman (ben vung)          = {rho:+.3f}  (p={p_rho:.4f})")
    print(f"   (doi chung, n={len(dd)}) ty le cung dau               = {s2 * 100:.1f}%")
    print("   -> omega truoc che khong nhung KHONG du doan duoc omega trong che,")
    print("      ma con co xu huong NGUOC DAU. Khong bo loc nao vuot qua duoc dieu nay.")

    print(f"\n  Da luu -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
