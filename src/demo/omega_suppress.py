"""
omega_suppress.py -- GIAI DOAN 3: triet tieu omega thay vi kep omega.

BOI CANH. Kep |omega| <= 0,05 rad/frame KHONG phai fix sach tren doan
MVI_40992 track 12: giup o mot so frame nhung lam TE HON o sai so cuoi doan 2
(267 -> 350 px). Nghi ngo: kep tao mot buoc nhay roi rac trong state, tuong tac
xau voi v/theta khien ban kinh v/omega doi huong sai thay vi chi giam do cong.

CACH LAM KHAC. Thay vi kep ve mot gia tri nho khac 0, dat omega = 0 HOAN TOAN
cho toan bo qua trinh ngoai suy mu. Khi omega = 0 thi phuong trinh CTRV suy bien:
        cx' = cx + v*cos(theta)*dt
        cy' = cy + v*sin(theta)*dt
tuc tam xe di THANG theo huong theta voi toc do v khong doi. Vi predict() giu
omega' = omega, chi can dat omega = 0 MOT LAN tai thoi diem mat quan sat la no
giu nguyen 0 suot doan che (khong co update de lam no khac 0 tro lai).

HAI BIEN THE, chay song song de so sanh:
    EKFS_cond   : chi triet tieu khi occlusion_ratio cua detection cuoi > nguong
                  (gia thuyet ban dau: detection bi cat xen -> omega nhieu)
    EKFS_always : luon triet tieu khi mat quan sat, khong dieu kien

VI SAO CAN CA HAI. Do duoc tren doan nay: occlusion_ratio tai frame neo cua
doan 2 (frame 523) chi la 0,077 - xe con nhin thay 92%. Nen bien the co dieu
kien KHONG kich hoat o day, va sanity check se bi vo hieu neu chi cai bien the
do. Bien the vo dieu kien bao dam omega = 0 that su de kiem chung.

SANITY CHECK BAT BUOC (buoc 3 cua yeu cau).
Ve toan hoc, khi omega = 0 thi CTRV suy bien gan giong CV. Vay sai so cua ban
"suppressed" PHAI gan bang sai so cua KF+CV tren dung doan nay.
    - Neu gan bang  -> xac nhan omega la nguyen nhan duy nhat.
    - Neu VAN lech xa -> con nguyen nhan KHAC ngoai omega. Ba cho phai kiem tra:
        (a) khoi tao state 2 khung hinh dau (initiate_from_motion)
        (b) ngoai suy (a, h) phan bbox scale
        (c) lech giua hai cach bieu dien van toc: CV dung vx, vy truc tiep;
            CTRV suy ra vx = v*cos(theta), vy = v*sin(theta). Neu theta uoc
            luong sai NGAY CA KHI omega = 0 thi huong suy ra van sai.
Script nay in thang so lieu cho ca ba cho do.

CACH DUNG
    python src/demo/omega_suppress.py                 # chi in so lieu
    python src/demo/omega_suppress.py --render        # them video 3 panel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from ekf_ctrv import EKFTrackerCTRV  # noqa: E402
from ekf_ctrv_clamped import EKFTrackerCTRVClamped  # noqa: E402
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH  # noqa: E402

VIDEO, TRACK = "MVI_40992", 12
OCC_THRESHOLD = 0.50          # nguong occlusion_ratio de kich hoat bien the co dieu kien
FRAME_W, FRAME_H = 960, 540
FPS = 10
PANEL_GAP = 6
TRAIL = 25

COL_GT = (255, 255, 255)
COL_OK = (0, 220, 0)          # dang duoc nap quan sat
COL_BLIND = (0, 0, 235)       # dang ngoai suy mu
COL_OCC = (0, 215, 255)

MODES = ["KF", "EKF", "EKFC", "EKFS_cond", "EKFS_always"]
LABEL = {
    "KF": "KF + CV",
    "EKF": "EKF + CTRV goc",
    "EKFC": "EKF + CTRV chan w",
    "EKFS_cond": "EKF + CTRV triet w (co dk)",
    "EKFS_always": "EKF + CTRV triet w",
}


def xyah(r) -> np.ndarray:
    return np.array([r.cx, r.cy, r.bb_width / max(r.bb_height, 1e-6), r.bb_height], float)


def merge_spans(o: pd.DataFrame) -> list[tuple[int, int]]:
    """Gop cac doan che chong nhau (bang >=0.10 va bang >=0.90 ta cung mot lan che)."""
    spans: list[tuple[int, int]] = []
    for r in o.sort_values("start_frame").itertuples():
        s, e = int(r.start_frame), int(r.end_frame)
        if spans and s <= spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    return spans


def run(t: pd.DataFrame, occ_frames: set[int], mode: str) -> dict:
    """Chay mot bo loc suot doi track. Chi `update` o frame KHONG bi che.

    Tra ve dict co:
        pos[frame]   -> (cx, cy, w, h) du doan
        state[frame] -> vector trang thai day du (de doc v, theta, omega)
        cov[frame]   -> ma tran hiep phuong sai
    """
    if mode == "KF":
        f = KalmanFilterXYAH()
        is_ctrv = False
    elif mode == "EKFC":
        f = EKFTrackerCTRVClamped()
        is_ctrv = True
    else:
        f = EKFTrackerCTRV()
        is_ctrv = True

    rows = list(t.itertuples())
    m, c = f.initiate(xyah(rows[0]))
    seeded = False
    last_obs = int(rows[0].frame)
    prev_vis = int(rows[0].frame) not in occ_frames

    def unpack(mean):
        if is_ctrv:
            return mean[f.CX], mean[f.CY], mean[f.A] * mean[f.H], mean[f.H]
        return mean[0], mean[1], mean[2] * mean[3], mean[3]

    pos = {int(rows[0].frame): unpack(m)}
    state = {int(rows[0].frame): m.copy()}
    cov = {int(rows[0].frame): c.copy()}

    for r in rows[1:]:
        fr = int(r.frame)
        vis = fr not in occ_frames

        if is_ctrv and not seeded and vis:
            # Khoang cach THAT giua hai quan sat (xem GIAI DOAN 0).
            m, c = f.initiate_from_motion(m, c, xyah(r), n_frames=max(1, fr - last_obs))
            seeded = True

        # --- TRIET TIEU OMEGA ngay tai thoi diem chuyen sang ngoai suy mu ---
        # predict() giu omega' = omega, va khong co update trong doan che,
        # nen dat mot lan la no giu 0 suot doan.
        if mode.startswith("EKFS") and prev_vis and not vis:
            trigger = True
            if mode == "EKFS_cond":
                last_row = t[t.frame == last_obs]
                occ_last = float(last_row.occlusion_ratio.iloc[0]) if len(last_row) else 0.0
                trigger = occ_last > OCC_THRESHOLD
            if trigger:
                m = m.copy()
                m[f.OMEGA] = 0.0

        m, c = f.predict(m, c)
        if vis:
            m, c = f.update(m, c, xyah(r))
            last_obs = fr
        prev_vis = vis
        pos[fr] = unpack(m)
        state[fr] = m.copy()
        cov[fr] = c.copy()

    return {"pos": pos, "state": state, "cov": cov, "filter": f, "is_ctrv": is_ctrv}


def draw_panel(im, title, gt_row, pred, trail_gt, trail_pr, in_occ, err, note):
    p = im.copy()
    H, W = p.shape[:2]
    col = COL_BLIND if in_occ else COL_OK
    if gt_row is not None:
        x, y, w, h = gt_row.bb_left, gt_row.bb_top, gt_row.bb_width, gt_row.bb_height
        cv2.rectangle(p, (int(x), int(y)), (int(x + w), int(y + h)), COL_GT, 2)
    cx, cy, w, h = pred
    cv2.rectangle(p, (int(cx - w / 2), int(cy - h / 2)),
                  (int(cx + w / 2), int(cy + h / 2)), col, 3)
    for tr, tc in ((trail_gt, COL_GT), (trail_pr, col)):
        pts = tr[-TRAIL:]
        for j in range(1, len(pts)):
            cv2.line(p, pts[j - 1], pts[j], tc, 2)
    if gt_row is not None and in_occ:
        cv2.line(p, (int(gt_row.cx), int(gt_row.cy)), (int(cx), int(cy)),
                 COL_BLIND, 1, cv2.LINE_AA)
    cv2.rectangle(p, (0, 0), (W, 56), (0, 0, 0), -1)
    cv2.putText(p, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)
    txt = "ngoai suy mu" if in_occ else "dang nap quan sat"
    if err is not None:
        txt += f"   sai so {err:6.1f} px"
    cv2.putText(p, txt, (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    if note:
        cv2.putText(p, note, (8, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)
    if in_occ:
        cv2.rectangle(p, (0, 0), (W - 1, H - 1), COL_OCC, 5)
    return p


def render(t, spans, occ_frames, res, out_path, panels):
    img_root = config.find_images_root()
    frames = [int(x) for x in t.frame]
    gt_rows = {int(r.frame): r for r in t.itertuples()}
    first = cv2.imread(str(img_root / VIDEO / f"img{frames[0]:05d}.jpg"))
    H, W = first.shape[:2]
    n = len(panels)
    out_w = W * n + PANEL_GAP * (n - 1)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         FPS, (out_w, H + 34))
    trails = {k: [] for k in panels}
    trail_gt = []
    fde_done = {k: [] for k in panels}
    seg_of = {f: i for i, (s, e) in enumerate(spans) for f in range(s, e + 1)}

    for fr in frames:
        im = cv2.imread(str(img_root / VIDEO / f"img{fr:05d}.jpg"))
        if im is None:
            continue
        g = gt_rows.get(fr)
        in_occ = fr in occ_frames
        trail_gt.append((int(g.cx), int(g.cy)))
        cells = []
        for k in panels:
            cx, cy, w, h = res[k]["pos"][fr]
            trails[k].append((int(cx), int(cy)))
            err = float(np.hypot(cx - g.cx, cy - g.cy)) if in_occ else None
            if in_occ and fr == spans[seg_of[fr]][1]:
                fde_done[k].append((seg_of[fr] + 1, float(np.hypot(cx - g.cx, cy - g.cy))))
            note = "   ".join(f"doan {i}: {v:.0f}px" for i, v in fde_done[k])
            cells.append(draw_panel(im, LABEL[k], g, (cx, cy, w, h),
                                    trail_gt, trails[k], in_occ, err, note))
        gap = np.full((H, PANEL_GAP, 3), 255, np.uint8)
        row = cells[0]
        for cpanel in cells[1:]:
            row = np.hstack([row, gap, cpanel])
        combo = np.vstack([row, np.full((34, row.shape[1], 3), 25, np.uint8)])
        Wc = combo.shape[1]
        span = max(1, frames[-1] - frames[0])
        for s, e in spans:
            xa = int((s - frames[0]) / span * (Wc - 1))
            xb = int((e - frames[0]) / span * (Wc - 1))
            cv2.rectangle(combo, (xa, H + 8), (max(xb, xa + 2), H + 26), COL_OCC, -1)
        xc = int((fr - frames[0]) / span * (Wc - 1))
        cv2.rectangle(combo, (xc - 1, H + 4), (xc + 1, H + 30), (255, 255, 255), -1)
        cv2.putText(combo, f"frame {fr}   vang = dang bi che", (8, H + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        vw.write(combo)
    vw.release()


def main() -> int:
    ap = argparse.ArgumentParser(description="Giai doan 3: triet tieu omega")
    ap.add_argument("--render", action="store_true", help="Xuat video 3 panel")
    a = ap.parse_args()

    gt = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    t = gt[(gt.video == VIDEO) & (gt.track_id == TRACK)].sort_values("frame").reset_index(drop=True)
    occ = pd.concat([pd.read_csv(config.INTERIM_DIR / f)
                     for f in ("full_occlusion_segments.csv", "occlusion_segments.csv")],
                    ignore_index=True)
    occ = occ[(occ.video == VIDEO) & (occ.track_id == TRACK)]
    spans = merge_spans(occ)
    occ_frames = {f for s, e in spans for f in range(s, e + 1)}

    out = config.RESULTS_DIR / "demo" / "diagnosis"
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print(f"GIAI DOAN 3 -- TRIET TIEU OMEGA -- {VIDEO} track {TRACK}")
    print("=" * 88)
    print(f"Doan che (da gop): " + ", ".join(f"{s}-{e} ({e-s+1}f)" for s, e in spans))

    res = {k: run(t, occ_frames, k) for k in MODES}

    # ---------------- BUOC 2: so lieu tai frame neo ----------------
    print("\n" + "=" * 88)
    print("BUOC 2 -- SO LIEU TAI FRAME CUOI TRUOC KHI MAT DAU (frame neo)")
    print("=" * 88)
    ekf = res["EKF"]["filter"]
    anchor_rows = []
    for i, (s, e) in enumerate(spans, 1):
        anchor = s - 1
        if anchor not in res["EKF"]["state"]:
            print(f"\n  Doan {i} (che {s}-{e}): khong co frame neo {anchor} trong track")
            continue
        x = res["EKF"]["state"][anchor]
        P = res["EKF"]["cov"][anchor]
        gr = t[t.frame == anchor]
        occ_r = float(gr.occlusion_ratio.iloc[0]) if len(gr) else np.nan
        om = float(x[ekf.OMEGA])
        var_om = float(P[ekf.OMEGA, ekf.OMEGA])
        v = float(x[ekf.V])
        th = float(x[ekf.THETA])
        print(f"\n  --- Doan {i}: che frame {s}-{e} ({e-s+1} frame), neo tai frame {anchor} ---")
        print(f"      omega_hat              = {om:+.6f} rad/frame  = {np.rad2deg(om):+9.3f} do/frame")
        print(f"      Var(omega) trong P     = {var_om:.6e}  -> do lech chuan "
              f"{np.sqrt(var_om):.4f} rad/f = {np.rad2deg(np.sqrt(var_om)):.2f} do/f")
        print(f"      |omega| / do lech chuan= {abs(om)/max(np.sqrt(var_om),1e-12):.2f}  "
              f"(nho hon 1 nghia la omega khong phan biet duoc voi 0)")
        print(f"      occlusion_ratio        = {occ_r:.3f}   -> bien the co dieu kien "
              f"{'KICH HOAT' if occ_r > OCC_THRESHOLD else 'KHONG kich hoat'} (nguong {OCC_THRESHOLD})")
        print(f"      v = {v:.3f} px/frame,  theta = {np.rad2deg(th):+.2f} do")
        print(f"      R = v/|omega|          = {abs(v/om) if abs(om)>1e-9 else float('inf'):.0f} px")
        print(f"      cung quet du kien      = {np.rad2deg(abs(om)*(e-s+1)):.1f} do")
        anchor_rows.append(dict(doan=i, anchor=anchor, omega=om, var_omega=var_om,
                                occlusion_ratio=occ_r, v=v, theta=th))
    pd.DataFrame(anchor_rows).to_csv(out / "phase3_anchors.csv", index=False)

    # ---------------- BUOC 3: SANITY CHECK ----------------
    print("\n" + "=" * 88)
    print("BUOC 3 -- SANITY CHECK: omega = 0 thi CTRV co suy bien ve CV khong?")
    print("=" * 88)
    print("\n[3a] So VECTO VAN TOC tai frame neo: CV dung (vx, vy); CTRV suy ra "
          "(v*cos(theta), v*sin(theta))")
    vel_rows = []
    for i, (s, e) in enumerate(spans, 1):
        anchor = s - 1
        if anchor not in res["KF"]["state"]:
            continue
        xk = res["KF"]["state"][anchor]
        xs = res["EKFS_always"]["state"][anchor]
        vx_kf, vy_kf = float(xk[4]), float(xk[5])
        v, th = float(xs[ekf.V]), float(xs[ekf.THETA])
        vx_ce, vy_ce = v * np.cos(th), v * np.sin(th)
        d = float(np.hypot(vx_ce - vx_kf, vy_ce - vy_kf))
        print(f"\n  Doan {i} (neo frame {anchor}):")
        print(f"     KF+CV        : vx = {vx_kf:+8.3f}, vy = {vy_kf:+8.3f}   "
              f"|v| = {np.hypot(vx_kf, vy_kf):7.3f} px/f")
        print(f"     CTRV suy ra  : vx = {vx_ce:+8.3f}, vy = {vy_ce:+8.3f}   "
              f"|v| = {v:7.3f} px/f")
        print(f"     Lech vecto van toc = {d:.3f} px/frame"
              f"   -> sau {e-s+1} frame ngoai suy tich luy ~{d*(e-s+1):.0f} px")
        vel_rows.append(dict(doan=i, vx_kf=vx_kf, vy_kf=vy_kf, vx_ctrv=vx_ce,
                             vy_ctrv=vy_ce, lech=d, T=e - s + 1))
    pd.DataFrame(vel_rows).to_csv(out / "phase3_velocity.csv", index=False)

    print("\n[3b] So NGOAI SUY (a, h) -- phan bbox scale")
    for i, (s, e) in enumerate(spans, 1):
        if e not in res["KF"]["state"]:
            continue
        xk, xs = res["KF"]["state"][e], res["EKFS_always"]["state"][e]
        gr = t[t.frame == e].iloc[0]
        print(f"  Doan {i} cuoi (frame {e}): GT h = {gr.bb_height:6.1f} | "
              f"KF h = {xk[3]:6.1f} | CTRV triet w h = {xs[ekf.H]:6.1f}   "
              f"(lech {abs(xk[3]-xs[ekf.H]):.2f} px)")

    print("\n[3c] SAI SO CUOI MOI DOAN CHE (FDE) -- day la phep so quyet dinh")
    fde = {k: {} for k in MODES}
    for i, (s, e) in enumerate(spans, 1):
        gr = t[t.frame == e]
        if not len(gr) or e not in res["KF"]["pos"]:
            continue
        gr = gr.iloc[0]
        for k in MODES:
            cx, cy, _, _ = res[k]["pos"][e]
            fde[k][i] = float(np.hypot(cx - gr.cx, cy - gr.cy))
    print(f"\n  {'doan':<7}{'che':<14}" + "".join(f"{LABEL[k]:>28}" for k in ["KF", "EKFS_always"])
          + f"{'chenh lech':>14}")
    print("  " + "-" * 78)
    verdicts = []
    for i, (s, e) in enumerate(spans, 1):
        if i not in fde["KF"]:
            continue
        a_, b_ = fde["KF"][i], fde["EKFS_always"][i]
        diff = b_ - a_
        rel = abs(diff) / max(a_, 1e-9) * 100
        verdicts.append((i, a_, b_, diff, rel))
        print(f"  {i:<7}{f'{s}-{e}':<14}{a_:>26.1f}px{b_:>26.1f}px{diff:>+12.1f}px")
    print()
    for i, a_, b_, diff, rel in verdicts:
        tag = "GAN BANG" if rel < 15 else ("LECH VUA" if rel < 50 else "LECH XA")
        print(f"    Doan {i}: chenh {rel:5.1f}% so voi KF+CV  -> {tag}")

    ok = all(r < 15 for _, _, _, _, r in verdicts)
    print("\n  " + "=" * 76)
    if ok:
        print("  KET LUAN SANITY CHECK: suppressed ~ KF+CV tren MOI doan.")
        print("  => Xac nhan omega la nguyen nhan duy nhat cua chenh lech.")
    else:
        print("  KET LUAN SANITY CHECK: suppressed VAN LECH XA KF+CV du omega = 0.")
        print("  => CON NGUYEN NHAN KHAC ngoai omega. Xem [3a] va [3d] o duoi.")
    print("  " + "=" * 76)

    # ---------------- [3d] DOI CHUNG TACH BIEN: omega=0 VA van toc dung ----------------
    # Neu chenh lech con lai la do VAN TOC chu khong phai omega, thi khi ep them
    # van toc bang dung van toc cua KF, CTRV phai suy bien TRUNG KHIT ve CV.
    print("\n[3d] DOI CHUNG TACH BIEN -- ep omega = 0 VA van toc = van toc cua KF")
    print("     (chi can thiep tai DUNG lan vao doan che dang xet, khong dung doan khac)")

    def run_forced(seg_idx: int, force_v: float | None):
        """omega = 0 (va tuy chon ep v) tai dung frame bat dau doan seg_idx."""
        s0 = spans[seg_idx][0]
        f2 = EKFTrackerCTRV()
        rows2 = list(t.itertuples())
        m2, c2 = f2.initiate(xyah(rows2[0]))
        sd, lo = False, int(rows2[0].frame)
        end = None
        for r in rows2[1:]:
            fr2 = int(r.frame)
            v2 = fr2 not in occ_frames
            if not sd and v2:
                m2, c2 = f2.initiate_from_motion(m2, c2, xyah(r), n_frames=max(1, fr2 - lo))
                sd = True
            if fr2 == s0:
                m2 = m2.copy()
                m2[f2.OMEGA] = 0.0
                if force_v is not None:
                    m2[f2.V] = force_v
            m2, c2 = f2.predict(m2, c2)
            if v2:
                m2, c2 = f2.update(m2, c2, xyah(r))
                lo = fr2
            if fr2 == spans[seg_idx][1]:
                end = (float(m2[f2.CX]), float(m2[f2.CY]))
        return end

    print(f"\n  {'doan':<7}{'KF + CV':>12}{'w=0, v uoc luong':>20}{'w=0, v = v cua KF':>21}"
          f"{'|chenh voi KF|':>17}")
    print("  " + "-" * 77)
    ctrl_rows = []
    for i, (s, e) in enumerate(spans, 1):
        gr = t[t.frame == e]
        if not len(gr):
            continue
        gr = gr.iloc[0]
        xk = res["KF"]["state"][s - 1] if (s - 1) in res["KF"]["state"] else None
        if xk is None:
            continue
        v_kf = float(np.hypot(xk[4], xk[5]))
        p_est = run_forced(i - 1, None)
        p_tru = run_forced(i - 1, v_kf)
        fde_kf = fde["KF"][i]
        fde_est = float(np.hypot(p_est[0] - gr.cx, p_est[1] - gr.cy))
        fde_tru = float(np.hypot(p_tru[0] - gr.cx, p_tru[1] - gr.cy))
        print(f"  {i:<7}{fde_kf:>10.1f}px{fde_est:>18.1f}px{fde_tru:>19.1f}px"
              f"{abs(fde_tru - fde_kf):>15.1f}px")
        ctrl_rows.append(dict(doan=i, v_kf=v_kf, fde_kf=fde_kf,
                              fde_omega0_vest=fde_est, fde_omega0_vkf=fde_tru,
                              lech_voi_kf=abs(fde_tru - fde_kf)))
    pd.DataFrame(ctrl_rows).to_csv(out / "phase3_control.csv", index=False)
    worst = max((r["lech_voi_kf"] for r in ctrl_rows), default=0.0)
    print(f"\n  Lech lon nhat so voi KF khi ep ca omega=0 va v dung: {worst:.1f} px")
    if worst < 25:
        print("  => CTRV suy bien DUNG ve CV khi omega=0 VA van toc khop.")
        print("     Vay chenh lech quan sat duoc KHONG phai do omega, ma do UOC LUONG VAN TOC.")
    else:
        print("  => Van con lech dang ke -> con nguyen nhan thu ba ngoai omega va van toc.")

    # ---------------- BUOC 5: bang so day du ----------------
    print("\n" + "=" * 88)
    print("BUOC 5 -- BANG SO SANH DAY DU (FDE tai cuoi moi doan che)")
    print("=" * 88)
    print(f"\n  {'doan':<7}{'che':<14}" + "".join(f"{LABEL[k]:>28}" for k in MODES))
    print("  " + "-" * 145)
    rows = []
    for i, (s, e) in enumerate(spans, 1):
        if i not in fde["KF"]:
            continue
        line = f"  {i:<7}{f'{s}-{e}':<14}"
        rec = dict(doan=i, start=s, end=e, T=e - s + 1)
        for k in MODES:
            line += f"{fde[k][i]:>26.1f}px"
            rec[k] = fde[k][i]
        rows.append(rec)
        print(line)
    df = pd.DataFrame(rows)
    if len(df):
        line = f"  {'TB':<7}{'':<14}" + "".join(f"{df[k].mean():>26.1f}px" for k in MODES)
        print(line)
    df.to_csv(out / "phase3_fde.csv", index=False)

    # ---------------- BUOC 4: video ----------------
    if a.render:
        panels = ["KF", "EKF", "EKFS_always"]
        vids = config.RESULTS_DIR / "demo" / "videos"
        vids.mkdir(parents=True, exist_ok=True)
        p = vids / f"suppress3_{VIDEO}_t{TRACK}.mp4"
        print(f"\n[BUOC 4] Render video 3 panel: {' | '.join(LABEL[k] for k in panels)}")
        render(t, spans, occ_frames, res, p, panels)
        print(f"  -> {p}")

    print(f"\n  So lieu da luu -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
