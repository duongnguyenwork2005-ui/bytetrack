"""
gt_omega.py -- DEMO Phan 1: tinh omega tu ground truth cho tung doan che khuat.

MUC DICH: cung cap omega_before / omega_during cho Phan 2 (phan loai Case A/B)
va Phan 3 (chon doan de render video).

    omega_before : toc do quay do tu GT trong cua so 10-15 frame NGAY TRUOC
                   khi bat dau bi che  -> thong tin ma tracker CO the biet
    omega_during : toc do quay do tu GT TRONG suot doan bi che
                   -> thong tin ma tracker KHONG the biet

Y nghia: neu khuc cua da hinh thanh TRUOC khi che (omega_before lon, cung dau
voi omega_during) thi CTRV co co so de ngoai suy dung. Neu khuc cua chi phat
sinh SAU khi mat quan sat thi khong mo hinh chuyen dong nao du doan duoc -
day la gioi han thong tin, khong phai gioi han thuat toan.

CACH TINH (giu don gian theo yeu cau demo):
    d      = hieu tam bbox giua cac frame lien tiep
    d_smooth = trung binh truot 3 diem (lam muot nhe, chong nhieu annotation)
    heading  = atan2(dy, dx)
    dtheta   = hieu heading lien tiep, GOI ve [-pi, pi)
    omega    = trung binh dtheta   [rad/frame]

MOT QUYET DINH KHONG CO TRONG YEU CAU NHUNG BAT BUOC PHAI CO:
Loc frame co toc do < MIN_SPEED. Ly do: khi xe gan nhu dung yen, huong di
chuyen khong xac dinh duoc va nhieu annotation vai pixel bi doc thanh "quay
rat nhanh". Da do o giai doan truoc: tren doan xe dung yen, MOI cach do do cong
deu bi thoi phong 5.9-7.2 lan. Neu bo buoc loc nay thi ti le Case A/B se sai
hoan toan.

CACH DUNG
    python src/demo/gt_omega.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# --- Tap video cua ban DEMO (chon bang tieu chi, khong chon tay) ---
# 8 video train co nhieu doan che khuat nhat trong khoang 10-120 frame.
DEMO_VIDEOS = ["MVI_40752", "MVI_63554", "MVI_63561", "MVI_40241",
               "MVI_63553", "MVI_63563", "MVI_40992", "MVI_63552"]

WIN_BEFORE = 15      # so frame toi da cua cua so truoc doan che
MIN_BEFORE = 10      # duoi nguong nay coi nhu khong du du lieu -> omega_before = NaN
MIN_DURING = 6       # doan che ngan hon thi omega_during khong dang tin
MIN_SPEED = 1.5      # px/frame - duoi nguong nay huong di chuyen vo nghia
SEG_MIN, SEG_MAX = 10, 120     # gioi han do dai doan che dua vao demo


def wrap(a):
    """Goi goc ve [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def omega_from_track(p: np.ndarray) -> float:
    """Toc do quay trung binh (rad/frame) tu chuoi tam bbox.

    Tra ve NaN neu khong du diem hoac xe gan nhu dung yen (huong khong xac dinh).
    """
    if len(p) < 4:
        return np.nan
    d = np.diff(p, axis=0)
    speed = np.hypot(d[:, 0], d[:, 1])
    if speed.mean() < MIN_SPEED:
        return np.nan
    # Lam muot nhe: trung binh truot 3 diem tren vector dich chuyen
    if len(d) >= 3:
        k = np.ones(3) / 3.0
        d = np.c_[np.convolve(d[:, 0], k, mode="valid"),
                  np.convolve(d[:, 1], k, mode="valid")]
    if len(d) < 2:
        return np.nan
    keep = np.hypot(d[:, 0], d[:, 1]) >= MIN_SPEED
    d = d[keep]
    if len(d) < 2:
        return np.nan
    heading = np.arctan2(d[:, 1], d[:, 0])
    dtheta = wrap(np.diff(heading))
    return float(np.mean(dtheta))


def load_segments(videos: list[str]) -> pd.DataFrame:
    """Tai dung cac doan che khuat DA PHAT HIEN SAN trong repo (khong tinh lai).

    Gop ca hai muc: che mot phan (>=0.10) va che gan hoan toan (>=0.90).
    """
    out = []
    for lvl, f in [("partial", "occlusion_segments.csv"),
                   ("full", "full_occlusion_segments.csv")]:
        d = pd.read_csv(config.INTERIM_DIR / f)
        d["lvl"] = lvl
        out.append(d)
    s = pd.concat(out, ignore_index=True)
    s = s[s.video.isin(videos)]
    return s[(s.length_frames >= SEG_MIN) & (s.length_frames <= SEG_MAX)].copy()


def main() -> int:
    pq = pd.read_parquet(config.INTERIM_DIR / "detrac_train_annotations.parquet")
    pq = pq[pq.video.isin(DEMO_VIDEOS)]
    g = {k: v.sort_values("frame") for k, v in pq.groupby(["video", "track_id"])}

    seg = load_segments(DEMO_VIDEOS)
    print(f"[demo] {len(DEMO_VIDEOS)} video, {len(seg)} doan che khuat "
          f"({SEG_MIN}-{SEG_MAX} frame)")

    rows = []
    for r in seg.itertuples(index=False):
        t = g.get((r.video, r.track_id))
        if t is None:
            continue
        before = t[(t.frame < r.start_frame) &
                   (t.frame >= r.start_frame - WIN_BEFORE)].sort_values("frame")
        during = t[(t.frame >= r.start_frame) &
                   (t.frame <= r.end_frame)].sort_values("frame")
        if len(before) < MIN_BEFORE or len(during) < MIN_DURING:
            continue
        rows.append(dict(
            video=r.video, track_id=int(r.track_id), seg_id=int(r.seg_id),
            lvl=r.lvl,
            frame_start=int(r.start_frame), frame_end=int(r.end_frame),
            T_occ=int(r.length_frames),
            omega_before=omega_from_track(np.c_[before.cx.values, before.cy.values]),
            omega_during=omega_from_track(np.c_[during.cx.values, during.cy.values]),
            occlusion_ratio_max=float(during.occlusion_ratio.max()),
        ))

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "demo"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "segments_omega.csv", index=False)

    ok = df.dropna(subset=["omega_before", "omega_during"])
    print(f"[demo] tinh duoc omega ca 2 phia: {len(ok)}/{len(df)} doan "
          f"({len(df) - len(ok)} doan bi loai do xe dung yen / thieu frame)")
    print(f"[demo] |omega_before| trung vi: {np.rad2deg(ok.omega_before.abs().median()):.4f} do/frame")
    print(f"[demo] |omega_during| trung vi: {np.rad2deg(ok.omega_during.abs().median()):.4f} do/frame")
    print(f"[demo] -> {out / 'segments_omega.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
