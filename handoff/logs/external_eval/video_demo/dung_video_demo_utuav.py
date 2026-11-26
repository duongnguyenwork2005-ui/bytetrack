"""
dung_video_demo_utuav.py -- (yeu cau nguoi dung 02/11) video demo PILOT UTUAV luoi 2x2: CV / EKF / UKF / PF100 (seed 0).

KHONG phai ket qua YOLO: ve tu output pilot 01/11 (`handoff/logs/external_mot_prep/02_pilot_utuav.md`), kich ban
`gapv2` = detection TONG HOP TU GT (nhieu 3 %, bo sot ngau nhien 15 %, them 8 frame mat lien tiep o mot nua track dong),
FPS 30 GIA DINH (chua xac minh). Hai doan validation 120 frame: B_straight_26401_26520, C_turn_20901_21020.
Anh 4K, xe rat nho -> moi o cat vung 1920x1080 quanh trong tam GT dong (class 1) cua ca doan (tinh tu GT, khong tu ket
qua tracker), thu 1/2. Phat cham 10 fps.

    python -B handoff/logs/external_eval/video_demo/dung_video_demo_utuav.py <thu_muc_ra>
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

GOC = Path(__file__).resolve().parents[4]
FONT, FONT_B = "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"
TRK = GOC / "data" / "processed" / "trackers" / "UTUAV-pilot"
TE = GOC / "results" / "trackeval" / "UTUAV-pilot"
GT = GOC / "data" / "processed" / "UTUAV-pilot"
ANH = GOC / "data" / "extracted" / "utuav_pilot" / "frames"
LOC = [("utuav_gapv2_cv", "CV (Kalman, vận tốc không đổi)"), ("utuav_gapv2_ekf", "EKF (CTRV)"),
       ("utuav_gapv2_ukf", "UKF (CTRV)"), ("utuav_gapv2_pf100-s0", "PF 100 hạt (CTRV, seed 0)")]
SEQ = [("B_straight_26401_26520", "B — đi thẳng"), ("C_turn_20901_21020", "C — có xe rẽ")]
CW, CH = 1920, 1080          # vung cat tren anh 4K
PW, PH = 960, 540
TH, BH = 92, 64
FPS_RA = 10


def mau_id(i: int) -> tuple:
    r = np.random.default_rng(int(i) * 7919 + 17)
    return tuple(int(x) for x in r.integers(70, 256, 3))


def main() -> int:
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    fr, fb, fn = ImageFont.truetype(FONT_B, 24), ImageFont.truetype(FONT_B, 22), ImageFont.truetype(FONT, 17)
    W, H = 2 * PW, TH + 2 * PH + BH
    tho = out_dir / "_tho_utuav.mp4"
    vw = cv2.VideoWriter(str(tho), cv2.VideoWriter_fourcc(*"mp4v"), FPS_RA, (W, H))
    for seq, nhan_seq in SEQ:
        g = pd.read_csv(GT / seq / "gt" / "gt.txt", header=None, names=["f", "id", "x", "y", "w", "h", "c", "cls", "vis"])
        g = g[g["cls"] == 1]
        cx, cy = float((g["x"] + g["w"] / 2).median()), float((g["y"] + g["h"] / 2).median())
        x0c = int(np.clip(cx - CW / 2, 0, 3840 - CW)); y0c = int(np.clip(cy - CH / 2, 0, 2160 - CH))
        runs, met = {}, {}
        for loc, _ in LOC:
            m = pd.read_csv(TRK / loc / "data" / f"{seq}.txt", header=None, names=["f", "id", "x", "y", "w", "h", "conf"])
            runs[loc] = {f: d for f, d in m.groupby("f")}
            pv = pd.read_csv(TE / loc / "per_video_metrics.csv")
            r = pv[pv["video"] == seq].iloc[0]
            met[loc] = (float(r["HOTA"]) * 100, int(r["IDSW"]))
        anh = sorted((ANH / seq).glob("*.jpg"))
        assert len(anh) == 120, (seq, len(anh))
        da_thay = {loc: set() for loc, _ in LOC}
        for f, p in enumerate(anh, 1):
            im = cv2.imread(str(p))[y0c:y0c + CH, x0c:x0c + CW]
            nho = cv2.resize(im, (PW, PH), interpolation=cv2.INTER_AREA)
            nen = np.zeros((H, W, 3), np.uint8); nen[:] = (28, 24, 24)
            for k, (loc, _) in enumerate(LOC):
                o = nho.copy()
                d = runs[loc].get(f)
                if d is not None:
                    for b in d.itertuples(index=False):
                        da_thay[loc].add(int(b.id))
                        p1 = (int((b.x - x0c) / 2), int((b.y - y0c) / 2))
                        p2 = (int((b.x + b.w - x0c) / 2), int((b.y + b.h - y0c) / 2))
                        if p2[0] < 0 or p2[1] < 0 or p1[0] > PW or p1[1] > PH:
                            continue
                        cm = mau_id(b.id)
                        cv2.rectangle(o, p1, p2, cm, 2)
                        cv2.putText(o, str(int(b.id)), (p1[0], max(p1[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, cm, 1)
                xo, yo = (k % 2) * PW, TH + (k // 2) * PH
                nen[yo:yo + PH, xo:xo + PW] = o
            img = Image.fromarray(cv2.cvtColor(nen, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(img)
            dr.text((14, 4), f"UTUAV PILOT · đoạn {nhan_seq} ({seq}) · frame {f}/120", font=fr, fill="white")
            dr.text((14, 36), "Detection TỔNG HỢP TỪ GT (nhiễu 3 %, bỏ sót 15 %, mất 8 frame liên tiếp ở một nửa xe) — "
                    "KHÔNG phải YOLO. FPS 30 GIẢ ĐỊNH, chưa xác minh.", font=fn, fill=(255, 180, 120))
            dr.text((14, 62), f"Ảnh drone 3840×2160, hiển thị vùng cắt {CW}×{CH} quanh nơi nhiều xe (tính từ GT), "
                    f"phát chậm {FPS_RA} fps.", font=fn, fill=(200, 200, 200))
            for k, (loc, nhan) in enumerate(LOC):
                xo, yo = (k % 2) * PW, TH + (k // 2) * PH
                hota, idsw = met[loc]
                dr.rounded_rectangle([xo + 8, yo + 8, xo + 470, yo + 86], 8, fill=(0, 0, 0))
                dr.text((xo + 18, yo + 10), nhan, font=fb, fill=(255, 220, 120))
                dr.text((xo + 18, yo + 44), f"HOTA đoạn {hota:.1f} · IDSW {idsw} · ID đã dùng {len(da_thay[loc])}"
                        .replace(".", ","), font=fn, fill="white")
                dr.rectangle([xo, yo, xo + PW - 1, yo + PH - 1], outline=(60, 60, 60), width=2)
            y = TH + 2 * PH + 6
            dr.text((14, y), "Pilot thăm dò 01/11 (handoff/logs/external_mot_prep/02_pilot_utuav.md), kịch bản gapv2. "
                    "HOTA/IDSW tính trên cả đoạn 120 frame (TrackEval).", font=fn, fill=(210, 210, 210))
            dr.text((14, y + 26), "Với YOLO11n mixed5 thật, pilot không ghép được xe nào (0/1.795) — vì vậy chưa có video "
                    "UTUAV từ detector thật.", font=fn, fill=(160, 160, 160))
            vw.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        print(seq, {k: v for k, v in met.items()}, "crop", (x0c, y0c), flush=True)
    vw.release()
    import imageio_ffmpeg
    ra = out_dir / "demo_utuav_pilot_gapv2_4_bo_loc.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tho), "-c:v", "libx264",
                    "-crf", "24", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(ra)],
                   check=True)
    tho.unlink()
    print("ghi", ra, f"{ra.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
