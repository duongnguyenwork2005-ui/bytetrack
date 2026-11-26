"""
dung_video_demo.py -- (R64, yeu cau nguoi dung 02/11) video demo luoi 2x2: CV / EKF / UKF / PF500 (seed 0) chay tren CUNG
mot chuoi, ve tu output MOT da cham cua R64 (khong chay lai tracker). Moi o: khung + ID theo mau, HOTA/IDSW cua chuoi
(per_video_metrics R64), so ID da dung tu dau chuoi (thay phan manh).

Quy tac chon (dat TRUOC khi xem video): chuoi DAU TIEN trong danh sach cua tap; detector co HOTA COMBINED (CV, c25) cao
nhat cua tap; nguong c25; PF500 seed 0. RoundaboutHD chi lay 900 frame dau (1 phut) cho gon.

    python -B handoff/logs/external_eval/video_demo/dung_video_demo.py <svmot|roundabout> <thu_muc_ra>
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
sys.path.insert(0, str(GOC / "src")); sys.path.insert(0, str(GOC / "src" / "r64_ngoai"))
import config  # noqa: E402
import danh_gia_ngoai as r64  # noqa: E402

FONT, FONT_B = "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"
LOC = [("cv", "CV (Kalman, vận tốc không đổi)"), ("ekf", "EKF (CTRV)"), ("ukf", "UKF (CTRV)"),
       ("pf500", "PF 500 hạt (CTRV, seed 0)")]
CHON = {"svmot": {"video": "seq_22_morning", "det": "yolov8n_coco", "n": None, "ten": "SVMOT"},
        "roundabout": {"video": "imagesc001", "det": "yolo11n_mixed5", "n": 900, "ten": "RoundaboutHD"}}
PW, PH = 960, 540            # kich thuoc moi o
TH, BH = 70, 64              # header / footer


def mau_id(i: int) -> tuple:
    r = np.random.default_rng(int(i) * 7919 + 17)
    return tuple(int(x) for x in r.integers(70, 256, 3))


def doc_frame(ds: str, t: dict, v: str, n: int):
    if ds == "svmot":
        for i, p in enumerate(r64.anh_svmot(t, v)[:n], 1):
            yield i, cv2.imread(str(p))
    else:
        cap = cv2.VideoCapture(str(t["nguon"] / v / "video.mp4"))
        for i in range(1, n + 1):
            ok, im = cap.read()
            if not ok:
                raise SystemExit(f"decode thieu frame {i}")
            yield i, im
        cap.release()


def main() -> int:
    ds, out_dir = sys.argv[1], Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
    c = CHON[ds]; t = r64.cau_hinh(ds, False); v = c["video"]
    n = c["n"] or r64.so_frame(t, v)
    H0, W0 = t["hw"]; sx, sy = PW / W0, PH / H0
    runs, met = {}, {}
    for loc, _ in LOC:
        ten = f"r64-{c['det']}-c25-{loc}" + ("-s0" if loc == "pf500" else "")
        mot = pd.read_csv(config.PROCESSED_DIR / "trackers" / t["split"] / ten / "data" / f"{v}.txt", header=None,
                          names=["f", "id", "x", "y", "w", "h", "conf"])
        runs[loc] = {f: d for f, d in mot.groupby("f")}
        pv = pd.read_csv(config.RESULTS_DIR / "trackeval" / t["split"] / ten / "per_video_metrics.csv")
        r = pv[pv["video"] == v].iloc[0]
        met[loc] = (float(r["HOTA"]) * 100, int(r["IDSW"]))
    fr, fb, fn = ImageFont.truetype(FONT_B, 24), ImageFont.truetype(FONT_B, 22), ImageFont.truetype(FONT, 17)
    W, H = 2 * PW, TH + 2 * PH + BH
    tho = out_dir / f"_tho_{ds}.mp4"
    vw = cv2.VideoWriter(str(tho), cv2.VideoWriter_fourcc(*"mp4v"), t["fps"], (W, H))
    da_thay = {loc: set() for loc, _ in LOC}
    for f, im in doc_frame(ds, t, v, n):
        nho = cv2.resize(im, (PW, PH), interpolation=cv2.INTER_AREA)
        nen = np.zeros((H, W, 3), np.uint8); nen[:] = (28, 24, 24)
        for k, (loc, _) in enumerate(LOC):
            o = nho.copy()
            d = runs[loc].get(f)
            if d is not None:
                for b in d.itertuples(index=False):
                    da_thay[loc].add(int(b.id))
                    cm = mau_id(b.id)
                    p1 = (int(b.x * sx), int(b.y * sy)); p2 = (int((b.x + b.w) * sx), int((b.y + b.h) * sy))
                    cv2.rectangle(o, p1, p2, cm, 2)
                    cv2.putText(o, str(int(b.id)), (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, cm, 2)
            x0, y0 = (k % 2) * PW, TH + (k // 2) * PH
            nen[y0:y0 + PH, x0:x0 + PW] = o
        img = Image.fromarray(cv2.cvtColor(nen, cv2.COLOR_BGR2RGB)); dr = ImageDraw.Draw(img)
        dr.text((14, 6), f"{c['ten']} · {v} · detector {c['det']} · ngưỡng c25 · cùng detection cho cả 4 bộ lọc",
                font=fr, fill="white")
        dr.text((14, 40), f"frame {f}/{n} · {t['fps']:.0f} fps · {W0}×{H0} thu nhỏ", font=fn, fill=(200, 200, 200))
        for k, (loc, nhan) in enumerate(LOC):
            x0, y0 = (k % 2) * PW, TH + (k // 2) * PH
            hota, idsw = met[loc]
            dr.rounded_rectangle([x0 + 8, y0 + 8, x0 + 470, y0 + 86], 8, fill=(0, 0, 0))
            dr.text((x0 + 18, y0 + 10), nhan, font=fb, fill=(255, 220, 120))
            dr.text((x0 + 18, y0 + 44), f"HOTA chuỗi {hota:.1f} · IDSW {idsw} · ID đã dùng {len(da_thay[loc])}"
                    .replace(".", ","), font=fn, fill="white")
            dr.rectangle([x0, y0, x0 + PW - 1, y0 + PH - 1], outline=(60, 60, 60), width=2)
        y = TH + 2 * PH + 6
        dr.text((14, y), "HOTA/IDSW tính trên cả chuỗi (TrackEval R64)"
                + (f", video chỉ hiển thị {n} frame đầu" if c["n"] else "")
                + ". Số trên khung = ID track; ID tăng nhanh → track bị cắt nhiều đoạn.", font=fn, fill=(210, 210, 210))
        dr.text((14, y + 26), "Dựng lại từ output MOT đã chấm của R64 (không chạy lại tracker). Chuỗi/detector chọn theo "
                "quy tắc định trước: chuỗi đầu tiên; detector HOTA COMBINED cao nhất.", font=fn, fill=(160, 160, 160))
        vw.write(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
        if f % 200 == 0:
            print(ds, f, "/", n, flush=True)
    vw.release()
    import imageio_ffmpeg
    ra = out_dir / f"demo_{ds}_{v}_4_bo_loc.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tho), "-c:v", "libx264",
                    "-crf", "24", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(ra)],
                   check=True)
    tho.unlink()
    print("ghi", ra, f"{ra.stat().st_size / 1e6:.1f} MB", {k: v_ for k, v_ in met.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
