"""
dung_video_so_sanh.py -- (R63, theo yeu cau nguoi dung 02/11) video so sanh TRUOC / SAU toi uu Nano (v16 640x640 vs C1
384x640) tren 300 frame MVI_39031, dung DU LIEU THAT cua cap do so 1 (CV) trong `g3/c1_so_cap_01`.

Nguon: cache detection do CHINH Nano ghi (`g3/nano_cache_300/cap1_cv_{A,B}`), tracker CV phat lai tren PC voi backend
`np` nhu Nano -> MOT phai TRUNG BYTE hash Nano trong `result.json` (A 9df71094..., B 62c4596e...; so sau khi bo CR) thi
moi ve. Anh nguon `data/extracted/DETRAC-Images/MVI_39031/img00001-00300.jpg` kiem sha256 theo MANIFEST goi Nano.
Toc do phat phan 1 = FPS xu ly DO TREN NANO (`result.json` fps) -> mo phong tien do xu ly, KHONG phai do lai tren PC.
Video duoc ve tren PC (OpenCV/PIL cua PC), khong phai quay man hinh Nano.

    python -B handoff/logs/review63/cong_cu/dung_video_so_sanh.py <thu_muc_ra>
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

os.environ["BYTETRACK_BACKEND"] = "np"
GOC = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GOC / "src"))
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import det_cache  # noqa: E402
import run_tracking as rt  # noqa: E402

R63 = GOC / "handoff" / "logs" / "review63" / "g3"
ANH = GOC / "data" / "extracted" / "DETRAC-Images" / "MVI_39031"
MANIFEST = GOC / "handoff" / "logs" / "review20" / "sao_luu_nano" / "goi_v5" / "MANIFEST_SHA256.txt"
FONT, FONT_B = "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"
FPS_RA = 25
W, H = 960, 540
NHANH = {"A": {"ten": "TRƯỚC TỐI ƯU", "phu": "v16 · đầu vào mạng 640×640 (43,75% là viền đệm)", "mau": (70, 90, 230)},
         "B": {"ten": "SAU TỐI ƯU", "phu": "C1 · đầu vào mạng 384×640 (bớt 40% pixel)", "mau": (30, 170, 60)}}


def kiem_anh() -> None:
    m = {}
    for dong in MANIFEST.read_text(encoding="utf-8").splitlines():
        h, p = dong.split(None, 1)
        if "MVI_39031_f0001-0300/img" in p:
            m[p.rsplit("/", 1)[1]] = h
    lech = [f for f, h in m.items() if hashlib.sha256((ANH / f).read_bytes()).hexdigest() != h]
    if len(m) != 300 or lech:
        raise SystemExit(f"anh nguon lech manifest Nano: {len(m)} anh, lech {lech[:3]}")


def phat_lai(nhanh: str, tmp: Path):
    kq = json.loads((R63 / "c1_so_cap_01" / f"cap1_cv_{nhanh}" / "result.json").read_text(encoding="utf-8"))
    goc = tmp / f"cache_{nhanh}"
    shutil.copytree(R63 / "nano_cache_300" / f"cap1_cv_{nhanh}", goc / "detections")
    det_cache.CACHE_ROOT = goc
    cfg = rt.load_tracker_cfg("bytetrack.yaml", {"track_buffer": rt.track_buffer_frames(1.2, 25.0)})
    vd = det_cache.open_cache("detections", "MVI_39031", 0.25, [1, 2, 3, 5, 7], 300, None)
    df, _, _ = rt.track_video(vd, cfg, 1 / 25.0, "steps", None, "MVI_39031")
    h = hashlib.sha256(df.to_csv(header=False, index=False, lineterminator="\n").encode()).hexdigest()
    if h != kq["hashes"]["data/MVI_39031.txt"]:
        raise SystemExit(f"{nhanh}: MOT phat lai {h[:12]} != Nano {kq['hashes']['data/MVI_39031.txt'][:12]}")
    print(f"{nhanh}: MOT trung byte Nano {h[:12]}, FPS Nano {kq['fps']:.3f}")
    return df, kq


def mau_id(i: int) -> tuple:
    r = np.random.default_rng(i * 7919)
    return tuple(int(x) for x in r.integers(60, 255, 3))


def ve_khung(img: np.ndarray, boxes) -> np.ndarray:
    img = img.copy()
    for _, b in boxes.iterrows():
        c = mau_id(int(b.id))
        p1, p2 = (int(b.bb_left), int(b.bb_top)), (int(b.bb_left + b.bb_width), int(b.bb_top + b.bb_height))
        cv2.rectangle(img, p1, p2, c, 2)
        cv2.putText(img, f"ID {int(b.id)}", (p1[0], max(p1[1] - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    return img


def main() -> int:
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    kiem_anh()
    tmp = Path(tempfile.mkdtemp())
    data = {n: phat_lai(n, tmp) for n in NHANH}
    anh = {f: cv2.imread(str(ANH / f"img{f:05d}.jpg")) for f in range(1, 301)}
    f_to, f_vua, f_nho = (ImageFont.truetype(FONT_B, 30), ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 17))
    f_so = ImageFont.truetype(FONT_B, 40)
    G = 8                                   # khe giua hai khung
    TW, TH, BH = 2 * W + G, 92, 120         # rong, cao header, cao footer
    VH = TH + H + BH
    p = out_dir / "so_sanh_truoc_sau_toi_uu_nano.mp4"
    vw = None
    for cc in ("avc1", "mp4v"):
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*cc), FPS_RA, (TW, VH))
        if vw.isOpened():
            print("codec", cc); break
    fps = {n: data[n][1]["fps"] for n in NHANH}
    tg = {n: data[n][1]["time"]["xu_ly_s"] for n in NHANH}
    d_by = {n: {f: d for f, d in data[n][0].groupby("frame")} for n in NHANH}
    import pandas as pd
    rong = pd.DataFrame(columns=rt.TRK_COLS)

    def khung(phan: str, t: float | None, frame_cung: int | None) -> np.ndarray:
        nen = Image.new("RGB", (TW, VH), (24, 24, 28))
        dr = ImageDraw.Draw(nen)
        for k, n in enumerate(NHANH):
            x0 = k * (W + G)
            if frame_cung is None:
                f = min(300, 1 + int(t * fps[n]))
                xong = t * fps[n] >= 300
            else:
                f, xong = frame_cung, False
            img = ve_khung(anh[f], d_by[n].get(f, rong))
            nen.paste(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), (x0, TH))
            mau = NHANH[n]["mau"]
            dr.rectangle([x0, 0, x0 + W, TH - 4], fill=mau)
            dr.text((x0 + 16, 8), NHANH[n]["ten"], font=f_to, fill="white")
            dr.text((x0 + 16, 52), NHANH[n]["phu"], font=f_vua, fill="white")
            # o so FPS + tien do (trong khung anh)
            dr.rounded_rectangle([x0 + 12, TH + 12, x0 + 300, TH + 118], 10, fill=(0, 0, 0))
            dr.text((x0 + 24, TH + 14), f"{fps[n]:.2f} FPS".replace(".", ","), font=f_so, fill=mau if n == "B" else (140, 160, 255))
            if frame_cung is None:
                tt = min(t, tg[n] * 300 / (fps[n] * tg[n])) if not xong else 300 / fps[n]
                dr.text((x0 + 24, TH + 66), f"frame {f}/300 · {tt:4.1f} s".replace(".", ","), font=f_vua, fill="white")
                dr.rectangle([x0, TH + H - 10, x0 + int(W * f / 300), TH + H], fill=mau)
                if xong:
                    dr.rounded_rectangle([x0 + W // 2 - 190, TH + H // 2 - 40, x0 + W // 2 + 190, TH + H // 2 + 40], 14,
                                         fill=mau)
                    dr.text((x0 + W // 2 - 170, TH + H // 2 - 26), f"XONG 300 frame: {300 / fps[n]:.1f} s".replace(".", ","),
                            font=f_to, fill="white")
            else:
                dr.text((x0 + 24, TH + 66), f"frame {f}/300 (cùng frame)", font=f_vua, fill="white")
        y = TH + H + 10
        if phan == "dua":
            dr.text((16, y), "Phần 1 — tốc độ: mỗi bên chạy đúng tốc độ xử lý đo trên Jetson Nano (cùng 300 frame, CV, hiển thị HDMI bật)",
                    font=f_vua, fill="white")
        else:
            dr.text((16, y), "Phần 2 — kết quả: phát cùng frame cạnh nhau ở 25 FPS để so khung phát hiện / ID", font=f_vua, fill="white")
        dr.text((16, y + 32), "FPS trung vị 8 cặp đo: CV 9,27 → 13,24 · EKF 9,23 → 13,20 (≈ +43%, chỉ đúng cho Nano này) · "
                              "trễ p90 −27% · chất lượng UA-DETRAC test40 (c25): ΔHOTA +0,09 / +0,20, ΔmAP50 −0,17, recall xe nhỏ −0,60",
                font=f_nho, fill=(220, 220, 220))
        dr.text((16, y + 58), "Video dựng trên PC từ detection do Nano ghi trong lượt đo R63 (cặp 1, CV, MVI_39031 UA-DETRAC); "
                              "tracker phát lại trùng byte output Nano. Không phải quay màn hình Nano.",
                font=f_nho, fill=(170, 170, 170))
        return cv2.cvtColor(np.asarray(nen), cv2.COLOR_RGB2BGR)

    t_max = 300 / min(fps.values()) + 2.0
    n1 = int(t_max * FPS_RA)
    for i in range(n1):
        vw.write(khung("dua", i / FPS_RA, None))
    for f in range(1, 301):
        vw.write(khung("cung", None, f))
    vw.release()
    shutil.rmtree(tmp, ignore_errors=True)
    print("ghi", p, f"{p.stat().st_size / 1e6:.1f} MB", f"{(n1 + 300) / FPS_RA:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
