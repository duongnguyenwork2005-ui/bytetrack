"""
side_by_side.py -- Ghep video song song: KF+CV (trai) vs EKF+CTRV (phai), tren
KET QUA TRACKER THAT (khong phai mo phong tu GT).

Khac voi render_videos.py (Phan 3 cua demo, nap GT roi CHI predict qua 1 doan
che de minh hoa co che), file nay ve THANG box+ID ma ByteTrack thuc su xuat ra
khi chay tren video, frame-by-frame, ca doan khong che lan doan bi che.

CACH DUNG
    python src/demo/side_by_side.py --video MVI_40131 --start 100 --end 400
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

TRK_COLS = ["frame", "id", "x", "y", "w", "h", "conf"]
FPS_OUT = 12
PANEL_GAP = 6


def color_for_id(tid: int) -> tuple[int, int, int]:
    """Mau on dinh theo track id (HSV -> BGR) de cung 1 ID luon cung 1 mau."""
    hue = (tid * 47) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_panel(frame: np.ndarray, frame_num: int, boxes: pd.DataFrame,
               title: str, in_occ: bool) -> np.ndarray:
    im = frame.copy()
    for r in boxes.itertuples():
        col = color_for_id(int(r.id))
        x1, y1, x2, y2 = int(r.x), int(r.y), int(r.x + r.w), int(r.y + r.h)
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
        cv2.putText(im, f"ID{int(r.id)}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    cv2.rectangle(im, (0, 0), (im.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(im, f"{title}  |  frame {frame_num}  |  {len(boxes)} track",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if in_occ:
        cv2.rectangle(im, (0, 0), (im.shape[1] - 1, im.shape[0] - 1), (0, 215, 255), 5)
    return im


def main() -> int:
    ap = argparse.ArgumentParser(description="Ghep video song song CV vs EKF+CTRV")
    ap.add_argument("--video", default="MVI_40131")
    ap.add_argument("--split-name", default="DETRAC-all")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--cv-tracker", default="yolov8n-bytetrack")
    ap.add_argument("--ekf-tracker", default="yolov8n-ekf-ctrv")
    ap.add_argument("--track-id", type=int, default=None,
                    help="Chi danh dau doan che cua DUNG track nay (khong thi lay ca video, "
                         "co the nham voi track khac dang bi che cung luc).")
    ap.add_argument("--diff-segments", action="store_true",
                    help="Danh dau cac doan ma KF va EKF cho KET CUC KHAC NHAU (doc tu "
                         "results/stratified/segment_outcomes_DETRAC-all.csv). Dung cho video "
                         "dai: bao nguoi xem biet CHO NAO dang co khac biet de nhin vao.")
    a = ap.parse_args()

    img_root = config.find_images_root()
    vdir = img_root / a.video
    n_img = len(list(vdir.glob("img*.jpg")))
    end = a.end or n_img
    print(f"[side_by_side] {a.video}: dung frame {a.start}-{end} (video co {n_img} anh)")

    def load(tracker):
        p = (config.PROCESSED_DIR / "trackers" / a.split_name / tracker / "data" /
             f"{a.video}.txt")
        return pd.read_csv(p, header=None, names=TRK_COLS)

    cv_df, ekf_df = load(a.cv_tracker), load(a.ekf_tracker)

    # --- Cac doan ma 2 model cho ket cuc KHAC NHAU (de danh dau tren video dai) ---
    diff_spans = []
    if a.diff_segments:
        # Panel phai la model nao thi phai doc bang ket cuc CUA MODEL DO. Dung
        # bang cua ban goc de danh dau video ban chan omega se chi sai cho.
        clamped = a.ekf_tracker.endswith("clamped")
        so = (config.RESULTS_DIR / "stratified" /
              ("segment_outcomes_DETRAC-all_clamped.csv" if clamped
               else "segment_outcomes_DETRAC-all.csv"))
        col_ekf = "EKF + CTRV (chan w)" if clamped else "EKF + CTRV"
        if so.exists():
            s = pd.read_csv(so)
            s = s[(s.in_common) & (s.video == a.video)]
            pv = s.pivot_table(index=["video", "track_id", "seg_id", "occlusion_level"],
                               columns="motion_model", values="status",
                               aggfunc="first").reset_index()
            pv = pv[pv["KF + CV"] != pv[col_ekf]]
            locs = []
            for lvl, f in [("full", "full_occlusion_segments.csv"),
                           ("partial", "occlusion_segments.csv")]:
                t = pd.read_csv(config.INTERIM_DIR / f)
                t["occlusion_level"] = lvl
                locs.append(t[["video", "track_id", "seg_id", "occlusion_level",
                               "start_frame", "end_frame"]])
            pv = pv.merge(pd.concat(locs, ignore_index=True),
                          on=["video", "track_id", "seg_id", "occlusion_level"], how="left")
            # Ten cot "KF + CV" co dau cach va dau + -> itertuples doi ten thanh _N.
            # Dung to_dict("records") de truy cap bang ten that, khong phu thuoc vi tri.
            for rec in pv.to_dict("records"):
                if pd.isna(rec.get("start_frame")):
                    continue
                diff_spans.append((int(rec["start_frame"]), int(rec["end_frame"]),
                                   int(rec["track_id"]), rec["KF + CV"], rec[col_ekf]))
            diff_spans.sort()
            print(f"[side_by_side] {len(diff_spans)} doan 2 model cho ket cuc KHAC NHAU:")
            for s0, e0, tid, kf, ekf in diff_spans:
                print(f"    track {tid}: frame {s0}-{e0}   KF={kf}  EKF={ekf}")

    # Danh dau doan bi che (>=0.10, gom ca mot phan lan gan hoan toan) de ve vien
    # vang, doi chieu voi annotation that.
    #
    # PHAI LOC THEO TRACK. Video dong xe thi gan nhu LUC NAO cung co MOT xe nao do
    # dang bi che: do khong loc, MVI_40992 bi to vang 1718/2160 frame (80%) -> vien
    # vang mat het y nghia. Uu tien: --track-id neu co, neu khong thi lay dung cac
    # track xuat hien trong diff_spans, cuoi cung moi lay ca video.
    if a.track_id is not None:
        keep_tracks = {a.track_id}
    elif diff_spans:
        keep_tracks = {t for _, _, t, _, _ in diff_spans}
    else:
        keep_tracks = None
    occ_frames = set()
    for f in ["full_occlusion_segments.csv", "occlusion_segments.csv"]:
        p = config.INTERIM_DIR / f
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[d.video == a.video]
        if keep_tracks is not None:
            d = d[d.track_id.isin(keep_tracks)]
        for r in d.itertuples():
            occ_frames.update(range(int(r.start_frame), int(r.end_frame) + 1))

    first = cv2.imread(str(vdir / f"img{a.start:05d}.jpg"))
    H, W = first.shape[:2]
    out_w = W * 2 + PANEL_GAP
    # Them hau to khi panel phai KHONG phai tracker EKF mac dinh, de ban chan
    # omega khong ghi de len video da render truoc do.
    sfx = "" if a.ekf_tracker == "yolov8n-ekf-ctrv" else "_" + a.ekf_tracker.split("-")[-1]
    out_path = config.RESULTS_DIR / "demo" / f"sbs_{a.video}_{a.start}_{end}{sfx}.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS_OUT, (out_w, H))

    cv_by_f = {k: v for k, v in cv_df.groupby("frame")}
    ekf_by_f = {k: v for k, v in ekf_df.groupby("frame")}

    n_written = 0
    for fr in range(a.start, end + 1):
        fp = vdir / f"img{fr:05d}.jpg"
        im = cv2.imread(str(fp))
        if im is None:
            continue
        in_occ = fr in occ_frames
        left = draw_panel(im, fr, cv_by_f.get(fr, cv_df.iloc[:0]), "KF + CV", in_occ)
        right = draw_panel(im, fr, ekf_by_f.get(fr, ekf_df.iloc[:0]),
                           "EKF + CTRV (chan w)" if sfx else "EKF + CTRV", in_occ)
        gap = np.full((H, PANEL_GAP, 3), 255, dtype=np.uint8)
        combo = np.hstack([left, gap, right])

        # Bao cho nguoi xem biet CHO NAO dang co khac biet giua 2 model
        for s0, e0, tid, kf_st, ekf_st in diff_spans:
            if s0 <= fr <= e0:
                cv2.rectangle(combo, (0, 0), (combo.shape[1] - 1, combo.shape[0] - 1),
                              (0, 0, 255), 8)
                msg = f"KHAC BIET  track {tid}:  KF={kf_st}  vs  EKF={ekf_st}"
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                x0 = (combo.shape[1] - tw) // 2
                cv2.rectangle(combo, (x0 - 12, H - 52), (x0 + tw + 12, H - 14), (0, 0, 255), -1)
                cv2.putText(combo, msg, (x0, H - 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
                break

        # Thanh tien trinh duoi day: vach do = cac diem co khac biet, vach trang =
        # vi tri hien tai. Video 3 phut ma khong co cai nay thi nguoi xem khong biet
        # con phai cho bao lau moi toi doan dang xem.
        if diff_spans:
            W_out, y0 = combo.shape[1], H - 10
            cv2.rectangle(combo, (0, y0), (W_out, H), (40, 40, 40), -1)
            span = max(1, end - a.start)
            for s0, e0, _t, _k, _e in diff_spans:
                xa = int((s0 - a.start) / span * (W_out - 1))
                xb = int((e0 - a.start) / span * (W_out - 1))
                cv2.rectangle(combo, (max(0, xa - 2), y0), (xb + 2, H), (0, 0, 255), -1)
            xc = int((fr - a.start) / span * (W_out - 1))
            cv2.rectangle(combo, (xc - 1, y0), (xc + 1, H), (255, 255, 255), -1)
        vw.write(combo)
        n_written += 1
    vw.release()
    print(f"[side_by_side] da ghi {n_written} frame -> {out_path}")
    scope = ("track %d" % a.track_id) if a.track_id is not None else (
        "cac track co khac biet" if diff_spans else "toan bo video")
    print(f"[side_by_side] {len(occ_frames & set(range(a.start, end + 1)))} frame "
          f"co che khuat (vien vang, pham vi: {scope})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
