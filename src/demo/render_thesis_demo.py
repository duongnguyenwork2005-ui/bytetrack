"""
render_thesis_demo.py -- Dung video demo 3 panel (KF+CV | EKF+CTRV | UKF+CTRV)
                         tu PREDICTION THAT cua tracker, cho slide khoa luan.

NGUYEN TAC
  1. Box va ID ve tren video la box/ID trong file prediction DA LUU
     (data/processed/trackers/<split>/<tracker>/data/<video>.txt) -- khong bia.
  2. "Trajectory du doan" trong luc track bi mat (LOST) KHONG co trong file da
     luu (ByteTrack chi xuat box o trang thai Tracked). No duoc lay bang cach
     CHAY LAI tracker voi cung weights/config/conf va doc thang trang thai
     Kalman/EKF/UKF trong runtime (src/demo/rerun_with_state_log.py). Lan chay
     lai PHAI khop 100% voi prediction da luu trong cua so clip, neu khong script
     tu dung va bao loi -- khong duoc dung du lieu chay lai khi no khong phan
     anh prediction goc.
  3. Trajectory quan sat that = GT (net LIEN, mau xanh la).
     Trajectory du doan cua tracker = net DUT, mau cua track.
  4. Doi ID cua target: doi mau box sang do va hien "ID cu -> ID moi" trong 15 frame.
  5. Moi con so tren video deu do tu chinh clip nay. Khong dua so mo phong hay
     thong ke nhom vao video neu khong ghi ro nguon.

BO CUC 1920x1080
  [0..60)      thanh tieu de: ten video, frame, thoi diem, trang thai che khuat
  [60..780)    3 panel 640x720, moi panel la MOT vung crop co dinh (giong nhau
               o ca 3 panel) phong to 1.6 lan quanh quy dao target
  [780..860)   dong trang thai tung panel: ID dang bam, ket cuc
  [860..1080)  chu thich mau + caption (chi hien o doan cuoi)

CACH DUNG
    python src/demo/render_thesis_demo.py --demo 1
    python src/demo/render_thesis_demo.py --demo 2
    python src/demo/render_thesis_demo.py --demo all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from occlusion_eval import iou_matrix, match_frame  # noqa: E402
from rerun_with_state_log import compare_to_saved, rerun  # noqa: E402

# ---------------------------------------------------------------------------
# Cau hinh 2 demo. Chon tu outputs/thesis_demos/candidates_demo{1,2}.csv
# (xem find_demo_candidates.py va DEMO_REPORT.md ve ly do chon).
# ---------------------------------------------------------------------------
DEMOS = {
    1: dict(
        name="demo_01_turning_occlusion",
        title="Demo 1 - Xe re goc nho, mat dau 39 frame trong luc bi che",
        video="MVI_20065", split="DETRAC-all", part="train",
        trackers={"cv": "runA-cv", "ekf_ctrv": "runA-ekf_ctrv", "ukf_ctrv": "runA-ukf_ctrv"},
        conf=0.25, gt_id=20, ev_start=239, ev_end=419,
        # Su kien che mot phan (GT) keo dai 239-419; doan tracker MAT DAU that la
        # 274-312 (39 frame). Clip tap trung quanh doan do.
        clip_start=250, clip_end=345, out_fps=12,
        key_before=273, key_during=293, key_after=313,
        # Cac so trong caption: goc re do tren GT (find_demo_candidates.py, turn_deg),
        # IoU tai frame 302 do tu match_cost_at_reacquisition.py. Xe di cham (~0,7 px/frame).
        caption=("Xe rẽ nhẹ (~21° trên toàn sự kiện GT, đi chậm ~0,7 px/frame) rồi mất dấu 39 frame (1,56 s) khi bị che một phần:\n"
                 "KF + CV (145 → 218) và EKF + CTRV (152 → 243) gán ID mới khi xe xuất hiện lại.\n"
                 "UKF + CTRV nối lại đúng ID 152: tại frame 302 box dự đoán khớp detection tốt hơn (IoU 0,83 so với 0,59 CV / 0,44 EKF).\n"
                 "Kết quả của riêng clip này."),
    ),
    2: dict(
        name="demo_02_heavy_occlusion",
        title="Demo 2 - Che khuat >= 90% phia sau xe khac",
        video="MVI_39781", split="DETRAC-all", part="train",
        trackers={"cv": "runA-cv", "ekf_ctrv": "runA-ekf_ctrv", "ukf_ctrv": "runA-ukf_ctrv"},
        conf=0.25, gt_id=48, ev_start=1590, ev_end=1619,
        clip_start=1550, clip_end=1626, out_fps=12,      # GT ket thuc o 1626 (xe roi canh)
        key_before=1598, key_during=1606, key_after=1619,
        # Diem ghep tai frame 1619 do tu match_cost_at_reacquisition.py (IoU x score, nguong 0,2).
        caption=("Che khuất ≥ 90 % trong 16 frame (0,64 s), xe đi thẳng, mất dấu 20 frame:\n"
                 "KF + CV nối lại đúng ID 643; EKF + CTRV (646 → 734) và UKF + CTRV (647 → 735) gán ID mới.\n"
                 "Chênh lệch rất mỏng: điểm ghép IoU × score tại frame 1619 là 0,227 (CV) so với 0,199 / 0,199 (ngưỡng ByteTrack 0,200).\n"
                 "Kết quả của riêng clip này — ngược với kỳ vọng ban đầu, và được giữ nguyên như vậy."),
    ),
}

PANEL_W, PANEL_H = 640, 720
CANVAS_W, CANVAS_H = 1920, 1080
TITLE_H, STATUS_H = 60, 80
Y_PANEL = TITLE_H
Y_STATUS = Y_PANEL + PANEL_H
Y_CAPTION = Y_STATUS + STATUS_H
CROP_W, CROP_H = 400, 450          # phong to 1.6x -> 640x720
TRAIL = 36                          # so frame ve duoi quy dao
SWITCH_FLASH = 15                   # so frame hien thong bao doi ID
CAPTION_HOLD_S = 3.5
IOU_MATCH = 0.5

COL_GT = (60, 200, 60)              # xanh la (BGR)
COL_SWITCH = (40, 40, 230)          # do
COL_OCC = (0, 200, 255)             # vang
COL_FULL = (0, 90, 255)             # cam dam - che gan hoan toan
PANEL_LABELS = {"cv": "KF + CV", "ekf_ctrv": "EKF + CTRV", "ukf_ctrv": "UKF + CTRV"}
FONT = "C:/Windows/Fonts/arial.ttf"
FONT_B = "C:/Windows/Fonts/arialbd.ttf"

TRK_COLS = ["frame", "id", "x", "y", "w", "h", "conf"]
GT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]


# ---------------------------------------------------------------------------
# Tien ich ve
# ---------------------------------------------------------------------------
def color_for_id(tid: int) -> tuple[int, int, int]:
    """Mau on dinh theo ID (HSV -> BGR), tranh xanh la (GT) va do (doi ID)."""
    hue = (int(hashlib.md5(str(tid).encode()).hexdigest(), 16) % 150) + 15   # 15..164
    if 45 <= hue <= 75:      # tranh dai xanh la cua GT
        hue += 40
    hsv = np.uint8([[[hue % 180, 210, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def dashed_line(im, p0, p1, col, thick=2, dash=8, gap=6):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    d = p1 - p0
    L = float(np.hypot(*d))
    if L < 1e-6:
        return
    u = d / L
    s = 0.0
    while s < L:
        e = min(s + dash, L)
        cv2.line(im, tuple(np.round(p0 + u * s).astype(int)),
                 tuple(np.round(p0 + u * e).astype(int)), col, thick, cv2.LINE_AA)
        s = e + gap


def dashed_rect(im, x1, y1, x2, y2, col, thick=2):
    dashed_line(im, (x1, y1), (x2, y1), col, thick)
    dashed_line(im, (x2, y1), (x2, y2), col, thick)
    dashed_line(im, (x2, y2), (x1, y2), col, thick)
    dashed_line(im, (x1, y2), (x1, y1), col, thick)


def put_text_pil(im_bgr, text, xy, size=22, color=(255, 255, 255), bold=False,
                 anchor="la", max_w=None):
    """Ve chu co dau bang PIL (cv2 khong ve duoc tieng Viet)."""
    pil = Image.fromarray(cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT_B if bold else FONT, size)
    rgb = (color[2], color[1], color[0])
    if max_w:
        # xuong dong tu dong theo chieu rong
        words, lines, cur = text.split(" "), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if dr.textlength(t, font=font) <= max_w:
                cur = t
            else:
                lines.append(cur); cur = w
        lines.append(cur)
        text = "\n".join(lines)
    dr.multiline_text(xy, text, font=font, fill=rgb, anchor=anchor if "\n" not in text else "la",
                      spacing=6)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Du lieu
# ---------------------------------------------------------------------------
def load_gt(split, video):
    d = pd.read_csv(config.PROCESSED_DIR / split / video / "gt" / "gt.txt", header=None, names=GT_COLS)
    d["cx"] = d.x + d.w / 2
    d["cy"] = d.y + d.h / 2
    return d


def load_pred(split, tracker, video):
    d = pd.read_csv(config.PROCESSED_DIR / "trackers" / split / tracker / "data" / f"{video}.txt",
                    header=None, names=TRK_COLS)
    d["cx"] = d.x + d.w / 2
    d["cy"] = d.y + d.h / 2
    return d


def match_target_per_frame(gt_t: pd.DataFrame, pred: pd.DataFrame, gt_all: pd.DataFrame):
    """Voi moi frame, tim tracker ID ghep voi GT target (IoU >= 0.5, Hungarian
    tren TOAN BO GT cua frame de tranh ghep nham sang xe khac). Tra ve dict
    frame -> tracker_id hoac None."""
    out = {}
    gid = int(gt_t.id.iloc[0])
    for fr, g in gt_all.groupby("frame"):
        p = pred[pred.frame == fr]
        if not len(p):
            out[int(fr)] = None
            continue
        iou = iou_matrix(g[["x", "y", "w", "h"]].to_numpy(float), p[["x", "y", "w", "h"]].to_numpy(float))
        gids = g.id.to_numpy()
        pids = p.id.to_numpy()
        hit = None
        for r, c in match_frame(iou, IOU_MATCH):
            if int(gids[r]) == gid:
                hit = int(pids[c])
        out[int(fr)] = hit
    return out


# ---------------------------------------------------------------------------
# Ve mot panel
# ---------------------------------------------------------------------------
def draw_panel(src_bgr, crop, fr, key, pred, states, gt_t, target_id_now, target_id_hist,
               ev, full_span, switch_info, trail_gt, trail_pred, in_lost):
    """Ve panel cho MOT tracker tai frame `fr`. Tat ca toa do source, crop sau."""
    x0, y0, x1, y1 = crop
    im = src_bgr.copy()

    # --- Box cua MOI track khac (mong, mo) de nhin boi canh ---
    pf = pred[pred.frame == fr]
    for r in pf.itertuples():
        if target_id_now is not None and int(r.id) == target_id_now:
            continue
        c = color_for_id(int(r.id))
        cv2.rectangle(im, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), c, 1)

    # --- GT cua target: box net dut xanh la + quy dao lien ---
    g = gt_t[gt_t.frame == fr]
    if len(g):
        g = g.iloc[0]
        dashed_rect(im, int(g.x), int(g.y), int(g.x + g.w), int(g.y + g.h), COL_GT, 2)
    for i in range(1, len(trail_gt)):
        cv2.line(im, trail_gt[i - 1], trail_gt[i], COL_GT, 3, cv2.LINE_AA)

    # --- Quy dao DU DOAN cua target (net dut, mau track) ---
    for i in range(1, len(trail_pred)):
        (p0, c0), (p1, c1) = trail_pred[i - 1], trail_pred[i]
        dashed_line(im, p0, p1, c1, 3, dash=9, gap=6)

    # --- Box target: dang bam (day) hoac dang du doan mu (net dut) ---
    hit_col = None
    if target_id_now is not None:
        row = pf[pf.id == target_id_now]
        if len(row):
            row = row.iloc[0]
            col = COL_SWITCH if switch_info else color_for_id(target_id_now)
            hit_col = col
            cv2.rectangle(im, (int(row.x), int(row.y)), (int(row.x + row.w), int(row.y + row.h)), col, 3)
            cv2.putText(im, f"ID {target_id_now}", (int(row.x), max(14, int(row.y) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    elif in_lost is not None:
        # track bi mat: ve box du doan tu trang thai runtime (net dut, mau track)
        lid, lcx, lcy, lw, lh = in_lost
        col = color_for_id(lid)
        dashed_rect(im, int(lcx - lw / 2), int(lcy - lh / 2), int(lcx + lw / 2), int(lcy + lh / 2), col, 3)
        cv2.putText(im, f"ID {lid} (du doan)", (int(lcx - lw / 2), max(14, int(lcy - lh / 2) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2, cv2.LINE_AA)

    # --- Crop + phong to ---
    panel = im[y0:y1, x0:x1]
    panel = cv2.resize(panel, (PANEL_W, PANEL_H), interpolation=cv2.INTER_CUBIC)

    # --- Vien trang thai che khuat ---
    if full_span and full_span[0] <= fr <= full_span[1]:
        cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), COL_FULL, 8)
    elif ev[0] <= fr <= ev[1]:
        cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), COL_OCC, 6)

    # --- Tieu de panel co dinh goc tren ---
    cv2.rectangle(panel, (0, 0), (PANEL_W, 40), (0, 0, 0), -1)
    cv2.putText(panel, PANEL_LABELS[key], (12, 28), cv2.FONT_HERSHEY_DUPLEX, 0.95,
                (255, 255, 255), 2, cv2.LINE_AA)

    # --- Thong bao doi ID ---
    if switch_info:
        old, new = switch_info
        msg = f"DOI ID: {old} -> {new}"
        (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.9, 2)
        cv2.rectangle(panel, (PANEL_W - tw - 28, 48), (PANEL_W - 6, 48 + th + 20), COL_SWITCH, -1)
        cv2.putText(panel, msg, (PANEL_W - tw - 18, 48 + th + 8), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
    return panel


# ---------------------------------------------------------------------------
# Chord effect (tinh TRUOC khi ve, de co the chu thich ngay tren video)
# ---------------------------------------------------------------------------
CHORD_MIN_FRAMES = 5      # it hon thi khong ket luan
CHORD_MIN_MARGIN_PX = 3.0  # |dev_gt| - |dev_pred| phai >= 3 px moi chu thich (nhieu annotation GT ~1-2 px)


def collect_lost_positions(tid_map_k: dict, states_k: pd.DataFrame, gt_t: pd.DataFrame,
                           f_start: int, f_end: int) -> list[tuple]:
    """Voi moi frame target KHONG ghep duoc nhung track dang theo con o trang thai
    LOST: tra ve (frame, sai so px, cx_pred, cy_pred, cx_gt, cy_gt).
    Cung mot may trang thai `last_id` nhu vong ve chinh."""
    last_id, rows = None, []
    for fr in range(f_start, f_end + 1):
        tid_now = tid_map_k.get(fr)
        follow_id = tid_now if tid_now is not None else last_id
        if tid_now is not None:
            last_id = tid_now
        if tid_now is not None or follow_id is None:
            continue
        row = states_k[(states_k.frame == fr) & (states_k.id == follow_id)]
        g = gt_t[gt_t.frame == fr]
        if len(row) and row.iloc[0].state == "lost" and len(g):
            r = row.iloc[0]
            rows.append((fr, float(np.hypot(r.cx - g.cx.iloc[0], r.cy - g.cy.iloc[0])),
                         float(r.cx), float(r.cy), float(g.cx.iloc[0]), float(g.cy.iloc[0])))
    return rows


def chord_effect(rows_: list[tuple], gt_t: pd.DataFrame) -> dict:
    """Du doan co "co vao phia trong cua" khong?

    Day cung = doan thang noi vi tri GT o frame ngay truoc khi mat dau va frame
    ngay sau khi ghep lai. Do khoang cach CO DAU tu diem du doan / diem GT toi
    day cung o moi frame mat dau. Neu GT cong ve mot phia (dev_gt cung dau) ma
    du doan nam GAN day cung hon (|dev_pred| < |dev_gt|) thi du doan dang "cat
    goc". `annotate` chi bat khi chenh lech >= CHORD_MIN_MARGIN_PX, vi GT cua
    UA-DETRAC la box ve tay, tam box nhieu co 1-2 px.
    """
    if len(rows_) < CHORD_MIN_FRAMES:
        return dict(n=len(rows_), note="qua it frame mat dau de do", annotate=False)
    f_lo, f_hi = rows_[0][0] - 1, rows_[-1][0] + 1
    a = gt_t[gt_t.frame == f_lo]; b = gt_t[gt_t.frame == f_hi]
    if not len(a) or not len(b):
        return dict(n=len(rows_), note="thieu GT o hai dau day cung", annotate=False)
    A = np.array([a.cx.iloc[0], a.cy.iloc[0]]); B = np.array([b.cx.iloc[0], b.cy.iloc[0]])
    u = B - A; L = float(np.hypot(*u)); u = u / max(L, 1e-9)
    nrm = np.array([-u[1], u[0]])
    dev_p = [float(np.dot(np.array([r[2], r[3]]) - A, nrm)) for r in rows_]
    dev_g = [float(np.dot(np.array([r[4], r[5]]) - A, nrm)) for r in rows_]
    mg, mp = float(np.mean(dev_g)), float(np.mean(dev_p))
    same_side = np.sign(mg) == np.sign(mp) or abs(mp) < 1.0
    inside = bool(abs(mg) > 2.0 and same_side and abs(mp) < abs(mg))
    margin = abs(mg) - abs(mp)
    annotate = bool(inside and margin >= CHORD_MIN_MARGIN_PX)
    if inside and not annotate:
        note = (f"du doan gan day cung hon GT {margin:.1f} px, DUOI nguong {CHORD_MIN_MARGIN_PX} px "
                "(nhieu annotation GT) -> khong chu thich tren video")
    elif annotate:
        note = f"du doan cat goc {margin:.1f} px so voi GT -> co chu thich 'Chord effect' tren video"
    else:
        note = "khong thay du doan co vao phia trong cua"
    return dict(n=len(rows_), chord_len_px=round(L, 1), frames=[rows_[0][0], rows_[-1][0]],
                gt_mean_dev_px=round(mg, 2), pred_mean_dev_px=round(mp, 2),
                pred_inside_curve=inside, margin_px=round(margin, 2),
                ratio_pred_over_gt=round(abs(mp) / abs(mg), 3) if abs(mg) > 1e-6 else None,
                annotate=annotate, note=note)


# ---------------------------------------------------------------------------
# Dung mot demo
# ---------------------------------------------------------------------------
def build_demo(k: int, out_root: Path, skip_rerun_if_cached: bool = True) -> dict:
    D = DEMOS[k]
    name, video, split = D["name"], D["video"], D["split"]
    ev = (D["ev_start"], D["ev_end"])
    f_start, f_end = D["clip_start"], D["clip_end"]
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rerun_dir = out_dir / "rerun"
    rerun_dir.mkdir(exist_ok=True)

    print("=" * 78)
    print(f"{D['title']}")
    print(f"  {video} ({split}), GT id {D['gt_id']}, su kien {ev[0]}-{ev[1]}, clip {f_start}-{f_end}")
    print("=" * 78)

    gt = load_gt(split, video)
    gt_t = gt[gt.id == D["gt_id"]].sort_values("frame")
    img_root = config.find_images_root()

    # Doan che gan hoan toan (>= 0.90) ben trong su kien (neu co)
    fcsv = config.part_file("full_occlusion_segments.csv", D["part"])
    fs = pd.read_csv(fcsv)
    fs = fs[(fs.video == video) & (fs.track_id == D["gt_id"]) & (fs.start_frame <= ev[1]) & (fs.end_frame >= ev[0])]
    full_span = (int(fs.start_frame.min()), int(fs.end_frame.max())) if len(fs) else None

    # ---- Prediction da luu + chay lai co log trang thai (doi chieu bat buoc) ----
    preds, states, verify = {}, {}, {}
    for key, tdir in D["trackers"].items():
        preds[key] = load_pred(split, tdir, video)
        cache_b = rerun_dir / f"rerun_boxes_{tdir}.csv"
        cache_s = rerun_dir / f"rerun_states_{tdir}.csv"
        cache_p = rerun_dir / f"rerun_pred_{tdir}.csv"
        if skip_rerun_if_cached and cache_b.exists() and cache_s.exists() and cache_p.exists():
            boxes, st = pd.read_csv(cache_b), pd.read_csv(cache_s)
            filt = "(cache)"
        else:
            boxes, st, pr, filt = rerun(video, split, key, D["conf"], f_end)
            boxes.to_csv(cache_b, index=False)
            st.to_csv(cache_s, index=False)
            pr.to_csv(cache_p, index=False)      # box du doan TRUOC ghep cap (match_cost_at_reacquisition.py dung)
        cmp = compare_to_saved(boxes, preds[key], f_start, f_end)
        cmp["filter_runtime"] = filt
        verify[key] = cmp
        print(f"  [{key:<9}] doi chieu chay lai vs da luu trong clip: "
              f"{cmp['boxes_matched_iou90_same_id']}/{cmp['n_saved']} box cung ID "
              f"({cmp['pct_same_id']}%), {cmp['frames_same_id_set']}/{cmp['frames']} frame cung tap ID")
        if cmp["pct_same_id"] < 99.0 or cmp["n_rerun"] != cmp["n_saved"]:
            raise SystemExit(f"  DUNG: lan chay lai KHONG khop prediction da luu cho {key}. "
                             f"Khong duoc dung trajectory du doan tu lan chay lai.")
        states[key] = st

    # ---- Ghep target <-> tracker ID tung frame ----
    tid_map = {key: match_target_per_frame(gt_t, preds[key], gt) for key in preds}

    # ---- Sai so du doan luc mat dau + chord effect (tinh truoc de chu thich) ----
    lost_err = {key: collect_lost_positions(tid_map[key], states[key], gt_t, f_start, f_end) for key in preds}
    chord = {key: chord_effect(lost_err[key], gt_t) for key in preds}
    for key in preds:
        print(f"  [{key:<9}] chord effect: {chord[key].get('note')}")

    # ---- Vung crop co dinh quanh quy dao target trong clip ----
    seg = gt_t[(gt_t.frame >= f_start) & (gt_t.frame <= f_end)]
    ccx, ccy = float(seg.cx.mean()), float(seg.cy.mean())
    H0, W0 = cv2.imread(str(img_root / video / f"img{f_start:05d}.jpg")).shape[:2]
    x0 = int(np.clip(ccx - CROP_W / 2, 0, W0 - CROP_W))
    y0 = int(np.clip(ccy - CROP_H / 2, 0, H0 - CROP_H))
    crop = (x0, y0, x0 + CROP_W, y0 + CROP_H)

    # ---- Dung tung frame ----
    tmp_mp4 = out_dir / "_raw_mp4v.mp4"
    vw = cv2.VideoWriter(str(tmp_mp4), cv2.VideoWriter_fourcc(*"mp4v"), D["out_fps"], (CANVAS_W, CANVAS_H))
    trails_gt, trails_pred = [], {key: [] for key in preds}
    last_id = {key: None for key in preds}
    switch_until = {key: (0, None) for key in preds}
    switch_log = {key: [] for key in preds}
    lost_frames = {key: 0 for key in preds}       # tong so frame mat dau trong clip
    lost_streak = {key: 0 for key in preds}       # mat dau LIEN TUC hien tai
    keyframes = {}
    first_rematch = {key: None for key in preds}

    for fr in range(f_start, f_end + 1):
        src = cv2.imread(str(img_root / video / f"img{fr:05d}.jpg"))
        if src is None:
            continue
        g = gt_t[gt_t.frame == fr]
        if len(g):
            trails_gt.append((int(g.cx.iloc[0]), int(g.cy.iloc[0])))
        trails_gt = trails_gt[-TRAIL:]

        panels = []
        status_lines = []
        for key in ("cv", "ekf_ctrv", "ukf_ctrv"):
            tid_now = tid_map[key].get(fr)
            # doi ID?
            switch_info = None
            if tid_now is not None and last_id[key] is not None and tid_now != last_id[key]:
                switch_until[key] = (fr + SWITCH_FLASH, (last_id[key], tid_now))
                switch_log[key].append(dict(frame=fr, old=int(last_id[key]), new=int(tid_now)))
            if fr < switch_until[key][0]:
                switch_info = switch_until[key][1]
            if tid_now is not None:
                if last_id[key] is not None and tid_now == last_id[key] and ev[0] <= fr <= ev[1] + 1:
                    pass
                last_id[key] = tid_now

            # vi tri du doan cua target tu runtime (tracked hoac lost) cho ID dang theo
            in_lost = None
            follow_id = tid_now if tid_now is not None else last_id[key]
            pt = None
            if follow_id is not None:
                st = states[key]
                row = st[(st.frame == fr) & (st.id == follow_id)]
                if len(row):
                    row = row.iloc[0]
                    pt = (int(row.cx), int(row.cy))
                    if tid_now is None and row.state == "lost":
                        in_lost = (int(follow_id), float(row.cx), float(row.cy), float(row.w), float(row.h))
                        lost_frames[key] += 1
                        lost_streak[key] += 1
            if tid_now is not None:
                lost_streak[key] = 0
            if pt is not None:
                trails_pred[key].append((pt, color_for_id(int(follow_id))))
            trails_pred[key] = trails_pred[key][-TRAIL:]
            if tid_now is not None and fr > ev[1] and first_rematch[key] is None:
                first_rematch[key] = fr

            panels.append(draw_panel(src, crop, fr, key, preds[key], states[key], gt_t, tid_now, None,
                                     ev, full_span, switch_info, trails_gt, trails_pred[key], in_lost))
            stt = (f"ID {tid_now}" if tid_now is not None else
                   (f"MAT DAU - du doan ID {follow_id}" if in_lost else "khong ghep duoc"))
            # Chu thich nhe "Chord effect" CHI khi do duoc du doan cat goc ro rang (xem chord_effect())
            ch = chord[key]
            if in_lost and ch.get("annotate") and ch["frames"][0] <= fr <= ch["frames"][1]:
                stt += "   [Chord effect]"
            status_lines.append(stt)

        # ---- Ghep canvas ----
        canvas = np.full((CANVAS_H, CANVAS_W, 3), 18, np.uint8)
        for i, p in enumerate(panels):
            canvas[Y_PANEL:Y_PANEL + PANEL_H, i * PANEL_W:(i + 1) * PANEL_W] = p
        # tieu de
        vis_now = float(g.vis.iloc[0]) if len(g) else float("nan")
        occ_txt = ("CHE GAN HOAN TOAN" if (full_span and full_span[0] <= fr <= full_span[1]) else
                   ("DANG BI CHE" if ev[0] <= fr <= ev[1] else
                    ("truoc che khuat" if fr < ev[0] else "sau che khuat")))
        t_s = (fr - f_start) / 25.0
        cv2.putText(canvas, f"{D['title']}   |   {video}  frame {fr}  t={t_s:4.1f}s   |   "
                            f"{occ_txt}  (GT nhin thay {vis_now*100:.0f}%)",
                    (16, 40), cv2.FONT_HERSHEY_DUPLEX, 0.85, (235, 235, 235), 1, cv2.LINE_AA)
        # trang thai tung panel
        for i, (key, s) in enumerate(zip(("cv", "ekf_ctrv", "ukf_ctrv"), status_lines)):
            col = COL_SWITCH if "MAT" in s or "khong" in s else (200, 255, 200)
            cv2.putText(canvas, s, (i * PANEL_W + 14, Y_STATUS + 34), cv2.FONT_HERSHEY_DUPLEX, 0.8, col, 1, cv2.LINE_AA)
            n_sw = len(switch_log[key])
            cv2.putText(canvas, f"doi ID trong clip: {n_sw}   |   mat dau lien tuc: {lost_streak[key]} frame",
                        (i * PANEL_W + 14, Y_STATUS + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
        # chu thich mau (co dinh)
        canvas = put_text_pil(canvas,
                              "Nét liền xanh lá = quỹ đạo & box GT   ·   Nét đứt màu = quỹ đạo dự đoán của tracker   ·   "
                              "Viền vàng = đang bị che   ·   Viền cam = che ≥ 90 %   ·   Đỏ = đổi ID",
                              (16, Y_CAPTION + 14), size=22, color=(200, 200, 200))
        vw.write(canvas)

        # ---- Keyframe (chon tuong minh tu phan tich chuoi ID, xem DEMO_REPORT.md) ----
        if fr == D["key_before"]:
            keyframes["before_occlusion"] = canvas.copy()
        if fr == D["key_during"]:
            keyframes["during_occlusion"] = canvas.copy()
        if fr == D["key_after"]:
            keyframes["reappearance"] = canvas.copy()

    # ---- Caption cuoi: giu frame cuoi + chu ----
    last = canvas.copy()
    last = put_text_pil(last, D["caption"], (16, Y_CAPTION + 50), size=28, color=(255, 255, 255), bold=True)
    for _ in range(int(CAPTION_HOLD_S * D["out_fps"])):
        vw.write(last)
    vw.release()

    # ---- Chuyen ma H.264 bang ffmpeg ----
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    final_mp4 = out_root / f"{name}.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(tmp_mp4),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
                    "-movflags", "+faststart", str(final_mp4)], check=True)
    tmp_mp4.unlink()

    for kname, kimg in keyframes.items():
        cv2.imwrite(str(out_dir / f"{kname}.png"), kimg)

    # ---- Metadata ----
    n_out = int(cv2.VideoCapture(str(final_mp4)).get(cv2.CAP_PROP_FRAME_COUNT))
    seg_vis = gt_t[(gt_t.frame >= ev[0]) & (gt_t.frame <= ev[1])]
    # Chi phi ghep tai cac frame tai ghep (neu da chay match_cost_at_reacquisition.py)
    mc_csv = out_dir / "match_cost_at_reacquisition.csv"
    match_cost = pd.read_csv(mc_csv).to_dict("records") if mc_csv.exists() else None

    per_tracker = {}
    for key, tdir in D["trackers"].items():
        errs = [e[1] for e in lost_err[key]]
        m = tid_map[key]
        ids_in_ev = sorted({v for f, v in m.items() if ev[0] <= f <= ev[1] and v is not None})
        per_tracker[key] = dict(
            prediction_file=str(config.PROCESSED_DIR / "trackers" / split / tdir / "data" / f"{video}.txt"),
            id_before=m.get(ev[0] - 1), id_after=m.get(ev[1] + 1),
            ids_matched_during_event=ids_in_ev,
            id_switches_in_clip=switch_log[key],
            frames_target_lost_in_clip=lost_frames[key],
            first_frame_rematched_after_event=first_rematch[key],
            recall_during_event=round(sum(1 for f in range(ev[0], ev[1] + 1) if m.get(f) is not None) / max(1, ev[1] - ev[0] + 1), 4),
            pred_error_px_during_lost=dict(n=len(errs), mean=round(float(np.mean(errs)), 2) if errs else None,
                                           max=round(float(np.max(errs)), 2) if errs else None,
                                           last=round(float(errs[-1]), 2) if errs else None),
            chord_effect=chord[key],
            rerun_verification=verify[key],
        )
    meta = dict(
        demo=name, source_video=video, split=split, detector="yolov8n.pt", conf=D["conf"],
        gt_target_id=D["gt_id"], clip_frames=[f_start, f_end], event_frames=list(ev),
        full_occlusion_frames=list(full_span) if full_span else None,
        occlusion_duration_frames=ev[1] - ev[0] + 1,
        occlusion_duration_s=round((ev[1] - ev[0] + 1) / 25.0, 2),
        occlusion_level=dict(vis_min=round(float(seg_vis.vis.min()), 3), vis_mean=round(float(seg_vis.vis.mean()), 3),
                             max_occlusion_ratio=round(1.0 - float(seg_vis.vis.min()), 3)),
        target_gt_box_px=dict(w=round(float(seg_vis.w.mean()), 1), h=round(float(seg_vis.h.mean()), 1)),
        crop_region=list(crop), output=dict(width=CANVAS_W, height=CANVAS_H, fps=D["out_fps"], codec="h264",
                                            n_frames=n_out, duration_s=round(n_out / D["out_fps"], 2)),
        trackers=per_tracker,
        reacquisition_match_cost=match_cost,
        note=("Box/ID lay tu prediction DA LUU. Quy dao du doan trong luc mat dau lay tu lan chay lai "
              "tracker voi cung weights/config, da doi chieu khop 100% voi prediction da luu trong cua so clip. "
              "Moi so trong metadata do tu chinh clip nay. reacquisition_match_cost: xem "
              "src/demo/match_cost_at_reacquisition.py (IoU x score giua box du doan va detection YOLO; "
              "ByteTrack ghep khi > 0.2)."),
    )
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  -> {final_mp4}  ({n_out} frame, {n_out / D['out_fps']:.1f}s)")
    print(f"  -> {out_dir}/{{before_occlusion,during_occlusion,reappearance}}.png, metadata.json")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", default="all", help="1, 2 hoac all")
    ap.add_argument("--out", default="outputs/thesis_demos")
    ap.add_argument("--no-cache", action="store_true", help="Chay lai tracker du da co cache")
    a = ap.parse_args()
    out_root = Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    ks = [1, 2] if a.demo == "all" else [int(a.demo)]
    for k in ks:
        build_demo(k, out_root, skip_rerun_if_cached=not a.no_cache)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
