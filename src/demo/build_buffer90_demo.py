"""Reproduce, audit, and render the selected track_buffer=90 demo.

The selected event is the top row produced by find_buffer90_candidates.py.
All tracker reruns use the existing Stage-E YAML files and are written only to
outputs/thesis_demos_buffer90/rerun/.  Existing tracking results are read-only.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from baseline_track import load_ignored_regions, mask_ignored_regions  # noqa: E402
from occlusion_eval import iou_matrix, match_frame  # noqa: E402
from rerun_with_state_log import compare_to_saved, rerun  # noqa: E402


OUT = Path("outputs/thesis_demos_buffer90")
RERUN = OUT / "rerun"
VIDEO = "MVI_40871"
SPLIT = "DETRAC-all"
GT_ID = 1
EVENT_ID = 0
EVENT = (149, 161)
REACQ = 162
CLIP = (124, 187)
KEYS = {"before_occlusion": 143, "during_occlusion": 156, "reappearance": REACQ}
CONF = 0.25
OUT_FPS = 12
TRACKERS = {
    "cv": ("sweep-cv-buf90", "configs/buffer_sweep/cv_buf90.yaml", "KF + CV"),
    "ekf": ("sweep-ekf_ctrv-buf90", "configs/buffer_sweep/ekf_ctrv_buf90.yaml", "EKF + CTRV"),
    "ukf": ("sweep-ukf_ctrv-buf90", "configs/buffer_sweep/ukf_ctrv_buf90.yaml", "UKF + CTRV"),
}
MOTION_MODEL = {"cv": "cv", "ekf": "ekf_ctrv", "ukf": "ukf_ctrv"}
GT_COLS = ["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"]
TRK_COLS = ["frame", "id", "x", "y", "w", "h", "conf"]
FONT = "C:/Windows/Fonts/arial.ttf"
FONT_B = "C:/Windows/Fonts/arialbd.ttf"
CANVAS = (1920, 1080)
PANEL_W, PANEL_H = 640, 720
CROP_W, CROP_H = 400, 450
IOU_GT = 0.5


def load_gt() -> pd.DataFrame:
    d = pd.read_csv(config.PROCESSED_DIR / SPLIT / VIDEO / "gt" / "gt.txt",
                    header=None, names=GT_COLS)
    d["cx"] = d.x + d.w / 2
    d["cy"] = d.y + d.h / 2
    return d


def load_saved(name: str) -> pd.DataFrame:
    p = config.PROCESSED_DIR / "trackers" / SPLIT / name / "data" / f"{VIDEO}.txt"
    d = pd.read_csv(p, header=None, names=TRK_COLS)
    d["cx"] = d.x + d.w / 2
    d["cy"] = d.y + d.h / 2
    return d


def target_timeline(gt: pd.DataFrame, pred: pd.DataFrame) -> dict[int, int | None]:
    out: dict[int, int | None] = {}
    for fr, g in gt[gt.frame.between(*CLIP)].groupby("frame"):
        p = pred[pred.frame == fr]
        hit = None
        if len(p):
            mat = iou_matrix(g[["x", "y", "w", "h"]].to_numpy(float),
                             p[["x", "y", "w", "h"]].to_numpy(float))
            gids, tids = g.id.to_numpy(), p.id.to_numpy()
            for ri, ci in match_frame(mat, IOU_GT):
                if int(gids[ri]) == GT_ID:
                    hit = int(tids[ci])
                    break
        if bool((g.id == GT_ID).any()):
            out[int(fr)] = hit
    return out


def id_mapping(rerun_boxes: pd.DataFrame, saved: pd.DataFrame) -> tuple[dict[int, int], dict]:
    """Map rerun labels to saved labels by repeated IoU>=0.9 box agreement."""
    votes: dict[tuple[int, int], int] = {}
    n_geom = n_raw = 0
    frames = range(CLIP[0], CLIP[1] + 1)
    for fr in frames:
        a = rerun_boxes[rerun_boxes.frame == fr]
        b = saved[saved.frame == fr]
        if not len(a) or not len(b):
            continue
        mat = iou_matrix(a[["x", "y", "w", "h"]].to_numpy(float),
                         b[["x", "y", "w", "h"]].to_numpy(float))
        for ri, ci in match_frame(mat, 0.9):
            ra, sb = int(a.id.iloc[ri]), int(b.id.iloc[ci])
            n_geom += 1
            n_raw += int(ra == sb)
            votes[(ra, sb)] = votes.get((ra, sb), 0) + 1
    mapping: dict[int, int] = {}
    used_saved: set[int] = set()
    for (ra, sb), _count in sorted(votes.items(), key=lambda x: x[1], reverse=True):
        if ra not in mapping and sb not in used_saved:
            mapping[ra] = sb
            used_saved.add(sb)
    n_mapped = sum(count for (ra, sb), count in votes.items() if mapping.get(ra) == sb)
    n_saved = int(saved.frame.between(*CLIP).sum())
    return mapping, {
        "frames": CLIP[1] - CLIP[0] + 1,
        "saved_boxes": n_saved,
        "geometry_iou90_matches": n_geom,
        "geometry_match_pct": round(100 * n_geom / max(1, n_saved), 3),
        "raw_same_id_matches": n_raw,
        "raw_same_id_pct": round(100 * n_raw / max(1, n_saved), 3),
        "mapped_same_id_matches": n_mapped,
        "mapped_same_id_pct": round(100 * n_mapped / max(1, n_saved), 3),
        "rerun_to_saved_id_map": {str(k): int(v) for k, v in sorted(mapping.items())},
    }


def remap_ids(df: pd.DataFrame, mapping: dict[int, int]) -> pd.DataFrame:
    out = df.copy()
    out["rerun_id"] = out.id.astype(int)
    out["id"] = out.id.astype(int).map(mapping).fillna(out.id).astype(int)
    return out


def replay_match_cost(gt: pd.DataFrame, saved: dict[str, pd.DataFrame],
                      pred_pool: dict[str, pd.DataFrame], maps: dict[str, dict[int, int]],
                      timelines: dict[str, dict[int, int | None]]) -> pd.DataFrame:
    from ultralytics import YOLO
    from ultralytics.trackers.utils import matching

    root = config.find_images_root() / VIDEO
    frame = cv2.imread(str(root / f"img{REACQ:05d}.jpg"))
    frame = mask_ignored_regions(frame, load_ignored_regions(VIDEO, SPLIT))
    result = YOLO(str(config.MODELS_DIR / "yolov8n.pt")).predict(
        frame, conf=CONF, classes=config.YOLO_VEHICLE_CLASSES, device="0", verbose=False
    )[0]
    det_xyxy = result.boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.cpu().numpy().astype(float)
    gt_row = gt[(gt.frame == REACQ) & (gt.id == GT_ID)]
    det_tlwh = np.c_[det_xyxy[:, 0], det_xyxy[:, 1],
                     det_xyxy[:, 2] - det_xyxy[:, 0], det_xyxy[:, 3] - det_xyxy[:, 1]]
    gt_iou = iou_matrix(gt_row[["x", "y", "w", "h"]].to_numpy(float), det_tlwh)[0]
    target_det = int(np.argmax(gt_iou))

    class Detection:
        def __init__(self, score: float):
            self.score = score

    detections = [Detection(float(score)) for score in scores]
    rows = []
    for key, (name, cfg, label) in TRACKERS.items():
        pool = pred_pool[key][pred_pool[key].frame == REACQ].reset_index(drop=True)
        boxes = [np.array([r.cx - r.w / 2, r.cy - r.h / 2,
                           r.cx + r.w / 2, r.cy + r.h / 2], dtype=np.float32)
                 for r in pool.itertuples()]
        raw_cost = matching.iou_distance(boxes, [x for x in det_xyxy])
        fused_cost = matching.fuse_score(raw_cost, detections)
        matches, _, _ = matching.linear_assignment(fused_cost, thresh=0.8)
        pairs = [(int(x[0]), int(x[1])) for x in matches]
        assigned_by_row = dict(pairs)
        assigned_to_target = next((ri for ri, di in pairs if di == target_det), None)

        candidate = pd.read_csv(OUT / "candidates_buffer90.csv")
        event = candidate[(candidate.video == VIDEO) & (candidate.track_id == GT_ID)
                          & (candidate.event_id == EVENT_ID)].iloc[0]
        old_saved_id = int(event[f"id_before_{key}"])
        inverse = {saved_id: rerun_id for rerun_id, saved_id in maps[key].items()}
        old_rerun_id = inverse[old_saved_id]
        row_index = int(pool.index[pool.rerun_id == old_rerun_id][0])
        pred = boxes[row_index]
        iou = float(1.0 - raw_cost[row_index, target_det])
        score = float(scores[target_det])
        similarity = float(1.0 - fused_cost[row_index, target_det])
        cost = float(fused_cost[row_index, target_det])
        assigned_det = assigned_by_row.get(row_index)
        winner_saved_id = None
        if assigned_to_target is not None:
            winner_saved_id = int(pool.id.iloc[assigned_to_target])
        actual_target_id = timelines[key].get(REACQ)
        rows.append({
            "frame": REACQ,
            "model": label,
            "tracker": name,
            "original_track_id_before": old_saved_id,
            "rerun_track_id_before_mapping": old_rerun_id,
            "pred_x1": float(pred[0]), "pred_y1": float(pred[1]),
            "pred_x2": float(pred[2]), "pred_y2": float(pred[3]),
            "det_x1": float(det_xyxy[target_det, 0]), "det_y1": float(det_xyxy[target_det, 1]),
            "det_x2": float(det_xyxy[target_det, 2]), "det_y2": float(det_xyxy[target_det, 3]),
            "detection_score": score,
            "detection_iou_with_gt": float(gt_iou[target_det]),
            "predict_detection_iou": iou,
            "fused_similarity_iou_times_score": similarity,
            "bytetrack_cost_1_minus_fused_similarity": cost,
            "match_cost_threshold": 0.8,
            "direct_pair_passes_threshold": bool(cost <= 0.8),
            "assigned_detection_index_by_full_lap": assigned_det,
            "target_detection_index": target_det,
            "old_track_assigned_target_detection": bool(assigned_det == target_det),
            "target_detection_winner_track_id": winner_saved_id,
            "actual_target_id_after_association": actual_target_id,
            "actual_old_id_preserved": bool(actual_target_id == old_saved_id),
        })
    return pd.DataFrame(rows)


def color_for_id(tid: int) -> tuple[int, int, int]:
    hue = int(hashlib.sha256(str(tid).encode()).hexdigest(), 16) % 180
    hsv = np.uint8([[[hue, 210, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def dashed_rect(im, box, color, thickness=3, dash=10):
    x1, y1, x2, y2 = map(int, box)
    for xa in range(x1, x2, dash * 2):
        cv2.line(im, (xa, y1), (min(xa + dash, x2), y1), color, thickness)
        cv2.line(im, (xa, y2), (min(xa + dash, x2), y2), color, thickness)
    for ya in range(y1, y2, dash * 2):
        cv2.line(im, (x1, ya), (x1, min(ya + dash, y2)), color, thickness)
        cv2.line(im, (x2, ya), (x2, min(ya + dash, y2)), color, thickness)


def pil_text(im: np.ndarray, text: str, xy: tuple[int, int], size: int,
             color=(255, 255, 255), bold=False, max_width: int | None = None) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FONT_B if bold else FONT, size)
    if max_width:
        lines, current = [], ""
        for word in text.split():
            trial = (current + " " + word).strip()
            if draw.textlength(trial, font=font) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        text = "\n".join(lines)
    draw.multiline_text(xy, text, font=font, fill=(color[2], color[1], color[0]), spacing=6)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def render(gt: pd.DataFrame, saved: dict[str, pd.DataFrame],
           pred_pool: dict[str, pd.DataFrame], timelines: dict[str, dict[int, int | None]],
           costs: pd.DataFrame, event_row: pd.Series) -> dict:
    root = config.find_images_root() / VIDEO
    target = gt[(gt.id == GT_ID) & gt.frame.between(*CLIP)]
    center_x, center_y = float(target.cx.mean()), float(target.cy.mean())
    sample = cv2.imread(str(root / f"img{CLIP[0]:05d}.jpg"))
    h0, w0 = sample.shape[:2]
    x0 = int(np.clip(center_x - CROP_W / 2, 0, w0 - CROP_W))
    y0 = int(np.clip(center_y - CROP_H / 2, 0, h0 - CROP_H))
    crop = (x0, y0, x0 + CROP_W, y0 + CROP_H)
    raw_path = OUT / "_demo_buffer90_raw.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS, CANVAS)
    keyframes: dict[str, np.ndarray] = {}
    prior_ids = {key: int(event_row[f"id_before_{key}"]) for key in TRACKERS}
    outcomes = {key: str(event_row[f"outcome_{key}"]) for key in TRACKERS}
    trails = {key: [] for key in TRACKERS}

    for fr in range(CLIP[0], CLIP[1] + 1):
        source = cv2.imread(str(root / f"img{fr:05d}.jpg"))
        panels = []
        for key, (_name, _cfg, label) in TRACKERS.items():
            im = source.copy()
            pf = saved[key][saved[key].frame == fr]
            target_id = timelines[key].get(fr)
            for r in pf.itertuples():
                col = color_for_id(int(r.id))
                thick = 4 if target_id is not None and int(r.id) == target_id else 1
                cv2.rectangle(im, (int(r.x), int(r.y)), (int(r.x + r.w), int(r.y + r.h)), col, thick)
                if thick > 1:
                    cv2.putText(im, f"ID {int(r.id)}", (int(r.x), max(18, int(r.y) - 7)),
                                cv2.FONT_HERSHEY_DUPLEX, 0.7, col, 2, cv2.LINE_AA)
                    trails[key].append((int(r.cx), int(r.cy)))
            trails[key] = trails[key][-25:]
            for i in range(1, len(trails[key])):
                cv2.line(im, trails[key][i - 1], trails[key][i], color_for_id(target_id or prior_ids[key]), 2)

            g = target[target.frame == fr]
            if len(g):
                gr = g.iloc[0]
                dashed_rect(im, (gr.x, gr.y, gr.x + gr.w, gr.y + gr.h), (60, 210, 60), 2)

            if fr == REACQ:
                c = costs[costs.tracker == TRACKERS[key][0]].iloc[0]
                dashed_rect(im, (c.pred_x1, c.pred_y1, c.pred_x2, c.pred_y2), (255, 80, 255), 3)
                cv2.rectangle(im, (int(c.det_x1), int(c.det_y1)), (int(c.det_x2), int(c.det_y2)),
                              (255, 220, 40), 3)

            panel = cv2.resize(im[crop[1]:crop[3], crop[0]:crop[2]], (PANEL_W, PANEL_H),
                               interpolation=cv2.INTER_CUBIC)
            if EVENT[0] <= fr <= EVENT[1]:
                cv2.rectangle(panel, (0, 0), (PANEL_W - 1, PANEL_H - 1), (0, 210, 255), 7)
            cv2.rectangle(panel, (0, 0), (PANEL_W, 46), (0, 0, 0), -1)
            cv2.putText(panel, label, (14, 33), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
            panels.append(panel)

        canvas = np.full((CANVAS[1], CANVAS[0], 3), 18, np.uint8)
        canvas[60:780] = np.hstack(panels)
        occ = "ĐANG BỊ CHE" if EVENT[0] <= fr <= EVENT[1] else (
            "TRƯỚC CHE KHUẤT" if fr < EVENT[0] else "SAU CHE KHUẤT")
        title = f"track_buffer=90 · {VIDEO} · GT {GT_ID} · frame {fr} · {occ}"
        canvas = pil_text(canvas, title, (18, 14), 28, bold=True)
        for i, key in enumerate(TRACKERS):
            now = timelines[key].get(fr)
            before, after = int(event_row[f"id_before_{key}"]), int(event_row[f"id_after_{key}"])
            cont = bool(event_row[f"id_continuous_{key}"])
            status = f"ID hiện tại: {now if now is not None else 'không ghép'} · {before} → {after} · {outcomes[key]} · continuous={int(cont)}"
            canvas = pil_text(canvas, status, (i * PANEL_W + 12, 802), 21,
                              color=(210, 255, 210) if outcomes[key] == "preserved" else (100, 180, 255))
        legend = f"GT: nét đứt xanh · Detection tại frame {REACQ}: xanh lam · Box predict trước ghép: tím · Viền vàng: sự kiện che khuất"
        canvas = pil_text(canvas, legend, (18, 862), 22, color=(205, 205, 205))
        if fr == REACQ:
            by_key = {key: costs[costs.tracker == TRACKERS[key][0]].iloc[0] for key in TRACKERS}
            summary = (f"Frame {REACQ} — điểm ghép IoU×score: "
                       f"CV {by_key['cv'].fused_similarity_iou_times_score:.3f} · "
                       f"EKF {by_key['ekf'].fused_similarity_iou_times_score:.3f} · "
                       f"UKF {by_key['ukf'].fused_similarity_iou_times_score:.3f}; ngưỡng 0,200")
            canvas = pil_text(canvas, summary, (18, 904), 25, bold=True)
        writer.write(canvas)
        for name, key_fr in KEYS.items():
            if fr == key_fr:
                keyframes[name] = canvas.copy()

    caption = ("Trong tình huống minh họa này với track_buffer=90, EKF giữ được ID còn CV không giữ được. "
               "Đây là một trường hợp cụ thể; không tự nó chứng minh EKF vượt trội trên toàn bộ dữ liệu.\n"
               f"Kết cục: CV {outcomes['cv']} ({prior_ids['cv']}→{int(event_row.id_after_cv)}) · "
               f"EKF {outcomes['ekf']} ({prior_ids['ekf']}→{int(event_row.id_after_ekf)}) · "
               f"UKF {outcomes['ukf']} ({prior_ids['ukf']}→{int(event_row.id_after_ukf)}).")
    final = pil_text(canvas.copy(), caption, (20, 940), 25, bold=True, max_width=1870)
    for _ in range(int(3.5 * OUT_FPS)):
        writer.write(final)
    writer.release()

    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    final_path = OUT / "demo_ekf_buffer90.mp4"
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(raw_path),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-preset", "slow", "-movflags", "+faststart", str(final_path)], check=True)
    raw_path.unlink()
    for name, image in keyframes.items():
        cv2.imwrite(str(OUT / f"{name}.png"), image)
    cap = cv2.VideoCapture(str(final_path))
    info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "codec": "H.264"}
    cap.release()
    return {"crop": list(crop), "output": info, "caption": caption}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    RERUN.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(OUT / "candidates_buffer90.csv")
    event = candidates[(candidates.video == VIDEO) & (candidates.track_id == GT_ID)
                       & (candidates.event_id == EVENT_ID)].iloc[0]
    if int(event["rank"]) != 1 or not bool(event.qualifies):
        raise SystemExit("Selected event is no longer the top qualifying candidate")

    gt = load_gt()
    saved, rerun_boxes, states, pred_pool = {}, {}, {}, {}
    mappings, verification, runtime = {}, {}, {}
    for key, (name, tracker_cfg, _label) in TRACKERS.items():
        saved[key] = load_saved(name)
        runtime[key] = {}
        boxes, state, pool, filt = rerun(
            VIDEO, SPLIT, MOTION_MODEL[key], CONF, CLIP[1], tracker_cfg=tracker_cfg,
            runtime_info=runtime[key]
        )
        basic = compare_to_saved(boxes, saved[key], CLIP[0], CLIP[1])
        mapping, mapped = id_mapping(boxes, saved[key])
        mappings[key] = mapping
        mapped["basic_check"] = basic
        verification[key] = mapped
        rerun_boxes[key] = remap_ids(boxes, mapping)
        states[key] = remap_ids(state, mapping)
        pred_pool[key] = remap_ids(pool, mapping)
        rerun_boxes[key].to_csv(RERUN / f"rerun_boxes_{name}.csv", index=False)
        states[key].to_csv(RERUN / f"rerun_states_{name}.csv", index=False)
        pred_pool[key].to_csv(RERUN / f"rerun_pred_{name}.csv", index=False)
        if mapped["geometry_match_pct"] < 98.0 or mapped["mapped_same_id_pct"] < 98.0:
            raise SystemExit(f"Rerun does not reproduce saved predictions for {name}: {mapped}")
        if runtime[key].get("track_buffer") != 90:
            raise SystemExit(f"Runtime track_buffer is not 90 for {name}: {runtime[key]}")
        print(name, filt, runtime[key], mapped)

    timelines = {key: target_timeline(gt, saved[key]) for key in TRACKERS}
    rerun_timelines = {key: target_timeline(gt, rerun_boxes[key]) for key in TRACKERS}
    for key in TRACKERS:
        before = rerun_timelines[key].get(EVENT[0] - 1)
        after = rerun_timelines[key].get(REACQ)
        expected_before = int(event[f"id_before_{key}"])
        expected_after = int(event[f"id_after_{key}"])
        target_ok = before == expected_before and after == expected_after
        verification[key]["target_id_before_event_rerun"] = before
        verification[key]["target_id_at_reappearance_rerun"] = after
        verification[key]["target_outcome_matches_saved"] = target_ok
        if not target_ok:
            raise SystemExit(
                f"Rerun target outcome differs for {key}: "
                f"before={before}/{expected_before}, after={after}/{expected_after}"
            )
    costs = replay_match_cost(gt, saved, pred_pool, mappings, timelines)
    costs.to_csv(OUT / "match_cost_at_reacquisition.csv", index=False)
    render_info = render(gt, saved, pred_pool, timelines, costs, event)

    meta = {
        "demo": "demo_ekf_buffer90",
        "selection": {"rank": 1, "candidate_table": str(OUT / "candidates_buffer90.csv")},
        "source": {"split": SPLIT, "part": "train", "video": VIDEO, "gt_id": GT_ID,
                   "event_frames": list(EVENT), "reappearance_frame": REACQ,
                   "reappearance_note": "First stable EKF/UKF reassociation after an unmatched streak that overlaps the annotated event; frame 162 is immediately after the merged event ends.",
                   "clip_frames": list(CLIP)},
        "configuration": {"condition": "Stage E extended (not A/B); A-like conf=0.25 with track_buffer=90",
                          "detector": "models/yolov8n.pt", "conf": CONF,
                          "classes": config.YOLO_VEHICLE_CLASSES, "seed": 0,
                          "tracker_runtime": runtime},
        "event": {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v)
                  for k, v in event.to_dict().items()},
        "rerun_verification": verification,
        "association_analysis_at_reappearance": [
            {k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v)
             for k, v in row.items()}
            for row in costs.to_dict("records")
        ],
        "render": render_info,
        "interpretation_limit": ("One selected train event; it illustrates a real outcome but does not establish "
                                 "dataset-wide superiority. Association values are an offline replay/contrast at "
                                 "the reacquisition frame, not a complete historical runtime association log."),
    }
    with open(OUT / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(costs.to_string(index=False))
    print(OUT / "demo_ekf_buffer90.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
