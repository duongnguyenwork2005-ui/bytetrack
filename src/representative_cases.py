"""
representative_cases.py -- (MOC 2, muc 2) Sinh `representative_cases.md`: voi MOI nguyen nhan chinh, chon
mot so ca THAT BAI tieu bieu + ca THANH CONG tieu bieu (cung tang), in du so lieu de nguoi review truy
nguoc ve log goc (`event_window_log.csv`, `event_causes_<cond>_<tracker>.csv`).

Tieu chi chon (xac dinh, khong ngau nhien): sap xep theo (has_full giam dan, length_frames giam dan,
video, track_id) roi lay `--n` ca dau cua moi nguyen nhan; ca thanh cong lay tuong tu trong cung tracker.

    python src/representative_cases.py --dir results/phase2/failure_analysis/expA_detrac_dev --tracker ekf --cond c25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

FAIL = ("switched", "lost_after")
COLS_NUM = ["length_frames", "n_gt_after", "first_det_frame", "gap_gt_to_det", "best_iou_high_after",
            "best_iou_low_after", "best_iou_cache_after", "best_cache_conf", "n_lost_at_fdet", "iou_pred_det",
            "center_offset_px", "fused_cost", "det_conf", "competitor_iou", "removed_frame"]


INT_COLS = {"id_before", "id_after", "det_matched_track", "det_stage", "id_after_new_frame", "first_det_frame",
            "removed_frame", "n_lost_at_fdet", "gap_gt_to_det", "n_gt_after", "hijack_frame", "hijack_other_gt",
            "track_matched_det", "center_offset_px"}


def fmt_case(r) -> str:
    def g(k, default="-"):
        v = getattr(r, k, None)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return default
        if isinstance(v, float):
            return f"{v:.0f}" if k in INT_COLS else f"{v:.3f}"
        return str(v)
    lines = [f"- **{r.video} / GT track {r.track_id} / event {r.event_id}** — che khuat frame "
             f"{r.start_frame}–{r.end_frame} ({r.length_frames} frame, {'FULL' if r.has_full else 'partial'}), "
             f"ket qua `{r.outcome}`, ID truoc {g('id_before')} → sau {g('id_after')}"]
    lines.append(f"  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = {g('first_det_frame')} "
                 f"(cach het su kien {g('gap_gt_to_det')} frame); IoU tot nhat sau su kien: high {g('best_iou_high_after')}, "
                 f"low {g('best_iou_low_after')}, cache(0,05) {g('best_iou_cache_after')} @conf {g('best_cache_conf')}")
    lines.append(f"  - track {g('id_before')} tai frame do: trang thai {g('state_at_fdet')}, da mat {g('n_lost_at_fdet')} buoc, "
                 f"IoU(du doan, det) = {g('iou_pred_det')}, lech tam {g('center_offset_px')} px, "
                 f"cost fused = {g('fused_cost')} (det conf {g('det_conf')}), det duoc ghep boi track {g('det_matched_track')} "
                 f"o vong {g('det_stage')}; track bi xoa tai frame {g('removed_frame')}")
    if getattr(r, "id_after_origin", ""):
        lines.append(f"  - nguon ID sau: {r.id_after_origin} (frame sinh {g('id_after_new_frame')})")
    lines.append(f"  - nhan: `{r.causes}` → nguyen nhan chinh **{r.primary_cause}**")
    lines.append(f"  - truy log: `event_window_log.csv` loc `video == \"{r.video}\" & gt_track == {r.track_id} "
                 f"& event_id == {r.event_id}` (cot kind = track/det_high/det_low/gt)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phase2/failure_analysis/expA_detrac_dev")
    ap.add_argument("--tracker", default="ekf")
    ap.add_argument("--cond", default="c25")
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    d = Path(a.dir)
    df = pd.read_csv(d / "event_causes_all.csv")
    df = df[(df.tracker == a.tracker) & (df.cond == a.cond)].copy()
    if not len(df):
        print("khong co du lieu"); return 1
    order = ["has_full", "length_frames"]
    fail = df[df.outcome.isin(FAIL)].sort_values(order + ["video", "track_id"], ascending=[False, False, True, True])
    ok = df[df.outcome == "preserved"].sort_values(order + ["video", "track_id"], ascending=[False, False, True, True])
    out = [f"# Ca tieu bieu — nguyen nhan mat ID ({a.tracker}, dieu kien {a.cond}, dev UA-DETRAC 12 video)", "",
           f"Sinh boi `src/representative_cases.py --dir {a.dir} --tracker {a.tracker} --cond {a.cond} --n {a.n}` "
           f"(chon xac dinh: uu tien tang FULL, su kien dai nhat). So lieu day du: `event_causes_{a.cond}_{a.tracker}.csv`; "
           f"log tung frame quanh su kien: `event_window_log.csv`.", "",
           "Dinh nghia ket qua (mau so = 184 su kien du dieu kien theo GT): `preserved` = ID ngay truoc == ID ngay sau; "
           "`switched` = khac; `lost_after` = co ID truoc, khong ghep lai duoc trong cua so sau (GT van con); "
           "`no_match_before` = tracker chua tung ghep GT truoc su kien (khong danh gia duoc phan giu ID).", ""]
    counts = fail.primary_cause.value_counts()
    for cause, n in counts.items():
        out.append(f"## {cause} — {n} ca that bai")
        sub = fail[fail.primary_cause == cause].head(a.n)
        out += [fmt_case(r) for r in sub.itertuples()]
        out.append("")
    out.append(f"## Ca THANH CONG tieu bieu (preserved) — {len(ok)} ca")
    out += [fmt_case(r) for r in ok.head(max(a.n, 4)).itertuples()]
    out.append("")
    p = d / f"representative_cases_{a.cond}_{a.tracker}.md"
    p.write_text("\n".join(out), encoding="utf-8")
    print(f"-> {p} ({len(counts)} nguyen nhan, {len(fail)} that bai, {len(ok)} thanh cong)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
