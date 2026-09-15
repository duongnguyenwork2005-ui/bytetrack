"""
cause_origin_table.py -- (MOC 2, lam ro diem 2) Tach BA CAU HOI KHAC NHAU trong phan tich mat ID va in
bang cheo + bang "cot log nao la bang chung cho nhan nao".

    Cau hoi 1 (nguyen nhan): VI SAO track CU khong duoc ghep lai?   -> cot `primary_cause` / `causes`
    Cau hoi 2 (hau qua)    : ID moi cua GT den TU DAU?               -> cot `id_after_origin`
                             (new_track_from_det = detection sinh track moi; existing_other_track = mot
                              track da co san bat duoc GT nay)
    Cau hoi 3 (kha nang danh gia): su kien co danh gia duoc khong?   -> `outcome` = no_match_before / eligible

Hai cau hoi dau DOC LAP: "ID moi sinh tu detection" KHONG phai mot nguyen nhan - no chi mo ta ID sau su
kien den tu dau; nguyen nhan nam o phia track cu (du doan lech / het han bo dem / nguong ghep / canh tranh).
Nhan `unknown` duoc GIU khi khong du bang chung.

    python src/cause_origin_table.py --dir results/phase2/failure_analysis/expA_detrac_dev --cond c25
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

#: nhan nguyen nhan -> (cot so lieu la bang chung, dieu kien ghi trong failure_analysis.analyze_event)
EVIDENCE = {
    "prediction_drift": ("state_at_fdet, n_lost_at_fdet, iou_pred_det, center_offset_px, det_conf",
                         "track con song tai frame detector thay lai GT, nhung IoU(box du doan, det) < 0.2"),
    "buffer_expired": ("state_at_fdet = removed, removed_frame, first_det_frame",
                       "track bi xoa (n_lost > track_buffer) TRUOC frame detector thay lai GT"),
    "score_fusion_reject": ("iou_pred_det, det_conf, fused_cost",
                            "IoU >= 0.2 nhung 1 - IoU*conf > match_thresh (0.8)"),
    "competition": ("det_matched_track, competitor_iou, det_stage",
                    "cap (track cu, det) hop le nhung det bi track KHAC lay o vong ghep"),
    "detector_miss": ("best_iou_high_after, best_iou_low_after, best_iou_cache_after, best_cache_conf, detector_miss_sub",
                      "khong co detection tier-high nao IoU >= 0.5 voi GT trong cua so sau su kien"),
    "hijacked_during": ("hijack_frame, hijack_other_gt",
                        "track cu (dang Tracked) ghep detection thuoc GT KHAC ngay trong su kien"),
    "track_matched_other": ("track_matched_det", "track cu ghep mot det khac tai frame do"),
    "matched_but_eval_mismatch": ("det_matched_track = id_before", "log cho thay DA ghep nhung cham su kien ra khac"),
    "unknown": ("(khong co)", "khong nhan nao du bang chung"),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/phase2/failure_analysis/expA_detrac_dev")
    ap.add_argument("--cond", default="c25")
    a = ap.parse_args()
    d = Path(a.dir)
    df = pd.read_csv(d / "event_causes_all.csv")
    df = df[df.cond == a.cond]
    fail = df[df.outcome.isin(["switched", "lost_after"])].copy()
    fail["stratum"] = fail.has_full.map({True: "full", False: "partial"})
    fail["id_after_origin"] = fail.id_after_origin.fillna("").replace("", "(khong co ID sau - lost_after)")

    cross = (fail.groupby(["tracker", "primary_cause", "id_after_origin"]).size()
             .reset_index(name="n").sort_values(["tracker", "primary_cause", "n"], ascending=[True, True, False]))
    cross.to_csv(d / f"cause_by_origin_{a.cond}.csv", index=False)

    ev = pd.DataFrame([{"primary_cause": k, "cot_log_la_bang_chung": v[0], "dieu_kien_gan_nhan": v[1],
                        "n_cv": int(((fail.tracker == "cv") & (fail.primary_cause == k)).sum()),
                        "n_ekf": int(((fail.tracker == "ekf") & (fail.primary_cause == k)).sum()),
                        "n_ukf": int(((fail.tracker == "ukf") & (fail.primary_cause == k)).sum()),
                        "n_pf500": int(((fail.tracker == "pf500") & (fail.primary_cause == k)).sum())}
                       for k, v in EVIDENCE.items()])
    ev.to_csv(d / f"cause_evidence_{a.cond}.csv", index=False)

    pd.set_option("display.width", 250)
    print(f"[{a.cond}] Bang cheo NGUYEN NHAN (track cu khong duoc ghep) x NGUON ID SAU (hau qua):")
    piv = fail.pivot_table(index=["tracker", "primary_cause"], columns="id_after_origin", values="n_gt_after",
                           aggfunc="size", fill_value=0)
    print(piv.to_string())
    print("\nCot log la bang chung cho tung nhan (so ca o dieu kien " + a.cond + "):")
    print(ev[["primary_cause", "n_cv", "n_ekf", "n_ukf", "n_pf500", "cot_log_la_bang_chung"]].to_string(index=False))
    print(f"\n-> {d / f'cause_by_origin_{a.cond}.csv'}, {d / f'cause_evidence_{a.cond}.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
