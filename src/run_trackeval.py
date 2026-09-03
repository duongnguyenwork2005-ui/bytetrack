"""
run_trackeval.py -- Chay TrackEval (JonathonLuiten/TrackEval, cai qua `pip install
trackeval`) de tinh MOTA, IDF1, HOTA cho ket qua tracker so voi ground truth.

TAI SAO GT CUA TA "VUA KHIT" VOI DATASET MOTChallenge2DBox CUA TrackEval?
--------------------------------------------------------------------------
TrackEval khong tu suy luan class tu ten - no dung 1 bang co dinh
(trackeval/datasets/mot_challenge_2d_box.py, da doi chieu source code):

    class_name_to_class_id = {
        'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4,
        'motorbike': 5, 'non_mot_vehicle': 6, 'static_person': 7,
        'distractor': 8, 'occluder': 9, 'occluder_on_ground': 10,
        'occluder_full': 11, 'reflection': 12, 'crowd': 13,
    }
    valid_classes = ['pedestrian']   # CLASSES_TO_EVAL chi nhan gia tri nay

O Giai doan 1 (to_motchallenge.py), ta gan class = 1 cho MOI bbox xe (gop 4
loai xe thanh 1 class de danh gia "theo vet phuong tien", xem config.MOT_CLASS_ID).
Vi 1 == class_name_to_class_id['pedestrian'], truyen CLASSES_TO_EVAL=['pedestrian']
se danh gia DUNG cac bbox xe cua ta - TrackEval chi so khop theo SO, khong quan
tam ten goi ngu nghia la gi.

CACH TRACKEVAL DOI CHIEU THU MUC (da doi chieu voi __init__ cua MotChallenge2DBox)
------------------------------------------------------------------------------------
    gt_set = BENCHMARK + '-' + SPLIT_TO_EVAL
    GT       : {GT_FOLDER}/{gt_set}/{seq}/gt/gt.txt
    seqmap   : {GT_FOLDER}/seqmaps/{gt_set}.txt
    tracker  : {TRACKERS_FOLDER}/{gt_set}/{tracker_name}/{TRACKER_SUB_FOLDER}/{seq}.txt

Voi to_motchallenge.py --split-name DETRAC-sample, ta da tao dung:
    data/processed/DETRAC-sample/<video>/gt/gt.txt
    data/processed/seqmaps/DETRAC-sample.txt
=> chi can dat BENCHMARK='DETRAC', SPLIT_TO_EVAL='sample', GT_FOLDER=data/processed,
   TRACKERS_FOLDER=data/processed/trackers la khop hoan toan, KHONG can doi ten
   thu muc nao them.

METRIC DUOC TINH
-----------------
- HOTA (Higher Order Tracking Accuracy): can bang giua do chinh xac phat hien
  (Detection Accuracy - DetA) va do chinh xac lien ket track (Association
  Accuracy - AssA). HOTA = sqrt(DetA * AssA). Day la metric duoc khuyen dung
  nhat hien nay vi khong thien vi ve 1 phia nhu MOTA/IDF1.
- CLEAR (cho ra MOTA, MOTP, so FP/FN/IDSW...): MOTA = 1 - (FN+FP+IDSW)/GT,
  thien ve do bao phu phat hien (detection), it nhay cam voi loi lien ket ID.
- Identity (cho ra IDF1, IDP, IDR): do do chinh xac GAN DUNG track id xuyen
  suot video, nhay cam voi ID switch hon MOTA - phu hop de danh gia rieng
  anh huong cua occlusion/curvature len KHA NANG GIU DUNG ID cua motion model.

CACH DUNG
---------
    python src/run_trackeval.py --split-name DETRAC-sample --tracker yolov8n-bytetrack
    python src/run_trackeval.py --split-name DETRAC-sample --tracker yolov8n-bytetrack --per-video
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def build_configs(split_name: str, tracker_name: str, videos: list[str] | None):
    """Dung config cho Dataset + Evaluator + Metric, khop voi cau truc thu muc cua ta."""
    import trackeval

    # 'DETRAC' + '-' + split_name_short se ghep lai thanh gt_set == thu muc ta da tao.
    # Vi split_name da la vd "DETRAC-sample" (co san tien to DETRAC-), ta tach ra
    # de truyen dung BENCHMARK/SPLIT_TO_EVAL ma TrackEval tu ghep lai.
    if not split_name.startswith("DETRAC-"):
        raise ValueError(f"split_name phai bat dau bang 'DETRAC-', nhan duoc: {split_name}")
    split_to_eval = split_name[len("DETRAC-"):]

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(config.PROCESSED_DIR),
        "TRACKERS_FOLDER": str(config.PROCESSED_DIR / "trackers"),
        "BENCHMARK": "DETRAC",
        "SPLIT_TO_EVAL": split_to_eval,
        "TRACKERS_TO_EVAL": [tracker_name],
        "CLASSES_TO_EVAL": ["pedestrian"],   # xem docstring: '1' trung voi class xe cua ta
        "SEQMAP_FOLDER": str(config.PROCESSED_DIR / "seqmaps"),
        "SKIP_SPLIT_FOL": False,
        "PRINT_CONFIG": False,
    })
    if videos:
        # Ghi de danh sach sequence + do dai, khong can dung seqmap file rieng.
        seq_info = {}
        for v in videos:
            ini = config.PROCESSED_DIR / split_name / v / "seqinfo.ini"
            import configparser
            cp = configparser.ConfigParser()
            cp.read(ini)
            seq_info[v] = int(cp["Sequence"]["seqLength"])
        dataset_config["SEQ_INFO"] = seq_info

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False,
        "NUM_PARALLEL_CORES": 1,
        "PRINT_RESULTS": True,
        "PRINT_CONFIG": False,
        "PRINT_ONLY_COMBINED": False,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": False,
        "TIME_PROGRESS": False,
        "LOG_ON_ERROR": str(config.RESULTS_DIR / "trackeval_error.log"),
    })

    metric_config = {"METRICS": ["HOTA", "CLEAR", "Identity"], "THRESHOLD": 0.5, "PRINT_CONFIG": False}
    return eval_config, dataset_config, metric_config


def run(split_name: str, tracker_name: str, videos: list[str] | None):
    """Chay TrackEval, tra ve (eval_results_dict, dataset)."""
    import trackeval

    eval_config, dataset_config, metric_config = build_configs(split_name, tracker_name, videos)

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics_list = [
        trackeval.metrics.HOTA(metric_config),
        trackeval.metrics.CLEAR(metric_config),
        trackeval.metrics.Identity(metric_config),
    ]

    results, _ = evaluator.evaluate([dataset], metrics_list)
    return results, dataset


def flatten_results(results: dict, tracker_name: str) -> pd.DataFrame:
    """Chuyen ket qua long nhau cua TrackEval thanh 1 DataFrame de doc/luu de dang.

    Cau truc goc: results['MotChallenge2DBox'][tracker][seq]['pedestrian'][metric][field]
    HOTA co field dang mang (theo 19 nguong alpha) -> lay trung binh (chinh no
    da la 'HOTA' tong hop) hoac lay gia tri ung voi alpha = 0.5 khi can chi tiet.
    """
    rows = []
    d = results["MotChallenge2DBox"][tracker_name]
    for seq, cls_dict in d.items():
        m = cls_dict.get("pedestrian")
        if m is None:
            continue
        row = {"video": seq}
        hota = m["HOTA"]
        row["HOTA"] = float(np.mean(hota["HOTA"])) if hasattr(hota["HOTA"], "__len__") else float(hota["HOTA"])
        row["DetA"] = float(np.mean(hota["DetA"]))
        row["AssA"] = float(np.mean(hota["AssA"]))
        clear = m["CLEAR"]
        row["MOTA"] = float(clear["MOTA"])
        row["MOTP"] = float(clear["MOTP"])
        row["FP"] = int(clear["CLR_FP"])
        row["FN"] = int(clear["CLR_FN"])
        row["IDSW"] = int(clear["IDSW"])
        row["GT_dets"] = int(clear["CLR_TP"]) + int(clear["CLR_FN"])
        ident = m["Identity"]
        row["IDF1"] = float(ident["IDF1"])
        row["IDP"] = float(ident["IDP"])
        row["IDR"] = float(ident["IDR"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Chay TrackEval (MOTA/IDF1/HOTA)")
    ap.add_argument("--split-name", default="DETRAC-sample",
                    help="Ten split GT (thu muc data/processed/<split-name>/)")
    ap.add_argument("--tracker", required=True,
                    help="Ten thu muc tracker trong data/processed/trackers/<split-name>/")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Chi eval cac video nay (mac dinh: toan bo seqmap cua split)")
    args = ap.parse_args()

    gt_dir = config.PROCESSED_DIR / args.split_name
    trk_dir = config.PROCESSED_DIR / "trackers" / args.split_name / args.tracker
    if not gt_dir.is_dir():
        print(f"[ERROR] Khong tim thay GT split: {gt_dir}")
        print("        Chay src/to_motchallenge.py --split-name ... truoc.")
        return 1
    if not trk_dir.is_dir():
        print(f"[ERROR] Khong tim thay ket qua tracker: {trk_dir}")
        print("        Chay src/baseline_track.py truoc.")
        return 1

    print(f"[trackeval] split={args.split_name}  tracker={args.tracker}")
    try:
        results, dataset = run(args.split_name, args.tracker, args.videos)
    except Exception as e:
        print(f"[ERROR] TrackEval that bai: {type(e).__name__}: {e}")
        raise

    df = flatten_results(results, args.tracker)
    # Tach dong 'COMBINED_SEQ' (tong hop toan bo video) khoi bang chi tiet tung video
    combined = df[df["video"] == "COMBINED_SEQ"]
    per_video = df[df["video"] != "COMBINED_SEQ"].sort_values("video")

    out_dir = config.RESULTS_DIR / "trackeval" / args.split_name / args.tracker
    out_dir.mkdir(parents=True, exist_ok=True)
    per_video.to_csv(out_dir / "per_video_metrics.csv", index=False)
    combined.to_csv(out_dir / "combined_metrics.csv", index=False)

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")

    print("\n" + "=" * 78)
    print(f"KET QUA THEO TUNG VIDEO - {args.tracker}")
    print("=" * 78)
    cols = ["video", "HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDP", "IDR", "FP", "FN", "IDSW"]
    print(per_video[cols].to_string(index=False))

    print("\n" + "=" * 78)
    print(f"KET QUA TONG HOP (COMBINED_SEQ) - {args.tracker}")
    print("=" * 78)
    if len(combined):
        c = combined.iloc[0]
        print(f"  HOTA  = {c['HOTA']:.4f}   (DetA = {c['DetA']:.4f}, AssA = {c['AssA']:.4f})")
        print(f"  MOTA  = {c['MOTA']:.4f}   (FP = {int(c['FP']):,}, FN = {int(c['FN']):,}, IDSW = {int(c['IDSW']):,})")
        print(f"  IDF1  = {c['IDF1']:.4f}   (IDP = {c['IDP']:.4f}, IDR = {c['IDR']:.4f})")
        print(f"  MOTP  = {c['MOTP']:.4f}")

    print(f"\n  Da luu:\n    {out_dir / 'per_video_metrics.csv'}\n    {out_dir / 'combined_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
