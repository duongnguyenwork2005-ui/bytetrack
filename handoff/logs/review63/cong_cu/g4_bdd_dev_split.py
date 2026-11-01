"""R63 G4 (Q3): dung split tracking BDD dev tu DUNG cac video BDD trong `dev` cua split mixed5 da khoa
(data/yolo_mixed5_v1/manifest/split_da_khoa.json; dev = BDD dev + dev12 UA-DETRAC). Khong chia lai, khong dung BDD val.

Nhan: box_track_20 native train (data/raw/bdd100k/labels/box_track_20/train/<video>.json) qua
`mot_datasets.bdd100k_mot.parse_video`; GT MOTChallenge qua `export_motchallenge(target='vehicle', benchmark='BDDDEV')`
-> data/processed/BDDDEV-train/ + seqmaps/BDDDEV-train.txt (lop 1 = car/truck/bus/motorcycle/bicycle; crowd, other
person/trailer/other vehicle, train -> lop 13 bo qua, nhu split BDD hien co). KHONG goi gt_dataframe (tranh ghi interim
parquet cho ca 1.400 video).

Kiem (loi -> exit 2, khong ghi gi): dung 139 video BDD; moi JSON doc duoc + schema; moi frame trong JSON co anh tren
dia; (frame, track_id) duy nhat; lop thuoc tap da biet; split dich chua ton tai. Ghi provenance (sha256 tung JSON,
dem box/track theo lop, ty le occluded, so dong eval/ignore) vao --ra.

    python handoff/logs/review63/cong_cu/g4_bdd_dev_split.py --ra handoff/logs/review63/g4/bdd_dev_split.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GOC / "src"))

import config  # noqa: E402
import pandas as pd  # noqa: E402
from mot_datasets import bdd100k_mot as bm  # noqa: E402

SPLIT_KHOA = GOC / "data" / "yolo_mixed5_v1" / "manifest" / "split_da_khoa.json"
SO_BDD_DEV = 139
BENCH, PART = "BDDDEV", "train"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", required=True, help="file JSON provenance (chua ton tai)")
    a = ap.parse_args()
    ra = Path(a.ra)
    if ra.exists():
        print("[LOI] da ton tai", ra); return 2
    split = bm.split_name_for(PART, "vehicle", BENCH)
    if (config.PROCESSED_DIR / split).exists() or (config.PROCESSED_DIR / "seqmaps" / f"{split}.txt").exists():
        print("[LOI] split da ton tai", split); return 2
    khoa = json.loads(SPLIT_KHOA.read_text(encoding="utf-8"))
    videos = sorted(v for v in khoa["dev"] if not v.startswith("MVI_"))
    if len(videos) != SO_BDD_DEV or len(set(videos)) != SO_BDD_DEV:
        print("[LOI] can %d video BDD dev, gap %d" % (SO_BDD_DEV, len(videos))); return 2
    ds = bm.BDD100KMOTDataset()
    loi, dfs, seq_len, nguon = [], [], {}, {}
    for v in videos:
        js = ds.labels_dir(PART) / f"{v}.json"
        if not js.is_file():
            loi.append(f"{v}: thieu {js}"); continue
        fr = ds.frames(v, PART)
        thieu = [f.name for f in fr if f.path is None or not Path(f.path).is_file()]
        if thieu:
            loi.append(f"{v}: {len(thieu)} frame khong co anh (vd {thieu[:2]})")
        df = bm.parse_video(js, PART)
        if df.duplicated(["frame", "track_id"]).any():
            loi.append(f"{v}: trung (frame, track_id)")
        la = set(df["category"]) - set(bm.ALL_CATEGORIES)
        if la:
            loi.append(f"{v}: lop la {sorted(la)}")
        seq_len[v] = len(fr)
        if len(df) and int(df["frame"].max()) > len(fr):
            loi.append(f"{v}: frame GT {int(df['frame'].max())} > so anh {len(fr)}")
        nguon[v] = {"json_sha256": sha(js), "n_frame": len(fr), "n_box": int(len(df)),
                    "n_track": int(df["track_id"].nunique()) if len(df) else 0}
        dfs.append(df)
    if loi:
        print("[LOI] %d van de, khong ghi:" % len(loi)); [print("  ", x) for x in loi[:30]]
        return 2
    df = pd.concat(dfs, ignore_index=True)
    kq = bm.export_motchallenge(df, PART, "vehicle", seq_lengths=seq_len, videos=videos, benchmark=BENCH)
    xe = df[df["category"].isin(bm.VEHICLE_CATEGORIES) & ~df["crowd"]]
    out = {"split": kq, "nguon_split": {"file": str(SPLIT_KHOA.relative_to(GOC)), "sha256": sha(SPLIT_KHOA),
                                        "sha256_danh_sach_dev_khoa": khoa.get("sha256_danh_sach")},
           "videos": videos, "n_frame_tong": sum(seq_len.values()),
           "theo_lop": {c: {"n_box": int((df["category"] == c).sum()),
                            "n_track": int(df[df["category"] == c].groupby(["video", "track_id"]).ngroups)}
                        for c in sorted(df["category"].unique())},
           "xe_eval": {"n_box": int(len(xe)), "ty_le_occluded": float(xe["occluded"].mean()) if len(xe) else None},
           "crowd_box": int(df["crowd"].sum()), "nguon_video": nguon,
           "ghi_chu": "occluded la co nhi phan cua BDD (khong phai muc do); visibility trong gt.txt la gia tri quy uoc "
                      "cua adapter (0.5 neu occluded) - phan tang che khuat BDD dung co occluded, khong dung visibility."}
    ra.parent.mkdir(parents=True, exist_ok=True)
    ra.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[g4_bdd_dev_split]", json.dumps(kq), "| frame", out["n_frame_tong"], "| xe eval", out["xe_eval"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
