"""
check_bdd100k.py -- Kiem tra du lieu BDD100K-MOT tren may, thong ke theo lop, khoang thoi
gian giua frame, va tao manifest split (dev tu train / final = val).

Buoc 1 (sau khi tai "MOT 2020 Labels" + "MOT 2020 Images" va giai nen vao data/raw/bdd100k):
    python src/check_bdd100k.py --part val                # schema + thong ke + kiem tra anh
    python src/check_bdd100k.py --part train --make-manifest --n-dev 40 --seed 0
Fixture (khong anh):
    python src/check_bdd100k.py --part val --data-root data/fixtures/bdd100k_mini --no-images

Ket qua: results/phase2/bdd100k/<part>_class_stats.csv, <part>_dt_stats.csv,
         <part>_schema_report.json, va manifest results/phase2/manifests/bdd100k_{dev,final_val}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from mot_datasets.bdd100k_mot import (BDD100KMOTDataset, SchemaError, check_schema, dt_stats,  # noqa: E402
                                      stats_report, track_gaps)

OUT_DIR = config.RESULTS_DIR / "phase2" / "bdd100k"
MANIFEST_DIR = config.RESULTS_DIR / "phase2" / "manifests"


def make_dev_manifest(df: pd.DataFrame, n_dev: int, seed: int) -> dict:
    """Chon n_dev video tu TRAIN theo tieu chi GT (truoc khi xem ket qua phuong phap nao).

    Tieu chi (tinh tu GT, doc lap thuat toan): co >= 1 track motorcycle (uu tien - muc tieu
    xe may), mat do xe/frame, ti le box occluded, so gap. Chon PHAN TANG: chia video theo
    (co_xe_may, tercile mat do) roi lay ngau nhien deu trong tung tang bang seed co dinh.
    """
    per = []
    gaps = track_gaps(df)
    for v, g in df.groupby("video"):
        veh = g[g["category"].isin(["car", "truck", "bus", "motorcycle", "bicycle"])]
        per.append({"video": v, "n_frames": int(g["frame"].max()), "n_tracks": int(g.groupby("track_id").ngroups),
                    "n_motorcycle_tracks": int(g[g.category == "motorcycle"]["track_id"].nunique()),
                    "n_bicycle_tracks": int(g[g.category == "bicycle"]["track_id"].nunique()),
                    "density_veh_per_frame": round(len(veh) / max(int(g["frame"].max()), 1), 3),
                    "pct_occluded": round(100 * g["occluded"].mean(), 2),
                    "n_gaps": int((gaps.video == v).sum()), "n_crowd": int(g["crowd"].sum())})
    stats = pd.DataFrame(per)
    stats["has_moto"] = stats["n_motorcycle_tracks"] > 0
    stats["density_tercile"] = pd.qcut(stats["density_veh_per_frame"].rank(method="first"), 3, labels=[0, 1, 2])
    rng = np.random.default_rng(seed)
    strata = stats.groupby(["has_moto", "density_tercile"], observed=True)
    quota = max(1, n_dev // max(strata.ngroups, 1))
    chosen = []
    for _, grp in strata:
        k = min(quota, len(grp))
        chosen += list(rng.choice(grp["video"].to_numpy(), size=k, replace=False))
    # bu cho du n_dev, uu tien video co xe may
    rest = stats[~stats.video.isin(chosen)].sort_values(["has_moto", "n_motorcycle_tracks"], ascending=False)
    for v in rest["video"]:
        if len(chosen) >= n_dev:
            break
        chosen.append(v)
    chosen = sorted(chosen[:n_dev])
    return {"name": "bdd100k_dev", "part": "train", "n_videos": len(chosen), "seed": seed,
            "criteria": "phan tang (co xe may) x (tercile mat do xe/frame); chon ngau nhien trong tang; bu theo so track xe may",
            "videos": chosen, "stats": stats[stats.video.isin(chosen)].to_dict("records")}


def main() -> int:
    ap = argparse.ArgumentParser(description="Kiem tra BDD100K-MOT")
    ap.add_argument("--part", default="val", choices=["train", "val"])
    ap.add_argument("--data-root", default=None, help="mac dinh data/raw/bdd100k")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--no-images", action="store_true", help="khong kiem tra file anh")
    ap.add_argument("--make-manifest", action="store_true")
    ap.add_argument("--n-dev", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rebuild", action="store_true", help="doc lai JSON du parquet da co")
    a = ap.parse_args()

    ds = BDD100KMOTDataset(a.data_root)
    videos = a.videos or ds.list_videos(a.part)
    if not videos:
        print(f"[ERROR] Khong co file JSON nao trong {ds.labels_dir(a.part)}.\n"
              f"        Can tai 'MOT 2020 Labels' (bdd-data.berkeley.edu) va giai nen sao cho co\n"
              f"        {ds.root}/labels/box_track_20/{a.part}/<video>.json")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- 1. Schema ---
    report = {"part": a.part, "root": str(ds.root), "n_videos": len(videos), "errors": [], "videos": {}}
    ok_videos = []
    for v in videos:
        try:
            st = check_schema(ds.load_json(v, a.part), v)
            report["videos"][v] = st
            ok_videos.append(v)
        except SchemaError as e:
            report["errors"].append(str(e))
    print(f"[schema] {len(ok_videos)}/{len(videos)} video hop le; {len(report['errors'])} loi")
    for e in report["errors"][:10]:
        print("   !!", e)
    if not ok_videos:
        return 1

    # --- 2. Anh ---
    if not a.no_images:
        missing = 0
        for v in ok_videos:
            fr = ds.frames(v, a.part)
            missing += sum(1 for f in fr if f.path is None or not f.path.exists())
        report["n_missing_images"] = missing
        print(f"[images] thieu {missing} anh trong {ds.images_dir(a.part)}")

    # --- 3. Thong ke ---
    df = ds.build_interim(a.part, ok_videos) if (a.rebuild or not ds.interim_parquet(a.part).exists()
                                                  or a.videos) else ds.gt_dataframe(a.part)
    df = df[df.video.isin(ok_videos)]
    cls_stats = stats_report(df)
    cls_stats.to_csv(OUT_DIR / f"{a.part}_class_stats.csv", index=False)
    dts = dt_stats(ds, a.part, ok_videos)
    dts.to_csv(OUT_DIR / f"{a.part}_dt_stats.csv", index=False)
    (OUT_DIR / f"{a.part}_schema_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print("\nTHONG KE THEO LOP (box_track_20, part =", a.part, ")")
    print(cls_stats.to_string(index=False))
    print("\nKHOANG THOI GIAN GIUA FRAME")
    print(f"  nguon dt: {dts.dt_source.value_counts().to_dict()}  median dt = {dts.dt_median_s.median():.4f} s "
          f"(danh nghia {ds.dt_nominal:.4f} s);  video co khoang > 1 buoc: {(dts.n_intervals_gt1_step > 0).sum()}; "
          f"khoang le: {dts.n_intervals_irregular.sum()}")
    print(f"\n  Da luu: {OUT_DIR / (a.part + '_class_stats.csv')}, {OUT_DIR / (a.part + '_dt_stats.csv')}")

    # --- 4. Manifest --- (du lieu o --data-root khac mac dinh -> tien to 'fixture_' de khong lan voi du lieu that)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    pre = "fixture_" if a.data_root else ""
    if a.part == "val":
        m = {"name": "bdd100k_final_val", "part": "val", "n_videos": len(ok_videos),
             "note": "Tap danh gia CUOI (val cong khai). KHONG dung de tuning.", "videos": ok_videos}
        (MANIFEST_DIR / f"{pre}bdd100k_final_val.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        print(f"  Manifest final: {MANIFEST_DIR / (pre + 'bdd100k_final_val.json')} ({len(ok_videos)} video)")
    if a.make_manifest and a.part == "train":
        m = make_dev_manifest(df, a.n_dev, a.seed)
        (MANIFEST_DIR / f"{pre}bdd100k_dev.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
        print(f"  Manifest dev: {MANIFEST_DIR / (pre + 'bdd100k_dev.json')} ({m['n_videos']} video, seed {a.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
