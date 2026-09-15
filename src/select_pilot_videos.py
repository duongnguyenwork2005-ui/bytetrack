"""
select_pilot_videos.py -- Chon video pilot (tap DEV) theo TIEU CHI GT, truoc khi xem
ket qua phuong phap nao. Luu manifest de tai lap lua chon.

UA-DETRAC (part train, 60 video): dung data/interim/video_selection_stats.csv (Giai doan 1):
    pct_tracks_occluded, n_full_occ_segs, pct_curved, n_sharp, n_tracks, sence_weather.
Tieu chi: phan tang theo (muc che khuat: tercile cua n_full_occ_segs) x (do cong: tercile
cua pct_curved), chon ngau nhien deu trong tang (seed co dinh), uu tien da dang thoi tiet.
Tap TEST (40 video) KHONG duoc dung o day (danh cho danh gia cuoi).

    python src/select_pilot_videos.py --dataset detrac --n 12 --seed 0
    -> results/phase2/manifests/detrac_dev.json  (+ detrac_test.json = 40 video test)
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

MANIFEST_DIR = config.RESULTS_DIR / "phase2" / "manifests"


def select_detrac(n: int, seed: int) -> dict:
    stats = pd.read_csv(config.INTERIM_DIR / "video_selection_stats.csv")
    train = set(config.list_videos("train"))
    stats = stats[stats.video.isin(train)].copy()
    stats["occ_tercile"] = pd.qcut(stats["n_full_occ_segs"].rank(method="first"), 3, labels=[0, 1, 2])
    stats["curve_tercile"] = pd.qcut(stats["pct_curved"].rank(method="first"), 3, labels=[0, 1, 2])
    rng = np.random.default_rng(seed)
    strata = stats.groupby(["occ_tercile", "curve_tercile"], observed=True)
    quota = max(1, n // max(strata.ngroups, 1))
    chosen = []
    for _, grp in strata:
        k = min(quota, len(grp))
        # trong tang: uu tien thoi tiet chua co (da dang), roi ngau nhien
        picked = []
        pool = grp.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        seen = set(stats.loc[stats.video.isin(chosen), "sence_weather"])
        for _, r in pool.iterrows():
            if len(picked) >= k:
                break
            if r.sence_weather not in seen:
                picked.append(r.video); seen.add(r.sence_weather)
        for _, r in pool.iterrows():
            if len(picked) >= k:
                break
            if r.video not in picked:
                picked.append(r.video)
        chosen += picked
    rest = stats[~stats.video.isin(chosen)].sort_values("n_full_occ_segs", ascending=False)
    for v in rest.video:
        if len(chosen) >= n:
            break
        chosen.append(v)
    chosen = sorted(chosen[:n])
    sel = stats[stats.video.isin(chosen)]
    return {"name": "detrac_dev", "dataset": "detrac", "part": "train", "split_name": "DETRAC-all",
            "n_videos": len(chosen), "seed": seed,
            "criteria": "phan tang tercile(n_full_occ_segs) x tercile(pct_curved) tren 60 video train; "
                        "chon ngau nhien trong tang, uu tien da dang thoi tiet; bu theo n_full_occ_segs",
            "videos": chosen,
            "stats": sel[["video", "sence_weather", "n_tracks", "pct_tracks_occluded", "n_full_occ_segs",
                          "pct_curved", "n_sharp", "occ_tercile", "curve_tercile"]].astype(str).to_dict("records")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="detrac")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    if a.dataset != "detrac":
        print("BDD100K: dung src/check_bdd100k.py --part train --make-manifest"); return 1
    m = select_detrac(a.n, a.seed)
    (MANIFEST_DIR / "detrac_dev.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    test = config.list_videos("test")
    (MANIFEST_DIR / "detrac_test.json").write_text(json.dumps(
        {"name": "detrac_test", "dataset": "detrac", "part": "test", "split_name": "DETRAC-test",
         "n_videos": len(test), "note": "40 video test UA-DETRAC - danh gia CUOI, khong tuning", "videos": test},
        indent=2), encoding="utf-8")
    print(f"detrac_dev: {m['n_videos']} video (seed {a.seed}):")
    print(pd.DataFrame(m["stats"]).to_string(index=False))
    print(f"detrac_test: {len(test)} video")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
