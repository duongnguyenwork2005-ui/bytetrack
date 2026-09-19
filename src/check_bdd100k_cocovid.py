"""
check_bdd100k_cocovid.py -- Kiem ke TOAN BO nguon nhan BDD100K-MOT dang COCO-video va doi chieu
voi anh tren dia, TRUOC khi khoa cau hinh thi nghiem.

Nguyen tac: KHONG am tham loai video thieu/loi khoi tap danh gia. Moi van de duoc liet ke tuong minh;
CLI tra ma loi != 0 neu co van de chan viec chay thi nghiem.

Kiem:
  1. Schema COCO-video (fail-loud) + tong so video/anh/annotation.
  2. Tham chieu: moi annotation tro toi mot `image_id` co that; moi anh tro toi mot `video_id` co that.
  3. `file_name` khop quy uoc `<video>/<video>-<7 chu so>.jpg` va khop ANH TREN DIA.
  4. Timeline: frame_id lien tuc hay co khoang bi bo qua; frame khong co annotation.
  5. ID theo video: `instance_id` dung chung giua cac video (bay cua nguon nay).
  6. bbox hop le (w, h > 0; nam trong khung anh 1280x720).
  7. Lop: chi 8 lop danh gia; thong ke theo lop, dac biet `motorcycle`.
  8. Vung bo qua: `iscrowd` / `ignore` dem RIENG (khong cong gop).

    python src/check_bdd100k_cocovid.py --labels <...json> [--out results/phase2/bdd_cocovid_check]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from mot_datasets.bdd100k_coco import (BDD100KCocoDataset, MISSING_FIELDS, check_schema,  # noqa: E402
                                       frame_timeline, parse_coco)

W, H = 1280, 720


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json")
    ap.add_argument("--part", default="val")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default="results/phase2/bdd_cocovid_check")
    ap.add_argument("--expect-videos", type=int, default=200)
    ap.add_argument("--expect-images", type=int, default=39973)
    ap.add_argument("--no-images", action="store_true", help="bo qua doi chieu anh tren dia")
    a = ap.parse_args()

    out = config.PROJECT_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    ds = BDD100KCocoDataset(a.labels, a.data_root)
    d = ds._load()
    stat = check_schema(d)
    van_de: list[str] = []

    print(f"[schema]  OK: {stat}")
    print(f"[schema]  THIEU so voi box_track_20 native: {list(MISSING_FIELDS)}")
    if stat["n_videos"] != a.expect_videos:
        van_de.append(f"so video {stat['n_videos']} != ky vong {a.expect_videos}")
    if stat["n_images"] != a.expect_images:
        van_de.append(f"so anh {stat['n_images']} != ky vong {a.expect_images}")
    print(f"[kiem ke] video {stat['n_videos']}/{a.expect_videos}, anh {stat['n_images']}/{a.expect_images}, "
          f"annotation {stat['n_annotations']}")

    # --- 2. tham chieu ---
    vid_ids = {v["id"] for v in d["videos"]}
    img = {i["id"]: i for i in d["images"]}
    ann_mo_coi = [a_["id"] for a_ in d["annotations"] if a_["image_id"] not in img]
    anh_mo_coi = [i["id"] for i in d["images"] if i["video_id"] not in vid_ids]
    if ann_mo_coi:
        van_de.append(f"{len(ann_mo_coi)} annotation tro toi image_id khong ton tai")
    if anh_mo_coi:
        van_de.append(f"{len(anh_mo_coi)} anh tro toi video_id khong ton tai")
    print(f"[tham chieu] annotation mo coi {len(ann_mo_coi)}, anh mo coi {len(anh_mo_coi)}")

    # --- 3 + 4. ten file, anh tren dia, timeline ---
    vid2name = {v["id"]: v["name"] for v in d["videos"]}
    tl = frame_timeline(d)
    rows, ten_sai, thieu_anh_tong = [], 0, 0
    for v in sorted(vid2name.values()):
        info = tl.get(v, {"n_images": 0, "first": 0, "last": 0, "missing_frames": []})
        n_disk = -1
        if not a.no_images:
            dd = ds.images_dir(a.part) / v
            n_disk = len(list(dd.glob("*.jpg"))) if dd.is_dir() else 0
            thieu = [f for f in ds.frames(v, a.part) if f.path is None]
            thieu_anh_tong += len(thieu)
        rows.append({"video": v, "n_images_label": info["n_images"], "frame_first": info["first"],
                     "frame_last": info["last"], "n_missing_frames": len(info["missing_frames"]),
                     "n_images_disk": n_disk})
    for im in d["images"]:
        v = vid2name.get(im["video_id"], "")
        if im["file_name"] != f"{v}/{v}-{int(im['frame_id']) + 1:07d}.jpg":
            ten_sai += 1
    if ten_sai:
        van_de.append(f"{ten_sai} anh co `file_name` khong dung quy uoc <video>/<video>-<7 chu so>.jpg")
    if thieu_anh_tong:
        van_de.append(f"thieu {thieu_anh_tong} anh tren dia")
    df_v = pd.DataFrame(rows)
    lech_disk = df_v[(df_v.n_images_disk >= 0) & (df_v.n_images_disk != df_v.n_images_label)]
    if len(lech_disk):
        van_de.append(f"{len(lech_disk)} video co so anh tren dia != so anh trong nhan")
    print(f"[anh]     file_name sai quy uoc: {ten_sai}; thieu tren dia: "
          f"{'(bo qua)' if a.no_images else thieu_anh_tong}; video lech so anh: {len(lech_disk)}")
    n_gap = int((df_v.n_missing_frames > 0).sum())
    print(f"[timeline] {n_gap} video co khoang frame BI BO QUA; "
          f"do dai: min {df_v.frame_last.min()} / max {df_v.frame_last.max()}")

    # --- 5. ID theo video ---
    img2vid = {i["id"]: i["video_id"] for i in d["images"]}
    inst2vid = collections.defaultdict(set)
    for an in d["annotations"]:
        if an["image_id"] in img2vid:
            inst2vid[an["instance_id"]].add(img2vid[an["image_id"]])
    dung_chung = sum(1 for s in inst2vid.values() if len(s) > 1)
    print(f"[ID]      {len(inst2vid)} instance_id, {dung_chung} dung chung o NHIEU video "
          f"-> BAT BUOC danh so theo (video, instance_id)")

    # --- 6 + 7 + 8. bbox / lop / vung bo qua (qua adapter, dung duong chay that) ---
    df = parse_coco(d, part=a.part)
    xau = df[(df.bb_width <= 0) | (df.bb_height <= 0)]
    ngoai = df[(df.bb_left < -1) | (df.bb_top < -1)
               | (df.bb_left + df.bb_width > W + 1) | (df.bb_top + df.bb_height > H + 1)]
    if len(xau):
        van_de.append(f"{len(xau)} bbox co w/h <= 0")
    print(f"[bbox]    w/h <= 0: {len(xau)}; vuot khung {W}x{H}: {len(ngoai)} "
          f"({100 * len(ngoai) / max(len(df), 1):.2f} %)")

    cls = df.category.value_counts()
    n_frame_khong_ann = int(stat["n_images"] - df.groupby(["video", "frame"]).ngroups)
    print(f"[lop]     " + ", ".join(f"{k}={v}" for k, v in cls.items()))
    print(f"[bo qua]  iscrowd={int(df.iscrowd.sum())}, ignore={int(df['ignore'].sum())} "
          f"(dem RIENG, khong cong gop; crowd hop nhat={int(df.crowd.sum())})")
    print(f"[frame]   {n_frame_khong_ann} anh KHONG co annotation "
          f"({100 * n_frame_khong_ann / max(stat['n_images'], 1):.2f} %) - VAN thuoc timeline hop le")

    df_v.to_csv(out / f"{a.part}_video_inventory.csv", index=False)
    cls.rename("n_boxes").to_csv(out / f"{a.part}_class_stats.csv")
    bao_cao = {"labels": str(a.labels), "part": a.part, **stat,
               "n_missing_images_on_disk": (None if a.no_images else thieu_anh_tong),
               "n_videos_with_frame_gaps": n_gap, "n_filename_mismatch": ten_sai,
               "n_instance_id": len(inst2vid), "n_instance_id_shared_across_videos": dung_chung,
               "n_bbox_nonpositive": int(len(xau)), "n_bbox_outside_frame": int(len(ngoai)),
               "n_images_without_annotation": n_frame_khong_ann,
               "iscrowd": int(df.iscrowd.sum()), "ignore": int(df["ignore"].sum()),
               "class_counts": {k: int(v) for k, v in cls.items()},
               "missing_fields": list(MISSING_FIELDS), "van_de": van_de}
    (out / f"{a.part}_report.json").write_text(json.dumps(bao_cao, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8")
    print(f"\n-> {out}")
    if van_de:
        print("[VAN DE] " + "; ".join(van_de))
        return 1
    print("[OK]     khong phat hien van de chan viec chay thi nghiem")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
