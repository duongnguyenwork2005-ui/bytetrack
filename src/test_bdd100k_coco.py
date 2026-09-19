"""
test_bdd100k_coco.py -- Adapter nhan BDD100K-MOT dang **COCO-video**
(`src/mot_datasets/bdd100k_coco.py`), viet cho review vong 5 muc 5.

Chay tren FIXTURE tong hop trong thu muc tam - khong can file nhan that 117 MB, khong dung cham
du lieu nghien cuu.

  1. `check_schema` FAIL-LOUD khi thieu khoa bat buoc hoac gap category la.
  2. Doc dung: so video / frame / annotation, toa do xywh, anh xa category.
  3. **ID THEO VIDEO**: `instance_id` dung chung giua cac video (do tren nhan that: 327/392) ->
     track_id phai duoc danh so lai theo (video, instance_id), KHONG duoc gop.
  4. Truong KHONG co trong nguon duoc ghi ro la THIEU, KHONG bi bia gia tri:
     occluded/truncated = None, occlusion_ratio/visibility = NaN, occlusion_source = "MISSING",
     dt_source = "nominal".
  5. `crowd` gom ca `iscrowd` LAN `ignore`; hai co goc van duoc giu rieng.
  6. `df_for_gt_export` dien visibility = 1.0 (PLACEHOLDER cho dinh dang MOTChallenge) nhung ban
     interim VAN giu NaN.
  7. `frames()` tra ve frame 1-based theo `frame_id` tang dan, `time_s` theo fps danh nghia.

    python src/test_bdd100k_coco.py
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines()[-6:]:
            print(f"         {line}")
    assert ok, name


def fixture() -> dict:
    """2 video x 3 frame. `instance_id` 1 va 2 DUNG CHUNG o ca hai video (giong nhan that)."""
    cats = [{"supercategory": "vehicle", "id": 3, "name": "car"},
            {"supercategory": "human", "id": 1, "name": "pedestrian"}]
    videos = [{"id": 1, "name": "vidA"}, {"id": 2, "name": "vidB"}]
    images, anns, aid = [], [], 0
    for vid, vname in ((1, "vidA"), (2, "vidB")):
        for f in range(3):
            iid = vid * 10 + f
            images.append({"id": iid, "video_id": vid, "frame_id": f, "width": 1280, "height": 720,
                           "file_name": f"{vname}/{vname}-{f + 1:07d}.jpg"})
            for inst in (1, 2):                       # <- CUNG instance_id o ca hai video
                aid += 1
                anns.append({"id": aid, "image_id": iid, "category_id": 3 if inst == 1 else 1,
                             "instance_id": inst, "iscrowd": 1 if (vid == 2 and inst == 2) else 0,
                             "ignore": 1 if (vid == 1 and f == 2 and inst == 2) else 0,
                             "bbox": [10.0 * inst + f, 20.0, 30.0, 40.0]})
    return {"categories": cats, "videos": videos, "images": images, "annotations": anns}


def main() -> int:
    failed = 0
    tmp = Path(tempfile.mkdtemp(prefix="bdd_coco_"))
    try:
        from mot_datasets.bdd100k_coco import (BDD100KCocoDataset, SchemaError, check_schema,
                                               df_for_gt_export, parse_coco)

        d = fixture()

        # --- 1. fail-loud ---
        loi = []
        for mo_ta, hong in (("thieu muc 'videos'", {k: v for k, v in d.items() if k != "videos"}),
                            ("annotations thieu 'instance_id'",
                             {**d, "annotations": [{k: v for k, v in a.items() if k != "instance_id"}
                                                   for a in d["annotations"]]}),
                            ("category la", {**d, "categories": [{"id": 9, "name": "ufo"}]})):
            try:
                check_schema(hong)
                loi.append(mo_ta)
            except SchemaError:
                pass
        _check("check_schema FAIL-LOUD voi nguon hong (thieu muc / thieu khoa / category la)",
               not loi, f"khong bao loi voi: {loi}")
        _check("check_schema chap nhan nguon dung", check_schema(d)["n_videos"] == 2)

        # --- 2. doc dung ---
        df = parse_coco(d, part="val")
        r0 = df[(df.video == "vidA") & (df.frame == 1) & (df.category == "car")].iloc[0]
        _check("doc dung so luong va toa do (bbox xywh -> bb_left/top/width/height, cx/cy)",
               len(df) == 12 and df.video.nunique() == 2
               and (r0.bb_left, r0.bb_top, r0.bb_width, r0.bb_height) == (10.0, 20.0, 30.0, 40.0)
               and (r0.cx, r0.cy) == (25.0, 40.0),
               f"n={len(df)} video={df.video.nunique()} box=({r0.bb_left},{r0.bb_top},"
               f"{r0.bb_width},{r0.bb_height}) tam=({r0.cx},{r0.cy})")

        # --- 3. ID THEO VIDEO (diem de sai nhat) ---
        chung = set(df[df.video == "vidA"].track_id) & set(df[df.video == "vidB"].track_id)
        n_track = df.track_id.nunique()
        _check("track_id danh so theo (video, instance_id): 2 instance_id dung chung o 2 video -> "
               "4 track RIENG, KHONG track nao dung chung giua hai video",
               n_track == 4 and not chung,
               f"n_track={n_track} (ky vong 4); track_id dung chung = {sorted(chung)}")

        # --- 4. truong thieu duoc ghi ro, khong bia ---
        _check("truong KHONG co trong nguon duoc ghi ro la THIEU (khong bia gia tri)",
               df.occluded.isna().all() and df.truncated.isna().all()
               and df.occlusion_ratio.isna().all() and df.visibility.isna().all()
               and (df.occlusion_source == "MISSING").all()
               and (df.dt_source == "nominal").all()
               and (df.label_source == "coco_video").all(),
               f"occlusion_ratio NaN: {df.occlusion_ratio.isna().all()}; "
               f"dt_source: {df.dt_source.unique().tolist()}")

        # --- 5. crowd = iscrowd HOAC ignore, giu rieng hai co goc ---
        n_crowd, n_isc, n_ign = int(df.crowd.sum()), int(df.iscrowd.sum()), int(df["ignore"].sum())
        _check("`crowd` gom ca iscrowd LAN ignore, hai co goc van duoc giu rieng",
               n_isc == 3 and n_ign == 1 and n_crowd == 4,
               f"crowd={n_crowd} iscrowd={n_isc} ignore={n_ign} (ky vong 4 / 3 / 1)")

        # --- 6. visibility placeholder chi o ban XUAT ---
        gt = df_for_gt_export(df)
        _check("`df_for_gt_export` dien visibility = 1.0 (PLACEHOLDER cho dinh dang MOTChallenge) "
               "nhung ban interim VAN giu NaN",
               (gt.visibility == 1.0).all() and df.visibility.isna().all(),
               f"export: {gt.visibility.unique().tolist()}; interim NaN: {df.visibility.isna().all()}")

        # --- 7. frames() ---
        lab = tmp / "labels.json"
        lab.write_text(json.dumps(d), encoding="utf-8")
        ds = BDD100KCocoDataset(lab, root=tmp / "khong_co_anh")
        fr = ds.frames("vidA", "val")
        _check("frames(): 1-based theo frame_id tang dan, time_s theo fps danh nghia (5 fps)",
               [f.index for f in fr] == [1, 2, 3]
               and np.allclose([f.time_s for f in fr], [0.0, 0.2, 0.4])
               and all(f.path is None for f in fr)      # khong co anh -> path None, khong nem loi
               and ds.list_videos("val") == ["vidA", "vidB"],
               f"index={[f.index for f in fr]} time_s={[round(f.time_s, 3) for f in fr]}")
    except AssertionError:
        failed += 1
    except Exception as e:  # noqa: BLE001
        failed += 1
        print(f"  [FAIL] loi khong mong doi: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
