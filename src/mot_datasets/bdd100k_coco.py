"""
bdd100k_coco.py -- Adapter RIENG cho nhan BDD100K-MOT o dinh dang **COCO-video**
(`bdd_box_track_val_cocofmt.json`), KHONG phai `box_track_20` native.

TAI SAO CO FILE NAY: nhan tren may la ban CHUYEN DOI sang COCO-video. Ep no qua
`bdd100k_mot.BDD100KMOTDataset` (cho dinh dang Scalabel) la sai: schema khac han va
`check_schema` se fail-loud. Adapter nay doc dung dinh dang co that, GIU nguyen thong tin co,
va GHI RO thong tin KHONG co - thay vi bia gia tri.

===========================================================================
CO GI / KHONG CO GI  (so voi box_track_20 native)
===========================================================================
CO   : videos[].name, images[].{video_id, frame_id, file_name, width, height},
       annotations[].{instance_id, category_id, bbox (xywh), iscrowd, ignore}
       -> du de: theo vet theo tung video, danh gia theo lop, va vung bo qua.
KHONG: `attributes.occluded`, `attributes.truncated`, `timestamp`.
       -> occluded/truncated/occlusion_ratio/visibility = **None/NaN** (KHONG bia 0.5 nhu ban
          native, vi o day khong co ca CO NHI PHAN de suy ra).
       -> `dt_source` = "nominal": delta t lay theo 5 fps danh nghia, KHONG co timestamp that.

HE QUA PHAI GHI RO KHI BAO CAO:
  - KHONG chay duoc phan tang che khuat (`occlusion_eval` theo occlusion_ratio) tren nguon nay.
  - Smoke chay bang nhan nay KHONG kiem chung duong nhan NATIVE; no kiem chung duong chay
    cache -> tracking -> evaluator.

===========================================================================
ID THEO VIDEO - diem de sai nhat
===========================================================================
`instance_id` trong file nay **dung chung giua cac video** (do duoc: 327/392 instance_id xuat hien
o nhieu hon mot video). Neu lay thang lam track_id thi track cua cac video khac nhau se bi gop.
Adapter nay danh so lai theo **(video, instance_id)** - xem `parse_coco()` va
`src/test_bdd100k_coco.py` phep kiem 3.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

from .base import FrameRef  # noqa: E402
from .bdd100k_mot import EVAL_CATEGORIES, FPS  # noqa: E402

#: Truong co trong box_track_20 native nhung KHONG co trong nguon COCO-video nay.
MISSING_FIELDS = ("occluded", "truncated", "timestamp")

#: Khoa bat buoc cua tung muc - thieu thi fail-loud thay vi doc sai am tham.
REQUIRED = {"videos": {"id", "name"},
            "images": {"id", "video_id", "frame_id", "file_name"},
            "annotations": {"id", "image_id", "category_id", "instance_id", "bbox"},
            "categories": {"id", "name"}}


class SchemaError(ValueError):
    """Nguon khong dung dinh dang COCO-video cua BDD100K-MOT."""


def check_schema(d: dict) -> dict:
    """Fail-loud neu thieu khoa bat buoc. Tra ve thong ke ngan."""
    for muc, khoa in REQUIRED.items():
        if muc not in d:
            raise SchemaError(f"thieu muc {muc!r} o cap cao nhat (co: {sorted(d)})")
        if not d[muc]:
            raise SchemaError(f"muc {muc!r} rong")
        thieu = khoa - set(d[muc][0])
        if thieu:
            raise SchemaError(f"{muc}[0] thieu khoa {sorted(thieu)} (co: {sorted(d[muc][0])})")
    ten = {c["name"] for c in d["categories"]}
    la = ten - set(EVAL_CATEGORIES)
    if la:
        raise SchemaError(f"category la (khong thuoc 8 lop danh gia cua BDD): {sorted(la)}")
    return {"n_videos": len(d["videos"]), "n_images": len(d["images"]),
            "n_annotations": len(d["annotations"]), "n_categories": len(d["categories"])}


class BDD100KCocoDataset:
    """Nhan BDD100K-MOT dang COCO-video + anh `images/track/<part>/<video>/`.

    Cung be mat voi `BDD100KMOTDataset` o nhung ham pipeline dung (`list_videos`, `frames`,
    `gt_dataframe`, `ignored_regions`, `orig_shape`, `dt_nominal`), de `det_cache` va
    `run_tracking` dung duoc ma khong phai sua.
    """

    name = "bdd100k_coco"
    dt_nominal = 1.0 / FPS
    orig_shape = (720, 1280)

    def __init__(self, labels_json, root=None, tag: str = "bddcoco_"):
        self.labels_json = Path(labels_json)
        self.root = Path(root) if root else config.RAW_DIR / "bdd100k"
        self.tag = tag
        self._d: dict | None = None

    # ------------------------------------------------------------------ doc nguon
    def _load(self) -> dict:
        if self._d is None:
            if not self.labels_json.is_file():
                raise FileNotFoundError(f"khong co file nhan COCO-video: {self.labels_json}")
            self._d = json.loads(self.labels_json.read_text(encoding="utf-8"))
            check_schema(self._d)
        return self._d

    def images_dir(self, part: str = "val") -> Path:
        return self.root / "images" / "track" / part

    def list_videos(self, part: str = "val") -> list[str]:
        return sorted(v["name"] for v in self._load()["videos"])

    def frames(self, video: str, part: str = "val") -> list[FrameRef]:
        """Frame cua mot video theo `frame_id` tang dan.

        `time_s` = frame_id / FPS - nguon KHONG co `timestamp`, xem MISSING_FIELDS.
        """
        d = self._load()
        vid = {v["name"]: v["id"] for v in d["videos"]}.get(video)
        if vid is None:
            raise KeyError(f"khong co video {video!r} trong {self.labels_json.name}")
        img_dir = self.images_dir(part)
        out = []
        for im in sorted((i for i in d["images"] if i["video_id"] == vid), key=lambda i: i["frame_id"]):
            p = img_dir / im["file_name"]        # file_name da gom <video>/<file>.jpg
            out.append(FrameRef(index=int(im["frame_id"]) + 1, path=p if p.exists() else None,
                                time_s=int(im["frame_id"]) / FPS, name=Path(im["file_name"]).name))
        return out

    def ignored_regions(self, video: str, part: str = "val") -> list:
        """BDD khong co vung bo qua tinh theo video (khac UA-DETRAC); vung bo qua la box crowd/ignore."""
        return []

    # ------------------------------------------------------------------ interim
    def interim_parquet(self, part: str) -> Path:
        return config.INTERIM_DIR / f"{self.tag}{part}_annotations.parquet"

    def build_interim(self, part: str = "val", videos: list[str] | None = None) -> pd.DataFrame:
        df = parse_coco(self._load(), part=part, videos=videos)
        out = self.interim_parquet(part)
        out.parent.mkdir(parents=True, exist_ok=True)
        if videos is None:
            df.to_parquet(out, index=False)
        return df

    def gt_dataframe(self, part: str = "val") -> pd.DataFrame:
        p = self.interim_parquet(part)
        return pd.read_parquet(p) if p.exists() else self.build_interim(part)


def parse_coco(d: dict, part: str = "val", videos: list[str] | None = None) -> pd.DataFrame:
    """COCO-video -> schema interim chung (cung ten cot voi `bdd100k_mot.parse_video`).

    track_id duoc danh so lai theo **(video, instance_id)**: `instance_id` trong nguon nay dung chung
    giua cac video, lay thang se GOP track cua cac video khac nhau.
    """
    check_schema(d)
    vid2name = {v["id"]: v["name"] for v in d["videos"]}
    cat2name = {c["id"]: c["name"] for c in d["categories"]}
    img = {i["id"]: i for i in d["images"]}
    keep = set(videos) if videos else None

    rows = []
    tid_map: dict[tuple[str, int], int] = {}
    for a in d["annotations"]:
        im = img[a["image_id"]]
        video = vid2name[im["video_id"]]
        if keep is not None and video not in keep:
            continue
        key = (video, int(a["instance_id"]))          # <-- ID THEO VIDEO
        tid = tid_map.setdefault(key, len(tid_map) + 1)
        x, y, w, h = (float(v) for v in a["bbox"])
        cat = cat2name[int(a["category_id"])]
        crowd = bool(a.get("iscrowd", 0)) or bool(a.get("ignore", 0))
        rows.append({
            "video": video, "frame": int(im["frame_id"]) + 1, "frame_name": Path(im["file_name"]).name,
            "time_s": int(im["frame_id"]) / FPS,
            "track_id": tid, "bb_left": x, "bb_top": y, "bb_width": w, "bb_height": h,
            "cx": x + w / 2, "cy": y + h / 2, "category": cat, "vehicle_type": cat,
            # --- KHONG CO trong nguon COCO-video: de None/NaN, KHONG bia gia tri ---
            "occluded": None, "truncated": None,
            "occlusion_ratio": np.nan, "visibility": np.nan, "occlusion_source": "MISSING",
            # `crowd` gom ca iscrowd LAN ignore: ca hai deu la "vung bo qua" khi danh gia.
            # Hai co goc duoc giu rieng de khong mat thong tin.
            "crowd": crowd, "iscrowd": bool(a.get("iscrowd", 0)), "ignore": bool(a.get("ignore", 0)),
            "part": part, "dt_source": "nominal", "label_source": "coco_video"})
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(["video", "frame", "track_id"]).reset_index(drop=True)
    return df


def df_for_gt_export(df: pd.DataFrame) -> pd.DataFrame:
    """Ban cho `export_motchallenge`: dien `visibility` = 1.0.

    QUAN TRONG: day la **PLACEHOLDER do DINH DANG MOTChallenge bat buoc cot `vis` phai la mot so**
    (de NaN thi TrackEval bao "Cannot convert gt data to float"). Nguon COCO-video KHONG co thong tin
    che khuat, nen 1.0 o day co nghia "KHONG BIET", **khong** phai "do duoc la nhin thay 100%".
    Ban interim (`parse_coco`) van giu NaN de khong mat su that. Moi phan tich dua tren visibility
    deu KHONG ap dung cho nguon nay.
    """
    out = df.copy()
    out["visibility"] = 1.0
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Kiem tra nhan BDD100K-MOT dang COCO-video + xuat GT MOTChallenge")
    ap.add_argument("--labels", required=True, help="duong dan file *_cocofmt.json")
    ap.add_argument("--part", default="val")
    ap.add_argument("--data-root", default=None, help="goc dataset (mac dinh data/raw/bdd100k)")
    ap.add_argument("--videos", nargs="*", default=None, help="chi xu ly cac video nay")
    ap.add_argument("--export-gt", action="store_true", help="xuat GT MOTChallenge")
    ap.add_argument("--benchmark", default="BDDSMOKE", help="tien to split GT (tach khoi BDD-val)")
    ap.add_argument("--no-images", action="store_true")
    a = ap.parse_args()

    ds = BDD100KCocoDataset(a.labels, a.data_root)
    stat = check_schema(ds._load())
    print(f"[schema] OK: {stat}")
    print(f"[schema] THIEU so voi box_track_20 native: {list(MISSING_FIELDS)} "
          f"-> occlusion_ratio/visibility = NaN, dt_source = 'nominal'")

    videos = a.videos or ds.list_videos(a.part)
    df = parse_coco(ds._load(), part=a.part, videos=videos)
    print(f"[du lieu] {df.video.nunique()} video, {len(df)} hang, "
          f"{df.track_id.nunique()} track (danh so theo (video, instance_id))")
    print(f"[lop]     {df.category.value_counts().to_dict()}")
    print(f"[bo qua]  crowd={int(df.crowd.sum())} (iscrowd={int(df.iscrowd.sum())}, "
          f"ignore={int(df['ignore'].sum())})")

    if not a.no_images:
        thieu = sum(1 for v in videos for f in ds.frames(v, a.part) if f.path is None)
        print(f"[anh]     thieu {thieu} anh trong {ds.images_dir(a.part)}")

    if a.export_gt:
        from .bdd100k_mot import export_motchallenge
        seq_len = {v: int(df[df.video == v].frame.max()) for v in videos}
        info = export_motchallenge(df_for_gt_export(df), a.part, "vehicle", seq_lengths=seq_len,
                                   videos=videos, benchmark=a.benchmark)
        print(f"[GT]      {info}")
        print("[GT]      LUU Y: cot `vis` trong gt.txt = 1.0 la PLACEHOLDER (dinh dang bat buoc mot so); "
              "nguon COCO-video khong co thong tin che khuat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
