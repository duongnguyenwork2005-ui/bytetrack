"""
bdd100k_mot.py -- Adapter BDD100K-MOT (box_track_20, dinh dang Scalabel) cho pipeline.

===========================================================================
1. NGUON VA SCHEMA (da doi chieu tai lieu chinh thuc, xac minh lai tren du lieu that)
===========================================================================
Goi tai (dang ky tai bdd-data.berkeley.edu): "MOT 2020 Images" + "MOT 2020 Labels".
Layout ky vong (chinh la layout goc sau khi giai nen; doi bang --data-root):

    <root>/labels/box_track_20/{train,val}/<video>.json
    <root>/images/track/{train,val,test}/<video>/<video>-<7 chu so>.jpg

Mot file JSON = list frame (theo frameIndex tang dan):
    {"name": str, "videoName": str, "frameIndex": int, ["timestamp": int ms],
     "labels": [{"id": str, "category": str,
                 "attributes": {"occluded": bool, "truncated": bool, "crowd": bool},
                 "box2d": {"x1", "y1", "x2", "y2"}}]}

    - 5 fps (lay mau lai tu video 30 Hz), ~200 frame / 40 s, 1280 x 720.
    - 8 lop danh gia: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle.
    - 3 lop distractor (bo qua): other person, trailer, other vehicle.
    - `crowd` = box bao mot NHOM doi tuong khong tach rieng -> vung bo qua khi danh gia
      (FP chua khop co IoA > 0.5 voi vung nay bi loai - trackeval/datasets/bdd100k.py).
    - `occluded` la CO NHI PHAN "bi che mot phan" - KHONG phai muc do che. Doi tuong bi
      che HOAN TOAN thi KHONG duoc annotate -> cung `id` xuat hien lai sau mot GAP.
      (Khac UA-DETRAC: annotate ca khi che 100% + co occlusion_ratio.)
    - rider = NGUOI ngoi tren xe hai banh, box RIENG; xe (motorcycle/bicycle) box rieng.
      Khong gop nguoi voi xe.
    - Khong co truong rieng cho ego-motion; camera gan tren xe (khac camera co dinh).

Schema CU (TrackEval pip 1.3.0 `datasets/bdd100k.py` dung `index`, `attributes.Crowd`)
-> check_schema() FAIL LOUD neu gap, de khong am tham doc sai.

===========================================================================
2. SCHEMA INTERIM CHUNG (giong UA-DETRAC de tai dung occlusion_segments / occlusion_eval)
===========================================================================
    video, frame (= frameIndex + 1, 1-based, LIEN TUC), frame_name, time_s, track_id (int),
    left, top, width, height, cx, cy, category, vehicle_type (= category),
    occluded, truncated, crowd (bool),
    occlusion_ratio: 0.5 neu occluded else 0.0   <- GIA TRI QUY UOC: "che mot phan, muc do
                     khong biet". Chon 0.5 de vuot nguong partial (0.10) nhung KHONG BAO GIO
                     vuot nguong full (0.90): trong BDD khong ton tai doan "che gan hoan toan
                     ma van co annotation"; su kien che hoan toan la GAP (xem occlusion_eval --gaps).
    occlusion_source: "unknown", visibility: 1 - occlusion_ratio

===========================================================================
3. XUAT MOTChallenge (theo lop) - danh gia bang run_trackeval --dataset-class mot_crowd
===========================================================================
    target = "vehicle": car + truck + bus + motorcycle + bicycle -> class 1 (danh gia CHINH,
             lien tuc voi UA-DETRAC "theo vet phuong tien"); train (tau hoa) -> vung bo qua.
    target = "<category>": chi lop do -> class 1 (bao cao RIENG o to / xe may).
    Vung bo qua (class 13 'crowd' cua bang MOT, conf = 0): moi box crowd (bat ky lop),
    3 lop distractor, va (voi target=vehicle) box `train`.
    Lop KHONG phai target, khong crowd: bo (giao thuc theo lop cua BDD).
    rider / pedestrian: chi thong ke, KHONG danh gia (detector COCO khong co lop 'rider').

    seqinfo.ini: frameRate 5, 1280x720, seqLength = so frame trong JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

from .base import DatasetSpec, FrameRef  # noqa: E402

EVAL_CATEGORIES = ("pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle")
DISTRACTOR_CATEGORIES = ("other person", "trailer", "other vehicle")
ALL_CATEGORIES = EVAL_CATEGORIES + DISTRACTOR_CATEGORIES
VEHICLE_CATEGORIES = ("car", "truck", "bus", "motorcycle", "bicycle")
SUPER_CATEGORIES = {"person": ("pedestrian", "rider"), "vehicle": ("car", "bus", "truck", "train"),
                    "bike": ("motorcycle", "bicycle")}
#: COCO id (yolov8n pretrained) -> lop BDD. person(0) KHONG map: COCO gop pedestrian + rider.
COCO_TO_BDD = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
BDD_TO_COCO = {v: k for k, v in COCO_TO_BDD.items()}
#: id class MOTChallenge dung cho vung bo qua (bang class_name_to_class_id cua TrackEval: 'crowd' = 13)
MOT_IGNORE_CLASS = 13
FPS = 5.0
IMG_W, IMG_H = 1280, 720
OCCLUDED_RATIO = 0.5       # gia tri quy uoc cho co `occluded` (xem docstring muc 2)

REQUIRED_FRAME_KEYS = ("name", "frameIndex", "labels")
REQUIRED_ATTR_KEYS = ("occluded", "truncated", "crowd")
OLD_SCHEMA_MARKERS = ("index", "Crowd", "Occluded", "Truncated")


class SchemaError(ValueError):
    pass


# ---------------------------------------------------------------------------
# 1. Kiem tra schema + doc mot video
# ---------------------------------------------------------------------------
def check_schema(frames: list, video_name: str = "?", strict_categories: bool = True) -> dict:
    """Kiem tra mot file JSON box_track_20. Tra ve thong ke nho; nem SchemaError neu sai.

    Fail loud (khong doan) khi: khong phai list; thieu name/frameIndex/labels; gap khoa cua
    schema cu (index / Crowd ...); frameIndex khong lien tuc tu 0; id trung trong 1 frame;
    category ngoai 11 lop; thieu box2d hoac box2d khong hop le; thieu attributes.
    """
    if not isinstance(frames, list) or not frames:
        raise SchemaError(f"{video_name}: JSON phai la list frame khong rong")
    stats = {"n_frames": len(frames), "n_labels": 0, "has_timestamp": all("timestamp" in f for f in frames),
             "categories": {}}
    for k, f in enumerate(frames):
        if not isinstance(f, dict):
            raise SchemaError(f"{video_name}: frame {k} khong phai dict")
        old = [m for m in OLD_SCHEMA_MARKERS if m in f]
        if old:
            raise SchemaError(f"{video_name}: gap khoa schema CU {old} (TrackEval bdd100k.py); can box_track_20")
        missing = [m for m in REQUIRED_FRAME_KEYS if m not in f]
        if missing:
            raise SchemaError(f"{video_name}: frame {k} thieu khoa {missing}")
        if not isinstance(f["frameIndex"], int) or f["frameIndex"] != k:
            raise SchemaError(f"{video_name}: frameIndex phai lien tuc tu 0 theo thu tu list "
                              f"(frame thu {k} co frameIndex={f['frameIndex']!r})")
        if not isinstance(f["labels"], list):
            raise SchemaError(f"{video_name}: frame {k} 'labels' phai la list")
        ids = set()
        for j, lb in enumerate(f["labels"]):
            for key in ("id", "category", "box2d"):
                if key not in lb:
                    raise SchemaError(f"{video_name}: frame {k} label {j} thieu '{key}'")
            if lb["id"] in ids:
                raise SchemaError(f"{video_name}: frame {k} co id trung {lb['id']!r}")
            ids.add(lb["id"])
            cat = lb["category"]
            if strict_categories and cat not in ALL_CATEGORIES:
                raise SchemaError(f"{video_name}: frame {k} category la {cat!r}, ngoai 11 lop box_track_20")
            attrs = lb.get("attributes")
            if not isinstance(attrs, dict):
                raise SchemaError(f"{video_name}: frame {k} label {j} thieu 'attributes'")
            old = [m for m in OLD_SCHEMA_MARKERS if m in attrs]
            if old:
                raise SchemaError(f"{video_name}: attributes dung khoa schema CU {old}")
            miss = [m for m in REQUIRED_ATTR_KEYS if m not in attrs]
            if miss:
                raise SchemaError(f"{video_name}: frame {k} label {j} attributes thieu {miss}")
            b = lb["box2d"]
            try:
                x1, y1, x2, y2 = (float(b[c]) for c in ("x1", "y1", "x2", "y2"))
            except (KeyError, TypeError, ValueError) as e:
                raise SchemaError(f"{video_name}: frame {k} label {j} box2d hong: {e}")
            if not (np.isfinite([x1, y1, x2, y2]).all() and x2 > x1 and y2 > y1):
                raise SchemaError(f"{video_name}: frame {k} label {j} box2d khong hop le {b}")
            stats["n_labels"] += 1
            stats["categories"][cat] = stats["categories"].get(cat, 0) + 1
    return stats


def parse_video(json_path: Path, part: str = "val") -> pd.DataFrame:
    """Doc mot video -> DataFrame theo schema interim chung (muc 2)."""
    frames = json.loads(Path(json_path).read_text(encoding="utf-8"))
    video = frames[0].get("videoName") or Path(json_path).stem
    check_schema(frames, video)
    has_ts = all("timestamp" in f for f in frames)
    t0 = float(frames[0]["timestamp"]) if has_ts else 0.0
    id_map: dict[str, int] = {}
    rows = []
    for f in frames:
        fi = int(f["frameIndex"])
        time_s = (float(f["timestamp"]) - t0) / 1000.0 if has_ts else fi / FPS
        for lb in f["labels"]:
            sid = str(lb["id"])
            if sid not in id_map:
                # id trong BDD la chuoi; anh xa sang int on dinh THEO THU TU XUAT HIEN trong video
                id_map[sid] = int(sid) if sid.isdigit() else len(id_map) + 1
            b = lb["box2d"]; a = lb["attributes"]
            x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
            occ = bool(a["occluded"])
            rows.append({"video": video, "frame": fi + 1, "frame_name": f["name"], "time_s": time_s,
                         "track_id": id_map[sid], "left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1,
                         "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "category": lb["category"],
                         "vehicle_type": lb["category"], "occluded": occ, "truncated": bool(a["truncated"]),
                         "crowd": bool(a["crowd"]), "occlusion_ratio": OCCLUDED_RATIO if occ else 0.0,
                         "occlusion_source": "unknown", "visibility": 1.0 - (OCCLUDED_RATIO if occ else 0.0),
                         "part": part, "dt_source": "timestamp" if has_ts else "nominal"})
    df = pd.DataFrame(rows)
    if len(df) and len(set(id_map.values())) != len(id_map):
        raise SchemaError(f"{video}: anh xa id -> int khong don anh (id so va id chuoi lan nhau)")
    return df


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------
class BDD100KMOTDataset(DatasetSpec):
    name = "bdd100k"
    fps = FPS
    img_w, img_h = IMG_W, IMG_H
    categories = ALL_CATEGORIES
    coco_to_dataset = dict(COCO_TO_BDD)
    category_to_coco = dict(BDD_TO_COCO)

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else config.RAW_DIR / "bdd100k"
        # root khac mac dinh (fixture, ban sao...) -> file interim mang ten thu muc goc de khong
        # lan voi du lieu that (vd bdd100k_mini_val_annotations.parquet)
        self.tag = "" if root is None else f"{self.root.name}_"

    def labels_dir(self, part: str) -> Path:
        return self.root / "labels" / "box_track_20" / part

    def images_dir(self, part: str) -> Path:
        return self.root / "images" / "track" / part

    def list_videos(self, part: str) -> list[str]:
        d = self.labels_dir(part)
        return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []

    def load_json(self, video: str, part: str) -> list:
        p = self.labels_dir(part) / f"{video}.json"
        if not p.exists():
            raise FileNotFoundError(p)
        return json.loads(p.read_text(encoding="utf-8"))

    def frames(self, video: str, part: str = "val") -> list[FrameRef]:
        """Danh sach frame tu JSON (thu tu frameIndex). time_s tu timestamp neu co, else frameIndex/5."""
        fr = self.load_json(video, part)
        check_schema(fr, video)
        has_ts = all("timestamp" in f for f in fr)
        t0 = float(fr[0]["timestamp"]) if has_ts else 0.0
        img_dir = self.images_dir(part) / video
        out = []
        for f in fr:
            fi = int(f["frameIndex"])
            t = (float(f["timestamp"]) - t0) / 1000.0 if has_ts else fi / FPS
            p = img_dir / f["name"]
            out.append(FrameRef(index=fi + 1, path=p if img_dir.is_dir() else None, time_s=t, name=f["name"]))
        return out

    def interim_parquet(self, part: str) -> Path:
        stem = f"{self.tag}{part}" if self.tag else f"bdd100k_{part}"
        return config.INTERIM_DIR / f"{stem}_annotations.parquet"

    def build_interim(self, part: str, videos: list[str] | None = None) -> pd.DataFrame:
        videos = videos or self.list_videos(part)
        if not videos:
            raise FileNotFoundError(f"Khong co file JSON nao trong {self.labels_dir(part)}")
        dfs = [parse_video(self.labels_dir(part) / f"{v}.json", part) for v in videos]
        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values(["video", "frame", "track_id"]).reset_index(drop=True)
        out = self.interim_parquet(part)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, index=False)
        return df

    def gt_dataframe(self, part: str = "val") -> pd.DataFrame:
        p = self.interim_parquet(part)
        if not p.exists():
            return self.build_interim(part)
        return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# 3. Thong ke
# ---------------------------------------------------------------------------
def track_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Khoang GAP (frame khong co annotation) cua tung (video, track_id): [gap_start, gap_end]."""
    rows = []
    for (v, tid), g in df.groupby(["video", "track_id"], sort=True):
        fr = np.sort(g["frame"].unique())
        d = np.diff(fr)
        for k in np.flatnonzero(d > 1):
            rows.append({"video": v, "track_id": tid, "last_seen": int(fr[k]), "reappear": int(fr[k + 1]),
                         "gap_start": int(fr[k] + 1), "gap_end": int(fr[k + 1] - 1), "gap_frames": int(d[k] - 1),
                         "category": g["category"].iloc[0]})
    return pd.DataFrame(rows, columns=["video", "track_id", "last_seen", "reappear", "gap_start", "gap_end",
                                       "gap_frames", "category"])


def stats_report(df: pd.DataFrame) -> pd.DataFrame:
    """Bang thong ke theo lop: video, track, box, do dai track, % occluded/truncated, crowd, gap."""
    rows = []
    gaps = track_gaps(df)
    for cat, g in df.groupby("category"):
        tl = g.groupby(["video", "track_id"])["frame"].nunique()
        gg = gaps[gaps.category == cat]
        rows.append({"category": cat, "n_videos": g["video"].nunique(), "n_tracks": len(tl),
                     "n_boxes": len(g), "track_len_min": int(tl.min()), "track_len_median": float(tl.median()),
                     "track_len_max": int(tl.max()), "pct_boxes_occluded": round(100 * g["occluded"].mean(), 2),
                     "pct_boxes_truncated": round(100 * g["truncated"].mean(), 2),
                     "n_crowd_boxes": int(g["crowd"].sum()), "n_tracks_with_gap": gg["track_id"].nunique() if len(gg) else 0,
                     "n_gaps": len(gg), "gap_frames_median": float(gg["gap_frames"].median()) if len(gg) else np.nan,
                     "evaluated": cat in EVAL_CATEGORIES})
    return pd.DataFrame(rows).sort_values("n_boxes", ascending=False).reset_index(drop=True)


def dt_stats(ds: BDD100KMOTDataset, part: str, videos: list[str]) -> pd.DataFrame:
    """Phan bo khoang thoi gian giua frame (giay) tung video: nguon, median, min, max, so khoang != danh nghia."""
    rows = []
    for v in videos:
        fr = ds.frames(v, part)
        t = np.array([f.time_s for f in fr])
        d = np.diff(t) if len(t) > 1 else np.array([])
        src = "timestamp" if any(("timestamp" in x) for x in ds.load_json(v, part)[:1]) else "nominal"
        irregular = int(np.sum(np.abs(d / ds.dt_nominal - np.round(d / ds.dt_nominal)) > 0.25)) if len(d) else 0
        skipped = int(np.sum(np.round(d / ds.dt_nominal) > 1)) if len(d) else 0
        rows.append({"video": v, "n_frames": len(fr), "dt_source": src,
                     "dt_median_s": float(np.median(d)) if len(d) else np.nan,
                     "dt_min_s": float(d.min()) if len(d) else np.nan, "dt_max_s": float(d.max()) if len(d) else np.nan,
                     "n_intervals_gt1_step": skipped, "n_intervals_irregular": irregular})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 4. Xuat MOTChallenge
# ---------------------------------------------------------------------------
def split_name_for(part: str, target: str, benchmark: str = "BDD") -> str:
    """Ten split MOTChallenge: BDD-val (vehicle) / BDD-val_car ... `benchmark` khac de tach fixture."""
    return f"{benchmark}-{part}" if target == "vehicle" else f"{benchmark}-{part}_{target}"


def _write_seqinfo(seq_dir: Path, video: str, seq_len: int):
    text = ("[Sequence]\n" f"name={video}\n" "imDir=img1\n" f"frameRate={int(FPS)}\n" f"seqLength={seq_len}\n"
            f"imWidth={IMG_W}\n" f"imHeight={IMG_H}\n" "imExt=.jpg\n")
    (seq_dir / "seqinfo.ini").write_text(text, encoding="utf-8")


def export_motchallenge(df: pd.DataFrame, part: str, target: str = "vehicle",
                        seq_lengths: dict[str, int] | None = None, out_root: Path | None = None,
                        videos: list[str] | None = None, benchmark: str = "BDD") -> dict:
    """Xuat GT theo lop (muc 3). Tra ve {split_name, n_videos, n_eval_rows, n_ignore_rows}."""
    if target != "vehicle" and target not in EVAL_CATEGORIES:
        raise ValueError(f"target phai la 'vehicle' hoac mot trong {EVAL_CATEGORIES}")
    out_root = Path(out_root) if out_root else config.PROCESSED_DIR
    split = split_name_for(part, target, benchmark)
    targets = set(VEHICLE_CATEGORIES) if target == "vehicle" else {target}
    ignore_cats = set(DISTRACTOR_CATEGORIES) | ({"train"} if target == "vehicle" else set())
    videos = videos or sorted(df["video"].unique())
    n_eval = n_ign = 0
    (out_root / "seqmaps").mkdir(parents=True, exist_ok=True)
    seqmap = ["name"]
    for v in videos:
        g = df[df["video"] == v]
        seq_dir = out_root / split / v
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        is_ignore = g["crowd"] | g["category"].isin(ignore_cats)
        is_eval = g["category"].isin(targets) & ~g["crowd"]
        rows = []
        for _, r in g[is_eval | is_ignore].iterrows():
            ign = bool(r["crowd"]) or r["category"] in ignore_cats
            rows.append((int(r["frame"]), int(r["track_id"]), r["left"], r["top"], r["width"], r["height"],
                         0 if ign else 1, MOT_IGNORE_CLASS if ign else config.MOT_CLASS_ID,
                         0.0 if ign else float(r["visibility"])))
            n_ign += ign; n_eval += (not ign)
        gt = pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"])
        gt = gt.sort_values(["frame", "id"])
        gt.to_csv(seq_dir / "gt" / "gt.txt", header=False, index=False, float_format="%.3f")
        seq_len = (seq_lengths or {}).get(v) or int(g["frame"].max())
        _write_seqinfo(seq_dir, v, seq_len)
        seqmap.append(v)
    (out_root / "seqmaps" / f"{split}.txt").write_text("\n".join(seqmap) + "\n", encoding="utf-8")
    vt = (df[df["video"].isin(videos)].groupby(["video", "track_id"])["category"].first().reset_index())
    vt.to_csv(out_root / split / "vehicle_types.csv", index=False)
    return {"split_name": split, "n_videos": len(videos), "n_eval_rows": int(n_eval), "n_ignore_rows": int(n_ign)}


def split_tracker_output_by_class(tracker_dir: Path, out_root: Path, part: str,
                                  targets: list[str], majority_vote: bool = True,
                                  benchmark: str = "BDD") -> dict:
    """Tach output tracker (data/*.txt + data_cls/*.csv, lop COCO) thanh cac split theo lop BDD.

    Mot track co the doi lop giua cac frame (STrack.cls bi ghi de moi lan khop) -> mac dinh
    gan MOT lop cho ca track theo DA SO (majority vote) de khong xe track thanh nhieu manh.
    target 'vehicle' = moi lop trong VEHICLE_CATEGORIES.
    """
    tracker_dir = Path(tracker_dir)
    name = tracker_dir.name
    written = {}
    for txt in sorted((tracker_dir / "data").glob("*.txt")):
        video = txt.stem
        cls_csv = tracker_dir / "data_cls" / f"{video}.csv"
        if not cls_csv.exists():
            raise FileNotFoundError(f"Thieu {cls_csv} (runner phai ghi data_cls)")
        det = pd.read_csv(txt, header=None, names=["frame", "id", "x", "y", "w", "h", "conf"]) if txt.stat().st_size else \
            pd.DataFrame(columns=["frame", "id", "x", "y", "w", "h", "conf"])
        cl = pd.read_csv(cls_csv)
        cl["bdd"] = cl["cls"].map(COCO_TO_BDD)
        if majority_vote and len(cl):
            maj = cl.groupby("id")["bdd"].agg(lambda s: s.value_counts().idxmax() if s.notna().any() else None)
            det = det.merge(maj.rename("bdd"), left_on="id", right_index=True, how="left")
        else:
            det = det.merge(cl[["frame", "id", "bdd"]], on=["frame", "id"], how="left")
        for target in targets:
            keep = det["bdd"].isin(VEHICLE_CATEGORIES) if target == "vehicle" else det["bdd"].eq(target)
            split = split_name_for(part, target, benchmark)
            d = out_root / "trackers" / split / name / "data"
            d.mkdir(parents=True, exist_ok=True)
            det.loc[keep, ["frame", "id", "x", "y", "w", "h", "conf"]].to_csv(d / f"{video}.txt", header=False, index=False)
            written[split] = written.get(split, 0) + int(keep.sum())
    return written
