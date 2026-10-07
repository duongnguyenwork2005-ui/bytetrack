"""
make_bdd5_train_fixture.py -- Sinh fixture TONG HOP gia lap BDD100K MOT split `train` (anh JPEG + nhan) de kiem
duong chuan bi dataset / huan luyen / danh gia detector 5 lop khi CHUA co du lieu that.

KHONG PHAI DU LIEU THAT: anh la nen nhieu + hinh chu nhat mau theo lop. Ket qua tren fixture chi chung minh
duong chay (doc/ghi, anh xa lop, toa do, manifest, checkpoint), KHONG noi gi ve chat luong detector.

Sinh CUNG noi dung o hai dinh dang nhan:
    <goc>/labels/box_track_20/train/<video>.json          (Scalabel, nhu goi MOT 2020 goc)
    <goc>/labels_cocovid/fx_box_track_train_cocofmt.json  (COCO-video: distractor -> ignore=1 voi lop car/truck/
                                                           pedestrian, crowd -> iscrowd=1)
    <goc>/images/track/train/<video>/<video>-<7 so>.jpg

Tinh huong co chu y (video 0 -- `ca_dac_biet=True`):
    - frame 3: chi co pedestrian (anh am tinh HOP LE cho 5 lop)
    - frame 4: khong co nhan nao nhung nam trong khoang co nhan (am tinh hop le)
    - 3 frame cuoi: KHONG co nhan (duoi video khong chac da gan nhan -> phai bi loai)
    - box car tran ra ngoai khung (phai cat), box bicycle < 1 px (phai bo)
    - vung crowd (car) de len mot car co nhan (khong duoc to xam pixel cua car do)
    - distractor 'other vehicle' / 'trailer' (to xam), 'other person', 'train', 'rider' (khong gan nhan)

    python src/mot_datasets/make_bdd5_train_fixture.py --out <thu muc> [--n-video 20 --n-frame 50 --w 1280 --h 720]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

#: mau BGR + kich thuoc tuong doi (w, h theo chieu rong anh) cua tung lop -- de detector hoc duoc tren fixture
KIEU = {"car": ((200, 80, 30), 0.09, 0.06), "truck": ((40, 40, 200), 0.16, 0.12), "bus": ((0, 200, 230), 0.22, 0.10),
        "motorcycle": ((40, 180, 40), 0.03, 0.05), "bicycle": ((200, 40, 200), 0.025, 0.04),
        "pedestrian": ((230, 230, 230), 0.015, 0.06), "rider": ((120, 120, 250), 0.015, 0.04),
        "train": ((90, 90, 90), 0.4, 0.15), "other vehicle": ((160, 160, 60), 0.08, 0.05),
        "trailer": ((60, 100, 160), 0.12, 0.08), "other person": ((210, 210, 210), 0.012, 0.05)}
COCO_CAT = ["pedestrian", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]
IGNORE_MAP = {"other vehicle": "car", "trailer": "truck", "other person": "pedestrian"}   # nhu bdd100k to_coco


def ten_video(seed: int, i: int) -> str:
    return f"fx{seed:02x}{i:06x}-{(i * 2654435761 + seed) & 0xFFFFFFFF:08x}"


def _lb(id_, cat, x1, y1, x2, y2, crowd=False):
    return {"id": str(id_), "category": cat, "attributes": {"occluded": False, "truncated": False, "crowd": crowd},
            "box2d": {"x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2)}}


def sinh_video(rng, video: str, n_frame: int, W: int, H: int, lop_video: list[str], dac_biet: bool) -> list[dict]:
    """Danh sach frame Scalabel cua mot video; doi tuong di chuyen tuyen tinh."""
    objs = []
    for k, cat in enumerate(lop_video):
        _, rw, rh = KIEU[cat]
        w, h = rw * W * rng.uniform(0.8, 1.3), rh * W * rng.uniform(0.8, 1.3)
        x, y = rng.uniform(0, W - w), rng.uniform(H * 0.35, H - h)
        vx = rng.uniform(-0.01, 0.01) * W
        objs.append([k + 1, cat, x, y, w, h, vx])
    frames = []
    for fi in range(n_frame):
        labels = []
        if dac_biet and fi == 3:
            labels = [_lb(900, "pedestrian", 0.1 * W, 0.5 * H, 0.12 * W, 0.6 * H)]
        elif dac_biet and (fi == 4 or fi >= n_frame - 3):
            labels = []
        else:
            for oid, cat, x, y, w, h, vx in objs:
                xx = (x + vx * fi) % max(1.0, W - w)
                labels.append(_lb(oid, cat, xx, y, xx + w, y + h))
            if dac_biet:
                labels.append(_lb(801, "car", W - 0.05 * W, 0.4 * H, W + 0.05 * W, 0.5 * H))          # tran khung
                labels.append(_lb(802, "bicycle", 0.3 * W, 0.3 * H, 0.3 * W + 0.4, 0.3 * H + 5))   # < 1 px
                labels.append(_lb(803, "car", 0.55 * W, 0.62 * H, 0.65 * W, 0.72 * H))              # nam trong crowd
                labels.append(_lb(804, "car", 0.5 * W, 0.6 * H, 0.8 * W, 0.8 * H, crowd=True))
                labels.append(_lb(805, "other vehicle", 0.02 * W, 0.8 * H, 0.1 * W, 0.9 * H))
                labels.append(_lb(806, "trailer", 0.85 * W, 0.2 * H, 0.95 * W, 0.3 * H))
                labels.append(_lb(807, "other person", 0.2 * W, 0.2 * H, 0.21 * W, 0.26 * H))
                labels.append(_lb(808, "train", 0.3 * W, 0.05 * H, 0.7 * W, 0.2 * H))
                labels.append(_lb(809, "rider", 0.12 * W, 0.4 * H, 0.13 * W, 0.45 * H))
                labels.append(_lb(810, "pedestrian", 0.4 * W, 0.3 * H, 0.42 * W, 0.4 * H, crowd=True))
        frames.append({"name": f"{video}-{fi + 1:07d}.jpg", "videoName": video, "frameIndex": fi, "labels": labels})
    return frames


def ve_anh(rng, frame: dict, W: int, H: int) -> np.ndarray:
    """Nen tan so thap + nhieu nhe (JPEG gan kich thuoc anh that, khong phai nhieu trang nen rat nang)."""
    import cv2
    nen = rng.integers(50, 130, size=(H // 16 + 2, W // 16 + 2, 3), dtype=np.uint8)
    im = cv2.resize(nen, (W, H), interpolation=cv2.INTER_LINEAR)
    im = np.clip(im.astype(np.int16) + rng.integers(-6, 7, size=(H, W, 3), dtype=np.int16), 0, 255).astype(np.uint8)
    for lb in frame["labels"]:
        b = lb["box2d"]
        x1, y1 = max(0, int(b["x1"])), max(0, int(b["y1"]))
        x2, y2 = min(W, int(np.ceil(b["x2"]))), min(H, int(np.ceil(b["y2"])))
        if x2 > x1 and y2 > y1:
            im[y1:y2, x1:x2] = KIEU[lb["category"]][0]
            im[y1:y2, x1:min(x2, x1 + 2)] = 20
    return im


def tao_fixture(goc: Path, n_video: int = 8, n_frame: int = 16, W: int = 160, H: int = 90, seed: int = 0,
                chat_luong: int = 90) -> dict:
    import cv2
    goc = Path(goc)
    rng = np.random.default_rng(seed)
    lab_dir = goc / "labels" / "box_track_20" / "train"
    img_root = goc / "images" / "track" / "train"
    lab_dir.mkdir(parents=True, exist_ok=True)
    coco = {"categories": [{"id": i + 1, "name": c, "supercategory": "x"} for i, c in enumerate(COCO_CAT)],
            "videos": [], "images": [], "annotations": []}
    cid = {c: i + 1 for i, c in enumerate(COCO_CAT)}
    videos = []
    for i in range(n_video):
        v = ten_video(seed, i)
        lop = ["car", "car", "truck"] + (["bus"] if i % 2 == 0 else []) + (["motorcycle"] if i % 3 == 0 else []) \
            + (["bicycle"] if i % 4 == 1 else []) + ["pedestrian"]
        frames = sinh_video(rng, v, n_frame, W, H, lop, dac_biet=(i == 0))
        (lab_dir / f"{v}.json").write_text(json.dumps(frames), encoding="utf-8")
        (img_root / v).mkdir(parents=True, exist_ok=True)
        coco["videos"].append({"id": i + 1, "name": v})
        for f in frames:
            ok, enc = cv2.imencode(".jpg", ve_anh(rng, f, W, H), [cv2.IMWRITE_JPEG_QUALITY, chat_luong])
            if not ok:
                raise RuntimeError(f"khong ma hoa duoc anh fixture {f['name']}")
            (img_root / v / f["name"]).write_bytes(enc.tobytes())
            iid = len(coco["images"]) + 1
            coco["images"].append({"id": iid, "video_id": i + 1, "frame_id": f["frameIndex"],
                                   "file_name": f"{v}/{f['name']}", "width": W, "height": H})
            for lb in f["labels"]:
                b = lb["box2d"]
                cat = IGNORE_MAP.get(lb["category"], lb["category"])
                coco["annotations"].append({
                    "id": len(coco["annotations"]) + 1, "image_id": iid, "category_id": cid[cat],
                    "instance_id": int(lb["id"]), "iscrowd": int(lb["attributes"]["crowd"]),
                    "ignore": int(lb["category"] in IGNORE_MAP),
                    "bbox": [b["x1"], b["y1"], b["x2"] - b["x1"], b["y2"] - b["y1"]]})
        videos.append(v)
    cdir = goc / "labels_cocovid"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "fx_box_track_train_cocofmt.json").write_text(json.dumps(coco), encoding="utf-8")
    return {"videos": videos, "n_frame": n_frame, "W": W, "H": H, "goc": str(goc)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-video", type=int, default=20)
    ap.add_argument("--n-frame", type=int, default=50)
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if Path(a.out).exists():
        raise SystemExit(f"{a.out} da ton tai")
    info = tao_fixture(Path(a.out), a.n_video, a.n_frame, a.w, a.h, a.seed)
    print(json.dumps({k: v for k, v in info.items() if k != "videos"} | {"n_video": len(info["videos"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
