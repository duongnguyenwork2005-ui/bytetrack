"""
extract_verify.py -- Giai nen cac file zip UA-DETRAC va kiem tra tinh toan ven.

CHUC NANG
---------
1. Giai nen data/raw/*.zip vao data/extracted/  (bo qua neu da giai nen roi).
2. TU DONG TIM thu muc anh, bat ke ban mirror long thu muc kieu gi
   (xem config.find_images_root).
3. Doi chieu 3 nguon thong tin de phat hien du lieu thieu / hong:
     - so luong video co annotation XML,
     - so frame duoc annotate trong tung XML (the <frame num=...>),
     - so file anh img*.jpg thuc su co tren dia.
4. Xuat bang tong hop ra data/interim/dataset_integrity.csv.

LUU Y VE SO FRAME
-----------------
Trong UA-DETRAC, the <frame> chi xuat hien cho frame CO it nhat 1 xe.
Vi vay n_frames_annotated (so the <frame>) thuong <= n_images (so anh tren dia).
Dieu do la BINH THUONG - da kiem chung: 82.085 frame co xe / 83.791 anh
(2.04% frame khong co xe nao).
Chi coi la LOI khi: max_frame_num > n_images (annotation tro toi anh khong ton tai).

GIAI NEN CHON LOC
-----------------
Cac ban mirror thuong dong goi CA tap train (60 video) LAN tap test (40 video)
trong cung 1 file zip ~10 GB. Tap test KHONG co ground truth cong khai nen vo
dung voi de tai nay. Vi vay MAC DINH script chi giai nen 60 video train
(tiet kiem khoang 4 GB dia). Dung --all-sequences neu muon giai nen tat ca.

CACH DUNG
---------
    python src/extract_verify.py                    # giai nen (chi train) + kiem tra
    python src/extract_verify.py --all-sequences    # giai nen ca tap test
    python src/extract_verify.py --verify-only      # chi kiem tra
    python src/extract_verify.py --images-zip D:/tai/ve/ua-detrac-orig.zip
"""
from __future__ import annotations

import argparse
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# 1. GIAI NEN
# ---------------------------------------------------------------------------
def extract_zip(zip_path: Path, dest_dir: Path, expect_subdir: str | None = None,
                member_filter=None, label: str = "") -> Path | None:
    """Giai nen `zip_path` vao `dest_dir`.

    expect_subdir  : neu thu muc nay da ton tai va khong rong -> bo qua (idempotent).
    member_filter  : ham nhan ten entry, tra ve True neu muon giai nen entry do.
                     Dung de chi lay 60 video train tu file zip chua ca train + test.
    """
    if not zip_path.exists():
        print(f"  [MISS] Khong co {zip_path.name}")
        return None

    if expect_subdir:
        target = dest_dir / expect_subdir
        if target.exists() and any(target.iterdir()):
            print(f"  [SKIP] Da giai nen truoc do: {target}")
            return target

    print(f"  [EXTRACT] {zip_path.name} ({zip_path.stat().st_size / 1e9:.2f} GB) -> {dest_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        if member_filter is not None:
            n_all = len(members)
            members = [m for m in members if member_filter(m.filename)]
            print(f"            loc {len(members):,} / {n_all:,} entry")
        for m in tqdm(members, desc=f"    {label or 'giai nen'}", unit="file", ncols=90):
            zf.extract(m, dest_dir)

    return dest_dir / expect_subdir if expect_subdir else dest_dir


def make_train_filter(videos: list[str]):
    """Tao ham loc chi giu entry thuoc 60 video train (hoac khong phai anh).

    Giu lai: moi entry KHONG nam trong thu muc MVI_* nao (annotation, toolkit...),
             va entry nam trong thu muc MVI_* thuoc danh sach train.
    """
    want = set(videos)
    pat = re.compile(r"(MVI_\d+)/", re.I)

    def keep(name: str) -> bool:
        m = pat.search(name.replace("\\", "/"))
        if m is None:
            return True                      # khong phai thu muc sequence -> giu
        return m.group(1).upper() in want    # chi giu sequence train

    return keep


def find_images_zip(explicit: str | None = None) -> Path | None:
    """Tim file zip chua anh trong data/raw (bo qua file annotation da biet)."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    ann_name = config.DRIVE_FILES["train_annotations_xml"]["filename"].lower()
    zips = [p for p in sorted(config.RAW_DIR.glob("*.zip")) if p.name.lower() != ann_name]
    if not zips:
        return None
    # Uu tien ten chuan, neu khong thi lay file lon nhat (anh nang hon annotation)
    preferred = config.RAW_DIR / config.DRIVE_FILES["train_images"]["filename"]
    if preferred in zips:
        return preferred
    return max(zips, key=lambda p: p.stat().st_size)


# ---------------------------------------------------------------------------
# 2. KIEM TRA TOAN VEN
# ---------------------------------------------------------------------------
def scan_xml(xml_path: Path) -> dict:
    """Doc nhanh 1 file XML, dem so frame / target / track id (khong parse day du).

    Dung ET.iterparse de khong load ca cay XML vao RAM.
    """
    n_frames = 0
    max_frame = 0
    min_frame = 10 ** 9
    n_targets = 0
    track_ids = set()
    n_occ = 0
    n_ignored_region = 0
    seq_name = xml_path.stem

    for event, elem in ET.iterparse(str(xml_path), events=("start", "end")):
        if event == "start":
            if elem.tag == "sequence":
                seq_name = elem.get("name", xml_path.stem)
            elif elem.tag == "frame":
                num = int(elem.get("num"))
                n_frames += 1
                max_frame = max(max_frame, num)
                min_frame = min(min_frame, num)
            elif elem.tag == "target":
                n_targets += 1
                track_ids.add(int(elem.get("id")))
            elif elem.tag == "region_overlap":
                n_occ += 1
        else:  # event == "end"
            if elem.tag == "box" and n_frames == 0:
                # box nam trong <ignored_region> (xuat hien truoc the <frame> dau tien)
                n_ignored_region += 1
            if elem.tag == "frame":
                elem.clear()  # giai phong RAM

    return {
        "video": seq_name,
        "xml_file": xml_path.name,
        "n_frames_annotated": n_frames,
        "min_frame_num": 0 if min_frame == 10 ** 9 else min_frame,
        "max_frame_num": max_frame,
        "n_target_instances": n_targets,     # tong so bbox (frame x xe)
        "n_tracks": len(track_ids),          # so xe duy nhat trong video
        "n_occlusion_regions": n_occ,        # so lan xuat hien <region_overlap>
        "n_ignored_regions": n_ignored_region,
    }


def count_images(images_root: Path | None, video: str) -> int:
    """Dem so file anh cua 1 video (0 neu chua tai bo anh)."""
    if images_root is None:
        return 0
    d = images_root / video
    if not d.is_dir():
        return 0
    return sum(1 for _ in d.glob("img*.jpg"))


def verify() -> pd.DataFrame:
    """Doi chieu XML voi anh tren dia, in bang tom tat va tra ve DataFrame."""
    ann_root = config.EXTRACTED_DIR / config.ANNOTATIONS_SUBDIR
    if not ann_root.is_dir():
        print(f"[ERROR] Chua co thu muc annotation: {ann_root}")
        return pd.DataFrame()

    xmls = sorted(ann_root.glob("*.xml"))
    print(f"\n[verify] Tim thay {len(xmls)} file XML trong {ann_root}")

    img_root = config.find_images_root()
    if img_root is None:
        print("[verify] Thu muc anh: CHUA CO (chua tai / chua giai nen bo anh)")
    else:
        print(f"[verify] Thu muc anh: {img_root}")
    has_images = img_root is not None

    rows = []
    for x in tqdm(xmls, desc="  quet XML", unit="video", ncols=90):
        r = scan_xml(x)
        r["n_images"] = count_images(img_root, r["video"])
        rows.append(r)

    df = pd.DataFrame(rows)

    # --- Danh gia trang thai tung video ---
    def status(row) -> str:
        if row["n_frames_annotated"] == 0:
            return "EMPTY_XML"
        if not has_images:
            return "NO_IMAGES_YET"
        if row["n_images"] == 0:
            return "MISSING_IMAGE_DIR"
        if row["max_frame_num"] > row["n_images"]:
            return "ANNOT_EXCEEDS_IMAGES"   # LOI THAT SU
        return "OK"

    df["status"] = df.apply(status, axis=1)

    # Ty le frame co annotation so voi tong so anh (NA neu chua co anh)
    n_img = df["n_images"].replace(0, pd.NA)
    df["annot_coverage"] = (df["n_frames_annotated"] / n_img).astype("Float64").round(4)

    out = config.INTERIM_DIR / "dataset_integrity.csv"
    df.to_csv(out, index=False)

    # --- In tom tat ---
    print("\n" + "=" * 78)
    print("TONG KET KIEM TRA TOAN VEN")
    print("=" * 78)
    print(f"  So video (XML)            : {len(df)}")
    print(f"  Tong frame co annotation  : {df['n_frames_annotated'].sum():,}")
    print(f"  Tong bbox (target)        : {df['n_target_instances'].sum():,}")
    print(f"  Tong track (xe)           : {df['n_tracks'].sum():,}")
    print(f"  Tong vung che khuat       : {df['n_occlusion_regions'].sum():,}")
    if has_images:
        n_i = int(df["n_images"].sum())
        n_a = int(df["n_frames_annotated"].sum())
        print(f"  Tong anh tren dia         : {n_i:,}")
        print(f"  Frame khong co xe nao     : {n_i - n_a:,} ({(n_i - n_a) / n_i * 100:.2f}%)")

    print("\n  Phan bo trang thai:")
    for k, v in df["status"].value_counts().items():
        print(f"    {k:<22} {v}")

    bad = df[df["status"] == "ANNOT_EXCEEDS_IMAGES"]
    if len(bad):
        print("\n  [CANH BAO] Video co annotation tro toi frame khong ton tai:")
        print(bad[["video", "max_frame_num", "n_images"]].to_string(index=False))

    miss = df[df["status"] == "MISSING_IMAGE_DIR"]
    if len(miss):
        print(f"\n  [CANH BAO] {len(miss)} video co XML nhung khong tim thay thu muc anh:")
        print("   ", ", ".join(miss["video"].head(10)))

    print(f"\n  Da luu bang chi tiet -> {out}")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Giai nen + kiem tra UA-DETRAC")
    ap.add_argument("--verify-only", action="store_true", help="Bo qua buoc giai nen")
    ap.add_argument("--all-sequences", action="store_true",
                    help="Giai nen ca 40 video test (mac dinh chi giai nen 60 video train)")
    ap.add_argument("--images-zip", default=None,
                    help="Duong dan file zip anh (mac dinh: tu tim trong data/raw)")
    args = ap.parse_args()

    if not args.verify_only:
        print("=" * 78)
        print("BUOC 1: GIAI NEN")
        print("=" * 78)

        # --- 1a. Annotation XML ---
        extract_zip(
            config.RAW_DIR / config.DRIVE_FILES["train_annotations_xml"]["filename"],
            config.EXTRACTED_DIR,
            expect_subdir=config.ANNOTATIONS_SUBDIR,
            label="annotation",
        )

        # --- 1b. Bo anh ---
        img_zip = find_images_zip(args.images_zip)
        if img_zip is None:
            print("  [MISS] Khong tim thay file zip anh nao trong data/raw/")
            print("         Xem huong dan tai o muc 'Tai du lieu thu cong' trong README.md")
        elif config.find_images_root() is not None:
            print(f"  [SKIP] Da co thu muc anh: {config.find_images_root()}")
        else:
            train_videos = config.list_train_videos()
            if args.all_sequences or not train_videos:
                mf = None
                print("  [INFO] Giai nen TAT CA sequence trong zip.")
            else:
                mf = make_train_filter(train_videos)
                print(f"  [INFO] Chi giai nen {len(train_videos)} video train "
                      f"(dung --all-sequences neu muon ca tap test).")
            extract_zip(img_zip, config.EXTRACTED_DIR, member_filter=mf, label="anh")

    print("\n" + "=" * 78)
    print("BUOC 2: KIEM TRA TOAN VEN")
    print("=" * 78)
    df = verify()
    return 0 if len(df) else 1


if __name__ == "__main__":
    raise SystemExit(main())
