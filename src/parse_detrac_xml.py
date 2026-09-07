"""
parse_detrac_xml.py -- Parse annotation XML cua UA-DETRAC thanh DataFrame chuan hoa.

CAU TRUC XML GOC (vi du MVI_20011.xml)
--------------------------------------
    <sequence name="MVI_20011">
      <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
      <ignored_region>
        <box left="778.75" top="24.75" width="181.75" height="63.5"/>   <- vung bo qua khi eval
        ...
      </ignored_region>
      <frame density="7" num="1">                                       <- num: 1-based
        <target_list>
          <target id="3">                                               <- id = track id
            <box left="545.2" top="88.27" width="35.25" height="30.08"/>
            <attribute orientation="2.75" speed="0.52" trajectory_length="105"
                       truncation_ratio="0" vehicle_type="car"/>
            <occlusion>
              <region_overlap left="553" top="88.27" width="27.45" height="1.52"
                              occlusion_id="5" occlusion_status="1"/>
            </occlusion>
          </target>
        </target_list>
      </frame>
    </sequence>


=============================================================================
DIEM MAU CHOT 1: XML KHONG CO SAN TRUONG occlusion_ratio
=============================================================================
UA-DETRAC chi cho biet VUNG CHONG LAN (region_overlap) trong khong gian anh.
Ta phai tu tinh ty le che khuat:

        occlusion_ratio = dien_tich( HOP cac vung che  giao  box ) / dien_tich(box)

Phai dung phep HOP (union) chu khong phai TONG, vi mot xe co the bi 2 xe khac
che va 2 vung chong lan co the giao nhau -> cong don se dem trung, ratio > 1.
Ket qua duoc clip ve [0, 1]. Tu do:  visibility = 1 - occlusion_ratio.


=============================================================================
DIEM MAU CHOT 2: Y NGHIA occlusion_status (da kiem chung bang thuc nghiem)
=============================================================================
Moi quan he che khuat giua 2 xe chi duoc khai bao MOT LAN, o MOT PHIA
(da kiem tra: khong ton tai cap khai bao doi xung nao trong toan bo dataset).
`occlusion_status` cho biet phia khai bao dong vai tro gi:

  status =  0 : xe khai bao BI CHE boi xe `occlusion_id`.
                Bang chung: 99.2% truong hop canh duoi box cua xe khai bao nam
                CAO hon trong anh (xa camera hon), va box nho hon (median ty le
                dien tich 0.53).

  status =  1 : xe khai bao DANG CHE xe `occlusion_id` (no o phia truoc).
                Bang chung: 99.4% truong hop canh duoi box cua xe khai bao nam
                THAP hon trong anh (gan camera hon), box lon hon (median 1.31).
                => Vung nay KHONG phai phan bi che cua xe khai bao,
                   ma la phan bi che cua xe `occlusion_id`.

  status = -1 : bi che boi VAT NEN TINH (cay, cot dien, bien bao...).
                Bang chung: 100% truong hop `occlusion_id` KHONG ung voi bat ky
                target nao trong cung frame.

HE QUA (rat quan trong, neu lam sai se tinh nham visibility):
Khi tinh do che khuat cua xe V tai 1 frame, phai gom:
    (a) cac vung V tu khai bao voi status = 0   (bi xe khac che)
    (b) cac vung V tu khai bao voi status = -1  (bi vat nen che)
    (c) cac vung do XE KHAC khai bao voi status = 1 va occlusion_id = V
va phai LOAI BO cac vung V khai bao voi status = 1 (do la phan V che xe khac).

Script nay lam dung nhu vay: doc ca frame -> phan phoi lai vung che ve dung xe
bi che -> moi tinh ty le.


OUTPUT
------
    data/interim/detrac_train_annotations.parquet   <- bang chinh (1 dong = 1 bbox)
    data/interim/detrac_train_annotations_sample.csv<- 2000 dong dau, mo bang Excel
    data/interim/detrac_ignored_regions.csv         <- vung bo qua theo tung video
    data/interim/detrac_sequence_attributes.csv     <- thoi tiet / trang thai camera

CACH DUNG
---------
    python src/parse_detrac_xml.py                     # parse toan bo 60 video
    python src/parse_detrac_xml.py --videos MVI_20011 MVI_20012
    python src/parse_detrac_xml.py --limit 5           # chi 5 video dau (test nhanh)
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


# ---------------------------------------------------------------------------
# Tien ich hinh hoc
# ---------------------------------------------------------------------------
def rect_intersection(a: tuple, b: tuple):
    """Giao cua 2 hinh chu nhat dang (x1, y1, x2, y2). None neu khong giao."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def union_area(rects: list) -> float:
    """Dien tich phan HOP cua danh sach hinh chu nhat (x1, y1, x2, y2).

    Thuat toan nen toa do (coordinate compression):
      1. Lay tat ca canh x va canh y lam luoi.
      2. Moi o luoi hoac nam tron trong mot hinh, hoac hoan toan ngoai
         -> chi can kiem tra diem giua o co thuoc hinh nao khong.
      3. Cong dien tich cac o duoc phu.

    So hinh chu nhat moi xe rat nho (thuong 1-3) nen chi phi khong dang ke.
    Cach nay tranh loi dem trung dien tich khi cac vung che khuat giao nhau.
    """
    rects = [r for r in rects if r is not None and r[2] > r[0] and r[3] > r[1]]
    if not rects:
        return 0.0
    if len(rects) == 1:
        r = rects[0]
        return (r[2] - r[0]) * (r[3] - r[1])

    xs = sorted({r[0] for r in rects} | {r[2] for r in rects})
    ys = sorted({r[1] for r in rects} | {r[3] for r in rects})

    total = 0.0
    for i in range(len(xs) - 1):
        x_lo, x_hi = xs[i], xs[i + 1]
        xm = 0.5 * (x_lo + x_hi)
        for j in range(len(ys) - 1):
            y_lo, y_hi = ys[j], ys[j + 1]
            ym = 0.5 * (y_lo + y_hi)
            if any(r[0] <= xm <= r[2] and r[1] <= ym <= r[3] for r in rects):
                total += (x_hi - x_lo) * (y_hi - y_lo)
    return total


# ---------------------------------------------------------------------------
# Doc thuoc tinh XML an toan
# ---------------------------------------------------------------------------
def fattr(elem, *names, default=np.nan) -> float:
    """Lay thuoc tinh dau tien tim thay trong `names` va ep ve float.

    Co nhieu `names` vi mot vai ban phat hanh UA-DETRAC dung `w`/`h`
    thay vi `width`/`height` trong the region_overlap.
    """
    for n in names:
        v = elem.get(n)
        if v is not None and v != "":
            try:
                return float(v)
            except ValueError:
                return default
    return default


def box_of(elem):
    """Doc the <box .../> hoac <region_overlap .../> -> (left, top, width, height)."""
    return (
        fattr(elem, "left", "x"),
        fattr(elem, "top", "y"),
        fattr(elem, "width", "w"),
        fattr(elem, "height", "h"),
    )


# ---------------------------------------------------------------------------
# Parse 1 file XML
# ---------------------------------------------------------------------------
def parse_one_xml(xml_path: Path):
    """Parse 1 video -> (rows_bbox, rows_ignored_region, sequence_attribute).

    Voi moi frame, xu ly theo 3 luot:
      PASS 1 - doc bbox + thuoc tinh cua tat ca xe, gom moi khai bao che khuat.
      PASS 2 - phan phoi tung vung che khuat ve dung XE BI CHE (xem docstring
               dau file: quy tac status 0 / 1 / -1).
      PASS 3 - tinh occlusion_ratio bang dien tich HOP, roi xuat dong du lieu.
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    video = root.get("name", xml_path.stem)

    # --- Thuoc tinh chung cua video ---
    seq_attr = {"video": video, "camera_state": None, "sence_weather": None}
    sa = root.find("sequence_attribute")
    if sa is not None:
        seq_attr["camera_state"] = sa.get("camera_state")
        seq_attr["sence_weather"] = sa.get("sence_weather")  # (chinh ta goc cua dataset)

    # --- Vung bo qua khi danh gia (via he, bai do xe xa...) ---
    ignored = []
    ir = root.find("ignored_region")
    if ir is not None:
        for k, b in enumerate(ir.findall("box")):
            l, t, w, h = box_of(b)
            ignored.append({"video": video, "region_idx": k,
                            "left": l, "top": t, "width": w, "height": h})

    rows = []
    n_bad_box = 0

    for frame in root.findall("frame"):
        frame_num = int(frame.get("num"))
        density = int(frame.get("density", 0))

        tl = frame.find("target_list")
        if tl is None:
            continue

        # =================================================================
        # PASS 1: doc bbox + thuoc tinh + moi khai bao che khuat
        # =================================================================
        targets = {}        # track_id -> thong tin xe
        order = []          # giu thu tu xuat hien de output on dinh
        declarations = []   # (declarer_id, other_id, status, region_xyxy)

        for tg in tl.findall("target"):
            track_id = int(tg.get("id"))

            b = tg.find("box")
            if b is None:
                n_bad_box += 1
                continue
            left, top, width, height = box_of(b)
            if not np.isfinite([left, top, width, height]).all() or width <= 0 or height <= 0:
                n_bad_box += 1
                continue

            at = tg.find("attribute")
            targets[track_id] = {
                "box": (left, top, left + width, top + height),
                "ltwh": (left, top, width, height),
                "area": width * height,
                "orientation": fattr(at, "orientation") if at is not None else np.nan,
                "speed": fattr(at, "speed") if at is not None else np.nan,
                "trajectory_length": fattr(at, "trajectory_length") if at is not None else np.nan,
                "truncation_ratio": fattr(at, "truncation_ratio", default=0.0) if at is not None else 0.0,
                "vehicle_type": at.get("vehicle_type") if at is not None else None,
            }
            order.append(track_id)

            occ = tg.find("occlusion")
            if occ is None:
                continue
            for ro in occ.findall("region_overlap"):
                ol, ot, ow, oh = box_of(ro)
                if not np.isfinite([ol, ot, ow, oh]).all() or ow <= 0 or oh <= 0:
                    continue
                try:
                    status = int(float(ro.get("occlusion_status", -1)))
                except (TypeError, ValueError):
                    status = -1
                try:
                    other_id = int(float(ro.get("occlusion_id", -1)))
                except (TypeError, ValueError):
                    other_id = -1
                declarations.append((track_id, other_id, status,
                                     (ol, ot, ol + ow, ot + oh)))

        # =================================================================
        # PASS 2: phan phoi vung che khuat ve dung XE BI CHE
        # =================================================================
        occ_by_vehicle = defaultdict(list)   # tid -> [region] bi xe khac che
        occ_by_background = defaultdict(list)  # tid -> [region] bi vat nen che
        occluder_ids = defaultdict(list)     # tid -> [id cua xe che no]
        raw_status = defaultdict(list)       # tid -> [status tho ma chinh no khai bao]

        for declarer, other, status, region in declarations:
            raw_status[declarer].append(status)

            if status == 1 and other in targets:
                # Xe khai bao dang CHE xe `other` -> vung nay thuoc ve `other`.
                occ_by_vehicle[other].append(region)
                occluder_ids[other].append(declarer)
            elif status == 0 and other in targets:
                # Xe khai bao BI xe `other` che.
                occ_by_vehicle[declarer].append(region)
                occluder_ids[declarer].append(other)
            else:
                # status == -1, hoac `other` khong ton tai trong frame
                # -> coi la bi vat nen tinh che.
                occ_by_background[declarer].append(region)

        # =================================================================
        # PASS 3: tinh ty le che khuat va xuat dong
        # =================================================================
        for track_id in order:
            t = targets[track_id]
            box, area = t["box"], t["area"]
            left, top, width, height = t["ltwh"]

            # Clip moi vung che vao trong bbox cua chinh xe do.
            rv = [r for r in (rect_intersection(x, box) for x in occ_by_vehicle.get(track_id, [])) if r]
            rb = [r for r in (rect_intersection(x, box) for x in occ_by_background.get(track_id, [])) if r]

            ratio_v = float(np.clip(union_area(rv) / area, 0.0, 1.0)) if area > 0 else 0.0
            ratio_b = float(np.clip(union_area(rb) / area, 0.0, 1.0)) if area > 0 else 0.0
            # Tong the phai tinh HOP cua ca 2 nhom (chung co the giao nhau),
            # khong duoc cong ratio_v + ratio_b.
            ratio_all = float(np.clip(union_area(rv + rb) / area, 0.0, 1.0)) if area > 0 else 0.0

            if rv and rb:
                source = "both"
            elif rv:
                source = "vehicle"
            elif rb:
                source = "background"
            else:
                source = "none"

            rows.append({
                "video": video,
                "frame": frame_num,
                "track_id": track_id,
                # bbox theo quy uoc MOTChallenge: goc trai-tren + kich thuoc
                "bb_left": left, "bb_top": top, "bb_width": width, "bb_height": height,
                # tam bbox - dung cho motion model (CV / CTRV) va tinh curvature
                "cx": left + width / 2.0,
                "cy": top + height / 2.0,
                "area": area,
                # thuoc tinh xe
                "vehicle_type": t["vehicle_type"],
                "orientation": t["orientation"],
                "speed": t["speed"],
                "trajectory_length": t["trajectory_length"],
                "truncation_ratio": t["truncation_ratio"],
                # --- che khuat (da phan phoi dung phia) ---
                "occlusion_ratio": ratio_all,
                "visibility": 1.0 - ratio_all,
                "occ_ratio_by_vehicle": ratio_v,
                "occ_ratio_by_background": ratio_b,
                "occlusion_source": source,
                "n_occlusion_regions": len(rv) + len(rb),
                "occluder_ids": ";".join(str(i) for i in occluder_ids.get(track_id, [])) or None,
                # gia tri THO ma chinh xe nay khai bao, giu lai de doi chieu tai lieu goc
                "occlusion_status_raw": ";".join(str(s) for s in raw_status.get(track_id, [])) or None,
                # bo sung
                "frame_density": density,
            })

    if n_bad_box:
        print(f"    [WARN] {xml_path.stem}: bo qua {n_bad_box} bbox loi/thieu.")

    return rows, ignored, seq_attr


# ---------------------------------------------------------------------------
# Parse toan bo dataset
# ---------------------------------------------------------------------------
def parse_dataset(xml_paths: list):
    all_rows, all_ignored, all_seq = [], [], []
    for p in tqdm(xml_paths, desc="  parse XML", unit="video", ncols=90):
        rows, ign, seq = parse_one_xml(p)
        all_rows.extend(rows)
        all_ignored.extend(ign)
        all_seq.append(seq)

    df = pd.DataFrame(all_rows)
    if len(df):
        df["frame"] = df["frame"].astype("int32")
        df["track_id"] = df["track_id"].astype("int32")
        df["vehicle_type"] = df["vehicle_type"].astype("category")
        df["occlusion_source"] = df["occlusion_source"].astype("category")
        # sap xep on dinh de moi lan chay cho ket qua giong nhau
        df = df.sort_values(["video", "frame", "track_id"]).reset_index(drop=True)

    return df, pd.DataFrame(all_ignored), pd.DataFrame(all_seq)


def summarize(df: pd.DataFrame) -> None:
    """In thong ke nhanh de kiem tra bang co hop ly khong."""
    n = len(df)
    print("\n" + "=" * 78)
    print("TOM TAT BANG ANNOTATION")
    print("=" * 78)
    print(f"  So dong (bbox)        : {n:,}")
    print(f"  So video              : {df['video'].nunique()}")
    print(f"  So track (video,id)   : {df.groupby(['video', 'track_id']).ngroups:,}")
    print(f"  Khoang frame          : {df['frame'].min()} .. {df['frame'].max()}")

    print("\n  Phan bo vehicle_type:")
    for k, v in df["vehicle_type"].value_counts().items():
        print(f"    {str(k):<10} {v:>9,}  ({v / n * 100:5.2f}%)")

    print("\n  Nguon che khuat (occlusion_source):")
    for k, v in df["occlusion_source"].value_counts().items():
        print(f"    {str(k):<12} {v:>9,}  ({v / n * 100:5.2f}%)")

    occ = df.loc[df["occlusion_ratio"] > 0, "occlusion_ratio"]
    n_thr = int((df["occlusion_ratio"] >= config.OCCLUSION_MIN_RATIO).sum())
    print("\n  occlusion_ratio:")
    print(f"    bbox co che khuat > 0        : {len(occ):,} ({len(occ) / n * 100:.2f}%)")
    print(f"    bbox >= nguong {config.OCCLUSION_MIN_RATIO:<4}          : {n_thr:,} ({n_thr / n * 100:.2f}%)")
    if len(occ):
        q = occ.quantile([0.25, 0.5, 0.75, 0.9, 0.99])
        print(f"    min / median / max           : {occ.min():.4f} / {occ.median():.4f} / {occ.max():.4f}")
        print("    quantile 25/50/75/90/99      : " +
              " / ".join(f"{q.loc[t]:.3f}" for t in [0.25, 0.5, 0.75, 0.9, 0.99]))

    print("\n  truncation_ratio (xe bi cat boi bien anh):")
    tr = df["truncation_ratio"]
    print(f"    bbox > 0                     : {(tr > 0).sum():,} ({(tr > 0).mean() * 100:.2f}%)")

    print("\n  Kiem tra bat thuong:")
    dup = df.duplicated(subset=["video", "frame", "track_id"]).sum()
    print(f"    trung (video,frame,track_id) : {dup}")
    oob = ((df["bb_left"] < -1) | (df["bb_top"] < -1) |
           (df["bb_left"] + df["bb_width"] > config.IMG_W + 1) |
           (df["bb_top"] + df["bb_height"] > config.IMG_H + 1)).sum()
    print(f"    bbox vuot khung anh          : {oob:,} ({oob / n * 100:.2f}%)")
    print(f"    thieu vehicle_type           : {df['vehicle_type'].isna().sum():,}")
    print(f"    visibility ngoai [0,1]       : {((df['visibility'] < 0) | (df['visibility'] > 1)).sum()}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Parse UA-DETRAC XML -> DataFrame")
    ap.add_argument("--videos", nargs="*", default=None, help="Chi parse cac video nay")
    ap.add_argument("--limit", type=int, default=None, help="Chi parse N video dau tien")
    ap.add_argument("--part", choices=["train", "test"], default="train",
                    help="Phan dataset can xu ly. Giai doan D dung --part test "
                         "(tap test 40 video). File ket qua cua train GIU NGUYEN ten cu.")
    args = ap.parse_args()

    ann_root = config.ann_dir(args.part)
    if not ann_root.is_dir():
        print(f"[ERROR] Chua co {ann_root}. Chay src/extract_verify.py truoc.")
        return 1

    xmls = sorted(ann_root.glob("*.xml"))
    if args.videos:
        want = set(args.videos)
        xmls = [p for p in xmls if p.stem in want]
    if args.limit:
        xmls = xmls[: args.limit]

    print(f"[parse] {len(xmls)} file XML")
    df, df_ign, df_seq = parse_dataset(xmls)
    if df.empty:
        print("[ERROR] Khong parse duoc dong nao.")
        return 1

    out_pq = config.ann_parquet(args.part)
    df.to_parquet(out_pq, index=False)
    df.head(2000).to_csv(config.part_file("detrac_train_annotations_sample.csv", args.part), index=False)
    df_ign.to_csv(config.part_file("detrac_ignored_regions.csv", args.part), index=False)
    df_seq.to_csv(config.part_file("detrac_sequence_attributes.csv", args.part), index=False)

    summarize(df)
    print("\n  Da luu:")
    print(f"    {out_pq}")
    print(f"    {config.part_file('detrac_train_annotations_sample.csv', args.part)}")
    print(f"    {config.part_file('detrac_ignored_regions.csv', args.part)}  ({len(df_ign)} vung)")
    print(f"    {config.part_file('detrac_sequence_attributes.csv', args.part)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
