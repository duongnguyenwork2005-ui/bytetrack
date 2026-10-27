"""
xuat_mixed5.py -- dung dataset tron 5 lop data/yolo_mixed5_v1 (BDD100K + UA-DETRAC), R52A.

    python -X utf8 src/mixed5/xuat_mixed5.py --buoc chia          # khoa configs/phase2/detector/splits/mixed5_v1.json
    python -X utf8 src/mixed5/xuat_mixed5.py --buoc ke-hoach      # chon frame + bam + kiem ro ri TRUOC khi ghi anh
    python -X utf8 src/mixed5/xuat_mixed5.py --buoc xuat-bdd      # hardlink/sao chep anh-nhan-ignore BDD (tiep tuc duoc)
    python -X utf8 src/mixed5/xuat_mixed5.py --buoc xuat-detrac   # xuat theo video, co gioi han thoi gian (tiep tuc duoc)
    python -X utf8 src/mixed5/xuat_mixed5.py --buoc manifest      # manifest v2 + data.yaml + kiem ro ri lan 2
    python -X utf8 src/mixed5/xuat_mixed5.py --buoc kiem          # bx.kiem_hop_dong_dataset + kiem bo sung tu XML
    python -X utf8 src/mixed5/xuat_mixed5.py --buoc thong-ke | overlay

Quy tac: docstring `src/mixed5/mixed5_lib.py`. Ma thoat: 0 dat; 2 thieu du lieu; 3 du lieu sai / ro ri / hop dong hong.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mixed5_lib as m  # noqa: E402
from mixed5_lib import GOC, LOP_5, bx  # noqa: E402

CFG_MAC_DINH = GOC / "configs/phase2/detector/mixed5_v1.yaml"


def doc_cfg(p: Path) -> dict:
    import yaml
    cfg = yaml.safe_load(p.read_text(encoding="utf-8"))
    cfg["_duong_dan"], cfg["_sha256"] = bx.rel(p, GOC), bx.sha256_file(p)
    if cfg["detrac"]["anh_xa_lop"] != m.ANH_XA_LOP:
        raise m.LoiDuLieu(f"anh_xa_lop trong cau hinh {cfg['detrac']['anh_xa_lop']} khac quyet dinh {m.ANH_XA_LOP}")
    return cfg


def P(s: str) -> Path:
    return GOC / s


def tam(cfg) -> Path:
    return P(cfg["thu_muc_tam"])


def doc_csv(p: Path) -> list[dict]:
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def nap_split(cfg) -> dict:
    return m.doc_split_mixed5(P(cfg["file_split"]), P(cfg["bdd"]["file_split"]))


def nap_bdd(cfg, sp) -> list[dict]:
    """Dong manifest bdd5_v1 (da kiem sha256 manifest, split goc, video theo split)."""
    d = P(cfg["bdd"]["dataset"])
    man = d / "manifest" / "anh.csv"
    if bx.sha256_file(man) != cfg["bdd"]["sha256_manifest_anh"]:
        raise m.LoiDuLieu(f"{man} khac sha256 da ghi trong cau hinh")
    tom = json.loads((d / "manifest" / "tom_tat.json").read_text(encoding="utf-8"))
    if tom["file_split"]["sha256"] != cfg["bdd"]["sha256_split"] or tom["manifest_anh"]["sha256"] != bx.sha256_file(man):
        raise m.LoiDuLieu("bdd5_v1: tom_tat khong khop split/manifest da khoa")
    rows = doc_csv(man)
    bsp = json.loads(P(cfg["bdd"]["file_split"]).read_text(encoding="utf-8"))
    for s, k in (("train", "train_fit"), ("dev", "dev")):
        if {r["video"] for r in rows if r["split"] == s} != set(bsp[k]):
            raise m.LoiDuLieu(f"bdd5_v1 {s}: video khac split {k}")
    return rows


def nap_tap_cam(cfg) -> dict:
    out = {}
    for t in cfg["tap_danh_gia_cam"]:
        b = m.doc_bang_sha(P(t["bang_sha256"]))
        if t.get("xml"):
            b["video"] |= {p.stem for p in P(t["xml"]).glob("*.xml")}
        b["nguon_anh"] = "detrac" if t["ten"] == "ua_detrac_test" else "bdd"
        out[t["ten"]] = b
    return out


# =========================================================================== buoc
def b_chia(cfg) -> dict:
    p = P(cfg["file_split"])
    if p.exists():
        raise m.LoiDuLieu(f"{p} da ton tai -- split da khoa, khong ghi de")
    d = cfg["detrac"]
    rec = m.tao_ban_ghi_split(cfg["ten"], P(cfg["bdd"]["file_split"]), P(d["xml"]), P(d["dev12"]), int(d["so_video_xml"]),
                              int(d["so_video_dev"]), {"duong_dan": cfg["_duong_dan"], "sha256": cfg["_sha256"]})
    m.ghi_json(p, rec)
    return {"file": bx.rel(p, GOC), "sha256": bx.sha256_file(p), "sha256_danh_sach": rec["sha256_danh_sach"],
            "train_fit": len(rec["train_fit"]), "dev": len(rec["dev"]),
            "detrac": {"train_fit": len(rec["detrac"]["train_fit"]), "dev": len(rec["detrac"]["dev"])}}


def b_ke_hoach(cfg) -> dict:
    t0 = time.time()
    sp = nap_split(cfg)
    bdd_rows = nap_bdd(cfg, sp)
    d = cfg["detrac"]
    buoc = int(d["buoc_frame"])
    ke, dem_video = [], {}
    for s, k in (("train", "train_fit"), ("dev", "dev")):
        for v in sp["detrac"][k]:
            x = m.doc_xml_detrac(P(d["xml"]) / f"{v}.xml")
            anh = m.liet_ke_anh_detrac(P(d["anh"]) / v)
            chon, dem = m.chon_frame_detrac(x["frames"].keys(), anh.keys(), buoc)
            dem["so_ignored_region"] = len(x["ignored"])
            dem_video[v] = {"split": s, **dem}
            for n in chon:
                src = P(d["anh"]) / v / anh[n]
                ke.append({"split": s, "video": v, "num": n, "anh_nguon": f"{v}/{anh[n]}",
                           "sha256_nguon": bx.sha256_file(src), "ten": m.ten_anh_detrac(v, n), "bytes": src.stat().st_size})
    # JPEG thieu EOI -> ultralytics ghi lai file (hardlink se lam doi ca bdd5_v1) -> dem truoc
    bdd_dir = P(cfg["bdd"]["dataset"])
    eoi_bdd = sorted(r["anh_ra"] for r in bdd_rows if not m.co_eoi_jpeg(bdd_dir / r["anh_ra"]))
    eoi_det = sorted(e["anh_nguon"] for e in ke if not m.co_eoi_jpeg(P(d["anh"]) / e["anh_nguon"]))
    tap = nap_tap_cam(cfg)
    gia = [{**{k: r[k] for k in ("split", "video", "anh_nguon", "sha256_nguon", "sha256_ra", "anh_ra")}, "nguon": "bdd"}
           for r in bdd_rows]
    gia += [{"split": e["split"], "video": e["video"], "anh_nguon": e["anh_nguon"], "sha256_nguon": e["sha256_nguon"],
             "sha256_ra": e["sha256_nguon"], "anh_ra": f"images/{e['split']}/{e['ten']}", "nguon": "detrac"} for e in ke]
    vs = {"train": set(sp["train_fit"]), "dev": set(sp["dev"])}
    ro_ri = m.kiem_ro_ri(gia, tap, vs)
    can = int(sum(e["bytes"] for e in ke) * 1.6)
    trong = shutil.disk_usage(GOC).free
    con_lai = (trong - can) / 2 ** 30
    if con_lai < float(cfg["o_dia"]["giu_lai_toi_thieu_gb"]):
        raise m.LoiDuLieu(f"o dia: trong {trong / 2**30:.1f} GB, can ~{can / 2**30:.1f} GB -> con {con_lai:.1f} GB < nguong")
    kh = {"sha256_split": bx.sha256_file(P(cfg["file_split"])), "cau_hinh": cfg["_sha256"], "buoc_frame": buoc,
          "detrac": ke, "dem_video": dem_video, "eoi_thieu": {"bdd": eoi_bdd, "detrac": eoi_det},
          "kiem_ro_ri_truoc_ghi": ro_ri, "o_dia": {"trong_gb": round(trong / 2**30, 2), "uoc_can_gb": round(can / 2**30, 2)},
          "thoi_gian_s": round(time.time() - t0, 1)}
    m.ghi_json(tam(cfg) / "ke_hoach.json", kh)
    return {"so_anh_detrac": {s: sum(1 for e in ke if e["split"] == s) for s in ("train", "dev")},
            "so_anh_bdd": {s: sum(1 for r in bdd_rows if r["split"] == s) for s in ("train", "dev")},
            "eoi_thieu": {"bdd": len(eoi_bdd), "detrac": len(eoi_det)}, "kiem_ro_ri_truoc_ghi": ro_ri,
            "o_dia": kh["o_dia"], "thoi_gian_s": kh["thoi_gian_s"]}


def nap_ke_hoach(cfg) -> dict:
    p = tam(cfg) / "ke_hoach.json"
    if not p.is_file():
        raise m.LoiDuLieu("chua co ke_hoach.json -- chay --buoc ke-hoach truoc")
    kh = json.loads(p.read_text(encoding="utf-8"))
    if kh["sha256_split"] != bx.sha256_file(P(cfg["file_split"])) or kh["cau_hinh"] != cfg["_sha256"]:
        raise m.LoiDuLieu("ke_hoach.json lap tren split/cau hinh khac -- lap lai ke hoach")
    return kh


def kiem_chua_xong(cfg) -> Path:
    out = P(cfg["dau_ra"])
    if (out / "manifest" / "tom_tat.json").exists():
        raise m.LoiDuLieu(f"{out} da co manifest (dataset da xuat) -- khong ghi lai")
    return out


def b_xuat_bdd(cfg) -> dict:
    t0 = time.time()
    sp = nap_split(cfg)
    kh = nap_ke_hoach(cfg)
    out = kiem_chua_xong(cfg)
    rows = nap_bdd(cfg, sp)
    src = P(cfg["bdd"]["dataset"])
    eoi = set(kh["eoi_thieu"]["bdd"])
    hl = cfg["bdd"]["cach_dat_file"] == "hardlink"
    dem = {"hardlink": 0, "sao_chep": 0}
    for r in rows:
        for k in ("anh_ra", "nhan_ra", "ignore_ra"):
            if r[k]:
                dem[m.dat_file_bdd(src / r[k], out / r[k], hl and r[k] not in eoi)] += 1
    m.ghi_json(tam(cfg) / "bdd_xong.json", {"so_dong": len(rows), "dem_file": dem})
    return {"so_dong": len(rows), "dem_file": dem, "thoi_gian_s": round(time.time() - t0, 1)}


def b_xuat_detrac(cfg, gioi_han_s: float) -> dict:
    t0 = time.time()
    nap_split(cfg)
    kh = nap_ke_hoach(cfg)
    out = kiem_chua_xong(cfg)
    d = cfg["detrac"]
    theo_video: dict[str, list] = {}
    for e in kh["detrac"]:
        theo_video.setdefault(e["video"], []).append(e)
    xong, lam = [], []
    for v, ds in theo_video.items():
        st = tam(cfg) / "detrac" / f"{v}.json"
        if st.is_file():
            xong.append(v)
            continue
        if time.time() - t0 > gioi_han_s:
            break
        x = m.doc_xml_detrac(P(d["xml"]) / f"{v}.xml")
        dem = {k: 0 for k in m.KHOA_DEM} | {"box_xml_khong_hop_le": 0}
        rows = []
        for e in ds:
            fr = m.frame_cho_bx(v, e["num"], x["frames"][e["num"]], x["ignored"], m.ANH_XA_LOP, dem)
            r = m.xuat_frame_detrac(P(d["anh"]) / e["anh_nguon"], fr, out, e["split"], v, e["ten"], dem)
            if r["sha256_nguon"] != e["sha256_nguon"]:
                raise m.LoiDuLieu(f"{e['anh_nguon']}: anh nguon doi sha256 sau buoc ke-hoach")
            rows.append(r)
        m.ghi_json(st, {"video": v, "rows": rows, "dem": dem})
        lam.append(v)
    con = [v for v in theo_video if v not in xong and v not in lam]
    return {"da_xong_truoc": len(xong), "lam_luot_nay": len(lam), "con_lai": len(con), "tong_video": len(theo_video),
            "thoi_gian_s": round(time.time() - t0, 1)}


def b_manifest(cfg) -> dict:
    sp = nap_split(cfg)
    kh = nap_ke_hoach(cfg)
    out = kiem_chua_xong(cfg)
    if not (tam(cfg) / "bdd_xong.json").is_file():
        raise m.LoiDuLieu("phan BDD chua xong")
    src = P(cfg["bdd"]["dataset"])
    rows, dat = [], {"hardlink": 0, "sao_chep": 0}
    for r in nap_bdd(cfg, sp):
        cach = {"hardlink" if Path.samefile(src / r[k], out / r[k]) else "sao_chep" for k in ("anh_ra", "nhan_ra", "ignore_ra")
                if r[k]}
        c = "hardlink" if cach == {"hardlink"} else "sao_chep"
        dat[c] += 1
        rows.append({**{k: r[k] for k in bx.COT_MANIFEST}, "nguon": "bdd", "dat_file": c})
    det_rows, dem = [], {k: 0 for k in m.KHOA_DEM} | {"box_xml_khong_hop_le": 0}
    videos = sorted({e["video"] for e in kh["detrac"]})
    for v in videos:
        st = tam(cfg) / "detrac" / f"{v}.json"
        if not st.is_file():
            raise m.LoiDuLieu(f"UA-DETRAC {v} chua xuat")
        j = json.loads(st.read_text(encoding="utf-8"))
        det_rows += j["rows"]
        for k in dem:
            dem[k] += j["dem"][k]
    if len(det_rows) != len(kh["detrac"]):
        raise m.LoiDuLieu(f"so dong DETRAC {len(det_rows)} != ke hoach {len(kh['detrac'])}")
    det_rows.sort(key=lambda r: (r["split"], r["video"], int(r["frame_idx"])))
    rows += det_rows
    ro_ri = m.kiem_ro_ri(rows, nap_tap_cam(cfg), {"train": set(sp["train_fit"]), "dev": set(sp["dev"])})
    dv = kh["dem_video"]
    tom_them = {
        "cau_hinh": {"duong_dan": cfg["_duong_dan"], "sha256": cfg["_sha256"]},
        "nguon": {
            "bdd": {"dataset": cfg["bdd"]["dataset"], "sha256_manifest_anh": cfg["bdd"]["sha256_manifest_anh"],
                    "file_split": cfg["bdd"]["file_split"], "sha256_split": cfg["bdd"]["sha256_split"],
                    "anh_goc_cua_anh_nguon": cfg["bdd"]["anh_goc"], "dat_file_theo_anh": dat,
                    "quy_tac": "dong manifest giu nguyen gia tri cac cot bdd5_v1 (frame_idx 0-based, buoc 5)"},
            "detrac": {"xml": cfg["detrac"]["xml"], "anh_goc_cua_anh_nguon": cfg["detrac"]["anh"],
                       "video": {"train": sp["detrac"]["train_fit"], "dev": sp["detrac"]["dev"]},
                       "dem_frame": {k: sum(x[k] for x in dv.values()) for k in next(iter(dv.values())) if k != "split"},
                       "dem_nhan": dem}},
        "quy_tac_detrac": {
            "buoc_frame": kh["buoc_frame"],
            "frame": "num cua <frame> (= so trong imgNNNNN.jpg, bat dau 1) chia het cho buoc_frame VA co <frame> trong XML; "
                     "anh khong co <frame> bi loai; cot frame_idx = num",
            "anh_xa_lop": m.ANH_XA_LOP, "anh_xa_ghi_chu": "van->car, others->truck la GIA DINH (quyet dinh nguoi dung 16/10)",
            "to_xam": "ignored_region cua video ap moi frame -> bdd_yolo_export.xuat_mot_anh (vai tro to_xam): tru pixel "
                      "trong box co nhan, mau [114,114,114], JPEG q95, ignore/<split>/<anh>.txt (x1 y1 x2 y2 px)",
            "am_tinh": "frame co <frame> nhung 0 box hop le -> giu, file nhan rong (nhu BDD: frame da gan nhan)",
            "box": "cat vao khung; < 1 px sau cat -> bo; box XML khong huu han / w,h <= 0 -> bo",
            "xe_hai_banh": "UA-DETRAC khong gan nhan motorcycle/bicycle -> nhieu am tinh gia; do o buoc danh gia sau"},
        "jpeg_thieu_eoi": {k: len(v) for k, v in kh["eoi_thieu"].items()},
        "kiem_trung": ro_ri,
    }
    tom = m.ghi_manifest(out, rows, P(cfg["file_split"]), sp, tom_them, f"cau hinh {cfg['ten']}")
    if not (out / ".gitignore").exists():
        (out / ".gitignore").write_text("*\n", encoding="utf-8")
    return {"so_dong": len(rows), "manifest_sha256": tom["manifest_anh"]["sha256"], "kiem_trung": ro_ri,
            "dat_file_bdd": dat, "dem_nhan_detrac": dem}


def b_kiem(cfg) -> dict:
    out = P(cfg["dau_ra"])
    t0 = time.time()
    hd = bx.kiem_hop_dong_dataset(out / "data.yaml")
    sp = nap_split(cfg)
    rows = doc_csv(out / "manifest" / "anh.csv")
    loi = []
    # (a) dong BDD trung NGUYEN gia tri voi manifest bdd5_v1
    bdd = {r["anh_ra"]: r for r in nap_bdd(cfg, sp)}
    mb = [r for r in rows if r["nguon"] == "bdd"]
    lech = [r["anh_ra"] for r in mb if r["anh_ra"] not in bdd or any(r[k] != bdd[r["anh_ra"]][k] for k in bx.COT_MANIFEST)]
    if lech or len(mb) != len(bdd):
        loi.append(f"BDD: {len(lech)} dong lech manifest bdd5_v1, {len(mb)} vs {len(bdd)} dong")
    # (b) UA-DETRAC: tinh lai tu XML (doc lap voi buoc xuat): tap frame, so box theo lop, so vung ignore
    d = cfg["detrac"]
    buoc = int(d["buoc_frame"])
    md = [r for r in rows if r["nguon"] == "detrac"]
    theo_v: dict[str, dict[int, dict]] = {}
    for r in md:
        theo_v.setdefault(r["video"], {})[int(r["frame_idx"])] = r
    dem_b = {"video": 0, "frame": 0, "box_lech": 0, "vung_lech": 0, "frame_thieu_thua": 0, "split_sai": 0}
    for s, k in (("train", "train_fit"), ("dev", "dev")):
        for v in sp["detrac"][k]:
            dem_b["video"] += 1
            x = m.doc_xml_detrac(P(d["xml"]) / f"{v}.xml")
            ky = {n for n in x["frames"] if n % buoc == 0}
            co = theo_v.get(v, {})
            if set(co) != ky:
                dem_b["frame_thieu_thua"] += len(set(co) ^ ky)
            for n in ky & set(co):
                r = co[n]
                dem_b["frame"] += 1
                dem_b["split_sai"] += int(r["split"] != s)
                W, H = int(r["w"]), int(r["h"])
                n_ky = {c: 0 for c in LOP_5}
                for vt, l, t, w, h in x["frames"][n]:
                    if w > 0 and h > 0 and bx._hop_le_box(l, t, l + w, t + h, W, H) is not None:
                        n_ky[m.ANH_XA_LOP[vt]] += 1
                dem_b["box_lech"] += int(any(int(r[f"n_{c}"]) != n_ky[c] for c in LOP_5))
                nv = sum(1 for l, t, w, h in x["ignored"] if bx._hop_le_box(l, t, l + w, t + h, W, H) is not None)
                dem_b["vung_lech"] += int(int(r["n_vung_to_xam"]) != nv)
    if set(theo_v) != set(sp["detrac"]["train_fit"]) | set(sp["detrac"]["dev"]):
        loi.append("UA-DETRAC: tap video trong manifest khac split")
    if any(dem_b[k] for k in ("box_lech", "vung_lech", "frame_thieu_thua", "split_sai")):
        loi.append(f"UA-DETRAC lech XML: {dem_b}")
    # (c) ro ri lan 3 tren manifest cuoi
    ro_ri = m.kiem_ro_ri(rows, nap_tap_cam(cfg), {"train": set(sp["train_fit"]), "dev": set(sp["dev"])})
    kq = {"hop_dong": hd, "bo_sung": {"bdd_trung_manifest_bdd5": {"so_dong": len(mb), "lech": len(lech)},
                                      "detrac_tinh_lai_tu_xml": dem_b, "kiem_ro_ri": ro_ri, "loi": loi},
          "thoi_gian_s": round(time.time() - t0, 1)}
    m.ghi_json(GOC / "results/phase2/mixed5/kiem_hop_dong.json", kq)
    if loi:
        raise m.LoiDuLieu("; ".join(loi))
    return kq


def b_thong_ke(cfg) -> dict:
    out = P(cfg["dau_ra"])
    tom = json.loads((out / "manifest" / "tom_tat.json").read_text(encoding="utf-8"))
    rows = doc_csv(out / "manifest" / "anh.csv")
    tk = tom["thong_ke"]
    dl = {}
    for r in rows:
        for k in ("anh_ra", "nhan_ra", "ignore_ra"):
            if r[k]:
                st = (out / r[k]).stat()
                e = dl.setdefault(r["nguon"], {"bytes": 0, "bytes_hardlink": 0, "so_file": 0})
                e["bytes"] += st.st_size
                e["bytes_hardlink"] += st.st_size if st.st_nlink > 1 else 0
                e["so_file"] += 1
    tr = tk["tong"]["train"]["so_anh"]
    kq = {"thong_ke": tk, "ti_le_anh_detrac_train": round(tk["detrac"]["train"]["so_anh"] / tr, 4),
          "ti_le_anh_detrac_dev": round(tk["detrac"]["dev"]["so_anh"] / tk["tong"]["dev"]["so_anh"], 4),
          "dung_luong": {k: {**v, "gb": round(v["bytes"] / 2**30, 3),
                             "gb_moi_tren_dia": round((v["bytes"] - v["bytes_hardlink"]) / 2**30, 3)} for k, v in dl.items()},
          "o_dia_trong_gb": round(shutil.disk_usage(GOC).free / 2**30, 2)}
    lines = ["| nguon | split | video | anh | am tinh | anh to xam | " + " | ".join(LOP_5) + " | tong box |",
             "|---|---|---:|---:|---:|---:|" + "---:|" * (len(LOP_5) + 1)]
    for ng in ("bdd", "detrac", "tong"):
        for s in ("train", "dev"):
            t = tk[ng][s]
            lines.append(f"| {ng} | {s} | {t['so_video']} | {t['so_anh']} | {t['am_tinh']} | {t['anh_to_xam']} | "
                         + " | ".join(str(t["box"][c]) for c in LOP_5) + f" | {t['tong_box']} |")
    lines += ["", "| nguon | split | " + " | ".join(f"anh co {c}" for c in LOP_5) + " |", "|---|---|" + "---:|" * len(LOP_5)]
    for ng in ("bdd", "detrac", "tong"):
        for s in ("train", "dev"):
            lines.append(f"| {ng} | {s} | " + " | ".join(str(tk[ng][s]["anh_co_lop"][c]) for c in LOP_5) + " |")
    kq["bang_md"] = "\n".join(lines)
    m.ghi_json(GOC / "results/phase2/mixed5/thong_ke.json", kq)
    (GOC / "results/phase2/mixed5/thong_ke.md").write_bytes((kq["bang_md"] + "\n").encode("utf-8"))
    return kq


def b_overlay(cfg) -> dict:
    import cv2
    out = P(cfg["dau_ra"])
    rows = doc_csv(out / "manifest" / "anh.csv")
    od = GOC / "results/phase2/mixed5/overlay"
    od.mkdir(parents=True, exist_ok=True)
    chon = []
    for dk in (lambda r: r["nguon"] == "detrac" and r["split"] == "train" and r["xu_ly"] == "to_xam",
               lambda r: r["nguon"] == "detrac" and r["split"] == "dev" and r["xu_ly"] == "to_xam",
               lambda r: r["nguon"] == "detrac" and r["split"] == "train" and int(r["n_truck"]) > 0,
               lambda r: r["nguon"] == "detrac" and r["split"] == "train" and r["xu_ly"] == "sao_chep",
               lambda r: r["nguon"] == "detrac" and r["am_tinh"] == "1",
               lambda r: r["nguon"] == "bdd" and r["split"] == "train" and r["xu_ly"] == "to_xam"):
        ds = [r for r in rows if dk(r) and r not in chon]
        if ds:
            chon.append(ds[len(ds) // 2])
    mau = [(0, 255, 0), (0, 165, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0)]
    ghi = []
    for r in chon:
        im = cv2.imread(str(out / r["anh_ra"]))
        H, W = im.shape[:2]
        for ln in (out / r["nhan_ra"]).read_text(encoding="utf-8").splitlines():
            c, cx, cy, w, h = ln.split()
            c, cx, cy, w, h = int(c), float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
            cv2.rectangle(im, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), mau[c], 2)
            cv2.putText(im, LOP_5[c], (int(cx - w / 2), max(10, int(cy - h / 2) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        mau[c], 1)
        if r["ignore_ra"]:
            for ln in (out / r["ignore_ra"]).read_text(encoding="utf-8").splitlines():
                x1, y1, x2, y2 = (int(float(v)) for v in ln.split())
                cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 1)
        p = od / f"{r['nguon']}_{r['split']}_{Path(r['anh_ra']).stem}.jpg"
        cv2.imwrite(str(p), im)
        ghi.append(bx.rel(p, GOC))
    return {"overlay": ghi}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CFG_MAC_DINH))
    ap.add_argument("--buoc", required=True, choices=["chia", "ke-hoach", "xuat-bdd", "xuat-detrac", "manifest", "kiem",
                                                      "thong-ke", "overlay"])
    ap.add_argument("--gioi-han-giay", type=float, default=480.0, help="xuat-detrac: dung sau video vuot moc nay")
    a = ap.parse_args()
    try:
        cfg = doc_cfg(Path(a.config))
        f = {"chia": b_chia, "ke-hoach": b_ke_hoach, "xuat-bdd": b_xuat_bdd, "manifest": b_manifest, "kiem": b_kiem,
             "thong-ke": b_thong_ke, "overlay": b_overlay}
        kq = b_xuat_detrac(cfg, a.gioi_han_giay) if a.buoc == "xuat-detrac" else f[a.buoc](cfg)
        print(json.dumps(kq, ensure_ascii=False, indent=1, default=str)[:12000])
        return 0
    except m.ThieuDuLieu as e:
        print(f"[THIEU DU LIEU] {e}", file=sys.stderr)
        return 2
    except m.LoiDuLieu as e:
        print(f"[LOI DU LIEU] {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
