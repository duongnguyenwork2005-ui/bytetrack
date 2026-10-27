"""
xuat_smoke_mixed5.py -- dung tap con SMOKE data/yolo_mixed5_v1_smoke (R52B): CHI kiem duong chay, KHONG dung cho ket qua.

    python -X utf8 src/mixed5/xuat_smoke_mixed5.py --buoc xuat     # hardlink file + manifest v2 (tiep tuc duoc)
    python -X utf8 src/mixed5/xuat_smoke_mixed5.py --buoc kiem     # bx.kiem_hop_dong_dataset + doi chieu nguon

QUY TAC TAP CON
1. Phan BDD = NGUYEN tap con smoke da dinh nghia trong configs/phase2/detector/bdd5_train.yaml (`tap_con.smoke`: 16 video
   train-fit / 4 video dev, buoc_frame 10, seed 0), lay tu data/yolo_bdd5_v1_smoke (da xuat tu vong 27). Tap video duoc
   tinh lai bang `bx.chon_tap_con` va phai trung video.json cua bdd5_v1_smoke; dataset nguon phai dat hop dong truoc.
2. Phan UA-DETRAC = `SO_VIDEO_DETRAC` video chon bang np.random.default_rng(SEED).permutation tren danh sach DA SAP XEP
   cua mixed5_v1.json (detrac.train_fit / detrac.dev) -- cung cach `bx.chon_tap_con`. Lay MOI dong cua cac video do
   trong manifest data/yolo_mixed5_v1 (tuc giu quy tac day du: num % 10 == 0 va co <frame> trong XML).
3. File anh / nhan / ignore: hardlink tu dataset nguon (khong doi byte); JPEG thieu EOI -> sao chep (ultralytics ghi
   lai file thieu EOI, hardlink se lam doi ca nguon). Dong manifest giu nguyen gia tri cot cua manifest nguon.
4. File split da khoa = configs/phase2/detector/splits/mixed5_v1.json (moi video smoke thuoc dung split cua no).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mixed5_lib as m  # noqa: E402
import xuat_mixed5 as xm  # noqa: E402
from mixed5_lib import GOC, bx  # noqa: E402

DAU_RA = GOC / "data/yolo_mixed5_v1_smoke"
NGUON_BDD = GOC / "data/yolo_bdd5_v1_smoke"
NGUON_FULL = GOC / "data/yolo_mixed5_v1"
CFG_BDD = GOC / "configs/phase2/detector/bdd5_train.yaml"
SO_VIDEO_DETRAC = {"train": 3, "dev": 1}
SEED = 0
KQ = GOC / "results/phase2/mixed5/smoke_dataset"


def chon_video_detrac(sp: dict, so: dict = SO_VIDEO_DETRAC, seed: int = SEED) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    out = {}
    for s, k in (("train", "train_fit"), ("dev", "dev")):
        v = sorted(sp["detrac"][k])
        out[s] = sorted(v[i] for i in rng.permutation(len(v))[:int(so[s])])
    return out


def chon_video_bdd(bdd_split: Path) -> dict[str, list[str]]:
    import yaml
    cfg = yaml.safe_load(CFG_BDD.read_text(encoding="utf-8"))
    sp = json.loads(bdd_split.read_text(encoding="utf-8"))
    return bx.chon_tap_con(sp, cfg, "smoke")


def chon_dong(rows_bdd: list[dict], rows_full: list[dict], vdet: dict) -> list[dict]:
    """Dong manifest smoke: moi dong BDD smoke + moi dong DETRAC cua video da chon (giu nguyen gia tri cot)."""
    out = [{**{k: r[k] for k in bx.COT_MANIFEST}, "nguon": "bdd", "dat_file": ""} for r in rows_bdd]
    chon = {(s, v) for s, vs in vdet.items() for v in vs}
    out += [{**{k: r[k] for k in bx.COT_MANIFEST}, "nguon": "detrac", "dat_file": ""} for r in rows_full
            if r["nguon"] == "detrac" and (r["split"], r["video"]) in chon]
    return out


def chuan_bi(cfg) -> dict:
    sp = xm.nap_split(cfg)
    hd_bdd = bx.kiem_hop_dong_dataset(NGUON_BDD / "data.yaml")
    tom_bdd = json.loads((NGUON_BDD / "manifest" / "tom_tat.json").read_text(encoding="utf-8"))
    if tom_bdd["file_split"]["sha256"] != cfg["bdd"]["sha256_split"]:
        raise m.LoiDuLieu("bdd5_v1_smoke lap tren split BDD khac bdd5_train_v1 da tham chieu trong mixed5_v1")
    vbdd = chon_video_bdd(GOC / cfg["bdd"]["file_split"])
    vj = json.loads((NGUON_BDD / "manifest" / "video.json").read_text(encoding="utf-8"))
    if {s: sorted(v) for s, v in vj.items()} != vbdd:
        raise m.LoiDuLieu("video bdd5_v1_smoke khac tap con smoke tinh lai tu bdd5_train.yaml")
    hd_full = json.loads((GOC / "results/phase2/mixed5/kiem_hop_dong.json").read_text(encoding="utf-8"))["hop_dong"]
    tom_full = json.loads((NGUON_FULL / "manifest" / "tom_tat.json").read_text(encoding="utf-8"))
    man_full = NGUON_FULL / "manifest" / "anh.csv"
    if bx.sha256_file(man_full) != tom_full["manifest_anh"]["sha256"] or hd_full["manifest_sha256"] != bx.sha256_file(man_full):
        raise m.LoiDuLieu("manifest mixed5_v1 khac ban da kiem hop dong R52A")
    vdet = chon_video_detrac(sp)
    rows = chon_dong(xm.doc_csv(NGUON_BDD / "manifest" / "anh.csv"), xm.doc_csv(man_full), vdet)
    video = {s: sorted(set(vbdd[s]) | set(vdet[s])) for s in ("train", "dev")}
    return {"sp": sp, "rows": rows, "video": video, "vbdd": vbdd, "vdet": vdet, "hd_bdd": hd_bdd,
            "manifest_full": bx.sha256_file(man_full), "manifest_bdd": tom_bdd["manifest_anh"]["sha256"]}


def b_xuat(cfg) -> dict:
    t0 = time.time()
    if (DAU_RA / "manifest" / "tom_tat.json").exists():
        raise m.LoiDuLieu(f"{DAU_RA} da co manifest -- khong ghi lai")
    c = chuan_bi(cfg)
    sp, rows = c["sp"], c["rows"]
    ro_ri = m.kiem_ro_ri(rows, xm.nap_tap_cam(cfg), {s: set(v) for s, v in c["video"].items()})
    for s in ("train", "dev"):
        for v in c["video"][s]:
            if v not in sp["train_fit" if s == "train" else "dev"]:
                raise m.LoiDuLieu(f"video smoke {v} khong thuoc {s} cua mixed5_v1.json")
    dem = {"hardlink": 0, "sao_chep": 0}
    for r in rows:
        src_dir = NGUON_BDD if r["nguon"] == "bdd" else NGUON_FULL
        cach = set()
        for k in ("anh_ra", "nhan_ra", "ignore_ra"):
            if r[k]:
                hl = not (k == "anh_ra" and not m.co_eoi_jpeg(src_dir / r[k]))
                cach.add(m.dat_file_bdd(src_dir / r[k], DAU_RA / r[k], hl))
        r["dat_file"] = "hardlink" if cach == {"hardlink"} else "sao_chep"
        dem[r["dat_file"]] += 1
    rows.sort(key=lambda r: (r["split"], r["nguon"], r["video"], int(r["frame_idx"])))
    man_dir = DAU_RA / "manifest"
    man_dir.mkdir(parents=True, exist_ok=True)
    man = man_dir / "anh.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=m.COT)
        w.writeheader()
        w.writerows(rows)
    (man_dir / "video.json").write_bytes((json.dumps(c["video"], indent=1) + "\n").encode("utf-8"))
    fsp = GOC / cfg["file_split"]
    (man_dir / "split_da_khoa.json").write_bytes(fsp.read_bytes())
    m.ghi_data_yaml(DAU_RA, f"cau hinh {cfg['ten']}, tap con smoke (src/mixed5/xuat_smoke_mixed5.py)")
    tom = {
        "ten": f"{sp['ten']}_smoke", "tap_con": "smoke", "loai": "SMOKE (tap con nho, KHONG dung cho ket qua nghien cuu)",
        "phien_ban_manifest": bx.PHIEN_BAN_MANIFEST, "phien_ban_quy_tac": m.PHIEN_BAN_QUY_TAC,
        "tao_luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "git_head": bx.git_head(), "lop": list(m.LOP_5),
        "file_split": {"duong_dan": bx.rel(fsp, GOC), "duong_dan_tuyet_doi": fsp.resolve().as_posix(),
                       "sha256": bx.sha256_file(fsp), "sha256_danh_sach": sp["sha256_danh_sach"],
                       "ban_sao": "manifest/split_da_khoa.json"},
        "sha256_data_yaml": bx.sha256_file(DAU_RA / "data.yaml"),
        "sha256_video_json": bx.sha256_file(man_dir / "video.json"),
        "video": {s: len(v) for s, v in c["video"].items()},
        "thong_ke": m.thong_ke_rows(rows),
        "quy_tac_tap_con": {
            "bdd": {"nguon": bx.rel(NGUON_BDD, GOC), "sha256_manifest_anh_nguon": c["manifest_bdd"],
                    "dinh_nghia": "bdd5_train.yaml tap_con.smoke (16 train / 4 dev video, buoc_frame 10, seed 0)",
                    "video": c["vbdd"]},
            "detrac": {"nguon": bx.rel(NGUON_FULL, GOC), "sha256_manifest_anh_nguon": c["manifest_full"],
                       "so_video": SO_VIDEO_DETRAC, "seed": SEED,
                       "cach_chon": "np.random.default_rng(seed).permutation tren danh sach da sap xep (nhu bx.chon_tap_con)",
                       "frame": "moi dong cua video da chon trong manifest mixed5_v1 (num % 10 == 0, co <frame>)",
                       "video": c["vdet"]},
            "dat_file": "hardlink tu dataset nguon; JPEG thieu EOI -> sao chep", "dat_file_theo_anh": dem},
        "kiem_trung": ro_ri,
        "manifest_anh": {"duong_dan": "manifest/anh.csv", "so_dong": len(rows), "sha256": bx.sha256_file(man),
                         "cot_them": ["nguon", "dat_file"]},
    }
    m.ghi_json(man_dir / "tom_tat.json", tom)
    if not (DAU_RA / ".gitignore").exists():
        (DAU_RA / ".gitignore").write_text("*\n", encoding="utf-8")
    return {"so_dong": len(rows), "video": tom["video"], "video_detrac": c["vdet"], "dat_file": dem,
            "manifest_sha256": tom["manifest_anh"]["sha256"], "kiem_trung": ro_ri, "thoi_gian_s": round(time.time() - t0, 1)}


def b_kiem(cfg) -> dict:
    t0 = time.time()
    hd = bx.kiem_hop_dong_dataset(DAU_RA / "data.yaml")
    c = chuan_bi(cfg)
    rows = xm.doc_csv(DAU_RA / "manifest" / "anh.csv")
    loi = []
    # dong smoke trung NGUYEN gia tri cot voi dong nguon; file la cung inode (hardlink) hoac cung sha256 (sao chep)
    ky = {(r["nguon"], r["anh_ra"]): r for r in c["rows"]}
    lech = [r["anh_ra"] for r in rows if (r["nguon"], r["anh_ra"]) not in ky
            or any(r[k] != ky[(r["nguon"], r["anh_ra"])][k] for k in bx.COT_MANIFEST)]
    if lech or len(rows) != len(ky):
        loi.append(f"{len(lech)} dong lech nguon; {len(rows)} vs {len(ky)} dong")
    khong_cung_file = 0
    for r in rows:
        src_dir = NGUON_BDD if r["nguon"] == "bdd" else NGUON_FULL
        for k in ("anh_ra", "nhan_ra", "ignore_ra"):
            if r[k] and not Path.samefile(src_dir / r[k], DAU_RA / r[k]) and \
                    bx.sha256_file(src_dir / r[k]) != bx.sha256_file(DAU_RA / r[k]):
                khong_cung_file += 1
    if khong_cung_file:
        loi.append(f"{khong_cung_file} file khac byte voi nguon")
    vj = json.loads((DAU_RA / "manifest" / "video.json").read_text(encoding="utf-8"))
    if vj != c["video"]:
        loi.append("video.json khac tap video tinh lai")
    ro_ri = m.kiem_ro_ri(rows, xm.nap_tap_cam(cfg), {s: set(v) for s, v in vj.items()})
    tom = json.loads((DAU_RA / "manifest" / "tom_tat.json").read_text(encoding="utf-8"))
    kq = {"hop_dong": hd, "hop_dong_nguon_bdd5_v1_smoke": {k: c["hd_bdd"][k] for k in ("loi", "ten", "manifest_sha256")},
          "bo_sung": {"dong_lech_nguon": len(lech), "file_khac_byte_nguon": khong_cung_file, "kiem_ro_ri": ro_ri,
                      "video_detrac": c["vdet"], "loi": loi},
          "thong_ke": tom["thong_ke"], "thoi_gian_s": round(time.time() - t0, 1)}
    m.ghi_json(KQ / "kiem_hop_dong_smoke.json", kq)
    if loi:
        raise m.LoiDuLieu("; ".join(loi))
    return kq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buoc", required=True, choices=["xuat", "kiem"])
    a = ap.parse_args()
    try:
        cfg = xm.doc_cfg(xm.CFG_MAC_DINH)
        kq = (b_xuat if a.buoc == "xuat" else b_kiem)(cfg)
        print(json.dumps(kq, ensure_ascii=False, indent=1, default=str)[:8000])
        return 0
    except m.ThieuDuLieu as e:
        print(f"[THIEU DU LIEU] {e}", file=sys.stderr)
        return 2
    except m.LoiDuLieu as e:
        print(f"[LOI DU LIEU] {e}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    sys.exit(main())
