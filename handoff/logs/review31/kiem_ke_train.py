#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 31 -- kiem ke doc lap anh/nhan BDD100K MOT train tren dia sau khi giai nen (chi doc, khong sua gi).

So voi hop cac member cua moi archive da xac minh (ban ghi DAT o 10_tai/): thieu tren dia, thua tren dia, file tam
.tmp_giai_nen con sot. Dem video / anh / byte; trung ten video train voi val (ky vong 0); trang thai thu muc nhan
box_track_20/train; dung luong trong. Ghi JSON vao --ra.
"""
import argparse
import collections
import datetime as dt
import json
import shutil
import sys
import zipfile
from pathlib import Path, PurePosixPath


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goc", default="data/raw")
    ap.add_argument("--thu-muc-tai", default="data/raw/downloads/bdd100k_mot20")
    ap.add_argument("--thu-muc-ban-ghi-tai", default="handoff/logs/review31/10_tai")
    ap.add_argument("--ra", default="handoff/logs/review31/21_kiem_ke_train.json")
    a = ap.parse_args()

    goc = Path(a.goc)
    anh_train = goc / "bdd100k/images/track/train"
    anh_val = goc / "bdd100k/images/track/val"
    nhan_train = goc / "bdd100k/labels/box_track_20/train"

    # 1. archive da xac minh va hop member
    archive = []
    mong_doi = {}
    for f in sorted(Path(a.thu_muc_ban_ghi_tai).glob("images20-track-train-*.zip.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        z = Path(a.thu_muc_tai) / r["ten"]
        dung = r.get("trang_thai") == "DAT" and z.exists() and z.stat().st_size == r.get("bytes")
        archive.append({"ten": r["ten"], "trang_thai": r.get("trang_thai"), "bytes": r.get("bytes"),
                        "sha256": r.get("sha256"), "md5": r.get("md5"), "dung_duoc": dung})
        if dung:
            with zipfile.ZipFile(z) as zf:
                for i in zf.infolist():
                    if not i.is_dir():
                        mong_doi[i.filename] = i.file_size

    # 2. tren dia
    tren_dia = {}
    tam = []
    video_rong = []
    if anh_train.is_dir():
        for v in sorted(p for p in anh_train.iterdir() if p.is_dir()):
            co = False
            for p in v.iterdir():
                if p.is_file():
                    co = True
                    rel = p.relative_to(goc).as_posix()
                    if p.name.endswith(".tmp_giai_nen"):
                        tam.append(rel)
                    else:
                        tren_dia[rel] = p.stat().st_size
            if not co:
                video_rong.append(v.name)
        le = [p.relative_to(goc).as_posix() for p in anh_train.iterdir() if p.is_file()]
    else:
        le = []
    thieu = sorted(set(mong_doi) - set(tren_dia))
    thua = sorted(set(tren_dia) - set(mong_doi))
    lech_kich_thuoc = sorted(k for k in set(mong_doi) & set(tren_dia) if mong_doi[k] != tren_dia[k])
    video_train = sorted({PurePosixPath(k).parts[4] for k in tren_dia})
    video_val = sorted(p.name for p in anh_val.iterdir() if p.is_dir()) if anh_val.is_dir() else []
    duoi = collections.Counter(PurePosixPath(k).suffix.lower() for k in tren_dia)
    anh_moi_video = collections.Counter(PurePosixPath(k).parts[4] for k in tren_dia)

    nhan = {"duong_dan": nhan_train.as_posix(), "ton_tai": nhan_train.is_dir(),
            "so_file_json": len(list(nhan_train.glob("*.json"))) if nhan_train.is_dir() else 0}
    kq = {
        "t": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "archive": archive,
        "so_archive_dat": sum(x["dung_duoc"] for x in archive),
        "anh_train": {
            "duong_dan": anh_train.as_posix(), "ton_tai": anh_train.is_dir(),
            "so_video": len(video_train), "so_anh": len(tren_dia), "tong_byte": sum(tren_dia.values()),
            "duoi_tep": dict(duoi),
            "anh_moi_video_min": min(anh_moi_video.values()) if anh_moi_video else 0,
            "anh_moi_video_max": max(anh_moi_video.values()) if anh_moi_video else 0,
            "video_rong": video_rong, "tep_le_ngoai_thu_muc_video": le[:20],
        },
        "so_voi_archive": {"so_member_mong_doi": len(mong_doi), "thieu_tren_dia": len(thieu), "vi_du_thieu": thieu[:20],
                           "thua_tren_dia": len(thua), "vi_du_thua": thua[:20],
                           "lech_kich_thuoc": len(lech_kich_thuoc), "vi_du_lech": lech_kich_thuoc[:20],
                           "file_tam_con_sot": len(tam), "vi_du_tam": tam[:20]},
        "trung_ten_video_train_val": sorted(set(video_train) & set(video_val)),
        "so_video_val": len(video_val),
        "nhan_train": nhan,
        "dung_luong_trong_byte": shutil.disk_usage(goc).free,
    }
    Path(a.ra).write_text(json.dumps(kq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    tom = {k: kq[k] for k in ("so_archive_dat", "trung_ten_video_train_val", "so_video_val", "nhan_train",
                              "dung_luong_trong_byte")}
    tom["anh_train"] = {k: v for k, v in kq["anh_train"].items() if k not in ("video_rong", "tep_le_ngoai_thu_muc_video")}
    tom["so_voi_archive"] = {k: v for k, v in kq["so_voi_archive"].items() if not k.startswith("vi_du")}
    print(json.dumps(tom, ensure_ascii=False, indent=1))
    loi = (kq["so_voi_archive"]["thieu_tren_dia"] or kq["so_voi_archive"]["thua_tren_dia"]
           or kq["so_voi_archive"]["lech_kich_thuoc"] or tam or kq["trung_ten_video_train_val"] or video_rong or le)
    return 1 if loi else 0


if __name__ == "__main__":
    sys.exit(main())
