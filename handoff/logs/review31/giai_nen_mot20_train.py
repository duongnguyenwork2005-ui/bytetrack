#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 31 -- giai nen anh BDD100K MOT 2020 train DA XAC MINH vao data/raw/.

Chi nhan archive co ban ghi DAT cua tai_mot20_train.py (<thu-muc-ban-ghi-tai>/<ten>.json) va kich thuoc file khop ban ghi.

Kiem truoc, CHUA ghi gi (sai -> exit 2):
  - moi member la tep dang  bdd100k/images/track/train/<video>/<anh>  (thu muc thi bo qua), khong duong dan tuyet doi,
    khong '..', khong '\\', khong long bdd100k/bdd100k, khong ky tu Windows cam trong ten;
  - khong trung duong dan giua cac archive voi noi dung khac (CRC/kich thuoc), khong hai duong dan chi khac hoa/thuong;
  - du dung luong trong cho phan chua co tren dia (+ du phong).
Dich = <goc>/<member>, goc mac dinh data/raw  ->  data/raw/bdd100k/images/track/train/<video>/<anh>.

Khi ghi:
  - dich chua co -> ghi <dich>.tmp_giai_nen roi doi ten (bi ngat giua chung khong de lai file do dang voi ten that);
  - dich da co va TRUNG tung byte voi member -> bo qua (dem);
  - dich da co ma KHAC noi dung -> KHONG ghi de, ghi vao danh sach xung dot; ket thuc exit 1;
  - zipfile kiem CRC khi doc tung member (sai -> dung, exit 1).
Khong xoa ZIP, khong cham images/track/val hay nhan. Ban ghi JSON moi archive + tong hop o --thu-muc-ban-ghi.
Archive da giai nen hoan tat (ban ghi hoan_tat, cung sha256 zip, 0 xung dot) -> bo qua, nen chay lai duoc sau moi phan
tai xong; --lam-lai de so lai tung byte. Phan kiem truoc van quet muc luc cua MOI archive da xac minh.
"""
import argparse
import collections
import datetime as dt
import json
import os
import re
import shutil
import sys
import time
import zipfile
import zlib
from pathlib import Path, PurePosixPath

TIEN_TO = ("bdd100k", "images", "track", "train")
KY_TU_CAM = re.compile(r'[<>:"|?*\x00-\x1f]')


def bay_gio():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def ghi_log(duong, **kv):
    kv = {"t": bay_gio(), **kv}
    dong = json.dumps(kv, ensure_ascii=False)
    with open(duong, "a", encoding="utf-8") as f:
        f.write(dong + "\n")
    print(dong, flush=True)


def kiem_member(ten):
    """Tra None neu hop le, hoac ly do sai."""
    if ten.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", ten) or "\\" in ten:
        return "duong dan tuyet doi hoac co '\\'"
    p = PurePosixPath(ten).parts
    if ".." in p or "." in p:
        return "co '..' hoac '.'"
    if tuple(p[:4]) != TIEN_TO:
        return f"khong bat dau bang {'/'.join(TIEN_TO)}/"
    if len(p) != 6:
        return f"do sau {len(p)} != 6 (bdd100k/images/track/train/<video>/<anh>)"
    if any(KY_TU_CAM.search(x) or x.endswith((" ", ".")) for x in p):
        return "ten co ky tu Windows cam hoac ket thuc bang ' '/'.'"
    return None


def trung_byte(z, info, dich, khoi=1 << 20):
    if dich.stat().st_size != info.file_size:
        return False
    with z.open(info) as a, open(dich, "rb") as b:
        while True:
            x, y = a.read(khoi), b.read(khoi)
            if x != y:
                return False
            if not x:
                return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thu-muc-tai", default="data/raw/downloads/bdd100k_mot20")
    ap.add_argument("--thu-muc-ban-ghi-tai", default="handoff/logs/review31/10_tai")
    ap.add_argument("--goc", default="data/raw", help="thu muc goc de dat duong dan member")
    ap.add_argument("--thu-muc-ban-ghi", default="handoff/logs/review31/11_giai_nen")
    ap.add_argument("--du-phong-gb", type=float, default=5.0)
    ap.add_argument("--chi-kiem", action="store_true", help="chi chay phan kiem truoc, khong ghi gi vao --goc")
    ap.add_argument("--lam-lai", action="store_true", help="xu ly lai ca archive da giai nen hoan tat (so tung byte)")
    a = ap.parse_args()

    tai, bgt, goc, ghi = Path(a.thu_muc_tai), Path(a.thu_muc_ban_ghi_tai), Path(a.goc), Path(a.thu_muc_ban_ghi)
    ghi.mkdir(parents=True, exist_ok=True)
    log = ghi / f"chay_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}{'_chi_kiem' if a.chi_kiem else ''}.jsonl"

    # 1. archive hop le = co ban ghi DAT + kich thuoc khop
    archive = []
    for f in sorted(bgt.glob("images20-track-train-*.zip.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        z = tai / r["ten"]
        ok = r.get("trang_thai") == "DAT" and z.exists() and z.stat().st_size == r.get("bytes")
        ghi_log(log, su_kien="archive", ten=r["ten"], trang_thai_tai=r.get("trang_thai"), sha256=r.get("sha256"),
                bytes=r.get("bytes"), dung_duoc=ok)
        if ok:
            archive.append((z, r))
    if not archive:
        ghi_log(log, su_kien="loi", ly_do="khong co archive nao da xac minh")
        return 2

    # 2. kiem truoc toan bo member, chua ghi gi
    theo_duong = {}
    theo_thuong = {}
    sai = []
    can_ghi_byte = 0
    for z, r in archive:
        with zipfile.ZipFile(z) as zf:
            for i in zf.infolist():
                if i.is_dir():
                    continue
                ly_do = kiem_member(i.filename)
                if ly_do:
                    sai.append((z.name, i.filename, ly_do))
                    continue
                cu = theo_duong.get(i.filename)
                if cu and (cu[1], cu[2]) != (i.CRC, i.file_size):
                    sai.append((z.name, i.filename, f"trung duong dan voi {cu[0]} nhung khac CRC/kich thuoc"))
                theo_duong.setdefault(i.filename, (z.name, i.CRC, i.file_size))
                k = i.filename.lower()
                if theo_thuong.setdefault(k, i.filename) != i.filename:
                    sai.append((z.name, i.filename, f"chi khac hoa/thuong voi {theo_thuong[k]}"))
                if not (goc / i.filename).exists():
                    can_ghi_byte += i.file_size
    video = collections.Counter(PurePosixPath(p).parts[4] for p in theo_duong)
    trong = shutil.disk_usage(goc).free
    ghi_log(log, su_kien="kiem_truoc", so_archive=len(archive), so_tep=len(theo_duong), so_video=len(video),
            so_sai=len(sai), vi_du_sai=sai[:20], can_ghi_byte=can_ghi_byte, trong_byte=trong)
    if sai:
        ghi_log(log, su_kien="loi", ly_do="member khong hop le / xung dot giua archive -- khong ghi gi")
        return 2
    if trong < can_ghi_byte + a.du_phong_gb * 1e9:
        ghi_log(log, su_kien="loi", ly_do="khong du dung luong trong")
        return 2
    if a.chi_kiem:
        ghi_log(log, su_kien="ket_thuc", che_do="chi_kiem")
        return 0

    # 3. ghi (archive da giai nen hoan tat voi cung sha256, 0 xung dot -> bo qua, tru khi --lam-lai)
    tong_xung_dot = []
    for z, r in archive:
        cu = ghi / f"{z.name}.json"
        if cu.exists() and not a.lam_lai:
            c = json.loads(cu.read_text(encoding="utf-8"))
            if c.get("hoan_tat") and c.get("sha256_zip") == r["sha256"] and c.get("so_xung_dot") == 0:
                ghi_log(log, su_kien="bo_qua_archive", ten=z.name, ly_do="da giai nen hoan tat truoc", t_xong=c.get("t_xong"))
                continue
        t0 = time.time()
        moi = trung = 0
        xung_dot = []
        vid = set()
        with zipfile.ZipFile(z) as zf:
            for i in zf.infolist():
                if i.is_dir():
                    continue
                dich = goc / i.filename
                vid.add(PurePosixPath(i.filename).parts[4])
                if dich.exists():
                    if trung_byte(zf, i, dich):
                        trung += 1
                    else:
                        xung_dot.append(i.filename)
                    continue
                dich.parent.mkdir(parents=True, exist_ok=True)
                tam = dich.with_name(dich.name + ".tmp_giai_nen")
                try:
                    with zf.open(i) as nguon, open(tam, "wb") as ra:
                        shutil.copyfileobj(nguon, ra, 1 << 20)
                except (zipfile.BadZipFile, zlib.error, EOFError) as e:
                    tam.unlink(missing_ok=True)
                    ghi_log(log, su_kien="loi", ten=z.name, member=i.filename, ly_do=f"doc member loi: {e!r}",
                            so_tep_ghi_moi_truoc_loi=moi)
                    return 1
                os.replace(tam, dich)
                moi += 1
        bg = {"ten": z.name, "sha256_zip": r["sha256"], "hoan_tat": True, "so_tep_ghi_moi": moi, "so_tep_trung_bo_qua": trung,
              "so_xung_dot": len(xung_dot), "vi_du_xung_dot": xung_dot[:50], "so_video": len(vid),
              "giay": round(time.time() - t0, 1), "t_xong": bay_gio()}
        (ghi / f"{z.name}.json").write_text(json.dumps(bg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        ghi_log(log, su_kien="giai_nen_xong", **{k: v for k, v in bg.items() if k != "vi_du_xung_dot"})
        tong_xung_dot += xung_dot

    con_tam = [p.as_posix() for p in (goc / Path(*TIEN_TO)).rglob("*.tmp_giai_nen")]
    ghi_log(log, su_kien="ket_thuc", so_xung_dot=len(tong_xung_dot), file_tam_con_lai=con_tam[:20])
    return 1 if tong_xung_dot or con_tam else 0


if __name__ == "__main__":
    sys.exit(main())
