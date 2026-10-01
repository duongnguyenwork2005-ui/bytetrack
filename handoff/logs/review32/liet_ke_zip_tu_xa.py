#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 32 -- liet ke muc luc (central directory) cua mot ZIP tren may chu bang HTTP Range, KHONG tai du lieu ben trong.

zipfile.ZipFile doc tren mot doi tuong file "ao": moi lan doc chi xin dung khoang byte can (khoi 1 MiB, co dem). Voi ZIP
lon chi ton phan duoi file + central directory (vai MB). Ghi JSON: URL cuoi sau chuyen huong, kich thuoc, so byte da tai,
so member, tien to thu muc, cac member khop mau (vd box_track_20).

Tuy chon:
  --ds-day-du F.tsv     ghi TOAN BO muc luc (ten, kich thuoc, kich thuoc nen, CRC32, offset) de doi chieu.
  --tai-tien-to P --dich D
                        CHI tai cac member bat dau bang P (vd bdd100k/labels/box_track_20/) vao D, giu duong dan;
                        zipfile kiem CRC32 khi doc (sai -> dung). Ghi file tam roi doi ten; file da co: trung byte -> bo
                        qua, khac -> KHONG ghi de (bao xung dot, exit 1). Khong tai member nao ngoai P.
"""
import argparse
import collections
import io
import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath

import requests

KHOI = 1 << 20


class FileHttp(io.RawIOBase):
    def __init__(self, url):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "Mozilla/5.0 (research data provenance check)"
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True, timeout=60, stream=True)
        r.close()
        if r.status_code != 206:
            raise OSError(f"may chu khong tra 206 cho Range (http {r.status_code})")
        m = re.search(r"/(\d+)$", r.headers.get("Content-Range", ""))
        if not m:
            raise OSError(f"khong co Content-Range: {r.headers}")
        self.url_cuoi = r.url
        self.lich_su = [(h.status_code, h.headers.get("Location", "")[:120]) for h in r.history]
        self.n = int(m.group(1))
        self.pos = 0
        self.dem = {}
        self.da_tai = 0
        self.so_yeu_cau = 1

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.n + off
        return self.pos

    def _khoi(self, i):
        if i not in self.dem:
            a = i * KHOI
            b = min(self.n, a + KHOI) - 1
            loi = None
            for _ in range(4):
                try:
                    r = self.s.get(self.url_cuoi, headers={"Range": f"bytes={a}-{b}"}, timeout=120)
                    if r.status_code == 206 and len(r.content) == b - a + 1:
                        break
                    loi = f"Range {a}-{b}: http {r.status_code}, {len(r.content)} B"
                except requests.RequestException as e:
                    loi = repr(e)
                time.sleep(3)
            else:
                raise OSError(loi)
            if len(self.dem) >= 64:  # gioi han bo nho dem: bo khoi cu nhat
                self.dem.pop(next(iter(self.dem)))
            self.dem[i] = r.content
            self.da_tai += len(r.content)
            self.so_yeu_cau += 1
        return self.dem[i]

    def read(self, k=-1):
        if k is None or k < 0:
            k = self.n - self.pos
        ra = bytearray()
        while k > 0 and self.pos < self.n:
            i, o = divmod(self.pos, KHOI)
            b = self._khoi(i)[o:o + k]
            ra += b
            self.pos += len(b)
            k -= len(b)
        return bytes(ra)

    def readinto(self, buf):
        d = self.read(len(buf))
        buf[:len(d)] = d
        return len(d)


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


def tai_member(z, infos, tien_to, dich):
    """Tai cac member bat dau bang tien_to vao dich (giu duong dan). Tra dict thong ke."""
    chon = [i for i in infos if i.filename.startswith(tien_to) and not i.is_dir()]
    sai = [i.filename for i in chon if i.filename.startswith("/") or "\\" in i.filename
           or ".." in PurePosixPath(i.filename).parts]
    if sai:
        raise ValueError(f"member khong an toan: {sai[:5]}")
    moi = trung = 0
    xung_dot = []
    for i in chon:
        ra = dich / i.filename
        if ra.exists():
            if trung_byte(z, i, ra):
                trung += 1
            else:
                xung_dot.append(i.filename)
            continue
        ra.parent.mkdir(parents=True, exist_ok=True)
        tam = ra.with_name(ra.name + ".tmp_tai")
        try:
            with z.open(i) as nguon, open(tam, "wb") as o:  # zipfile kiem CRC32 khi doc het member
                while True:
                    b = nguon.read(1 << 20)
                    if not b:
                        break
                    o.write(b)
        except BaseException:
            tam.unlink(missing_ok=True)
            raise
        os.replace(tam, ra)
        moi += 1
    return {"tien_to": tien_to, "so_member_chon": len(chon), "so_ghi_moi": moi, "so_trung_bo_qua": trung,
            "so_xung_dot": len(xung_dot), "vi_du_xung_dot": xung_dot[:20],
            "tong_byte_giai_nen": sum(i.file_size for i in chon), "tong_byte_nen": sum(i.compress_size for i in chon)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--ra", required=True, help="file JSON ket qua")
    ap.add_argument("--mau", default=r"box_track|labels/|\.json$", help="regex member can liet ke chi tiet")
    ap.add_argument("--ds-day-du", help="ghi toan bo muc luc ra file TSV")
    ap.add_argument("--tai-tien-to", help="chi tai cac member bat dau bang tien to nay")
    ap.add_argument("--dich", help="thu muc dich cho --tai-tien-to")
    a = ap.parse_args()
    f = FileHttp(a.url)
    z = zipfile.ZipFile(f)
    infos = z.infolist()
    if a.ds_day_du:
        with open(a.ds_day_du, "w", encoding="utf-8", newline="\n") as o:
            o.write("ten\tkich_thuoc\tkich_thuoc_nen\tcrc32\toffset\n")
            for i in infos:
                o.write(f"{i.filename}\t{i.file_size}\t{i.compress_size}\t{i.CRC:08x}\t{i.header_offset}\n")
    ket_qua_tai = None
    if a.tai_tien_to:
        if not a.dich:
            ap.error("--tai-tien-to can --dich")
        byte_truoc = f.da_tai
        ket_qua_tai = tai_member(z, infos, a.tai_tien_to, Path(a.dich))
        ket_qua_tai["byte_da_tai_cho_member"] = f.da_tai - byte_truoc
    z.close()
    tep = [i for i in infos if not i.is_dir()]
    mau = re.compile(a.mau)
    khop = [i for i in tep if mau.search(i.filename)]
    kq = {
        "url": a.url, "url_cuoi_sau_chuyen_huong": f.url_cuoi.split("?")[0], "chuyen_huong": f.lich_su,
        "kich_thuoc_byte": f.n, "byte_da_tai_de_doc_muc_luc": f.da_tai, "so_yeu_cau_range": f.so_yeu_cau,
        "so_member": len(infos), "so_tep": len(tep),
        "tien_to_3_cap": dict(collections.Counter("/".join(i.filename.split("/")[:3]) for i in tep).most_common(30)),
        "duoi_tep": dict(collections.Counter(i.filename.rsplit(".", 1)[-1].lower() if "." in i.filename else "" for i in tep).most_common(15)),
        "mau": a.mau, "so_tep_khop_mau": len(khop),
        "tien_to_tep_khop_mau": dict(collections.Counter("/".join(i.filename.split("/")[:-1]) for i in khop).most_common(30)),
        "vi_du_khop_mau": [[i.filename, i.file_size, i.compress_size, f"{i.CRC:08x}"] for i in khop[:15]],
        "tong_giai_nen_khop_mau": sum(i.file_size for i in khop),
        "tong_nen_khop_mau": sum(i.compress_size for i in khop),
        "tai_member": ket_qua_tai,
    }
    with open(a.ra, "w", encoding="utf-8", newline="\n") as o:
        o.write(json.dumps(kq, ensure_ascii=False, indent=2) + "\n")
    tom = {k: kq[k] for k in ("kich_thuoc_byte", "byte_da_tai_de_doc_muc_luc", "so_yeu_cau_range", "so_tep", "so_tep_khop_mau",
                              "tien_to_3_cap", "tien_to_tep_khop_mau", "tai_member")}
    sys.stdout.write(json.dumps(tom, ensure_ascii=True, indent=1) + "\n")
    return 1 if ket_qua_tai and ket_qua_tai["so_xung_dot"] else 0


if __name__ == "__main__":
    sys.exit(main())
