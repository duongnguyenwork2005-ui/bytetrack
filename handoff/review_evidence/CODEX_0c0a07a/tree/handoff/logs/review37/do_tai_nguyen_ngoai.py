#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 35 -- do tai nguyen BEN NGOAI runner (khong sua src/train_detector.py). Hai che do:
(Ban sao vong 37 tu logs/review35/do_tai_nguyen_ngoai.py; chi THEM tuy chon `ram --tho <csv>` ghi MAU THO co moc thoi
gian -- RSS tien trinh chinh, tong RSS chinh + con, so tien trinh con, RAM he thong con trong -- de tinh lai dinh
tung thoi diem; phan con lai giu nguyen.)

1) gan moc gio cho stdout:   <lenh> 2>&1 | python do_tai_nguyen_ngoai.py moc-gio --ra <log>
   Moi dong (tach theo \\n va \\r -- thanh tien do ultralytics) duoc ghi kem so giay tu luc bat dau, de suy ra thoi gian
   khoi tao (tu luc chay toi dong "Starting training") ma khong can sua runner.
   Sua sau smoke lan 1 (08_stderr_tac_vu_nen_lan1.txt): ban dau doc stdin o che do text theo locale (cp1252,
   surrogateescape) -> ky tu thanh tien do "━" (UTF-8 E2 94 81) thanh '\\udc81', ghi log UTF-8 bi UnicodeEncodeError,
   bo doc chet, pipe dut -> runner bao OSError [Errno 22]. Nay doc stdin dang BYTE, giai ma UTF-8 errors="replace"
   (stdout ultralytics la UTF-8; byte khong hop le -> U+FFFD) va KHONG BAO GIO ngung doc giua chung: loi khi ghi
   mot dong chi duoc dem (dong cuoi log) roi doc tiep toi EOF, de ben ghi khong bao gio bi dut pipe vi bo do.
2) do RAM tien trinh:         python do_tai_nguyen_ngoai.py ram --mau train_detector.py --ra <json> [--chu-ky 1]
   Cho tien trinh co command line chua --mau xuat hien, lay mau RSS cua no + MOI tien trinh con (worker dataloader)
   moi --chu-ky giay, toi khi no ket thuc. Ghi dinh RSS (MiB), trung vi, so mau, RAM he thong con trong thap nhat.
   RSS tren Windows = working set; tong RSS cua nhieu tien trinh co the dem trung trang chia se, nen co the cao hon
   bo nho thuc dung.
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def moc_gio(ra):
    t0 = time.time()
    fd = sys.stdin.fileno()
    buf = b""
    so_loi = 0
    with open(ra, "w", encoding="utf-8", errors="backslashreplace", newline="\n") as f:
        def ghi(dong):
            nonlocal so_loi
            try:
                s = dong.decode("utf-8", "replace")
                if s.strip():
                    f.write(f"[+{time.time() - t0:8.1f}s] {s}\n")
                    f.flush()
            except Exception:  # noqa: BLE001 -- khong de bo do lam dut pipe cua ben ghi
                so_loi += 1
        while True:
            try:
                khoi = os.read(fd, 65536)
            except OSError:
                break
            if not khoi:
                break
            phan = re.split(rb"[\r\n]", buf + khoi)
            buf = phan.pop()
            for p in phan:
                ghi(p)
        ghi(buf)
        if so_loi:
            try:
                f.write(f"[moc-gio] {so_loi} dong khong ghi duoc (da bo qua, van doc het stdin)\n")
            except Exception:  # noqa: BLE001
                pass
    return so_loi


def do_ram(mau, ra, chu_ky, tho=None):
    import psutil
    t0 = time.time()
    p = None
    while p is None and time.time() - t0 < 600:
        for x in psutil.process_iter(["pid", "cmdline"]):
            try:
                if x.info["cmdline"] and any(mau in s for s in x.info["cmdline"]) and x.pid != psutil.Process().pid:
                    p = x
                    break
            except (psutil.Error, TypeError):
                pass
        if p is None:
            time.sleep(0.5)
    if p is None:
        Path(ra).write_text(json.dumps({"loi": f"khong thay tien trinh chua {mau!r} trong 600 s"}) + "\n", encoding="utf-8")
        return 1
    mau_rss, mau_con, ram_trong, mau_tho = [], [], [], []
    t_bat_dau = time.time()
    while True:
        try:
            if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
                break
            ds = [p] + p.children(recursive=True)
            tong = 0
            for q in ds:
                try:
                    tong += q.memory_info().rss
                except psutil.Error:
                    pass
            mau_rss.append(tong / 2 ** 20)
            mau_con.append(len(ds) - 1)
            ram_trong.append(psutil.virtual_memory().available / 2 ** 20)
            try:
                rss_chinh = p.memory_info().rss / 2 ** 20
            except psutil.Error:
                rss_chinh = None
            mau_tho.append((round(time.time(), 3), round(rss_chinh) if rss_chinh is not None else None,
                            round(mau_rss[-1]), mau_con[-1], round(ram_trong[-1])))
        except psutil.NoSuchProcess:
            break
        time.sleep(chu_ky)
    s = sorted(mau_rss)
    kq = {"pid": p.pid, "mau_lenh": mau, "chu_ky_s": chu_ky, "so_mau": len(mau_rss), "thoi_gian_do_s": round(time.time() - t_bat_dau, 1),
          "rss_tien_trinh_va_con_max_mib": round(max(mau_rss)) if mau_rss else None,
          "rss_tien_trinh_va_con_trung_vi_mib": round(s[len(s) // 2]) if s else None,
          "so_tien_trinh_con_max": max(mau_con) if mau_con else None,
          "ram_he_thong_con_trong_min_mib": round(min(ram_trong)) if ram_trong else None,
          "ram_he_thong_con_trong_trung_vi_mib": round(sorted(ram_trong)[len(ram_trong) // 2]) if ram_trong else None,
          "pham_vi": "RSS (working set Windows) cua tien trinh train + moi tien trinh con; tong co the dem trung trang chia se "
                     "nen co the cao hon bo nho thuc dung. RAM he thong con trong do psutil.virtual_memory().available"}
    if tho:
        import csv
        with open(tho, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_unix", "rss_tien_trinh_chinh_mib", "rss_chinh_va_con_mib", "so_tien_trinh_con",
                        "ram_he_thong_con_trong_mib"])
            w.writerows(mau_tho)
        kq["mau_tho"] = str(tho)
    Path(ra).write_text(json.dumps(kq, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(kq, ensure_ascii=True))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("moc-gio")
    a1.add_argument("--ra", required=True)
    a2 = sub.add_parser("ram")
    a2.add_argument("--mau", required=True)
    a2.add_argument("--ra", required=True)
    a2.add_argument("--chu-ky", type=float, default=1.0)
    a2.add_argument("--tho", default=None, help="CSV mau tho co moc thoi gian (vong 37)")
    a = ap.parse_args()
    if a.cmd == "moc-gio":
        return 4 if moc_gio(a.ra) else 0           # 4 = co dong khong ghi duoc (stdin van duoc doc het)
    return do_ram(a.mau, a.ra, a.chu_ky, a.tho)


if __name__ == "__main__":
    sys.exit(main())
