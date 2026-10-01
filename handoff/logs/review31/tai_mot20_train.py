#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 31 -- tai va xac minh anh BDD100K MOT 2020 train (images20-track-train-*.zip) tu dl.yf.io.

Khong doan so phan hay ten file: danh sach phan lay tu listing THAT cua http://dl.yf.io/bdd100k/mot20/, doc lai luc chay
va phai trung ban da luu (00_dl_yf_io_mot20_listing.html). Dung HTTP vi HTTPS cua may chu nay loi chung chi.

Moi phan <ten>:
  1. HEAD -> HTTP 200, Content-Type application/zip, Accept-Ranges bytes, Content-Length.
  2. <ten>.md5 cua may chu -> HTTP 200, dung 32 ky tu hex; luu canh file tai.
  3. curl -f -C - vao <ten>.part (tai tiep sau mat mang), vong ngoai thu lai co gian cach. Moi luot: Content-Type tra ve
     phai la application/zip, neu khong thi cat .part ve kich thuoc truoc luot do (khong de trang HTML lan vao du lieu).
  4. Kiem: kich thuoc == Content-Length; 4 byte dau PK\\x03\\x04; md5 == .md5 may chu; sha256; zipfile.testzip()
     (CRC tung member); cau truc duong dan member (tien to, so video, duoi file, duong dan tuyet doi / '..').
  5. Dat -> doi ten .part thanh <ten>, ghi ban ghi JSON vao --thu-muc-ban-ghi.
File cuoi da co ban ghi DAT cung so byte -> bo qua (khong tai, khong bam lai tru khi --kiem-lai).
Khoa <ten>.khoa (tao doc quyen) cho phep chay song song nhieu tien trinh: phan dang bi khoa thi bo qua (KHOA), khong ghi
chung vao mot .part. Tien trinh chet de lai khoa -> xoa tay sau khi chac chan khong con tien trinh nao tai phan do.

Ma thoat: 0 moi phan duoc chon deu DAT; 1 co phan loi; 2 loi tien quyet (listing khac ban da luu, thieu dung luong...).
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

NGUON = "http://dl.yf.io/bdd100k/mot20/"
MAU_TRAIN = re.compile(r'href="(images20-track-train-(\d+)\.zip)"')
MAGIC_ZIP = b"PK\x03\x04"


def bay_gio():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def ghi_log(duong, **kv):
    kv = {"t": bay_gio(), **kv}
    dong = json.dumps(kv, ensure_ascii=False)
    with open(duong, "a", encoding="utf-8") as f:
        f.write(dong + "\n")
    print(dong, flush=True)


def head(curl_exe, url):
    r = subprocess.run([curl_exe, "-sS", "-I", "--max-time", "60", url], capture_output=True, timeout=120)
    dong = [d.strip() for d in r.stdout.decode("latin-1").replace("\r", "").split("\n") if d.strip()]
    if r.returncode != 0 or not dong:
        return {"curl_exit": r.returncode, "http": None, "headers": {},
                "loi": r.stderr.decode("utf-8", "replace").strip()}
    m = re.match(r"HTTP/\S+\s+(\d+)", dong[0])
    h = {}
    for d in dong[1:]:
        if ":" in d:
            k, v = d.split(":", 1)
            h[k.strip().lower()] = v.strip()
    return {"curl_exit": 0, "http": int(m.group(1)) if m else None, "headers": h}


def lay_nho(curl_exe, url):
    """GET noi dung nho (listing, .md5). Tra (http, bytes, curl_exit, stderr)."""
    r = subprocess.run([curl_exe, "-sS", "--max-time", "120", "-w", "\n%{http_code}", url],
                       capture_output=True, timeout=180)
    than, _, ma = r.stdout.rpartition(b"\n")
    try:
        http = int(ma.decode().strip())
    except ValueError:
        http = None
    return http, than, r.returncode, r.stderr.decode("utf-8", "replace").strip()


def phan_train(html_bytes):
    return sorted({m.group(1) for m in MAU_TRAIN.finditer(html_bytes.decode("utf-8", "replace"))},
                  key=lambda t: int(MAU_TRAIN.search(f'href="{t}"').group(2)))


def content_type_tu_header(hdr):
    """Content-Type cua phan hoi HTTP cuoi trong file header curl -D; '' neu khong co/khong doc duoc."""
    try:
        dong = hdr.read_bytes().decode("latin-1").replace("\r", "").split("\n")
    except OSError:
        return ""
    ct = ""
    for d in dong:
        if d.startswith("HTTP/"):
            ct = ""
        elif d.lower().startswith("content-type:"):
            ct = d.split(":", 1)[1].strip()
    return ct


def tai_part(curl_exe, url, part, tong, log, cap_lien_tiep=60, cap_tong=1000):
    """Tai/tai tiep vao part cho toi du `tong` byte. Tra (ok, ly_do)."""
    lien_tiep = 0
    luot = 0
    while True:
        co = part.stat().st_size if part.exists() else 0
        if co == tong:
            return True, "du Content-Length"
        if co > tong:
            return False, f".part lon hon Content-Length ({co} > {tong})"
        if lien_tiep >= cap_lien_tiep or luot >= cap_tong:
            return False, f"dung sau {luot} luot ({lien_tiep} luot lien tiep khong tien trien)"
        luot += 1
        t0 = time.time()
        hdr = part.with_name(part.name + ".hdr")
        hdr.unlink(missing_ok=True)
        r = subprocess.run([curl_exe, "-f", "-sS", "--connect-timeout", "30",
                            "--speed-limit", "20000", "--speed-time", "120",
                            "-C", "-", "-o", str(part), "-D", str(hdr),
                            "-w", "%{http_code}|%{content_type}|%{size_download}|%{speed_download}", url],
                           capture_output=True)
        w = r.stdout.decode("latin-1").strip().split("|")
        http, ctype = (w + ["", ""])[:2]
        ctype_hdr = content_type_tu_header(hdr)
        sau = part.stat().st_size if part.exists() else 0
        bg = dict(su_kien="luot_tai", url=url, luot=luot, tu_byte=co, den_byte=sau, curl_exit=r.returncode,
                  http=http, content_type=ctype, content_type_header=ctype_hdr,
                  toc_do_Bps=(w[3] if len(w) > 3 else ""),
                  giay=round(time.time() - t0, 1), stderr=r.stderr.decode("utf-8", "replace").strip()[-300:])
        # Chi cat khi BIET CHAC Content-Type khac zip. curl bi giet giua chung khong kip in -w (ctype rong) -- ban cu
        # coi rong la "khong phai zip" va cat mat du lieu tot (su co 03:46 vong 31). Khong ro -> giu byte, md5 cuoi kiem.
        ctype_hl = (ctype or ctype_hdr).split(";")[0].strip().lower()
        if sau > co and not ctype_hl:
            bg["content_type_khong_ro"] = True
        if sau > co and ctype_hl and ctype_hl != "application/zip":
            with open(part, "r+b") as f:
                f.truncate(co)
            bg["cat_ve_byte"] = co
            ghi_log(log, **bg)
            return False, f"luot {luot}: Content-Type {ctype_hl!r} khong phai application/zip -- da cat .part ve {co}"
        ghi_log(log, **bg)
        if r.returncode in (33, 36):
            return False, f"curl exit {r.returncode}: may chu khong cho tai tiep (range)"
        if r.returncode == 22 and http in ("401", "403", "404", "410", "416"):
            return False, f"HTTP {http}"
        lien_tiep = 0 if sau > co else lien_tiep + 1
        if r.returncode != 0 or sau <= co:
            time.sleep(min(120, 10 * max(1, lien_tiep)))


def bam(duong, khoi=8 << 20):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(duong, "rb") as f:
        while True:
            b = f.read(khoi)
            if not b:
                break
            md5.update(b)
            sha.update(b)
    return md5.hexdigest(), sha.hexdigest()


def cau_truc_zip(duong):
    with zipfile.ZipFile(duong) as z:
        infos = z.infolist()
        crc_loi = z.testzip()
    tep = [i for i in infos if not i.is_dir()]
    tien_to = collections.Counter("/".join(i.filename.split("/")[:4]) for i in tep)
    video = collections.Counter()
    sau_video = collections.Counter()
    for i in tep:
        p = i.filename.split("/")
        if len(p) >= 6 and p[:4] == ["bdd100k", "images", "track", "train"]:
            video[p[4]] += 1
            sau_video[len(p)] += 1
    khong_an_toan = [i.filename for i in infos
                     if i.filename.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", i.filename)
                     or ".." in i.filename.replace("\\", "/").split("/") or "\\" in i.filename]
    return {
        "testzip": "OK" if crc_loi is None else f"CRC loi o {crc_loi}",
        "so_member": len(infos), "so_tep": len(tep), "so_thu_muc": len(infos) - len(tep),
        "tien_to_4_cap": dict(tien_to.most_common(20)),
        "so_video": len(video), "anh_moi_video_min": min(video.values()) if video else 0,
        "anh_moi_video_max": max(video.values()) if video else 0,
        "do_sau_duong_dan_tep": dict(sau_video),
        "duoi_tep": dict(collections.Counter(Path(i.filename).suffix.lower() for i in tep)),
        "kieu_nen": dict(collections.Counter(str(i.compress_type) for i in tep)),
        "tong_giai_nen_byte": sum(i.file_size for i in tep),
        "long_bdd100k_bdd100k": any(i.filename.startswith("bdd100k/bdd100k/") for i in infos),
        "duong_dan_khong_an_toan": khong_an_toan[:20], "so_duong_dan_khong_an_toan": len(khong_an_toan),
        "ten_dai_nhat": max((len(i.filename) for i in infos), default=0),
        "vi_du_tep": [i.filename for i in tep[:3]],
    }


def xu_ly(ten, curl_exe, dich, ghi, kiem_lai, nguon=NGUON):
    url = nguon + ten
    log = ghi / f"{ten}.su_kien.jsonl"
    file_ban_ghi = ghi / f"{ten}.json"
    cuoi, part, md5f = dich / ten, dich / (ten + ".part"), dich / (ten + ".md5")
    bg = {"ten": ten, "url": url, "url_md5": url + ".md5", "t_bat_dau": bay_gio(), "trang_thai": "LOI"}

    def xong(trang_thai, ly_do=None):
        bg["trang_thai"], bg["t_xong"] = trang_thai, bay_gio()
        if ly_do:
            bg["ly_do"] = ly_do
        file_ban_ghi.write_text(json.dumps(bg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        ghi_log(log, su_kien="ket_qua", ten=ten, trang_thai=trang_thai, ly_do=ly_do)
        return trang_thai == "DAT"

    if cuoi.exists() and file_ban_ghi.exists() and not kiem_lai:
        cu = json.loads(file_ban_ghi.read_text(encoding="utf-8"))
        if cu.get("trang_thai") == "DAT" and cu.get("bytes") == cuoi.stat().st_size:
            ghi_log(log, su_kien="bo_qua", ten=ten, ly_do="da xac minh truoc", sha256=cu.get("sha256"))
            return True

    h = head(curl_exe, url)
    hh = h["headers"]
    bg["head"] = {"http": h["http"], "content_type": hh.get("content-type"), "content_length": hh.get("content-length"),
                  "accept_ranges": hh.get("accept-ranges"), "last_modified": hh.get("last-modified"),
                  "etag": hh.get("etag"), "server": hh.get("server"), "curl_exit": h["curl_exit"]}
    ghi_log(log, su_kien="head", **bg["head"])
    if h["http"] != 200 or (hh.get("content-type") or "").lower() != "application/zip" or "content-length" not in hh:
        return xong("LOI", f"HEAD khong hop le: {bg['head']}")
    tong = int(hh["content-length"])
    bg["content_length"] = tong

    http, than, ce, err = lay_nho(curl_exe, url + ".md5")
    m = re.fullmatch(rb"\s*([0-9a-f]{32})\s*(\S+)?\s*", than or b"")
    if http != 200 or not m:
        return xong("LOI", f".md5 may chu khong hop le: http={http} curl_exit={ce} than={than[:80]!r} {err}")
    md5_may_chu = m.group(1).decode()
    md5f.write_bytes(than)
    bg["md5_may_chu"] = md5_may_chu

    if not cuoi.exists():
        ok, ly_do = tai_part(curl_exe, url, part, tong, log)
        if not ok:
            return xong("LOI", f"tai: {ly_do}")
        part.with_name(part.name + ".hdr").unlink(missing_ok=True)
        nguon_kiem = part
    else:
        ghi_log(log, su_kien="kiem_file_co_san", ten=ten, bytes=cuoi.stat().st_size)
        nguon_kiem = cuoi

    kich_thuoc = nguon_kiem.stat().st_size
    bg["bytes"] = kich_thuoc
    if kich_thuoc != tong:
        return xong("LOI", f"kich thuoc {kich_thuoc} != Content-Length {tong}")
    with open(nguon_kiem, "rb") as f:
        dau = f.read(4)
    bg["magic"] = dau.hex()
    if dau != MAGIC_ZIP:
        return xong("LOI", f"4 byte dau {dau!r} khong phai PK\\x03\\x04 (co the la trang HTML)")
    ghi_log(log, su_kien="bam_bat_dau", ten=ten)
    md5, sha = bam(nguon_kiem)
    bg["md5"], bg["sha256"] = md5, sha
    if md5 != md5_may_chu:
        return xong("LOI", f"md5 {md5} != may chu {md5_may_chu}")
    ghi_log(log, su_kien="testzip_bat_dau", ten=ten, md5=md5, sha256=sha)
    try:
        bg["zip"] = cau_truc_zip(nguon_kiem)
    except (zipfile.BadZipFile, OSError) as e:
        return xong("LOI", f"zipfile: {e!r}")
    if bg["zip"]["testzip"] != "OK":
        return xong("LOI", bg["zip"]["testzip"])
    if nguon_kiem is part:
        part.replace(cuoi)
    bg["duong_dan"] = cuoi.as_posix()
    return xong("DAT")


def pid_con_song(pid):
    """True neu tien trinh `pid` con chay. Khong xac dinh duoc -> True (than trong: giu khoa).

    Windows: OpenProcess + GetExitCodeProcess (KHONG dung os.kill(pid, 0) -- tren Windows lenh do GIET tien trinh).
    """
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ctypes.get_last_error() != 87  # 87 = ERROR_INVALID_PARAMETER: khong co PID nay
    try:
        ma = wintypes.DWORD()
        if not k32.GetExitCodeProcess(h, ctypes.byref(ma)):
            return True
        return ma.value == 259  # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


def don_khoa_cu(dich, log):
    """Xoa .khoa cua tien trinh DA CHET (vd bi giet khi het han tac vu nen, dong cua so bang X). Khoa cua tien trinh
    con song, hoac khoa khong doc duoc, giu nguyen. Chay TRUOC khi khoi dong cac tien trinh tai, khong chay song song."""
    da_xoa = []
    for k in sorted(dich.glob("*.khoa")):
        try:
            noi = json.loads(k.read_text(encoding="utf-8"))
            pid = int(noi["pid"])
        except (OSError, ValueError, KeyError, TypeError) as e:
            ghi_log(log, su_kien="khoa_khong_doc_duoc", khoa=k.name, loi=repr(e))
            continue
        if pid == os.getpid() or pid_con_song(pid):
            ghi_log(log, su_kien="khoa_con_song", khoa=k.name, pid=pid)
            continue
        k.unlink(missing_ok=True)
        da_xoa.append(k.name)
        ghi_log(log, su_kien="xoa_khoa_cu", khoa=k.name, noi_dung=noi)
    return da_xoa


def xu_ly_co_khoa(ten, curl_exe, dich, ghi, kiem_lai, nguon, log_chung):
    khoa = dich / (ten + ".khoa")
    try:
        fd = os.open(khoa, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        ghi_log(log_chung, su_kien="bo_qua_khoa", ten=ten, khoa=khoa.read_text(encoding="utf-8", errors="replace"))
        return None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps({"pid": os.getpid(), "t": bay_gio()}))
    try:
        return xu_ly(ten, curl_exe, dich, ghi, kiem_lai, nguon)
    finally:
        khoa.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nguon", default=NGUON, help="thu muc may chu (chi doi khi kiem thu cuc bo)")
    ap.add_argument("--dich", default="data/raw/downloads/bdd100k_mot20")
    ap.add_argument("--thu-muc-ban-ghi", default="handoff/logs/review31/10_tai")
    ap.add_argument("--listing-da-luu", default="handoff/logs/review31/00_dl_yf_io_mot20_listing.html")
    ap.add_argument("--phan", nargs="*", type=int, help="chi xu ly cac so phan nay (phai co trong listing)")
    ap.add_argument("--curl", default=shutil.which("curl"))
    ap.add_argument("--kiem-lai", action="store_true", help="bam va testzip lai ca file da co ban ghi DAT")
    ap.add_argument("--du-phong-gb", type=float, default=2.0, help="dung luong trong phai con sau khi tai xong")
    ap.add_argument("--don-khoa-cu", action="store_true",
                    help="CHI xoa .khoa cua tien trinh da chet roi thoat (chay truoc khi khoi dong lai cac tien trinh tai)")
    a = ap.parse_args()

    dich, ghi = Path(a.dich), Path(a.thu_muc_ban_ghi)
    dich.mkdir(parents=True, exist_ok=True)
    ghi.mkdir(parents=True, exist_ok=True)
    if a.don_khoa_cu:
        log_don = ghi / f"don_khoa_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
        da_xoa = don_khoa_cu(dich, log_don)
        ghi_log(log_don, su_kien="ket_thuc_don_khoa", da_xoa=da_xoa)
        return 0
    log_chung = ghi / f"chay_{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{'_'.join(map(str, a.phan or ['tat_ca']))}.jsonl"
    if not a.curl:
        ghi_log(log_chung, su_kien="loi", ly_do="khong tim thay curl")
        return 2
    ban_curl = subprocess.run([a.curl, "--version"], capture_output=True).stdout.decode("latin-1").splitlines()[0]
    ghi_log(log_chung, su_kien="bat_dau", curl=a.curl, curl_version=ban_curl, nguon=a.nguon, dich=dich.as_posix(),
            python=sys.version.split()[0], phan=a.phan)

    http, than, ce, err = lay_nho(a.curl, a.nguon)
    da_luu = Path(a.listing_da_luu).read_bytes()
    ds_song, ds_luu = phan_train(than or b""), phan_train(da_luu)
    ghi_log(log_chung, su_kien="listing", http=http, curl_exit=ce, train_song=ds_song, train_da_luu=ds_luu)
    if http != 200 or not ds_song or ds_song != ds_luu:
        ghi_log(log_chung, su_kien="loi", ly_do="listing song khac ban da luu hoac khong doc duoc", stderr=err)
        return 2
    chon = ds_song if not a.phan else [t for t in ds_song if int(MAU_TRAIN.search(f'href="{t}"').group(2)) in a.phan]
    if a.phan and len(chon) != len(set(a.phan)):
        ghi_log(log_chung, su_kien="loi", ly_do=f"--phan {a.phan} co so khong nam trong listing {ds_song}")
        return 2

    con_lai = 0
    for t in chon:
        h = head(a.curl, a.nguon + t)
        cl = int(h["headers"].get("content-length", 0))
        da_co = (dich / t).stat().st_size if (dich / t).exists() else (
            (dich / (t + ".part")).stat().st_size if (dich / (t + ".part")).exists() else 0)
        con_lai += max(0, cl - da_co)
    trong = shutil.disk_usage(dich).free
    ghi_log(log_chung, su_kien="dung_luong", trong_byte=trong, can_tai_them_byte=con_lai,
            du_phong_byte=int(a.du_phong_gb * 1e9))
    if trong < con_lai + a.du_phong_gb * 1e9:
        ghi_log(log_chung, su_kien="loi", ly_do="khong du dung luong trong")
        return 2

    ket_qua = {}
    for t in chon:
        ket_qua[t] = xu_ly_co_khoa(t, a.curl, dich, ghi, a.kiem_lai, a.nguon, log_chung)
    nhan = {t: ("KHOA" if v is None else "DAT" if v else "LOI") for t, v in ket_qua.items()}
    ghi_log(log_chung, su_kien="ket_thuc", ket_qua=nhan)
    return 1 if "LOI" in nhan.values() else 0


if __name__ == "__main__":
    sys.exit(main())
