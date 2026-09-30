# Kiem thu cuc bo tai_mot20_train.py voi may chu http.server tren 127.0.0.1 (khong cham mang ngoai, khong cham data/).
import hashlib, io, json, os, shutil, socket, subprocess, sys, time, zipfile
from pathlib import Path

GOC = Path(__file__).resolve().parent
REPO = Path(r"D:\LABOTARY\bytetrack")
SCRIPT = REPO / "handoff/logs/review31/tai_mot20_train.py"
PY = sys.executable
SRV = GOC / "srv"
OUT = GOC / "out"
for d in (SRV, OUT):
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)


def zip_mau(so_video=2, so_anh=3):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w", zipfile.ZIP_STORED) as z:
        for v in range(so_video):
            ten = f"vid{v:02d}-aaaaaaaa"
            for k in range(1, so_anh + 1):
                z.writestr(f"bdd100k/images/track/train/{ten}/{ten}-{k:07d}.jpg", os.urandom(5000))
    return b.getvalue()


def dat(ten, noi_dung, md5=None):
    (SRV / ten).write_bytes(noi_dung)
    (SRV / (ten + ".md5")).write_text(md5 or hashlib.md5(noi_dung).hexdigest())


dat("images20-track-train-1.zip", zip_mau())
dat("images20-track-train-2.zip", zip_mau(), md5="0" * 32)                    # md5 may chu sai
z3 = zip_mau(3, 50)
dat("images20-track-train-3.zip", z3)                                           # se co .part do dang
dat("images20-track-train-4.zip", b"<html><body>404 not found</body></html>" * 10)  # HTML gia zip
z5 = zip_mau(4, 60)
dat("images20-track-train-5.zip", z5)                       # co Range + ngat ket noi giua chung o luot dau
(SRV / "images20-track-train-5.zip.range").write_text("1")
(SRV / "images20-track-train-5.zip.cat").write_text(str(int(len(z5) * 0.4)))
z6 = zip_mau(4, 60)
dat("images20-track-train-6.zip", z6)                       # co Range, .part do dang co san
(SRV / "images20-track-train-6.zip.range").write_text("1")

port = 8765
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
srv = subprocess.Popen([PY, str(GOC / "may_chu.py"), str(port), str(SRV)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.5)
NG = f"http://127.0.0.1:{port}/"
listing = subprocess.run(["curl", "-sS", NG], capture_output=True).stdout
(GOC / "listing_luu.html").write_bytes(listing)


def chay(*args):
    r = subprocess.run([PY, str(SCRIPT), "--nguon", NG, "--dich", str(OUT / "dl"), "--thu-muc-ban-ghi",
                        str(OUT / "ghi"), "--listing-da-luu", str(GOC / "listing_luu.html"), *args],
                       capture_output=True, text=True, cwd=str(REPO))
    return r.returncode, r.stdout, r.stderr


def bg(ten):
    return json.loads((OUT / "ghi" / f"{ten}.json").read_text(encoding="utf-8"))


kq = []
def kiem(ten, dk, chi_tiet=""):
    kq.append((ten, bool(dk)))
    print(("PASS" if dk else "FAIL"), ten, chi_tiet)

try:
    # 1. tai moi phan 1 -> DAT, doi ten, ban ghi day du
    rc, o, e = chay("--phan", "1")
    b = bg("images20-track-train-1.zip")
    kiem("1 tai moi DAT", rc == 0 and b["trang_thai"] == "DAT", f"rc={rc} {b.get('ly_do')} {e[-300:]}")
    kiem("1 file cuoi co, .part khong", (OUT / "dl/images20-track-train-1.zip").exists()
         and not (OUT / "dl/images20-track-train-1.zip.part").exists())
    noi = (SRV / "images20-track-train-1.zip").read_bytes()
    kiem("1 sha256 + md5 + bytes dung", b.get("sha256") == hashlib.sha256(noi).hexdigest()
         and b.get("md5") == hashlib.md5(noi).hexdigest() and b.get("bytes") == len(noi))
    z = b.get("zip", {}); kiem("1 cau truc zip", z.get("so_video") == 2 and z.get("so_tep") == 6 and z.get("testzip") == "OK"
         and z.get("long_bdd100k_bdd100k") is False and z.get("so_duong_dan_khong_an_toan") == 0, json.dumps(z)[:300])
    # 2. chay lai -> bo qua
    rc, o, e = chay("--phan", "1")
    kiem("2 chay lai bo qua", rc == 0 and '"su_kien": "bo_qua"' in o, o[-300:])
    # 3. md5 may chu sai -> LOI, .part con
    rc, o, e = chay("--phan", "2")
    b = bg("images20-track-train-2.zip")
    kiem("3 md5 sai -> LOI", rc == 1 and b["trang_thai"] == "LOI" and "md5" in b["ly_do"], b.get("ly_do"))
    kiem("3 khong doi ten khi LOI", (OUT / "dl/images20-track-train-2.zip.part").exists()
         and not (OUT / "dl/images20-track-train-2.zip").exists())
    # 4. .part do dang + may chu khong ho tro Range -> LOI range, .part giu nguyen
    (OUT / "dl/images20-track-train-3.zip.part").write_bytes(z3[:1000])
    rc, o, e = chay("--phan", "3")
    b = bg("images20-track-train-3.zip")
    kiem("4 khong range -> LOI", rc == 1 and b["trang_thai"] == "LOI", b.get("ly_do"))
    kiem("4 .part khong bi ghi them", (OUT / "dl/images20-track-train-3.zip.part").stat().st_size == 1000)
    # 5. HTML gia .zip -> LOI magic
    rc, o, e = chay("--phan", "4")
    b = bg("images20-track-train-4.zip")
    kiem("5 HTML -> LOI magic", rc == 1 and b["trang_thai"] == "LOI" and "4 byte dau" in b["ly_do"], b.get("ly_do"))
    # 6. listing khac -> exit 2
    (GOC / "listing_khac.html").write_bytes(listing.replace(b"images20-track-train-4.zip", b"images20-track-train-9.zip"))
    r = subprocess.run([PY, str(SCRIPT), "--nguon", NG, "--dich", str(OUT / "dl"), "--thu-muc-ban-ghi", str(OUT / "ghi"),
                        "--listing-da-luu", str(GOC / "listing_khac.html"), "--phan", "1"], capture_output=True, text=True)
    kiem("6 listing khac -> exit 2", r.returncode == 2, r.stdout[-200:])
    # 7. so phan khong co trong listing -> exit 2
    rc, o, e = chay("--phan", "8")
    kiem("7 --phan ngoai listing -> exit 2", rc == 2, o[-200:])
    # 8. file cuoi co san khong ban ghi -> kiem khong tai lai
    shutil.copy(SRV / "images20-track-train-3.zip", OUT / "dl/images20-track-train-3.zip")
    rc, o, e = chay("--phan", "3")
    b = bg("images20-track-train-3.zip")
    kiem("8 file co san -> kiem DAT", rc == 0 and b["trang_thai"] == "DAT" and '"kiem_file_co_san"' in o, b.get("ly_do"))
    # 9. mat ket noi giua chung -> luot 2 tai tiep bang Range (206) -> DAT, sha dung
    t0 = time.time()
    rc, o, e = chay("--phan", "5")
    b = bg("images20-track-train-5.zip")
    luot = [json.loads(l) for l in (OUT / "ghi/images20-track-train-5.zip.su_kien.jsonl").read_text(encoding="utf-8").splitlines()
            if '"luot_tai"' in l]
    kiem("9 ngat giua chung -> tai tiep DAT", rc == 0 and b["trang_thai"] == "DAT"
         and b.get("sha256") == hashlib.sha256(z5).hexdigest(), f"{b.get('ly_do')} {time.time()-t0:.0f}s")
    kiem("9 luot 1 dut (exit!=0, 200), luot 2 tu byte cu (206)", len(luot) == 2 and luot[0]["curl_exit"] != 0
         and luot[0]["http"] == "200" and luot[1]["http"] == "206" and luot[1]["tu_byte"] == luot[0]["den_byte"] > 0,
         json.dumps([(l["curl_exit"], l["http"], l["tu_byte"], l["den_byte"]) for l in luot]))
    # 10. .part do dang co san (dung byte) -> 1 luot 206 -> DAT
    (OUT / "dl/images20-track-train-6.zip.part").write_bytes(z6[:int(len(z6) * 0.3)])
    rc, o, e = chay("--phan", "6")
    b = bg("images20-track-train-6.zip")
    luot = [json.loads(l) for l in (OUT / "ghi/images20-track-train-6.zip.su_kien.jsonl").read_text(encoding="utf-8").splitlines()
            if '"luot_tai"' in l]
    kiem("10 .part co san -> 206 -> DAT", rc == 0 and b["trang_thai"] == "DAT" and len(luot) == 1
         and luot[0]["http"] == "206" and b.get("sha256") == hashlib.sha256(z6).hexdigest(), b.get("ly_do"))
    kiem("10 khong con file khoa sau khi xong", not list((OUT / "dl").glob("*.khoa")))
    # 11. phan dang bi khoa (tien trinh khac) -> bo qua KHOA, khong cham .part, exit 0 khi khong co LOI
    (OUT / "dl/images20-track-train-2.zip.khoa").write_text('{"pid": 999999}')
    truoc = (OUT / "dl/images20-track-train-2.zip.part").stat().st_size
    rc, o, e = chay("--phan", "2", "1")
    kiem("11 khoa -> KHOA, phan khac van xu ly", rc == 0 and '"bo_qua_khoa"' in o
         and '"images20-track-train-2.zip": "KHOA"' in o and '"images20-track-train-1.zip": "DAT"' in o, o[-300:])
    kiem("11 .part cua phan bi khoa khong doi, khoa cua nguoi khac giu nguyen",
         (OUT / "dl/images20-track-train-2.zip.part").stat().st_size == truoc
         and (OUT / "dl/images20-track-train-2.zip.khoa").exists())
finally:
    srv.terminate()
print("TONG", sum(v for _, v in kq), "/", len(kq), "PASS")
sys.exit(0 if all(v for _, v in kq) else 1)
