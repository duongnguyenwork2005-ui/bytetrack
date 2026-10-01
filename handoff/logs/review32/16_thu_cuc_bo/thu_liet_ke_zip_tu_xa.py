# Kiem thu cuc bo liet_ke_zip_tu_xa.py qua may chu Range gia lap (khong cham mang ngoai, khong cham data/).
import json, os, shutil, socket, subprocess, sys, time, zipfile
from pathlib import Path

G = Path(__file__).resolve().parent
REPO = Path(r"D:\LABOTARY\bytetrack")
SCRIPT = REPO / "handoff/logs/review32/liet_ke_zip_tu_xa.py"
MAY_CHU = G / "may_chu_range.py"
PY = sys.executable
SRV, OUT = G / "srv", G / "out"
for d in (SRV, OUT):
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
kq = []


def kiem(ten, dk, ct=""):
    kq.append(bool(dk))
    print("PASS" if dk else "FAIL", ten, ct)


mem = {f"bdd100k/labels/box_track_20/train/v{k:02d}.json": json.dumps([{"name": f"v{k:02d}-0000001.jpg"}] * (k + 1)).encode()
       for k in range(5)}
mem.update({f"bdd100k/labels/box_track_20/val/w{k}.json": b"[]" for k in range(2)})
mem.update({f"bdd100k/images/track/val/w0/w0-{k:07d}.jpg": os.urandom(300000) for k in range(8)})  # khong duoc tai
with zipfile.ZipFile(SRV / "goi.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for k, v in mem.items():
        z.writestr(k, v)
(SRV / "goi.zip.range").write_text("1")
# ban hong CRC: lat 1 byte trong du lieu nen cua member train/v03
b = bytearray((SRV / "goi.zip").read_bytes())
with zipfile.ZipFile(SRV / "goi.zip") as z:
    i = z.getinfo("bdd100k/labels/box_track_20/train/v03.json")
    off = i.header_offset + 30 + len(i.filename.encode()) + len(i.extra)
b[off + 2] ^= 0xFF
(SRV / "hong.zip").write_bytes(bytes(b))
(SRV / "hong.zip.range").write_text("1")

s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
srv = subprocess.Popen([PY, str(MAY_CHU), str(port), str(SRV)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(1.5)
NG = f"http://127.0.0.1:{port}/"


def chay(ten_zip, *them):
    r = subprocess.run([PY, str(SCRIPT), NG + ten_zip, "--ra", str(OUT / "ra.json"), *them], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


try:
    rc, o, e = chay("goi.zip", "--ds-day-du", str(OUT / "ds.tsv"), "--tai-tien-to", "bdd100k/labels/box_track_20/",
                    "--dich", str(OUT / "d"))
    ra = json.loads((OUT / "ra.json").read_text(encoding="utf-8"))
    tai = sorted(p.relative_to(OUT / "d").as_posix() for p in (OUT / "d").rglob("*") if p.is_file())
    kiem("1 chi tai dung tien to (7 json, 0 anh)", rc == 0 and tai == sorted(k for k in mem if "box_track_20" in k), f"rc={rc} {tai} {e[-300:]}")
    kiem("1 noi dung trung byte", all((OUT / "d" / k).read_bytes() == v for k, v in mem.items() if "box_track_20" in k))
    kiem("1 TSV day du 15 dong + tieu de", len((OUT / "ds.tsv").read_text(encoding="utf-8").splitlines()) == 16)
    kiem("1 byte tai cho member << tong zip (khong tai anh)", ra["tai_member"]["byte_da_tai_cho_member"] < 2 * (1 << 20)
         and ra["tai_member"]["so_ghi_moi"] == 7, json.dumps(ra["tai_member"]))
    rc, o, e = chay("goi.zip", "--tai-tien-to", "bdd100k/labels/box_track_20/", "--dich", str(OUT / "d"))
    ra = json.loads((OUT / "ra.json").read_text(encoding="utf-8"))
    kiem("2 chay lai -> trung, 0 ghi moi", rc == 0 and ra["tai_member"]["so_trung_bo_qua"] == 7 and ra["tai_member"]["so_ghi_moi"] == 0)
    p = OUT / "d/bdd100k/labels/box_track_20/train/v01.json"
    p.write_bytes(b"KHAC" + p.read_bytes()[4:])
    rc, o, e = chay("goi.zip", "--tai-tien-to", "bdd100k/labels/box_track_20/", "--dich", str(OUT / "d"))
    ra = json.loads((OUT / "ra.json").read_text(encoding="utf-8"))
    kiem("3 file khac noi dung -> xung dot, khong ghi de, exit 1", rc == 1 and ra["tai_member"]["so_xung_dot"] == 1
         and p.read_bytes()[:4] == b"KHAC")
    rc, o, e = chay("hong.zip", "--tai-tien-to", "bdd100k/labels/box_track_20/", "--dich", str(OUT / "h"))
    kiem("4 CRC hong -> dung (exit != 0), khong de file v03 that", rc != 0 and "CRC" in (o + e)
         and not (OUT / "h/bdd100k/labels/box_track_20/train/v03.json").exists()
         and not list((OUT / "h").rglob("*.tmp_tai")), f"rc={rc} {e[-200:]}")
finally:
    srv.terminate()
print("TONG", sum(kq), "/", len(kq), "PASS")
sys.exit(0 if all(kq) else 1)
