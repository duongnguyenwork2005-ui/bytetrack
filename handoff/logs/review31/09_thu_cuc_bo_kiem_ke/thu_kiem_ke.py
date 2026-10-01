# Kiem thu cuc bo kiem_ke_train.py (khong cham data/ cua repo).
import hashlib, json, shutil, subprocess, sys, zipfile
from pathlib import Path

G = Path(__file__).resolve().parent / "env"
REPO = Path(r"D:\LABOTARY\bytetrack")
SCRIPT = REPO / "handoff/logs/review31/kiem_ke_train.py"
PY = sys.executable
kq = []


def kiem(ten, dk, ct=""):
    kq.append(bool(dk))
    print("PASS" if dk else "FAIL", ten, ct)


def dung():
    shutil.rmtree(G, ignore_errors=True)
    for x in ("tai", "bgt", "goc/bdd100k/images/track/val/vv1", "goc/bdd100k/images/track/val/vv2"):
        (G / x).mkdir(parents=True)
    mem = {f"bdd100k/images/track/train/{v}/{v}-{k:07d}.jpg": bytes([k]) * (100 + k) for v in ("ta", "tb") for k in (1, 2, 3)}
    z = G / "tai/images20-track-train-1.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for m, b in mem.items():
            zf.writestr(m, b)
    b = z.read_bytes()
    (G / "bgt/images20-track-train-1.zip.json").write_text(json.dumps(
        {"ten": z.name, "trang_thai": "DAT", "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}), encoding="utf-8")
    for m, b in mem.items():
        p = G / "goc" / m
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    return mem


def chay():
    r = subprocess.run([PY, str(SCRIPT), "--goc", str(G / "goc"), "--thu-muc-tai", str(G / "tai"),
                        "--thu-muc-ban-ghi-tai", str(G / "bgt"), "--ra", str(G / "ra.json")],
                       capture_output=True, text=True, encoding="utf-8")
    return r.returncode, json.loads((G / "ra.json").read_text(encoding="utf-8")) if (G / "ra.json").exists() else {}, r.stderr


mem = dung()
rc, k, e = chay()
kiem("1 sach -> exit 0, dem dung", rc == 0 and k["anh_train"]["so_video"] == 2 and k["anh_train"]["so_anh"] == 6
     and k["so_archive_dat"] == 1 and k["so_voi_archive"]["thieu_tren_dia"] == 0 and k["trung_ten_video_train_val"] == []
     and k["so_video_val"] == 2 and k["nhan_train"]["ton_tai"] is False, f"rc={rc} {e[-300:]}")
kiem("1 tong byte", k["anh_train"]["tong_byte"] == sum(len(b) for b in mem.values()))
(G / "goc/bdd100k/images/track/train/ta/ta-0000002.jpg").unlink()
(G / "goc/bdd100k/images/track/train/tb/tb-0000009.jpg").write_bytes(b"x")
(G / "goc/bdd100k/images/track/train/tb/tb-0000003.jpg.tmp_giai_nen").write_bytes(b"x")
(G / "goc/bdd100k/images/track/train/tb/tb-0000001.jpg").write_bytes(b"ngan")
(G / "goc/bdd100k/images/track/train/vv1").mkdir()
(G / "goc/bdd100k/images/track/train/vv1/vv1-0000001.jpg").write_bytes(b"x")
rc, k, e = chay()
s = k.get("so_voi_archive", {})
kiem("2 thieu/thua/lech/tam/trung val -> exit 1", rc == 1 and s["thieu_tren_dia"] == 1 and s["thua_tren_dia"] == 2
     and s["lech_kich_thuoc"] == 1 and s["file_tam_con_sot"] == 1 and k["trung_ten_video_train_val"] == ["vv1"], json.dumps(s))
(G / "goc/bdd100k/labels/box_track_20/train").mkdir(parents=True)
(G / "goc/bdd100k/labels/box_track_20/train/ta.json").write_text("[]")
rc, k, e = chay()
kiem("3 nhan train dem json", k["nhan_train"] == {"duong_dan": (G / "goc/bdd100k/labels/box_track_20/train").as_posix(),
                                                  "ton_tai": True, "so_file_json": 1}, json.dumps(k["nhan_train"]))
print("TONG", sum(kq), "/", len(kq), "PASS")
sys.exit(0 if all(kq) else 1)
