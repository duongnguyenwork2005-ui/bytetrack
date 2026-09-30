# Kiem thu cuc bo giai_nen_mot20_train.py tren zip tong hop (khong cham data/ cua repo).
import hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

GOC_THU = Path(__file__).resolve().parent
REPO = Path(r"D:\LABOTARY\bytetrack")
SCRIPT = REPO / "handoff/logs/review31/giai_nen_mot20_train.py"
PY = sys.executable
kq = []


def kiem(ten, dk, chi_tiet=""):
    kq.append((ten, bool(dk)))
    print(("PASS" if dk else "FAIL"), ten, chi_tiet)


def moi_truong(ten):
    d = GOC_THU / ten
    shutil.rmtree(d, ignore_errors=True)
    for x in ("tai", "bgt", "goc", "ghi"):
        (d / x).mkdir(parents=True)
    return d


def tao_zip(d, so, members, nen=zipfile.ZIP_DEFLATED, trang_thai="DAT"):
    ten = f"images20-track-train-{so}.zip"
    z = d / "tai" / ten
    with zipfile.ZipFile(z, "w", nen) as zf:
        for m, noi in members.items():
            if m.endswith("/"):
                zf.writestr(zipfile.ZipInfo(m), b"")
            else:
                zf.writestr(m, noi)
    b = z.read_bytes()
    (d / "bgt" / f"{ten}.json").write_text(json.dumps({"ten": ten, "trang_thai": trang_thai, "bytes": len(b),
                                                       "sha256": hashlib.sha256(b).hexdigest()}), encoding="utf-8")
    return z


def chay(d, *them):
    r = subprocess.run([PY, str(SCRIPT), "--thu-muc-tai", str(d / "tai"), "--thu-muc-ban-ghi-tai", str(d / "bgt"),
                        "--goc", str(d / "goc"), "--thu-muc-ban-ghi", str(d / "ghi"), "--du-phong-gb", "0", *them],
                       capture_output=True, text=True, encoding="utf-8")
    return r.returncode, r.stdout, r.stderr


def anh(v, k):
    return f"bdd100k/images/track/train/{v}/{v}-{k:07d}.jpg"


def ds_tep(goc):
    return sorted(p.relative_to(goc).as_posix() for p in goc.rglob("*") if p.is_file())


# 1-3: binh thuong, chay lai, xung dot
d = moi_truong("binh_thuong")
m1 = {"bdd100k/": b"", "bdd100k/images/track/train/va/": b"", **{anh("va", k): os.urandom(3000) for k in (1, 2, 3)}}
m2 = {**{anh("vb", k): os.urandom(3000) for k in (1, 2)}, anh("va", 3): m1[anh("va", 3)]}  # trung duong dan + trung noi dung
tao_zip(d, 1, m1)
tao_zip(d, 2, m2, nen=zipfile.ZIP_STORED)
rc, o, e = chay(d, "--chi-kiem")
kiem("0 --chi-kiem khong ghi gi", rc == 0 and ds_tep(d / "goc") == [], f"rc={rc} {e[-300:]}")
rc, o, e = chay(d)
tep = ds_tep(d / "goc")
kiem("1 giai nen dung vi tri", rc == 0 and tep == sorted({anh("va", 1), anh("va", 2), anh("va", 3), anh("vb", 1), anh("vb", 2)}),
     f"rc={rc} {tep} {e[-300:]}")
kiem("1 noi dung trung byte", all((d / "goc" / k).read_bytes() == v for k, v in {**m1, **m2}.items() if not k.endswith("/")))
b2 = json.loads((d / "ghi/images20-track-train-2.zip.json").read_text(encoding="utf-8"))
kiem("1 member trung giua archive (cung noi dung) -> bo qua", b2["so_tep_trung_bo_qua"] == 1 and b2["so_tep_ghi_moi"] == 2, str(b2))
kiem("1 khong con file tam, khong long bdd100k/bdd100k", not list((d / "goc").rglob("*.tmp_giai_nen"))
     and not (d / "goc/bdd100k/bdd100k").exists())
rc, o, e = chay(d)
kiem("2a chay lai -> bo qua archive da hoan tat", rc == 0 and o.count('"bo_qua_archive"') == 2, o[-300:])
rc, o, e = chay(d, "--lam-lai")
b1 = json.loads((d / "ghi/images20-track-train-1.zip.json").read_text(encoding="utf-8"))
kiem("2b --lam-lai -> so tung byte, tat ca trung, 0 ghi moi", rc == 0 and b1["so_tep_ghi_moi"] == 0
     and b1["so_tep_trung_bo_qua"] == 3, str(b1))
tao_zip(d, 3, {anh("vc", 1): b"c" * 50})
rc, o, e = chay(d)
kiem("2c them archive moi -> chi giai nen archive moi", rc == 0 and o.count('"bo_qua_archive"') == 2
     and (d / "goc" / anh("vc", 1)).read_bytes() == b"c" * 50, o[-300:])
sua = d / "goc" / anh("va", 2)
sua.write_bytes(b"KHAC" + m1[anh("va", 2)][4:])  # cung kich thuoc, khac noi dung
rc, o, e = chay(d, "--lam-lai")
b1 = json.loads((d / "ghi/images20-track-train-1.zip.json").read_text(encoding="utf-8"))
kiem("3 xung dot -> exit 1, khong ghi de", rc == 1 and b1["so_xung_dot"] == 1 and sua.read_bytes()[:4] == b"KHAC", str(b1))

# 4-8: kiem truoc -> exit 2, khong ghi gi
for ten, members, mong in [
    ("long", {"bdd100k/bdd100k/images/track/train/va/va-0000001.jpg": b"x"}, "khong bat dau"),
    ("slip", {"bdd100k/images/track/train/../../../x.jpg": b"x"}, ".."),
    ("sau", {"bdd100k/images/track/train/va/con/va-0000001.jpg": b"x"}, "do sau"),
    ("val", {"bdd100k/images/track/val/va/va-0000001.jpg": b"x"}, "khong bat dau"),
]:
    d = moi_truong("sai_" + ten)
    tao_zip(d, 1, {anh("ok", 1): b"ok", **members})
    rc, o, e = chay(d)
    kiem(f"4 {ten} -> exit 2, khong ghi gi", rc == 2 and ds_tep(d / "goc") == [] and mong in o, f"rc={rc} {o[-250:]}")
d = moi_truong("xung_dot_archive")
tao_zip(d, 1, {anh("va", 1): b"a" * 10})
tao_zip(d, 2, {anh("va", 1): b"b" * 10})
rc, o, e = chay(d)
kiem("5 cung duong dan khac noi dung giua archive -> exit 2", rc == 2 and ds_tep(d / "goc") == [] and "khac CRC" in o, o[-250:])
d = moi_truong("hoa_thuong")
tao_zip(d, 1, {anh("va", 1): b"a", anh("va", 1).replace("/va-", "/VA-"): b"b"})
rc, o, e = chay(d)
kiem("6 chi khac hoa/thuong -> exit 2", rc == 2 and ds_tep(d / "goc") == [] and "hoa/thuong" in o, o[-250:])
d = moi_truong("chua_dat")
tao_zip(d, 1, {anh("va", 1): b"a"}, trang_thai="LOI")
rc, o, e = chay(d)
kiem("7 ban ghi tai khong DAT -> khong dung, exit 2", rc == 2 and ds_tep(d / "goc") == [], o[-200:])
d = moi_truong("sai_kich_thuoc")
z = tao_zip(d, 1, {anh("va", 1): b"a"})
with open(z, "ab") as f:
    f.write(b"\0")
rc, o, e = chay(d)
kiem("8 file zip doi kich thuoc sau khi xac minh -> khong dung, exit 2", rc == 2 and ds_tep(d / "goc") == [], o[-200:])

# 9: CRC hong trong du lieu member -> exit 1, khong de file tam / file do dang
d = moi_truong("crc")
noi = os.urandom(20000)
z = tao_zip(d, 1, {anh("va", 1): b"tot" * 100, anh("va", 2): noi}, nen=zipfile.ZIP_STORED)
b = bytearray(z.read_bytes())
vt = bytes(b).find(noi[:64])
b[vt + 5000] ^= 0xFF
z.write_bytes(bytes(b))
(d / "bgt/images20-track-train-1.zip.json").write_text(json.dumps({"ten": z.name, "trang_thai": "DAT", "bytes": len(b),
                                                                   "sha256": hashlib.sha256(b).hexdigest()}), encoding="utf-8")
rc, o, e = chay(d)
kiem("9 CRC hong -> exit 1, khong file tam, khong file hong", rc == 1 and "doc member loi" in o
     and not list((d / "goc").rglob("*.tmp_giai_nen")) and not (d / "goc" / anh("va", 2)).exists(), f"rc={rc} {o[-300:]} {e[-200:]}")

print("TONG", sum(v for _, v in kq), "/", len(kq), "PASS")
sys.exit(0 if all(v for _, v in kq) else 1)
