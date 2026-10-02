#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 34 -- kiem thu nhap_nhan_train.py tren fixture tam (khong cham data/ that).

Ca: nhap sach; chay lai (da co, khong ghi lai); dich co file khac hash -> dung, khong ghi de; dich co file la -> dung;
nguon lech hash -> dung truoc khi ghi; nguon thieu file -> dung; COCO lech gia tri nghiem thu -> dung; --chi-kiem khong ghi.
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "nhap_nhan_train.py"
TIEN_TO = "bdd100k/labels/box_track_20/train/"
kq = []


def kiem(ten, dk, ct=""):
    kq.append(bool(dk))
    print("PASS" if dk else "FAIL", ten, ct if not dk else "")


def dung(goc, n=3):
    noi = {f"v{k}.json": json.dumps([{"name": f"v{k}-0000001.jpg", "labels": []}] * (k + 1)).encode() for k in range(n)}
    src = goc / "cl" / TIEN_TO
    src.mkdir(parents=True)
    for t, b in noi.items():
        (src / t).write_bytes(b)
    (goc / "cl" / "bdd100k/labels/box_track_20/val").mkdir(parents=True)
    ((goc / "cl" / "bdd100k/labels/box_track_20/val") / "w.json").write_bytes(b"[]")  # khong duoc nhap
    dong = ["duong_dan\tbyte\tcrc32_muc_luc_zip\tcrc32_tinh_lai\tsha256\tmd5"]
    for t, b in noi.items():
        dong.append(f"{TIEN_TO}{t}\t{len(b)}\tx\tx\t{hashlib.sha256(b).hexdigest()}\tx")
    dong.append("bdd100k/labels/box_track_20/val/w.json\t2\tx\tx\tx\tx")
    (goc / "manifest.tsv").write_text("\n".join(dong) + "\n", encoding="utf-8")
    (goc / "coco.json").write_bytes(b"{}")
    return noi


def chay(goc, *them, coco_sha=None):
    r = subprocess.run([sys.executable, str(SCRIPT), "--goc-cach-ly", str(goc / "cl"), "--dich", str(goc / "dich"),
                        "--manifest", str(goc / "manifest.tsv"), "--coco-val", str(goc / "coco.json"),
                        "--coco-val-sha256", coco_sha or hashlib.sha256(b"{}").hexdigest(), "--so-file", "3",
                        "--ra", str(goc / "ra.json"), "--ra-tsv", str(goc / "ra.tsv"), *them], capture_output=True, text=True)
    return r.returncode, json.loads((goc / "ra.json").read_text(encoding="utf-8"))


with tempfile.TemporaryDirectory() as td:
    g = Path(td); noi = dung(g)
    rc, k = chay(g, "--chi-kiem")
    kiem("1 --chi-kiem: exit 0, khong ghi gi", rc == 0 and not (g / "dich").exists())
    rc, k = chay(g)
    tep = sorted(p.name for p in (g / "dich").iterdir())
    kiem("2 nhap sach: exit 0, dung 3 file, trung byte, khong nhap val", rc == 0 and tep == sorted(noi)
         and all((g / "dich" / t).read_bytes() == b for t, b in noi.items()) and k["nhap"]["so_ghi_moi"] == 3, json.dumps(k)[:300])
    kiem("2 nguon cach ly giu nguyen", all((g / "cl" / TIEN_TO / t).read_bytes() == b for t, b in noi.items()))
    rc, k = chay(g)
    kiem("3 chay lai: da co 3, ghi moi 0", rc == 0 and k["nhap"] == {"so_ghi_moi": 0, "so_da_co_giu_nguyen": 3}, json.dumps(k)[:300])
    (g / "dich" / "v1.json").write_bytes(b"KHAC")
    rc, k = chay(g)
    kiem("4 dich khac hash -> exit 2, khong ghi de", rc == 2 and (g / "dich" / "v1.json").read_bytes() == b"KHAC"
         and k["kiem_dich_truoc"]["so_khac_hash"] == 1, json.dumps(k)[:300])

with tempfile.TemporaryDirectory() as td:
    g = Path(td); dung(g)
    (g / "dich").mkdir()
    (g / "dich" / "provenance.json").write_bytes(b"{}")
    rc, k = chay(g)
    kiem("5 dich co file la -> exit 2, khong xoa, khong nhap", rc == 2 and sorted(p.name for p in (g / "dich").iterdir()) == ["provenance.json"], json.dumps(k)[:300])

with tempfile.TemporaryDirectory() as td:
    g = Path(td); dung(g)
    (g / "cl" / TIEN_TO / "v2.json").write_bytes(b"[1]")
    rc, k = chay(g)
    kiem("6 nguon lech hash -> exit 2 truoc khi ghi", rc == 2 and not (g / "dich").exists() and k["kiem_truoc"]["so_lech_byte_hoac_sha256"] == 1, json.dumps(k)[:300])

with tempfile.TemporaryDirectory() as td:
    g = Path(td); dung(g)
    (g / "cl" / TIEN_TO / "v0.json").unlink()
    rc, k = chay(g)
    kiem("7 nguon thieu file -> exit 2", rc == 2 and not (g / "dich").exists() and k["kiem_truoc"]["so_thieu_o_nguon"] == 1, json.dumps(k)[:300])

with tempfile.TemporaryDirectory() as td:
    g = Path(td); dung(g)
    rc, k = chay(g, coco_sha="0" * 64)
    kiem("8 COCO val khac gia tri nghiem thu -> exit 2", rc == 2 and not (g / "dich").exists(), json.dumps(k)[:300])

print("TONG", sum(kq), "/", len(kq), "PASS")
sys.exit(0 if all(kq) else 1)
