# -*- coding: utf-8 -*-
"""Vong 44 (R43-B1): doi chieu sha256 cua file support (CHI DOC).

    python handoff/logs/review44/doi_chieu_hash_support.py --base 4d54744 --target HEAD --out <json> -- <duong dan...>

Moi file: sha256 blob tai base va target (packager `--support` gui DUNG blob target, chi nhan file base == target);
sha256 file tren dia (working tree Windows, `core.autocrlf=true` + `* text=auto` -> co the la CRLF); so dong CRLF tren
dia; sha256 sau khi doi CRLF -> LF tren ban sao trong bo nho (PHAI == blob neu khac biet chi la EOL luc checkout); commit
cuoi cung cham file. So voi bang hash cua review a03f6e1 (cac hash do lay tu log chay vong 43 = file TREN DIA)."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BANG_REVIEW = {   # CODEX_REVIEW_a03f6e1.md muc 2 ("SHA256 duoc ghi trong ho so chay")
    "src/eval_detector_5lop.py": "30a1fd5a63641f1cebc22e4e7ef7e4600a48a45ed6f7d533bf319af0fa156f7f",
    "src/bdd_yolo_export.py": "1ca964b0fc81701ac746335fa20ac6e5231322a4cdd11ad47753a2b25fc31493",
    "src/detector_classes.py": "7bba3439987377cb2f161eb31e2cf0afafae3e814f165279087b7a3780699c3e",
    "src/mot_datasets/bdd100k_coco.py": "5028ffb98ddff6ba423bef14ff9331fd02f8e40b1055438c88224182914f8363",
}


def git(*a: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, check=True).stdout


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True)
    ap.add_argument("--target", default="HEAD")
    ap.add_argument("--out", required=True)
    ap.add_argument("duong_dan", nargs="+")
    a = ap.parse_args()
    base, target = git("rev-parse", a.base).decode().strip(), git("rev-parse", a.target).decode().strip()
    ds, loi = [], 0
    for p in a.duong_dan:
        b_base, b_tgt = git("show", f"{base}:{p}"), git("show", f"{target}:{p}")
        dia = (REPO / p).read_bytes()
        lf = dia.replace(b"\r\n", b"\n")
        eol = git("ls-files", "--eol", "--", p).decode().split()
        cuoi = git("log", "-1", "--format=%h %ad %s", "--date=short", target, "--", p).decode().strip()
        r = {"path": p, "bytes_blob": len(b_tgt), "sha256_blob_base": sha(b_base), "sha256_blob_target": sha(b_tgt),
             "base_bang_target": sha(b_base) == sha(b_tgt), "sha256_dia": sha(dia), "dong_crlf_tren_dia": dia.count(b"\r\n"),
             "sha256_dia_doi_crlf_ve_lf": sha(lf), "dia_sau_doi_eol_bang_blob": sha(lf) == sha(b_tgt),
             "git_eol": " ".join(eol[:3]), "commit_cuoi_cham_file": cuoi[:110]}
        if p in BANG_REVIEW:
            h = BANG_REVIEW[p]
            r["sha256_bang_review"] = h
            r["bang_review_khop"] = ("blob" if h == r["sha256_blob_target"] else
                                     "file tren dia (CRLF)" if h == r["sha256_dia"] else "KHONG KHOP")
        r["dat"] = r["base_bang_target"] and r["dia_sau_doi_eol_bang_blob"] and r.get("bang_review_khop") != "KHONG KHOP"
        loi += not r["dat"]
        ds.append(r)
        print(f"[{'DAT' if r['dat'] else 'KHONG DAT'}] {p}: blob {r['sha256_blob_target'][:12]} (base==target "
              f"{r['base_bang_target']}) | dia {r['sha256_dia'][:12]} CRLF {r['dong_crlf_tren_dia']} | dia->LF == blob "
              f"{r['dia_sau_doi_eol_bang_blob']} | {r['git_eol']}"
              + (f" | bang review {r['sha256_bang_review'][:12]} = {r['bang_review_khop']}" if p in BANG_REVIEW else "")
              + f" | {r['commit_cuoi_cham_file'][:60]}")
    Path(a.out).write_text(json.dumps({"base": base, "target": target, "file": ds, "khong_dat": loi},
                                      ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"KET LUAN: {len(ds) - loi}/{len(ds)} DAT")
    return 0 if loi == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
