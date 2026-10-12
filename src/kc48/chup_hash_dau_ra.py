"""
chup_hash_dau_ra.py -- R48B, sua loi hau audit F1 (review_003): chup sha256 moi file dau ra cua audit trong
results/phase2/kc48_audit/ (gom _dem/ va overlay/, tru chinh cac file chup) TRUOC va SAU khi chay lai, roi so sanh de
chung minh chi cac dau ra quy dao thay doi. CHI DOC du lieu; chi ghi results/phase2/kc48_audit/_dem/hash_<ten>.json.

Voi file .json, ngoai sha256 byte con ghi sha256 "noi dung" da bo cac khoa thoi gian (tao_luc, giay, cong_bo_luc) o
cap dau, de tach "chi doi dau thoi gian" voi "doi noi dung".

Lenh:
  python src/kc48/chup_hash_dau_ra.py --ghi <ten>
  python src/kc48/chup_hash_dau_ra.py --so <ten_truoc> <ten_sau>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
OUT = GOC / "results" / "phase2" / "kc48_audit"
DEM = OUT / "_dem"
KHOA_THOI_GIAN = ("tao_luc", "giay", "cong_bo_luc")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def chup() -> dict:
    kq = {}
    for p in sorted(OUT.rglob("*")):
        if not p.is_file() or (p.parent == DEM and p.name.startswith("hash_")):
            continue
        b = p.read_bytes()
        muc = {"sha256": _sha(b), "byte": len(b)}
        if p.suffix == ".json":
            try:
                d = json.loads(b.decode("utf-8"))
                if isinstance(d, dict):
                    d = {k: v for k, v in d.items() if k not in KHOA_THOI_GIAN}
                muc["sha256_noi_dung"] = _sha(json.dumps(d, ensure_ascii=False, sort_keys=True).encode("utf-8"))
            except ValueError:
                muc["sha256_noi_dung"] = "khong_doc_duoc_json"
        kq[p.relative_to(OUT).as_posix()] = muc
    return kq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ghi", metavar="TEN")
    g.add_argument("--so", nargs=2, metavar=("TRUOC", "SAU"))
    a = ap.parse_args(argv)
    if a.ghi:
        if not a.ghi.replace("_", "").isalnum():
            raise SystemExit("ten chi gom chu, so, '_'")
        p = (DEM / f"hash_{a.ghi}.json").resolve()
        if OUT.resolve() not in p.parents:
            raise SystemExit(f"tu choi ghi ngoai {OUT}")
        kq = chup()
        p.write_text(json.dumps(kq, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"[chup] {len(kq)} file -> {p.relative_to(GOC).as_posix()}")
        return 0
    t, s = (json.loads((DEM / f"hash_{x}.json").read_text(encoding="utf-8")) for x in a.so)
    giu, chi_tg, doi = [], [], []
    for k in sorted(set(t) & set(s)):
        if t[k]["sha256"] == s[k]["sha256"]:
            giu.append(k)
        elif t[k].get("sha256_noi_dung") and t[k].get("sha256_noi_dung") == s[k].get("sha256_noi_dung"):
            chi_tg.append(k)
        else:
            doi.append(k)
    print(f"[so] {a.so[0]} -> {a.so[1]}: giu nguyen byte {len(giu)}, chi doi khoa thoi gian {len(chi_tg)}, "
          f"doi noi dung {len(doi)}, them {len(set(s) - set(t))}, mat {len(set(t) - set(s))}")
    for ten, ds in (("chi_doi_khoa_thoi_gian", chi_tg), ("doi_noi_dung", doi), ("them", sorted(set(s) - set(t))),
                    ("mat", sorted(set(t) - set(s)))):
        for k in ds:
            print(f"  {ten}: {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
