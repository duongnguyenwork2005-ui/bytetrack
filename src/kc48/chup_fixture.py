"""
chup_fixture.py -- R48C (quyet dinh nguoi dung 11/10/2026): chup TRUOC / SAU khi chay du src/test_pf_ctrv.py (gom
test_11_runner_seeds) de chung minh test 11 chi doi cac thu muc fixture duoc phep:

  data/interim/det_cache/fixture_pf_seed_test/
  data/processed/trackers/DETRAC-sample/p2-pftest-s0/
  data/processed/trackers/DETRAC-sample/p2-pftest-s1/

- Trong 3 thu muc fixture: danh sach file + sha256 + byte + mtime_ns.
- Phan con lai cua data/interim/ va data/processed/ (tru 3 thu muc tren): danh sach file + byte + mtime_ns (khong hash
  noi dung: cache/ket qua lon; doi noi dung ma giu nguyen ca byte lan mtime_ns la khong the voi ghi file thong thuong).
CHI DOC data/; chi ghi results/phase2/kc48_pf_code/fixture_{truoc,sau}.json.

Lenh:
  python src/kc48/chup_fixture.py --ghi truoc     (tu choi neu fixture_truoc.json da co -- giu bang chung)
  python src/kc48/chup_fixture.py --ghi sau       (ghi them muc "so_voi_truoc"; exit 1 neu co thay doi ngoai fixture)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase2" / "kc48_pf_code"
FIXTURE = ("data/interim/det_cache/fixture_pf_seed_test",
           "data/processed/trackers/DETRAC-sample/p2-pftest-s0",
           "data/processed/trackers/DETRAC-sample/p2-pftest-s1")
VUNG_NGOAI = ("data/interim", "data/processed")


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _trong_fixture(rel: str) -> bool:
    return any(rel == d or rel.startswith(d + "/") for d in FIXTURE)


def chup() -> dict:
    fx = {}
    for d in FIXTURE:
        goc = ROOT / d
        fx[d] = {"ton_tai": goc.is_dir(), "file": {}}
        if goc.is_dir():
            for p in sorted(goc.rglob("*")):
                if p.is_file():
                    st = p.stat()
                    fx[d]["file"][p.relative_to(goc).as_posix()] = {"sha256": _sha(p), "byte": st.st_size,
                                                                   "mtime_ns": st.st_mtime_ns}
    ngoai, n_bo_qua = {}, 0
    for v in VUNG_NGOAI:
        for p in (ROOT / v).rglob("*"):
            rel = p.relative_to(ROOT).as_posix()
            if _trong_fixture(rel):
                n_bo_qua += p.is_file()
                continue
            if p.is_file():
                st = p.stat()
                ngoai[rel] = [st.st_size, st.st_mtime_ns]
    return {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "fixture": fx,
            "ngoai_fixture": {"vung": list(VUNG_NGOAI), "n_file": len(ngoai), "file_byte_mtime_ns": ngoai}}


def so(t: dict, s: dict) -> dict:
    kq = {"fixture": {}}
    for d in FIXTURE:
        a, b = t["fixture"][d]["file"], s["fixture"][d]["file"]
        kq["fixture"][d] = {"ton_tai_truoc": t["fixture"][d]["ton_tai"], "ton_tai_sau": s["fixture"][d]["ton_tai"],
                            "giu_nguyen_sha": sorted(k for k in set(a) & set(b) if a[k]["sha256"] == b[k]["sha256"]),
                            "doi_sha": sorted(k for k in set(a) & set(b) if a[k]["sha256"] != b[k]["sha256"]),
                            "ghi_lai_mtime_doi": sorted(k for k in set(a) & set(b) if a[k]["mtime_ns"] != b[k]["mtime_ns"]),
                            "them": sorted(set(b) - set(a)), "mat": sorted(set(a) - set(b))}
    a, b = t["ngoai_fixture"]["file_byte_mtime_ns"], s["ngoai_fixture"]["file_byte_mtime_ns"]
    doi = sorted(k for k in set(a) & set(b) if a[k] != b[k])
    them, mat = sorted(set(b) - set(a)), sorted(set(a) - set(b))
    kq["ngoai_fixture"] = {"n_file_truoc": len(a), "n_file_sau": len(b), "doi_byte_hoac_mtime": doi, "them": them,
                           "mat": mat}
    kq["chi_fixture_thay_doi"] = not (doi or them or mat)
    return kq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghi", choices=("truoc", "sau"), required=True)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"fixture_{a.ghi}.json"
    if a.ghi == "truoc" and p.exists():
        print(f"tu choi: {p.relative_to(ROOT).as_posix()} da co (giu bang chung truoc khi chay)")
        return 2
    kq = chup()
    rc = 0
    if a.ghi == "sau":
        pt = OUT / "fixture_truoc.json"
        if not pt.exists():
            print("thieu fixture_truoc.json")
            return 2
        kq["so_voi_truoc"] = so(json.loads(pt.read_text(encoding="utf-8")), kq)
        rc = 0 if kq["so_voi_truoc"]["chi_fixture_thay_doi"] else 1
    p.write_text(json.dumps(kq, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for d in FIXTURE:
        print(f"[{a.ghi}] {d}: {len(kq['fixture'][d]['file'])} file")
    print(f"[{a.ghi}] ngoai fixture: {kq['ngoai_fixture']['n_file']} file -> {p.relative_to(ROOT).as_posix()}")
    if a.ghi == "sau":
        s = kq["so_voi_truoc"]
        for d, v in s["fixture"].items():
            print(f"  {d}: giu sha {len(v['giu_nguyen_sha'])}, doi sha {len(v['doi_sha'])}, ghi lai (mtime doi) "
                  f"{len(v['ghi_lai_mtime_doi'])}, them {len(v['them'])}, mat {len(v['mat'])}")
        n = s["ngoai_fixture"]
        print(f"  ngoai fixture: doi {len(n['doi_byte_hoac_mtime'])}, them {len(n['them'])}, mat {len(n['mat'])} "
              f"-> chi_fixture_thay_doi = {s['chi_fixture_thay_doi']}")
        for k in (n["doi_byte_hoac_mtime"] + n["them"] + n["mat"])[:20]:
            print(f"    {k}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
