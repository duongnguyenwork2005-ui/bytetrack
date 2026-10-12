"""
ghi_sha256.py -- R48C: chup sha256 + kich thuoc + thoi diem cua cac file (vd giao thuc truoc khi chay so sanh).

Chi doc cac file dau vao; ghi mot JSON vao results/phase2/kc48_pf_code/<ten>.json. Khong ghi de ban chup cu:
neu file JSON da ton tai thi them muc moi vao danh sach `lan_chup`.

    python src/kc48/ghi_sha256.py --ten giao_thuc_sha256 handoff/logs/review48/04_giao_thuc_kiem_chung_pf.md

--tien-to SHA256 BYTE: kiem them sha256 cua BYTE byte dau moi file phai bang SHA256 (vd phan da khoa cua giao thuc
truoc khi chi them phu luc o cuoi); ghi ket qua vao muc chup, exit 1 neu lech.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "phase2" / "kc48_pf_code"


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ten", required=True, help="ten file JSON dau ra (khong duoi)")
    ap.add_argument("--ghi-chu", default="")
    ap.add_argument("--tien-to", nargs=2, metavar=("SHA256", "BYTE"), default=None)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args(argv)
    muc = {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "ghi_chu": a.ghi_chu, "files": {}}
    rc = 0
    for f in a.files:
        p = (ROOT / f).resolve()
        if not p.is_file():
            print(f"THIEU: {f}")
            return 2
        b = p.read_bytes()
        muc["files"][Path(f).as_posix()] = {"sha256": hashlib.sha256(b).hexdigest(), "byte": len(b),
                                            "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(p.stat().st_mtime))}
        print(f"{hashlib.sha256(b).hexdigest()}  {len(b):>8} B  {Path(f).as_posix()}")
        if a.tien_to:
            n = int(a.tien_to[1])
            h = hashlib.sha256(b[:n]).hexdigest()
            khop = h == a.tien_to[0] and len(b) >= n
            muc["files"][Path(f).as_posix()]["tien_to"] = {"byte": n, "sha256": h, "mong": a.tien_to[0], "khop": khop}
            print(f"  tien to {n} B: {h} -> khop = {khop}")
            rc = rc or (0 if khop else 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{a.ten}.json"
    d = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {"lan_chup": []}
    d["lan_chup"].append(muc)
    out.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {out.relative_to(ROOT).as_posix()} (lan chup thu {len(d['lan_chup'])})")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
