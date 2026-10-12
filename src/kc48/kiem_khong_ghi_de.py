"""
kiem_khong_ghi_de.py -- R48D: kiem CHI DOC rang R48D khong ghi vao ket qua cu.

Quet mtime moi file trong cac vung "khong duoc doi" o R48D va liet ke file co mtime >= moc bat dau R48D
(lan chup giao thuc R48D, dong 1 cua results/phase2/kc48_mo_phong/giao_thuc_sha256_r48d.jsonl). mtime chi la bang chung
gian tiep (ghi file thi mtime doi; doi noi dung ma giu mtime thi khong bat duoc), khong thay cho hash truoc/sau.

Vung quet:
  - results/trackeval/DETRAC-all/*   tru p2-kc48_pf_dev-* (dau ra R48D)
  - results/phase2/*                 tru kc48_mo_phong/, kc48_pf_dev/ (dau ra R48D) va kc48_audit/overlay/
  - data/processed/, data/interim/   (R48D khong duoc ghi; test 11 khong chay o R48D)
  - handoff/logs/review40..review47/
Chi ghi results/phase2/kc48_pf_dev/kiem_khong_ghi_de.json. Exit 0 neu khong file nao vi pham, 1 neu co.

Lenh: python src/kc48/kiem_khong_ghi_de.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

GOC = Path(__file__).resolve().parents[2]
NHAT_KY_KHOA = GOC / "results" / "phase2" / "kc48_mo_phong" / "giao_thuc_sha256_r48d.jsonl"
OUT = GOC / "results" / "phase2" / "kc48_pf_dev" / "kiem_khong_ghi_de.json"


def _moc() -> tuple[float, str]:
    dong = NHAT_KY_KHOA.read_text(encoding="utf-8").splitlines()[0]
    d = json.loads(dong)
    for k in ("thoi_diem", "luc", "tao_luc", "thoi_gian"):
        if k in d:
            s = d[k]
            break
    else:
        raise SystemExit(f"khong thay khoa thoi gian trong dong 1 cua {NHAT_KY_KHOA}: {sorted(d)}")
    return datetime.fromisoformat(s).timestamp(), s


def _duyet(goc: Path, bo_qua) -> tuple[int, list[str]]:
    n, moi = 0, []
    if not goc.exists():
        return 0, []
    for thu_muc, con, tep in os.walk(goc):
        tm = Path(thu_muc)
        con[:] = [c for c in con if not bo_qua(tm / c)]
        for t in tep:
            p = tm / t
            if bo_qua(p):
                continue
            n += 1
            if p.stat().st_mtime >= MOC:
                moi.append(p.relative_to(GOC).as_posix())
    return n, sorted(moi)


MOC, MOC_STR = _moc()


def main() -> int:
    te = GOC / "results" / "trackeval" / "DETRAC-all"
    ph = GOC / "results" / "phase2"
    vung = {
        "results/trackeval/DETRAC-all (tru p2-kc48_pf_dev-*)":
            (te, lambda p: p.parent == te and p.name.startswith("p2-kc48_pf_dev-")),
        "results/phase2 (tru kc48_mo_phong, kc48_pf_dev, kc48_audit/overlay)":
            (ph, lambda p: (p.parent == ph and p.name in ("kc48_mo_phong", "kc48_pf_dev"))
             or p == ph / "kc48_audit" / "overlay"),
        "data/processed": (GOC / "data" / "processed", lambda p: False),
        "data/interim": (GOC / "data" / "interim", lambda p: False),
    }
    for i in range(40, 48):
        vung[f"handoff/logs/review{i}"] = (GOC / "handoff" / "logs" / f"review{i}", lambda p: False)
    kq, tong_vp = {}, 0
    for ten, (goc, bo) in vung.items():
        n, moi = _duyet(goc, bo)
        kq[ten] = {"so_file_quet": n, "so_file_mtime_sau_moc": len(moi), "file": moi[:200]}
        tong_vp += len(moi)
        print(f"[{ten}] quet {n} file, mtime >= moc: {len(moi)}")
        for m in moi[:20]:
            print(f"   {m}")
    raw = sorted(te.glob("p2-expB_detrac_dev_fix1-*/raw_fields.json"))
    kq["tham_khao_raw_fields_expB"] = {
        p.relative_to(GOC).as_posix(): datetime.fromtimestamp(p.stat().st_mtime).astimezone().isoformat(timespec="seconds")
        for p in raw}
    bao = {"moc_bat_dau_r48d": MOC_STR, "nguon_moc": NHAT_KY_KHOA.relative_to(GOC).as_posix() + " dong 1",
           "tao_luc": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
           "gioi_han": "bang chung mtime, khong phai hash truoc/sau", "vung": kq, "tong_vi_pham": tong_vp,
           "ket_luan": "khong file cu nao co mtime sau moc" if tong_vp == 0 else "CO file cu bi ghi sau moc"}
    OUT.write_text(json.dumps(bao, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"[ket luan] {bao['ket_luan']} -> {OUT.relative_to(GOC).as_posix()}")
    return 0 if tong_vp == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
