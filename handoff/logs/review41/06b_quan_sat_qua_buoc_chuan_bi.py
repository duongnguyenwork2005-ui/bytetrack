#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 41 -- doi chung cua observe_csv_risks.py (Codex, review 421dd53): CUNG 3 tinh huong, CUNG du lieu, CUNG writer
that va CUNG dieu kien 'risk_observed', nhung ca theo_batch.csv cung di qua BUOC CHUAN BI GHI cua resume that
(sao_luu_truoc_tiep_tuc -> sua_duoi_csv) truoc khi goi het_epoch_train (script goc goi thang writer, bo qua buoc nay).
Hai ca results.csv giong het script goc (script goc da goi sao_luu_truoc_tiep_tuc). Khong thay the test tich hop (nhom C).

    python handoff/logs/review41/06b_quan_sat_qua_buoc_chuan_bi.py --out handoff/logs/review41/06b_quan_sat_qua_buoc_chuan_bi.json
"""
import argparse
import csv
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path

GOC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(GOC / "src"))
import tiep_tuc_huan_luyen as t  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    observations = []
    with tempfile.TemporaryDirectory(prefix="r41_csv_") as td:
        root = Path(td)
        for name, tail in [("results_nul", b"\x00" * 4), ("results_incomplete_epoch10", b"1")]:
            r = root / name
            r.mkdir()
            header = b"epoch,time,x\r\n"
            body = b"".join(f"{i},{i * 10},0.5\r\n".encode() for i in range(1, 10))
            original = header + body + tail
            p = r / "results.csv"
            p.write_bytes(original)
            (r / "checkpoint.pt").write_bytes(b"marker-only")
            t.ghi_nhat_ky(r, "bat_dau", lan=0)
            err = None
            try:
                t.sao_luu_truoc_tiep_tuc(r, r / "checkpoint.pt", 1, epoch_da_luu=9)
            except Exception as e:  # noqa: BLE001
                err = {"type": type(e).__name__, "message": str(e), "is_LoiTiepTuc": isinstance(e, t.LoiTiepTuc)}
            rows = list(csv.DictReader(io.StringIO(p.read_text())))
            observations.append({
                "case": name, "error": err, "trailing_bytes_hex": p.read_bytes()[-20:].hex(), "last_parsed_row": rows[-1],
                "backup_exact": (r / "khoi_phuc/truoc_tiep_tuc_1/results.csv").read_bytes() == original,
                "risk_observed": bool(err and err["type"] == "ValueError" and not err["is_LoiTiepTuc"])
                if name == "results_nul" else err is None and p.read_bytes().endswith(b"\r\n1") and rows[-1].get("time") is None})
        that = {"t": 1790000000.5, "ni": 0, "loss_tong": 1.5, "box_loss": 1.5, "cls_loss": 1.5, "dfl_loss": 1.5,
                "loss_huu_han": True, "lr_weight": 0.01, "lr_bn": 0.01, "lr_bias": 0.01, "momentum_weight": 0.9,
                "momentum_bn": 0.9, "momentum_bias": 0.9, "accumulate": 8, "ema_updates": 3, "batch_size": 8,
                "thu_tu_batch": "0123456789abcdef"}
        # (a) DUNG du lieu script goc: hang mau chi co lan/epoch/i (o bat buoc khac rong) -> KHONG hop le theo dinh dang
        #     that cua runner; sau khi them '0,1,' no thanh hang hong o GIUA -> buoc chuan bi phai TU CHOI (LoiTiepTuc),
        #     file giu nguyen, khong noi tiep. (b) cung tinh huong voi hang runner THUC TE (du o bat buoc) -> cat duoi.
        for case, hang_dau, hang_moi in (("batch_du_lieu_codex_qua_chuan_bi", {"lan": 0, "epoch": 1, "i": 0},
                                          {"lan": 1, "epoch": 2, "i": 0}),
                                         ("batch_hang_thuc_te_qua_chuan_bi", {"lan": 0, "epoch": 1, "i": 0, **that},
                                          {"lan": 1, "epoch": 2, "i": 0, **that})):
            r = root / case
            r.mkdir()
            buf = io.StringIO(newline="")
            writer = csv.DictWriter(buf, fieldnames=t.COT_BATCH)
            writer.writeheader()
            writer.writerow(hang_dau)
            p = r / t.TEN_BATCH_CSV
            original = buf.getvalue().encode() + b"0,1,"
            p.write_bytes(original)
            (r / "checkpoint.pt").write_bytes(b"marker-only")
            t.ghi_nhat_ky(r, "bat_dau", lan=0)
            err = None
            try:
                t.sao_luu_truoc_tiep_tuc(r, r / "checkpoint.pt", 1, epoch_da_luu=1)
            except Exception as e:  # noqa: BLE001
                err = {"type": type(e).__name__, "message": str(e)[:160], "is_LoiTiepTuc": isinstance(e, t.LoiTiepTuc)}
            if err is None:
                recorder = t.GhiLuotDayDu({}, r, {}, "tiep_tuc", 1, None, {})
                recorder._rows = [hang_moi]
                recorder.het_epoch_train(None)   # writer that cua runner
            rows = list(csv.DictReader(io.StringIO(p.read_text())))
            observations.append({"case": case, "expected_fields": len(t.COT_BATCH), "buoc_chuan_bi": err or "khong loi",
                                 "file_khong_doi": p.read_bytes() == original,
                                 "unexpected_extra_fields": rows[-1].get(None) if rows else None,
                                 "hang_cuoi_lan_epoch_i": [rows[-1].get(k) for k in ("lan", "epoch", "i")] if rows else None,
                                 "risk_observed": bool(rows) and None in rows[-1] and rows[-1]["lan"] == "0"})
    src = GOC / "src" / "tiep_tuc_huan_luyen.py"
    out = {"scope": __doc__, "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
           "observations": observations, "known_risks_reproduced": sum(bool(x["risk_observed"]) for x in observations)}
    a.out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
