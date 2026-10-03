#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 40 -- doi chung cua REPRO_REVIEW_1454b3d.py: CUNG 4 ca, CUNG hau to, CUNG hai lan append va CUNG tieu chi,
nhung chen BUOC CHUAN BI GHI cua resume that (sua_duoi_nhat_ky, goi trong sao_luu_truoc_tiep_tuc) truoc lan append dau.
Khong thay the test tich hop (nhom N cua src/test_tiep_tuc_huan_luyen.py chay qua train_detector.py that).

    python handoff/logs/review40/06b_repro_qua_buoc_chuan_bi.py --out handoff/logs/review40/06b_repro_qua_buoc_chuan_bi.json
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

GOC = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(GOC / "src"))
import tiep_tuc_huan_luyen as mod  # noqa: E402

CA = [("clean_control", b""), ("partial_tail_no_newline", b'{"loai":"epoch_da_luu","epoch":2'),
      ("partial_tail_with_newline", b'{"loai":"epoch_da_luu","epoch":2\n'),
      ("valid_record_no_newline", b'{"loai":"ket_thuc","trang_thai":"bi_ngat"}')]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    cases = []
    with tempfile.TemporaryDirectory(prefix="r40_journal_") as tmp:
        for name, suffix in CA:
            run = Path(tmp) / name
            run.mkdir()
            mod.ghi_nhat_ky(run, "bat_dau", che_do_luot="moi", lan=0, ten="fixture")
            mod.ghi_nhat_ky(run, "epoch_da_luu", epoch=1, lan=0)
            p = run / mod.TEN_NHAT_KY
            prefix = p.read_bytes()
            p.write_bytes(prefix + suffix)
            goc = p.read_bytes()
            before, warnings = mod.doc_nhat_ky(run)
            (run / "kp").mkdir()
            xu_ly = mod.sua_duoi_nhat_ky(run, run / "kp", mod.phan_tich_duoi_nhat_ky(goc))   # buoc chuan bi ghi
            mod.ghi_nhat_ky(run, "bat_dau", che_do_luot="tiep_tuc", lan=1, ten="fixture")
            mod.ghi_nhat_ky(run, "epoch_da_luu", epoch=2, lan=1)
            error, after = None, []
            try:
                after, _ = mod.doc_nhat_ky(run)
            except mod.LoiTiepTuc as exc:
                error = str(exc).replace(str(run), "<fixture>")
            expected = len(before) + 2
            ok = error is None and len(after) == expected and after[:len(before)] == before
            cases.append({"case": name, "accepted_events_before": len(before), "warnings_before": warnings,
                          "hanh_dong": xu_ly["hanh_dong"], "bo_bytes": xu_ly["bo_bytes"],
                          "ban_sao_trung_goc": (run / "kp" / mod.TEN_NHAT_KY).read_bytes() == goc,
                          "expected_events_after": expected, "actual_events_after": len(after), "error_after": error,
                          "original_valid_prefix_unchanged": p.read_bytes().startswith(prefix), "pass": ok,
                          "journal_after": p.read_text(encoding="utf-8")})
    src = GOC / "src" / "tiep_tuc_huan_luyen.py"
    result = {"source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
              "scope": "doc/ghi_nhat_ky + sua_duoi_nhat_ky that (buoc chuan bi ghi cua resume); khong torch/checkpoint",
              "bug_count": sum(not c["pass"] for c in cases), "cases": cases}
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["bug_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
