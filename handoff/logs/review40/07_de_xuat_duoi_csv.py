#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 40 -- bang chung cho DE XUAT NGOAI PHAM VI R39-F1 (CHUA sua; chi thu muc tam, khong dong vao repo).

    python handoff/logs/review40/07_de_xuat_duoi_csv.py
1. results.csv: sao_luu_truoc_tiep_tuc cat dong theo int(float(truong dau)) -- dong cuoi GHI DO co lam resume dung bang
   loi tho (khong phai LoiTiepTuc) hoac giu lai dong rac khong?
2. theo_batch.csv: neu dong cuoi ghi do roi csv append tiep (nhu het_epoch_train) thi doc lai ra sao?
"""
import csv
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import tiep_tuc_huan_luyen as ttl  # noqa: E402

HDR = b"epoch,time,x\r\n1,10,0.5\r\n2,20,0.6\r\n"
for ten, duoi in (("byte_nul", b"\x00\x00\x00\x00"), ("so_do_dang", b"1"), ("dong_do", b"3,30,0.")):
    with tempfile.TemporaryDirectory() as t:
        r = Path(t)
        (r / "results.csv").write_bytes(HDR + duoi)
        (r / "nguon.pt").write_bytes(b"ck")
        ttl.ghi_nhat_ky(r, "bat_dau", che_do_luot="moi", lan=0)
        try:
            kq = ttl.sao_luu_truoc_tiep_tuc(r, r / "nguon.pt", 1, epoch_da_luu=2)
            print(f"results.csv duoi {ten}: khong loi; {kq['results_csv']}; 24 byte cuoi con lai "
                  f"{(r / 'results.csv').read_bytes()[-24:]!r}")
        except Exception as e:  # noqa: BLE001
            print(f"results.csv duoi {ten}: {type(e).__name__}: {str(e)[:70]} | la LoiTiepTuc: "
                  f"{isinstance(e, ttl.LoiTiepTuc)}")
rows = list(csv.DictReader(io.StringIO("lan,epoch,i\r\n0,1,0\r\n0,1,1\r\n0,1," + "1,2,0\r\n")))
print(f"theo_batch.csv: dong cuoi '0,1,' ghi do + append '1,2,0' -> dong doc lai {rows[-1]}")
