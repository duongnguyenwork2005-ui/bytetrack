#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 36 -- doi chieu hai JSON du toan: MOI gia tri so (int/float/bool) va cau truc khoa/do dai list phai GIONG HET;
chi duoc khac o gia tri chuoi (cau chu). Dung de chung minh dinh chinh R35-F2 khong doi so lieu.

Chay:  python handoff/logs/review36/doi_chieu_so_json.py <json goc> <json sinh lai>
Exit 0: cau truc + moi so giong het (chuoi co the khac, duoc liet ke); 1: co khac biet ve so hoac cau truc; 2: loi dau vao.
"""
import json
import sys
from pathlib import Path


def duyet(a, b, duong, khac_so, khac_cau_truc, khac_chuoi, dem):
    if isinstance(a, dict) and isinstance(b, dict):
        if list(a) != list(b):
            khac_cau_truc.append(f"{duong}: khoa {list(a)} != {list(b)}")
            return
        for k in a:
            duyet(a[k], b[k], f"{duong}.{k}", khac_so, khac_cau_truc, khac_chuoi, dem)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            khac_cau_truc.append(f"{duong}: do dai {len(a)} != {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            duyet(x, y, f"{duong}[{i}]", khac_so, khac_cau_truc, khac_chuoi, dem)
    elif isinstance(a, (bool, int, float)) and isinstance(b, (bool, int, float)) and type(a) is type(b):
        dem["so"] += 1
        if a != b:
            khac_so.append(f"{duong}: {a!r} != {b!r}")
    elif a is None and b is None:
        dem["null"] += 1
    elif isinstance(a, str) and isinstance(b, str):
        dem["chuoi"] += 1
        if a != b:
            khac_chuoi.append(duong)
    else:
        khac_cau_truc.append(f"{duong}: kieu {type(a).__name__} != {type(b).__name__}")


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    try:
        a = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        b = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"[LOI DAU VAO] {e}")
        return 2
    khac_so, khac_cau_truc, khac_chuoi = [], [], []
    dem = {"so": 0, "null": 0, "chuoi": 0}
    duyet(a, b, "$", khac_so, khac_cau_truc, khac_chuoi, dem)
    print(f"so sanh: {sys.argv[1]}  vs  {sys.argv[2]}")
    print(f"gia tri so da so: {dem['so']} | null: {dem['null']} | chuoi: {dem['chuoi']}")
    print(f"khac SO: {len(khac_so)}" + "".join(f"\n  {x}" for x in khac_so))
    print(f"khac CAU TRUC: {len(khac_cau_truc)}" + "".join(f"\n  {x}" for x in khac_cau_truc))
    print(f"chuoi da doi (chi cau chu): {len(khac_chuoi)}" + "".join(f"\n  {x}" for x in khac_chuoi))
    ok = not khac_so and not khac_cau_truc
    print("KET LUAN:", "DAT -- moi gia tri so va cau truc giong het" if ok else "KHAC -- xem tren")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
