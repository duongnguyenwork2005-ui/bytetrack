#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 39 -- TINH LAI DOC LAP so lieu RAM cua 6 luot kiem GPU tu ram_mau_tho.csv (CHI DOC).

    python handoff/logs/review39/07_tinh_lai_ram_doc_lap.py > handoff/logs/review39/07_tinh_lai_ram_doc_lap.log
KHONG dung ham tom tat cua src/chay_giam_sat.py (lenh `tinh-lai` cua cong cu dung lai chinh ham do). Ma rieng, chi module
csv/json, kiem:
  1. moi dong CSV co pid_chinh = PID trong tom_tat.json = PID runner TU GHI trong su kien bat_dau cua nhat ky luot
     (ban chep 05_gpu/<model>_luot/*nhat_ky_luot.jsonl) -- nguon thu ba, do runner ghi bang os.getpid();
  2. t_unix tang dan; khong o nao bang 0; dong trong co ghi chu;
  3. so mau, max rss_chinh / rss_tong / so_con, min ram trong toan may, khoang do -- doi chieu tom_tat.json;
  4. ten cac tien trinh con xuat hien (chi_tiet_con);
  5. khoang do bao phu thoi gian lenh (tom_tat.thoi_gian_s).
"""
import csv
import json
import sys
from pathlib import Path

L = Path(__file__).resolve().parent / "05_gpu"


def nhat_ky(model: str, t: str) -> list[dict]:
    f = L / f"{model}_luot" / f"{'C' if t == 'C' else 'I'}__nhat_ky_luot.jsonl"
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]


def main() -> int:
    loi, bang = [], []
    for model in ("yolov8n", "yolo11n"):
        for t in ("C", "I", "R"):
            d = L / f"{model}_{t}"
            tom = json.loads((d / "tom_tat.json").read_text(encoding="utf-8"))
            rows = list(csv.DictReader(open(d / "ram_mau_tho.csv", newline="", encoding="utf-8")))
            lan = 1 if t == "R" else 0
            pid_nk = [s["pid"] for s in nhat_ky(model, t) if s["loai"] == "bat_dau" and s.get("lan") == lan]
            pids = {r["pid_chinh"] for r in rows}
            t_unix = [float(r["t_unix"]) for r in rows]
            rong = [i for i, r in enumerate(rows) if any(r[c] in ("", None) for c in
                                                         ("rss_chinh_mib", "rss_tong_mib", "ram_he_thong_con_trong_mib"))]
            khong = [i for i, r in enumerate(rows) if any(r[c] not in ("", None) and float(r[c]) == 0 for c in
                                                          ("rss_chinh_mib", "rss_tong_mib", "ram_he_thong_con_trong_mib"))]
            ten_con = sorted({p.split(":")[1] for r in rows for p in r["chi_tiet_con"].split(";") if p.count(":") == 2})
            kq = {"luot": f"{model}_{t}", "so_mau": len(rows), "pid_csv": sorted(pids), "pid_tom_tat": tom["pid"],
                  "pid_nhat_ky": pid_nk, "exit": tom["exit_code"],
                  "rss_chinh_max": max(float(r["rss_chinh_mib"]) for r in rows),
                  "rss_tong_max": max(float(r["rss_tong_mib"]) for r in rows),
                  "so_con_max": max(int(r["so_con"]) for r in rows),
                  "ram_trong_min": min(float(r["ram_he_thong_con_trong_mib"]) for r in rows),
                  "khoang_do_s": round(max(t_unix) - min(t_unix), 1), "thoi_gian_lenh_s": tom["thoi_gian_s"],
                  "ten_con": ten_con}
            r_tom = tom["ram"]
            dk = {
                "pid_csv == tom_tat == nhat_ky": kq["pid_csv"] == [str(tom["pid"])] and pid_nk == [tom["pid"]],
                "t tang dan": all(b > a for a, b in zip(t_unix, t_unix[1:])),
                "khong o rong / 0": not rong and not khong,
                "so_mau khop": len(rows) == r_tom["so_mau"],
                "rss_chinh_max khop": kq["rss_chinh_max"] == r_tom["rss_chinh_mib"]["max"],
                "rss_tong_max khop": kq["rss_tong_max"] == r_tom["rss_tong_mib"]["max"],
                "so_con_max khop": kq["so_con_max"] == r_tom["so_con"]["max"],
                "ram_trong_min khop": kq["ram_trong_min"] == r_tom["ram_he_thong_con_trong_mib"]["min"],
                "khoang_do khop": kq["khoang_do_s"] == r_tom["khoang_do_s"],
                "khoang do >= thoi gian lenh - 5 s": kq["khoang_do_s"] >= tom["thoi_gian_s"] - 5,
            }
            kq["kiem"] = dk
            bang.append(kq)
            loi += [f"{kq['luot']}: {k}" for k, v in dk.items() if not v]
    for kq in bang:
        print(json.dumps(kq, ensure_ascii=False))
    print(f"KET QUA: {len(bang)} luot, {sum(len(k['kiem']) for k in bang) - len(loi)}/"
          f"{sum(len(k['kiem']) for k in bang)} dieu kien dat" + (f"; LOI: {loi}" if loi else ""))
    return 0 if not loi else 1


if __name__ == "__main__":
    sys.exit(main())
