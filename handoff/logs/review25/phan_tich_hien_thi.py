"""Vong 25: tong hop cac luot 02_hien_thi_trt.sh (ban sao tren PC) -> bang so do tung luot + theo (tracker, hien thi).
MO TA, khong kiem dinh; khong xep hang CV/EKF (decode OpenCV 3.2 co hai muc, dieu kien chua kiem soat cho so sanh do).

    python phan_tich_hien_thi.py <thu_muc_ket_qua_02> [ket_qua.json]
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


def tv(v):
    v = [x for x in v if x is not None]
    return statistics.median(v) if v else None


def main() -> int:
    goc = Path(sys.argv[1])
    tn = {}
    p_tn = goc / "tai_nguyen.csv"
    if p_tn.exists():
        for r in csv.DictReader(p_tn.open(encoding="utf-8")):
            tn.setdefault(r["luot"], {})[r["moc"]] = r
    hang = []
    for d in sorted(p for p in goc.iterdir() if p.is_dir() and p.name.startswith("luot") and "_replay" not in p.name):
        m = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
        tg, dem, h = m["thoi_gian"], m["dem"], m.get("hien_thi") or {}
        hd = h.get("dem") or {}
        te = (m.get("thiet_bi") or {}).get("tegrastats") or {}
        tt = tg.get("tom_tat_sau_warmup") or {}
        so = int(d.name[4:6])
        t0, t1 = tn.get(str(so), {}).get("truoc", {}), tn.get(str(so), {}).get("sau", {})
        hang.append({
            "luot": d.name, "tracker": d.name.split("_")[1], "hien_thi": d.name.split("_")[2],
            "trang_thai": m["trang_thai"], "frame_xu_ly": dem["n_frame_da_xu_ly"], "frame_khai_bao": dem["n_frames_khai_bao"],
            "n_hien_thi": hd.get("n_hien_thi"), "n_bo_hien_thi": hd.get("n_bo_hien_thi"), "n_khong_ghep": hd.get("n_khong_ghep"),
            "fps_xu_ly": tg.get("fps_xu_ly"), "fps_on_dinh": tt.get("fps_on_dinh"), "fps_hien_thi": h.get("fps_hien_thi"),
            "decode_ms_tv": (tt.get("decode_ms") or {}).get("trung_vi"), "infer_ms_tv": (tt.get("infer_ms") or {}).get("trung_vi"),
            "tre_ms_tv": (tt.get("tre_ms") or {}).get("trung_vi"), "ve_ms_tv": (h.get("ve_ms") or {}).get("trung_vi"),
            "imshow_ms_tv": (h.get("imshow_ms") or {}).get("trung_vi"), "khoi_tao_s": tg.get("khoi_tao_s"), "tong_s": tg.get("tong_s"),
            "ram_MB_max": te.get("ram_dung_MB_max"), "swap_MB_max": te.get("swap_dung_MB_max"),
            "cpu_pct_tb": te.get("cpu_pct_tb"), "gr3d_pct_tb": (te.get("gr3d_pct_tb_max") or [None])[0],
            "nhiet_max_C": te.get("nhiet_max_C"), "mem_available_kB_truoc": t0.get("mem_available_kB"),
            "mem_available_kB_sau": t1.get("mem_available_kB"), "con_sau": t1.get("con_detector_dang_chay"),
            "cua_so_sau": t1.get("cua_so_xem")})
    cot = ["luot", "trang_thai", "frame_xu_ly", "n_hien_thi", "n_bo_hien_thi", "fps_xu_ly", "fps_hien_thi", "decode_ms_tv",
           "infer_ms_tv", "tre_ms_tv", "ve_ms_tv", "imshow_ms_tv", "ram_MB_max", "cpu_pct_tb", "gr3d_pct_tb", "khoi_tao_s"]
    f = lambda v: "-" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    print(" | ".join(cot))
    for r in hang:
        print(" | ".join(f(r[k]) for k in cot))
    nhom = {}
    for r in hang:
        nhom.setdefault((r["tracker"], r["hien_thi"]), []).append(r)
    print("\ntrung vi theo nhom (tracker, hien thi): so luot | fps_xu_ly | fps_hien_thi | decode_ms | ram_MB_max | cpu_pct_tb")
    tong = {}
    for k, rs in sorted(nhom.items()):
        t = {"n_luot": len(rs), "fps_xu_ly": tv([r["fps_xu_ly"] for r in rs]), "fps_hien_thi": tv([r["fps_hien_thi"] for r in rs]),
             "decode_ms_tv": tv([r["decode_ms_tv"] for r in rs]), "ram_MB_max": tv([r["ram_MB_max"] for r in rs]),
             "cpu_pct_tb": tv([r["cpu_pct_tb"] for r in rs]), "n_bo_hien_thi": [r["n_bo_hien_thi"] for r in rs],
             "fps_xu_ly_tung_luot": [round(r["fps_xu_ly"], 3) for r in rs]}
        tong["%s_%s" % k] = t
        print(f"{k[0]:>3} {k[1]:>3}: {t['n_luot']} | {f(t['fps_xu_ly'])} ({t['fps_xu_ly_tung_luot']}) | {f(t['fps_hien_thi'])} | "
              f"{f(t['decode_ms_tv'])} | {f(t['ram_MB_max'])} | {f(t['cpu_pct_tb'])} | bo hien thi {t['n_bo_hien_thi']}")
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps({"luot": hang, "nhom": tong,
                                                 "ghi_chu": "mo ta; khong kiem dinh; khong xep hang CV/EKF"},
                                                ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
