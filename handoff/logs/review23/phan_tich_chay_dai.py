# -*- coding: utf-8 -*-
"""Tinh lai DOC LAP tu du lieu tho cua 4 luot chay day du 1470 frame (vong 23) -- ban sao tu Nano trong `nano_dai_v23/`.

Moi luot: trang thai/ma thoat; frame 1..1470 xu ly DUNG MOT LAN (pipeline + timing), ke ca frame 0 detection sau loc;
live vs replay: so byte MOT + data_cls, va so tung hang (frame, id, bbox, conf) KHONG anh xa lai ID; FPS tinh lai tu
`xu_ly_s` va moc `t_xong_cha`; moi frame nam trong cua so do; trung vi stage sau warm-up 25; tai nguyen truoc/sau luot;
tien trinh con da thoat. Them: so MOT/cache giua cac luot cung cau hinh va giua cac luot (tinh xac dinh).

    python handoff/logs/review23/phan_tich_chay_dai.py
"""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

G = Path(__file__).resolve().parent
D = G / "nano_dai_v23" / "results_nano" / "dai_v23"
W = 25
out_lines, loi = [], []


def p(s=""):
    out_lines.append(s)
    print(s)


def doc(f):
    with open(f, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


def tv(xs):
    return statistics.median(xs) if xs else None


def p90(xs):
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))] if s else None


def mot_rows(f):
    rows = []
    for ln in Path(f).read_text(encoding="utf-8").splitlines():
        v = ln.split(",")
        if len(v) >= 7:
            rows.append((int(v[0]), int(v[1]), *[float(x) for x in v[2:7]]))
    return rows


luots = sorted(q for q in D.iterdir() if q.is_dir() and q.name.startswith("luot") and not q.name.endswith(("_replay", "_cache")))
p(f"# phan tich 4 luot chay dai (vong 23) tu {D.relative_to(G)} -- {len(luots)} luot")
kq = []
for L in luots:
    m = json.loads((L / "run_meta.json").read_text(encoding="utf-8"))
    th, dem = m["thoi_gian"], m["dem"]
    pl = doc(L / "pipeline" / "MVI_39031.csv")
    tm = doc(L / "timing" / "MVI_39031.csv")
    f_pl = [int(r["frame"]) for r in pl]
    f_tm = [int(r["frame"]) for r in tm]
    mot_frame_moi_lan = f_pl == list(range(1, 1471)) and f_tm == list(range(1, 1471))
    n_rong = sum(1 for r in pl if int(r["n_det_loc"]) == 0)
    n_rong_tho = sum(1 for r in pl if int(r["n_det_raw"]) == 0)
    # live vs replay
    R = D / f"{L.name}_replay"
    live_mot, rep_mot = L / "data" / "MVI_39031.txt", R / "data" / "MVI_39031.txt"
    byte_mot = live_mot.read_bytes() == rep_mot.read_bytes()
    byte_cls = (L / "data_cls" / "MVI_39031.csv").read_bytes() == (R / "data_cls" / "MVI_39031.csv").read_bytes()
    a, b = mot_rows(live_mot), mot_rows(rep_mot)
    cung_so = len(a) == len(b)
    dbb = max((max(abs(x - y) for x, y in zip(ra[2:6], rb[2:6])) for ra, rb in zip(a, b)), default=0.0)
    dcf = max((abs(ra[6] - rb[6]) for ra, rb in zip(a, b)), default=0.0)
    cung_id = all(ra[:2] == rb[:2] for ra, rb in zip(a, b))
    pj = json.loads((D / f"parity_{L.name}.json").read_text(encoding="utf-8"))
    # FPS + cua so
    xong = [float(r["t_xong_cha"]) for r in pl]
    dau = [float(r["t_bat_dau_con"]) for r in pl]
    moc = th["moc_monotonic"]
    trong_cua_so = min(dau) >= moc["gui_bat_dau"] and max(xong) <= moc["ket_thuc_xu_ly"]
    fps_xl = 1470 / th["xu_ly_s"]
    fps_od = (1470 - W) / (xong[-1] - xong[W - 1])
    bao_phu = max(xong) - min(dau)
    sau = pl[W:]
    upd = {int(r["frame"]): float(r["t_update_ms"]) for r in tm}
    st = {k: tv([float(r[k]) for r in sau]) for k in ("decode_ms", "pre_ms", "infer_ms", "exec_ms", "post_ms",
                                                      "cho_credit_ms", "truyen_ms", "cho_doc_ms", "loc_ms", "tre_ms")}
    st["tre_ms_p90"] = p90([float(r["tre_ms"]) for r in sau])
    st["t_update_ms"] = tv([upd[int(r["frame"])] for r in sau])
    lech_meta = abs(fps_xl - th["fps_xu_ly"]) + abs(fps_od - th["tom_tat_sau_warmup"]["fps_on_dinh"])
    teg = m["thiet_bi"]["tegrastats"]
    r = {"luot": L.name, "trang_thai": m["trang_thai"], "ma": m["ma_thoat"], "n_nhan": dem["n_frame_nhan"],
         "n_xu_ly": dem["n_frame_da_xu_ly"], "n_khai_bao": dem["n_frames_khai_bao"], "eos_gui": m["eos_con"]["n_frame_gui"],
         "frame_1_1470_dung_1_lan": mot_frame_moi_lan, "n_frame_0_det_sau_loc": n_rong, "n_frame_0_det_tho": n_rong_tho,
         "n_hang_mot": len(a), "n_id": len({x[1] for x in a}), "byte_mot": byte_mot, "byte_cls": byte_cls,
         "so_hang_cung": cung_so, "max_bbox": dbb, "max_conf": dcf, "id_tung_hang_cung": cung_id,
         "parity_nano_dat": all(v.get("dat") for v in pj.values()) if isinstance(pj, dict) else pj,
         "fps_xu_ly": fps_xl, "fps_on_dinh": fps_od, "xu_ly_s": th["xu_ly_s"], "bao_phu_s": bao_phu,
         "moi_frame_trong_cua_so": trong_cua_so, "lech_voi_meta": lech_meta, "khoi_tao_s": th["khoi_tao_s"],
         "chuan_bi_cha_s": th["chuan_bi_cha_s"], "ket_thuc_s": th["ket_thuc_s"], "ghi_s": th["ghi_s"],
         "max_in_flight": m["eos_con"]["max_in_flight"], "con_returncode": m["con"]["returncode"],
         "ram_MB_max": teg.get("ram_dung_MB_max"), "swap_MB_max": teg.get("swap_dung_MB_max"),
         "cpu_MHz": teg.get("cpu_MHz_min_max"), "nhiet_max": teg.get("nhiet_max_C"), **st,
         "mot_sha256": sha(live_mot), "cache_sha256": sha(D / f"{L.name}_cache" / "MVI_39031.npz")}
    kq.append(r)
    for ten, ok in (("hoan_tat", r["trang_thai"] == "hoan_tat" and r["ma"] == 0), ("1470 dung 1 lan", mot_frame_moi_lan),
                    ("n = 1470", r["n_xu_ly"] == r["n_nhan"] == r["eos_gui"] == 1470), ("MOT byte", byte_mot),
                    ("cls byte", byte_cls), ("so hang", cung_so), ("ID tung hang", cung_id), ("bbox/conf 0", dbb == 0 and dcf == 0),
                    ("trong cua so", trong_cua_so), ("fps khop meta", lech_meta < 1e-9), ("con thoat 0", r["con_returncode"] == 0),
                    ("K<=4", r["max_in_flight"] <= 4)):
        if not ok:
            loi.append(f"{L.name}: {ten}")

p("\n## 1. Hoan tat, dem frame, live vs replay (so thang, khong anh xa ID)")
p("luot | trang thai | nhan/xu ly/EOS | 1..1470 dung 1 lan | frame 0 det (loc/tho) | hang MOT | ID | byte MOT/cls | bbox/conf max | ID tung hang")
for r in kq:
    p(f"{r['luot']} | {r['trang_thai']} ma {r['ma']} | {r['n_nhan']}/{r['n_xu_ly']}/{r['eos_gui']} | {r['frame_1_1470_dung_1_lan']} | "
      f"{r['n_frame_0_det_sau_loc']}/{r['n_frame_0_det_tho']} | {r['n_hang_mot']} | {r['n_id']} | {r['byte_mot']}/{r['byte_cls']} | "
      f"{r['max_bbox']:.0e}/{r['max_conf']:.0e} | {r['id_tung_hang_cung']}")
p("\n## 2. Thoi gian (tinh lai tu CSV/moc goc; tom tat bo 25 frame dau)")
p("luot | fps_xu_ly | fps_on_dinh | xu_ly_s | bao phu s | trong cua so | khoi tao s | ket thuc s | ghi s | decode | pre | infer | post | cho doc | update | tre tv (p90)")
for r in kq:
    p(f"{r['luot']} | {r['fps_xu_ly']:.4f} | {r['fps_on_dinh']:.4f} | {r['xu_ly_s']:.3f} | {r['bao_phu_s']:.3f} | {r['moi_frame_trong_cua_so']} | "
      f"{r['khoi_tao_s']:.2f} | {r['ket_thuc_s']:.2f} | {r['ghi_s']:.2f} | {r['decode_ms']:.2f} | {r['pre_ms']:.2f} | {r['infer_ms']:.2f} | "
      f"{r['post_ms']:.2f} | {r['cho_doc_ms']:.2f} | {r['t_update_ms']:.3f} | {r['tre_ms']:.1f} ({r['tre_ms_p90']:.1f})")
p("\n## 3. Tai nguyen truoc/sau moi luot (tai_nguyen.csv) + tegrastats")
tn = doc(D / "tai_nguyen.csv")
for x in tn:
    p(f"{x['luot']} {x['cau_hinh']} {x['moc']:5s} {x['thoi_diem']} MemAvailable {x['mem_available_kB']} kB swap {x['swap_used_MB']} MB "
      f"CPU {x['cpu_therm_C']} GPU {x['gpu_therm_C']} C con dang chay {x['so_tien_trinh_con_dang_chay']} ext4 {x['ext4_errors']} boot {x['boot']}")
for r in kq:
    p(f"{r['luot']}: tegrastats RAM max {r['ram_MB_max']} MB, swap max {r['swap_MB_max']} MB, CPU MHz {r['cpu_MHz']}, "
      f"nhiet max CPU {r['nhiet_max'].get('CPU')} GPU {r['nhiet_max'].get('GPU')}")
truoc = [int(x["mem_available_kB"]) for x in tn if x["moc"] == "truoc"]
sau_ = [x for x in tn if x["moc"] == "sau"]
boot_set = {x["boot"] for x in tn}
p(f"MemAvailable truoc moi luot: {truoc} (chenh dau-cuoi {truoc[-1] - truoc[0]:+d} kB)")
p(f"tien trinh con con chay sau moi luot: {[x['so_tien_trinh_con_dang_chay'] for x in sau_]} | ext4 errors: "
  f"{sorted({x['ext4_errors'] for x in tn})} | boot: {sorted(boot_set)}")
if any(x["so_tien_trinh_con_dang_chay"] != "0" for x in sau_):
    loi.append("con tien trinh con sau luot")
if len(boot_set) != 1:
    loi.append("Nano khoi dong lai giua chung")
p("\n## 4. Tinh xac dinh giua cac luot")
by = {}
for r in kq:
    by.setdefault(r["luot"].split("_")[1], []).append(r)
for cfg, rs in by.items():
    p(f"{cfg}: MOT sha256 {[r['mot_sha256'][:16] for r in rs]} -> {'GIONG HET' if len({r['mot_sha256'] for r in rs}) == 1 else 'KHAC'}")
p(f"det cache (npz) sha256 ca 4 luot: {[r['cache_sha256'][:16] for r in kq]} -> "
  f"{'GIONG HET' if len({r['cache_sha256'] for r in kq}) == 1 else 'KHAC (xem noi dung)'}")
p("\n## 5. Theo cau hinh (trung vi, min-max; 2 luot/cau hinh -- mo ta)")
for cfg, rs in by.items():
    for k in ("fps_xu_ly", "fps_on_dinh", "t_update_ms", "tre_ms", "decode_ms", "infer_ms"):
        v = [r[k] for r in rs]
        p(f"{cfg} {k}: {statistics.median(v):.4f} ({min(v):.4f}-{max(v):.4f})")
p(f"\nKET QUA: {'DAT' if not loi else 'CO VAN DE'} {loi}")
(G / "05_phan_tich_chay_dai.json").write_text(json.dumps({"luot": kq, "tai_nguyen": tn, "loi": loi}, ensure_ascii=False,
                                                          indent=2, default=str), encoding="utf-8")
(G / "05_phan_tich_chay_dai.log").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
sys.exit(1 if loi else 0)
