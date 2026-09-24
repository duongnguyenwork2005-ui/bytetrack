# -*- coding: utf-8 -*-
"""Tinh lai benchmark CV <-> EKF (vong 20, lan 2) TU DU LIEU THO da sao luu tu Nano.

Chi doc: timing/<video>.csv cua 12 luot trong sao_luu_nano/goi_v5/.../NANO-bench/ va bench_cv_vs_ekf.json.
Khong import ma cua repo (doc lap voi cong cu `nano_bench_tracker.py`). Chay tu goc repo:
    python handoff/logs/review20/tinh_lai_bench.py
Ghi: handoff/logs/review20/09_tinh_lai_tu_raw.{log,json}
"""
import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parent
TRK = GOC / "sao_luu_nano/goi_v5/data/processed/trackers/NANO-bench"
JS = GOC / "sao_luu_nano/goi_v5/results_nano/bench_cv_ekf/lan2/bench_cv_vs_ekf.json"
WARMUP = 50
VIDEO = "MVI_39031"
out_lines = []


def p(s=""):
    out_lines.append(s)
    print(s)


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def q_ca_hai(xs):
    """Q1/Q3 theo hai quy uoc cua `statistics.quantiles` (n=4)."""
    ex = statistics.quantiles(xs, n=4, method="exclusive")
    inc = statistics.quantiles(xs, n=4, method="inclusive")
    return {"exclusive_(cong_cu)": [ex[0], ex[2]], "inclusive": [inc[0], inc[2]]}


def doc_csv(f):
    with f.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows


bc = json.loads(JS.read_text(encoding="utf-8"))
p(f"# Tinh lai benchmark tu timing CSV tho (doc lap voi cong cu). Nguon: {TRK.relative_to(GOC)}")
p(f"# bench_cv_vs_ekf.json sha256 {sha(JS)}")
p(f"# warm-up bo {WARMUP} frame dau moi luot (theo giao_thuc.warmup_frames = {bc['giao_thuc']['warmup_frames']})")
assert bc["giao_thuc"]["warmup_frames"] == WARMUP and bc["giao_thuc"]["videos"] == [VIDEO]
assert bc["giao_thuc"]["pairs"] == 6

loi = []
luots = []
ndets_ref = None
ntracks_ref = {}
p()
p("## 1. Tung luot: so frame, thong ke tu CSV, doi chieu voi JSON cua cong cu")
p("chay r cap thu_tu cau_hinh tracker        n_nguon n_do  n_steps!=1  upd_tb  upd_tv  pre_tb  assoc_tb(cot) assoc_tb(upd-pre) |d_json|max  wall_s")
moc_truoc = None
for chay, l in enumerate(bc["luots"], 1):
    moc = l["thiet_bi_truoc"]["thoi_diem"]
    if moc_truoc is not None and not moc > moc_truoc:
        loi.append(f"{l['tracker_name']}: thoi diem khong tang ({moc} <= {moc_truoc})")
    moc_truoc = moc
    f = TRK / l["tracker_name"] / "timing" / f"{VIDEO}.csv"
    rows = doc_csv(f)
    frames = [int(r["frame"]) for r in rows]
    if frames != list(range(1, len(rows) + 1)):
        loi.append(f"{l['tracker_name']}: cot frame khong phai 1..N")
    do = rows[WARMUP:]
    upd = [float(r["t_update_ms"]) for r in do]
    pre = [float(r["t_predict_ms"]) for r in do]
    asc = [float(r["t_assoc_ms"]) for r in do]
    n_steps_khac_1 = sum(int(r["n_steps"]) != 1 for r in rows)
    ndets = [int(r["n_dets"]) for r in rows]
    ntr = [int(r["n_tracks"]) for r in rows]
    if ndets_ref is None:
        ndets_ref = ndets
    elif ndets != ndets_ref:
        loi.append(f"{l['tracker_name']}: n_dets khac luot dau")
    ntracks_ref.setdefault(l["cau_hinh"], ntr)
    if ntr != ntracks_ref[l["cau_hinh"]]:
        loi.append(f"{l['tracker_name']}: n_tracks khac luot cung cau hinh")
    tinh = {"n_frame_nguon": len(rows), "n_frame_do": len(do),
            "update_ms_tb": statistics.mean(upd), "update_ms_trung_vi": statistics.median(upd),
            "predict_ms_tb": statistics.mean(pre), "predict_ms_trung_vi": statistics.median(pre),
            "assoc_ms_tb_cot": statistics.mean(asc),
            "assoc_ms_tb_upd_tru_pre": statistics.mean(upd) - statistics.mean(pre),
            "update_ms_min": min(upd), "update_ms_max": max(upd)}
    lech = max(abs(tinh[k] - l[k]) for k in ("update_ms_tb", "update_ms_trung_vi",
                                               "predict_ms_tb", "predict_ms_trung_vi"))
    if tinh["n_frame_do"] != l["n_frame_do"]:
        loi.append(f"{l['tracker_name']}: n_frame_do {tinh['n_frame_do']} != JSON {l['n_frame_do']}")
    if lech > 1e-9:
        loi.append(f"{l['tracker_name']}: lech JSON {lech}")
    if l["exit"] != 0:
        loi.append(f"{l['tracker_name']}: exit {l['exit']}")
    luots.append({"thu_tu_chay": chay, "thoi_diem_truoc": moc, **{k: l[k] for k in ("luot", "cap", "thu_tu_trong_cap", "cau_hinh", "tracker_name", "exit", "wall_s")},
                  "tegrastats": l["tegrastats"],
                  "CPU_therm_truoc": l["thiet_bi_truoc"]["nhiet_do_C"].get("CPU-therm"),
                  "CPU_therm_sau": l["thiet_bi_sau"]["nhiet_do_C"].get("CPU-therm"),
                  "timing_csv_sha256": sha(f), "n_steps_khac_1": n_steps_khac_1, **tinh,
                  "lech_max_voi_json": lech})
    thu_tu = "AB" if l["cap"] <= 3 else "BA"
    p(f"{chay:>4} {l['luot']} {l['cap']:>3} {thu_tu}/{l['thu_tu_trong_cap']}  {l['cau_hinh']:<4} {l['tracker_name']:<14} "
      f"{len(rows):>7} {len(do):>4} {n_steps_khac_1:>10}  {tinh['update_ms_tb']:.4f} {tinh['update_ms_trung_vi']:.4f} "
      f"{tinh['predict_ms_tb']:.4f} {tinh['assoc_ms_tb_cot']:.4f}        {tinh['assoc_ms_tb_upd_tru_pre']:.4f}"
      f"            {lech:.1e}  {l['wall_s']:.2f}")

p(f"tong so dong assoc: |t_assoc_ms - (t_update_ms - t_predict_ms)| max = "
  f"{max(abs(x['assoc_ms_tb_cot'] - x['assoc_ms_tb_upd_tru_pre']) for x in luots):.2e} (theo trung binh luot)")

# ---------------------------------------------------------------- ghep cap
p()
p("## 2. Ghep cap theo `cap` (1-3 AB = cv->ekf, 4-6 BA = ekf->cv); hieu so = EKF - CV")
hieu = []
for c in range(1, 7):
    cap = [x for x in luots if x["cap"] == c]
    if len(cap) != 2 or {x["cau_hinh"] for x in cap} != {"cv", "ekf"}:
        loi.append(f"cap {c} khong du cv+ekf")
        continue
    cv = next(x for x in cap if x["cau_hinh"] == "cv")
    ek = next(x for x in cap if x["cau_hinh"] == "ekf")
    thu_tu = "AB" if cv["thu_tu_trong_cap"] == 1 else "BA"
    if thu_tu != ("AB" if c <= 3 else "BA"):
        loi.append(f"cap {c}: thu tu {thu_tu} khong dung lich")
    h = {"cap": c, "thu_tu": thu_tu, "chay_cv": cv["thu_tu_chay"], "chay_ekf": ek["thu_tu_chay"],
         "d_update_ms": ek["update_ms_tb"] - cv["update_ms_tb"],
         "d_update_trung_vi_ms": ek["update_ms_trung_vi"] - cv["update_ms_trung_vi"],
         "d_predict_ms": ek["predict_ms_tb"] - cv["predict_ms_tb"],
         "d_assoc_ms": ek["assoc_ms_tb_upd_tru_pre"] - cv["assoc_ms_tb_upd_tru_pre"],
         "d_wall_s": ek["wall_s"] - cv["wall_s"],
         "ti_le_update_pct": 100.0 * (ek["update_ms_tb"] / cv["update_ms_tb"] - 1.0)}
    hj = next(x for x in bc["hieu_so_theo_cap"] if x["cap"] == c)
    lech = max(abs(h["d_update_ms"] - hj["d_update_ms"]), abs(h["d_predict_ms"] - hj["d_predict_ms"]))
    if lech > 1e-9 or hj["thu_tu"] != thu_tu:
        loi.append(f"cap {c}: lech JSON {lech} / thu tu {hj['thu_tu']}")
    h["lech_max_voi_json"] = lech
    hieu.append(h)
    p(f"cap {c} {thu_tu} (chay {cv['thu_tu_chay']:>2} cv, {ek['thu_tu_chay']:>2} ekf): d_update {h['d_update_ms']:+.4f}  "
      f"d_predict {h['d_predict_ms']:+.4f}  d_assoc {h['d_assoc_ms']:+.4f}  d_update(trung vi frame) "
      f"{h['d_update_trung_vi_ms']:+.4f}  d_wall {h['d_wall_s']:+.3f} s  ({h['ti_le_update_pct']:+.1f} %)  |lech JSON| {lech:.1e}")

# ---------------------------------------------------------------- tong hop
def tom_tat(ten, xs):
    s = sorted(xs)
    r = {"n": len(s), "trung_vi": statistics.median(s), **q_ca_hai(s), "min": s[0], "max": s[-1],
         "so_cap_duong": sum(x > 0 for x in s), "gia_tri_sap_xep": s}
    q = r["exclusive_(cong_cu)"]; qi = r["inclusive"]
    p(f"{ten:<26} trung vi {r['trung_vi']:+.4f} | Q1-Q3 exclusive(cong cu) [{q[0]:+.4f}, {q[1]:+.4f}] "
      f"inclusive [{qi[0]:+.4f}, {qi[1]:+.4f}] | min/max {r['min']:+.4f}/{r['max']:+.4f} | duong {r['so_cap_duong']}/{r['n']}")
    return r


p()
p("## 3. Tong hop 6 cap (mo ta; IQR KHONG phai khoang tin cay, 6 cap khong du de noi 'co y nghia thong ke')")
th = {"d_update_ms": tom_tat("d_update (ms/frame, tb)", [h["d_update_ms"] for h in hieu]),
      "d_predict_ms": tom_tat("d_predict (ms/frame, tb)", [h["d_predict_ms"] for h in hieu]),
      "d_assoc_ms": tom_tat("d_assoc=upd-pre (ms/frame)", [h["d_assoc_ms"] for h in hieu]),
      "d_update_trung_vi_ms": tom_tat("d_update (trung vi frame)", [h["d_update_trung_vi_ms"] for h in hieu]),
      "d_wall_s": tom_tat("d_wall (s/luot)", [h["d_wall_s"] for h in hieu]),
      "ti_le_update_pct": tom_tat("ti le update (%)", [h["ti_le_update_pct"] for h in hieu])}
tj = bc["tong_hop"]
lech_th = max(abs(th["d_update_ms"]["trung_vi"] - tj["trung_vi_d_update_ms"]),
              abs(th["d_update_ms"]["exclusive_(cong_cu)"][0] - tj["iqr"][0]),
              abs(th["d_update_ms"]["exclusive_(cong_cu)"][1] - tj["iqr"][1]),
              abs(th["d_update_ms"]["min"] - tj["min"]), abs(th["d_update_ms"]["max"] - tj["max"]))
if lech_th > 1e-9 or tj["n_cap"] != 6:
    loi.append(f"tong_hop lech JSON {lech_th}")
p(f"doi chieu tong_hop cua cong cu (trung vi, IQR, min, max, n_cap=6): |lech| max = {lech_th:.1e}")

p()
p("## 4. Theo cau hinh (6 luot moi cau hinh)")
theo_ch = {}
for ch in ("cv", "ekf"):
    xs = [x for x in luots if x["cau_hinh"] == ch]
    d = {}
    for k in ("update_ms_tb", "update_ms_trung_vi", "predict_ms_tb", "assoc_ms_tb_upd_tru_pre", "wall_s"):
        v = sorted(x[k] for x in xs)
        d[k] = {"trung_vi": statistics.median(v), "min": v[0], "max": v[-1]}
    d["tegra_nhiet_max_C"] = [min(x["tegrastats"]["nhiet_max_C"] for x in xs), max(x["tegrastats"]["nhiet_max_C"] for x in xs)]
    d["tegra_n_mau"] = sorted({x["tegrastats"]["n_mau"] for x in xs})
    theo_ch[ch] = d
    p(f"{ch:<4} update tb: trung vi {d['update_ms_tb']['trung_vi']:.3f} ({d['update_ms_tb']['min']:.3f}-{d['update_ms_tb']['max']:.3f}) | "
      f"predict {d['predict_ms_tb']['trung_vi']:.3f} ({d['predict_ms_tb']['min']:.3f}-{d['predict_ms_tb']['max']:.3f}) | "
      f"assoc {d['assoc_ms_tb_upd_tru_pre']['trung_vi']:.3f} ({d['assoc_ms_tb_upd_tru_pre']['min']:.3f}-{d['assoc_ms_tb_upd_tru_pre']['max']:.3f}) | "
      f"wall {d['wall_s']['min']:.2f}-{d['wall_s']['max']:.2f} s | tegrastats nhiet max {d['tegra_nhiet_max_C'][0]}-{d['tegra_nhiet_max_C'][1]} C, n_mau {d['tegra_n_mau']}")
ti_le = 100.0 * th["d_update_ms"]["trung_vi"] / theo_ch["cv"]["update_ms_tb"]["trung_vi"]
p(f"trung vi d_update / trung vi CV update = {ti_le:+.1f} %")

p()
p("## 5. Tinh xac dinh cua output (cung cache, cung cau hinh)")
p(f"n_dets theo frame giong nhau o ca 12 luot: {'CO' if not any('n_dets' in e for e in loi) else 'KHONG'}")
p(f"n_tracks theo frame giong nhau trong 6 luot moi cau hinh: {'CO' if not any('n_tracks' in e for e in loi) else 'KHONG'}")
p(f"thu tu chay theo thiet_bi_truoc.thoi_diem tang ngat: {'CO' if not any('thoi diem' in e for e in loi) else 'KHONG'}")
p(f"frame co n_steps != 1 (elapsed-mode steps): {sorted({x['n_steps_khac_1'] for x in luots})} (moi luot)")
p("hash data/MVI_39031.txt ca 12 luot: xem 09a_hash_output_12_luot.log (tren Nano, chi doc)")

p()
p(f"## KET QUA: {'DAT - so lieu tinh lai tu CSV tho khop JSON cua cong cu (|lech| <= 1e-9), 6 cap du AB/BA' if not loi else 'CO LOI'}")
for e in loi:
    p(f"  [LOI] {e}")

(GOC / "09_tinh_lai_tu_raw.json").write_text(json.dumps(
    {"nguon_json_sha256": sha(JS), "warmup_frames": WARMUP, "luots": luots, "hieu_so_theo_cap": hieu,
     "tong_hop_6_cap": th, "theo_cau_hinh": theo_ch, "ti_le_trung_vi_pct": ti_le,
     "ghi_chu": "IQR la mo ta phan tan cua 6 hieu so, KHONG phai khoang tin cay; so lieu la tracker-from-cache, "
                "khong gom detector, khong phai FPS end-to-end",
     "loi": loi}, ensure_ascii=False, indent=2), encoding="utf-8")
(GOC / "09_tinh_lai_tu_raw.log").write_text("\n".join(out_lines) + "\n", encoding="utf-8")
sys.exit(1 if loi else 0)
