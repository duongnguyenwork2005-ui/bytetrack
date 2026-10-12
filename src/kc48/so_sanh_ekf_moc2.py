"""
so_sanh_ekf_moc2.py -- R48, buoc Pfe7c525a (chi doc): so tung dong file tracking EKF cua smoke kc48 voi run moc 2.

  smoke (eaf2e87): results/phase2/kc48_pf_smoke/_processed/trackers/DETRAC-all/p2-kc48_pf_smoke-A-ekf/data/<video>.txt
  moc 2 (1c7ce8c): data/processed/trackers/DETRAC-all/p2-expB_detrac_dev_fix1-A-bt-ekf/data/<video>.txt

Voi moi video: sha256 + kieu xuong dong; so dong; so dong trung theo vi tri, dong dau tien lech; so theo frame (tap dong
cua frame, khoa (frame, id)): frame dau tien lech, phan loai (chi khac thu tu / cung khoa khac so / khoa chi co o mot
ben / bbox trung nhung khac id), do lech so lon nhat tren khoa chung; track dau tien lech. So them data_cls/<video>.csv.
Quy uoc hash: tinh sha256 moi file theo HAI quy uoc -- byte tho (src/run_fingerprint.py tai 1c7ce8c) va noi dung chuan
hoa CRLF -> LF (tu 935a445) -- roi doi chieu voi tracking.per_video[:16] trong eval_meta.json cua tung run, de biet
"sha256 khac" trong eval_meta la do noi dung hay do quy uoc.
Ghi results/phase2/kc48_ekf_tai_lap/so_sanh_dong.json + frame_<video>.csv. Exit 0 khi doc du file va ghi xong (lech
hay khong nam trong JSON), 2 khi thieu file.

    python src/kc48/so_sanh_ekf_moc2.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CU = ROOT / "data/processed/trackers/DETRAC-all/p2-expB_detrac_dev_fix1-A-bt-ekf"
MOI = ROOT / "results/phase2/kc48_pf_smoke/_processed/trackers/DETRAC-all/p2-kc48_pf_smoke-A-ekf"
VIDEOS = ("MVI_20052", "MVI_20062")
OUT = ROOT / "results/phase2/kc48_ekf_tai_lap"
COT = ("x", "y", "w", "h", "score")
NGUONG = (0.0, 1e-9, 1e-6, 1e-3, 1e-1, 1.0, 10.0)  # bac do lech tuyet doi (px / diem score)


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def doc(p: Path) -> dict:
    b = p.read_bytes()
    lines = b.decode("utf-8").splitlines()
    return {"path": _rel(p), "sha256": hashlib.sha256(b).hexdigest(),
            "sha256_lf": hashlib.sha256(b.replace(b"\r\n", b"\n")).hexdigest(), "bytes": len(b),
            "crlf": b.count(b"\r\n"), "lf": b.count(b"\n"), "ket_thuc_xuong_dong": b.endswith(b"\n"),
            "lines": lines}


EVAL_META = {"cu": ROOT / "results/trackeval/DETRAC-all/p2-expB_detrac_dev_fix1-A-bt-ekf/eval_meta.json",
             "moi": ROOT / "results/trackeval/DETRAC-all/p2-kc48_pf_smoke-A-ekf/eval_meta.json"}


def doi_chieu_eval_meta(video: str, a: dict, b: dict) -> dict:
    """Hash 16 ky tu da ghi trong eval_meta cua moi run khop quy uoc nao (byte tho / LF) cua file nao."""
    out = {}
    for ten, p in EVAL_META.items():
        em = json.loads(p.read_text(encoding="utf-8"))
        ghi = (em.get("tracking") or {}).get("per_video", {}).get(video)
        out[ten] = {"eval_meta": _rel(p), "git_commit": (em.get("code") or {}).get("git_commit"),
                    "evaluated_at": em.get("evaluated_at"), "per_video_ghi": ghi,
                    "khop": [f"{f}:{q}" for f, d in (("file_cu", a), ("file_moi", b))
                             for q, k in (("byte_tho", "sha256"), ("lf", "sha256_lf")) if ghi and d[k][:16] == ghi]}
    return out


def tach(lines: list[str]) -> dict:
    """frame -> {id: (chuoi goc, [float x, y, w, h, score, ...])}; giu ca thu tu dong trong frame."""
    by_f: dict[int, dict[int, tuple[str, list[float]]]] = defaultdict(dict)
    order: dict[int, list[int]] = defaultdict(list)
    for ln in lines:
        f = ln.split(",")
        fr, tid = int(f[0]), int(f[1])
        if tid in by_f[fr]:
            raise ValueError(f"khoa (frame, id) lap: {fr},{tid}")
        by_f[fr][tid] = (",".join(f[2:]), [float(v) for v in f[2:]])
        order[fr].append(tid)
    return {"by_f": by_f, "order": order}


def so_vi_tri(a: list[str], b: list[str]) -> dict:
    n = min(len(a), len(b))
    trung = sum(1 for i in range(n) if a[i] == b[i])
    dau = next((i for i in range(n) if a[i] != b[i]), None)
    if dau is None and len(a) != len(b):
        dau = n
    out = {"n_dong_cu": len(a), "n_dong_moi": len(b), "n_dong_trung_vi_tri": trung,
           "n_dong_khac_vi_tri": max(len(a), len(b)) - trung, "dong_dau_lech_1based": None}
    if dau is not None:
        out["dong_dau_lech_1based"] = dau + 1
        out["dong_cu"] = a[dau] if dau < len(a) else None
        out["dong_moi"] = b[dau] if dau < len(b) else None
    ca, cb = Counter(a), Counter(b)
    out["n_dong_chi_o_cu_multiset"] = sum((ca - cb).values())
    out["n_dong_chi_o_moi_multiset"] = sum((cb - ca).values())
    return out


def so_frame(cu: dict, moi: dict, video: str) -> dict:
    fa, fb = cu["by_f"], moi["by_f"]
    frames = sorted(set(fa) | set(fb))
    rows, max_cot = [], [0.0] * len(COT)
    dem_nguong = Counter()
    tong = Counter()
    dau = {"frame_lech_noi_dung": None, "frame_lech_khoa": None, "frame_chi_khac_thu_tu": []}
    track_dau: dict[int, int] = {}  # id -> frame dau tien id do lech (gia tri, hoac chi co o mot ben)
    for fr in frames:
        a, b = fa.get(fr, {}), fb.get(fr, {})
        ka, kb = set(a), set(b)
        chung = ka & kb
        khac_so, max_f = [], 0.0
        for tid in sorted(chung):
            if a[tid][0] != b[tid][0]:
                d = [abs(x - y) for x, y in zip(a[tid][1], b[tid][1])]
                for j, v in enumerate(d[:len(COT)]):
                    max_cot[j] = max(max_cot[j], v)
                m = max(d[:4])  # bbox
                max_f = max(max_f, m)
                for t in NGUONG:
                    if m > t:
                        dem_nguong[f">{t:g}"] += 1
                khac_so.append(tid)
                track_dau.setdefault(tid, fr)
        for tid in (ka ^ kb):
            track_dau.setdefault(tid, fr)
        # bbox trung chuoi (x,y,w,h) nhung id khac: dau hieu danh so lai id
        bb_a = Counter(",".join(v[0].split(",")[:4]) for v in a.values())
        bb_b = Counter(",".join(v[0].split(",")[:4]) for v in b.values())
        n_bb_trung = sum((bb_a & bb_b).values())
        same_set = (ka == kb) and not khac_so
        if same_set and cu["order"].get(fr) != moi["order"].get(fr):
            dau["frame_chi_khac_thu_tu"].append(fr)
        if not same_set:
            if dau["frame_lech_noi_dung"] is None:
                dau["frame_lech_noi_dung"] = fr
            if ka != kb and dau["frame_lech_khoa"] is None:
                dau["frame_lech_khoa"] = fr
        tong["frame"] += 1
        tong["frame_lech"] += int(not same_set)
        tong["khoa_chung"] += len(chung)
        tong["khoa_chung_khac_so"] += len(khac_so)
        tong["khoa_chi_cu"] += len(ka - kb)
        tong["khoa_chi_moi"] += len(kb - ka)
        rows.append({"video": video, "frame": fr, "n_cu": len(a), "n_moi": len(b), "n_khoa_chung": len(chung),
                     "n_khoa_chung_khac_so": len(khac_so), "n_chi_cu": len(ka - kb), "n_chi_moi": len(kb - ka),
                     "n_bbox_trung_chuoi": n_bb_trung, "max_lech_bbox_px": max_f,
                     "id_khac_so": " ".join(map(str, khac_so)),
                     "id_chi_cu": " ".join(map(str, sorted(ka - kb))), "id_chi_moi": " ".join(map(str, sorted(kb - ka)))})
    chi_tiet = None
    f0 = dau["frame_lech_noi_dung"]
    if f0 is not None:
        a, b = fa.get(f0, {}), fb.get(f0, {})
        chi_tiet = {"frame": f0,
                    "cu": {str(t): a[t][0] for t in sorted(a)}, "moi": {str(t): b[t][0] for t in sorted(b)},
                    "lech_tung_id": {str(t): [abs(x - y) for x, y in zip(a[t][1], b[t][1])]
                                     for t in sorted(set(a) & set(b)) if a[t][0] != b[t][0]}}
    ids_cu = {t for d in fa.values() for t in d}
    ids_moi = {t for d in fb.values() for t in d}
    track_dau_sx = sorted(track_dau.items(), key=lambda kv: (kv[1], kv[0]))
    return {"tong": dict(tong), "max_lech_tung_cot_khoa_chung": dict(zip(COT, max_cot)),
            "dem_khoa_chung_lech_bbox_theo_nguong": dict(dem_nguong), "dau": {**dau,
            "frame_chi_khac_thu_tu": dau["frame_chi_khac_thu_tu"][:20],
            "n_frame_chi_khac_thu_tu": len(dau["frame_chi_khac_thu_tu"])},
            "chi_tiet_frame_dau_lech": chi_tiet,
            "track_dau_lech_10_dau": [{"id": t, "frame": f} for t, f in track_dau_sx[:10]],
            "n_track_lech": len(track_dau), "n_id_cu": len(ids_cu), "n_id_moi": len(ids_moi),
            "frame_min_max_cu": [min(fa), max(fa)] if fa else None,
            "frame_min_max_moi": [min(fb), max(fb)] if fb else None}, rows


def main() -> int:
    thieu = [_rel(p) for v in VIDEOS for p in (CU / "data" / f"{v}.txt", MOI / "data" / f"{v}.txt",
                                               CU / "data_cls" / f"{v}.csv", MOI / "data_cls" / f"{v}.csv")
             if not p.is_file()]
    if thieu:
        print("thieu file:", thieu)
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    ket_qua = {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cu": _rel(CU), "moi": _rel(MOI),
               "nguong_lech_bbox": list(NGUONG), "videos": {}}
    for v in VIDEOS:
        a, b = doc(CU / "data" / f"{v}.txt"), doc(MOI / "data" / f"{v}.txt")
        ca, cb = doc(CU / "data_cls" / f"{v}.csv"), doc(MOI / "data_cls" / f"{v}.csv")
        tom, rows = so_frame(tach(a["lines"]), tach(b["lines"]), v)
        with (OUT / f"frame_{v}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        ket_qua["videos"][v] = {
            "file": {k: {kk: vv for kk, vv in d.items() if kk != "lines"} for k, d in
                     (("data_cu", a), ("data_moi", b), ("cls_cu", ca), ("cls_moi", cb))},
            "byte_trung_khit": a["sha256"] == b["sha256"],
            "eval_meta_quy_uoc_hash": doi_chieu_eval_meta(v, a, b),
            "data_vi_tri": so_vi_tri(a["lines"], b["lines"]),
            "cls_vi_tri": so_vi_tri(ca["lines"], cb["lines"]),
            "theo_frame": tom, "csv_frame": _rel(OUT / f"frame_{v}.csv")}
        t = tom["tong"]
        print(f"{v}: dong cu/moi {len(a['lines'])}/{len(b['lines'])}, trung vi tri "
              f"{ket_qua['videos'][v]['data_vi_tri']['n_dong_trung_vi_tri']}, dong dau lech "
              f"{ket_qua['videos'][v]['data_vi_tri']['dong_dau_lech_1based']}; frame dau lech "
              f"{tom['dau']['frame_lech_noi_dung']} (khoa: {tom['dau']['frame_lech_khoa']}); frame lech "
              f"{t['frame_lech']}/{t['frame']}; khoa chung khac so {t['khoa_chung_khac_so']}/{t['khoa_chung']}; "
              f"chi cu/moi {t['khoa_chi_cu']}/{t['khoa_chi_moi']}; max lech {tom['max_lech_tung_cot_khoa_chung']}")
        em = ket_qua["videos"][v]["eval_meta_quy_uoc_hash"]
        print(f"  byte trung khit: {a['sha256'] == b['sha256']}; eval_meta cu {em['cu']['per_video_ghi']} khop "
              f"{em['cu']['khop']}; eval_meta moi {em['moi']['per_video_ghi']} khop {em['moi']['khop']}")
    p = OUT / "so_sanh_dong.json"
    p.write_text(json.dumps(ket_qua, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {_rel(p)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
