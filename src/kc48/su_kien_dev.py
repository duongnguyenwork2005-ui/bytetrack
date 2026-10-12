"""
su_kien_dev.py -- R48D: hai tang su kien cua giao thuc 04 muc 5.3 tren pipeline dev12 kc48_pf_dev (chi doc).

Tang 1 "che khuat" : su kien cua src/occlusion_eval.py (build_events + add_gt_eligibility, nguon doan che
                     data/interim theo config.part_file(..., "train")), ngu canh 1,2 s = 30 frame, nhu moc 2.
                     Bao ca tat ca su kien va tang con has_full (co doan che >= 0,90 = "che hoan toan").
Tang 2 "xe nho canh xe lon": tieu chi khoa `canh_xe_lon` (results/phase2/kc48_audit/tieu_chi_cong_bo.json):
                     DETRAC nho = car, lon = bus | others; dien tich lon >= 2 x nho va IoA(nho, lon) >= 0,10 cung
                     frame; su kien = chuoi frame lien tiep (cach <= 1 frame nhan) cua cung cap (nho, lon). Dung
                     lai ham cua src/kc48/audit_du_lieu.py (doc_xml_detrac, _ioa, dem_su_kien). Phai khop 88 su kien /
                     65 track nho / 9 video (03 muc 5.2), lech -> exit 2, khong cham.

Cham: ghep GT-tracker tung frame bang occlusion_eval.match_video (Hungarian IoU >= 0,5), ket cuc bang
occlusion_eval.evaluate_events (preserved / switched / lost_after / no_match_before; id_continuous). Du dieu kien theo
GT (add_gt_eligibility): loai gt_missing_before (track bat dau trong su kien / su kien cham dau video), gt_missing_after
(track ket thuc trong su kien / cham cuoi video). Giao thuc 04 muc 5.3 con loai "khong ghep duoc truoc su kien": ty le
giu ID chinh tinh tren su kien ghep duoc truoc (`giu_id_khi_co_truoc`); ty le tren moi su kien du dieu kien
(no_match_before tinh la khong giu, cach cua moc 2) bao kem. PF: trung binh 3 seed trong tung su kien.
Bootstrap theo video (B = 5000, seed 0) cho hieu voi ekf; doi chieu tren su kien ghep duoc truoc o CA HAI bien the.

    python src/kc48/su_kien_dev.py --phan cham --bien-the ekf pf100      # cham, ghi su_kien/ket_cuc_<run>.csv
    python src/kc48/su_kien_dev.py --phan tong_hop                       # bang + bootstrap (khi da cham du)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "kc48"))
import config  # noqa: E402
import occlusion_eval as oe  # noqa: E402
import audit_du_lieu as au  # noqa: E402

EXP = "kc48_pf_dev"
PROC = ROOT / "results/phase2/kc48_pf_dev/_processed"
OUT = ROOT / "results/phase2/kc48_pf_dev/su_kien"
VIDEOS = json.loads((ROOT / "results/phase2/manifests/detrac_dev.json").read_text(encoding="utf-8"))["videos"]
BIEN_THE = {"ekf": [None], "ekf-chinh": [None], "pf100": [0, 1, 2], "pf100-k1": [0, 1, 2], "pf100-k2": [0, 1, 2],
            "pf100-k1k2": [0, 1, 2], "pf300": [0, 1, 2], "pf500": [0, 1, 2], "pf300-k1": [0, 1, 2],
            "pf500-k1": [0, 1, 2]}
CONTEXT = 30                       # 1,2 s x 25 fps
KY_VONG_XE_NHO = (88, 65, 9)       # 03 muc 5.2


def ten_run(bt, s):
    return f"p2-{EXP}-A-{bt}" + (f"-s{s}" if s is not None else "")


def doc_gt() -> pd.DataFrame:
    gts = []
    for v in VIDEOS:
        d = pd.read_csv(PROC / "DETRAC-all" / v / "gt" / "gt.txt", header=None, names=oe.GT_COLS)
        d["video"] = v
        gts.append(d)
    gt = pd.concat(gts, ignore_index=True)
    return gt[gt["cls"] == config.MOT_CLASS_ID].reset_index(drop=True)


def su_kien_che(gt: pd.DataFrame) -> pd.DataFrame:
    partial = pd.read_csv(config.part_file("occlusion_segments.csv", "train"))
    full = pd.read_csv(config.part_file("full_occlusion_segments.csv", "train"))
    partial, full = partial[partial.video.isin(VIDEOS)], full[full.video.isin(VIDEOS)]
    ev = oe.add_gt_eligibility(oe.build_events(partial, full, None), gt, CONTEXT)
    ev["tang"] = "che_khuat"
    return ev


def su_kien_xe_nho(gt: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    tc = json.loads((ROOT / "results/phase2/kc48_audit/tieu_chi_cong_bo.json").read_text(encoding="utf-8"))
    tc = tc.get("tieu_chi", tc)["canh_xe_lon"]
    nho_l, lon_l = set(tc["nho"]["detrac"]), set(tc["lon"]["detrac"])
    xml = ROOT / "data/extracted/DETRAC-Train-Annotations-XML"
    rows, n_ev_tong, nho_tong, video_co = [], 0, 0, 0
    gt_fr = {k: set(v.frame) for k, v in gt.groupby(["video", "id"])}
    lech_frame = 0
    for v in VIDEOS:
        _, _, _, frames = au.doc_xml_detrac(xml / f"{v}.xml")
        pairs = []
        for num, _d, ts in frames:
            nho, lon = [], []
            for t in ts:
                x1, y1, x2, y2 = t["box"]
                ar = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if t["loai"] in nho_l:
                    nho.append((t["id"], t["box"], ar))
                if t["loai"] in lon_l:
                    lon.append((t["id"], t["box"], ar))
            for ts_, bs, as_ in nho:
                for tl, bl, al in lon:
                    if tl != ts_ and al >= 2 * as_ and au._ioa(bs, bl) >= 0.10:
                        pairs.append((ts_, tl, num))
        n_ev, n_nho = au.dem_su_kien(pairs)
        n_ev_tong += n_ev
        nho_tong += n_nho
        video_co += int(n_ev > 0)
        by = {}
        for s, l, f in pairs:
            by.setdefault((s, l), set()).add(f)
        for (s, l), fs in by.items():
            fs = sorted(fs)
            st = fs[0]
            for a, b in zip(fs, fs[1:] + [None]):
                if b is None or b - a > 1:
                    rows.append({"video": v, "track_id": int(s), "track_lon": int(l), "start_frame": st,
                                 "end_frame": a})
                    lech_frame += sum(1 for f in range(st, a + 1) if f not in gt_fr.get((v, int(s)), set())
                                      and f in fs)
                    st = b
    ev = pd.DataFrame(rows).sort_values(["video", "track_id", "start_frame", "track_lon"]).reset_index(drop=True)
    ev["length_frames"] = ev.end_frame - ev.start_frame + 1
    ev["event_id"] = ev.groupby(["video", "track_id"]).cumcount()
    ev["has_full"] = False
    ev = oe.add_gt_eligibility(ev, gt, CONTEXT)
    ev["tang"] = "xe_nho_canh_xe_lon"
    kiem = {"n_su_kien": n_ev_tong, "n_track_nho": nho_tong, "n_video": video_co, "ky_vong": list(KY_VONG_XE_NHO),
            "khop": (n_ev_tong, nho_tong, video_co) == KY_VONG_XE_NHO, "n_dong_su_kien": len(ev),
            "frame_su_kien_khong_co_trong_gt_txt": int(lech_frame)}
    return ev, kiem


def cham(bien_the: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    gt = doc_gt()
    ev_c = su_kien_che(gt)
    ev_n, kiem = su_kien_xe_nho(gt)
    (OUT / "kiem_su_kien_xe_nho.json").write_text(json.dumps(kiem, indent=2, ensure_ascii=False), encoding="utf-8")
    ev_c.to_csv(OUT / "su_kien_che_khuat.csv", index=False)
    ev_n.to_csv(OUT / "su_kien_xe_nho.csv", index=False)
    print("kiem xe nho:", kiem)
    if not kiem["khop"]:
        print("[DUNG] su kien xe nho khong khop 88 / 65 / 9 -> khong cham")
        return 2
    for bt in bien_the:
        for s in BIEN_THE[bt]:
            run = ten_run(bt, s)
            t0 = time.perf_counter()
            mats = []
            for v in VIDEOS:
                f = PROC / "trackers/DETRAC-all" / run / "data" / f"{v}.txt"
                t = pd.read_csv(f, header=None, names=oe.TRK_COLS) if f.stat().st_size > 0 \
                    else pd.DataFrame(columns=oe.TRK_COLS)
                m = oe.match_video(gt[gt.video == v], t)
                m["video"] = v
                mats.append(m)
            match = pd.concat(mats, ignore_index=True)
            out = []
            for ev in (ev_c, ev_n):
                r = oe.evaluate_events(ev.drop(columns=["tang", "track_lon"], errors="ignore"), match, gt, CONTEXT)
                r["tang"] = ev["tang"].iloc[0]
                out.append(r)
            res = pd.concat(out, ignore_index=True)
            res["bien_the"], res["seed"], res["run"] = bt, -1 if s is None else s, run
            res.to_csv(OUT / f"ket_cuc_{run}.csv", index=False)
            print(f"{run}: {time.perf_counter() - t0:.1f} s")
    return 0


def _boot(d: pd.DataFrame, col: str, rng, B=5000):
    tong = d.groupby("video")[col].agg(["sum", "size"])
    sm, sz = tong["sum"].to_numpy(), tong["size"].to_numpy()
    idx = rng.integers(0, len(sm), size=(B, len(sm)))
    bo = sm[idx].sum(1) / np.maximum(sz[idx].sum(1), 1)
    return float(sm.sum() / sz.sum()), float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))


def tong_hop() -> int:
    parts = []
    for bt, ss in BIEN_THE.items():
        for s in ss:
            p = OUT / f"ket_cuc_{ten_run(bt, s)}.csv"
            if not p.exists():
                print(f"[THIEU] {p.name} -> chay --phan cham truoc")
                return 1
            parts.append(pd.read_csv(p))
    r = pd.concat(parts, ignore_index=True)
    r = r[r.eligible].copy()
    r["giu"] = (r.outcome == "preserved").astype(float)
    r["co_truoc"] = (r.outcome != "no_match_before").astype(float)
    r["lien_tuc"] = r.id_continuous.astype(float)
    key = ["tang", "video", "track_id", "event_id", "start_frame"]
    tb = r.groupby(["bien_the"] + key + ["has_full"], dropna=False)[["giu", "co_truoc", "lien_tuc"]].mean().reset_index()
    # PF: co_truoc trung binh seed; su kien "ghep duoc truoc" o muc bien the khi co_truoc == 1 o MOI seed
    tb["co_truoc_moi_seed"] = tb.co_truoc == 1.0
    tang_con = {"che_khuat": tb[tb.tang == "che_khuat"], "che_khuat_has_full": tb[(tb.tang == "che_khuat") & tb.has_full],
                "xe_nho_canh_xe_lon": tb[tb.tang == "xe_nho_canh_xe_lon"]}
    rng = np.random.default_rng(0)
    rows = []
    for ten, d in tang_con.items():
        e = d[d.bien_the == "ekf"].set_index(key)
        for bt in BIEN_THE:
            a = d[d.bien_the == bt].set_index(key)
            ct = a[a.co_truoc_moi_seed]
            row = {"tang": ten, "bien_the": bt, "n_video": a.reset_index().video.nunique(), "n_su_kien_du_dk": len(a),
                   "n_no_match_before_tb": float((1 - a.co_truoc).sum()), "ty_le_giu_moi_su_kien": a.giu.mean(),
                   "n_su_kien_co_truoc": len(ct), "ty_le_giu_khi_co_truoc": ct.giu.mean(),
                   "ty_le_lien_tuc_moi_su_kien": a.lien_tuc.mean()}
            if bt != "ekf":
                if row["n_video"] >= 5:
                    dd = (a.giu - e.giu.reindex(a.index)).rename("d").reset_index()
                    row["hieu_giu_moi_su_kien"], row["ci_lo"], row["ci_hi"] = _boot(dd, "d", rng)
                    chung = a.index[a.co_truoc_moi_seed & e.co_truoc_moi_seed.reindex(a.index).fillna(False)
                                    .astype(bool)]
                    dc = (a.giu.loc[chung] - e.giu.loc[chung]).rename("d").reset_index()
                    row["n_su_kien_co_truoc_ca_hai"] = len(dc)
                    row["hieu_giu_khi_co_truoc"], row["ci_lo_co_truoc"], row["ci_hi_co_truoc"] = _boot(dc, "d", rng)
                else:
                    row["ghi_chu"] = "< 5 video: chi mo ta"
            rows.append(row)
    bang = pd.DataFrame(rows)
    bang.to_csv(OUT / "bang_su_kien.csv", index=False)
    loai = []
    for ten, p in (("che_khuat", "su_kien_che_khuat.csv"), ("xe_nho_canh_xe_lon", "su_kien_xe_nho.csv")):
        ev = pd.read_csv(OUT / p)
        loai.append({"tang": ten, "n_su_kien": len(ev), "n_du_dk": int(ev.eligible.sum()),
                     "n_video_du_dk": int(ev[ev.eligible].video.nunique()),
                     "n_track_du_dk": int(ev[ev.eligible][["video", "track_id"]].drop_duplicates().shape[0]),
                     **{f"loai_{k}": int(n) for k, n in ev.loc[~ev.eligible, "exclusion_reason"].value_counts().items()},
                     "n_has_full_du_dk": int((ev.eligible & ev.has_full).sum())})
    pd.DataFrame(loai).to_csv(OUT / "co_mau_va_loai.csv", index=False)
    print(pd.DataFrame(loai).to_string())
    print(bang.to_string(float_format=lambda x: f"{x:.3f}"))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phan", choices=["cham", "tong_hop"], required=True)
    ap.add_argument("--bien-the", nargs="*", default=list(BIEN_THE))
    a = ap.parse_args(argv)
    config.PROCESSED_DIR = PROC
    return cham(a.bien_the) if a.phan == "cham" else tong_hop()


if __name__ == "__main__":
    raise SystemExit(main())
