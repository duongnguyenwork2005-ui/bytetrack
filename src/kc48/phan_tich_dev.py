"""
phan_tich_dev.py -- R48D: phan tich pipeline dev12 cua kc48_pf_dev theo giao thuc 04 muc 5.2-5.3 (chi doc).

1. Mac dinh khong doi hanh vi tren du lieu THAT: file tracking `ekf`, `pf100-s0`, `pf100-s1` cua kc48_pf_dev (ma sau
   khi them K1 / hook) so sha256 voi smoke R48C (ma tai eaf2e87, truoc khi sua) tren MVI_20052, MVI_20062.
2. Bang chinh COMBINED (HOTA, DetA, AssA, IDF1, MOTA, IDSW) tu raw_fields.json; PF: COMBINED tung seed roi trung
   binh 3 seed, ca trong moi mau bootstrap (src/kc48/boot_seed.py, vong sua F3; ban cu trung binh truong tho truoc)
   + khoang giua cac seed. Bootstrap ghep cap theo video, B = 5000, seed 0.
3. Phan tang (04 muc 5.3): ghep GT-tracker tung frame bang Hungarian CO RANG BUOC IoU >= 0.5
   (occlusion_eval.match_frame, vong sua F1; ban cu loc nguong sau Hungarian); moi track GT: ty_le_giu_id,
   n_doi_id. Tang: lop (vehicle_type XML), kich thuoc (tam phan vi trung vi h tren toan bo track GT dev12),
   nhom cua (quy_dao/detrac_train.csv loai_max; do nhay loai_p90). Bootstrap theo video cho hieu voi ekf.
   CHUA CHAY o day: su kien che khuat (occlusion_eval) va su kien xe nho canh xe lon (ghi ro trong bao cao).

    python src/kc48/phan_tich_dev.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "kc48"))
import boot_stats as bs  # noqa: E402
import boot_seed as bsd  # noqa: E402

EXP = "kc48_pf_dev"
OUT = ROOT / "results/phase2/kc48_pf_dev/phan_tich"
PROC = ROOT / "results/phase2/kc48_pf_dev/_processed"
TE = ROOT / "results/trackeval/DETRAC-all"
SMOKE = ROOT / "results/phase2/kc48_pf_smoke/_processed/trackers/DETRAC-all"
VIDEOS = json.loads((ROOT / "results/phase2/manifests/detrac_dev.json").read_text(encoding="utf-8"))["videos"]
BIEN_THE = {"ekf": [None], "ekf-chinh": [None], "pf100": [0, 1, 2], "pf100-k1": [0, 1, 2], "pf100-k2": [0, 1, 2],
            "pf100-k1k2": [0, 1, 2], "pf300": [0, 1, 2], "pf500": [0, 1, 2], "pf300-k1": [0, 1, 2],
            "pf500-k1": [0, 1, 2]}
DOI_CHIEU = [("P1", "pf100", "ekf"), ("P2", "pf100-k1", "pf100"), ("P2", "pf100-k2", "pf100"),
             ("P2", "pf100-k1k2", "pf100"), ("P2", "ekf-chinh", "ekf"), ("P3", "pf100-k1", "ekf"),
             ("P3", "pf300", "ekf"), ("P3", "pf500", "ekf"), ("P3", "pf300-k1", "ekf"), ("P3", "pf500-k1", "ekf"),
             ("P3", "pf300-k1", "pf300"), ("P3", "pf500-k1", "pf500")]
METRICS = ["HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW"]


def ten_run(bt, s):
    return f"p2-{EXP}-A-{bt}" + (f"-s{s}" if s is not None else "")


def kiem_mac_dinh() -> dict:
    out = []
    for bt, s, sm in (("ekf", None, "ekf"), ("pf100", 0, "pf100-s0"), ("pf100", 1, "pf100-s1")):
        for v in ("MVI_20052", "MVI_20062"):
            a = PROC / "trackers/DETRAC-all" / ten_run(bt, s) / "data" / f"{v}.txt"
            b = SMOKE / f"p2-kc48_pf_smoke-A-{sm}" / "data" / f"{v}.txt"
            ha, hb = hashlib.sha256(a.read_bytes()).hexdigest(), hashlib.sha256(b.read_bytes()).hexdigest()
            out.append({"run_dev": ten_run(bt, s), "run_smoke": f"p2-kc48_pf_smoke-A-{sm}", "video": v,
                        "sha256_dev": ha, "sha256_smoke": hb, "trung": ha == hb})
    return {"mo_ta": "file tracking (byte tho) cua ma sau khi them K1 / hook, mac dinh, so voi smoke R48C tao tai "
                     "eaf2e87 (truoc khi sua ma)", "tat_ca_trung": all(r["trung"] for r in out), "chi_tiet": out}


def bang_chinh():
    raws = {bt: [bs.load_raw_fields(TE / ten_run(bt, s) / "raw_fields.json") for s in ss] for bt, ss in BIEN_THE.items()}
    rows = []
    for bt, rl in raws.items():
        c = bsd.combine_tb_seed(rl, VIDEOS)          # vong sua F3: COMBINED tung seed roi trung binh (04 muc 5.2)
        per = [bs.combine(r, VIDEOS) for r in rl]
        row = {"bien_the": bt, "n_seed": len(rl)}
        for m in METRICS:
            row[m] = c[m]
            if len(per) > 1:
                row[m + "_min_seed"] = min(p[m] for p in per)
                row[m + "_max_seed"] = max(p[m] for p in per)
        rows.append(row)
    bc = pd.DataFrame(rows)
    dc = []
    for ma, a, b in DOI_CHIEU:
        r = bsd.paired_bootstrap_seed(raws[a], raws[b], VIDEOS, 5000, 0, METRICS)
        for m in METRICS:
            x = r[m]
            dc.append({"doi_chieu": ma, "a": a, "b": b, "chi_so": m, "a_gt": x["a"], "b_gt": x["b"], "hieu": x["diff"],
                       "ci_lo": x["ci_low"], "ci_hi": x["ci_high"], "ci_loai_0": bool(x["ci_low"] > 0 or x["ci_high"] < 0)})
    return bc, pd.DataFrame(dc)


# ----------------------------------------------------------------------------------------------------------------
def _doc_mot(p: Path, gt: bool) -> pd.DataFrame:
    d = pd.read_csv(p, header=None)
    if gt and d.shape[1] > 6:
        d = d[d[6] != 0]
    d = d.iloc[:, :6]
    d.columns = ["f", "id", "x", "y", "w", "h"]
    return d


def ghep_frame(g: pd.DataFrame, t: pd.DataFrame) -> dict:
    """Vong sua F1: ghep 1-1 CO RANG BUOC IoU >= 0.5 (occlusion_eval.match_frame: cap duoi nguong bi phat BIG_M TRUOC
    Hungarian). Ban cu giai Hungarian tren moi IoU roi moi loc nguong -> co the mat / doi cap hop le.
    Tra ve {gt_id: (tracker_id, iou)}."""
    import occlusion_eval as oe
    iou = oe.iou_matrix(g[["x", "y", "w", "h"]].to_numpy(float), t[["x", "y", "w", "h"]].to_numpy(float))
    return {int(g.id.iloc[i]): (int(t.id.iloc[j]), float(iou[i, j])) for i, j in oe.match_frame(iou, 0.5)}


def giu_id(gt: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    ghep = {}
    tg = dict(tuple(tr.groupby("f")))
    for f, g in gt.groupby("f"):
        t = tg.get(f)
        if t is None or len(t) == 0:
            continue
        for gid, (tid, _) in ghep_frame(g, t).items():
            ghep[(gid, int(f))] = tid
    rows = []
    for gid, g in gt.groupby("id"):
        seq = [ghep.get((int(gid), int(f))) for f in sorted(g.f)]
        m = [x for x in seq if x is not None]
        chinh = max(set(m), key=m.count) if m else None
        doi = sum(1 for a, b in zip(m[:-1], m[1:]) if a != b)
        rows.append({"gt_id": int(gid), "n_frame": len(seq), "ty_le_giu_id": (m.count(chinh) / len(seq)) if m else 0.0,
                     "n_doi_id": doi, "h_trung_vi": float(g.h.median())})
    return pd.DataFrame(rows)


def phan_tang():
    import audit_du_lieu as au
    qd = pd.read_csv(ROOT / "results/phase2/kc48_audit/_dem/quy_dao/detrac_train.csv")
    qd = qd[qd.video.isin(VIDEOS)]
    xml_dir = ROOT / "data/extracted/DETRAC-Train-Annotations-XML"
    meta, kiem_id = [], []
    gts = {}
    for v in VIDEOS:
        gt = _doc_mot(PROC / "DETRAC-all" / v / "gt" / "gt.txt", gt=True)
        gts[v] = gt
        _, _, _, frames = au.doc_xml_detrac(xml_dir / f"{v}.xml")
        lop, nf_xml = {}, {}
        for _, _, ts in frames:
            for t in ts:
                lop.setdefault(int(t["id"]), t["loai"])
                nf_xml[int(t["id"])] = nf_xml.get(int(t["id"]), 0) + 1
        nf_gt = gt.groupby("id").size().to_dict()
        qv = qd[qd.video == v].set_index("track")
        for gid, n in nf_gt.items():
            kiem_id.append({"video": v, "gt_id": int(gid), "n_frame_gt": int(n), "n_frame_xml": nf_xml.get(int(gid), 0),
                            "n_frame_quy_dao": int(qv.n_frame.get(int(gid), -1)) if int(gid) in qv.index else -1})
            meta.append({"video": v, "gt_id": int(gid), "lop": lop.get(int(gid)),
                         "loai_max": qv.loai_max.get(int(gid)) if int(gid) in qv.index else None,
                         "loai_p90": qv.loai_p90.get(int(gid)) if int(gid) in qv.index else None})
    meta = pd.DataFrame(meta)
    kid = pd.DataFrame(kiem_id)
    kid["khop_xml"] = kid.n_frame_gt == kid.n_frame_xml
    kid["khop_quy_dao"] = kid.n_frame_gt == kid.n_frame_quy_dao
    rows = []
    for bt, ss in BIEN_THE.items():
        for s in ss:
            for v in VIDEOS:
                tr = _doc_mot(PROC / "trackers/DETRAC-all" / ten_run(bt, s) / "data" / f"{v}.txt", gt=False)
                g = giu_id(gts[v], tr)
                g["video"], g["bien_the"], g["seed"] = v, bt, -1 if s is None else s
                rows.append(g)
    tk = pd.concat(rows, ignore_index=True).merge(meta, on=["video", "gt_id"], how="left")
    q = tk[(tk.bien_the == "ekf")].h_trung_vi.quantile([1 / 3, 2 / 3]).to_list()
    tk["kich_thuoc"] = np.where(tk.h_trung_vi <= q[0], "nho", np.where(tk.h_trung_vi <= q[1], "vua", "lon"))
    nhom = {"cua": "cua", "cua_gat": "cua", "thang": "thang", "cong_nhe": "cong_nhe"}
    tk["nhom_cua"] = tk.loai_max.map(nhom).fillna("loai")
    tk["nhom_cua_p90"] = tk.loai_p90.map(nhom).fillna("loai")
    # PF: trung binh 3 seed trong tung track GT
    tb = tk.groupby(["bien_the", "video", "gt_id", "lop", "kich_thuoc", "nhom_cua", "nhom_cua_p90"], dropna=False)[
        ["ty_le_giu_id", "n_doi_id"]].mean().reset_index()
    rng = np.random.default_rng(0)
    out = []
    for tang in ("lop", "kich_thuoc", "nhom_cua", "nhom_cua_p90"):
        for gia_tri, s in tb.groupby(tang, dropna=False):
            vids = sorted(set(s.video))
            for bt in BIEN_THE:
                a = s[s.bien_the == bt]
                row = {"tang": tang, "gia_tri": gia_tri, "bien_the": bt, "n_video": len(vids),
                       "n_track": int(len(a[["video", "gt_id"]].drop_duplicates())) if len(a) else 0,
                       "ty_le_giu_id_tb": a.ty_le_giu_id.mean(),
                       "n_doi_id_tb": a.n_doi_id.mean()}
                if bt != "ekf" and len(vids) >= 5:
                    e = s[s.bien_the == "ekf"].set_index(["video", "gt_id"]).ty_le_giu_id
                    x = a.set_index(["video", "gt_id"]).ty_le_giu_id
                    d = (x - e.reindex(x.index)).reset_index()
                    tong = d.groupby("video").ty_le_giu_id.agg(["sum", "size"]).reindex(vids)
                    sm, sz = tong["sum"].to_numpy(), tong["size"].to_numpy()
                    idx = rng.integers(0, len(vids), size=(5000, len(vids)))
                    bo = sm[idx].sum(1) / np.maximum(sz[idx].sum(1), 1)
                    row.update(hieu_giu_id_vs_ekf=float(sm.sum() / sz.sum()), ci_lo=float(np.percentile(bo, 2.5)),
                               ci_hi=float(np.percentile(bo, 97.5)))
                elif bt != "ekf":
                    row["ghi_chu"] = "< 5 video: chi mo ta"
                out.append(row)
    return tk, kid, pd.DataFrame(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import luu_ban_cu
    luu_ban_cu.luu([OUT / n for n in ("bang_chinh.csv", "doi_chieu_bootstrap.csv", "giu_id_theo_track.csv",
                                      "kiem_id_gt_xml_quy_dao.csv", "phan_tang.csv", "mac_dinh_vs_smoke.json")],
                   OUT / "_ban_cu_truoc_vong_sua")
    km = kiem_mac_dinh()
    (OUT / "mac_dinh_vs_smoke.json").write_text(json.dumps(km, indent=2, ensure_ascii=False), encoding="utf-8")
    print("mac dinh == smoke eaf2e87:", km["tat_ca_trung"])
    bc, dc = bang_chinh()
    bc.to_csv(OUT / "bang_chinh.csv", index=False)
    dc.to_csv(OUT / "doi_chieu_bootstrap.csv", index=False)
    print(bc[["bien_the"] + METRICS].to_string(float_format=lambda x: f"{x:.4f}"))
    print(dc[dc.chi_so.isin(["HOTA", "IDF1", "IDSW", "AssA"])].to_string(float_format=lambda x: f"{x:+.4f}"))
    tk, kid, pt = phan_tang()
    tk.to_csv(OUT / "giu_id_theo_track.csv", index=False)
    kid.to_csv(OUT / "kiem_id_gt_xml_quy_dao.csv", index=False)
    pt.to_csv(OUT / "phan_tang.csv", index=False)
    print(f"kiem id: {len(kid)} track GT; khop so frame XML {int(kid.khop_xml.sum())}, khop quy_dao "
          f"{int(kid.khop_quy_dao.sum())}")
    print(pt[pt.bien_the.isin(["ekf", "pf100", "pf100-k1", "pf500-k1"])].to_string(float_format=lambda x: f"{x:.3f}"))
    return 0 if km["tat_ca_trung"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
