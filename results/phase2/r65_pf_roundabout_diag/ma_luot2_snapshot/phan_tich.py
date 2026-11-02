"""
phan_tich.py -- (R65) thong ke chan doan PF500 tren RoundaboutHD tu output da qua cong parity (nhanh `on`).

Theo GPT 02/11 Q2 (`hoi_codex_R64_viec_tiep_20261102/answer.json`). Chi doc output; khong chay tracker.
    1. Chuyen dong GT lien tiep: d = ||c_t - c_{t-1}|| / h_{t-1}, |h_t / h_{t-1} - 1|.
    2. Sai so du doan TRUOC association (box du doan trong assoc_log `tracks`): noi track -> GT bang ghep mot-mot
       (Hungarian, IoU >= 0,5) giua output MOT va GT o QUAN SAT TRUOC (frame output gan nhat < t); theo CUNG GT sang
       frame t: e = ||c_pred - c_GT|| / h_GT, |w/w_GT - 1|, |h/h_GT - 1|, IoU(pred, GT). Bao so mau, do phu (khong noi
       duoc / thieu GT), trung vi / p90 / p95; tren toan bo mau hop le tung bo loc VA tren tap (video, GT, frame) chung.
    3. Phan loai ket qua association cua cac mau tren: ghep dung GT / ghep det cua vat khac / khong co det phu hop
       (IoU det-GT < 0,5) / det phu hop bi track khac lay (tranh chap) / det phu hop con trong nhung chi phi khong dat.
    4. Su kien new/lost/refind/removed; phan manh: so track khac nhau noi vao moi GT, so lan doi track lien tiep.
    5. PF: ESS/N truoc resample (chi dong update khong suy bien), ty le < 0,2 / < 0,5, ty le resample, so to tien
       khac nhau / N (quan_sat pf_update, tu last_resample_idx), spread_xy/h, spread_v/h, spread_theta, maha2_best,
       init_motion, suy bien (tach rieng); khoang cach update (gap) vs trang thai Lost.
Quy tac hoa: Hungarian scipy tren ma tran sap theo id tang dan (xac dinh).

    python -B src/r65_pf_roundabout/phan_tich.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

GOC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(GOC / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
import chan_doan as cd  # noqa: E402
from assoc_log import load_log  # noqa: E402

OUT = cd.KQ / "phan_tich"
IOU_LK = 0.5
MATCH_THRESH = 0.8          # bytetrack.yaml / YAML CTRV: match_thresh (tang 1, fuse_score)
GAP_BIN = [0, 1, 3, 9, 10_000]
GAP_NHAN = ["1", "2-3", "4-9", ">=10"]       # gap = t - frame quan sat truoc; so frame THIEU quan sat = gap - 1
GAP0_BIN = [-1, 0, 1, 3, 9, 10_000]
GAP0_NHAN = ["0 (update thuong)", "1", "2-3", "4-9", ">=10"]


def q(x, p):
    x = np.asarray(x, dtype=float)
    return float(np.quantile(x, p)) if len(x) else float("nan")


def tom_tat(x) -> dict:
    x = np.asarray(x, dtype=float)
    return {"n": int(len(x)), "trung_vi": q(x, 0.5), "p90": q(x, 0.9), "p95": q(x, 0.95), "tb": float(np.mean(x)) if len(x) else float("nan")}


def iou_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a (n,4) b (m,4) tlwh -> IoU (n,m)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0][:, None], b[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1][:, None], b[:, 1][None]), 0, None)
    inter = iw * ih
    return inter / (a[:, 2:4].prod(1)[:, None] + b[:, 2:4].prod(1)[None] - inter)


def doc_gt(v: str) -> pd.DataFrame:
    g = pd.read_csv(config.PROCESSED_DIR / cd.SPLIT / v / "gt" / "gt.txt", header=None,
                    names=["frame", "id", "x", "y", "w", "h", "conf", "cls", "vis"])
    return g[g["conf"] > 0].sort_values(["frame", "id"]).reset_index(drop=True)


def doc_mot(run: str, v: str) -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED_DIR / "trackers" / cd.SPLIT / run / "data" / f"{v}.txt", header=None,
                       names=["frame", "id", "x", "y", "w", "h", "conf"]).sort_values(["frame", "id"])


def lien_ket(mot: pd.DataFrame, gt: pd.DataFrame) -> dict:
    """(frame, track_id) -> gt_id, ghep mot-mot IoU >= 0,5 tung frame."""
    out = {}
    gtf = {f: d for f, d in gt.groupby("frame")}
    for f, m in mot.groupby("frame"):
        g = gtf.get(f)
        if g is None:
            continue
        iou = iou_mat(m[["x", "y", "w", "h"]].to_numpy(), g[["x", "y", "w", "h"]].to_numpy())
        # sua theo review GPT 02/11: cap IoU < 0,5 KHONG kha thi ngay tu dau (khong ghep Hungarian roi moi loc)
        r, c = linear_sum_assignment(np.where(iou >= IOU_LK, -iou, 1e6))
        for i, j in zip(r, c):
            if iou[i, j] >= IOU_LK:
                out[(int(f), int(m["id"].iat[i]))] = int(g["id"].iat[j])
    return out


def chuyen_dong_gt() -> pd.DataFrame:
    rows = []
    for v in cd.VIDEOS:
        g = doc_gt(v)
        g["cx"], g["cy"] = g["x"] + g["w"] / 2, g["y"] + g["h"] / 2
        for gid, d in g.groupby("id"):
            d = d.sort_values("frame")
            lt = d["frame"].diff().to_numpy() == 1
            dx = np.hypot(d["cx"].diff(), d["cy"].diff()).to_numpy() / d["h"].shift().to_numpy()
            dh = np.abs(d["h"].to_numpy() / d["h"].shift().to_numpy() - 1)
            for a, b, h in zip(dx[lt], dh[lt], d["h"].to_numpy()[lt]):
                rows.append((v, gid, a, b, h))
    return pd.DataFrame(rows, columns=["video", "gt_id", "d_tren_h", "dh_rel", "h"])


def mau_du_doan(run: str) -> pd.DataFrame:
    """Moi dong = track (Tracked/Lost, da kich hoat) o frame t co box du doan, kem lien ket GT va ket qua ghep."""
    run_dir = config.PROCESSED_DIR / "trackers" / cd.SPLIT / run
    rows = []
    for v in cd.VIDEOS:
        gt = doc_gt(v)
        mot = doc_mot(run, v)
        lk = lien_ket(mot, gt)
        tr, de, _ = load_log(run_dir / "assoc_log", v)
        if not (tr["frame"] == tr["tracker_frame_id"]).all():
            raise RuntimeError(f"{run}/{v}: frame video != tracker_frame_id")
        tr = tr[(tr["state"].isin([1, 2])) & tr["is_activated"]]
        out_fr = {tid: d["frame"].to_numpy() for tid, d in mot.groupby("id")}
        gt_by = {(int(f), int(i)): (x, y, w, h) for f, i, x, y, w, h in
                 gt[["frame", "id", "x", "y", "w", "h"]].itertuples(index=False)}
        gt_fr = {f: d for f, d in gt.groupby("frame")}
        de_fr = {f: d for f, d in de.groupby("frame")}
        for r in tr.itertuples(index=False):
            t, tid = int(r.frame), int(r.track_id)
            fr = out_fr.get(tid)
            k = np.searchsorted(fr, t) - 1 if fr is not None else -1
            if k < 0:
                rows.append((v, t, tid, int(r.state), -1, -1, "khong_co_output_truoc") + (np.nan,) * 6 + ("", np.nan)); continue
            fp = int(fr[k])
            gid = lk.get((fp, tid))
            if gid is None:
                rows.append((v, t, tid, int(r.state), t - fp, -1, "khong_lien_ket") + (np.nan,) * 6 + ("", np.nan)); continue
            gb = gt_by.get((t, gid))
            if gb is None:
                rows.append((v, t, tid, int(r.state), t - fp, gid, "thieu_gt") + (np.nan,) * 6 + ("", np.nan)); continue
            gx, gy, gw, gh = gb
            px, py, pw, ph = r.x, r.y, r.w, r.h
            e = np.hypot(px + pw / 2 - gx - gw / 2, py + ph / 2 - gy - gh / 2) / gh
            iou_pg = float(iou_mat(np.array([[px, py, pw, ph]]), np.array([[gx, gy, gw, gh]]))[0, 0])
            # ket qua association
            d = de_fr.get(t)
            gbox = np.array([[gx, gy, gw, gh]])
            iou_dg = iou_mat(d[["x", "y", "w", "h"]].to_numpy(), gbox)[:, 0] if d is not None else np.zeros(0)
            iou_pd, chi_phi = np.nan, np.nan
            if int(r.matched_det) >= 0:
                m = d[d["det_idx"] == int(r.matched_det)]
                ok = len(m) and iou_mat(m[["x", "y", "w", "h"]].to_numpy()[:1], gbox)[0, 0] >= IOU_LK
                kq = "ghep_dung" if ok else "ghep_vat_khac"
            elif not len(iou_dg) or iou_dg.max() < IOU_LK:
                kq = "khong_co_det"
            else:
                # sua theo review GPT 02/11: xet MOI det phu hop (IoU det-GT >= 0,5), khong chi det gan GT nhat.
                # Kha thi: tang 1 (det high, Tracked + Lost) chi phi fused 1 - IoU*conf <= match_thresh 0,8;
                # tang 2 (det low, chi Tracked) 1 - IoU <= 0,5.
                ph_ = d[iou_dg >= IOU_LK]
                iou_p = iou_mat(np.array([[px, py, pw, ph]]), ph_[["x", "y", "w", "h"]].to_numpy(float))[0]
                cp = np.where(ph_["tier"].to_numpy() == "high", 1 - iou_p * ph_["conf"].to_numpy(float), 1 - iou_p)
                kha_thi = np.where(ph_["tier"].to_numpy() == "high", cp <= MATCH_THRESH,
                                   (cp <= 0.5) & (int(r.state) == 1))
                j = int(np.argmin(cp))
                iou_pd, chi_phi = float(iou_p[j]), float(cp[j])
                if not kha_thi.any():
                    kq = "chi_phi_khong_dat"
                elif ((ph_["matched_track"].to_numpy()[kha_thi] >= 0)
                      & (ph_["matched_track"].to_numpy()[kha_thi] != tid)).all():
                    kq = "tranh_chap"
                else:
                    kq = "kha_thi_con_trong"
            rows.append((v, t, tid, int(r.state), t - fp, gid, "hop_le", e, abs(pw / gw - 1), abs(ph / gh - 1), iou_pg,
                         iou_pd, chi_phi, kq, e * gh))
    cols = ["video", "frame", "track_id", "state", "gap", "gt_id", "loai", "e_tam_tren_h", "e_w_rel", "e_h_rel",
            "iou_du_doan_gt", "iou_du_doan_det", "chi_phi_fused", "ket_qua_ghep", "e_tam_px"]
    df = pd.DataFrame(rows, columns=cols)
    df["gap_nhom"] = pd.cut(df["gap"], GAP_BIN, labels=GAP_NHAN)
    return df


def phan_manh(run: str) -> dict:
    n_track, n_doi = [], 0
    for v in cd.VIDEOS:
        lk = lien_ket(doc_mot(run, v), doc_gt(v))
        s = pd.Series(lk)
        if not len(s):
            continue
        d = pd.DataFrame({"frame": [k[0] for k in s.index], "tid": [k[1] for k in s.index], "gid": s.values})
        for _, g in d.sort_values("frame").groupby("gid"):
            n_track.append(g["tid"].nunique())
            n_doi += int((g["tid"].diff().fillna(0) != 0).sum())
    return {"gt_co_lien_ket": len(n_track), "track_moi_gt_tb": float(np.mean(n_track)),
            "track_moi_gt_p90": q(n_track, 0.9), "so_lan_doi_track_lien_tiep": n_doi}


def su_kien(run: str) -> dict:
    run_dir = config.PROCESSED_DIR / "trackers" / cd.SPLIT / run
    ev = pd.concat([load_log(run_dir / "assoc_log", v)[2] for v in cd.VIDEOS])
    out = {k: int((ev["kind"] == k).sum()) for k in ("new", "lost", "refind", "removed", "removed_unconfirmed")}
    rf = ev[ev["kind"] == "refind"]["n_lost"]
    out["refind_n_lost"] = tom_tat(rf)
    return out


def pf_thong_ke(run: str) -> dict:
    run_dir = config.PROCESSED_DIR / "trackers" / cd.SPLIT / run
    dg, pu, hh = [], [], []
    for v in cd.VIDEOS:
        d = pd.read_csv(run_dir / "pf_diag" / f"{v}.csv"); d["video"] = v; dg.append(d)
        u = pd.read_csv(cd.TRACE / run / f"{v}_pf_update.csv"); u["video"] = v; pu.append(u)
        tr = load_log(run_dir / "assoc_log", v)[0][["frame", "track_id", "h"]]; tr["video"] = v; hh.append(tr)
    dg, pu, hh = pd.concat(dg), pd.concat(pu), pd.concat(hh).drop_duplicates(["video", "frame", "track_id"])
    up = dg[(dg["kind"] == "update") & (~dg["degenerate"].astype(bool))]
    up = up.merge(hh, on=["video", "frame", "track_id"], how="left")
    rs = pu[pu["resampled"].astype(bool) & ~pu["degenerate"].astype(bool)]
    return {
        "dong_update_hop_le": int(len(up)), "dong_suy_bien": int(dg["degenerate"].astype(bool).sum()),
        "dong_init_motion": int((dg["kind"] == "init_motion").sum()),
        "ess_ratio": tom_tat(up["ess_ratio"]),
        "ty_le_ess_duoi_0_2": float((up["ess_ratio"] < 0.2).mean()),
        "ty_le_ess_duoi_0_5": float((up["ess_ratio"] < 0.5).mean()),
        "ty_le_resample": float(up["resampled"].astype(bool).mean()),
        "to_tien_tren_N_khi_resample": tom_tat(rs["n_to_tien"] / 500.0),
        "spread_xy_tren_h": tom_tat((up["spread_xy"] / up["h"]).dropna()),
        "spread_v_tren_h": tom_tat((up["spread_v"] / up["h"]).dropna()),
        "spread_theta": tom_tat(up["spread_theta"]),
        "maha2_best": tom_tat(up["maha2_best"]),
        "h_khong_khop": int(up["h"].isna().sum()),
        "gap_x_trang_thai": {f"gap={g}|state={s}": int(n) for (g, s), n in
                             pu.assign(g=pd.cut(pu["gap"], GAP0_BIN, labels=GAP0_NHAN)).groupby(["g", "state"],
                                                                                                observed=True).size().items()},
    }


def clear_evaluator(run: str) -> dict:
    """Frag / MT / PT / ML theo CLEAR cua TrackEval (sua theo review GPT 02/11: phan manh theo evaluator khoa)."""
    import run_trackeval as rte
    results, _ = rte.run(cd.SPLIT, run, cd.VIDEOS, "mot")
    d = next(iter(results.values()))[run]
    out = {}
    for seq, m in d.items():
        c = m["pedestrian"]["CLEAR"]
        out[seq] = {k: int(c[k]) for k in ("Frag", "MT", "PT", "ML", "IDSW", "CLR_TP", "CLR_FN", "CLR_FP")}
    return out


def prior_vs_sai_so(run: str, mau: pd.DataFrame) -> dict:
    """Do trai dam hat NGAY SAU du doan, truoc association (quan_sat `pf_prior`; sua theo review GPT 02/11: khong so do
    trai SAU update voi dich chuyen GT) so voi sai so tam thuc te cua cung (track, frame). spread_xy = do lech RMS ban
    kinh 2D cua dam hat (px)."""
    pr = pd.concat([pd.read_csv(cd.TRACE / run / f"{v}_pf_prior.csv").assign(video=v) for v in cd.VIDEOS])
    pr = pr.drop_duplicates(["video", "frame", "track_id"])
    h = mau[mau["loai"] == "hop_le"].merge(pr[["video", "frame", "track_id", "h_du_doan", "spread_xy", "spread_v"]],
                                           on=["video", "frame", "track_id"], how="left")
    thieu = int(h["spread_xy"].isna().sum())
    h = h.dropna(subset=["spread_xy"])

    def khoi(d):
        ty = d["e_tam_px"] / d["spread_xy"]
        return {"n": int(len(d)), "spread_xy_tren_h": tom_tat(d["spread_xy"] / d["h_du_doan"]),
                "spread_v_tren_h": tom_tat(d["spread_v"] / d["h_du_doan"]), "e_px_tren_spread": tom_tat(ty),
                "ty_le_e_le_1_spread": float((ty <= 1).mean()), "ty_le_e_le_2_spread": float((ty <= 2).mean())}
    return {"mau_noi_duoc": int(len(h)), "mau_khong_co_prior": thieu, "tat_ca": khoi(h),
            "theo_gap": {str(g): khoi(d) for g, d in h.groupby("gap_nhom", observed=True)}}

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    par = json.loads((cd.LOG / "parity.json").read_text(encoding="utf-8"))
    if not par["dat"]:
        print("[DUNG] parity chua dat"); return 2
    cd.kiem_manifest()
    kq = {"nguon": "nhanh on (da qua parity)", "chuyen_dong_gt": {}, "bo_loc": {}}
    gm = chuyen_dong_gt()
    gm.to_csv(OUT / "chuyen_dong_gt.csv", index=False)
    kq["chuyen_dong_gt"] = {"tat_ca": {"d_tren_h": tom_tat(gm["d_tren_h"]), "dh_rel": tom_tat(gm["dh_rel"]),
                                       "h_px": tom_tat(gm["h"])},
                            **{v: {"d_tren_h": tom_tat(g["d_tren_h"]), "h_px": tom_tat(g["h"])}
                               for v, g in gm.groupby("video")}}
    runs = {f"{l}" + (f"-s{s}" if s is not None else ""): cd.ten("on", l, s) for l in cd.LOC for s in cd.don_vi(l)}
    mau = {}
    for k, run in runs.items():
        print("phan tich", run, flush=True)
        df = mau_du_doan(run)
        df.to_csv(OUT / f"mau_{k}.csv.gz", index=False)
        mau[k] = df
        h = df[df["loai"] == "hop_le"]
        kq["bo_loc"][k] = {
            "run": run, "do_phu": df["loai"].value_counts().to_dict(),
            "e_tam_tren_h_theo_gap": {str(g): tom_tat(d["e_tam_tren_h"]) for g, d in h.groupby("gap_nhom", observed=True)},
            "e_tam_tren_h_theo_state": {str(s): tom_tat(d["e_tam_tren_h"]) for s, d in h.groupby("state")},
            "e_h_rel": tom_tat(h["e_h_rel"]), "e_w_rel": tom_tat(h["e_w_rel"]),
            "iou_du_doan_gt": tom_tat(h["iou_du_doan_gt"]),
            "ty_le_iou_du_doan_gt_duoi_0_5": float((h["iou_du_doan_gt"] < 0.5).mean()),
            "ket_qua_ghep": h["ket_qua_ghep"].value_counts().to_dict(),
            "ket_qua_ghep_theo_gap": {str(g): d["ket_qua_ghep"].value_counts().to_dict()
                                      for g, d in h.groupby("gap_nhom", observed=True)},
            "chi_phi_khong_dat": {"iou_du_doan_det": tom_tat(h.loc[h["ket_qua_ghep"] == "chi_phi_khong_dat", "iou_du_doan_det"]),
                                  "chi_phi_fused": tom_tat(h.loc[h["ket_qua_ghep"] == "chi_phi_khong_dat", "chi_phi_fused"])},
            "phan_manh": phan_manh(run), "su_kien": su_kien(run),
        }
        kq["bo_loc"][k]["clear_evaluator"] = clear_evaluator(run)
        if k.startswith("pf500"):
            kq["bo_loc"][k]["pf"] = pf_thong_ke(run)
            kq["bo_loc"][k]["pf_prior"] = prior_vs_sai_so(run, df)
    # tap chung: (video, gt_id, frame) hop le, gap = 1, co o MOI bo loc
    khoa = None
    for k, df in mau.items():
        s = set(map(tuple, df[(df["loai"] == "hop_le") & (df["gap"] == 1)][["video", "gt_id", "frame"]].to_numpy()))
        khoa = s if khoa is None else khoa & s
    kq["tap_chung_gap1"] = {"n": len(khoa)}
    for k, df in mau.items():
        h = df[(df["loai"] == "hop_le") & (df["gap"] == 1)]
        h = h[[tuple(x) in khoa for x in h[["video", "gt_id", "frame"]].to_numpy()]].drop_duplicates(["video", "gt_id", "frame"])
        kq["tap_chung_gap1"][k] = {"e_tam_tren_h": tom_tat(h["e_tam_tren_h"]), "iou_du_doan_gt": tom_tat(h["iou_du_doan_gt"]),
                                   "ket_qua_ghep": h["ket_qua_ghep"].value_counts().to_dict()}
    # metric R64 (TrackEval) de doi chieu
    for k, run in runs.items():
        c = pd.read_csv(config.RESULTS_DIR / "trackeval" / cd.SPLIT / run / "combined_metrics.csv").iloc[0]
        kq["bo_loc"][k]["metric"] = {m: float(c[m]) for m in ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "FP", "FN")}
    (OUT / "thong_ke.json").write_text(json.dumps(kq, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    # bang phang
    dong = []
    for k, b in kq["bo_loc"].items():
        r = {"bo_loc": k, **{f"m_{m}": x for m, x in b["metric"].items()}}
        for g, s in b["e_tam_tren_h_theo_gap"].items():
            r[f"e_gap{g}_trung_vi"], r[f"e_gap{g}_p90"], r[f"e_gap{g}_n"] = s["trung_vi"], s["p90"], s["n"]
        n = sum(b["ket_qua_ghep"].values())
        for kk, x in b["ket_qua_ghep"].items():
            r[f"ghep_{kk}"] = x / n
        r.update({f"pm_{kk}": x for kk, x in b["phan_manh"].items()})
        r.update({f"clear_{kk}": x for kk, x in b["clear_evaluator"]["COMBINED_SEQ"].items()})
        r.update({f"sk_{kk}": x for kk, x in b["su_kien"].items() if not isinstance(x, dict)})
        dong.append(r)
    pd.DataFrame(dong).to_csv(cd.KQ / "thong_ke.csv", index=False)
    print("xong ->", OUT / "thong_ke.json", cd.KQ / "thong_ke.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
