"""
test_sua_r48d.py -- test cho vong sua R48D (Codex review_003 F1-F3). Chi doc; khong ghi file.

F1  ghep phan tang co rang buoc IoU >= 0.5 (phan_tich_dev.ghep_frame -> occlusion_eval.match_frame):
    1. phan vi du tong hop: Hungarian tren moi IoU roi loc nguong mat cap hop le; ghep co rang buoc giu duoc.
    2. giu_id tren bang tong hop dung phan vi du do: track GT duoc ghep.
    3. ca that MVI_20052 / run ekf / frame 445: GT 38 -> ban cu 350 (IoU 0,72466), ban moi 290 (IoU 0,74063).
F2  tach tai khoi tao khoi ESS / systematic resampling (thi_nghiem_mo_phong.chay_mot):
    4. ep nhanh tai khoi tao o MOI update (degenerate_maha2 < 0): ESS / resample = NaN, dem rieng, mau so dung.
    5. n_deg theo TUNG quy dao (hai quy dao noi tiep cung bo loc), khong phai bo dem tich luy.
    6. chan doan khong doi dau ra bo loc: cung chuoi predict / update (co do lech xa ep tai khoi tao), PF bat
       enable_diag vs khong bat -> particle, logw, mean, cov trung khit tung buoc.
    7. tron hai co che: o ca mot quy dao co ca tai khoi tao lan update thuong, ESS hop le moi diem >= 1/N.
F3  bootstrap COMBINED tung seed roi trung binh (boot_seed):
    8. cac seed co thong ke khac nhau: combine_tb_seed == trung binh boot_stats.combine tung seed va KHAC
       boot_stats.combine(average_seeds) (HOTA, IDF1); IDSW (tuyen tinh) bang nhau.
    9. moi mau bootstrap == tinh tay (cung day rng), va mot seed moi ben trung khit boot_stats.paired_bootstrap.
   10. so that pf500-k1 (raw_fields dev12): HOTA tb seed = 0,5547572114, ban cu 0,5547515202 (so cua Codex).

    python src/kc48/test_sua_r48d.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "kc48"))
import boot_seed as bsd  # noqa: E402
import boot_stats as bs  # noqa: E402
import mo_phong_quy_dao as mp  # noqa: E402
import occlusion_eval as oe  # noqa: E402
import phan_tich_dev as pdv  # noqa: E402
import thi_nghiem_mo_phong as tn  # noqa: E402
from pf_ctrv import PFCTRV, PFTrackView  # noqa: E402

KQ: list[tuple[str, bool, str]] = []


def kiem(ten: str, dk: bool, ghi: str = "") -> None:
    KQ.append((ten, bool(dk), ghi))
    print(("PASS " if dk else "FAIL ") + ten + (f"  [{ghi}]" if ghi else ""), flush=True)


def ghep_cu(iou: np.ndarray, nguong: float = 0.5) -> list[tuple[int, int]]:
    """Cach ghep cua phan_tich_dev TRUOC vong sua (chi de tai hien phan vi du)."""
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(-iou)
    return [(int(i), int(j)) for i, j in zip(r, c) if iou[i, j] >= nguong]


# ---------------------------------------------------------------------------------------------------------- F1
def test_1_phan_vi_du():
    iou = np.array([[0.60, 0.45], [0.45, 0.00]])
    cu, moi = ghep_cu(iou), oe.match_frame(iou, 0.5)
    kiem("F1.1 loc sau Hungarian mat cap hop le (0,0)", cu == [], f"cu={cu}")
    kiem("F1.1 ghep co rang buoc giu (0,0)", moi == [(0, 0)], f"moi={moi}")


def test_2_giu_id_tong_hop():
    # hai GT, hai tracker, 3 frame; IoU dung ma tran phan vi du o moi frame -> GT 1 phai giu ID tracker 10
    def box(x, y):
        return [x, y, 10.0, 10.0]
    # GT1 (0, 0); tracker 10 dich x 2,5 -> IoU 0,6; tracker 11 dich y -3,79 -> IoU 0,45;
    # GT2 = tracker 10 dich y +3,79 -> IoU(GT2, 10) 0,45, IoU(GT2, 11) ~0,10. Hungarian tu do chon (0,1) + (1,0) = 0,9
    gt, tr = [], []
    for f in (1, 2, 3):
        gt += [[f, 1, *box(0.0, 0.0)], [f, 2, *box(2.5, 3.7931034)]]
        tr += [[f, 10, *box(2.5, 0.0)], [f, 11, *box(0.0, -3.7931034)]]
    gt = pd.DataFrame(gt, columns=["f", "id", "x", "y", "w", "h"])
    tr = pd.DataFrame(tr, columns=["f", "id", "x", "y", "w", "h"])
    iou = oe.iou_matrix(gt[gt.f == 1][["x", "y", "w", "h"]].to_numpy(float),
                        tr[tr.f == 1][["x", "y", "w", "h"]].to_numpy(float))
    kiem("F1.2 bang tong hop dung dang phan vi du", ghep_cu(iou) == [] and oe.match_frame(iou) == [(0, 0)],
         np.array2string(iou, precision=3))
    g = pdv.giu_id(gt, tr).set_index("gt_id")
    kiem("F1.2 giu_id: GT 1 giu ID 3/3 frame", g.loc[1, "ty_le_giu_id"] == 1.0, str(g.loc[1].to_dict()))


def test_3_ca_that_mvi_20052_445():
    v = "MVI_20052"
    gt = pdv._doc_mot(pdv.PROC / "DETRAC-all" / v / "gt" / "gt.txt", gt=True)
    tr = pdv._doc_mot(pdv.PROC / "trackers/DETRAC-all" / pdv.ten_run("ekf", None) / "data" / f"{v}.txt", gt=False)
    g, t = gt[gt.f == 445], tr[tr.f == 445]
    iou = oe.iou_matrix(g[["x", "y", "w", "h"]].to_numpy(float), t[["x", "y", "w", "h"]].to_numpy(float))
    i38 = int(np.flatnonzero(g.id.to_numpy() == 38)[0])
    cu = {int(g.id.iloc[i]): (int(t.id.iloc[j]), float(iou[i, j])) for i, j in ghep_cu(iou)}
    moi = pdv.ghep_frame(g, t)
    kiem("F1.3 ban cu: GT 38 -> 350, IoU 0,72466", cu.get(38, (None,))[0] == 350 and
         abs(cu[38][1] - 0.72466) < 5e-6, str(cu.get(38)))
    kiem("F1.3 ban moi: GT 38 -> 290, IoU 0,74063", moi.get(38, (None,))[0] == 290 and
         abs(moi[38][1] - 0.74063) < 5e-6, str(moi.get(38)))
    kiem("F1.3 ban moi khong it cap hop le hon ban cu", len(moi) >= len(cu), f"cu {len(cu)}, moi {len(moi)}")
    kiem("F1.3 moi cap ban moi IoU >= 0,5", all(x[1] >= 0.5 for x in moi.values()))
    j290 = int(np.flatnonzero(t.id.to_numpy() == 290)[0])
    kiem("F1.3 IoU(38, 290) dung", abs(iou[i38, j290] - 0.74063) < 5e-6, f"{iou[i38, j290]:.5f}")


# ---------------------------------------------------------------------------------------------------------- F2
def _bl(N=100, maha2=1e4, k1=False):
    pf = PFCTRV(num_particles=N, degenerate_maha2=maha2, optimal_proposal=k1)
    pf.enable_diag()
    return pf


def test_4_ep_tai_khoi_tao():
    qd = mp.sinh_quy_dao("cua_deu", 0, 24.0)
    do = mp.sinh_do(qd, 1.0, 10)
    bl = _bl(maha2=-1.0)                     # -2 * max ll >= 0 > -1 -> MOI update vao nhanh tai khoi tao
    r = tn.chay_mot(bl, qd, do, 0)
    co = np.asarray(do["co_do"], bool)
    n_upd, n_bam = int(co[1:].sum()), int(co[10:do["t_gap"]].sum())
    kiem("F2.4 moi update la tai khoi tao: n_deg == n_update", r["n_deg"] == r["n_update"] == n_upd,
         f"n_deg {r['n_deg']}, n_update {r['n_update']}, co_do {n_upd}")
    kiem("F2.4 dem rieng khi bam: n_reinit_bam == n_upd_bam > 0", r["n_reinit_bam"] == r["n_upd_bam"] == n_bam > 0,
         f"{r['n_reinit_bam']} / {r['n_upd_bam']}")
    kiem("F2.4 ESS / resample khong lay tu quy uoc tai khoi tao",
         all(np.isnan(r[k]) for k in ("ess_bam", "rs_bam", "tt_bam", "ess_tr", "rs_tr")) and r["reinit_tr"] == 1.0,
         str({k: r[k] for k in ("ess_bam", "rs_bam", "ess_tr", "rs_tr", "reinit_tr")}))
    kiem("F2.4 dau ra bo loc van huu han", np.isfinite(r["e_r"]) and np.isfinite(r["rmse_h"]))


def test_5_n_deg_theo_quy_dao():
    bl = _bl(maha2=-1.0)
    qa, qb = mp.sinh_quy_dao("thang", 0, 24.0), mp.sinh_quy_dao("thang", 1, 24.0)
    ra = tn.chay_mot(bl, qa, mp.sinh_do(qa, 1.0, 10), 0)
    rb = tn.chay_mot(bl, qb, mp.sinh_do(qb, 1.0, 20), 0)
    kiem("F2.5 n_deg quy dao 2 khong cong don quy dao 1",
         rb["n_deg"] == rb["n_update"] and bl.n_degenerate == ra["n_deg"] + rb["n_deg"],
         f"a {ra['n_deg']}, b {rb['n_deg']}, tich luy {bl.n_degenerate}")


def _chay_api(diag: bool, k1: bool, rough: float):
    """Chuoi predict / update qua PFTrackView; buoc 25 do lech 2000 px -> ep tai khoi tao. Tra ve lich su trang thai."""
    pf = PFCTRV(num_particles=200, optimal_proposal=k1, roughening=rough)
    if diag:
        pf.enable_diag()
    pf.set_seed(3, video="test_f2")
    qd = mp.sinh_quy_dao("cua_gat", 2, 48.0)
    do = mp.sinh_do(qd, 1.0, 10)
    z, co = do["z"], np.asarray(do["co_do"], bool)
    m, P = pf.initiate(z[0])
    trk = type("T", (), {})()
    trk.track_id, trk.frame_id, trk._pending_frame_id, trk._k1_prior = 1, 0, 0, None
    trk.rng = pf.make_rng(1)
    trk.particles = pf.sample_initial(m, P, trk.rng)
    trk.logw = np.full(pf.num_particles, -np.log(pf.num_particles))
    view = PFTrackView(pf, trk)
    lich_su, lost = [], False
    for t in range(1, qd["T"]):
        trk._pending_frame_id = t
        m, P = view.predict(m, P, lost=lost)
        if co[t]:
            zz = z[t] + (np.array([2000.0, 0, 0, 0]) if t == 25 else 0.0)
            trk.frame_id = t - 1
            m, P = view.update(m, P, zz)
            lost = False
        else:
            lost = True
        lich_su.append((trk.particles.copy(), trk.logw.copy(), m.copy(), P.copy()))
    return lich_su, pf


def test_6_chan_doan_khong_doi_dau_ra():
    for k1, rough in ((False, 0.0), (True, 0.0), (False, 1.0), (True, 1.0)):
        a, pa = _chay_api(True, k1, rough)
        b, pb = _chay_api(False, k1, rough)
        trung = len(a) == len(b) and all(all(np.array_equal(x, y) for x, y in zip(ra, rb)) for ra, rb in zip(a, b))
        n_kt = sum(1 for r in pa.diag_rows if r[2] == "update" and r[9])
        kiem(f"F2.6 diag bat / tat cho dau ra trung khit (k1={k1}, rough={rough})",
             trung and pa.n_degenerate == pb.n_degenerate >= 1 and n_kt == pa.n_degenerate,
             f"tai khoi tao {pa.n_degenerate}, dong diag {n_kt}")


def test_7_tron_co_che():
    # nguong suy bien thap vua phai -> mot so update tai khoi tao, mot so khong
    qd = mp.sinh_quy_dao("nhieu_duoi_day", 4, 24.0)
    do = mp.sinh_do(qd, 2.0, 10)
    bl = _bl(maha2=30.0)
    ch = []
    r = tn.chay_mot(bl, qd, do, 1, chuoi=ch)
    upd = [x for x in bl.diag_rows if x[2] == "update"]
    kt = [x for x in upd if x[9]]
    hop_le = [x[3] for x in upd if not x[9]]
    c = pd.DataFrame(ch)
    kiem("F2.7 quy dao co ca hai co che", 0 < len(kt) < len(upd), f"tai khoi tao {len(kt)} / {len(upd)}")
    kiem("F2.7 ESS/N hop le >= 1/N", min(hop_le) >= 1.0 / bl.num_particles - 1e-12, f"min {min(hop_le):.4f}")
    kiem("F2.7 chuoi: buoc tai khoi tao co ess NaN, so buoc dung",
         int(c.tai_khoi_tao.fillna(False).astype(bool).sum()) == len(kt) and
         c[c.tai_khoi_tao.fillna(False).astype(bool)].ess.isna().all() and
         c[c.tai_khoi_tao.fillna(True) == False].ess.notna().all())  # noqa: E712
    kiem("F2.7 n_deg == so dong diag tai khoi tao", r["n_deg"] == len(kt))


# ---------------------------------------------------------------------------------------------------------- F3
def _raw(tp, fn, fp, ass, idtp, idfn, idfp, ctp, cfn, cfp, idsw):
    a = np.linspace(1.0, 0.4, 19)
    return {"HOTA_TP": list(tp * a), "HOTA_FN": list(fn + tp * (1 - a)), "HOTA_FP": list(fp + tp * (1 - a)),
            "AssA": list(ass * a), "AssRe": list(ass * a), "AssPr": list(ass * a),
            "IDTP": idtp, "IDFN": idfn, "IDFP": idfp, "CLR_TP": ctp, "CLR_FN": cfn, "CLR_FP": cfp, "IDSW": idsw}


SEED_A = {"v1": _raw(900, 100, 50, 0.8, 800, 200, 150, 900, 100, 50, 5),
          "v2": _raw(100, 400, 20, 0.3, 50, 450, 70, 100, 400, 20, 9),
          "v3": _raw(500, 50, 300, 0.6, 400, 150, 400, 500, 50, 300, 2)}
SEED_B = {"v1": _raw(300, 700, 10, 0.5, 100, 900, 210, 300, 700, 10, 11),
          "v2": _raw(450, 50, 200, 0.9, 440, 60, 210, 450, 50, 200, 1),
          "v3": _raw(200, 350, 40, 0.2, 90, 460, 150, 200, 350, 40, 7)}
VID = ["v1", "v2", "v3"]


def test_8_thu_tu_tinh():
    moi = bsd.combine_tb_seed([SEED_A, SEED_B], VID)
    tay = {m: np.mean([bs.combine(SEED_A, VID)[m], bs.combine(SEED_B, VID)[m]]) for m in bs.METRICS}
    cu = bs.combine(bs.average_seeds([SEED_A, SEED_B]), VID)
    kiem("F3.8 combine_tb_seed == trung binh COMBINED tung seed", all(abs(moi[m] - tay[m]) < 1e-12 for m in bs.METRICS))
    kiem("F3.8 khac thu tu cu o HOTA, IDF1", abs(moi["HOTA"] - cu["HOTA"]) > 1e-4 and abs(moi["IDF1"] - cu["IDF1"]) > 1e-4,
         f"HOTA moi {moi['HOTA']:.6f} cu {cu['HOTA']:.6f}; IDF1 moi {moi['IDF1']:.6f} cu {cu['IDF1']:.6f}")
    kiem("F3.8 IDSW (tuyen tinh) bang nhau", abs(moi["IDSW"] - cu["IDSW"]) < 1e-9)


def test_9_bootstrap():
    r = bsd.paired_bootstrap_seed([SEED_A, SEED_B], [SEED_B], VID, n_boot=40, seed=7, tra_mau=True)
    rng = np.random.default_rng(7)
    sai = 0.0
    for i in range(40):
        idx = rng.integers(0, 3, size=3)
        vv = [VID[k] for k in idx]
        a = np.mean([bs.combine(SEED_A, vv)["HOTA"], bs.combine(SEED_B, vv)["HOTA"]])
        b = bs.combine(SEED_B, vv)["HOTA"]
        sai = max(sai, abs((a - b) - r["_mau"]["HOTA"][i]))
    kiem("F3.9 tung mau bootstrap == tinh tay", sai < 1e-12, f"sai lon nhat {sai:.2e}")
    p1 = bsd.paired_bootstrap_seed([SEED_A], [SEED_B], VID, n_boot=200, seed=0)
    p0 = bs.paired_bootstrap(SEED_A, SEED_B, VID, n_boot=200, seed=0)
    kiem("F3.9 mot seed moi ben trung khit boot_stats.paired_bootstrap",
         all(p1[m][k] == p0[m][k] for m in bs.METRICS for k in ("a", "b", "diff", "ci_low", "ci_high")))


def test_10_so_that_pf500_k1():
    rl = [bs.load_raw_fields(pdv.TE / pdv.ten_run("pf500-k1", s) / "raw_fields.json") for s in (0, 1, 2)]
    moi = bsd.combine_tb_seed(rl, pdv.VIDEOS)["HOTA"]
    cu = bs.combine(bs.average_seeds(rl), pdv.VIDEOS)["HOTA"]
    kiem("F3.10 pf500-k1 HOTA tb seed 0,5547572114", abs(moi - 0.5547572114) < 1e-9, f"{moi:.10f}")
    kiem("F3.10 ban cu 0,5547515202", abs(cu - 0.5547515202) < 1e-9, f"{cu:.10f}")


def main() -> int:
    tests = [test_1_phan_vi_du, test_2_giu_id_tong_hop, test_3_ca_that_mvi_20052_445, test_4_ep_tai_khoi_tao,
             test_5_n_deg_theo_quy_dao, test_6_chan_doan_khong_doi_dau_ra, test_7_tron_co_che, test_8_thu_tu_tinh,
             test_9_bootstrap, test_10_so_that_pf500_k1]
    for f in tests:
        try:
            f()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            kiem(f.__name__ + " (ngoai le)", False)
    n_ok = sum(1 for _, ok, _ in KQ if ok)
    print(f"\n{n_ok}/{len(KQ)} PASS")
    return 0 if n_ok == len(KQ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
