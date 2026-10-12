"""
kiem_loc_quy_dao.py -- R48B, kiem sua loi hau audit F1 (review_003): track nghi_nhieu theo MAX dpsi phai bi loai khoi
tap hop le cua CA chi so chinh (max) LAN do nhay (p90); sau loc n_hop_le = thang + cong_nhe + cua + cua_gat.
CHI DOC; voi --ghi chi ghi results/phase2/kc48_audit/kiem_loc_quy_dao.json.

  A  ham thuan (khong doc du lieu): phan_loai_track(179, 41), bien nguong 150, tong_hop_quy_dao tren 7 track nho
  B  quy dao tong hop qua ham quy_dao() that: cung tron 41 do/s (hop le, cua) va cung + doan quay dau (max > 150,
     p90 trong [30, 60) -> phai bi loai o ca hai chi so)
  C  dau ra that: CSV tung track _dem/quy_dao/*.csv va quy_dao_cua.csv (luat loc, bat bien, tong hop lai khop CSV,
     vi du review MVI_20064 track 34, so review_003 tu tinh), tieu chi cong bo khong doi (sha256)

Lenh: C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/kiem_loc_quy_dao.py [--ghi]
Exit 0 khi moi kiem dat, 1 khi co kiem hong.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys

import audit_du_lieu as ad   # cung thu muc src/kc48 (sys.path[0] khi chay script)

SHA_TIEU_CHI = "d7543fffa6aeeccd8ba864bc189025bc4c99e308bcb0347708e1aa6d7615f3ec"   # 03_audit_du_lieu.md muc 1
TAP = ("bdd_train", "bdd_val", "detrac_train", "detrac_test")
NHOM = ad.NHOM_CD
# so review_003 (F1) tu tinh tren CSV tung track, bdd_trainfit, moi lop
SO_REVIEW = {"n_truoc_loc_nhieu": 41329, "nghi_nhieu": 8162, "n_hop_le": 33167, "cua_hoac_gat_max": 12437,
             "nghi_nhieu_bi_tinh_cua_hoac_gat_truoc_loc_p90": 3405}


class Kq:
    def __init__(self):
        self.ds = []

    def kiem(self, ten: str, dat: bool, chi_tiet=None):
        self.ds.append({"kiem": ten, "dat": bool(dat), "chi_tiet": chi_tiet})
        print(f"  [{'DAT ' if dat else 'HONG'}] {ten}" + ("" if chi_tiet is None else f" -- {chi_tiet}"))


def bat_bien(rows: list[dict]) -> list:
    """n_hop_le = tong 4 nhom = n_track - loai_ngan - loai_cham - nghi_nhieu; tra ve cac khoa vi pham."""
    sai = []
    for r in rows:
        v = {k: int(r[k]) for k in ("n_track", "loai_ngan", "loai_cham", "nghi_nhieu", "n_hop_le", *NHOM)}
        if not (v["n_hop_le"] == sum(v[x] for x in NHOM)
                == v["n_track"] - v["loai_ngan"] - v["loai_cham"] - v["nghi_nhieu"]):
            sai.append((r["tap_con"], r["lop"], r["chi_so"], v["n_hop_le"], sum(v[x] for x in NHOM)))
    return sai


def phan_a(kq: Kq, g: dict):
    print("A. ham thuan")
    r = ad.phan_loai_track(179.0, 41.0, g)
    kq.kiem("A1 max=179, p90=41 -> loai_max = loai_p90 = nghi_nhieu, loai_p90_tho = cua",
            r == {"loai_max": "nghi_nhieu", "loai_p90": "nghi_nhieu", "loai_p90_tho": "cua"}, r)
    r = ad.phan_loai_track(150.0, 41.0, g)
    kq.kiem("A2 max=150 (nguong la '>' theo tieu chi) -> khong loai; loai_max = cua_gat, loai_p90 = cua",
            r == {"loai_max": "cua_gat", "loai_p90": "cua", "loai_p90_tho": "cua"}, r)
    rows = [{"tap_con": "t", "lop": "car", **ad.phan_loai_track(m, p, g)}
            for m, p in ((179.0, 41.0), (170.0, 155.0), (10.0, 5.0), (45.0, 20.0), (90.0, 50.0))]
    rows += [{"tap_con": "t", "lop": "car", "loai_max": x, "loai_p90": x, "loai_p90_tho": x}
             for x in ("loai_ngan", "loai_cham")]
    out = {(o["lop"], o["chi_so"]): o for o in ad.tong_hop_quy_dao(rows)}
    mong = {"loai_max": dict(n_track=7, loai_ngan=1, loai_cham=1, n_truoc_loc_nhieu=5, nghi_nhieu=2, n_hop_le=3,
                             thang=1, cong_nhe=0, cua=1, cua_gat=1, truoc_loc_cua=1, truoc_loc_cua_gat=1,
                             truoc_loc_nghi_nhieu=2, nghi_nhieu_bi_tinh_cua_hoac_gat_truoc_loc=0),
            "loai_p90": dict(n_track=7, loai_ngan=1, loai_cham=1, n_truoc_loc_nhieu=5, nghi_nhieu=2, n_hop_le=3,
                             thang=1, cong_nhe=1, cua=1, cua_gat=0, truoc_loc_cua=2, truoc_loc_cua_gat=0,
                             truoc_loc_nghi_nhieu=1, nghi_nhieu_bi_tinh_cua_hoac_gat_truoc_loc=1)}
    for cs, m in mong.items():
        for lop in ("car", "_tat_ca"):
            o = out[(lop, cs)]
            lech = {k: (o[k], v) for k, v in m.items() if o[k] != v}
            kq.kiem(f"A3 tong_hop_quy_dao 7 track ({lop}, {cs}): dem sau / truoc loc dung ky vong", not lech,
                    lech or None)
    kq.kiem("A4 bat bien n_hop_le = tong 4 nhom sau loc tren moi hang tong hop", not bat_bien(list(out.values())))


def cung_tron(n_cung: int, n_quay: int, fps=25.0, v=10.0, omega_deg_s=41.0, h=40.0):
    """Da giac deu (buoc v px, quay omega/fps moi frame) -> moi dinh nam tren mot duong tron; n_quay frame cuoi di
    nguoc lai chinh duong cu (c(N+j) = c(N-2-j)), tao mot lan quay dau."""
    w = math.radians(omega_deg_s) / fps
    x = y = th = 0.0
    cx, cy = [], []
    for _ in range(n_cung):
        cx.append(x)
        cy.append(y)
        x, y, th = x + v * math.cos(th), y + v * math.sin(th), th + w
    for j in range(n_quay):
        cx.append(cx[n_cung - 2 - j])
        cy.append(cy[n_cung - 2 - j])
    n = len(cx)
    return list(range(n)), cx, cy, [h] * n


def phan_b(kq: Kq, tcq: dict):
    print("B. quy dao tong hop qua quy_dao() (fps 25, 41 do/s)")
    r = ad.quy_dao(*cung_tron(600, 0), 25.0, tcq)
    kq.kiem("B1 cung tron deu: max = p90 = 41 do -> cua o ca hai chi so (khong bi loai)",
            r["loai_max"] == r["loai_p90"] == r["loai_p90_tho"] == "cua"
            and abs(r["max_dpsi_deg"] - 41) < 0.01 and abs(r["p90_dpsi_deg"] - 41) < 0.01,
            {k: r[k] for k in ("max_dpsi_deg", "p90_dpsi_deg", "loai_max", "loai_p90", "loai_p90_tho", "n_dpsi")})
    r = ad.quy_dao(*cung_tron(600, 60), 25.0, tcq)
    kq.kiem("B2 cung + quay dau: max > 150, p90 trong [30, 60) -> nghi_nhieu o ca max lan p90; p90 tho = cua",
            r["loai_max"] == r["loai_p90"] == "nghi_nhieu" and r["loai_p90_tho"] == "cua"
            and r["max_dpsi_deg"] > 150 and 30 <= r["p90_dpsi_deg"] < 60,
            {k: r[k] for k in ("max_dpsi_deg", "p90_dpsi_deg", "loai_max", "loai_p90", "loai_p90_tho", "n_dpsi")})


def phan_c(kq: Kq, tc: dict, sha_tc: str) -> dict:
    print("C. dau ra that")
    kq.kiem("C0 tieu_chi_cong_bo.json khong doi (sha256 nhu luc cong bo)", sha_tc == SHA_TIEU_CHI, sha_tc)
    g = tc["quy_dao"]["nguong_deg"]
    nguong = [g[k] for k in ("thang_duoi", "cong_nhe_duoi", "cua_duoi", "nghi_nhieu_tren")]
    rows, du_cot = [], True
    for tap in TAP:
        rs = ad.doc_csv(ad.DEM / "quy_dao" / f"{tap}.csv")
        thieu = [c for c in ad.COT_QD if c not in rs[0]]
        du_cot &= not thieu
        kq.kiem(f"C1 _dem/quy_dao/{tap}.csv du cot (gom loai_p90_tho)", not thieu,
                {"n_track": len(rs), "thieu_cot": thieu})
        rows += rs
    vp1 = [r for r in rows if r["loai_max"] == "nghi_nhieu" and r["loai_p90"] != "nghi_nhieu"]
    kq.kiem("C2 luat loc: loai_max = nghi_nhieu => loai_p90 = nghi_nhieu (moi track)", not vp1,
            {"n_vi_pham": len(vp1), "n_vi_pham_bdd_trainfit_p90_cua_hoac_gat": sum(
                1 for r in vp1 if r["tap_con"] == "bdd_trainfit" and r["loai_p90"] in ("cua", "cua_gat"))})
    if not du_cot:
        kq.kiem("C3-C6 bo qua: CSV tung track theo schema cu (chua chay lai phan nhan)", False)
        return {}
    vp2 = [r for r in rows if r["loai_max"] != "nghi_nhieu" and r["loai_p90"] != r["loai_p90_tho"]]
    sai_tinh, bien = [], 0
    for r in rows:
        if r["max_dpsi_deg"] == "":
            if not r["loai_max"] == r["loai_p90"] == r["loai_p90_tho"] in ("loai_ngan", "loai_cham"):
                sai_tinh.append(r)
            continue
        m, p = float(r["max_dpsi_deg"]), float(r["p90_dpsi_deg"])
        if any(abs(x - t) <= 0.005 for x in (m, p) for t in nguong):
            bien += 1   # gia tri da lam tron 2 chu so, sat nguong: khong tai tinh duoc chac chan
            continue
        if ad.phan_loai_track(m, p, g) != {k: r[k] for k in ("loai_max", "loai_p90", "loai_p90_tho")}:
            sai_tinh.append(r)
    kq.kiem("C3 track khong nghi_nhieu: loai_p90 = loai_p90_tho", not vp2, {"n_vi_pham": len(vp2)})
    kq.kiem("C4 tai tinh nhan tu max/p90 (nguong cong bo) khop CSV tung track", not sai_tinh,
            {"n_track": len(rows), "n_sai": len(sai_tinh), "n_sat_nguong_bo_qua": bien,
             "vi_du_sai": [(r["video"], r["track"]) for r in sai_tinh[:5]]})
    lai = {(o["tap_con"], o["lop"], o["chi_so"]): o for o in ad.tong_hop_quy_dao(rows)}
    csv_th = ad.doc_csv(ad.OUT / "quy_dao_cua.csv")
    lech = [(r["tap_con"], r["lop"], r["chi_so"], c) for r in csv_th for c in ad.COT_QD_TH
            if str(lai.get((r["tap_con"], r["lop"], r["chi_so"]), {}).get(c, "<thieu>")) != r.get(c, "<thieu cot>")]
    kq.kiem("C5 quy_dao_cua.csv = tong_hop_quy_dao(CSV tung track) (moi cot, moi hang)",
            not lech and len(csv_th) == len(lai), {"n_hang_csv": len(csv_th), "n_hang_tinh": len(lai),
                                                    "n_o_lech": len(lech), "vi_du": lech[:5]})
    sai_bb = bat_bien(csv_th)
    cap = {}
    for r in csv_th:
        cap.setdefault((r["tap_con"], r["lop"]), set()).add((r["nghi_nhieu"], r["n_hop_le"], r["n_truoc_loc_nhieu"]))
    kq.kiem("C6 quy_dao_cua.csv: n_hop_le = tong 4 nhom = n_track - ngan - cham - nghi_nhieu; max va p90 cung tap "
            "hop le", not sai_bb and all(len(v) == 1 for v in cap.values()),
            {"n_hang": len(csv_th), "vi_pham_bat_bien": sai_bb[:5],
             "cap_lech_tap_hop_le": [k for k, v in cap.items() if len(v) != 1][:5]})
    vd = [r for r in rows if r["video"] == "MVI_20064" and r["track"] == "34"]
    kq.kiem("C7 vi du review: MVI_20064 track 34 (max 179,18; p90 41,43) -> nghi_nhieu o ca hai chi so, p90 tho = cua",
            len(vd) == 1 and vd[0]["max_dpsi_deg"] == "179.18" and vd[0]["p90_dpsi_deg"] == "41.43"
            and vd[0]["loai_max"] == vd[0]["loai_p90"] == "nghi_nhieu" and vd[0]["loai_p90_tho"] == "cua",
            [{k: r[k] for k in ("tap_con", "max_dpsi_deg", "p90_dpsi_deg", "loai_max", "loai_p90", "loai_p90_tho")}
             for r in vd])
    m, p = lai[("bdd_trainfit", "_tat_ca", "loai_max")], lai[("bdd_trainfit", "_tat_ca", "loai_p90")]
    so = {"n_truoc_loc_nhieu": m["n_truoc_loc_nhieu"], "nghi_nhieu": m["nghi_nhieu"], "n_hop_le": m["n_hop_le"],
          "cua_hoac_gat_max": m["cua"] + m["cua_gat"],
          "nghi_nhieu_bi_tinh_cua_hoac_gat_truoc_loc_p90": p["nghi_nhieu_bi_tinh_cua_hoac_gat_truoc_loc"]}
    kq.kiem("C8 bdd_trainfit (moi lop) khop so review_003 tu tinh", so == SO_REVIEW,
            {"tinh": so, "review_003": SO_REVIEW, "ti_le_max_sau_loc": m["ti_le_cua_hoac_gat_tren_hop_le"],
             "ti_le_max_truoc_loc": m["ti_le_cua_hoac_gat_truoc_loc"]})
    return {"n_track": len(rows), "bdd_trainfit": so}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ghi", action="store_true", help="ghi results/phase2/kc48_audit/kiem_loc_quy_dao.json")
    a = ap.parse_args(argv)
    tc, sha_tc = ad.nap_tieu_chi()
    kq = Kq()
    phan_a(kq, tc["quy_dao"]["nguong_deg"])
    phan_b(kq, tc["quy_dao"])
    so = phan_c(kq, tc, sha_tc)
    n_hong = sum(not x["dat"] for x in kq.ds)
    print(f"[kiem_loc_quy_dao] {len(kq.ds) - n_hong}/{len(kq.ds)} dat")
    if a.ghi:
        p = ad._duong_ghi(ad.OUT / "kiem_loc_quy_dao.json")
        p.write_text(json.dumps({"lenh": "python src/kc48/kiem_loc_quy_dao.py " + " ".join(sys.argv[1:]),
                                 "tao_luc": dt.datetime.now().isoformat(timespec="seconds"), "git_head": ad.git_head(),
                                 "tieu_chi_sha256": sha_tc, "kiem": kq.ds, "so": so,
                                 "n_dat": len(kq.ds) - n_hong, "n_hong": n_hong},
                                ensure_ascii=False, indent=1, default=str) + "\n", encoding="utf-8")
        print(f"[kiem_loc_quy_dao] ghi {p.relative_to(ad.GOC).as_posix()}")
    return 1 if n_hong else 0


if __name__ == "__main__":
    sys.exit(main())
