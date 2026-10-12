"""
kiem_pixel_to_xam.py -- R48E: kiem PIXEL CHI DOC cua 160 anh "to xam" trong mau 320 anh cua R48B (khoa `mau_yolo`
trong results/phase2/kc48_audit/tieu_chi_cong_bo.json, seed 48). Moi anh: so anh xuat data/yolo_bdd5_v1 voi anh goc
BDD train.

Vung tinh (tai suy DOC LAP tu JSON Scalabel goc theo quy tac 4-5 docstring src/bdd_yolo_export.py, khong goi exporter):
  V = hop cac vung crowd / distractor xe (to_xam), G = V tru hop box co nhan (phai xam), K = V giao hop box co nhan
  (box co nhan chong len vung, phai giu nguyen), O = ngoai V (phai giu nguyen). Phan "loi" cua moi vung bo bien JPEG:
  pixel ma cua so (2r+1)^2 quanh no (cat theo khung anh) khong cham pixel khac loai (G: chi G; O/K: khong co G).
Tieu chi, dung sai va nguong ghi o --phan cong_bo TRUOC khi tinh (khong ghi de); --phan tinh doc lai, ghi sha256.

KHONG sua / tao file trong data/: anh chi mo de doc; moi file ghi kiem bang `_duong_ghi` (chi results/phase2/kc48_audit/).

Lenh:
  C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/kiem_pixel_to_xam.py --phan cong_bo
  C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/kiem_pixel_to_xam.py --phan tinh
  C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/kiem_pixel_to_xam.py --phan chan_doan  (hau kiem K)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_du_lieu import (DISTRACTOR_BDD, LOP5, OUT, P, _duong_ghi, doc_csv, doc_json, ghi_csv,  # noqa: E402
                           ghi_json, git_head, nap_tieu_chi, sha256_file, vai_tro)

FILE_TC = OUT / "tieu_chi_pixel_to_xam.json"
FILE_CSV = OUT / "pixel_to_xam_tung_anh.csv"
FILE_TOM = OUT / "pixel_to_xam_tom_tat.json"
FILE_LOG = OUT / "pixel_to_xam_nhat_ky.jsonl"

TIEU_CHI = {
    "phien_ban": "kc48-pixel-to-xam-v1",
    "mau": ("dung 160 anh xu_ly == to_xam trong mau 320 anh cua R48B (tieu_chi_cong_bo.json khoa mau_yolo, seed 48): "
            "chon lai bang dung thu tuc phan_yolo cua src/kc48/audit_du_lieu.py va doi chieu tung (split, video, anh) "
            "voi results/phase2/kc48_audit/yolo_kiem_mau.csv; lech -> dung, khong tinh"),
    "giai_ma": "PIL Image.open(...).convert('RGB') cho ca anh goc va anh xuat; uint8; so cung kich thuoc",
    "vung": {
        "raster": "hinh chu nhat px [floor(y1):ceil(y2), floor(x1):ceil(x2)] sau khi cat vao khung; box < 1 px bo qua",
        "V": "hop vung to_xam (crowd thuoc 5 lop, other vehicle, trailer)",
        "G": "V tru hop box co nhan -> ky vong (114,114,114)",
        "K": "V giao hop box co nhan -> ky vong giu nguyen anh goc (doi tuong co nhan khong bi xoa)",
        "O": "ngoai V -> ky vong giu nguyen anh goc",
        "ban_kinh_loi_px": 8,
        "loi": ("G_loi: pixel G co cua so (2r+1)^2 (cat theo khung) toan G; O_loi / K_loi: pixel O / K co cua so khong "
                "chua pixel G. r = 8 px = mot khoi DCT 8x8; bien JPEG (ringing, chroma 4:2:0) nam trong dai nay, va quy "
                "uoc raster floor/ceil o bien khong anh huong phan loi"),
    },
    "xap_xi_xam": "max theo kenh |I_ra - 114| <= 6",
    "ly_do_dung_sai": ("exporter ghi JPEG chat luong 95 bang cv2 (src/bdd_yolo_export.py:574); bang luong tu hoa o q95 = "
                       "10 % bang chuan (buoc 1-12), khoi 8x8 phang 114 giu DC gan dung; 6 muc chua chenh lech bo giai "
                       "ma (cv2 luc ghi, PIL luc kiem) va lam tron YCbCr<->RGB"),
    "chi_so": {
        "ti_le_xam": "so pixel xap_xi_xam / so pixel vung",
        "mae": "trung binh |I_ra - I_goc| tren moi kenh moi pixel cua vung (thang 0-255)",
        "psnr_db": "10 log10(255^2 / MSE), MSE tren moi kenh moi pixel cua vung; MSE = 0 -> inf",
    },
    "nguong": {
        "T1_vung_to": "G_loi khac rong va ti_le_xam(I_ra, G_loi) >= 0.99",
        "T2_ngoai_vung": "O_loi: psnr_db >= 35 va mae <= 2.0",
        "T3_box_chong_vung": "neu K_loi khac rong: psnr_db >= 35 va mae <= 2.0; K_loi rong -> khong_ap_dung",
        "ket_qua_anh": ("dat = T1 dat va T2 dat va T3 in {dat, khong_ap_dung}; khong_dat = co tieu chi khong dat; "
                        "chua_ket_luan = G_loi rong (vung qua mong de tach khoi bien JPEG) va khong tieu chi nao khong dat"),
    },
    "bao_cao_them_khong_nguong": [
        "ti_le_xam(I_goc, G_loi): doi chung - anh goc da xam san o vung nay thi T1 khong phan biet duoc",
        "ti_le_xam(I_ra, G), mae / psnr tren O, K day du (gom dai bien)",
        "ti_le_xam tren K (anh ra va anh goc), so box co nhan cham V",
        "p99 |I_ra - I_goc| tren O_loi",
        "sha256 anh xuat so voi manifest; vung tai suy so voi file ignore (dung sai 0.006 px, nhu R48B)",
    ],
    "pham_vi": "chi 160 anh mau; KHONG ket luan cho toan bo 18.600 anh to xam cua data/yolo_bdd5_v1",
}


def phan_cong_bo() -> None:
    if FILE_TC.exists():
        raise SystemExit(f"{FILE_TC} da ton tai: khong ghi de tieu chi da cong bo")
    ghi_json(FILE_TC, {**TIEU_CHI, "tao_luc": dt.datetime.now().isoformat(timespec="seconds"), "git_head": git_head()})
    print("[cong_bo]", FILE_TC, sha256_file(FILE_TC))


# =========================================================================== mau
def chon_mau() -> tuple[list[dict], str]:
    """Lap lai dung thu tuc chon mau cua audit_du_lieu.phan_yolo (seed, tang, thu tu)."""
    tc, sha_tc = nap_tieu_chi()
    my = tc["mau_yolo"]
    rows = doc_csv(P["yolo"] / "manifest" / "anh.csv")
    tang = defaultdict(list)
    for r in rows:
        hiem = int(r.get("n_motorcycle") or 0) + int(r.get("n_bicycle") or 0) > 0
        tang[(r["split"], r["xu_ly"], int(hiem))].append(r)
    rng = np.random.default_rng(int(my["seed"]))
    mau = []
    for k in sorted(tang):
        L = tang[k]
        idx = rng.choice(len(L), size=min(len(L), int(my["so_moi_tang"])), replace=False)
        mau += [dict(L[i], _hiem=k[2]) for i in sorted(idx)]
    return mau, sha_tc


# =========================================================================== hinh hoc
def _rect(m: np.ndarray, b) -> None:
    x1, y1, x2, y2 = b
    m[int(math.floor(y1)):int(math.ceil(y2)), int(math.floor(x1)):int(math.ceil(x2))] = True


def dem_cua_so(m: np.ndarray, r: int) -> np.ndarray:
    """So pixel True cua m trong cua so (2r+1)^2 quanh moi pixel (cat theo khung), bang anh tich phan."""
    H, W = m.shape
    S = np.zeros((H + 1, W + 1), np.int64)
    S[1:, 1:] = np.cumsum(np.cumsum(m.astype(np.int64), 0), 1)
    y0 = np.clip(np.arange(H) - r, 0, H)[:, None]
    y1 = np.clip(np.arange(H) + r + 1, 0, H)[:, None]
    x0 = np.clip(np.arange(W) - r, 0, W)[None, :]
    x1 = np.clip(np.arange(W) + r + 1, 0, W)[None, :]
    return S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]


def ti_le_xam(im: np.ndarray, m: np.ndarray, tol: int):
    n = int(m.sum())
    if n == 0:
        return ""
    d = np.abs(im[m].astype(np.int16) - 114).max(axis=1)
    return round(float((d <= tol).mean()), 6)


def sai_so(a: np.ndarray, b: np.ndarray, m: np.ndarray) -> dict:
    if not m.any():
        return {"mae": "", "psnr": "", "p99": ""}
    d = a[m].astype(np.float64) - b[m].astype(np.float64)
    mse = float((d * d).mean())
    return {"mae": round(float(np.abs(d).mean()), 4),
            "psnr": "inf" if mse == 0 else round(10 * math.log10(255.0 ** 2 / mse), 3),
            "p99": round(float(np.percentile(np.abs(d), 99)), 2)}


def _psnr_ok(s: dict, nguong_psnr: float, nguong_mae: float) -> bool:
    return (s["psnr"] == "inf" or s["psnr"] >= nguong_psnr) and s["mae"] <= nguong_mae


# =========================================================================== tinh
def phan_tinh() -> None:
    from PIL import Image
    if not FILE_TC.exists():
        raise SystemExit("chua co tieu_chi_pixel_to_xam.json: chay --phan cong_bo truoc")
    tcp, sha_tcp = doc_json(FILE_TC), sha256_file(FILE_TC)
    r_loi = int(tcp["vung"]["ban_kinh_loi_px"])
    tol = 6
    if tcp["xap_xi_xam"] != "max theo kenh |I_ra - 114| <= 6" or r_loi != 8:
        raise SystemExit("tieu chi da cong bo khac hang so trong ma: dung")
    NG_T1, NG_PSNR, NG_MAE = 0.99, 35.0, 2.0

    mau, sha_tc = chon_mau()
    mau_xam = [r for r in mau if r["xu_ly"] == "to_xam"]
    ref = {(x["split"], x["video"], x["anh"]) for x in doc_csv(OUT / "yolo_kiem_mau.csv") if x["xu_ly"] == "to_xam"}
    got = {(r["split"], r["video"], r["anh_nguon"].replace("\\", "/").split("/")[-1]) for r in mau_xam}
    if len(mau) != 320 or len(mau_xam) != 160 or got != ref:
        raise SystemExit(f"mau khong khop R48B: n={len(mau)}, to_xam={len(mau_xam)}, khop_csv={got == ref}")

    id5 = {c: i for i, c in enumerate(LOP5)}
    cache: dict[str, dict] = {}
    out = []
    for r in mau_xam:
        v = r["video"]
        ten = r["anh_nguon"].replace("\\", "/").split("/")[-1]
        src, dst = P["bdd_anh_train"] / v / ten, P["yolo"] / r["anh_ra"]
        if v not in cache:
            if len(cache) > 50:
                cache.clear()
            cache[v] = {f["name"]: f for f in doc_json(P["bdd_nhan_train"] / f"{v}.json")}
        fr = cache[v][ten]
        with Image.open(src) as im:
            goc = np.asarray(im.convert("RGB"))
        with Image.open(dst) as im:
            ra = np.asarray(im.convert("RGB"))
        H, W = goc.shape[:2]
        kq = {"split": r["split"], "co_lop_hiem": r["_hiem"], "video": v, "anh": ten, "anh_ra": r["anh_ra"],
              "W": W, "H": H, "cung_kich_thuoc": int(ra.shape == goc.shape),
              "sha_anh_ra_khop_manifest": int(sha256_file(dst) == r["sha256_ra"])}
        if ra.shape != goc.shape:
            kq["ket_qua"] = "khong_dat"
            out.append(kq)
            continue
        vung, box = [], []
        for lb in fr.get("labels") or []:
            b = lb.get("box2d")
            if not b:
                continue
            at = lb.get("attributes") or {}
            vt = vai_tro(lb["category"], bool(at.get("crowd", False)), lb["category"] in DISTRACTOR_BDD)
            if vt == "lop_khac":
                continue
            x1, x2 = (max(0.0, min(float(W), float(b[k]))) for k in ("x1", "x2"))
            y1, y2 = (max(0.0, min(float(H), float(b[k]))) for k in ("y1", "y2"))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            (box if vt == "gan_nhan" else vung).append((x1, y1, x2, y2))
        pi = P["yolo"] / r["ignore_ra"] if r.get("ignore_ra") else None
        gv = sorted(tuple(float(u) for u in t.split()) for t in pi.read_text(encoding="utf-8").splitlines() if t) \
            if pi is not None and pi.exists() else []
        kq["vung_khop_file_ignore"] = int(len(gv) == len(vung) and all(
            max(abs(u - w) for u, w in zip(a, b)) <= 0.006 for a, b in zip(sorted(vung), gv)))

        mV, mB = np.zeros((H, W), bool), np.zeros((H, W), bool)
        for b in vung:
            _rect(mV, b)
        n_cham = 0
        for b in box:
            mb = np.zeros((H, W), bool)
            _rect(mb, b)
            n_cham += int((mb & mV).any())
            mB |= mb
        mG, mK, mO = mV & ~mB, mV & mB, ~mV
        g_quanh = dem_cua_so(mG, r_loi)
        mG_loi = mG & (dem_cua_so(~mG, r_loi) == 0)
        mO_loi, mK_loi = mO & (g_quanh == 0), mK & (g_quanh == 0)
        kq.update({"n_vung": len(vung), "n_box_nhan": len(box), "n_box_cham_vung": n_cham,
                   "px_V": int(mV.sum()), "px_G": int(mG.sum()), "px_K": int(mK.sum()),
                   "px_G_loi": int(mG_loi.sum()), "px_K_loi": int(mK_loi.sum()), "px_O_loi": int(mO_loi.sum()),
                   "ti_le_xam_G_loi_ra": ti_le_xam(ra, mG_loi, tol), "ti_le_xam_G_loi_goc": ti_le_xam(goc, mG_loi, tol),
                   "ti_le_xam_G_ra": ti_le_xam(ra, mG, tol)})
        for ten_v, m in (("O_loi", mO_loi), ("O", mO), ("K_loi", mK_loi), ("K", mK)):
            s = sai_so(ra, goc, m)
            kq[f"mae_{ten_v}"], kq[f"psnr_{ten_v}"] = s["mae"], s["psnr"]
            if ten_v == "O_loi":
                kq["p99_absdiff_O_loi"] = s["p99"]
            if ten_v == "K_loi":
                sK = s
        sO = sai_so(ra, goc, mO_loi)
        kq["ti_le_xam_K_ra"], kq["ti_le_xam_K_goc"] = ti_le_xam(ra, mK, tol), ti_le_xam(goc, mK, tol)

        kq["T1"] = "khong_ap_dung" if not mG_loi.any() else \
            ("dat" if kq["ti_le_xam_G_loi_ra"] >= NG_T1 else "khong_dat")
        kq["T2"] = "dat" if mO_loi.any() and _psnr_ok(sO, NG_PSNR, NG_MAE) else "khong_dat"
        kq["T3"] = "khong_ap_dung" if not mK_loi.any() else ("dat" if _psnr_ok(sK, NG_PSNR, NG_MAE) else "khong_dat")
        t = (kq["T1"], kq["T2"], kq["T3"])
        kq["ket_qua"] = "khong_dat" if "khong_dat" in t else ("chua_ket_luan" if kq["T1"] == "khong_ap_dung" else "dat")
        out.append(kq)

    ghi_csv(FILE_CSV, out)

    def pv(key, rows):
        x = [float(q[key]) for q in rows if q.get(key) not in ("", None, "inf")]
        n_inf = sum(1 for q in rows if q.get(key) == "inf")
        if not x:
            return {"n": 0, "n_inf": n_inf}
        return {"n": len(x), "n_inf": n_inf, "min": round(min(x), 4),
                **{f"p{p}": round(float(np.percentile(x, p)), 4) for p in (5, 50, 95)}, "max": round(max(x), 4)}

    tom = {
        "chay_luc": dt.datetime.now().isoformat(timespec="seconds"), "git_head": git_head(),
        "tieu_chi_pixel_sha256": sha_tcp, "tieu_chi_pixel_tao_luc": tcp.get("tao_luc"),
        "tieu_chi_mau_r48b_sha256": sha_tc,
        "n_anh": len(out), "theo_tang": dict(Counter(f"{q['split']}|hiem={q['co_lop_hiem']}" for q in out)),
        "ket_qua": dict(Counter(q["ket_qua"] for q in out)),
        "T1": dict(Counter(q.get("T1") for q in out)), "T2": dict(Counter(q.get("T2") for q in out)),
        "T3": dict(Counter(q.get("T3") for q in out)),
        "ket_qua_theo_tang": {k: dict(Counter(q["ket_qua"] for q in out if f"{q['split']}|hiem={q['co_lop_hiem']}" == k))
                              for k in sorted({f"{q['split']}|hiem={q['co_lop_hiem']}" for q in out})},
        "cung_kich_thuoc": sum(int(q["cung_kich_thuoc"]) for q in out),
        "sha_anh_ra_khop_manifest": sum(int(q["sha_anh_ra_khop_manifest"]) for q in out),
        "vung_khop_file_ignore": sum(int(q.get("vung_khop_file_ignore", 0)) for q in out),
        "anh_co_box_cham_vung": sum(1 for q in out if int(q.get("n_box_cham_vung", 0)) > 0),
        "tong_box_cham_vung": sum(int(q.get("n_box_cham_vung", 0)) for q in out),
        "anh_co_K_loi": sum(1 for q in out if int(q.get("px_K_loi", 0)) > 0),
        "tong_px": {k: sum(int(q.get(k, 0)) for q in out) for k in ("px_V", "px_G", "px_K", "px_G_loi", "px_K_loi")},
        "phan_vi": {k: pv(k, out) for k in ("ti_le_xam_G_loi_ra", "ti_le_xam_G_loi_goc", "ti_le_xam_G_ra",
                                            "mae_O_loi", "psnr_O_loi", "p99_absdiff_O_loi", "mae_O", "psnr_O",
                                            "mae_K_loi", "psnr_K_loi", "mae_K", "psnr_K", "ti_le_xam_K_ra",
                                            "ti_le_xam_K_goc")},
        "anh_khong_dat": [{k: q.get(k) for k in ("split", "video", "anh", "T1", "T2", "T3", "ti_le_xam_G_loi_ra",
                                                 "psnr_O_loi", "mae_O_loi", "psnr_K_loi", "mae_K_loi")}
                          for q in out if q["ket_qua"] != "dat"],
        "pham_vi": TIEU_CHI["pham_vi"],
    }
    ghi_json(FILE_TOM, tom)
    print("[tinh]", json.dumps({k: tom[k] for k in ("n_anh", "ket_qua", "T1", "T2", "T3")}, ensure_ascii=False))


# =========================================================================== chan doan (hau kiem)
FILE_CD_CSV = OUT / "pixel_to_xam_chan_doan_K.csv"
FILE_CD_TOM = OUT / "pixel_to_xam_chan_doan_K.json"


def phan_chan_doan() -> None:
    """HAU KIEM, viet SAU khi thay ket qua --phan tinh (3 anh T3 khong_dat); KHONG doi T1-T3 / ket_qua da cong bo.
    Cau hoi: sai so tren K_loi den tu xu ly vung (exporter dong vao box co nhan) hay tu nen JPEG lai?
    Doi chung 1 (nen lai): anh goc nen lai trong bo nho bang cv2 q95, KHONG to xam -> mae tren cung K_loi.
    Doi chung 2 (cung anh): mae cua anh ra tren pixel box co nhan nam ngoai V (B & O_loi)."""
    import cv2
    from PIL import Image
    tom_cu = doc_json(FILE_TOM)
    rows = [r for r in doc_csv(FILE_CSV) if r.get("px_K_loi") and int(r["px_K_loi"]) > 0]
    mau = {(r["split"], r["video"], r["anh_nguon"].replace("\\", "/").split("/")[-1]): r for r in chon_mau()[0]}
    r_loi = int(doc_json(FILE_TC)["vung"]["ban_kinh_loi_px"])
    out = []
    for q in rows:
        r = mau[(q["split"], q["video"], q["anh"])]
        v, ten = q["video"], q["anh"]
        src, dst = P["bdd_anh_train"] / v / ten, P["yolo"] / r["anh_ra"]
        fr = {f["name"]: f for f in doc_json(P["bdd_nhan_train"] / f"{v}.json")}[ten]
        with Image.open(src) as im:
            goc = np.asarray(im.convert("RGB"))
        with Image.open(dst) as im:
            ra = np.asarray(im.convert("RGB"))
        ok, enc = cv2.imencode(".jpg", cv2.imread(str(src), cv2.IMREAD_COLOR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise SystemExit(f"cv2 khong nen duoc {src}")
        nen_lai = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        H, W = goc.shape[:2]
        mV, mB = np.zeros((H, W), bool), np.zeros((H, W), bool)
        for lb in fr.get("labels") or []:
            b = lb.get("box2d")
            if not b:
                continue
            at = lb.get("attributes") or {}
            vt = vai_tro(lb["category"], bool(at.get("crowd", False)), lb["category"] in DISTRACTOR_BDD)
            if vt == "lop_khac":
                continue
            x1, x2 = (max(0.0, min(float(W), float(b[k]))) for k in ("x1", "x2"))
            y1, y2 = (max(0.0, min(float(H), float(b[k]))) for k in ("y1", "y2"))
            if x2 - x1 < 1.0 or y2 - y1 < 1.0:
                continue
            _rect(mB if vt == "gan_nhan" else mV, (x1, y1, x2, y2))
        mG = mV & ~mB
        g_quanh = dem_cua_so(mG, r_loi)
        mK_loi, mBO_loi = mV & mB & (g_quanh == 0), mB & ~mV & (g_quanh == 0)
        if int(mK_loi.sum()) != int(q["px_K_loi"]):
            raise SystemExit(f"K_loi tai suy lech --phan tinh o {ten}")
        sK, sN, sB = sai_so(ra, goc, mK_loi), sai_so(nen_lai, goc, mK_loi), sai_so(ra, goc, mBO_loi)
        out.append({"split": q["split"], "video": v, "anh": ten, "T3": q["T3"], "px_K_loi": q["px_K_loi"],
                    "mae_K_loi_ra": sK["mae"], "psnr_K_loi_ra": sK["psnr"],
                    "mae_K_loi_nen_lai_goc": sN["mae"], "psnr_K_loi_nen_lai_goc": sN["psnr"],
                    "mae_K_loi_ra_tru_nen_lai": round(sK["mae"] - sN["mae"], 4),
                    "mae_K_loi_ra_vs_nen_lai": sai_so(ra, nen_lai, mK_loi)["mae"],
                    "px_box_ngoai_V_loi": int(mBO_loi.sum()), "mae_box_ngoai_V_loi_ra": sB["mae"]})
    ghi_csv(FILE_CD_CSV, out)
    d = [float(o["mae_K_loi_ra_tru_nen_lai"]) for o in out]
    tom = {"chay_luc": dt.datetime.now().isoformat(timespec="seconds"), "git_head": git_head(),
           "loai": "hau_kiem_sau_khi_xem_ket_qua; khong doi ket_qua T1-T3 da cong bo",
           "tom_tat_tinh_sha256": sha256_file(FILE_TOM), "tom_tat_tinh_chay_luc": tom_cu.get("chay_luc"),
           "cv2": cv2.__version__, "n_anh_co_K_loi": len(out),
           "chenh_mae_ra_tru_nen_lai": {"min": round(min(d), 4), "p50": round(float(np.median(d)), 4),
                                        "max": round(max(d), 4), "n_abs_le_0_1": sum(abs(x) <= 0.1 for x in d)},
           "anh_T3_khong_dat": [o for o in out if o["T3"] == "khong_dat"]}
    ghi_json(FILE_CD_TOM, tom)
    print("[chan_doan]", json.dumps({k: tom[k] for k in ("n_anh_co_K_loi", "chenh_mae_ra_tru_nen_lai")}))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R48E kiem pixel anh to xam (chi doc)")
    ap.add_argument("--phan", required=True, choices=["cong_bo", "tinh", "chan_doan"])
    a = ap.parse_args(argv)
    t0, bat_dau = time.time(), dt.datetime.now().isoformat(timespec="seconds")
    {"cong_bo": phan_cong_bo, "tinh": phan_tinh, "chan_doan": phan_chan_doan}[a.phan]()
    p = _duong_ghi(FILE_LOG)
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"phan": a.phan, "bat_dau": bat_dau, "giay": round(time.time() - t0, 1),
                            "git_head": git_head(), "argv": sys.argv[1:]}, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
