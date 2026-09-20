"""
nano_parity.py -- So hai ket qua tracking (MOTChallenge) voi nhau: ban CAN KIEM so voi ban THAM CHIEU.

Dung cho: (1) Jetson Nano so voi PC; (2) chay TRUC TIEP tren video so voi REPLAY tu detection cache
(moc he thong PC). Neu ket qua tracking khac nhau thi so toc do khong so sanh duoc.

TOLERANCE DUOC CONG BO TRUOC KHI SO (hang so ngay duoi day, in ra dau moi lan chay):

  bbox      : sai so tuyet doi <= TOL_BBOX pixel cho tung toa do (x, y, w, h)
  confidence: sai so tuyet doi <= TOL_CONF
  LIEN KET ID: **KHONG co tolerance** -- phai khop TUYET DOI.
              Association la quyet dinh ROI RAC: mot khac biet nghia la tracker da ghep khac,
              khong phai "sai so so hoc". Kiem theo HAI muc:
                (a) gia tri ID tho phai bang nhau -- ghep hang bang KHOA (frame, id);
                (b) PHAN HOACH track phai tuong duong -- ghep hinh hoc 1-1 theo tolerance trong
                    tung frame, roi kiem SONG ANH ID CO DINH tren TOAN sequence.
              (b) yeu hon (a): neu chi danh so lech thi (a) fail nhung (b) pass -- bao cao tach
              rieng de biet dung loai khac biet.

CACH GHEP HANG (review 8bab154, F2 -- ban truoc sap theo (frame, x, y) roi so theo chi so hang:
SAI, vi toa do duoc phep lech toi TOL_BBOX nen thu tu tuyet doi cua toa do khong con la khoa; hai
box gan nhau doi cho sau sai so nho, hai box trung (x, y) phu thuoc thu tu dong trong file):
  - Muc (a): khoa (frame, id). Tap khoa hai ben phai BANG NHAU va KHONG co khoa trung.
  - Muc (b): trong moi frame, ghep 1-1 bang Hungarian tren khoang cach Chebyshev cua 4 toa do,
    chi chap nhan cap co khoang cach <= TOL_BBOX. Neu mot box co >1 ung vien trong tolerance thi
    ghi nhan MO HO (khong tu chon). Song anh ID duoc suy tu TOAN BO cac cap da ghep, khong anh
    xa lai tung frame.

VI SAO bbox co tolerance ma ID thi khong: hai kien truc (x86-64 va aarch64) co the dung BLAS/libm
khac nhau, nen phep toan dau phay dong co the lech o chu so cuoi. Lech do chap nhan duoc mien la
khong doi quyet dinh ghep. Neu no doi quyet dinh ghep thi (a)/(b) se bat duoc.

    python src/nano_parity.py --ref <thu muc tham chieu> --test <thu muc can kiem> [--videos v1 v2]
    python src/nano_parity.py --ref data/processed/trackers/X/pc-cv --test .../nano-cv --json out.json

Tra exit 0 neu DAT ca ba muc, 1 neu khong.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- TOLERANCE (cong bo truoc)
TOL_BBOX = 1e-3          # pixel, sai so tuyet doi tung toa do
TOL_CONF = 1e-6          # diem confidence
TOL_ID = None            # KHONG co tolerance cho lien ket ID

COLS = ("frame", "id", "x", "y", "w", "h", "conf")


def doc_mot(p: Path) -> np.ndarray:
    """Doc file MOTChallenge -> mang (n, 7). File rong -> mang (0, 7)."""
    if not p.exists():
        raise FileNotFoundError(f"khong co {p}")
    txt = p.read_text(encoding="utf-8").strip()
    if not txt:
        return np.zeros((0, 7), dtype=np.float64)
    rows = [[float(x) for x in ln.split(",")[:7]] for ln in txt.splitlines() if ln.strip()]
    return np.asarray(rows, dtype=np.float64).reshape(-1, 7)


# --------------------------------------------------------------------------- muc (a): khoa (frame, id)
def _theo_khoa(a: np.ndarray) -> tuple[dict, list]:
    """{(frame, id): hang} va danh sach khoa TRUNG (mot tracker khong duoc ghi hai hang cung id
    trong cung frame -- neu co thi file khong hop le de so)."""
    d, trung = {}, []
    for i in range(len(a)):
        k = (int(a[i, 0]), int(a[i, 1]))
        if k in d:
            trung.append(k)
        d[k] = a[i]
    return d, trung


def so_theo_id_tho(a: np.ndarray, b: np.ndarray) -> dict:
    """Muc (a): ghep bang khoa (frame, id) roi so bbox/conf. Tra ve dict ket qua."""
    da, trung_a = _theo_khoa(a)
    db, trung_b = _theo_khoa(b)
    r: dict = {"khoa_trung_ref": len(trung_a), "khoa_trung_test": len(trung_b),
               "max_bbox": 0.0, "max_conf": 0.0}
    if trung_a or trung_b:
        r.update(id_tho_khop=False, ly_do=f"khoa (frame,id) TRUNG: ref {len(trung_a)}, test {len(trung_b)}"
                 " -- file khong hop le de so")
        return r
    ka, kb = set(da), set(db)
    chi_ref, chi_test = ka - kb, kb - ka
    r.update(n_chi_ref=len(chi_ref), n_chi_test=len(chi_test))
    if chi_ref or chi_test:
        r.update(id_tho_khop=False,
                 ly_do=f"tap khoa (frame,id) khac: {len(chi_ref)} chi co o ref, {len(chi_test)} chi co o test")
        # van tinh lech tren phan giao de bao cao
        chung = sorted(ka & kb)
    else:
        chung = sorted(ka)
        r["id_tho_khop"] = True
    if chung:
        A = np.stack([da[k] for k in chung]); B = np.stack([db[k] for k in chung])
        r["max_bbox"] = float(np.max(np.abs(A[:, 2:6] - B[:, 2:6])))
        r["max_conf"] = float(np.max(np.abs(A[:, 6] - B[:, 6])))
    else:
        r["max_bbox"] = r["max_conf"] = 0.0
    return r


# --------------------------------------------------------------------------- muc (b): ghep hinh hoc + song anh
def _ghep_hinh_hoc_mot_frame(A: np.ndarray, B: np.ndarray) -> tuple[list, int, str]:
    """Ghep 1-1 cac hang cua mot frame theo khoang cach Chebyshev 4 toa do, nguong TOL_BBOX.

    Tra ve (cap [(i_ref, j_test)], so_box_mo_ho, ly_do_khong_ghep_duoc).
    Mo ho = mot box co >1 ung vien trong tolerance (hinh hoc khong phan biet duoc).
    """
    if len(A) != len(B):
        return [], 0, f"so hang khac nhau trong frame: {len(A)} vs {len(B)}"
    if len(A) == 0:
        return [], 0, ""
    D = np.max(np.abs(A[:, None, 2:6] - B[None, :, 2:6]), axis=2)       # (n, n) Chebyshev
    ung_vien = (D <= TOL_BBOX)
    mo_ho = int(np.sum(ung_vien.sum(axis=1) > 1) + np.sum(ung_vien.sum(axis=0) > 1))
    from scipy.optimize import linear_sum_assignment
    ri, cj = linear_sum_assignment(D)
    cap = [(int(i), int(j)) for i, j in zip(ri, cj) if D[i, j] <= TOL_BBOX]
    if len(cap) < len(A):
        return cap, mo_ho, f"{len(A) - len(cap)}/{len(A)} box khong co doi tac trong tolerance"
    return cap, mo_ho, ""


def so_phan_hoach(a: np.ndarray, b: np.ndarray) -> dict:
    """Muc (b): ghep hinh hoc tung frame, roi kiem song anh ID co dinh tren toan sequence."""
    fa = defaultdict(list); fb = defaultdict(list)
    for i in range(len(a)):
        fa[int(a[i, 0])].append(i)
    for j in range(len(b)):
        fb[int(b[j, 0])].append(j)
    if set(fa) != set(fb):
        return {"phan_hoach_khop": False, "ly_do": "danh sach frame khac nhau"}
    anh_xa: dict[int, set] = defaultdict(set)      # id_ref -> {id_test}
    nguoc: dict[int, set] = defaultdict(set)       # id_test -> {id_ref}
    n_mo_ho, loi_frame, mx_bbox, mx_conf = 0, [], 0.0, 0.0
    for f in sorted(fa):
        A, B = a[fa[f]], b[fb[f]]
        cap, mo_ho, ly_do = _ghep_hinh_hoc_mot_frame(A, B)
        n_mo_ho += mo_ho
        if ly_do:
            loi_frame.append((f, ly_do))
            continue
        for i, j in cap:
            anh_xa[int(A[i, 1])].add(int(B[j, 1]))
            nguoc[int(B[j, 1])].add(int(A[i, 1]))
            mx_bbox = max(mx_bbox, float(np.max(np.abs(A[i, 2:6] - B[j, 2:6]))))
            mx_conf = max(mx_conf, float(abs(A[i, 6] - B[j, 6])))
    r: dict = {"n_frame_mo_ho_hinh_hoc": n_mo_ho, "n_frame_loi_ghep": len(loi_frame),
               "max_bbox_hinh_hoc": mx_bbox, "max_conf_hinh_hoc": mx_conf}
    if loi_frame:
        f, why = loi_frame[0]
        r.update(phan_hoach_khop=False, ly_do=f"{len(loi_frame)} frame khong ghep hinh hoc duoc"
                 f" (vd frame {f}: {why})")
        return r
    xau = [k for k, v in anh_xa.items() if len(v) != 1] + [k for k, v in nguoc.items() if len(v) != 1]
    if xau:
        r.update(phan_hoach_khop=False,
                 ly_do=f"song anh ID KHONG co dinh tren sequence: {len(xau)} ID anh xa toi >1 ID ben kia"
                 " (tracker da ghep khac)")
        return r
    r.update(phan_hoach_khop=True, n_track=len(anh_xa),
             chi_danh_so_lech=any(k != next(iter(v)) for k, v in anh_xa.items()))
    if n_mo_ho:
        r["ghi_chu"] = (f"{n_mo_ho} box co >1 ung vien hinh hoc trong tolerance -- song anh van co dinh "
                        "nhung ket luan phan hoach o cac box do dua vao Hungarian, khong phai hinh hoc thuan")
    return r


# --------------------------------------------------------------------------- mot video
def so_mot_video(ref: Path, test: Path) -> dict:
    a, b = doc_mot(ref), doc_mot(test)
    r: dict = {"video": ref.stem, "n_ref": len(a), "n_test": len(b)}
    if len(a) != len(b):
        r.update(dat=False, ly_do=f"so hang khac nhau: {len(a)} vs {len(b)}")
        return r
    if len(a) == 0:
        r.update(dat=True, max_bbox=0.0, max_conf=0.0, id_tho_khop=True, phan_hoach_khop=True)
        return r

    ra = so_theo_id_tho(a, b)
    khoa_khop = bool(ra.get("id_tho_khop"))
    a_dat = khoa_khop and ra["max_bbox"] <= TOL_BBOX and ra["max_conf"] <= TOL_CONF
    r.update(max_bbox=ra["max_bbox"], max_conf=ra["max_conf"])

    # (b) chay khi (a) KHONG dat TRON VEN. Luu y: hoan doi ID (1<->2) lam tap khoa (frame,id) VAN
    # bang nhau nhung noi dung duoi khoa khac -> chi nhin (a) se bao nham "bbox lech"; (b) moi noi
    # duoc do la doi cach danh so hay doi lien ket.
    if a_dat:
        rb = {"phan_hoach_khop": True, "max_bbox_hinh_hoc": ra["max_bbox"],
              "max_conf_hinh_hoc": ra["max_conf"]}
    else:
        rb = so_phan_hoach(a, b)
    ph_ok = bool(rb.get("phan_hoach_khop"))
    r.update(phan_hoach_khop=ph_ok, phan_hoach_ly_do=rb.get("ly_do", ""))
    for k in ("n_frame_mo_ho_hinh_hoc", "chi_danh_so_lech", "ghi_chu", "max_bbox_hinh_hoc",
              "max_conf_hinh_hoc"):
        if k in rb:
            r[k] = rb[k]

    # id_tho_khop = khoa khop VA noi dung duoi khoa khop; neu (b) chi ra "chi danh so lech" thi (a)
    # that bai vi DANH SO, khong phai vi bbox -> bao dung loai loi
    danh_so_lech = ph_ok and (rb.get("chi_danh_so_lech") or not khoa_khop)
    r["id_tho_khop"] = bool(a_dat) if not danh_so_lech else False
    bbox_de_xet = rb.get("max_bbox_hinh_hoc", r["max_bbox"]) if ph_ok else r["max_bbox"]
    conf_de_xet = rb.get("max_conf_hinh_hoc", r["max_conf"]) if ph_ok else r["max_conf"]

    r["dat"] = bool(a_dat and ph_ok)
    if not r["dat"]:
        vi = []
        if danh_so_lech:
            vi.append("ID tho KHAC nhung PHAN HOACH tuong duong (song anh co dinh) => chi lech cach "
                      f"danh so; bbox lech hinh hoc {bbox_de_xet:.3e}")
        else:
            if bbox_de_xet > TOL_BBOX:
                vi.append(f"bbox lech {bbox_de_xet:.3e} > {TOL_BBOX:.0e}")
            if conf_de_xet > TOL_CONF:
                vi.append(f"conf lech {conf_de_xet:.3e} > {TOL_CONF:.0e}")
            if not khoa_khop:
                vi.append("ID tho: " + ra.get("ly_do", "khac"))
        if not ph_ok:
            vi.append(f"phan hoach track KHAC: {r['phan_hoach_ly_do']}")
        r["ly_do"] = "; ".join(vi) or "khong dat"
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="So hai ket qua tracking (can kiem vs tham chieu)")
    ap.add_argument("--ref", required=True, help="thu muc ban THAM CHIEU, chua data/*.txt hoac *.txt")
    ap.add_argument("--test", required=True, help="thu muc ban CAN KIEM")
    ap.add_argument("--videos", nargs="*", default=None, help="chi so cac video nay")
    ap.add_argument("--json", default=None, help="ghi ket qua ra file JSON")
    a = ap.parse_args()

    def data_dir(p: str) -> Path:
        d = Path(p)
        return d / "data" if (d / "data").is_dir() else d

    ref, test = data_dir(a.ref), data_dir(a.test)
    print("=" * 78)
    print("TOLERANCE DA CONG BO (truoc khi so):")
    print(f"  bbox       : |delta| <= {TOL_BBOX:.0e} pixel moi toa do")
    print(f"  confidence : |delta| <= {TOL_CONF:.0e}")
    print("  lien ket ID: KHONG co tolerance -- (a) khoa (frame,id) phai bang nhau; "
          "(b) phan hoach track tuong duong qua song anh ID co dinh")
    print("=" * 78)
    print(f"tham chieu: {ref}")
    print(f"can kiem  : {test}\n")

    vids = a.videos or sorted(p.stem for p in ref.glob("*.txt"))
    if not vids:
        print(f"[LOI] khong co file .txt nao trong {ref}")
        return 1

    ket = []
    for v in vids:
        try:
            ket.append(so_mot_video(ref / f"{v}.txt", test / f"{v}.txt"))
        except FileNotFoundError as e:
            ket.append({"video": v, "dat": False, "ly_do": str(e)})

    print(f"{'video':<28} {'hang':>7} {'max|d bbox|':>12} {'max|d conf|':>12}  ket qua")
    print("-" * 78)
    for r in ket:
        mb = f"{r['max_bbox']:.3e}" if "max_bbox" in r else "-"
        mc = f"{r['max_conf']:.3e}" if "max_conf" in r else "-"
        print(f"{r['video']:<28} {r.get('n_ref', '-'):>7} {mb:>12} {mc:>12}  "
              f"{'DAT' if r.get('dat') else 'KHONG DAT: ' + r.get('ly_do', '?')}")
        if r.get("ghi_chu"):
            print(f"{'':<28} ghi chu: {r['ghi_chu']}")

    n_dat = sum(1 for r in ket if r.get("dat"))
    print("-" * 78)
    print(f"KET QUA: {n_dat}/{len(ket)} video DAT")
    if n_dat != len(ket):
        print("\n=> KHONG duoc do toc do khi ket qua tracking con khac: so do se khong so sanh duoc.")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(
            {"tolerance": {"bbox": TOL_BBOX, "conf": TOL_CONF, "id": TOL_ID},
             "ref": str(ref), "test": str(test), "n_dat": n_dat, "n_video": len(ket),
             "ket_qua": ket}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"-> {a.json}")
    return 0 if n_dat == len(ket) else 1


if __name__ == "__main__":
    raise SystemExit(main())
