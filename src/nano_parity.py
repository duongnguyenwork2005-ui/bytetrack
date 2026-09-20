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
  - Muc (b): trong moi frame, tap UNG VIEN = cac cap (ref, test) co Chebyshev 4 toa do <= TOL_BBOX
    va |d conf| <= TOL_CONF. Song anh ID xay tu cac cap KHONG MO HO (1-1 duy nhat) roi lan truyen
    bang rang buoc TOAN SEQUENCE; box con nhieu ung vien ma rang buoc khong phan biet duoc ->
    trang thai MO HO (khong PASS, khong FAIL). KHONG dung Hungarian de chon tuy y (review 8abb368).
    Ket qua khong phu thuoc thu tu dong trong file.

CONG CU NAY CHI CO HAI FILE TRACKING: no bao dieu QUAN SAT duoc (bbox/conf lech, khoa (frame,id),
phan hoach khop/khac/mo ho). No KHONG ket luan nguyen nhan (detector hay tracker) -- muon biet phai
kiem detection dau vao bang phep kiem rieng (vd so npz cua hai lan chay).

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
def _ung_vien_mot_frame(A: np.ndarray, B: np.ndarray) -> "np.ndarray | None":
    """Ma tran ung vien (n_ref, n_test): True khi khoang cach Chebyshev 4 toa do <= TOL_BBOX.
    Chi dung HINH HOC de dinh danh detection; confidence la dai luong quan sat RIENG (bao o muc (a)),
    khong tham gia ghep -- neu khong, chi lech conf da lam "khong ghep duoc" va che mat ket luan
    phan hoach. None neu so hang khac nhau."""
    if len(A) != len(B):
        return None
    if len(A) == 0:
        return np.zeros((0, 0), dtype=bool)
    D = np.max(np.abs(A[:, None, 2:6] - B[None, :, 2:6]), axis=2)
    return D <= TOL_BBOX


def so_phan_hoach(a: np.ndarray, b: np.ndarray) -> dict:
    """Muc (b): phan hoach track co tuong duong qua MOT song anh ID co dinh tren toan sequence?

    (review 8abb368, PC-F3) KHONG dung Hungarian de chon giua nhieu ung vien: lua chon tuy y cua no
    khong phai bang chung. Thay vao do:
      1. Tim cac cap (ref_id, test_id) KHONG MO HO: box ref co dung 1 ung vien va box test do cung
         co dung 1 ung vien (trong frame do) -> mo neo cua song anh.
      2. Lan truyen bang RANG BUOC TOAN SEQUENCE: voi box con mo ho, neu ref_id da co anh xa
         (tu neo) va anh xa do nam trong tap ung vien cua box -> chot. Lap den khi khong doi.
      3. Con box chua chot -> trang thai `mo_ho` (khong DAT, khong FAIL): du lieu hai file khong
         du de phan biet. Khong bien mo ho thanh PASS.
      4. Neu anh xa thu duoc khong phai song anh (mot ID ung voi >1 ID ben kia) -> phan hoach KHAC.
    Ket qua KHONG phu thuoc thu tu dong trong file: chi dung tap ung vien va tap ID.

    Tra ve dict co `trang_thai` in {"khop", "khac", "mo_ho", "loi_ghep"} va cac truong mo ta.
    """
    fa = defaultdict(list); fb = defaultdict(list)
    for i in range(len(a)):
        fa[int(a[i, 0])].append(i)
    for j in range(len(b)):
        fb[int(b[j, 0])].append(j)
    if set(fa) != set(fb):
        return {"trang_thai": "loi_ghep", "phan_hoach_khop": False, "ly_do": "danh sach frame khac nhau"}

    # --- tap ung vien theo frame ---
    frames = []
    loi_frame = []
    for f in sorted(fa):
        A, B = a[fa[f]], b[fb[f]]
        M = _ung_vien_mot_frame(A, B)
        if M is None:
            loi_frame.append((f, f"so hang khac nhau trong frame: {len(A)} vs {len(B)}"))
            continue
        if len(A) and (not M.any(axis=1).all() or not M.any(axis=0).all()):
            n = int((~M.any(axis=1)).sum() + (~M.any(axis=0)).sum())
            loi_frame.append((f, f"{n} box khong co doi tac trong tolerance"))
            continue
        frames.append((f, A, B, M))
    if loi_frame:
        f, why = loi_frame[0]
        return {"trang_thai": "loi_ghep", "phan_hoach_khop": False, "n_frame_loi_ghep": len(loi_frame),
                "ly_do": f"{len(loi_frame)} frame khong ghep hinh hoc duoc (vd frame {f}: {why})"}

    # --- 1. neo khong mo ho ---
    anh_xa: dict[int, int] = {}          # ref_id -> test_id
    nguoc: dict[int, int] = {}           # test_id -> ref_id
    xung_dot = []

    def them(r_id: int, t_id: int) -> None:
        if anh_xa.get(r_id, t_id) != t_id or nguoc.get(t_id, r_id) != r_id:
            xung_dot.append((r_id, t_id, anh_xa.get(r_id), nguoc.get(t_id)))
        anh_xa.setdefault(r_id, t_id); nguoc.setdefault(t_id, r_id)

    n_box, n_mo_ho_ban_dau = 0, 0
    for f, A, B, M in frames:
        for i in range(len(A)):
            n_box += 1
            js = np.flatnonzero(M[i])
            if len(js) == 1 and M[:, js[0]].sum() == 1:
                them(int(A[i, 1]), int(B[js[0], 1]))
            else:
                n_mo_ho_ban_dau += 1

    # --- 2. lan truyen rang buoc toan sequence ---
    chua_chot = []
    doi = True
    while doi:
        doi = False
        chua_chot = []
        for f, A, B, M in frames:
            for i in range(len(A)):
                r_id = int(A[i, 1])
                js = np.flatnonzero(M[i])
                if len(js) == 1 and M[:, js[0]].sum() == 1:
                    continue                                    # da la neo
                ung = {int(B[j, 1]) for j in js}
                if r_id in anh_xa and anh_xa[r_id] in ung:
                    continue                                    # nhat quan voi anh xa da co
                if r_id in anh_xa and anh_xa[r_id] not in ung:
                    xung_dot.append((r_id, None, anh_xa[r_id], None))
                    continue
                # ref_id chua co anh xa: neu trong tap ung vien chi con DUNG 1 test_id chua bi chiem -> chot
                con = [t for t in ung if t not in nguoc]
                if len(con) == 1:
                    them(r_id, con[0]); doi = True
                else:
                    chua_chot.append((f, r_id, sorted(ung)))

    r: dict = {"n_box": n_box, "n_box_mo_ho_ban_dau": n_mo_ho_ban_dau, "n_box_chua_chot": len(chua_chot),
               "n_track_ref": len({int(x) for x in a[:, 1]}), "n_track_test": len({int(x) for x in b[:, 1]})}
    if xung_dot:
        r.update(trang_thai="khac", phan_hoach_khop=False,
                 ly_do=f"song anh ID KHONG co dinh tren sequence: {len(xung_dot)} xung dot "
                       "(mot ID ung voi >1 ID ben kia) -> phan hoach track KHAC")
        return r
    if chua_chot:
        f0, rid0, ung0 = chua_chot[0]
        r.update(trang_thai="mo_ho", phan_hoach_khop=None,
                 ly_do=f"MO HO: {len(chua_chot)}/{n_box} box co nhieu ung vien hinh hoc trong tolerance va "
                       f"rang buoc toan sequence khong phan biet duoc (vd frame {f0}, ref id {rid0} <-> "
                       f"test id {ung0}). Hai file KHONG du du lieu de ket luan phan hoach.")
        return r
    r.update(trang_thai="khop", phan_hoach_khop=True, n_track=len(anh_xa),
             chi_danh_so_lech=any(k != v for k, v in anh_xa.items()))
    return r


# --------------------------------------------------------------------------- mot video
def so_mot_video(ref: Path, test: Path) -> dict:
    """So mot video. Bao RIENG tung dieu quan sat duoc; KHONG suy nguyen nhan.

    `dat` (PASS day du) chi khi: khoa (frame,id) bang nhau, bbox/conf trong tolerance, phan hoach khop.
    Trang thai phan hoach `mo_ho` KHONG phai PASS.
    """
    a, b = doc_mot(ref), doc_mot(test)
    r: dict = {"video": ref.stem, "n_ref": len(a), "n_test": len(b)}
    if len(a) != len(b):
        r.update(dat=False, ly_do=f"so hang khac nhau: {len(a)} vs {len(b)}", phan_hoach="chua_xet")
        return r
    if len(a) == 0:
        r.update(dat=True, max_bbox=0.0, max_conf=0.0, id_tho_khop=True, phan_hoach="khop",
                 phan_hoach_khop=True)
        return r

    ra = so_theo_id_tho(a, b)
    khoa_khop = bool(ra.get("id_tho_khop"))
    bbox_ok, conf_ok = ra["max_bbox"] <= TOL_BBOX, ra["max_conf"] <= TOL_CONF
    a_dat = khoa_khop and bbox_ok and conf_ok
    r.update(max_bbox=ra["max_bbox"], max_conf=ra["max_conf"], khoa_frame_id_khop=khoa_khop)

    # (b) chay khi (a) khong dat TRON VEN (hoan doi ID lam tap khoa van bang nhau nhung noi dung khac)
    if a_dat:
        rb = {"trang_thai": "khop", "phan_hoach_khop": True}
    else:
        rb = so_phan_hoach(a, b)
    ph = rb.get("trang_thai", "loi_ghep")
    r.update(phan_hoach=ph, phan_hoach_khop=rb.get("phan_hoach_khop"), phan_hoach_ly_do=rb.get("ly_do", ""))
    for k in ("n_box_mo_ho_ban_dau", "n_box_chua_chot", "chi_danh_so_lech", "n_track_ref", "n_track_test"):
        if k in rb:
            r[k] = rb[k]

    danh_so_lech = (ph == "khop") and (rb.get("chi_danh_so_lech") or not khoa_khop)
    r["id_tho_khop"] = bool(khoa_khop and not danh_so_lech)
    r["dat"] = bool(a_dat and ph == "khop")
    if not r["dat"]:
        vi = []
        # --- quan sat, KHONG suy nguyen nhan ---
        if not bbox_ok:
            vi.append(f"bbox lech {ra['max_bbox']:.3e} > {TOL_BBOX:.0e}")
        if not conf_ok:
            vi.append(f"conf lech {ra['max_conf']:.3e} > {TOL_CONF:.0e}")
        if not khoa_khop:
            vi.append("khoa (frame,id): " + ra.get("ly_do", "khac"))
        if ph == "khop":
            if danh_so_lech:
                vi.append("phan hoach: KHOP qua song anh co dinh (chi lech cach danh so)")
            elif khoa_khop:
                vi.append("ID va phan hoach: KHOP (khac biet chi o bbox/conf; nguyen nhan: CHUA XAC DINH "
                          "tu hai file tracking -- can kiem detection dau vao rieng)")
        elif ph == "mo_ho":
            vi.append("phan hoach: MO HO -- " + rb.get("ly_do", ""))
        elif ph == "khac":
            vi.append("phan hoach: KHAC -- " + rb.get("ly_do", ""))
        else:
            vi.append("phan hoach: khong ghep duoc -- " + rb.get("ly_do", ""))
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

    print(f"{'video':<28} {'hang':>7} {'max|d bbox|':>12} {'max|d conf|':>12} {'phan hoach':>10}  ket qua")
    print("-" * 90)
    for r in ket:
        mb = f"{r['max_bbox']:.3e}" if "max_bbox" in r else "-"
        mc = f"{r['max_conf']:.3e}" if "max_conf" in r else "-"
        print(f"{r['video']:<28} {r.get('n_ref', '-'):>7} {mb:>12} {mc:>12} {r.get('phan_hoach', '-'):>10}  "
              f"{'DAT' if r.get('dat') else 'KHONG DAT: ' + r.get('ly_do', '?')}")

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
