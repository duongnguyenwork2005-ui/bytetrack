"""
nano_parity.py -- So ket qua tracking cua Jetson Nano voi BAN THAM CHIEU tren PC.

Dung TRUOC khi do toc do: neu ket qua tracking khac nhau thi so toc do khong so sanh duoc.

TOLERANCE DUOC CONG BO TRUOC KHI SO (hang so ngay duoi day, in ra dau moi lan chay):

  bbox      : sai so tuyet doi <= TOL_BBOX pixel cho tung toa do (x, y, w, h)
  confidence: sai so tuyet doi <= TOL_CONF
  LIEN KET ID: **KHONG co tolerance** -- phai khop TUYET DOI.
              Association la quyet dinh ROI RAC: mot khac biet nghia la tracker da ghep khac,
              khong phai "sai so so hoc". Kiem theo HAI muc:
                (a) gia tri ID tho phai bang nhau;
                (b) PHAN HOACH track phai tuong duong (song anh 1-1 giua hai tap ID).
              (b) yeu hon (a): neu chi danh so lech thi (a) fail nhung (b) pass -- bao cao tach
              rieng de biet dung loai khac biet.

VI SAO bbox co tolerance ma ID thi khong: hai kien truc (x86-64 va aarch64) co the dung BLAS/libm
khac nhau, nen phep toan dau phay dong co the lech o chu so cuoi. Lech do chap nhan duoc mien la
khong doi quyet dinh ghep. Neu no doi quyet dinh ghep thi (a)/(b) se bat duoc.

    python src/nano_parity.py --ref <thu muc PC> --test <thu muc Nano> [--videos v1 v2]
    python src/nano_parity.py --ref data/processed/trackers/X/pc-cv --test .../nano-cv --json out.json

Tra exit 0 neu DAT ca ba muc, 1 neu khong.
"""
from __future__ import annotations

import argparse
import json
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


def phan_hoach(a: np.ndarray) -> dict:
    """track_id -> tuple cac (frame, thu tu trong frame). Dung de so PHAN HOACH, bo qua ten ID."""
    out: dict[float, list] = {}
    for i in range(len(a)):
        out.setdefault(a[i, 1], []).append(i)
    return out


def song_anh_1_1(pa: dict, pb: dict) -> tuple[bool, str]:
    """Hai phan hoach co tuong duong qua mot song anh 1-1 khong (bo qua gia tri ID)?"""
    if len(pa) != len(pb):
        return False, f"so track khac nhau: {len(pa)} vs {len(pb)}"
    # khoa cua mot track = tap chi so hang (da sap xep) -> hai phan hoach bang nhau khi tap khoa bang nhau
    ka = sorted(tuple(v) for v in pa.values())
    kb = sorted(tuple(v) for v in pb.values())
    if ka != kb:
        khac = sum(1 for x, y in zip(ka, kb) if x != y)
        return False, f"{khac}/{len(ka)} track co tap hang khac nhau"
    return True, ""


def so_mot_video(ref: Path, test: Path) -> dict:
    a, b = doc_mot(ref), doc_mot(test)
    r: dict = {"video": ref.stem, "n_ref": len(a), "n_test": len(b)}
    if len(a) != len(b):
        r.update(dat=False, ly_do=f"so hang khac nhau: {len(a)} vs {len(b)}")
        return r
    if len(a) == 0:
        r.update(dat=True, max_bbox=0.0, max_conf=0.0, id_tho_khop=True, phan_hoach_khop=True)
        return r
    # Sap theo (frame, x, y) -- KHONG dung `id` lam khoa sap xep. Neu dung `id` thi chi so hang
    # se phu thuoc cach danh so track, lam phep kiem PHAN HOACH duoi day mat y nghia (hai file
    # danh so khac nhau se bi sap khac nhau roi ket luan sai). Khoa nay chi phu thuoc vi tri box,
    # nen cung mot detection nam cung chi so o ca hai file.
    ia = np.lexsort((a[:, 3], a[:, 2], a[:, 0]))
    ib = np.lexsort((b[:, 3], b[:, 2], b[:, 0]))
    a, b = a[ia], b[ib]

    if not np.array_equal(a[:, 0], b[:, 0]):
        r.update(dat=False, ly_do="danh sach frame khac nhau")
        return r

    d_bbox = float(np.max(np.abs(a[:, 2:6] - b[:, 2:6])))
    d_conf = float(np.max(np.abs(a[:, 6] - b[:, 6])))
    id_tho = bool(np.array_equal(a[:, 1], b[:, 1]))
    ph_ok, ph_vi = song_anh_1_1(phan_hoach(a), phan_hoach(b))

    r.update(max_bbox=d_bbox, max_conf=d_conf, id_tho_khop=id_tho,
             phan_hoach_khop=ph_ok, phan_hoach_ly_do=ph_vi,
             dat=bool(d_bbox <= TOL_BBOX and d_conf <= TOL_CONF and id_tho and ph_ok))
    if not r["dat"]:
        vi = []
        if d_bbox > TOL_BBOX:
            vi.append(f"bbox lech {d_bbox:.3e} > {TOL_BBOX:.0e}")
        if d_conf > TOL_CONF:
            vi.append(f"conf lech {d_conf:.3e} > {TOL_CONF:.0e}")
        if not id_tho:
            n = int(np.sum(a[:, 1] != b[:, 1]))
            vi.append(f"{n}/{len(a)} hang co ID tho khac"
                      + (" (nhung PHAN HOACH van tuong duong => chi lech cach danh so)"
                         if ph_ok else ""))
        if not ph_ok:
            vi.append(f"phan hoach track KHAC: {ph_vi}")
        r["ly_do"] = "; ".join(vi)
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description="So tracking Nano voi ban tham chieu PC")
    ap.add_argument("--ref", required=True, help="thu muc ban THAM CHIEU (PC), chua data/*.txt hoac *.txt")
    ap.add_argument("--test", required=True, help="thu muc ban CAN KIEM (Nano)")
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
    print(f"  lien ket ID: KHONG co tolerance -- ID tho phai bang nhau VA phan hoach track tuong duong")
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
