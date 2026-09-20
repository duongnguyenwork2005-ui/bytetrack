"""
test_nano_parity.py -- Bo so ket qua tracking Nano voi ban tham chieu PC (`src/nano_parity.py`).

Cong cu nay quyet dinh co duoc do toc do tren Nano hay khong, nen no phai:
  - CHAP NHAN sai lech dau phay dong nho hon tolerance da cong bo (kien truc khac nhau),
  - TU CHOI moi khac biet ve LIEN KET ID (quyet dinh roi rac, khong phai sai so),
  - phan biet "danh so track lech" voi "ghep track khac han" -- hai loai loi rat khac nhau.

Cac phep kiem:
  1. Hai file y het            -> DAT.
  2. bbox lech TRONG tolerance -> DAT.
  3. bbox lech VUOT tolerance  -> KHONG DAT, bao dung ly do.
  4. conf lech vuot tolerance  -> KHONG DAT.
  5. ID danh so lai nhung PHAN HOACH y het -> ID tho KHAC, phan hoach KHOP (phan biet duoc).
  6. Ghep track KHAC han (doi mot detection sang track khac) -> phan hoach KHONG khop.
  7. So hang khac nhau         -> KHONG DAT.
  8. Hai file RONG             -> DAT.
  9. Thieu file ben can kiem   -> KHONG DAT, khong nem ra ngoai.

Phep kiem 5 la ly do khoa sap xep KHONG duoc chua `id`: neu sap theo (frame, id, x) thi hai file
danh so khac nhau se bi sap khac thu tu, va phep so phan hoach mat y nghia.

    python src/test_nano_parity.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
_RESULTS: list[tuple[str, bool]] = []


def _check(name: str, ok: bool, detail: str = "") -> None:
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines()[-5:]:
            print(f"         {line}")
    assert ok, name


#: 4 frame, 2 track. Track 1 di ngang o y=100; track 2 o y=300.
GOC = [
    (1, 1, 100.0, 100.0, 50.0, 40.0, 0.90),
    (1, 2, 600.0, 300.0, 60.0, 50.0, 0.80),
    (2, 1, 110.0, 100.0, 50.0, 40.0, 0.91),
    (2, 2, 610.0, 300.0, 60.0, 50.0, 0.81),
    (3, 1, 120.0, 100.0, 50.0, 40.0, 0.92),
    (3, 2, 620.0, 300.0, 60.0, 50.0, 0.82),
    (4, 1, 130.0, 100.0, 50.0, 40.0, 0.93),
    (4, 2, 630.0, 300.0, 60.0, 50.0, 0.83),
]


def viet(p: Path, rows) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(f"{int(f)},{int(i)},{x!r},{y!r},{w!r},{h!r},{c!r}\n"
                         for f, i, x, y, w, h, c in rows), encoding="utf-8")


def main() -> int:  # noqa: C901
    failed = 0
    tmp = Path(tempfile.mkdtemp(prefix="nanoparity_"))
    try:
        import nano_parity as np_

        ref = tmp / "ref"
        viet(ref / "v1.txt", GOC)

        def so(rows, ten="t") -> dict:
            d = tmp / ten
            viet(d / "v1.txt", rows)
            return np_.so_mot_video(ref / "v1.txt", d / "v1.txt")

        # --- 1. y het ---
        r = so(GOC, "t1")
        _check("hai file y het -> DAT, lech bang 0",
               r["dat"] and r["max_bbox"] == 0.0 and r["id_tho_khop"] and r["phan_hoach_khop"],
               f"{r}")

        # --- 2. bbox lech TRONG tolerance ---
        nho = np_.TOL_BBOX / 2
        r = so([(f, i, x + nho, y, w, h, c) for f, i, x, y, w, h, c in GOC], "t2")
        _check(f"bbox lech {nho:.1e} (< tolerance {np_.TOL_BBOX:.0e}) -> DAT",
               r["dat"] and 0 < r["max_bbox"] <= np_.TOL_BBOX, f"max_bbox={r['max_bbox']:.3e}")

        # --- 3. bbox lech VUOT tolerance ---
        lon = np_.TOL_BBOX * 10
        r = so([(f, i, x + lon, y, w, h, c) for f, i, x, y, w, h, c in GOC], "t3")
        _check(f"bbox lech {lon:.1e} (> tolerance) -> KHONG DAT, ly do noi ro 'bbox'",
               (not r["dat"]) and "bbox" in r["ly_do"], r.get("ly_do", ""))

        # --- 4. conf lech vuot tolerance ---
        r = so([(f, i, x, y, w, h, c + 1e-3) for f, i, x, y, w, h, c in GOC], "t4")
        _check("conf lech vuot tolerance -> KHONG DAT, ly do noi ro 'conf'",
               (not r["dat"]) and "conf" in r["ly_do"], r.get("ly_do", ""))

        # --- 5. ID danh so lai, PHAN HOACH y het ---
        doi_so = {1: 7, 2: 9}
        r = so([(f, doi_so[i], x, y, w, h, c) for f, i, x, y, w, h, c in GOC], "t5")
        _check("ID danh so lai (1->7, 2->9) nhung PHAN HOACH y het: ID tho KHAC, phan hoach KHOP "
               "-> phan biet duoc 'lech danh so' voi 'ghep khac'",
               (not r["dat"]) and (not r["id_tho_khop"]) and r["phan_hoach_khop"]
               and "danh so" in r["ly_do"], f"{r.get('ly_do','')}")

        # --- 6. ghep track KHAC han ---
        xau = list(GOC)
        xau[6] = (4, 2, 130.0, 100.0, 50.0, 40.0, 0.93)     # frame 4 cua track 1 -> gan sang track 2
        r = so(xau, "t6")
        _check("doi mot detection sang track khac -> PHAN HOACH khong khop (bat duoc loi ghep)",
               (not r["dat"]) and (not r["phan_hoach_khop"]), f"{r.get('ly_do','')}")

        # --- 7. so hang khac nhau ---
        r = so(GOC[:-1], "t7")
        _check("so hang khac nhau -> KHONG DAT",
               (not r["dat"]) and "so hang" in r["ly_do"], r.get("ly_do", ""))

        # --- 8. hai file RONG ---
        (tmp / "ref_rong").mkdir(parents=True, exist_ok=True)
        (tmp / "ref_rong" / "v1.txt").write_text("", encoding="utf-8")
        (tmp / "t8").mkdir(parents=True, exist_ok=True)
        (tmp / "t8" / "v1.txt").write_text("", encoding="utf-8")
        r = np_.so_mot_video(tmp / "ref_rong" / "v1.txt", tmp / "t8" / "v1.txt")
        _check("hai file RONG -> DAT (khong co track nao la mot ket qua hop le)",
               r["dat"] and r["n_ref"] == 0, f"{r}")

        # --- 9. thieu file ben can kiem ---
        loi = ""
        try:
            np_.so_mot_video(ref / "v1.txt", tmp / "khong_co" / "v1.txt")
        except FileNotFoundError as e:
            loi = str(e)
        _check("thieu file ben can kiem -> FileNotFoundError ro rang (CLI bat va bao KHONG DAT)",
               "khong co" in loi, loi[:120])

        # --- 10. tolerance duoc cong bo thanh hang so, khong an trong ham ---
        _check("tolerance la hang so module, cong bo duoc truoc khi so "
               f"(bbox={np_.TOL_BBOX:.0e}, conf={np_.TOL_CONF:.0e}, id={np_.TOL_ID})",
               np_.TOL_BBOX > 0 and np_.TOL_CONF > 0 and np_.TOL_ID is None)
    except AssertionError:
        failed += 1
    except Exception as e:  # noqa: BLE001
        failed += 1
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] loi khong mong doi: {type(e).__name__}: {e}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
