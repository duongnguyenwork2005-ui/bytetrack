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
 11. (F2, review 8bab154) Hai box co x GAN NHAU, chi lech 0,0002 px -> DAT (ban cu bao 400 px).
 12. (F2) Hai box TRUNG (x, y), khac (w, h); file test chi DAO THU TU dong -> DAT (ban cu bao 10 px).
 13. (F2) Doi so ID lam DAO THU TU ID (1<->2) nhung lien ket y het -> ID tho khac, phan hoach KHOP.
 14. (F2) Thuc su DOI LIEN KET o mot frame (hai track hoan doi box) -> phan hoach KHONG khop, FAIL.
 15. Khoa (frame, id) TRUNG trong mot file -> khong hop le de so, KHONG DAT.
 16. (F3, review 8abb368) Chi DAO THU TU DONG cua test (giu nguyen tap ban ghi) -> ket luan Y HET.
 17. (F3) Hai track bbox/conf GIONG HET o hai frame, ref ID 1/2, test ID 8/9: hinh hoc khong phan
     biet duoc -> trang thai MO HO (khong PASS, khong FAIL), va dao thu tu 2 dong frame 2 -> VAN mo ho.
 18. (F3) Chi doi confidence trong file ket qua -> KHONG quy nguyen nhan cho detector; ghi CHUA XAC DINH.
 19. (F3) Neo khong mo ho o frame khac GIAI DUOC mo ho o frame nay bang rang buoc toan sequence.

Cac ca 11-14 la ly do KHONG duoc ghep hang theo thu tu sap xep toa do: toa do duoc phep lech toi
tolerance nen thu tu tuyet doi khong con la khoa. Muc (a) ghep bang khoa (frame, id); muc (b) ghep
hinh hoc 1-1 theo tolerance trong tung frame va kiem song anh ID co dinh tren TOAN sequence.

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
        xau[6] = (4, 3, 130.0, 100.0, 50.0, 40.0, 0.93)     # frame 4: box cua track 1 -> track MOI id 3
        r = so(xau, "t6")
        _check("track 1 bi cat sang ID moi (3) o frame 4 -> song anh ID khong co dinh -> PHAN HOACH "
               "khong khop (bat duoc loi ghep)",
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

        # --- 11. (F2) hai box x gan nhau, lech 0,0002 px -> phai DAT ---
        ref2 = tmp / "ref_f2"
        F2_REF = [(1, 1, 20.0, 100.0, 50.0, 40.0, 0.9), (1, 2, 40.0, 500.0, 60.0, 50.0, 0.8),
                  (2, 1, 100.0, 100.0, 50.0, 40.0, 0.9), (2, 2, 100.0001, 500.0, 60.0, 50.0, 0.8)]
        F2_TEST = [(1, 1, 20.0, 100.0, 50.0, 40.0, 0.9), (1, 2, 40.0, 500.0, 60.0, 50.0, 0.8),
                   (2, 1, 100.0002, 100.0, 50.0, 40.0, 0.9), (2, 2, 99.9999, 500.0, 60.0, 50.0, 0.8)]
        viet(ref2 / "v1.txt", F2_REF)
        viet(tmp / "t11" / "v1.txt", F2_TEST)
        r = np_.so_mot_video(ref2 / "v1.txt", tmp / "t11" / "v1.txt")
        _check("(F2) hai box x gan nhau (100.0 / 100.0001), test lech 0,0002 px -> DAT, "
               "max_bbox = 2e-4 (ban cu ghep nham hang, bao 400 px va conf lech 0,1)",
               r["dat"] and abs(r["max_bbox"] - 2e-4) < 1e-9 and r["max_conf"] == 0.0,
               f"max_bbox={r['max_bbox']:.3e} max_conf={r['max_conf']:.3e} ly_do={r.get('ly_do','')}")

        # --- 12. (F2) hai box TRUNG (x, y), khac (w, h); test chi dao thu tu dong -> DAT ---
        ref3 = tmp / "ref_f2b"
        TRUNG = [(1, 1, 50.0, 50.0, 30.0, 30.0, 0.9), (1, 2, 50.0, 50.0, 40.0, 40.0, 0.8)]
        viet(ref3 / "v1.txt", TRUNG)
        viet(tmp / "t12" / "v1.txt", list(reversed(TRUNG)))
        r = np_.so_mot_video(ref3 / "v1.txt", tmp / "t12" / "v1.txt")
        _check("(F2) hai box trung (x,y) khac (w,h), test chi DAO THU TU dong -> DAT, lech 0 "
               "(ban cu bao 10 px)",
               r["dat"] and r["max_bbox"] == 0.0, f"max_bbox={r['max_bbox']} ly_do={r.get('ly_do','')}")

        # --- 13. (F2) doi so ID lam DAO THU TU ID nhung lien ket y het ---
        dao = {1: 2, 2: 1}
        r = so([(f, dao[i], x, y, w, h, c) for f, i, x, y, w, h, c in GOC], "t13")
        _check("(F2) hoan doi so ID (1<->2), lien ket y het -> ID tho KHAC nhung phan hoach KHOP, "
               "bao 'chi lech cach danh so'",
               (not r["dat"]) and (not r["id_tho_khop"]) and r["phan_hoach_khop"]
               and r.get("chi_danh_so_lech") is True and "danh so" in r["ly_do"],
               f"ly_do={r.get('ly_do','')}")

        # --- 14. (F2) THUC SU doi lien ket o mot frame: hai track hoan doi box tai frame 3 ---
        swap = list(GOC)
        swap[4] = (3, 2, 120.0, 100.0, 50.0, 40.0, 0.92)    # box cua track 1 -> gan cho track 2
        swap[5] = (3, 1, 620.0, 300.0, 60.0, 50.0, 0.82)    # box cua track 2 -> gan cho track 1
        r = so(swap, "t14")
        _check("(F2) hai track HOAN DOI box tai mot frame (lien ket that su khac) -> phan hoach "
               "KHONG khop, song anh ID khong co dinh -> FAIL",
               (not r["dat"]) and (not r["phan_hoach_khop"]) and "song anh" in r["phan_hoach_ly_do"],
               f"ly_do={r.get('ly_do','')}")

        # --- 15. khoa (frame, id) trung -> khong hop le ---
        r = so(GOC + [(1, 1, 999.0, 999.0, 5.0, 5.0, 0.5)], "t15")
        _check("khoa (frame,id) TRUNG trong file test -> khong hop le de so, KHONG DAT",
               (not r["dat"]) and ("so hang" in r["ly_do"] or "TRUNG" in r["ly_do"]),
               f"ly_do={r.get('ly_do','')}")

        # --- 16. (F3) dao thu tu dong -> ket luan y het ---
        import random
        rows_dao = list(GOC)
        random.Random(7).shuffle(rows_dao)
        r_goc = so(GOC, "t16a")
        r_dao = so(rows_dao, "t16b")
        khoa_so = ("dat", "max_bbox", "max_conf", "id_tho_khop", "phan_hoach", "phan_hoach_khop")
        _check("(F3) chi DAO THU TU DONG cua test (cung tap ban ghi) -> ket luan y het",
               all(r_goc.get(k) == r_dao.get(k) for k in khoa_so),
               f"goc={[r_goc.get(k) for k in khoa_so]} dao={[r_dao.get(k) for k in khoa_so]}")
        swap_dao = list(swap)                           # ca doi lien ket that (phep 14) cung phai on dinh
        random.Random(11).shuffle(swap_dao)
        r_s1, r_s2 = so(swap, "t16c"), so(swap_dao, "t16d")
        _check("(F3) ca doi lien ket that: dao thu tu dong van FAIL voi cung trang thai phan hoach",
               (not r_s1["dat"]) and (not r_s2["dat"]) and r_s1["phan_hoach"] == r_s2["phan_hoach"] == "khac",
               f"{r_s1['phan_hoach']} / {r_s2['phan_hoach']}")

        # --- 17. (F3) fixture Codex: hai track giong het -> MO HO, on dinh khi dao dong ---
        B = (10.0, 20.0, 30.0, 40.0, 0.9)
        refc = tmp / "ref_codex"
        viet(refc / "v1.txt", [(1, 1, *B), (1, 2, *B), (2, 1, *B), (2, 2, *B)])
        viet(tmp / "t17a" / "v1.txt", [(1, 8, *B), (1, 9, *B), (2, 8, *B), (2, 9, *B)])
        viet(tmp / "t17b" / "v1.txt", [(1, 8, *B), (1, 9, *B), (2, 9, *B), (2, 8, *B)])
        ra_ = np_.so_mot_video(refc / "v1.txt", tmp / "t17a" / "v1.txt")
        rb_ = np_.so_mot_video(refc / "v1.txt", tmp / "t17b" / "v1.txt")
        _check("(F3) hai track bbox/conf GIONG HET, ID 1/2 vs 8/9: phan hoach = MO HO (khong PASS, khong "
               "FAIL), va DAO 2 dong frame 2 -> VAN mo ho (ban cu: khop -> khac)",
               ra_["phan_hoach"] == "mo_ho" and rb_["phan_hoach"] == "mo_ho"
               and (not ra_["dat"]) and (not rb_["dat"]) and ra_["phan_hoach_khop"] is None,
               f"a={ra_['phan_hoach']} b={rb_['phan_hoach']} ly_do={ra_.get('ly_do','')[:90]}")

        # --- 18. (F3) chi doi conf trong file ket qua -> khong quy cho detector ---
        r = so([(f, i, x, y, w, h, c + 2e-5) for f, i, x, y, w, h, c in GOC], "t18")
        _check("(F3) chi doi confidence 2e-5 trong file ket qua: KHONG DAT, KHONG quy nguyen nhan cho "
               "detector, ghi CHUA XAC DINH",
               (not r["dat"]) and "detector" not in r["ly_do"].lower().replace("chua xac dinh", "")
               and "CHUA XAC DINH" in r["ly_do"] and r["id_tho_khop"] and r["phan_hoach"] == "khop",
               r.get("ly_do", ""))

        # --- 19. (F3) neo o frame khac giai duoc mo ho ---
        # frame 1: hai box GIONG HET (mo ho); frame 2: hai box KHAC vi tri (neo). Test danh so 8/9.
        ref19 = tmp / "ref19"
        viet(ref19 / "v1.txt", [(1, 1, *B), (1, 2, *B), (2, 1, 100.0, 100.0, 50.0, 40.0, 0.9),
                                (2, 2, 600.0, 300.0, 60.0, 50.0, 0.8)])
        viet(tmp / "t19" / "v1.txt", [(1, 8, *B), (1, 9, *B), (2, 8, 100.0, 100.0, 50.0, 40.0, 0.9),
                                      (2, 9, 600.0, 300.0, 60.0, 50.0, 0.8)])
        r = np_.so_mot_video(ref19 / "v1.txt", tmp / "t19" / "v1.txt")
        _check("(F3) frame 1 mo ho nhung frame 2 co neo 1->8, 2->9: rang buoc toan sequence GIAI duoc -> "
               "phan hoach KHOP (chi lech danh so), khong con mo ho",
               r["phan_hoach"] == "khop" and r.get("chi_danh_so_lech") is True and r.get("n_box_chua_chot") == 0,
               f"phan_hoach={r['phan_hoach']} chua_chot={r.get('n_box_chua_chot')} ly_do={r.get('ly_do','')[:80]}")

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
