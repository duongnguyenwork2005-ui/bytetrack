"""
test_render_mot_video.py -- Kiem `render_mot_video.py` (render HAU KY bbox/ID/lop tu anh + MOT) va bo kiem
`kiem_video_render.py` (vong 23). Can opencv trong interpreter chay test; khong co -> SKIP (vd env tracker tren Nano).

  1. Render 30 frame tong hop (2 track chay ngang, frame 11-13 KHONG co box, lop car/truck) -> exit 0, `<out>.json` dung
     so frame/fps/kich thuoc/so box; video doc lai du 30 frame.
  2. Bo kiem voi DUNG dau vao -> exit 0 (moi phep DAT).
  3. Bo kiem voi MOT LECH +25 px -> exit 1 (bbox sai cho).
  4. Bo kiem voi fps khai bao KHAC (video 25 fps, kiem 10) -> exit 1.
  5. Bo kiem khi video THIEU frame (render 20, kiem 30) -> exit 1.
  6. Bo kiem voi TEN LOP sai (cls car -> truck cho mot track) -> exit 1 (phep ten lop KHONG DAT).
  7. `render_mot_video.py` va `kiem_video_render.py` dung cu phap Python 3.6 (khong `from __future__ import annotations`,
     khong walrus, khong f-string long nhau) -- kiem bang `ast` voi feature_version=(3, 6).
Review 1d5ffaa, N23-F1 -- so frame hien thi phai duoc kiem DOC LAP tren MOI frame (bbox/ID/lop/fps/do dai giu dung):
  8. so frame thay bang 'f999' tren 27 frame CO box (kich ban cua Codex) -> exit 1, phep so frame KHONG DAT, phep khung DAT.
  9. MAT so frame tren frame co box -> exit 1.
 10. so frame LECH MOT (f<i+1>) tren frame co box -> exit 1 (nhan dang ung vien, khong chi ti le diem).
 11. so frame sai CHI tren 3 frame KHONG co box (11-13) -> exit 1 (giu kiem frame khong box).

    python src/test_render_mot_video.py
"""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
_R: list[tuple[str, str]] = []


def check(name, ok, detail=""):
    _R.append((name, "PASS" if ok else "FAIL"))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    for ln in str(detail).strip().splitlines()[-5:]:
        print(f"         {ln}")


def tao(tmp: Path, cv2, n=30):
    d = tmp / "anh"
    d.mkdir()
    rng = np.random.default_rng(0)
    nen = (rng.integers(40, 200, size=(240, 320, 3))).astype(np.uint8)
    nen = cv2.GaussianBlur(nen, (7, 7), 0)                       # ket cau that-ish cho codec
    mot, cls = [], ["frame,id,cls"]
    for i in range(1, n + 1):
        img = np.roll(nen, i, axis=1).copy()
        cv2.imwrite(str(d / f"img{i:05d}.jpg"), img)
        if 11 <= i <= 13:
            continue
        mot.append(f"{i},1,{20 + 3 * i},{40},60,40,0.9")
        mot.append(f"{i},2,{230 - 3 * i},{150},70,50,0.8")
        cls += [f"{i},1,2", f"{i},2,7"]
    (tmp / "mot.txt").write_text("\n".join(mot) + "\n", encoding="utf-8")
    (tmp / "cls.csv").write_text("\n".join(cls) + "\n", encoding="utf-8")
    return d


def ghi_video_sua(tmp: Path, anh: Path, cv2, ten: str, sua) -> Path:
    """Video 25 fps cung anh/bbox/ID/lop voi render dung; `sua(i, co_box, goc, frame)` doi RIENG vung so frame."""
    import render_mot_video as rmv
    boxes = rmv.doc_mot(tmp / "mot.txt")
    lop = rmv.doc_lop(tmp / "cls.csv")
    out = tmp / (ten + ".mp4")
    wr = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    for i, p in enumerate(rmv.liet_ke(anh), 1):
        goc = cv2.imread(str(p))
        fr = goc.copy()
        rmv.ve_frame(cv2, fr, boxes.get(i, []), lop, i)
        sua(i, bool(boxes.get(i)), goc, fr)
        wr.write(fr)
    wr.release()
    return out


def chay(args):
    return subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")


def main() -> int:
    try:
        import cv2
    except ImportError as e:
        print(f"  [SKIP] test_render_mot_video: khong co opencv trong {PY} ({e})")
        print("\nKET QUA: 0/0 phep kiem tra PASS, 1 SKIP")
        return 0
    tmp = Path(tempfile.mkdtemp(prefix="render_test_"))
    try:
        anh = tao(tmp, cv2)
        vid = tmp / "out.mp4"
        r = chay(["src/render_mot_video.py", "--images", str(anh), "--mot", str(tmp / "mot.txt"), "--cls", str(tmp / "cls.csv"),
                  "--fps", "25", "--out", str(vid)])
        info = json.loads(Path(str(vid) + ".json").read_text(encoding="utf-8")) if Path(str(vid) + ".json").exists() else {}
        cap = cv2.VideoCapture(str(vid))
        n = 0
        while cap.read()[0]:
            n += 1
        cap.release()
        check("1. render 30 frame: exit 0, <out>.json dung (30 frame, 25 fps, 320x240, 54 box), doc lai du 30 frame",
              r.returncode == 0 and info.get("n_frame") == 30 and info.get("fps") == 25 and info.get("kich_thuoc_wh") == [320, 240]
              and info.get("n_box") == 54 and n == 30, f"exit {r.returncode} info={info} doc lai {n}\n{r.stdout[-200:]}{r.stderr[-300:]}")
        base = ["src/kiem_video_render.py", "--video", str(vid), "--images", str(anh), "--cls", str(tmp / "cls.csv"),
                "--buoc-lop", "3"]
        r2 = chay(base + ["--mot", str(tmp / "mot.txt"), "--fps", "25"])
        check("2. bo kiem voi dau vao dung -> exit 0 (DAT)", r2.returncode == 0, r2.stdout[-500:])
        lech = tmp / "mot_lech.txt"
        lech.write_text("\n".join(",".join([v[0], v[1], str(float(v[2]) + 25)] + v[3:]) for v in
                                  (ln.split(",") for ln in (tmp / "mot.txt").read_text().split())) + "\n", encoding="utf-8")
        r3 = chay(base + ["--mot", str(lech), "--fps", "25"])
        check("3. MOT lech +25 px -> exit 1 (bbox sai cho bi bat)", r3.returncode == 1 and "KHONG DAT" in r3.stdout, r3.stdout[-400:])
        r4 = chay(base + ["--mot", str(tmp / "mot.txt"), "--fps", "10"])
        check("4. fps khai bao khac video -> exit 1", r4.returncode == 1 and "fps video" in r4.stdout, r4.stdout[-300:])
        vid20 = tmp / "out20.mp4"
        chay(["src/render_mot_video.py", "--images", str(anh), "--mot", str(tmp / "mot.txt"), "--cls", str(tmp / "cls.csv"),
              "--fps", "25", "--out", str(vid20), "--max-frames", "20"])
        r5 = chay(["src/kiem_video_render.py", "--video", str(vid20), "--images", str(anh), "--mot", str(tmp / "mot.txt"),
                   "--cls", str(tmp / "cls.csv"), "--fps", "25"])
        check("5. video thieu frame (20) khi kiem 30 anh -> exit 1", r5.returncode == 1 and "20 vs 30" in r5.stdout, r5.stdout[-300:])
        sai = tmp / "cls_sai.csv"
        sai.write_text((tmp / "cls.csv").read_text().replace(",1,2\n", ",1,7\n"), encoding="utf-8")
        r6 = chay(["src/kiem_video_render.py", "--video", str(vid), "--images", str(anh), "--mot", str(tmp / "mot.txt"),
                   "--cls", str(sai), "--fps", "25", "--buoc-lop", "3"])
        check("6. ten lop sai (car -> truck cho track 1) -> exit 1, phep ten lop KHONG DAT",
              r6.returncode == 1 and "KHONG DAT] ten lop" in r6.stdout, r6.stdout[-400:])
        loi36 = []
        for f in ("render_mot_video.py", "kiem_video_render.py"):
            src = (ROOT / "src" / f).read_text(encoding="utf-8")
            try:
                t = ast.parse(src, feature_version=(3, 6))
                if any(isinstance(nd, ast.ImportFrom) and nd.module == "__future__" and
                       any(al.name == "annotations" for al in nd.names) for nd in ast.walk(t)):
                    loi36.append(f"{f}: from __future__ import annotations (khong co o 3.6)")
            except SyntaxError as e:
                loi36.append(f"{f}: {e}")
        check("7. render_mot_video.py + kiem_video_render.py dung cu phap Python 3.6", not loi36, loi36)

        def xoa_so(goc, fr):
            fr[:24, :120] = goc[:24, :120]                # chi vung so frame; nhan doi tuong nam duoi

        def viet(fr, chu):                                 # cung kieu chu voi render; KHONG goi ma dang kiem
            cv2.putText(fr, chu, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        ca = {
            "8. so frame 'f999' tren 27 frame co box (kich ban Codex)":
                lambda i, cb, g, fr: (xoa_so(g, fr), viet(fr, "f999")) if cb else None,
            "9. MAT so frame tren frame co box":
                lambda i, cb, g, fr: xoa_so(g, fr) if cb else None,
            "10. so frame LECH MOT (f<i+1>) tren frame co box":
                lambda i, cb, g, fr: (xoa_so(g, fr), viet(fr, "f%d" % (i + 1))) if cb else None,
            "11. so frame sai CHI tren frame KHONG co box (11-13)":
                lambda i, cb, g, fr: (xoa_so(g, fr), viet(fr, "f999")) if not cb else None,
        }
        for ten, sua in ca.items():
            v = ghi_video_sua(tmp, anh, cv2, "sua_%s" % ten.split(".")[0], sua)
            js = tmp / ("kq_%s.json" % ten.split(".")[0])
            rr = chay(["src/kiem_video_render.py", "--video", str(v), "--images", str(anh), "--mot", str(tmp / "mot.txt"),
                       "--cls", str(tmp / "cls.csv"), "--fps", "25", "--buoc-lop", "3", "--json", str(js)])
            kq = json.loads(js.read_text(encoding="utf-8")) if js.exists() else {"phep": {}}
            ph = kq["phep"]
            so_ok = next((val for k, val in ph.items() if k.startswith("so frame hien thi")), None)
            khung_ok = next((val for k, val in ph.items() if k.startswith("khung bbox")), None)
            check(ten + " -> exit 1: phep so frame KHONG DAT, phep khung bbox/ID van DAT",
                  rr.returncode == 1 and so_ok is False and khung_ok is True and not kq.get("dat"),
                  f"exit {rr.returncode} so_frame={so_ok} khung={khung_ok} loi_so={kq.get('loi_so', [])[:2]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(1 for _, s in _R if s == "PASS")
    n_fail = sum(1 for _, s in _R if s == "FAIL")
    print(f"\nKET QUA: {n_ok}/{n_ok + n_fail} phep kiem tra PASS")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
