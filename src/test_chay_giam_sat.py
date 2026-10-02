"""
test_chay_giam_sat.py -- Kiem `chay_giam_sat.py` (vong 39): bam DUNG PID do cong cu tao, khong ghi 0 cho so khong doc
duoc, giu exit code that, khong lam dut pipe. Tien trinh con la script Python tong hop trong thu muc tam; khong train,
khong cham du lieu du an.

  M1  Shell/monitor mo phong co CUNG ten script trong command line + nhieu tien trinh Python khac dang chay: cong cu
      bam dung tien trinh con cua no (PID con tu in ra), cot pid_chinh khong doi, RSS phan anh bo nho cua con.
  M2  Worker duoc tao roi ket thuc giua chung: so_con tang roi ve 0, rss_tong > rss_chinh khi worker con song; moi mau
      deu co so (khong 0 gia).
  M3  Tien trinh con thoat loi (exit 3) -> cong cu exit 3, log `# exit=3`, tom_tat ghi 3.
  M4  Tien trinh con bi kill tu ben ngoai -> cong cu tra dung ma cua con; lay mau dung, ly do ghi ro; khong mau 0.
  M5  Ma NTSTATUS lon (0xC000013A) giu nguyen qua cong cu (returncode 32 bit cua Python).
  M6  Ctrl+Break gui cho nhom tien trinh (cong cu + con): cong cu KHONG chet truoc con, ghi tin hieu, tra ma cua con.
  M7  stdout: ky tu thanh tien do UTF-8 + \\r, byte hong, 50.000 dong, dong cuoi khong newline -> log du, exit 0; ca khi
      tien trinh con tu reconfigure UTF-8 (--giu-moi-truong).
  M8  `gan`: PID khong ton tai / create_time sai -> exit 2, khong tao thu muc; PID hop le ket thuc giua chung -> dung,
      ly do ro, chi co mau hop le.
  M9  Thu muc dau ra da co noi dung -> exit 2 (khong ghi de).
  M10 `trang-thai`: ket_thuc (exit) / dang_chay / bi_ngat_khong_co_exit_code (PID chet ma chua ghi ket_thuc).
  M11 `tinh-lai` tinh lai tu CSV khop tom_tat; sua tom_tat -> phat hien (exit 1).

    python src/test_chay_giam_sat.py
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

SRC = Path(__file__).resolve().parent
CONG_CU = SRC / "chay_giam_sat.py"
PY = sys.executable
KQ: list[tuple[str, bool]] = []
SKIP: list[tuple[str, str]] = []


def _check(ten, ok, ct=""):
    KQ.append((ten, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {ten}" + (f"  ({str(ct)[:500]})" if ct else ""), flush=True)


def _skip(ten, ly_do):
    SKIP.append((ten, ly_do))
    print(f"  [SKIP] {ten}  ({ly_do}) -- CHUA KIEM, khong tinh PASS", flush=True)


def _chay(nhom, fn):
    print(f"\n== {nhom}", flush=True)
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        _check(f"{nhom}: loi khong mong doi {type(e).__name__}: {e}", False)
        traceback.print_exc()


def script(tmp: Path, ten: str, noi_dung: str) -> Path:
    p = tmp / ten
    p.write_text(noi_dung, encoding="utf-8")
    return p


def giam_sat(out: Path, lenh: list[str], them: list[str] | None = None, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([PY, str(CONG_CU), "chay", "--thu-muc", str(out), "--chu-ky", "0.25", "--khong-in",
                           *(them or []), "--", *lenh], capture_output=True, timeout=300, **kw)


def doc_csv(out: Path) -> list[dict]:
    p = out / "ram_mau_tho.csv"
    return list(csv.DictReader(open(p, newline="", encoding="utf-8"))) if p.is_file() else []


def doc_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


CON_CAP_PHAT = r'''
import os, sys, time
print("PID_CON=%d" % os.getpid(), flush=True)
x = bytearray(160 * 2**20)
for i in range(0, len(x), 4096):
    x[i] = 1
time.sleep(2.5)
print("xong", flush=True)
'''

CON_WORKER = r'''
import multiprocessing as mp, os, time

def lam(n):
    x = bytearray(90 * 2**20)
    for i in range(0, len(x), 4096):
        x[i] = 1
    time.sleep(3)

if __name__ == "__main__":
    print("PID_CON=%d" % os.getpid(), flush=True)
    ps = [mp.Process(target=lam, args=(i,)) for i in range(2)]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    print("worker xong", flush=True)
    time.sleep(2)
'''

CON_IN_NHIEU = r'''
import sys, time
for i in range(3):
    sys.stdout.write("Scanning ━━━━─── %d%%\r" % (i * 33)); sys.stdout.flush(); time.sleep(0.05)
sys.stdout.write("Scanning ━━━━━━ 100%\n")
sys.stdout.write("emoji ✅ ±\n"); sys.stdout.flush()
sys.stdout.buffer.write(b"byte hong \x81\xff\n"); sys.stdout.flush()
for i in range(50000):
    sys.stdout.write("%d/50000 ━─\n" % i)
sys.stdout.write("cuoi khong newline"); sys.stdout.flush()
'''

CON_TU_UTF8 = "import sys\nsys.stdout.reconfigure(encoding='utf-8')\n" + CON_IN_NHIEU


def m1(tmp: Path):
    import psutil
    nhu_monitor = subprocess.Popen([PY, "-c", "import time; time.sleep(120)", "train_detector.py", "--gia-monitor"])
    python_khac = subprocess.Popen([PY, "-c", "import time; x = bytearray(50 * 2**20); time.sleep(120)"])
    try:
        time.sleep(1.0)
        con = script(tmp, "train_detector.py", CON_CAP_PHAT)             # tien trinh con CUNG ten script
        out = tmp / "m1"
        r = giam_sat(out, [PY, str(con)])
        log = (out / "log.txt").read_text(encoding="utf-8")
        pid_con = int(log.split("PID_CON=")[1].split()[0])
        tt, tom, rows = doc_json(out / "trang_thai.json"), doc_json(out / "tom_tat.json"), doc_csv(out)
        _check("M1a exit 0; PID trong trang_thai/tom_tat = PID tien trinh con tu in ra; KHAC PID shell gia monitor",
               r.returncode == 0 and tt["pid"] == tom["pid"] == pid_con != nhu_monitor.pid,
               f"exit {r.returncode}, pid {tom['pid']}, con {pid_con}, gia monitor {nhu_monitor.pid}")
        _check("M1b create_time ghi lai = create_time that cua PID do (doi chieu psutil truoc khi con thoat)",
               tt["create_time"] is not None and tom["create_time"] == tt["create_time"], str(tt["create_time"]))
        pid_csv = {int(x["pid_chinh"]) for x in rows}
        rss = [float(x["rss_chinh_mib"]) for x in rows if x["rss_chinh_mib"]]
        _check("M1c moi mau CSV deu cua PID con; RSS chinh max >= 150 MiB (con cap phat 160 MiB) -- khong phai shell gia "
               "(~10 MiB) hay python khac (~60 MiB)", pid_csv == {pid_con} and rss and max(rss) >= 150,
               f"pid csv {pid_csv}, rss max {max(rss) if rss else None}, so mau {len(rows)}")
        _check("M1d khong mau nao co rss rong / 0", all(x["rss_chinh_mib"] and float(x["rss_chinh_mib"]) > 0 for x in rows),
               f"{len(rows)} mau")
    finally:
        for p in (nhu_monitor, python_khac):
            p.kill()
            p.wait()


def m2(tmp: Path):
    con = script(tmp, "con_worker.py", CON_WORKER)
    out = tmp / "m2"
    r = giam_sat(out, [PY, str(con)])
    rows = doc_csv(out)
    so_con = [int(x["so_con"]) for x in rows if x["so_con"]]
    co_worker = [x for x in rows if x["so_con"] and int(x["so_con"]) >= 2]
    sau = rows[rows.index(co_worker[-1]) + 1:] if co_worker else []
    _check("M2a exit 0; co mau voi >= 2 worker con, rss_tong - rss_chinh >= 150 MiB khi worker song",
           r.returncode == 0 and co_worker and max(float(x["rss_tong_mib"]) - float(x["rss_chinh_mib"])
                                                   for x in co_worker) >= 150,
           f"so_con max {max(so_con) if so_con else None}, {len(co_worker)} mau co worker")
    _check("M2b sau khi worker ket thuc: so_con ve 0 trong mau sau; chi_tiet_con liet ke PID:ten:RSS khi con song",
           any(int(x["so_con"]) == 0 for x in sau) and all(x["chi_tiet_con"].count(":") >= 4 for x in co_worker),
           f"{len(sau)} mau sau, vd chi tiet {co_worker[0]['chi_tiet_con'] if co_worker else None}")
    _check("M2c khong mau nao 0 gia: rss_chinh co so > 0 o moi mau",
           all(x["rss_chinh_mib"] and float(x["rss_chinh_mib"]) > 0 for x in rows), f"{len(rows)} mau")


def m3(tmp: Path):
    out = tmp / "m3"
    r = giam_sat(out, [PY, "-c", "print('sap loi'); import sys; sys.exit(3)"])
    log = (out / "log.txt").read_text(encoding="utf-8")
    tom = doc_json(out / "tom_tat.json")
    _check("M3 tien trinh con exit 3 -> cong cu exit 3, log ket thuc '# exit=3', tom_tat exit 3, trang_thai ket_thuc",
           r.returncode == 3 and log.rstrip().endswith("# exit=3") and tom["exit_code"] == 3
           and doc_json(out / "trang_thai.json")["trang_thai"] == "ket_thuc", f"exit {r.returncode}")


def m4(tmp: Path):
    import psutil
    out = tmp / "m4"
    p = subprocess.Popen([PY, str(CONG_CU), "chay", "--thu-muc", str(out), "--chu-ky", "0.25", "--khong-in", "--", PY,
                          "-c", "import time; print('bat dau', flush=True); time.sleep(60)"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for _ in range(100):
            if (out / "trang_thai.json").is_file():
                break
            time.sleep(0.1)
        time.sleep(1.5)
        tt = doc_json(out / "trang_thai.json")
        psutil.Process(tt["pid"]).kill()
        rc = p.wait(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    tom, rows = doc_json(out / "tom_tat.json"), doc_csv(out)
    _check("M4a tien trinh con bi kill -> cong cu thoat sau con, exit = ma cua con (khac 0), tom_tat cung ma",
           (rc & 0xFFFFFFFF) == tom["exit_code"] and tom["exit_code"] != 0,
           f"cong cu {rc}, tom_tat {tom['exit_code']} ({tom['exit_code_hex']})")
    _check("M4b lay mau dung voi ly do 'tien trinh chinh da ket thuc' hoac 'dung theo yeu cau'; khong mau 0 / rong",
           rows and all(x["rss_chinh_mib"] and float(x["rss_chinh_mib"]) > 0 for x in rows)
           and ("ket thuc" in tom["ly_do_dung_lay_mau"] or "theo yeu cau" in tom["ly_do_dung_lay_mau"]),
           f"{len(rows)} mau, ly do {tom['ly_do_dung_lay_mau']}")


def m5(tmp: Path):
    out = tmp / "m5"
    r = giam_sat(out, [PY, "-c", "import ctypes; ctypes.windll.kernel32.ExitProcess(0xC000013A)"])
    tom = doc_json(out / "tom_tat.json")
    _check("M5 ma NTSTATUS 0xC000013A giu nguyen: returncode 32 bit cua cong cu == 3221225786, tom_tat cung ma",
           r.returncode & 0xFFFFFFFF == 0xC000013A and tom["exit_code"] == 0xC000013A,
           f"returncode {r.returncode} ({r.returncode & 0xFFFFFFFF:#x}), tom_tat {tom['exit_code_hex']}")


def m6(tmp: Path):
    if os.name != "nt":
        _skip("M6 Ctrl+Break", "chi Windows")
        return
    out = tmp / "m6"
    p = subprocess.Popen([PY, str(CONG_CU), "chay", "--thu-muc", str(out), "--chu-ky", "0.25", "--khong-in", "--", PY,
                          "-c", "import time; print('cho', flush=True); time.sleep(60)"],
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
    try:
        for _ in range(100):
            if (out / "trang_thai.json").is_file():
                break
            time.sleep(0.1)
        time.sleep(1.5)
        try:
            os.kill(p.pid, signal.CTRL_BREAK_EVENT)
        except OSError as e:
            p.kill()
            p.wait()
            _skip("M6 Ctrl+Break gui cho nhom tien trinh", f"khong gui duoc CTRL_BREAK_EVENT trong moi truong nay: {e}")
            return
        rc = p.wait(timeout=60)
    finally:
        if p.poll() is None:
            p.kill()
    if not (out / "tom_tat.json").is_file():
        _check("M6 Ctrl+Break: cong cu ghi tom_tat sau khi con thoat", False, f"cong cu exit {rc}, khong co tom_tat")
        return
    tom = doc_json(out / "tom_tat.json")
    _check("M6 Ctrl+Break toi nhom (cong cu + con): cong cu ghi nhan tin hieu, KHONG chet truoc con, ghi tom_tat va "
           "tra dung ma cua con", tom["ctrl_c"] and (rc & 0xFFFFFFFF) == tom["exit_code"] and tom["exit_code"] != 0,
           f"cong cu {rc & 0xFFFFFFFF:#x}, con {tom['exit_code_hex']}, tin hieu {tom['ctrl_c']}")


def m7(tmp: Path):
    for ten, src, them in (("m7a", CON_IN_NHIEU, []), ("m7b", CON_TU_UTF8, ["--giu-moi-truong"])):
        out = tmp / ten
        con = script(tmp, f"{ten}.py", src)
        r = giam_sat(out, [PY, str(con)], them)
        dong = [ln.split("] ", 1)[1] for ln in (out / "log.txt").read_text(encoding="utf-8").split("\n")
                if ln.startswith("[+")]
        mong = (["Scanning ━━━━─── 0%", "Scanning ━━━━─── 33%", "Scanning ━━━━─── 66%", "Scanning ━━━━━━ 100%",
                 "emoji ✅ ±", "byte hong ��"] + [f"{i}/50000 ━─" for i in range(50000)] + ["cuoi khong newline"])
        _check(f"M7 {ten}: exit 0, log du {len(mong)} dong dung noi dung (\\r tach dong, byte hong -> U+FFFD, dong cuoi "
               f"khong newline)" + (" -- con tu reconfigure UTF-8, cong cu khong them env" if them else ""),
               r.returncode == 0 and dong == mong, f"exit {r.returncode}, {len(dong)} dong, lech dau tien "
               f"{next(((i, a, b) for i, (a, b) in enumerate(zip(dong, mong)) if a != b), None)}")


def m8(tmp: Path):
    import psutil
    pid_chet = max(psutil.pids()) + 4099
    while psutil.pid_exists(pid_chet):
        pid_chet += 4
    out = tmp / "m8a"
    r = subprocess.run([PY, str(CONG_CU), "gan", "--pid", str(pid_chet), "--create-time", "1.0", "--thu-muc", str(out)],
                       capture_output=True, text=True, timeout=60)
    _check("M8a gan PID khong ton tai -> exit 2, KHONG tao thu muc / mau", r.returncode == 2 and not out.exists(),
           r.stderr.strip()[-120:])
    song = subprocess.Popen([PY, "-c", "import time; x = bytearray(80 * 2**20); time.sleep(3)"])
    try:
        ct = psutil.Process(song.pid).create_time()
        out_b = tmp / "m8b"
        r = subprocess.run([PY, str(CONG_CU), "gan", "--pid", str(song.pid), "--create-time", str(ct - 5),
                            "--thu-muc", str(out_b)], capture_output=True, text=True, timeout=60)
        _check("M8b gan voi create_time SAI (PID tai su dung) -> exit 2, khong tao thu muc",
               r.returncode == 2 and not out_b.exists(), r.stderr.strip()[-120:])
        out_c = tmp / "m8c"
        r = subprocess.run([PY, str(CONG_CU), "gan", "--pid", str(song.pid), "--create-time", repr(ct), "--thu-muc",
                            str(out_c), "--chu-ky", "0.25"], capture_output=True, text=True, timeout=60)
        tom, rows = doc_json(out_c / "tom_tat.json"), doc_csv(out_c)
        _check("M8c gan PID hop le ket thuc giua chung -> exit 0, dung voi ly do, chi mau hop le (> 0), exit_code None",
               r.returncode == 0 and "ket thuc" in tom["ly_do_dung_lay_mau"] and rows
               and all(x["rss_chinh_mib"] and float(x["rss_chinh_mib"]) > 0 for x in rows) and tom["exit_code"] is None
               and {int(x["pid_chinh"]) for x in rows} == {song.pid}, f"{len(rows)} mau, ly do {tom['ly_do_dung_lay_mau']}")
    finally:
        song.kill()
        song.wait()


def m9(tmp: Path):
    out = tmp / "m9"
    out.mkdir()
    (out / "cu.txt").write_text("cu", encoding="utf-8")
    r = giam_sat(out, [PY, "-c", "print(1)"])
    _check("M9 thu muc dau ra da co noi dung -> exit 2, file cu giu nguyen, khong chay lenh",
           r.returncode == 2 and sorted(p.name for p in out.iterdir()) == ["cu.txt"], r.stderr.decode()[-120:])


def m10(tmp: Path):
    import psutil

    def tt(d):
        r = subprocess.run([PY, str(CONG_CU), "trang-thai", "--thu-muc", str(d)], capture_output=True, text=True,
                           timeout=60)
        return json.loads(r.stdout)
    a = tt(tmp / "m3")
    _check("M10a trang-thai luot da xong -> ket_thuc, exit 3", a["trang_thai"] == "ket_thuc" and a["exit_code"] == 3,
           str(a))
    d = tmp / "m10_chet"
    d.mkdir()
    (d / "trang_thai.json").write_text(json.dumps({"trang_thai": "dang_chay", "pid": 4, "create_time": 1.0,
                                                   "pid_cong_cu": 8, "create_time_cong_cu": 2.0}), encoding="utf-8")
    b = tt(d)
    _check("M10b trang_thai 'dang_chay' nhung ca hai PID khong con (create_time khac) -> bi_ngat_khong_co_exit_code",
           b["trang_thai"] == "bi_ngat_khong_co_exit_code", str(b))
    d2 = tmp / "m10_song"
    d2.mkdir()
    me = psutil.Process()
    (d2 / "trang_thai.json").write_text(json.dumps({"trang_thai": "dang_chay", "pid": me.pid,
                                                    "create_time": me.create_time(), "pid_cong_cu": None,
                                                    "create_time_cong_cu": None}), encoding="utf-8")
    c = tt(d2)
    _check("M10c PID con song dung create_time -> dang_chay", c["trang_thai"] == "dang_chay", str(c))


def m11(tmp: Path):
    r = subprocess.run([PY, str(CONG_CU), "tinh-lai", "--thu-muc", str(tmp / "m2")], capture_output=True, text=True,
                       timeout=60)
    _check("M11a tinh-lai tu CSV khop tom_tat (exit 0)", r.returncode == 0 and json.loads(r.stdout)["khop_tom_tat"],
           r.stdout[-200:])
    p = tmp / "m2" / "tom_tat.json"
    t = doc_json(p)
    t["ram"]["rss_tong_mib"]["max"] = 1.0
    p.write_text(json.dumps(t), encoding="utf-8")
    r = subprocess.run([PY, str(CONG_CU), "tinh-lai", "--thu-muc", str(tmp / "m2")], capture_output=True, text=True,
                       timeout=60)
    _check("M11b tom_tat bi sua -> tinh-lai phat hien (exit 1)", r.returncode == 1, r.stdout[-120:])


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="test_giam_sat_"))
    try:
        for ten, fn in (("M1 bam dung PID giua shell cung ten + python khac", m1), ("M2 worker tao / ket thuc", m2),
                        ("M3 thoat loi", m3), ("M4 bi kill", m4), ("M5 ma NTSTATUS", m5), ("M6 Ctrl+Break", m6),
                        ("M7 stdout UTF-8 / byte hong / khoi luong", m7), ("M8 gan PID", m8), ("M9 khong ghi de", m9),
                        ("M10 trang-thai", m10), ("M11 tinh-lai", m11)):
            _chay(ten, lambda fn=fn: fn(tmp))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n = sum(ok for _, ok in KQ)
    print(f"\nKET QUA: {n}/{len(KQ)} PASS" + (f", {len(SKIP)} SKIP (CHUA KIEM): {[s for s, _ in SKIP]}" if SKIP else ""))
    return 0 if KQ and n == len(KQ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
