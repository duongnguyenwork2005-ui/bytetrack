#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Vong 35 -- regression cho do_tai_nguyen_ngoai.py moc-gio (smoke lan 1: runner bao OSError [Errno 22] vi bo doc chet).

Chay:  python handoff/logs/review35/test_do_tai_nguyen_ngoai.py [--ma PATH]
  --ma PATH  ban do_tai_nguyen_ngoai.py can kiem (mac dinh: file canh test nay). Voi ban f27895a (ma da chay o smoke
             lan 1) phep T1/T2 PHAI FAIL (tai hien loi); ban da sua phai DAT het.
Ben ghi gia lap runner: reconfigure stdout sang UTF-8 nhu ultralytics.utils.set_logging tren Windows, ghi thanh tien do
"━─" ket thuc bang \\r, emoji, byte khong hop le UTF-8, dong cuoi khong co newline; tiep tuc ghi sau khi bo doc co the da
chet. Chi dung thu muc tam; khong cham data/, models/, results/.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

KET_QUA = []

BEN_GHI_T1 = r'''
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")      # nhu ultralytics.utils.set_logging khi stdout khong phai utf-8
try:
    sys.stdout.write("train: Scanning a ━━━━─── 50%\r"); sys.stdout.flush()
    sys.stdout.write("train: Scanning a ━━━━━━ 100%\n"); sys.stdout.flush()
    for i in range(20):
        time.sleep(0.05)
        sys.stdout.write("dong %d ✅ ±\n" % i); sys.stdout.flush()
    sys.stdout.buffer.write(b"byte hong \x81\xff\xfe\n"); sys.stdout.flush()
    sys.stdout.write("cuoi khong newline"); sys.stdout.flush()
except OSError as e:
    sys.stderr.write("BEN_GHI_OSERROR errno=%s %s\n" % (e.errno, e)); sys.stderr.flush()
    os._exit(7)
os._exit(0)
'''

BEN_GHI_T2 = r'''
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
try:
    for i in range(50000):
        sys.stdout.write("%d/50000 ━━━─%s" % (i, "\r" if i % 2 else "\n"))
    sys.stdout.write("XONG\n"); sys.stdout.flush()
except OSError as e:
    sys.stderr.write("BEN_GHI_OSERROR errno=%s %s\n" % (e.errno, e)); sys.stderr.flush()
    os._exit(7)
os._exit(0)
'''


def _check(ten, dk, chi_tiet=""):
    KET_QUA.append((ten, bool(dk)))
    print(f"[{'PASS' if dk else 'FAIL'}] {ten}" + (f" -- {chi_tiet}" if chi_tiet else ""), flush=True)


def moi_truong(utf8):
    """utf8=False: nhu smoke lan 1 (khong PYTHONIOENCODING/PYTHONUTF8 -> stdin pipe theo locale, vd cp1252);
    utf8=True: PYTHONIOENCODING=utf-8. Tien trinh con KHONG ke thua tuy y bien cua tien trinh test."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    if utf8:
        env["PYTHONIOENCODING"] = "utf-8"
    return env


def ma_hoa_stdin_pipe(env):
    r = subprocess.run([sys.executable, "-c", "import sys; print(sys.stdin.encoding, sys.stdin.errors)"],
                       stdin=subprocess.PIPE, capture_output=True, env=env, timeout=60)
    return r.stdout.decode("ascii", "replace").strip()


def chay_ong(ma, ben_ghi, log, env, timeout=120):
    """ben_ghi | python ma moc-gio --ra log; tra (rc ghi, stderr ghi, rc doc, stderr doc, thoi gian)."""
    t0 = time.time()
    g = subprocess.Popen([sys.executable, "-c", ben_ghi], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    d = subprocess.Popen([sys.executable, str(ma), "moc-gio", "--ra", str(log)], stdin=g.stdout,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    g.stdout.close()                                # chi tien trinh doc giu dau doc cua pipe
    _, e_d = d.communicate(timeout=timeout)
    e_g = g.stderr.read()
    g.wait(timeout=timeout)
    return g.returncode, e_g.decode("utf-8", "replace"), d.returncode, e_d.decode("utf-8", "replace"), time.time() - t0


def doc_log(log):
    dong, moc = [], []
    if not Path(log).is_file():
        return dong, moc
    for s in Path(log).read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^\[\+\s*([0-9.]+)s\] (.*)$", s)
        if m:
            moc.append(float(m.group(1)))
            dong.append(m.group(2))
    return dong, moc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ma", default=str(Path(__file__).with_name("do_tai_nguyen_ngoai.py")))
    a = ap.parse_args()
    ma = Path(a.ma).resolve()
    print(f"# ma kiem: {ma}\n# python: {sys.executable} {sys.version.split()[0]}", flush=True)
    with tempfile.TemporaryDirectory(prefix="test_moc_gio_") as td:
        for utf8, nhan in ((False, "E1 nhu smoke lan 1, khong PYTHONIOENCODING"), (True, "E2 PYTHONIOENCODING=utf-8")):
            env = moi_truong(utf8)
            print(f"# moi truong {nhan}: stdin pipe cua tien trinh con = {ma_hoa_stdin_pipe(env)}", flush=True)
            # T1 -- dung chuoi ky tu da lam chet bo doc o smoke lan 1
            log = Path(td) / f"t1_{int(utf8)}.log"
            rc_g, e_g, rc_d, e_d, dt = chay_ong(ma, BEN_GHI_T1, log, env)
            _check(f"{nhan[:2]} T1a ben ghi (gia lap runner) khong bi dut pipe: exit 0", rc_g == 0,
                   f"exit {rc_g} {e_g.strip()[:200]}")
            _check(f"{nhan[:2]} T1b bo doc moc-gio exit 0, khong traceback", rc_d == 0 and "Traceback" not in e_d,
                   f"exit {rc_d} {e_d.strip().splitlines()[-1][:200] if e_d.strip() else ''}")
            dong, moc = doc_log(log)
            mong = (["train: Scanning a ━━━━─── 50%", "train: Scanning a ━━━━━━ 100%"] +
                    [f"dong {i} ✅ ±" for i in range(20)] + ["byte hong ���", "cuoi khong newline"])
            _check(f"{nhan[:2]} T1c log du 24 dong dung noi dung (tach \\r va \\n, UTF-8, byte hong -> U+FFFD, "
                   "dong cuoi khong newline)", dong == mong, f"{len(dong)} dong; dong lech dau tien: "
                   f"{next(((i, x, y) for i, (x, y) in enumerate(zip(dong, mong)) if x != y), None)}")
            _check(f"{nhan[:2]} T1d moc gio khong giam va phu khoang ghi (>= 0,5 s)",
                   bool(moc) and all(x <= y for x, y in zip(moc, moc[1:])) and moc[-1] - moc[0] >= 0.5,
                   f"moc dau {moc[:1]} cuoi {moc[-1:]}")
            # T2 -- khoi lon, nhieu \r, ky tu thanh tien do: khong treo, du dong
            log2 = Path(td) / f"t2_{int(utf8)}.log"
            rc_g, e_g, rc_d, e_d, dt = chay_ong(ma, BEN_GHI_T2, log2, env)
            dong2, _ = doc_log(log2)
            _check(f"{nhan[:2]} T2a 50.000 dong (\\r/\\n xen ke) + XONG: ca hai ben exit 0", rc_g == 0 and rc_d == 0,
                   f"ghi {rc_g} doc {rc_d} {e_g.strip()[:120]} {e_d.strip()[-160:]}")
            _check(f"{nhan[:2]} T2b log du 50.001 dong, dung thu tu", len(dong2) == 50001 and bool(dong2)
                   and dong2[-1] == "XONG" and dong2[0] == "0/50000 ━━━─" and dong2[49999] == "49999/50000 ━━━─",
                   f"{len(dong2)} dong, {dt:.1f} s")
    n_dat = sum(ok for _, ok in KET_QUA)
    print(f"# tong: {n_dat}/{len(KET_QUA)} PASS", flush=True)
    return 0 if n_dat == len(KET_QUA) else 1


if __name__ == "__main__":
    sys.exit(main())
