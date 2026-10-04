# -*- coding: utf-8 -*-
"""Vong 42: DUNG luot training theo yeu cau nguoi dung ("tam dung" -> "Dung training ngay", 2026-10-04 ~08:33).

    python handoff/logs/review42/dung_luot.py --giam-sat <thu muc giam sat> [--cho-s 120]

1. Doc PID runner + PID cong cu tu trang_thai.json (doi chieu create_time).
2. Gui CTRL_C_EVENT vao console AN cua luot (AttachConsole(PID cong cu) + GenerateConsoleCtrlEvent, chinh tien trinh nay
   bo qua Ctrl+C). Runner co nhanh rieng cho KeyboardInterrupt: ghi ket qua do dang + nhat ky ket_thuc 'bi_ngat', exit 130.
3. Cho toi da --cho-s giay cho runner thoat. Neu van con: DUNG CUNG cay tien trinh runner (tien trinh con truoc, roi
   runner) = duong "kill roi resume" da kiem o test; cong cu giam sat tu ghi exit code khi runner het.
Khong dung toi checkpoint / nhat ky; khong resume."""
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import sys
import time
from pathlib import Path

import psutil


def iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def song(pid, ct) -> bool:
    try:
        return abs(psutil.Process(int(pid)).create_time() - float(ct)) <= 0.01
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError):
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--giam-sat", required=True)
    ap.add_argument("--cho-s", type=float, default=120)
    a = ap.parse_args()
    tt = json.loads((Path(a.giam_sat) / "trang_thai.json").read_text(encoding="utf-8"))
    pid, ct, pcc, ctcc = tt["pid"], tt["create_time"], tt["pid_cong_cu"], tt["create_time_cong_cu"]
    print(f"{iso()} truoc khi dung: runner {pid} song={song(pid, ct)}, cong cu {pcc} song={song(pcc, ctcc)}", flush=True)
    if not song(pid, ct):
        print("runner khong con chay -- khong lam gi")
        return 2
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.FreeConsole()
    if not k.AttachConsole(int(pcc)):
        print(f"{iso()} AttachConsole({pcc}) loi {ctypes.get_last_error()} -- bo qua Ctrl+C, chuyen sang dung cung")
        gui = False
    else:
        k.SetConsoleCtrlHandler(None, True)            # chinh tien trinh nay bo qua Ctrl+C
        gui = bool(k.GenerateConsoleCtrlEvent(0, 0))   # CTRL_C_EVENT toi moi tien trinh dung chung console cua luot
        err = ctypes.get_last_error()
        time.sleep(0.5)
        k.FreeConsole()
        print(f"{iso()} CTRL_C_EVENT gui={gui} (loi {err}) vao console cua cong cu {pcc}", flush=True)
    t0 = time.time()
    while time.time() - t0 < a.cho_s and song(pid, ct):
        time.sleep(2)
    if not song(pid, ct):
        print(f"{iso()} runner da thoat sau {time.time() - t0:.0f} s sau Ctrl+C", flush=True)
        cach = "ctrl_c"
    else:
        print(f"{iso()} runner VAN chay sau {a.cho_s:.0f} s -> dung cung cay tien trinh runner", flush=True)
        p = psutil.Process(int(pid))
        con = p.children(recursive=True)
        for c in con:
            try:
                c.kill()
            except psutil.Error:
                pass
        p.kill()
        psutil.wait_procs([p, *con], timeout=30)
        cach = "dung_cung"
    t1 = time.time()
    while time.time() - t1 < 60 and song(pcc, ctcc):      # cho cong cu ghi exit code + tom tat
        time.sleep(2)
    tt2 = json.loads((Path(a.giam_sat) / "trang_thai.json").read_text(encoding="utf-8"))
    print(f"{iso()} cach dung: {cach}; runner song={song(pid, ct)}, cong cu song={song(pcc, ctcc)}; trang_thai.json: "
          f"{tt2.get('trang_thai')} exit_code={tt2.get('exit_code')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
