# -*- coding: utf-8 -*-
"""Vong 42: theo doi luot training dai -- CHI DOC (khong gui tin hieu, khong ghi vao thu muc luot / thu muc giam sat).

    python handoff/logs/review42/theo_doi_luot.py --giam-sat <thu muc giam sat> --run-dir <thu muc luot> --log <file>
        [--toi-da-s 6600] [--chu-ky-s 60] [--moc-dinh-ky-s 1800] [--dung-yen-s 1200] [--thoat-khi-epoch-moi]
    python handoff/logs/review42/theo_doi_luot.py ... --mot-lan        # chi chup mot anh roi thoat

Moi anh chup = MOT dong JSON noi vao --log (luc bat dau, moi epoch moi luu, dinh ky, khi ket thuc / bi ngat / dung yen /
het gio cho) va in ra stdout. Nguon: trang_thai.json cua chay_giam_sat (PID + create_time), nhat_ky_luot.jsonl cua runner
(su kien bat_dau / epoch_da_luu / ket_thuc), dong cuoi log.txt (tien do ultralytics), psutil (RSS tien trinh con cua cong
cu + cac tien trinh con cua no, RAM toan may, pin), nvidia-smi, dung luong dia.
Thoat 0 trong moi truong hop doc duoc. HET GIO CHO KHONG CO NGHIA TRAINING THAT BAI -- chi la het mot luot cho."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import psutil

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\[K")
EPOCH = re.compile(r"^\s*(\d+)/(\d+)\s+(\S+G)\s")
BATCH = re.compile(r"(\d+)%\s+\S+\s+(\d+)/(\d+)")


def iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def song(pid, ct) -> bool:
    if pid is None or ct is None:
        return False
    try:
        return abs(psutil.Process(int(pid)).create_time() - float(ct)) <= 0.01
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
        return False


def doc_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def nhat_ky(run_dir: Path) -> dict:
    p = run_dir / "nhat_ky_luot.jsonl"
    ep, kt, bd, hong = [], None, [], 0
    if p.is_file():
        for dong in p.read_bytes().split(b"\n"):
            if not dong.strip():
                continue
            try:
                e = json.loads(dong)
            except json.JSONDecodeError:
                hong += 1
                continue
            loai = e.get("loai")
            if loai == "epoch_da_luu":
                m = e.get("metrics") or {}
                ep.append({"lan": e.get("lan"), "epoch": e.get("epoch"), "fitness": e.get("fitness"),
                           "best_fitness": e.get("best_fitness"), "dung": e.get("dung"),
                           "mAP50-95(B)": m.get("metrics/mAP50-95(B)"), "mAP50(B)": m.get("metrics/mAP50(B)"),
                           "best_ghi": e.get("best_ghi_epoch_nay"), "t": e.get("t_iso")})
            elif loai == "ket_thuc":
                kt = {k: e.get(k) for k in ("lan", "trang_thai", "exit_code", "kiem_sau_train_dat", "loi",
                                            "epoch_da_luu_lan_nay", "t_iso")}
            elif loai == "bat_dau":
                bd.append({k: e.get(k) for k in ("lan", "che_do_luot", "start_epoch", "tong_epoch", "pid", "t_iso")})
    return {"co_file": p.is_file(), "bat_dau": bd, "epoch_da_luu": ep, "ket_thuc": kt, "dong_hong": hong}


def tien_do(log: Path) -> dict:
    if not log.is_file():
        return {"co_log": False}
    st = log.stat()
    with open(log, "rb") as f:
        f.seek(max(0, st.st_size - 65536))
        dong = [ANSI.sub("", x.decode("utf-8", "replace")) for x in f.read().split(b"\n") if x.strip()]
    kq = {"co_log": True, "log_bytes": st.st_size, "log_tuoi_s": round(time.time() - st.st_mtime, 1)}
    for x in reversed(dong):
        noi_dung = x.split("] ", 1)[-1] if x.startswith("[+") else x
        m, b = EPOCH.match(noi_dung), BATCH.search(noi_dung)
        if "mAP50" in noi_dung and b:
            kq.update(pha="val", val_phan_tram=int(b.group(1)), val_batch=f"{b.group(2)}/{b.group(3)}")
            continue
        if m:
            kq.update(epoch_dang_chay=f"{m.group(1)}/{m.group(2)}", gpu_mem_ultralytics=m.group(3))
            if b:
                kq.update(batch=f"{b.group(2)}/{b.group(3)}", phan_tram=int(b.group(1)))
            kq.setdefault("pha", "train")
            break
    kq["dong_cuoi"] = (dong[-1][:220] if dong else None)
    return kq


def ram(pid, ct) -> dict:
    out = {"ram_he_thong_con_trong_mib": round(psutil.virtual_memory().available / 2 ** 20),
           "pagefile_dung_gib": round(psutil.swap_memory().used / 2 ** 30, 2)}
    if song(pid, ct):
        try:
            p = psutil.Process(int(pid))
            con = p.children(recursive=True)
            rc = 0
            for c in con:
                try:
                    rc += c.memory_info().rss
                except psutil.Error:
                    pass
            out.update(rss_chinh_mib=round(p.memory_info().rss / 2 ** 20), so_con=len(con),
                       rss_con_mib=round(rc / 2 ** 20))
        except psutil.Error as e:
            out["loi_doc"] = type(e).__name__
    return out


def gpu() -> dict:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw,pstate",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        mem, util, temp, pw, ps = (x.strip() for x in r.stdout.strip().splitlines()[0].split(","))
        return {"vram_dung_mib": int(mem), "util_pct": int(util), "nhiet_c": int(temp), "w": pw, "pstate": ps}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {"loi": "nvidia-smi khong doc duoc"}


def anh_chup(gs: Path, run_dir: Path, su_kien: str) -> dict:
    tt = doc_json(gs / "trang_thai.json")
    if tt is None:
        trang = "chua_co_trang_thai"
        song_con = song_cc = False
    else:
        song_con, song_cc = song(tt.get("pid"), tt.get("create_time")), song(tt.get("pid_cong_cu"),
                                                                             tt.get("create_time_cong_cu"))
        trang = tt.get("trang_thai")
        if trang == "dang_chay" and not (song_con or song_cc):
            trang = "bi_ngat_khong_co_exit_code"
    nk = nhat_ky(run_dir)
    b = psutil.sensors_battery()
    return {"t": iso(), "su_kien": su_kien, "trang_thai": trang, "exit_code": (tt or {}).get("exit_code"),
            "pid": (tt or {}).get("pid"), "pid_cong_cu": (tt or {}).get("pid_cong_cu"), "song_con": song_con,
            "song_cong_cu": song_cc, "epoch_da_luu": [e["epoch"] for e in nk["epoch_da_luu"]],
            "epoch_cuoi": nk["epoch_da_luu"][-1] if nk["epoch_da_luu"] else None, "ket_thuc_nhat_ky": nk["ket_thuc"],
            "bat_dau_nhat_ky": nk["bat_dau"][-1] if nk["bat_dau"] else None, "nhat_ky_dong_hong": nk["dong_hong"],
            "tien_do": tien_do(gs / "log.txt"), "ram": ram((tt or {}).get("pid"), (tt or {}).get("create_time")),
            "gpu": gpu(), "pin": {"pct": b.percent, "cam_sac": b.power_plugged} if b else None,
            "dia_trong_gib": {"C": round(shutil.disk_usage("C:\\").free / 2 ** 30, 1),
                              "D": round(shutil.disk_usage("D:\\").free / 2 ** 30, 1)}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--giam-sat", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--toi-da-s", type=float, default=6600)
    ap.add_argument("--chu-ky-s", type=float, default=60)
    ap.add_argument("--moc-dinh-ky-s", type=float, default=1800)
    ap.add_argument("--dung-yen-s", type=float, default=1200, help="log.txt khong doi qua so giay nay khi con chay")
    ap.add_argument("--thoat-khi-epoch-moi", action="store_true")
    ap.add_argument("--mot-lan", action="store_true")
    a = ap.parse_args()
    gs, rd, lg = Path(a.giam_sat), Path(a.run_dir), Path(a.log)

    def ghi(s: dict) -> None:
        dong = json.dumps(s, ensure_ascii=False)
        with open(lg, "a", encoding="utf-8", newline="\n") as f:
            f.write(dong + "\n")
        print(dong, flush=True)

    t0 = time.time()
    s = anh_chup(gs, rd, "anh_chup" if a.mot_lan else "bat_dau_theo_doi")
    ghi(s)
    if a.mot_lan:
        return 0
    n0, moc = len(s["epoch_da_luu"]), time.time()
    while True:
        time.sleep(a.chu_ky_s)
        s = anh_chup(gs, rd, "kiem")
        if s["trang_thai"] == "bi_ngat_khong_co_exit_code":
            time.sleep(10)                         # tranh dua voi luc cong cu dang ghi ket_thuc
            s = anh_chup(gs, rd, "kiem")
        if s["trang_thai"] in ("ket_thuc", "bi_ngat_khong_co_exit_code", "khong_khoi_dong_duoc"):
            s["su_kien"] = s["trang_thai"]
            ghi(s)
            return 0
        tuoi = (s.get("tien_do") or {}).get("log_tuoi_s")
        if tuoi is not None and tuoi > a.dung_yen_s:
            s["su_kien"] = f"dung_yen (log.txt khong doi {tuoi:.0f} s)"
            ghi(s)
            return 0
        if len(s["epoch_da_luu"]) > n0:
            s["su_kien"] = "epoch_moi"
            ghi(s)
            if a.thoat_khi_epoch_moi:
                return 0
            n0 = len(s["epoch_da_luu"])
        if time.time() - moc >= a.moc_dinh_ky_s:
            s["su_kien"] = "dinh_ky"
            ghi(s)
            moc = time.time()
        if time.time() - t0 >= a.toi_da_s:
            s["su_kien"] = "het_gio_cho (training van chay; khong phai that bai)"
            ghi(s)
            return 0


if __name__ == "__main__":
    sys.exit(main())
