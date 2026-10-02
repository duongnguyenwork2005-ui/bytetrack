"""
chay_giam_sat.py -- Khoi chay MOT lenh (vd training dai) va giam sat no bang PID DO CHINH CONG CU TAO RA (vong 39).

Thay cho `do_tai_nguyen_ngoai.py ram --mau` (vong 35/37, giu nguyen lam bang chung). Cach cu chon tien trinh DAU TIEN co
chuoi ten script trong command line, nen o vong 38 da bam nham shell cua lenh Monitor.

    python src/chay_giam_sat.py chay --thu-muc <thu muc MOI> [--chu-ky 1] [--khong-in] -- <lenh> [doi so ...]
    python src/chay_giam_sat.py gan --pid N --create-time T --thu-muc <thu muc MOI> [--chu-ky 1] [--toi-da-s S]
    python src/chay_giam_sat.py trang-thai --thu-muc <thu muc>
    python src/chay_giam_sat.py tinh-lai --thu-muc <thu muc>

`chay`
  - Popen(lenh, stdout=PIPE, stderr=STDOUT, stdin=DEVNULL). PID + create_time cua CHINH tien trinh con ghi vao
    trang_thai.json ngay khi tao. Mac dinh them PYTHONIOENCODING=utf-8, PYTHONUNBUFFERED=1 cho tien trinh con (chi khi
    chua dat; ghi vao trang_thai), de tien trinh con Python khong tu chet vi in ky tu ngoai cp1252 ra pipe;
    --giu-moi-truong de tat.
  - Doc stdout dang BYTE (os.read). log.txt ghi moi dong kem moc gio (tach \\n va \\r, UTF-8 errors=replace). Dong thoi in
    nguyen byte ra man hinh (tu tat neu ghi man hinh loi). KHONG BAO GIO ngung doc truoc EOF, nen tien trinh con khong
    bao gio bi dut pipe vi cong cu nay (su co vong 35).
  - Luong lay mau RAM moi --chu-ky giay tren psutil.Process(pid), moi mau kiem lai create_time:
    - ghi RSS tien trinh chinh, RSS TUNG tien trinh con (worker DataLoader tao / ket thuc giua chung duoc ghi dung luc)
      va RAM he thong con trong;
    - tien trinh chinh het, khong doc duoc, hoac PID bi tai su dung -> DUNG lay mau va ghi ly do;
    - KHONG BAO GIO ghi 0 thay cho so khong doc duoc (de trong + ghi_chu).
  - Ctrl+C: console gui Ctrl+C cho ca cong cu lan tien trinh con. Cong cu ghi lai thoi diem, KHONG thoat truoc con; cho
    con thoat roi ghi exit code.
  - Exit code cua cong cu = exit code THAT cua tien trinh con, ke ca ma NTSTATUS lon tren Windows. log.txt ket thuc bang
    `# exit=<ma>`; tom_tat.json va trang_thai.json ghi lai ma nay.
`gan`
  - Gan vao tien trinh DA CO (khong do cong cu tao). Phai cho --pid VA --create-time (lay tu psutil hoac trang_thai.json).
  - PID khong ton tai hoac create_time khac (PID tai su dung) -> tu choi, exit 2, khong ghi mau nao.
  - Khong co exit code cua tien trinh duoc gan.
`trang-thai`
  - Doc trang_thai.json, kiem PID cong cu / tien trinh con con song (doi chieu create_time).
  - Tra: dang_chay | ket_thuc (exit) | bi_ngat_khong_co_exit_code (vd dong cua so, mat dien).
  - Dung lenh nay thay cho viec DOAN trang thai tu mot monitor het han.
`tinh-lai`
  - Tinh lai tom tat tu ram_mau_tho.csv va doi chieu voi tom_tat.json.

RSS = working set tren Windows. Tong RSS cua nhieu tien trinh co the dem trung trang chia se, nen co the cao hon bo nho
thuc dung. Cong cu nay CHI do RAM; khong gop VRAM (runner ghi VRAM rieng bang nvidia-smi).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

COT = ["t_unix", "t_tuong_doi_s", "pid_chinh", "rss_chinh_mib", "so_con", "rss_con_mib", "rss_tong_mib",
       "ram_he_thong_con_trong_mib", "chi_tiet_con", "ghi_chu"]
GHI_CHU_RSS = ("RSS = working set Windows; rss_tong = chinh + moi tien trinh con (dem trung trang chia se -> co the cao "
               "hon bo nho thuc dung); ram_he_thong_con_trong = psutil.virtual_memory().available (toan may); KHONG gom VRAM")


def _iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def ma_thoat_giu_bit(code: int) -> int:
    """sys.exit() cua Python tren Windows tran voi ma >= 2**31 (vd NTSTATUS 0xC000013A -> 0xFFFFFFFF). Doi sang int32 co
    dau de mau bit (= ma tien trinh cha thay) duoc giu nguyen."""
    code = int(code) & 0xFFFFFFFF
    return code - (1 << 32) if code > 0x7FFFFFFF else code


def ghi_json_nguyen_tu(p: Path, o: dict) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(o, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def thu_muc_moi(d: Path) -> str | None:
    if d.exists() and (not d.is_dir() or any(d.iterdir())):
        return f"{d} da ton tai va khong rong -- khong ghi de ket qua giam sat cu"
    d.mkdir(parents=True, exist_ok=True)
    return None


# =========================================================================== lay mau RAM
class LayMau(threading.Thread):
    def __init__(self, pid: int, create_time: float, csv_path: Path, chu_ky: float, t0: float, toi_da_s: float = 0):
        super().__init__(daemon=True)
        self.pid, self.create_time, self.csv_path, self.chu_ky, self.t0 = pid, create_time, csv_path, chu_ky, t0
        self.toi_da_s = toi_da_s
        self.dung = threading.Event()
        self.n = 0
        self.ly_do_dung = None

    def run(self):
        import psutil
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(COT)
            f.flush()
            try:
                p = psutil.Process(self.pid)
            except psutil.NoSuchProcess:
                self.ly_do_dung = "tien trinh chinh khong ton tai khi bat dau lay mau"
                return
            while not self.dung.is_set():
                t = time.time()
                if self.toi_da_s and t - self.t0 > self.toi_da_s:
                    self.ly_do_dung = f"het thoi gian gan toi da {self.toi_da_s} s"
                    break
                ghi_chu = []
                try:
                    if abs(p.create_time() - self.create_time) > 0.01:
                        self.ly_do_dung = "PID bi tai su dung (create_time khac) -- dung, khong ghi mau"
                        break
                    if p.status() == psutil.STATUS_ZOMBIE:
                        self.ly_do_dung = "tien trinh chinh da ket thuc (zombie)"
                        break
                    rss_chinh = p.memory_info().rss
                    ds_con = p.children(recursive=True)
                except psutil.NoSuchProcess:
                    self.ly_do_dung = "tien trinh chinh da ket thuc"
                    break
                except psutil.AccessDenied:
                    w.writerow([round(t, 3), round(t - self.t0, 1), self.pid, "", "", "", "", "", "",
                                "AccessDenied tien trinh chinh -- khong doc duoc (khong ghi 0)"])
                    f.flush()
                    self.dung.wait(self.chu_ky)
                    continue
                rss_con, chi_tiet = 0, []
                for c in ds_con:
                    try:
                        r = c.memory_info().rss
                        rss_con += r
                        chi_tiet.append(f"{c.pid}:{c.name()}:{r / 2 ** 20:.0f}")
                    except psutil.NoSuchProcess:
                        ghi_chu.append(f"con {c.pid} ket thuc giua luc doc")
                    except psutil.AccessDenied:
                        ghi_chu.append(f"con {c.pid} AccessDenied (khong cong)")
                avail = psutil.virtual_memory().available
                w.writerow([round(t, 3), round(t - self.t0, 1), self.pid, round(rss_chinh / 2 ** 20, 1), len(chi_tiet),
                            round(rss_con / 2 ** 20, 1), round((rss_chinh + rss_con) / 2 ** 20, 1),
                            round(avail / 2 ** 20), ";".join(chi_tiet), "; ".join(ghi_chu)])
                f.flush()
                self.n += 1
                self.dung.wait(self.chu_ky)
            if self.ly_do_dung is None:
                self.ly_do_dung = "dung theo yeu cau (tien trinh chinh da thoat)"


def tinh_tom_tat(csv_path: Path) -> dict:
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8"))) if csv_path.is_file() else []

    def so(c):
        return [float(r[c]) for r in rows if r.get(c) not in ("", None)]

    def tk(x):
        if not x:
            return None
        s = sorted(x)
        return {"n": len(s), "min": s[0], "trung_vi": s[len(s) // 2], "max": s[-1]}
    t = so("t_unix")
    return {"so_mau": len(rows), "so_mau_khong_doc_duoc": sum(1 for r in rows if r.get("rss_chinh_mib") in ("", None)),
            "tu_t": min(t) if t else None, "den_t": max(t) if t else None,
            "khoang_do_s": round(max(t) - min(t), 1) if t else None,
            "pid_trong_csv": sorted({int(r["pid_chinh"]) for r in rows if r.get("pid_chinh")}),
            "rss_chinh_mib": tk(so("rss_chinh_mib")), "rss_tong_mib": tk(so("rss_tong_mib")),
            "so_con": tk(so("so_con")), "ram_he_thong_con_trong_mib": tk(so("ram_he_thong_con_trong_mib")),
            "so_mau_co_ghi_chu": sum(1 for r in rows if r.get("ghi_chu")), "ghi_chu_cot": GHI_CHU_RSS}


# =========================================================================== chay
def chay(a) -> int:
    import psutil
    out = Path(a.thu_muc)
    loi = thu_muc_moi(out)
    if loi:
        print(f"[LOI] {loi}", file=sys.stderr)
        return 2
    lenh = list(a.lenh)
    if lenh and lenh[0] == "--":
        lenh = lenh[1:]
    if not lenh:
        print("[LOI] thieu lenh sau --", file=sys.stderr)
        return 2
    t0 = time.time()
    env = dict(os.environ)
    env_them = {}
    if not a.giu_moi_truong:                     # tien trinh con Python ghi pipe bang UTF-8, khong dem -> khong tu chet vi
        for k, v in (("PYTHONIOENCODING", "utf-8"), ("PYTHONUNBUFFERED", "1")):   # ky tu ngoai cp1252
            if k not in env:
                env[k] = env_them[k] = v
    try:
        proc = subprocess.Popen(lenh, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                bufsize=0, env=env)
    except OSError as e:
        ghi_json_nguyen_tu(out / "trang_thai.json", {"trang_thai": "khong_khoi_dong_duoc", "loi": str(e), "lenh": lenh})
        print(f"[LOI] khong khoi dong duoc lenh: {e}", file=sys.stderr)
        return 2
    try:
        ct = psutil.Process(proc.pid).create_time()
    except psutil.NoSuchProcess:
        ct = None
    tt = {"trang_thai": "dang_chay", "pid_cong_cu": os.getpid(), "create_time_cong_cu": psutil.Process().create_time(),
          "pid": proc.pid, "create_time": ct, "lenh": lenh, "cwd": os.getcwd(), "bat_dau": _iso(), "t0": t0,
          "env_them_cho_tien_trinh_con": env_them}
    ghi_json_nguyen_tu(out / "trang_thai.json", tt)
    lm = None
    if ct is not None:
        lm = LayMau(proc.pid, ct, out / "ram_mau_tho.csv", a.chu_ky, t0)
        lm.start()
    ctrl_c = []

    def nhan_tin_hieu(sig, frame):              # khong thoat truoc tien trinh con; con cung nhan Ctrl+C tu console
        ctrl_c.append({"tin_hieu": int(sig), "t_tuong_doi_s": round(time.time() - t0, 1)})
    signal.signal(signal.SIGINT, nhan_tin_hieu)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, nhan_tin_hieu)
    echo, loi_in, so_dong = not a.khong_in, 0, 0
    fd = proc.stdout.fileno()
    with open(out / "log.txt", "w", encoding="utf-8", errors="backslashreplace", newline="\n") as log:
        log.write(f"# {_iso()} chay_giam_sat: pid {proc.pid} create_time {ct}\n# lenh: {subprocess.list2cmdline(lenh)}\n")
        log.flush()

        def ghi(dong: bytes):
            nonlocal so_dong
            s = dong.decode("utf-8", "replace")
            if s.strip():
                log.write(f"[+{time.time() - t0:9.1f}s] {s}\n")
                log.flush()
                so_dong += 1
        buf = b""
        while True:
            try:
                khoi = os.read(fd, 65536)
            except OSError:
                break
            if not khoi:
                break
            if echo:
                try:
                    sys.stdout.buffer.write(khoi)
                    sys.stdout.buffer.flush()
                except Exception:  # noqa: BLE001 -- man hinh loi khong duoc lam dut pipe cua tien trinh con
                    echo, loi_in = False, loi_in + 1
            phan = re.split(rb"[\r\n]", buf + khoi)
            buf = phan.pop()
            for p_ in phan:
                ghi(p_)
        ghi(buf)
        code = proc.wait()
        t1 = time.time()
        if lm is not None:
            lm.dung.set()
            lm.join(timeout=10)
        log.write(f"# {_iso()} ket thuc sau {t1 - t0:.1f} s\n# exit={code}\n")
    tom = {"lenh": lenh, "pid": proc.pid, "create_time": ct, "exit_code": code,
           "exit_code_hex": f"0x{code & 0xFFFFFFFF:08X}", "bat_dau": tt["bat_dau"], "ket_thuc": _iso(),
           "thoi_gian_s": round(t1 - t0, 1), "so_dong_log": so_dong, "ctrl_c": ctrl_c, "loi_in_man_hinh": loi_in,
           "chu_ky_s": a.chu_ky, "ly_do_dung_lay_mau": lm.ly_do_dung if lm else "khong lay mau (tien trinh het truoc)",
           "ram": tinh_tom_tat(out / "ram_mau_tho.csv")}
    ghi_json_nguyen_tu(out / "tom_tat.json", tom)
    ghi_json_nguyen_tu(out / "trang_thai.json", {**tt, "trang_thai": "ket_thuc", "exit_code": code, "ket_thuc": _iso()})
    return code


# =========================================================================== gan
def gan(a) -> int:
    import psutil
    try:
        p = psutil.Process(a.pid)
        ct = p.create_time()
    except psutil.NoSuchProcess:
        print(f"[LOI] PID {a.pid} khong ton tai -- khong gan, khong ghi mau", file=sys.stderr)
        return 2
    if abs(ct - a.create_time) > 0.01:
        print(f"[LOI] PID {a.pid} co create_time {ct} khac {a.create_time} (PID bi tai su dung?) -- khong gan",
              file=sys.stderr)
        return 2
    out = Path(a.thu_muc)
    loi = thu_muc_moi(out)
    if loi:
        print(f"[LOI] {loi}", file=sys.stderr)
        return 2
    try:
        cmd = p.cmdline()
    except psutil.Error:
        cmd = None
    t0 = time.time()
    lm = LayMau(a.pid, ct, out / "ram_mau_tho.csv", a.chu_ky, t0, a.toi_da_s)
    lm.start()
    dung = []
    signal.signal(signal.SIGINT, lambda s, f: (dung.append(1), lm.dung.set()))
    while lm.is_alive():
        lm.join(timeout=0.5)
    tom = {"che_do": "gan", "pid": a.pid, "create_time": ct, "cmdline": cmd, "bat_dau": dt.datetime.fromtimestamp(t0)
           .astimezone().isoformat(timespec="seconds"), "ket_thuc": _iso(), "ly_do_dung_lay_mau": lm.ly_do_dung,
           "exit_code": None, "ghi_chu": "gan vao tien trinh co san -> khong co exit code", "chu_ky_s": a.chu_ky,
           "ram": tinh_tom_tat(out / "ram_mau_tho.csv")}
    ghi_json_nguyen_tu(out / "tom_tat.json", tom)
    return 0


# =========================================================================== trang-thai / tinh-lai
def song(pid, create_time) -> bool:
    import psutil
    if pid is None or create_time is None:
        return False
    try:
        return abs(psutil.Process(int(pid)).create_time() - float(create_time)) <= 0.01
    except psutil.NoSuchProcess:
        return False


def trang_thai(a) -> int:
    p = Path(a.thu_muc) / "trang_thai.json"
    if not p.is_file():
        print(json.dumps({"trang_thai": "khong_co_trang_thai", "thu_muc": str(a.thu_muc)}, ensure_ascii=False))
        return 2
    tt = json.loads(p.read_text(encoding="utf-8"))
    if tt.get("trang_thai") == "ket_thuc":
        kq = {"trang_thai": "ket_thuc", "exit_code": tt.get("exit_code"), "ket_thuc": tt.get("ket_thuc")}
    elif tt.get("trang_thai") == "dang_chay":
        con, cc = song(tt.get("pid"), tt.get("create_time")), song(tt.get("pid_cong_cu"), tt.get("create_time_cong_cu"))
        kq = {"trang_thai": "dang_chay" if (con or cc) else "bi_ngat_khong_co_exit_code",
              "tien_trinh_con_song": con, "cong_cu_song": cc, "pid": tt.get("pid"), "bat_dau": tt.get("bat_dau"),
              "ghi_chu": None if (con or cc) else "ca cong cu lan tien trinh con khong con chay ma chua ghi ket_thuc "
                                                  "(vd dong cua so terminal / mat dien) -> khong co exit code"}
    else:
        kq = tt
    print(json.dumps(kq, ensure_ascii=False))
    return 0


def tinh_lai(a) -> int:
    d = Path(a.thu_muc)
    lai = tinh_tom_tat(d / "ram_mau_tho.csv")
    tom = json.loads((d / "tom_tat.json").read_text(encoding="utf-8")) if (d / "tom_tat.json").is_file() else {}
    khop = tom.get("ram") == lai
    print(json.dumps({"tinh_lai": lai, "khop_tom_tat": khop}, ensure_ascii=False, indent=1))
    return 0 if khop else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("chay")
    a1.add_argument("--thu-muc", required=True)
    a1.add_argument("--chu-ky", type=float, default=1.0)
    a1.add_argument("--khong-in", action="store_true", help="khong in stdout cua lenh ra man hinh (van ghi log)")
    a1.add_argument("--giu-moi-truong", action="store_true",
                    help="KHONG them PYTHONIOENCODING=utf-8 / PYTHONUNBUFFERED=1 cho tien trinh con")
    a1.add_argument("lenh", nargs=argparse.REMAINDER)
    a2 = sub.add_parser("gan")
    a2.add_argument("--pid", type=int, required=True)
    a2.add_argument("--create-time", type=float, required=True)
    a2.add_argument("--thu-muc", required=True)
    a2.add_argument("--chu-ky", type=float, default=1.0)
    a2.add_argument("--toi-da-s", type=float, default=0, help="dung sau S giay (0 = toi khi tien trinh het)")
    a3 = sub.add_parser("trang-thai")
    a3.add_argument("--thu-muc", required=True)
    a4 = sub.add_parser("tinh-lai")
    a4.add_argument("--thu-muc", required=True)
    a = ap.parse_args()
    return {"chay": chay, "gan": gan, "trang-thai": trang_thai, "tinh-lai": tinh_lai}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(ma_thoat_giu_bit(main()))
