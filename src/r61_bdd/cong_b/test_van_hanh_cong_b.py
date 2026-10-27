"""
test_van_hanh_cong_b.py -- Fixture CPU cua lop van hanh cong B (plan chuan bi B0/B1, 26/10/2026; sua B0-F1..F9).

    C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -B src/r61_bdd/cong_b/test_van_hanh_cong_b.py \
        --phien results/phase2/r61_bdd/tests/cong_b_20261026/lan_02 --ngan-sach-giay 420 --gioi-han-giay 540

(Q9MG/QUVZ 27/10: lan_01 dung hoi sau crash native cua fixture pid12248 -> lenh B1 doi sang lan_02; lan_01 CHI DOC, xem
CHUYEN_PHIEN / chuyen_phien().)

Du lieu TONG HOP (the gioi / giao thuc / ModelGia cua test_chuan_bi_bdd; khong GPU, khong du lieu that). Dung A tong hop
DUNG cau truc goi A da nghiem thu bang helper hien co (KHONG chay runner/fixture A cu). Moi lenh van hanh B chay qua DUNG
CLI [python, -B, van_hanh_cong_b.py, ...]; moi lenh nhan --han-chot = han chung cua lan goi (khong cap lai ngan sach) va
--so-doan 1.
Cha fixture CHI thu vien chuan + numpy (P43284b78): moi buoc can thu vien ben thu ba (the gioi / tien kiem A / cache / goi A
+ anh khoa / doi chieu bootstrap) chay qua lenh con [python, -B, THIS, --viec <ten> <vong> <han> <mem> <lenh>] trong job
(viec_con -> lenh()); sys.modules cua cha ghi o moi ket_thuc (lan_goi + khoi_dong) va tong_ket chi dat khi khong co
ultralytics/torch, ngoai thu vien chuan chi numpy.
Bao ve toan lan goi (F1/F2):
  - TRAN CUNG: faulthandler.dump_traceback_later(exit=True) dat NGAY dau module (truoc chup pyc / import ma du an) o
    bat_dau + --gioi-han-giay - 8 s; bao ca import, thiet lap, tai tinh, chup pyc cuoi, thu hoi.
  - JOB GOC: fixture tu gan vao Job Object KILL_ON_JOB_CLOSE TRUOC khi tao bat ky con nao -> con thuoc job goc ngay tu
    CreateProcess (khong co khoang ho Popen -> Assign); fixture chet o bat ky dau -> kernel dong handle -> ca cay bi giet.
    Moi lenh con con co job lenh rieng (long trong job goc) de dem / thu hoi cay theo lenh; loi Python giua Popen va
    Assign -> giet con + TerminateJobObject truoc khi nem lai.
  - HAN MEM: khong bat dau lenh con / buoc thiet lap / ca / tai tinh trong tien trinh khi qua ngan sach mem (420 s) hoac
    khong con du du phong toi han chung (= tran - du phong ket thuc, du phong >= 2 x thoi gian chup pyc + 25 s) -> exit 10.
Checkpoint: lenh con <vong>/lenh/<ten>.json (co -> dung lai, KHONG chay lai); ca <vong>/ca/<NN>_<ten>.json (kem danh muc
kiem BAT BUOC 'can'); quan sat / tai tinh nang <vong>/nho/; buoc thiet lap <vong>/thiet_lap/ (cache tiep tuc theo video).
Moi vong <phien>/vong_NN/ gan snapshot ma; ma doi -> vong moi.
So crash ngoai du kien (F8): <phien>/so_crash_ngoai_du_kien.json.
  - F8a: journal khoi dong <phien>/khoi_dong/<id>.bat_dau.json (thu vien chuan, sau kiem doi so / duong, TRUOC chup pyc va
    import ma du an) + <id>.ket_thuc.json khi ket thuc co kiem soat; thieu ket_thuc = lan goi bi ngat o bat ky giai doan.
  - F8b: loi tiem DA XAC NHAN theo loai (diem/kieu tiem giai thich dung phan loai) ghi 'tiem'; con lai ghi 'muc' (ngoai du
    kien). Ra soat MOI vong + journal o dau / cuoi moi lan goi VA ngay sau MOI lenh con (truoc lenh ke tiep). Moi muc ngoai
    du kien khong phai crash native (timeout, traceback, exit 1, thieu log, Job/resume, PID, cha gian doan, fixture bi ngat,
    cay chua thu hoi) -> dung ngay; crash native ngoai du kien -> dung o su co thu 4.
  - F8c: dem theo su_co duy nhat (ket qua lenh + ket_thuc lan chay cung stt = mot su co).
  Dung -> <phien>/dung_fixture.json -> moi lan goi sau exit 40 (DUNG HOI), kiem TRUOC chon vong; khong reset khi doi vong.
Bytecode: chup .pyc (danh sach, sha256, size, mtime_ns) trong pham vi hop dong TRUOC import ma du an/ben thu ba va SAU khi
thu hoi cay con, moi lan goi; khong xoa pyc.
Ket luan (F7): moi lan goi khi ma tran du, TINH LAI tu bang chung day du (moi ca co mat, moi kiem bat buoc co va dat, so
sanh pyc MOI lan goi cua vong dat, khong crash ngoai du kien trong vong); ket_qua/lan_NN.json giu lich su, ket_qua_test.json
= ket luan lan goi cuoi. Exit: 0 dat; 10 con viec (chay lai DUNG lenh); 2 khong dat / doi so sai; 40 dung hoi.
"""
from __future__ import annotations

import faulthandler
import hashlib
import json
import os
import sys
import sysconfig
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
T_BAT_DAU = time.time()
ROOT = Path(__file__).resolve().parents[3]
THIS = Path(__file__).resolve()
TRAN_TOI_DA = 540.0
DU_PHONG_TRAN = 8.0           # tran cung = bat dau + gioi han - 8 s (de kernel thu hoi cay truoc 540 s)

if sys.argv[1:2] == ["--ngu"]:                         # CHAU cua tham do: chi ngu, bi thu hoi theo job; khong import gi them
    time.sleep(float(sys.argv[2]))
    sys.exit(0)
CHE_DO = "tham_do" if sys.argv[1:2] == ["--tham-do"] else "viec" if sys.argv[1:2] == ["--viec"] else "chinh"
CAM_CHA = ("ultralytics", "torch")             # P43284b78: cha fixture KHONG duoc nap (buoc can chung chay qua --viec)
CHO_PHEP_CHA = ("numpy",)                      # ngoai thu vien chuan, cha chi duoc nap them goi nay


def _goc_site() -> list[str]:
    """Ten goc cac module da nap tu site-packages (CHI thu vien chuan; khong import gi)."""
    sp = os.path.normcase(os.path.abspath(sysconfig.get_paths()["purelib"]))
    return sorted({n.split(".")[0] for n, m in list(sys.modules.items())
                   if os.path.normcase(os.path.abspath(str(getattr(m, "__file__", None) or os.devnull))).startswith(sp + os.sep)})


_TB_DAU = _goc_site()                          # nap san luc khoi dong (.pth cua site, vd _distutils_hack), truoc moi import


_TAO_TOI: list = []                            # thoi diem tao (FILETIME) cua CHINH tien trinh nay, doc mot lan


def _tao_toi():
    if not _TAO_TOI:
        try:
            _TAO_TOI.append(VA.thoi_diem_tao(VA._k32().GetCurrentProcess()))
        except Exception:                      # noqa: BLE001 -- khong doc duoc -> ghi None (bang chung thieu, khong doan)
            _TAO_TOI.append(None)
    return _TAO_TOI[0]


def ben_thu_ba(diem: str | None = None, day_du: bool = False) -> dict:
    """Bang chung sys.modules THUC TE cua tien trinh nay (khong xoa module nao): goi ben thu ba da nap, phan nap them sau
    khoi dong, goi cam co mat; gan pid + thoi diem tao + lan goi + snapshot + diem kiem (Q73M)."""
    goc = _goc_site()
    them = [n for n in goc if n not in _TB_DAU]
    out = {"pid": os.getpid(), "thoi_diem_tao": _tao_toi(), "lan": getattr(T, "LAN", None),
           "snapshot": getattr(T, "SNAP", None), "diem": diem, "luc": round(time.time(), 3), "so_module": len(sys.modules),
           "da_nap": goc, "luc_khoi_dong": _TB_DAU, "them_sau_khoi_dong": them,
           "cam_co_mat": [n for n in CAM_CHA if any(k == n or k.startswith(n + ".") for k in list(sys.modules))],
           "ngoai_cho_phep": [n for n in them if n not in CHO_PHEP_CHA]}
    if day_du:
        sp = os.path.normcase(os.path.abspath(sysconfig.get_paths()["purelib"]))
        out["module_site_packages"] = sorted(
            n for n, m in list(sys.modules.items())
            if os.path.normcase(os.path.abspath(str(getattr(m, "__file__", None) or os.devnull))).startswith(sp + os.sep))
    return out


def tb_sach(x) -> bool:
    return isinstance(x, dict) and not x.get("cam_co_mat") and not x.get("ngoai_cho_phep")


def _tran_som(argv: list[str]) -> float:
    """Doc --gioi-han-giay SOM chi de dat tran cung truoc import / chup pyc (kiem day du o chay())."""
    try:
        return min(TRAN_TOI_DA, float(argv[argv.index("--gioi-han-giay") + 1]))
    except (ValueError, IndexError):
        return TRAN_TOI_DA


if CHE_DO == "chinh":                                  # F1: tran cung bao ca import / chup pyc / thiet lap / ket thuc
    faulthandler.dump_traceback_later(max(1.0, T_BAT_DAU + _tran_som(sys.argv[1:]) - DU_PHONG_TRAN - time.time()), exit=True)

# ---------------------------------------------------------------------------
# F8a: journal KHOI DONG (CHI thu vien chuan) -- sau kiem doi so / duong ghi, TRUOC chup pyc va import ma du an.
# <phien>/khoi_dong/<id>.bat_dau.json (fsync) ... <id>.ket_thuc.json (ghi khi lan goi ket thuc co kiem soat). Thieu
# ket_thuc o lan goi sau = lan goi bi ngat (crash / tran cung / kill) o BAT KY giai doan, ke ca truoc lan_goi cua vong.
# ---------------------------------------------------------------------------
VUNG_SOM = ROOT / "results" / "phase2" / "r61_bdd" / "tests" / "cong_b_20261026"
KHOI_DONG: dict = {"id": None, "dir": None, "fh": None, "loi": None}
MODULE_DU_AN = ("hop_dong_bdd", "kiem_dau_vao", "van_hanh_cong_b", "dieu_phoi_bdd", "van_hanh_cong_a", "numpy", "pandas")


def _reparse_som(p: Path) -> bool:
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return False
    return os.path.islink(p) or bool(getattr(st, "st_file_attributes", 0) & 0x400)   # FILE_ATTRIBUTE_REPARSE_POINT


def _kiem_duong_som(p: Path) -> str | None:
    """Lexical (duoi VUNG_SOM, khac VUNG_SOM) + realpath = abspath + KHONG reparse o moi thanh phan da ton tai."""
    a, v = os.path.normcase(os.path.abspath(p)), os.path.normcase(os.path.abspath(VUNG_SOM))
    if a == v or not a.startswith(v + os.sep):
        return f"{p} khong nam duoi {VUNG_SOM}"
    if os.path.normcase(os.path.realpath(a)) != a:
        return f"{p} resolve thanh {os.path.realpath(a)}"
    cur = Path(os.path.abspath(p))
    for part in [cur, *cur.parents]:
        if _reparse_som(part):
            return f"{p} di qua reparse {part}"
    return None


def _tao_som(d: Path) -> None:
    """mkdir tung cap (chi duoi VUNG_SOM), kiem lai reparse / realpath SAU moi mkdir."""
    chua = []
    cur = Path(os.path.abspath(d))
    while not cur.exists():
        chua.append(cur)
        cur = cur.parent
    for x in reversed(chua):
        if not os.path.normcase(str(x)).startswith(os.path.normcase(os.path.abspath(VUNG_SOM))):
            raise OSError(f"tao {x} ngoai {VUNG_SOM}")
        x.mkdir(exist_ok=False)
        loi = _kiem_duong_som(x) if x != Path(os.path.abspath(VUNG_SOM)) else None
        if loi or _reparse_som(x):
            raise OSError(f"sau mkdir: {loi or x}")


def _ghi_som(p: Path, obj) -> None:
    with open(p, "x", encoding="utf-8") as f:          # "x": khong ghi de
        f.write(json.dumps(obj, ensure_ascii=False, indent=1) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _doi_so_som(argv: list[str]) -> Path | None:
    """--phien / --ngan-sach-giay / --gioi-han-giay hop le (cung dieu kien voi chay()) -> duong phien; sai -> None."""
    try:
        ph = Path(os.path.abspath(ROOT / argv[argv.index("--phien") + 1]))
        ns_, gh = float(argv[argv.index("--ngan-sach-giay") + 1]), float(argv[argv.index("--gioi-han-giay") + 1])
    except (ValueError, IndexError):
        return None
    return ph if 0 < ns_ <= 420 and ns_ < gh <= TRAN_TOI_DA else None


def khoi_dong_som(ph: Path) -> None:
    """Journal khoi dong + log faulthandler rieng (tran cung ghi vao day den khi bao_ve_lan_goi dat lai)."""
    loi = _kiem_duong_som(ph) or _kiem_duong_som(ph / "khoi_dong")
    if loi:
        KHOI_DONG["loi"] = loi
        return
    d = ph / "khoi_dong"
    _tao_som(d)
    kid = time.strftime("%Y%m%d_%H%M%S", time.localtime(T_BAT_DAU)) + f"_{int(T_BAT_DAU * 1000) % 1000:03d}_pid{os.getpid()}"
    _ghi_som(d / f"{kid}.bat_dau.json", {"id": kid, "che_do": CHE_DO, "pid": os.getpid(), "bat_dau": T_BAT_DAU,
                                         "executable": sys.executable, "orig_argv": list(getattr(sys, "orig_argv", [])),
                                         "luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "ghi_luc_epoch": time.time(),
                                         "module_du_an_da_nap": sorted(n for n in MODULE_DU_AN if n in sys.modules)})
    fh = open(d / f"{kid}.faulthandler.log", "x", encoding="utf-8")
    faulthandler.enable(file=fh, all_threads=True)
    KHOI_DONG.update(id=kid, dir=d, fh=fh)


if CHE_DO == "chinh" and _doi_so_som(sys.argv[1:]) is not None:
    khoi_dong_som(_doi_so_som(sys.argv[1:]))
    if KHOI_DONG["fh"] is not None:                    # tran cung (giu nguyen moc) ghi traceback vao log khoi dong
        faulthandler.dump_traceback_later(max(1.0, T_BAT_DAU + _tran_som(sys.argv[1:]) - DU_PHONG_TRAN - time.time()),
                                          exit=True, file=KHOI_DONG["fh"])
if CHE_DO == "tham_do" and sys.argv[2:3] == ["ngat_khoi_dong"]:
    # Tham do F8a: journal khoi dong cua mot phien con roi crash native NGAY -- truoc chup pyc / import ma du an
    _d = Path(os.path.abspath(sys.argv[3])) if len(sys.argv) > 3 else None
    if _d is None or _d.name != "ngat_khoi_dong" or _d.parent.name != "tham_do" or _kiem_duong_som(_d / "phien_kd"):
        print("tham do ngat_khoi_dong: doi so / duong sai", file=sys.stderr)
        sys.exit(2)
    khoi_dong_som(_d / "phien_kd")
    if KHOI_DONG["id"] is None:
        print(f"tham do ngat_khoi_dong: {KHOI_DONG['loi']}", file=sys.stderr)
        sys.exit(2)
    faulthandler._read_null()                          # access violation THAT (0xC0000005), khong qua CRT


def chup_pyc() -> dict:
    """CHI thu vien chuan (goi TRUOC import ma du an/ben thu ba). {duong: [size, mtime_ns, sha256]}."""
    t0 = time.time()
    pv = json.loads((ROOT / "configs/phase2/r61_bdd/hop_dong_cong_b.json").read_bytes())["bytecode"]["pham_vi_pyc"]
    sp, lib = Path(sysconfig.get_paths()["purelib"]), Path(sysconfig.get_paths()["stdlib"])
    goc = [ROOT / r for r in pv["repo"]] + [sp / n for n in pv["site_packages"]] + ([lib] if pv.get("stdlib") else [])
    out = {}
    for g in goc:
        for d, dirs, files in os.walk(g, followlinks=False):
            if Path(d) == lib:
                dirs[:] = [x for x in dirs if x != "site-packages"]
            for f in files:
                if f.endswith(".pyc"):
                    p = Path(d) / f
                    st = os.lstat(p)
                    out[p.as_posix()] = [st.st_size, st.st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest()]
    return {"goc": [g.as_posix() for g in goc], "so_pyc": len(out), "pyc": dict(sorted(out.items())),
            "giay": round(time.time() - t0, 3), "t0": t0}


_PYC_TRUOC = chup_pyc() if CHE_DO == "chinh" else None
_BC = {"executable": sys.executable, "orig_argv": list(getattr(sys, "orig_argv", [])),
       "flags_dont_write_bytecode": sys.flags.dont_write_bytecode, "sys_dont_write_bytecode": sys.dont_write_bytecode,
       "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE")}
sys.path.insert(0, str(ROOT / "src" / "r61_bdd"))
sys.path.insert(0, str(ROOT / "src" / "r61_bdd" / "cong_b"))
import hop_dong_bdd as H  # noqa: E402
import kiem_dau_vao as KD  # noqa: E402
import van_hanh_cong_b as VB  # noqa: E402
D, VA, CB = KD.D, KD.VA, KD.CB

RUNNER = VB.SCRIPT
CAN_TOI_THIEU = 75.0          # khong khoi chay lenh con khi con < muc nay toi han chung
MEM_TOI_THIEU_RUNNER = 30.0   # khong khoi chay lenh runner khi ngan sach mem con < muc nay (runner can >= 1 don vi)
LOAI_CRASH_CON = ("crash_native", "cha_gian_doan", "het_gio", "loi_python", "exit1_khong_traceback")
NGUONG_NGOAI_NATIVE = 4       # crash native ngoai du kien thu 4 / phien fixture -> dung hoi
THAM_DO = ("ngat_giua_popen_gan", "loi_python_giua_popen_gan", "tran_cung_trong_tien_trinh", "thieu_du_phong")
KQ: list[dict] = []
KQ_CAN: list[str] = []


class HetGio(Exception):
    """Khong du ngan sach / du phong cho viec tiep -> fixture exit 10 (tiep tuc lan goi sau)."""


class T:
    V: Path
    PH: Path
    HAN: float
    MEM: float
    TRAN: float
    LAN: int
    HD: Path
    GT: Path
    A: Path
    FX: Path
    BV: dict = {}
    SNAP: str | None = None       # sha256 snapshot cua vong (gan vao bang chung ben_thu_ba_cha)
    CP: dict | None = None        # Q9MG/QUVZ: ket qua xac minh chuyen phien cua lan goi nay


def kiem(ten: str, ok: bool, chi_tiet=None):
    KQ.append({"kiem": ten, "dat": bool(ok), "chi_tiet": chi_tiet})
    print(("[DAT] " if ok else "[KHONG DAT] ") + ten + ("" if ok else f" -- {json.dumps(chi_tiet, ensure_ascii=False, default=str)[:1200]}"))


def yeu_cau(*ten: str) -> None:
    """F7: khai bao kiem BAT BUOC cua ca TRUOC moi nhanh; kiem khong chay (dieu kien truoc hong, loi giua ca) = KHONG DAT."""
    KQ_CAN.extend(ten)


def can_tg(ten: str, giay: float) -> None:
    """F1: checkpoint truoc viec TRONG tien trinh / lenh con: qua ngan sach mem hoac khong du `giay` toi han chung -> HetGio
    TRUOC khi bat dau (lan goi sau tiep tuc tu checkpoint)."""
    now = time.time()
    if now > T.MEM or now + giay > T.HAN:
        raise HetGio(f"{ten}: can {giay:.0f} s, con {T.HAN - now:.0f} s toi han chung (ngan sach mem con {T.MEM - now:.0f} s)")


# ---------------------------------------------------------------------------
# Job Object: job goc cua fixture + job tung lenh (long trong job goc); tran cung
# ---------------------------------------------------------------------------
def _job():
    import ctypes
    from ctypes import wintypes

    class CoBan(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class MoRong(ctypes.Structure):
        _fields_ = [("Basic", CoBan), ("Io", ctypes.c_uint64 * 6)] + \
                   [(n, ctypes.c_size_t) for n in ("ProcessMemoryLimit", "JobMemoryLimit", "PeakProcess", "PeakJob")]

    class KeToan(ctypes.Structure):
        _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                    ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                    ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                    ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]
    k = VA._k32()
    k.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
    k.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    return ctypes, k, MoRong, KeToan


def tao_job():
    ctypes, k, MoRong, _ = _job()
    j = k.CreateJobObjectW(None, None)
    if not j:
        return None
    info = MoRong()
    info.Basic.LimitFlags = 0x2000                     # KILL_ON_JOB_CLOSE
    return j if k.SetInformationJobObject(j, 9, ctypes.byref(info), ctypes.sizeof(info)) else None


def dem_job(j) -> dict:
    ctypes, k, _, KeToan = _job()
    a = KeToan()
    ok = k.QueryInformationJobObject(j, 1, ctypes.byref(a), ctypes.sizeof(a), None)
    return {"ok": bool(ok), "active": int(a.ActiveProcesses), "tong": int(a.TotalProcesses)}


def bao_ve_lan_goi(tran: float, fh) -> dict:
    """F1/F2: (1) gan CHINH tien trinh nay vao job goc KILL_ON_JOB_CLOSE (van_hanh_cong_a.gan_job_cha) TRUOC khi tao con;
    (2) dat lai tran cung (dump_traceback_later exit=True) ghi traceback vao log rieng cua lan goi."""
    job = VA.gan_job_cha() if VA._JOB is None else {"dat": True, "gan": "da gan"}
    con = tran - time.time()
    faulthandler.dump_traceback_later(max(1.0, con), exit=True, file=fh)
    return {**job, "tran_cung_epoch": round(tran, 3), "tran_cung_con_giay": round(con, 3)}


def env_con() -> dict:
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    return env


def khoi_chay_trong_job(cmd: list[str], env: dict, fo, fe, han: float, tiem: str | None = None, sau_tao=None) -> dict:
    """Con tao TAM DUNG -- da thuoc job goc ngay tu CreateProcess (F2) -- xac nhan IsProcessInJob(job goc), gan job lenh,
    tiep tuc, cho toi `han` (het -> TerminateJobObject), cho ActiveProcesses = 0. Loi Python o BAT KY dau -> giet con +
    TerminateJobObject truoc khi nem lai; crash native cua tien trinh nay -> job goc dong -> kernel giet ca cay.
    tiem (CHI tham do): 'giua_popen_va_gan' (crash native) / 'giua_popen_va_gan_python' (RuntimeError)."""
    import subprocess
    ctypes, k, _, _ = _job()
    j, p = tao_job(), None
    kq = {"ma": None, "het_gio": False, "gan_job_fixture": False, "trong_job_goc": None, "cay": {}, "pid": None,
          "thoi_diem_tao": None}
    try:
        p = subprocess.Popen(cmd, cwd=str(H.ROOT), env=env, stdout=fo, stderr=fe, creationflags=0x4)   # CREATE_SUSPENDED
        kq.update(pid=p.pid, thoi_diem_tao=VA.thoi_diem_tao(int(p._handle)), trong_job_goc=VA.trong_job(p))
        if sau_tao is not None:
            sau_tao(p)
        if tiem == "giua_popen_va_gan":
            VA.crash_native()
        if tiem == "giua_popen_va_gan_python":
            raise RuntimeError("[TIEM LOI PYTHON] giua Popen va AssignProcessToJobObject")
        kq["gan_job_fixture"] = bool(j) and bool(k.AssignProcessToJobObject(j, int(p._handle)))
        if not kq["gan_job_fixture"] or kq["trong_job_goc"] is not True:
            p.kill()                                   # khong bao dam thu hoi -> khong cho con chay lenh nao
            kq["ma"] = p.wait()
            kq["cay"] = dem_job(j) if j else {}
            return kq
        tt = VA.tiep_tuc_con(p)
        try:
            if not tt["dat"]:
                raise subprocess.TimeoutExpired(cmd, 0)
            kq["ma"] = p.wait(timeout=max(1.0, han - time.time()))
        except subprocess.TimeoutExpired:
            k.TerminateJobObject(j, 0xDEAD)
            kq["ma"], kq["het_gio"] = p.wait(), True
        cay = {}
        for _ in range(150):                           # cho ca cay (cha B, con, git ...) ket thuc
            cay = dem_job(j)
            if cay["active"] == 0:
                break
            time.sleep(0.1)
        if cay.get("active"):
            k.TerminateJobObject(j, 0xDEAD)
            time.sleep(0.5)
            cay = {**dem_job(j), "phai_terminate": True}
        kq["cay"] = cay
        return kq
    except BaseException:
        if p is not None and p.poll() is None:
            p.kill()
            p.wait()
        if j:
            k.TerminateJobObject(j, 0xDEAD)
        raise
    finally:
        if j:
            k.CloseHandle(j)


# ---------------------------------------------------------------------------
# Lenh con (checkpoint theo ten)
# ---------------------------------------------------------------------------
def lan_chay(S: Path | None) -> list[dict]:
    return VA.doc_lan_chay(S) if S is not None and (S / "van_hanh" / "lan_chay").is_dir() else []


def lenh(ten: str, argv: list[str], S: Path | None = None, can: float = CAN_TOI_THIEU, script: Path | None = None) -> dict:
    d = T.V / "lenh"
    p_kq, p_bd = d / f"{ten}.json", d / f"{ten}.bat_dau.json"
    if p_kq.is_file():
        return H.doc_json(p_kq)
    if p_bd.is_file():                                 # lan goi truoc bi ngat giua lenh: khong chay lai
        m = {"ten": ten, "exit": None, "bi_gian_doan": True, "stdout": "", "stderr": "", "lan_chay_moi": None,
             "ben_thu_ba_cha": ben_thu_ba(f"sau_lenh:{ten}")}
        H.ghi_json(p_kq, m, [T.V])
        ap_so_sau_lenh(T.PH, (T.V.name, T.LAN), p_kq, S)
        return m
    can_tg(f"lenh {ten}", can)
    if not T.BV.get("dat"):
        raise H.LoiHopDong("khong co job goc cua fixture -- khong khoi chay lenh con")
    now = time.time()
    if script is None and "--ngan-sach-giay" not in argv and argv[:1] != ["--in-ma-b"] and T.MEM - now < MEM_TOI_THIEU_RUNNER:
        # lan_02/vong_01: cr2_tiep_1 khoi chay khi ngan sach mem con 2 s -> runner het han_mem truoc don vi dau (exit 10 rong)
        raise HetGio(f"lenh {ten}: ngan sach mem con {T.MEM - now:.0f} s < {MEM_TOI_THIEU_RUNNER:.0f} s cho runner")
    if script is None:
        cmd = [H.PYTHON, "-B", str(RUNNER), *argv]
        if "--han-chot" not in argv:
            cmd += ["--han-chot", f"{T.HAN:.3f}"]
        if "--ngan-sach-giay" not in argv and argv[:1] != ["--in-ma-b"]:
            cmd += ["--ngan-sach-giay", f"{max(1.0, min(420.0, T.MEM - now)):.1f}"]
    else:
        cmd = [H.PYTHON, "-B", str(script), *argv]
    n0 = len(lan_chay(S))
    H.ghi_json(p_bd, {"ten": ten, "argv": cmd, "luc": time.strftime("%Y-%m-%dT%H:%M:%S")}, [T.V])
    fo_p, fe_p = H.kiem_duong_ghi(d / f"{ten}.stdout.txt", [T.V]), H.kiem_duong_ghi(d / f"{ten}.stderr.txt", [T.V])
    t0 = time.time()
    with open(fo_p, "wb") as fo, open(fe_p, "wb") as fe:
        r = khoi_chay_trong_job(cmd, env_con(), fo, fe, T.HAN)
    ma = r["ma"]
    out, err = fo_p.read_bytes().decode("utf-8", "replace"), fe_p.read_bytes().decode("utf-8", "replace")
    ds = lan_chay(S)
    m = {"ten": ten, "argv": cmd, "exit": ma, "exit_hex": None if ma is None else hex(ma & 0xFFFFFFFF),
         "giay": round(time.time() - t0, 3), "het_gio_fixture": r["het_gio"], "gan_job_fixture": r["gan_job_fixture"],
         "trong_job_goc": r["trong_job_goc"], "pid": r["pid"], "thoi_diem_tao": r["thoi_diem_tao"], "cay": r["cay"],
         "stdout": out[-20000:], "stderr": err[-20000:], "lan_chay_moi": len(ds) - n0 if S is not None else None,
         "phien": str(S) if S is not None else None, "lan_chay_stt": [x["stt"] for x in ds[n0:]] if S is not None else [],
         "ket_thuc": (ds[-1]["ket_thuc"] if ds and len(ds) > n0 else None),
         "ket_qua_con": (H.doc_json(ds[-1]["dir"] / "ket_qua_con.json")
                         if ds and len(ds) > n0 and (ds[-1]["dir"] / "ket_qua_con.json").is_file() else None),
         "ben_thu_ba_cha": ben_thu_ba(f"sau_lenh:{ten}")}   # Q73M: sys.modules cua cha NGAY sau moi lenh, truoc checkpoint
    H.ghi_json(p_kq, m, [T.V])
    print(f"  [lenh] {ten}: exit {ma} ({m['giay']} s, cay active {r['cay'].get('active')})")
    ap_so_sau_lenh(T.PH, (T.V.name, T.LAN), p_kq, S)   # F8b: ghi so + ap quyet dinh dung NGAY, truoc lenh ke tiep
    return m


def chu(m: dict) -> str:
    kt = m.get("ket_thuc") or {}
    kc = m.get("ket_qua_con") or {}
    return "\n".join([m.get("stdout", ""), m.get("stderr", ""), *kt.get("ly_do_hop_dong", []), str(kc.get("loi", "")),
                      json.dumps(kt.get("quyet_dinh"), ensure_ascii=False, default=str)])


def kiem_lenh(ten: str, m: dict, exit_ky: int, nhom: str | None = None, *can: str):
    s = chu(m)
    ok = m.get("exit") == exit_ky and (nhom is None or f"[{nhom}]" in s) and all(c in s for c in can)
    kiem(ten, ok, {"exit": m.get("exit"), "ky_vong": exit_ky, "nhom": nhom, "can": can, "trich": s[-1500:]})


def phe_duyet(S: Path, ten: str | None = None, **doi) -> Path:
    hd = KD.nap_hop_dong(T.HD)
    d = {"loai": "phe_duyet_cong_b", "duyet": True, "goi_a_sha256": hd["a"]["goi_a_sha256"],
         "khoa_sha256": hd["a"]["khoa_sha256"], "review_a_sha256": hd["review_a"]["sha256"],
         "hop_dong_cong_b_sha256": hd["_sha256"], "ma_b_sha256": KD.ma_b(hd)["sha256"], "phien_b": H.rel(S),
         "plan_thuc_thi_sha256": "0" * 64, "nguoi_duyet": "fixture (tong hop, KHONG phai phe duyet that)", "luc": "fixture"}
    d.update(doi)
    p = T.FX / "phe_duyet" / f"{ten or S.name}.json"
    if not p.is_file():
        H.ghi_json(p, d, [T.V])
    return p


def b(S: Path, buoc: str, *them, pd: Path | None = None, hd: Path | None = None) -> list[str]:
    return ["--cong", "B", "--phien", str(S), "--buoc", buoc, "--so-doan", "1", "--hop-dong", str(hd or T.HD),
            "--giao-thuc", str(T.GT), "--phe-duyet", str(pd or phe_duyet(S)), *them]


def tl(*ds) -> list[str]:
    return ["--tiem-loi", json.dumps([dict(x) for x in ds])]


def S_(ten: str) -> Path:
    return T.V / "phien" / ten


def chay_toi_xong(S: Path, ten: str, buoc: str, *them, toi_da: int = 6) -> dict:
    m = {}
    for i in range(toi_da):
        m = lenh(f"{ten}_{i}", b(S, buoc, *them), S)
        if m.get("exit") != 10:
            return m
    return m


def mo(S: Path, ten: str) -> bool:
    a = lenh(f"{ten}__tien_kiem", b(S, "tien-kiem"), S)
    c = lenh(f"{ten}__mo_phien", b(S, "mo-phien"), S)
    return a.get("exit") == 0 and c.get("exit") == 0


# ---------------------------------------------------------------------------
# Sua tam the gioi fixture + phuc hoi (F3: journal la dau vao KHONG tin)
# ---------------------------------------------------------------------------
def vung_sua() -> list[Path]:
    """Chi the gioi fixture (fx/) va cac phien B fixture duoc sua tam; khong bao gio so sach cua chinh fixture."""
    return [T.FX, T.V / "phien"]


class sua_tam:
    """Sua TAM mot file cua the gioi fixture; ban goc luu ben vung (khoi_phuc/) TRUOC khi sua, phuc hoi khi ra khoi khoi
    (va o dau moi lan goi neu lan truoc bi ngat). Duong kiem lexical/realpath/reparse TRUOC moi ghi."""

    def __init__(self, p: Path, moi: bytes | None):
        self.p, self.moi = Path(p), moi
        self.k = T.V / "khoi_phuc" / (hashlib.sha256(str(self.p).encode()).hexdigest()[:16] + ".json")

    def __enter__(self):
        H.kiem_duong_ghi(self.p, vung_sua())
        H.kiem_duong_ghi(self.k, [T.V / "khoi_phuc"])
        if not self.k.is_file():
            co = self.p.is_file()
            if co:
                H.ghi_bytes(self.k.with_suffix(".bytes"), self.p.read_bytes(), [T.V / "khoi_phuc"], ghi_de=True)
            H.ghi_json(self.k, {"duong": str(self.p), "co": co}, [T.V / "khoi_phuc"], ghi_de=True)
        if self.moi is None:
            if self.p.is_file():                       # "xoa" tam = chuyen RA NGOAI the gioi (khoi_phuc/), khong de lai file la
                nguon = H.kiem_duong_ghi(self.p, vung_sua())
                os.replace(nguon, H.kiem_duong_ghi(self.k.with_suffix(".goc"), [T.V / "khoi_phuc"]))
        else:
            H.ghi_bytes(self.p, self.moi, vung_sua(), ghi_de=True)
        return self

    def __exit__(self, *a):
        phuc_hoi(self.k)
        return False


def phuc_hoi(k: Path, goc: Path | None = None) -> None:
    """Phuc hoi tu journal `k` (duoi `goc`, mac dinh <vong>/khoi_phuc). Journal = dau vao khong tin (F3): JSON + schema, ten
    journal phai rang buoc duong (sha256(duong)[:16]), duong tuyet doi; NGAY TRUOC moi move/restore kiem lexical + realpath +
    reparse ca NGUON lan DICH (dich chi trong vung_sua). Sai -> LoiHopDong, khong dong vao file nao."""
    goc = goc or T.V / "khoi_phuc"
    k = H.kiem_duong_ghi(k, [goc])
    m, loi = KD.doc_json_dv(k, f"journal {k.name}")
    if loi or not isinstance(m.get("duong"), str) or not isinstance(m.get("co"), bool):
        raise H.LoiHopDong(f"journal phuc hoi hong: {loi or m}")
    if hashlib.sha256(m["duong"].encode()).hexdigest()[:16] != k.stem or not Path(m["duong"]).is_absolute():
        raise H.LoiHopDong(f"journal {k.name} khong rang buoc duong {m['duong']!r} (ten / duong tuyet doi)")
    p = H.kiem_duong_ghi(Path(m["duong"]), vung_sua())
    if m["co"]:
        g = k.with_suffix(".goc")
        if g.is_file():
            nguon, dich = H.kiem_duong_ghi(g, [goc]), H.kiem_duong_ghi(p, vung_sua())
            os.replace(nguon, dich)
        else:
            nguon = H.kiem_duong_ghi(k.with_suffix(".bytes"), [goc])
            H.ghi_bytes(p, nguon.read_bytes(), vung_sua(), ghi_de=True)
    elif p.is_file():                                  # file them tam: chuyen ra vung bo/ cua vong (khong xoa)
        bo = H.tao_thu_muc(T.V / "bo", [T.V]) / f"{k.stem}_{int(time.time() * 1000)}_{p.name}"
        nguon, dich = H.kiem_duong_ghi(p, vung_sua()), H.kiem_duong_ghi(bo, [T.V / "bo"])
        os.replace(nguon, dich)
    os.replace(H.kiem_duong_ghi(k, [goc]), H.kiem_duong_ghi(k.with_name(k.name + f".da_phuc_hoi_{int(time.time() * 1000)}"), [goc]))


def nho(ten: str, fn, can: float = 0.0):
    """Quan sat / tai tinh ghi BEN VUNG ngay lan dau (lan goi sau phat lai, khong quan sat lai trang thai da doi).
    can > 0: viec nang trong tien trinh -> checkpoint thoi gian TRUOC khi tinh (F1)."""
    p = T.V / "nho" / f"{ten}.json"
    if p.is_file():
        return H.doc_json(p)["v"]
    if can:
        can_tg(f"nho {ten}", can)
    v = fn()
    H.ghi_json(p, {"v": v}, [T.V])
    return json.loads(json.dumps(v, default=str))


def stt_cua(m: dict) -> int:
    return int((m.get("ket_thuc") or {}).get("stt") or 10 ** 6)


def lat_byte(p: Path) -> bytes:
    b0 = p.read_bytes()
    return b0[:-1] + bytes([b0[-1] ^ 1]) if b0 else b"x"


def nguyen_A() -> bool:
    return H.bang_hash_thu_muc(T.A, ()) == H.doc_json(T.V / "thiet_lap" / "anh_A.json")


# ---------------------------------------------------------------------------
# Thiet lap: the gioi + A tong hop + hop dong / giao thuc / review fixture
# ---------------------------------------------------------------------------
def buoc_tl(ten: str, fn, can: float, tiep_tuc: bool = False) -> None:
    """Buoc thiet lap TRONG tien trinh. <ten>.xong.json = checkpoint; <ten>.trang_thai.json: 'dang_chay' con lai tu lan goi
    truoc = bi ngat giua buoc -> KHONG lam lai (LoiHopDong); 'tam_dung' (buoc tiep_tuc dung sach o ranh gioi don vi khi het
    ngan sach) -> lam tiep. Khong du du phong -> HetGio TRUOC khi bat dau (F1)."""
    d = T.V / "thiet_lap"
    if (d / f"{ten}.xong.json").is_file():
        return
    tt_p = d / f"{ten}.trang_thai.json"
    if tt_p.is_file():
        tt = H.doc_json(tt_p)
        if tt.get("trang_thai") != "tam_dung" or not tiep_tuc:
            raise H.LoiHopDong(f"thiet lap {ten} bi ngat giua buoc o lan goi truoc ({tt}) -- khong lam lai (vong can ma moi)")
    can_tg(f"thiet lap {ten}", can)
    H.ghi_json(tt_p, {"trang_thai": "dang_chay", "luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid()}, [T.V], ghi_de=True)
    t0 = time.time()
    try:
        them = fn() or {}
    except HetGio as e:
        H.ghi_json(tt_p, {"trang_thai": "tam_dung", "ly_do": str(e), "luc": time.strftime("%Y-%m-%dT%H:%M:%S")}, [T.V], ghi_de=True)
        raise
    H.ghi_json(d / f"{ten}.xong.json", {"giay": round(time.time() - t0, 3), **them,
                                        "ben_thu_ba_cha": ben_thu_ba(f"sau_thiet_lap:{ten}")}, [T.V])
    H.ghi_json(tt_p, {"trang_thai": "xong", "luc": time.strftime("%Y-%m-%dT%H:%M:%S")}, [T.V], ghi_de=True)


def viec_con(ten: str, can: float):
    """P43284b78: buoc can thu vien ben thu ba (ultralytics/torch/cv2/pandas/trackeval) chay trong LENH CON
    [python, -B, THIS, --viec ...] qua lenh() (job goc + job lenh, han chung T.HAN, so crash xet ngay sau lenh); cha fixture chi
    thu vien chuan + numpy. Moi lan thu = mot lenh moi viec_<ten>_NN (lenh da co ket qua khong chay lai). exit 0 -> ket qua
    <vong>/viec/<lenh>.json; 10 -> HetGio (lan goi sau tiep tuc); con lai -> LoiHopDong."""
    tl = "viec_" + ten.replace(":", "__")
    k = len(list((T.V / "lenh").glob(f"{tl}_[0-9][0-9].json"))) if (T.V / "lenh").is_dir() else 0
    ma = f"{tl}_{k:02d}"
    m = lenh(ma, ["--viec", ten, str(T.V), f"{T.HAN:.3f}", f"{T.MEM:.3f}", ma], can=can, script=THIS)
    p = T.V / "viec" / f"{ma}.json"
    if m.get("exit") == 0 and p.is_file():
        return H.doc_json(p)["v"]
    if m.get("exit") == H.EXIT_CHUA_XONG:
        raise HetGio(f"viec con {ten} ({ma}) chua xong: {m.get('stdout', '')[-300:]}")
    raise H.LoiHopDong(f"viec con {ten} ({ma}): exit {m.get('exit')} -- {chu(m)[-800:]}")


def tl_the_gioi():
    import test_chuan_bi_bdd as F
    vung = [T.V]
    sha = F.tao_the_gioi(T.FX, vung)
    g = H.doc_json(F.giao_thuc_fixture(H.doc_json(D.GIAO_THUC), T.FX, sha, vung))
    H.ghi_bytes(T.FX / "PLAN_DA_CHOT.md", CB.PLAN.read_bytes(), vung)      # BAN SAO PLAN (chi doc PLAN that)
    g["van_hanh_a"] = {"plan": "PLAN_DA_CHOT.md", "chup_dau": "chup_dau", "kich_thuoc_doan_bam": 8}
    H.ghi_json(T.GT, g, vung)
    return {"so_run": len(D.Ctx(T.GT).runs())}


def tl_A_tien_kiem():
    ctx, A, vung = D.Ctx(T.GT), T.A, [T.V]
    D.buoc_tien_kiem(ctx, A, vung)
    VA.chup_a(T.FX / "PLAN_DA_CHOT.md", A / "tien_kiem_a" / "chup", vung)
    H.ghi_json(A / "tien_kiem_a" / "so_sanh.json", {"dat": True, "ghi_chu": "A tong hop fixture B"}, vung)
    H.ghi_json(A / "tien_kiem_a" / "moi_truong.json", {"fixture": True}, vung)


def tl_A_cache(det: str):
    """Cache A tong hop theo detector: D.buoc_cache tiep tuc theo video (marker); ngan sach = thoi gian con lai toi han
    chung tru du phong 30 s (F1: khong cap ngan sach vo han)."""
    import test_chuan_bi_bdd as F
    ctx = D.Ctx(T.GT)
    D.NAP_MODEL = lambda d, cfg: F.ModelGia(cfg["loai_checkpoint"])
    con = min(T.MEM, T.HAN - 30.0) - time.time()
    if con < 5:
        raise HetGio(f"cache A tong hop {det}: con {con:.0f} s")
    try:
        D.buoc_cache(ctx, T.A, det, [T.V], "cpu", None, con)
    except H.ChuaXong as e:
        raise HetGio(f"cache A tong hop {det}: {e}") from None


def tl_A_goi():
    ctx, A, vung = D.Ctx(T.GT), T.A, [T.V]
    loi = [e for d in ctx.detectors() for v in ctx.videos for e in D.kiem_cache_video(ctx, d, A / "cache" / d, v)]
    if loi:
        raise H.LoiHopDong(f"kiem cache A tong hop: {loi[:3]}")
    H.ghi_json(A / "kiem_cache" / "ket_qua_fixture.json", {"dat": True, "loi": []}, vung)
    H.ghi_json(A / "smoke" / "ket_qua_smoke.json", {"fixture": "smoke toi gian (cau truc); B khong doc smoke"}, vung)
    kp = A / "khoa" / "khoa.json"
    k = D.anh_khoa(ctx, A)
    k["phu_luc_van_hanh"] = VA.phu_luc(ctx, A)
    H.ghi_json(kp, k, vung)
    H.ghi_marker(A / "khoa" / "khoa.xong.json", [kp], vung, {"buoc": "khoa", "luot": VA.LUOT})
    R = A / "kiem_khoa" / "vong_01"
    H.ghi_json(R / "ket_qua.json", {"dat": True, "loi": [], "khoa_sha256": H.sha256_file(kp)}, vung)
    H.ghi_marker(R / "xong.json", [R / "ket_qua.json"], vung, {"buoc": "kiem-khoa"})
    vh = A / "van_hanh"
    H.ghi_json(vh / "phien.json", {"luot": VA.LUOT, "phien": H.rel(A), "fixture": True}, vung)
    for f, b0 in (("goi.json", b'{"stt": 1}\n'), ("ket_thuc.json", b'{"stt": 1, "exit": 0}\n'), ("stdout.txt", b"ok\n"),
                  ("stderr.txt", b""), ("faulthandler.log", b"")):
        H.ghi_bytes(vh / "lan_chay" / "0001_tien-kiem" / f, b0, vung)
    for i in (1, 2):
        H.ghi_bytes(vh / "cha" / f"20261024_00000{i}_pid{i}.faulthandler.log", b"", vung)
    H.ghi_bytes(vh / "goi_cha.jsonl", b'{"exit": 0}\n', vung)
    H.ghi_json(vh / "so_crash.json", {"dem": {"tong": 0}}, vung)
    H.ghi_json(vh / "dung_da_go" / "dung_lan_0010.json", {"stt": 10, "fixture": "dung da go tong hop"}, vung)
    H.ghi_json(vh / "dung_da_go" / "quyet_dinh_go_dung.json", {"quyet_dinh": "go_dung_tiep_tuc_cung_phien_A", "fixture": True}, vung)
    nkc = A / "goi" / "nhat_ky_crash_chot.json"
    H.ghi_json(nkc, {"dem": {"tong": 0}, "chot_tai_stt": 2}, vung)
    tep = {r: H.sha256_file(H.ROOT / r) for r in VA.payload(A, 2)}
    tep[H.rel(nkc)] = H.sha256_file(nkc)
    goi = {"luot": VA.LUOT, "phien_A": H.rel(A), "khoa": H.rel(kp), "khoa_sha256": H.sha256_file(kp),
           "giao_thuc_sha256": H.sha256_file(T.GT), "hop_dong_lenh_sha256": H.sha256_file(VA.HOP_DONG),
           "kiem_khoa_vong": H.rel(R), "kiem_khoa_ket_qua_sha256": H.sha256_file(R / "ket_qua.json"),
           "stt_dong_goi_dau": 2, "loai_tru": VA.LOAI_TRU_GOI, "phuc_hoi_dong_goi": [], "so_file": len(tep),
           "file": dict(sorted(tep.items()))}
    H.ghi_json(A / "goi_a.json", goi, vung)
    H.ghi_bytes(A / "SHA256SUMS", "".join(f"{h}  {r}\n" for r, h in sorted(tep.items())).encode("utf-8"), vung)
    H.ghi_marker(A / "goi_a.xong.json", [A / "goi_a.json", A / "SHA256SUMS"], vung,
                 {"buoc": "dong-goi-a", "luot": VA.LUOT, "khoa_sha256": goi["khoa_sha256"]})
    H.ghi_json(A / "dong_goi" / "danh_muc.json", {"fixture": True}, vung)
    for n in ("0002_dong-goi-a", "0003_kiem-goi-a"):
        for f in ("goi.json", "ket_thuc.json", "stdout.txt", "stderr.txt"):
            H.ghi_bytes(vh / "lan_chay" / n / f, b"{}\n", vung)
    K = vh / "kiem_goi_a" / "vong_01"
    H.ghi_json(K / "ket_qua.json", {"dat": True, "loi": [], "goi_a_sha256": H.sha256_file(A / "goi_a.json"),
                                    "khoa_sha256": goi["khoa_sha256"]}, vung)
    H.ghi_marker(K / "xong.json", [K / "ket_qua.json"], vung, {"buoc": "kiem-goi-a"})
    H.ghi_json(K / "bam" / "danh_muc.json", {"fixture": True}, vung)
    H.ghi_json(K / "bam" / "doan" / "0000.json", {}, vung)
    return {"so_file_goi": len(tep)}


def tl_hop_dong():
    import copy
    vung, A = [T.V], T.A
    hd = copy.deepcopy(H.doc_json(KD.HOP_DONG_B))
    review = T.FX / "review_a.json"
    H.ghi_json(review, {"verdict": "approve", "issues": [], "fixture": "review tong hop, khong co goi_a_sha256 cap dau"}, vung)
    H.ghi_json(T.FX / "review_a_revise.json", {"verdict": "revise", "fixture": True}, vung)
    H.ghi_bytes(T.FX / "review_a_cat_cut.json", b'{"verdict": "appr', vung)                 # F9: JSON cat cut, digest ghim dung
    t = CB.tach_plan((T.FX / "PLAN_DA_CHOT.md").read_bytes())
    vh = A / "van_hanh"
    hd.update({"che_do": "fixture", "nguon": "FIXTURE tong hop (tests/cong_b_20261026); KHONG phai hop dong that"})
    hd["a"].update({"phien": H.rel(A), "goi_a_sha256": H.sha256_file(A / "goi_a.json"),
                    "khoa_sha256": H.sha256_file(A / "khoa" / "khoa.json"), "sha256sums_sha256": H.sha256_file(A / "SHA256SUMS"),
                    "kiem_khoa_sha256": H.sha256_file(A / "kiem_khoa" / "vong_01" / "ket_qua.json"), "stt_dong_goi_dau": 2,
                    "so_file_goi": H.doc_json(A / "goi_a.json")["so_file"]})
    hd["plan"].update({"duong": H.rel(T.FX / "PLAN_DA_CHOT.md"), **{k: t[k] for k in KD.PLAN_KHOA}})
    hd["review_a"].update({"duong": H.rel(review), "sha256": H.sha256_file(review)})
    hd["phu_luc_dung_da_go"] = [{"duong": f"van_hanh/dung_da_go/{n}", "so_byte": (vh / "dung_da_go" / n).stat().st_size,
                                 "sha256": H.sha256_file(vh / "dung_da_go" / n)}
                                for n in ("dung_lan_0010.json", "quyet_dinh_go_dung.json")]
    so = {"so_crash": 1, "goi_cha": 1, "log_cha": 2, "lan_chay_sau_goi": 8, "kiem_goi_a": 4}
    for x in hd["van_hanh_a"]["loai_tru"]:
        x["so_file"] = so[x["ten"]]
        if x["ten"] == "lan_chay_sau_goi":
            x["mau"] = "^lan_chay/(0002_dong-goi-a|0003_kiem-goi-a)/(goi\\.json|ket_thuc\\.json|stdout\\.txt|stderr\\.txt)$"
    hd["deadline"]["du_phong_don_vi"] = 20
    H.ghi_json(T.HD, hd, vung)
    for ten, rv in (("hop_dong_review_revise.json", "review_a_revise.json"), ("hop_dong_review_cat_cut.json", "review_a_cat_cut.json")):
        hv = copy.deepcopy(hd)
        hv["review_a"].update({"duong": H.rel(T.FX / rv), "sha256": H.sha256_file(T.FX / rv)})
        H.ghi_json(T.FX / ten, hv, vung)
    hb = T.HD.read_bytes()
    H.ghi_bytes(T.FX / "hop_dong_cat_cut.json", hb[:len(hb) // 2], vung)                   # F9: hop dong JSON cat cut
    hs = copy.deepcopy(hd)
    hs["a"]["so_file_goi"] = str(hs["a"]["so_file_goi"])                                     # F9: sai kieu truong
    H.ghi_json(T.FX / "hop_dong_sai_kieu.json", hs, vung)
    gb = T.GT.read_bytes()                                                                   # B0-F9: giao thuc hong
    H.ghi_bytes(T.FX / "giao_thuc_cat_cut.json", gb[:len(gb) // 2], vung)
    g0 = H.doc_json(T.GT)
    H.ghi_json(T.FX / "manifest_sai_kieu.json", {"videos": "fxv-a", "fixture": "videos sai kieu (chuoi)"}, vung)
    gm = copy.deepcopy(g0)
    gm["tap_du_lieu"]["manifest"]["rel"] = os.path.relpath(T.FX / "manifest_sai_kieu.json", g0["goc"]).replace("\\", "/")
    H.ghi_json(T.FX / "giao_thuc_manifest_sai_kieu.json", gm, vung)
    gs = copy.deepcopy(g0)
    gs["tracker"]["fps"] = str(gs["tracker"]["fps"])
    H.ghi_json(T.FX / "giao_thuc_sai_kieu.json", gs, vung)
    gf = copy.deepcopy(g0)                             # B0-F9: fps=0 -> D.Ctx 1.0/fps; lop B phai chan truoc (exit 2)
    gf["tracker"]["fps"] = 0
    H.ghi_json(T.FX / "giao_thuc_fps_0.json", gf, vung)
    H.ghi_json(T.V / "thiet_lap" / "anh_A.json", H.bang_hash_thu_muc(A, ()), vung)
    return {"hop_dong_sha256": H.sha256_file(T.HD)}


THAT = {"plan": "handoff/PLAN_DA_CHOT.md", "hop_dong_lenh": "configs/phase2/r61_bdd/hop_dong_lenh.json",
        "giao_thuc": "configs/phase2/r61_bdd/giao_thuc.json"}
A_THAT_TEP = ("goi_a.json", "goi_a.xong.json", "SHA256SUMS", "khoa/khoa.json", "khoa/khoa.xong.json",
              "kiem_khoa/vong_01/ket_qua.json", "van_hanh/dung_da_go/dung_lan_0010.json",
              "van_hanh/dung_da_go/quyet_dinh_go_dung.json", "van_hanh/so_crash.json", "van_hanh/phien.json")


def trang_thai_that() -> dict:
    """CHI DOC: PLAN, ma r61 cu, hop dong/giao thuc, file dinh danh phien A that, thu muc thuc_thi (khong bam du lieu lon)."""
    return {"tep": {r: H.sha256_file(H.ROOT / r) for r in THAT.values()},
            "A_that": {r: H.sha256_file(VA.PHIEN_A_THAT / r) if (VA.PHIEN_A_THAT / r).is_file() else None for r in A_THAT_TEP},
            "ma_r61_cu": {n: H.sha256_file(H.SRC / "r61_bdd" / n) for n in sorted(os.listdir(H.SRC / "r61_bdd")) if n.endswith(".py")},
            "thuc_thi_ton_tai": H.VUNG_THUC_THI.exists()}


def ca_thiet_lap():
    yeu_cau("hop_dong_fixture_hop_le", "in_ma_b_exit_0", "tuong_thich_khoa_A_ma_nguon_khong_doi_khi_them_cong_b",
            "tuong_thich_khoa_A_phu_luc_ma_r61_khong_doi", "fixture_chay_bang_interpreter_va_co_B")
    # P43284b78: the gioi / tien kiem A / cache / goi + anh khoa nap ultralytics, torch, cv2, pandas -> lenh con --viec
    buoc_tl("the_gioi", lambda: viec_con("the_gioi", 60), 60)
    buoc_tl("A_tien_kiem", lambda: viec_con("A_tien_kiem", 60), 60)
    for det in D.Ctx(T.GT).detectors():
        buoc_tl(f"A_cache_{det}", lambda det=det: viec_con(f"A_cache:{det}", 45), 45, tiep_tuc=True)
    buoc_tl("A_goi", lambda: viec_con("A_goi", 120), 120)
    buoc_tl("hop_dong", tl_hop_dong, 30)                # chi thu vien chuan: giu trong tien trinh
    hd = KD.nap_hop_dong(T.HD)
    kiem("hop_dong_fixture_hop_le", hd["_fixture"] and H._trong(T.HD, KD.VUNG_FIXTURE_B))
    m = lenh("thiet_lap__in_ma_b", ["--in-ma-b", "--hop-dong", str(T.HD)])
    kiem_lenh("in_ma_b_exit_0", m, 0, None, KD.ma_b(hd)["sha256"])
    # Tuong thich khoa A khi them src/r61_bdd/cong_b/: khoa A THAT (chi doc) khong chua file cong_b
    k = H.doc_json(VA.PHIEN_A_THAT / "khoa" / "khoa.json")
    ma_nay = {p.relative_to(H.SRC).as_posix(): H.sha256_file(p) for p in D.dong_import_tinh(H.SRC)}
    r61 = H.SRC / "r61_bdd"
    ma_r61 = {f"r61_bdd/{n}": H.sha256_file(r61 / n) for n in sorted(os.listdir(r61)) if n.endswith(".py")}
    kiem("tuong_thich_khoa_A_ma_nguon_khong_doi_khi_them_cong_b",
         k["ma_nguon"] == ma_nay and not any("cong_b" in r for r in ma_nay), {"lech": sorted(set(k["ma_nguon"]) ^ set(ma_nay))})
    kiem("tuong_thich_khoa_A_phu_luc_ma_r61_khong_doi", k["phu_luc_van_hanh"]["ma_r61"] == ma_r61,
         {"lech": sorted(f for f in set(ma_r61) | set(k["phu_luc_van_hanh"]["ma_r61"])
                         if ma_r61.get(f) != k["phu_luc_van_hanh"]["ma_r61"].get(f))})
    bc, loi = VB.kiem_bytecode()
    kiem("fixture_chay_bang_interpreter_va_co_B", not loi, bc)


# ---------------------------------------------------------------------------
# Ca: bao ve toan lan goi (F1/F2) -- tham do bang CHINH ham cua fixture trong tien trinh con rieng
# ---------------------------------------------------------------------------
def chay_tham_do(loai: str, tran: float) -> tuple[dict, dict]:
    d = T.V / "tham_do" / loai
    m = lenh(f"bv_{loai}", ["--tham-do", loai, str(d), f"{tran:g}"], can=40, script=THIS)

    def qs():
        out = {f: (H.doc_json(d / f) if (d / f).is_file() else None)
               for f in ("tham_do.json", "chau.json", "don_dep.json", "het_gio.json", "da_lam.json")}
        c = out["chau.json"]
        out["chau_trang_thai"] = VA.trang_thai_con(c["pid"], c.get("thoi_diem_tao")) if c else None
        fh = d / "faulthandler.log"
        out["faulthandler"] = fh.read_bytes().decode("utf-8", "replace")[-2000:] if fh.is_file() else ""
        return out
    return m, nho(f"bv_{loai}", qs)


def cay_tu_thu_hoi(m: dict) -> bool:
    """Cay cua lenh ve 0 MA fixture KHONG phai TerminateJobObject (chau chet theo job goc cua tham do)."""
    c = m.get("cay") or {}
    return c.get("active") == 0 and not c.get("phai_terminate") and not m.get("het_gio_fixture")


def ca_bao_ve():
    yeu_cau("bv_job_goc_fixture_dat", "bv_tran_cung_da_dat", "bv_ngat_giua_popen_va_gan_chau_bi_thu_hoi",
            "bv_loi_python_giua_popen_va_gan_don_dep", "bv_tran_cung_trong_tien_trinh_thu_hoi_cay",
            "bv_thieu_du_phong_khong_bat_dau_viec")
    kiem("bv_job_goc_fixture_dat", T.BV.get("dat") is True, T.BV)
    kiem("bv_tran_cung_da_dat", T.BV.get("tran_cung_epoch") is not None
         and T.BV["tran_cung_epoch"] <= T_BAT_DAU + TRAN_TOI_DA - DU_PHONG_TRAN + 1e-3, T.BV)
    m, q = chay_tham_do("ngat_giua_popen_gan", 30)
    c, tt = q["chau.json"] or {}, q["chau_trang_thai"] or {}
    kiem("bv_ngat_giua_popen_va_gan_chau_bi_thu_hoi",
         m.get("exit") not in (0, 1, 2, 10, 40, None) and c.get("trong_job_goc") is True
         and c.get("argv", [None, None])[:2] == [H.PYTHON, "-B"] and tt.get("con_song") is False and cay_tu_thu_hoi(m),
         {"exit": m.get("exit_hex"), "chau": c, "tt": tt, "cay": m.get("cay")})
    m, q = chay_tham_do("loi_python_giua_popen_gan", 30)
    dd = q["don_dep.json"] or {}
    kiem("bv_loi_python_giua_popen_va_gan_don_dep",
         m.get("exit") == 1 and "Traceback" in m.get("stderr", "") and (dd.get("chau") or {}).get("con_song") is False
         and (q["chau_trang_thai"] or {}).get("con_song") is False and cay_tu_thu_hoi(m),
         {"exit": m.get("exit"), "don_dep": dd, "cay": m.get("cay")})
    m, q = chay_tham_do("tran_cung_trong_tien_trinh", 8)
    kiem("bv_tran_cung_trong_tien_trinh_thu_hoi_cay",
         m.get("exit") == 1 and "Timeout (" in q["faulthandler"] and (m.get("giay") or 1e9) < 8 + 20
         and (q["chau_trang_thai"] or {}).get("con_song") is False and cay_tu_thu_hoi(m),
         {"exit": m.get("exit"), "giay": m.get("giay"), "fh": q["faulthandler"][:300], "tt": q["chau_trang_thai"]})
    m, q = chay_tham_do("thieu_du_phong", 20)
    kiem("bv_thieu_du_phong_khong_bat_dau_viec", m.get("exit") == 10 and q["het_gio.json"] is not None
         and q["da_lam.json"] is None and q["chau.json"] is None, {"exit": m.get("exit"), "het_gio": q["het_gio.json"]})


def tham_do(argv: list[str]) -> int:
    """Tien trinh THAM DO (con cua fixture, cung interpreter + -B): kiem bao ve lan goi bang CHINH bao_ve_lan_goi /
    khoi_chay_trong_job / can_tg. Chi ghi duoi <vong>/tham_do/<loai>/."""
    try:
        loai, d, tran = argv[0], Path(os.path.abspath(argv[1])), float(argv[2])
    except (IndexError, ValueError):
        print("tham do: doi so sai", file=sys.stderr)
        return H.EXIT_HOP_DONG
    if loai not in THAM_DO or not H._trong(d, KD.VUNG_FIXTURE_B) or d.parent.name != "tham_do" or not 0 < tran <= 60:
        print("tham do: doi so sai", file=sys.stderr)
        return H.EXIT_HOP_DONG
    bc, loi = VB.kiem_bytecode()
    if loi:
        print(f"[LOI HOP DONG] {loi}", file=sys.stderr)
        return H.EXIT_HOP_DONG
    vung = [d]
    H.kiem_duong_ghi(d, [d.parent])
    H.tao_thu_muc(d, [d.parent])
    fh = open(H.kiem_duong_ghi(d / "faulthandler.log", vung), "w", encoding="utf-8")
    faulthandler.enable(file=fh, all_threads=True)
    T.HAN = T.MEM = time.time() + tran
    bv = bao_ve_lan_goi(time.time() + tran, fh)
    H.ghi_json(d / "tham_do.json", {"loai": loai, "pid": os.getpid(), "bytecode": bc, "bao_ve": bv}, vung)
    if not bv.get("dat"):
        return VA.EXIT_DUNG
    if loai == "thieu_du_phong":
        try:
            can_tg("viec trong tien trinh (mo phong tai tinh / chup pyc)", tran + 30)
        except HetGio as e:
            H.ghi_json(d / "het_gio.json", {"ly_do": str(e)}, vung)
            return H.EXIT_CHUA_XONG
        H.ghi_json(d / "da_lam.json", {}, vung)
        return 0
    cmd = [H.PYTHON, "-B", str(THIS), "--ngu", "120"]

    def ghi_chau(p):
        H.ghi_json(d / "chau.json", {"pid": p.pid, "thoi_diem_tao": VA.thoi_diem_tao(int(p._handle)), "argv": cmd,
                                     "trong_job_goc": VA.trong_job(p)}, vung)
    with open(H.kiem_duong_ghi(d / "chau.stdout.txt", vung), "wb") as fo, \
            open(H.kiem_duong_ghi(d / "chau.stderr.txt", vung), "wb") as fe:
        if loai == "tran_cung_trong_tien_trinh":
            import subprocess
            p = subprocess.Popen(cmd, cwd=str(H.ROOT), env=env_con(), stdout=fo, stderr=fe)   # ke thua job goc
            ghi_chau(p)
            time.sleep(tran * 10)                      # viec trong tien trinh vuot tran -> tran cung _exit(1)
            return 0
        tiem = "giua_popen_va_gan" if loai == "ngat_giua_popen_gan" else "giua_popen_va_gan_python"
        try:
            khoi_chay_trong_job(cmd, env_con(), fo, fe, time.time() + 30, tiem=tiem, sau_tao=ghi_chau)
        except RuntimeError as e:
            c = H.doc_json(d / "chau.json")
            H.ghi_json(d / "don_dep.json", {"loi": str(e), "chau": VA.trang_thai_con(c["pid"], c["thoi_diem_tao"])}, vung)
            raise
    return 0


# ---------------------------------------------------------------------------
# Ca kiem
# ---------------------------------------------------------------------------
def dau_marker(S: Path, run: str, v: str):
    p = S / "tracking" / "BDDCOCOVID-val" / run / "xong" / f"{v}.xong.json"
    return [H.sha256_file(p), os.stat(p).st_mtime_ns] if p.is_file() else None


def bat_dau_trong(S: Path, stt_tu: int) -> list[str]:
    return [e["don_vi"] for x in lan_chay(S) if x["stt"] >= stt_tu
            for e in VA.doc_nhat_ky(x["dir"] / "tien_do.jsonl")[0] if e.get("su_kien") == "bat_dau" and e.get("pha") == "chay"]


def bam(ps: list[Path]) -> dict:
    return {H.rel(p): (H.sha256_file(p) if p.is_file() else None) for p in ps}


def ca_thanh_cong():
    yeu_cau("tc_tien_kiem_exit_0", "tc_manifest_co_phu_luc_va_dinh_danh", "tc_manifest_phan_loai_moi_file_van_hanh",
            "tc_mo_phien_exit_0", "tc_tracking_exit_10_tiep_tuc", "tc_gioi_han_5_dung_5_don_vi", "tc_tracking_xong_exit_0",
            "tc_tracking_du_don_vi", "tc_resume_khong_ghi_lai_don_vi_hoan_tat", "tc_resume_khong_bat_dau_lai_don_vi_hoan_tat",
            "tc_cham_gioi_han_2_exit_10", "tc_cham_exit_0", "tc_lap_tracker_exit_0", "tc_cham_lai_exit_0",
            "tc_tong_hop_gioi_han_2_exit_10", "tc_tong_hop_exit_0", "tc_bootstrap_chia_khoi_trung_nguyen_khoi_dong_mau_thu_tu",
            "tc_bootstrap_ci_108_byte_trung_nguyen_khoi", "tc_bootstrap_so_khoi", "tc_hau_kiem_nguyen_khoi_dieu_phoi_dat",
            "tc_kiem_khoa_exit_0", "tc_hau_kiem_gioi_han_2_exit_10", "tc_hau_kiem_exit_0", "tc_hau_kiem_dat",
            "tc_thieu_du_phong_exit_10_khong_tao_lan_chay", "hk_ci_108_bi_sua_phat_hien", "hk_khoi_bootstrap_bi_sua_phat_hien",
            "hk_phu_luc_doi_sau_mo_phien_chan", "hk_file_la_van_hanh_sau_mo_phien_chan",
            "hk_artifact_tai_lap_bi_sua_phat_hien", "lap_tracker_da_xong_kiem_lai_artifact_lech_chan",
            "hk_artifact_cham_lai_bi_sua_phat_hien", "cham_lai_da_xong_kiem_lai_artifact_lech_chan",
            "tc_bang_ket_qua_lap_cham_lai_khai_dat_khong_doi_khi_sua", "hk_dat_lai_sau_khoi_phuc",
            "tc_A_nguyen_byte_sau_thu_sua", "tc_moi_cha_con_dung_interpreter_va_B")
    S = S_("thanh_cong")
    ctx = D.Ctx(T.GT)
    runs = [r["run"] for r in ctx.runs()]
    m = lenh("tc_tien_kiem", b(S, "tien-kiem"), S)
    kiem_lenh("tc_tien_kiem_exit_0", m, 0)
    if (S / "tien_kiem_b" / "ket_qua.json").is_file():
        kq = H.doc_json(S / "tien_kiem_b" / "ket_qua.json")
        man = kq["manifest"]
        kiem("tc_manifest_co_phu_luc_va_dinh_danh", len(man["phu_luc_dung_da_go"]) == 2 and man["goi_a_sha256"] and
             all(set(x) == {"duong", "so_byte", "sha256"} for x in man["phu_luc_dung_da_go"]), man["phu_luc_dung_da_go"])
        loai = sorted({v[0] for v in man["van_hanh_a"].values()})
        kiem("tc_manifest_phan_loai_moi_file_van_hanh", loai == ["goi_a", "loai_tru:goi_cha", "loai_tru:kiem_goi_a",
                                                                   "loai_tru:lan_chay_sau_goi", "loai_tru:log_cha",
                                                                   "loai_tru:so_crash", "phu_luc"], loai)
    kiem_lenh("tc_mo_phien_exit_0", lenh("tc_mo_phien", b(S, "mo-phien"), S), 0)
    m = lenh("tc_tracking_gioi_han_5", b(S, "tracking", "--gioi-han-don-vi", "5"), S)
    kiem_lenh("tc_tracking_exit_10_tiep_tuc", m, 10)
    dau5 = nho("tc_dau5", lambda: {**{f"{runs[0]}:{v}": dau_marker(S, runs[0], v) for v in ctx.videos},
                                   **{f"{runs[1]}:{v}": dau_marker(S, runs[1], v) for v in ctx.videos[:2]}})
    kiem("tc_gioi_han_5_dung_5_don_vi", all(dau5.values()) and len(dau5) == 5, dau5)
    stt_sau = stt_cua(m) + 1
    m = chay_toi_xong(S, "tc_tracking", "tracking")
    kiem_lenh("tc_tracking_xong_exit_0", m, 0)
    du = sum(1 for r in runs for v in ctx.videos if dau_marker(S, r, v))
    kiem("tc_tracking_du_don_vi", du == len(runs) * len(ctx.videos), du)
    kiem("tc_resume_khong_ghi_lai_don_vi_hoan_tat", all(dau_marker(S, *k.split(":")) == x for k, x in dau5.items()))
    kiem("tc_resume_khong_bat_dau_lai_don_vi_hoan_tat",
         not (set(f"tracking:{k}" for k in dau5) & set(bat_dau_trong(S, stt_sau))))
    kiem_lenh("tc_cham_gioi_han_2_exit_10", lenh("tc_cham_gioi_han_2", b(S, "cham", "--gioi-han-don-vi", "2"), S), 10)
    kiem_lenh("tc_cham_exit_0", chay_toi_xong(S, "tc_cham", "cham"), 0)
    kiem_lenh("tc_lap_tracker_exit_0", chay_toi_xong(S, "tc_lap", "lap-tracker"), 0)
    kiem_lenh("tc_cham_lai_exit_0", chay_toi_xong(S, "tc_cham_lai", "cham-lai"), 0)
    kiem_lenh("tc_tong_hop_gioi_han_2_exit_10", lenh("tc_tong_hop_gh2", b(S, "tong-hop", "--gioi-han-don-vi", "2"), S), 10)
    kiem_lenh("tc_tong_hop_exit_0", chay_toi_xong(S, "tc_tong_hop", "tong-hop"), 0)
    if (S / "tong_hop" / "xong.json").is_file():
        # F1 + P43284b78: tai tinh (pandas / boot_stats) trong lenh con --viec, checkpoint truoc khi chay
        dc = nho("tc_doi_chieu_bootstrap", lambda: viec_con("doi_chieu_bootstrap", 150), can=150)
        kiem("tc_bootstrap_chia_khoi_trung_nguyen_khoi_dong_mau_thu_tu", dc["dong_trung"] and not dc["mau_lech"], dc)
        kiem("tc_bootstrap_ci_108_byte_trung_nguyen_khoi", dc["csv_trung"] and dc["so_ref"] == 108, dc["so_ref"])
        kiem("tc_bootstrap_so_khoi", len(dc["khoi"]) == dc["so_khoi_ky_vong"], dc["khoi"])
        kiem("tc_hau_kiem_nguyen_khoi_dieu_phoi_dat", dc["kiem_lai_tong_hop"] == [], dc["kiem_lai_tong_hop"])
    kiem_lenh("tc_kiem_khoa_exit_0", lenh("tc_kiem_khoa", b(S, "kiem-khoa"), S), 0)
    kiem_lenh("tc_hau_kiem_gioi_han_2_exit_10", lenh("tc_hau_kiem_gh2", b(S, "hau-kiem", "--gioi-han-don-vi", "2"), S), 10)
    m = chay_toi_xong(S, "tc_hau_kiem", "hau-kiem")
    kiem_lenh("tc_hau_kiem_exit_0", m, 0)
    hk = H.doc_json(S / "hau_kiem.json") if (S / "hau_kiem.json").is_file() else {}
    kiem("tc_hau_kiem_dat", hk.get("dat") is True and hk.get("loi") == [], hk)
    n0 = len(lan_chay(S))
    m = lenh("tc_thieu_du_phong", b(S, "hau-kiem", "--han-chot", f"{time.time() + 20:.3f}"), S)
    kiem("tc_thieu_du_phong_exit_10_khong_tao_lan_chay", m.get("exit") == 10 and m.get("lan_chay_moi") == 0
         and "THIEU THOI GIAN" in m.get("stdout", ""), {"exit": m.get("exit"), "moi": m.get("lan_chay_moi"), "n0": n0})
    # Hau kiem tai tinh: dau ra bi sua du kiem_chung khai dat -> phat hien; khoi phuc -> dat lai
    with sua_tam(S / "tong_hop" / "ci_108.csv", lat_byte(S / "tong_hop" / "ci_108.csv")):
        m = lenh("tc_hk_sua_ci", b(S, "hau-kiem"), S)
    kiem_lenh("hk_ci_108_bi_sua_phat_hien", m, 2, None, "ci_108.csv khac tinh lai")
    with sua_tam(S / "tong_hop" / "khoi" / "0000.npy", lat_byte(S / "tong_hop" / "khoi" / "0000.npy")):
        m = lenh("tc_hk_sua_khoi", b(S, "hau-kiem"), S)
    kiem_lenh("hk_khoi_bootstrap_bi_sua_phat_hien", m, 2, None, "0000")
    pl = T.A / "van_hanh" / "dung_da_go" / "dung_lan_0010.json"
    with sua_tam(pl, pl.read_bytes() + b" "):
        m = lenh("tc_hk_sua_phu_luc", b(S, "hau-kiem"), S)
    kiem_lenh("hk_phu_luc_doi_sau_mo_phien_chan", m, 2, "phu_luc", "so_byte")
    with sua_tam(T.A / "van_hanh" / "la_sau_mo.txt", b"la"):
        m = lenh("tc_la_van_hanh_sau_mo", b(S, "tracking"), S)
    kiem_lenh("hk_file_la_van_hanh_sau_mo_phien_chan", m, 2, "van_hanh", "thua van_hanh")
    # F5: artifact tai lap / cham lai bi sua trong khi bang ket_qua (khai trung_byte=true) giu nguyen -> hau-kiem va
    # buoc da xong deu phat hien qua marker + bam + so byte tung run
    tlB = ctx.gt["tai_lap_cong_B"]
    bang = [S / "lap" / f"ket_qua_{tlB['video']}.json", S / "lap" / f"ket_qua_{tlB['video']}.xong.json",
            S / "cham_lai" / "ket_qua.json", S / "cham_lai" / "ket_qua.xong.json"]
    bang0 = nho("tc_bang_lap_cham_lai_truoc_sua", lambda: bam(bang))
    p_lap = S / "lap" / ctx.split / runs[0] / "data" / f"{tlB['video']}.txt"
    with sua_tam(p_lap, lat_byte(p_lap)):
        m1 = lenh("tc_hk_sua_lap", b(S, "hau-kiem"), S)
        m2 = lenh("tc_lap_lai_sua_lap", b(S, "lap-tracker"), S)
    kiem_lenh("hk_artifact_tai_lap_bi_sua_phat_hien", m1, 2, None, "tai lap:", runs[0])
    kiem_lenh("lap_tracker_da_xong_kiem_lai_artifact_lech_chan", m2, 2, None, "lap-tracker kiem lai", runs[0])
    p_cl = S / "cham_lai" / tlB["cham_lai"][0] / "raw_fields.json"
    with sua_tam(p_cl, lat_byte(p_cl)):
        m1 = lenh("tc_hk_sua_cham_lai", b(S, "hau-kiem"), S)
        m2 = lenh("tc_cham_lai_lai_sua", b(S, "cham-lai"), S)
    kiem_lenh("hk_artifact_cham_lai_bi_sua_phat_hien", m1, 2, None, "cham lai:", tlB["cham_lai"][0])
    kiem_lenh("cham_lai_da_xong_kiem_lai_artifact_lech_chan", m2, 2, None, "cham-lai kiem lai", tlB["cham_lai"][0])
    kiem("tc_bang_ket_qua_lap_cham_lai_khai_dat_khong_doi_khi_sua", bam(bang) == bang0 and all(bang0.values()), bang0)
    m = lenh("tc_hk_dat_lai", b(S, "hau-kiem"), S)
    kiem_lenh("hk_dat_lai_sau_khoi_phuc", m, 0)
    kiem("tc_A_nguyen_byte_sau_thu_sua", nguyen_A())
    # Bang chung -B / interpreter cua moi lan chay cha/con trong phien thanh cong
    ok, bad = 0, []
    for x in lan_chay(S):
        g = x["goi"] or {}
        kc = H.doc_json(x["dir"] / "ket_qua_con.json") if (x["dir"] / "ket_qua_con.json").is_file() else {}
        dat = g.get("argv_con", [None, None])[:2] == [H.PYTHON, "-B"] and "-B" in (g.get("bytecode_cha") or {}).get("tuy_chon", []) \
            and "-B" in ((kc.get("bytecode") or {}).get("tuy_chon") or [])
        ok += dat
        if not dat:
            bad.append(x["ten"])
    kiem("tc_moi_cha_con_dung_interpreter_va_B", not bad and ok > 0, bad)


def vc_doi_chieu() -> dict:
    """(Chay trong tien trinh --viec) doi chieu bootstrap chia khoi voi nguyen khoi tren phien thanh_cong."""
    import thong_ke_bdd as Tk
    import thong_ke_tiep_tuc as TT
    S, ctx = S_("thanh_cong"), D.Ctx(T.GT)
    raw, comb, _ = D.doc_cham(ctx, S)
    nhom = TT.nhom_raws(ctx.runs(), raw)
    cs = D.doc_chi_so(ctx, S / "tong_hop" / "chi_so_bootstrap.npy")
    ps = sorted((S / "tong_hop" / "khoi").glob("[0-9][0-9][0-9][0-9].npy"))
    mau = TT.ghep([TT.doc_npy(p) for p in ps], ctx.so_mau)
    _, obs = TT.chuan_bi(nhom, ctx.videos)
    rows = TT.hang_ci(nhom, obs, mau)
    dc = TT.doi_chieu_nguyen_khoi(nhom, ctx.videos, cs, mau, rows)
    ref = Tk.bang_ci(nhom, ctx.videos, cs)
    return {"dong_trung": bool(dc["dong_trung"]), "mau_lech": list(dc["mau_lech"]), "so_dong": dc["so_dong"],
            "csv_trung": (S / "tong_hop" / "ci_108.csv").read_bytes() == D.csv_bytes(ref), "so_ref": len(ref),
            "khoi": [p.name for p in ps], "so_khoi_ky_vong": len(TT.khoi(ctx.so_mau, 64)),
            "kiem_lai_tong_hop": D.kiem_lai_tong_hop(ctx, S)}


VIEC = {"the_gioi": tl_the_gioi, "A_tien_kiem": tl_A_tien_kiem, "A_goi": tl_A_goi, "doi_chieu_bootstrap": vc_doi_chieu}


def viec(argv: list[str]) -> int:
    """Tien trinh VIEC (P43284b78; con cua fixture qua lenh(), cung interpreter + -B, trong job goc + job lenh): MOT buoc can
    thu vien ben thu ba. Chi ghi duoi <vong>/ (cung vung voi buoc trong tien trinh truoc day); ket qua <vong>/viec/<lenh>.json
    (khong ghi de). Han = han chung cua lan goi fixture (khong cap lai). Exit: 0 xong; 10 chua xong (cache tam dung);
    2 doi so / hop dong; traceback 1 (so crash cua cha xet ngay sau lenh -> dung)."""
    try:
        ten, V, han, mem, ma = argv[0], Path(os.path.abspath(argv[1])), float(argv[2]), float(argv[3]), argv[4]
    except (IndexError, ValueError):
        print("viec: doi so sai", file=sys.stderr)
        return H.EXIT_HOP_DONG
    ph = V.parent
    if len(argv) != 5 or not (ten in VIEC or ten.startswith("A_cache:")) or not H._trong(ph, KD.VUNG_FIXTURE_B) \
            or ph == KD.VUNG_FIXTURE_B or not (V.name.startswith("vong_") and V.name[5:].isdigit()) \
            or not (V / "vong.json").is_file() or ma != "viec_" + ten.replace(":", "__") + ma[-3:] \
            or not (ma[-3] == "_" and ma[-2:].isdigit()) or not time.time() < han:
        print(f"viec: doi so sai {argv}", file=sys.stderr)
        return H.EXIT_HOP_DONG
    bc, loi = VB.kiem_bytecode()
    if loi:
        print(f"[LOI HOP DONG] {loi}", file=sys.stderr)
        return H.EXIT_HOP_DONG
    faulthandler.enable(all_threads=True)              # stderr = file cua lenh (bang chung crash native)
    faulthandler.dump_traceback_later(max(1.0, han - 2.0 - time.time()), exit=True)
    T.V, T.PH, T.HAN, T.MEM = V, ph, han, mem
    gan_vong(V)
    p = H.kiem_duong_ghi(V / "viec" / f"{ma}.json", [V])
    if p.exists():
        print(f"[LOI HOP DONG] {H.rel(p)} da ton tai (khong ghi de)", file=sys.stderr)
        return H.EXIT_HOP_DONG
    H.chuan_bi_moi_truong(V, [V])                      # TRUOC import ben thu ba (cau hinh / TEMP trong vong)
    tb0 = ben_thu_ba()
    try:
        if ten.startswith("A_cache:"):
            det = ten.split(":", 1)[1]
            if det not in D.Ctx(T.GT).detectors():
                raise H.LoiHopDong(f"detector {det!r} khong co trong giao thuc fixture")
            v = tl_A_cache(det)
        else:
            v = VIEC[ten]()
    except HetGio as e:
        print(f"[CHUA XONG] {e}")
        return H.EXIT_CHUA_XONG
    except H.LoiHopDong as e:
        print(f"[LOI HOP DONG] {e}", file=sys.stderr)
        return H.EXIT_HOP_DONG
    H.ghi_json(p, {"v": v, "ten": ten, "pid": os.getpid(), "bytecode": bc, "ben_thu_ba_truoc": tb0,
                   "ben_thu_ba_sau": ben_thu_ba()}, [V])
    faulthandler.cancel_dump_traceback_later()
    return 0


def ca_kiem_lai_buoc_xong():
    """F4: buoc tien-kiem / mo-phien da co marker hop le VAN kiem lai dau vao hien tai truoc khi bo qua."""
    yeu_cau(*[f"kl_{bt}_lai_{x}" for bt in ("tk", "mp") for x in ("plan_muc0_doi_chan", "dung_json_chan", "phu_luc_doi_chan",
                                                                    "dat_sau_khoi_phuc")],
            "kl_artifact_hoan_tat_giu_nguyen", "kl_A_nguyen_byte")
    S = S_("thanh_cong")
    tep = [S / "tien_kiem_b" / "ket_qua.json", S / "tien_kiem_b" / "xong.json", S / "phien.json", S / "manifest_b.json",
           S / "mo_phien.xong.json"]
    dau = nho("kl_dau", lambda: bam(tep))
    plan = T.FX / "PLAN_DA_CHOT.md"
    pb = plan.read_bytes()
    i, j = CB.tach_plan(pb)["byte_muc0"]
    k = (i + j) // 2
    plan_moi = pb[:k] + (b"X" if pb[k:k + 1] != b"X" else b"Y") + pb[k + 1:]
    pl = T.A / "van_hanh" / "dung_da_go" / "quyet_dinh_go_dung.json"
    for buoc, bt in (("tien-kiem", "tk"), ("mo-phien", "mp")):
        with sua_tam(plan, plan_moi):
            m = lenh(f"kl_{bt}_plan", b(S, buoc), S)
        kiem_lenh(f"kl_{bt}_lai_plan_muc0_doi_chan", m, 2, "plan", "sha256_toan_bo")
        with sua_tam(T.A / "van_hanh" / "dung.json", b"{}"):
            m = lenh(f"kl_{bt}_dung", b(S, buoc), S)
        kiem_lenh(f"kl_{bt}_lai_dung_json_chan", m, 2, "van_hanh", "dung.json")
        with sua_tam(pl, pl.read_bytes() + b"\n"):
            m = lenh(f"kl_{bt}_phu_luc", b(S, buoc), S)
        kiem_lenh(f"kl_{bt}_lai_phu_luc_doi_chan", m, 2, "phu_luc", "sha256")
        m = lenh(f"kl_{bt}_dat", b(S, buoc), S)
        x = [y for y in lan_chay(S) if y["stt"] == stt_cua(m)]
        ev = VA.doc_nhat_ky(x[0]["dir"] / "tien_do.jsonl")[0] if x else []
        dv_kl = "tien-kiem:kiem_lai" if buoc == "tien-kiem" else "mo-phien:chuan_bi"   # don vi kiem lai cua con
        kiem(f"kl_{bt}_lai_dat_sau_khoi_phuc", m.get("exit") == 0
             and any(e.get("su_kien") == "xong" and e.get("don_vi") == dv_kl for e in ev)
             and any(e.get("su_kien") == "bo_qua" and e.get("don_vi") == buoc for e in ev), {"exit": m.get("exit"), "su_kien": ev[-6:]})
    kiem("kl_artifact_hoan_tat_giu_nguyen", bam(tep) == dau and all(dau.values()), {"truoc": dau, "sau": bam(tep)})
    kiem("kl_A_nguyen_byte", nguyen_A() and plan.read_bytes() == pb)


AM_TEN = ("thieu_phu_luc", "doi_phu_luc", "thua_dung_da_go", "file_la_van_hanh", "dung_json_chan", "plan_muc0_doi",
          "plan_truoc_muc0_doi", "plan_sau_muc0_doi", "payload_npz_doi", "payload_thua", "provenance_cache_doi",
          "goi_a_json_doi", "sha256sums_doi", "marker_goi_hong", "khoa_doi", "spec_cau_hinh_doi_anh_khoa", "review_digest_doi",
          "json_goi_cat_cut", "json_goi_sai_kieu", "json_kiem_goi_a_cat_cut")
AM_PD = ("pd_duyet_false", "pd_sai_goi", "pd_sai_phien", "pd_sai_ma_b")
AM_DOI_SO = ("so_doan_2", "ngan_sach_421", "cong_A")
AM_JSON_HD = ("json_review_cat_cut", "json_hop_dong_cat_cut", "json_hop_dong_sai_kieu_truong")
AM_JSON_GT = ("json_giao_thuc_cat_cut", "json_giao_thuc_manifest_sai_kieu", "json_giao_thuc_truong_sai_kieu",
              "json_giao_thuc_fps_0")


def ca_am_dau_vao():
    yeu_cau(*[f"am_{t}" for t in AM_TEN], *[f"am_{t}_A_nguyen_byte_sau_khoi_phuc" for t in AM_TEN],
            "am_review_verdict_khong_approve", *[f"am_{t}" for t in AM_PD], *[f"am_{t}_khong_ghi_truoc_kiem" for t in AM_PD],
            "am_json_phe_duyet_cat_cut", "am_json_phe_duyet_cat_cut_khong_ghi_truoc_kiem", *[f"am_{t}" for t in AM_JSON_HD],
            *[f"am_{t}" for t in AM_JSON_GT], *[f"am_{t}_khong_ghi_truoc_kiem" for t in AM_JSON_GT],
            "am_con_goi_stt_null", "am_con_goi_stt_null_khong_ghi",
            "am_that_khong_phe_duyet", "am_that_tiem_loi_bi_chan", "am_that_khong_tao_thu_muc", *[f"am_{t}" for t in AM_DOI_SO],
            "am_duong_traversal_thoat_vung_that", "am_duong_traversal", "am_duong_traversal_khong_ghi", "am_duong_reparse",
            "am_A_nguyen_byte_cuoi")
    A, vh = T.A, T.A / "van_hanh"
    pl0, pl1 = vh / "dung_da_go" / "dung_lan_0010.json", vh / "dung_da_go" / "quyet_dinh_go_dung.json"
    plan = T.FX / "PLAN_DA_CHOT.md"
    pb = plan.read_bytes()
    t = CB.tach_plan(pb)
    i, j = t["byte_muc0"]
    npz = sorted((A / "cache").glob("*/*.npz"))[0]
    prov = sorted((A / "cache").glob("*/fxv-a.json"))[0]
    yaml = sorted((T.FX / "yaml").glob("*.yaml"))[0]
    gb = (A / "goi_a.json").read_bytes()
    kq_kg = vh / "kiem_goi_a" / "vong_01" / "ket_qua.json"
    ca = [("thieu_phu_luc", pl0, None, "phu_luc", ("thieu",)),
          ("doi_phu_luc", pl1, pl1.read_bytes() + b"\n", "phu_luc", ("sha256",)),
          ("thua_dung_da_go", vh / "dung_da_go" / "them.json", b"{}", "phu_luc", ("thua dung_da_go",)),
          ("file_la_van_hanh", vh / "la.txt", b"x", "van_hanh", ("thua van_hanh",)),
          ("dung_json_chan", vh / "dung.json", b"{}", "van_hanh", ("chan: van_hanh/dung.json",)),
          ("plan_muc0_doi", plan, pb[:(i + j) // 2] + (b"X" if pb[(i + j) // 2:(i + j) // 2 + 1] != b"X" else b"Y")
           + pb[(i + j) // 2 + 1:], "plan", ("sha256_toan_bo",)),
          ("plan_truoc_muc0_doi", plan, b"X" + pb[1:], "plan", ("sha256_truoc_muc0",)),
          ("plan_sau_muc0_doi", plan, pb[:-1] + b"X", "plan", ("sha256_sau_muc0",)),
          ("payload_npz_doi", npz, lat_byte(npz), "payload", ("lech: ",)),
          ("payload_thua", npz.parent / "thua.npz", b"x", "payload", ("thua: ",)),
          ("provenance_cache_doi", prov, lat_byte(prov), "payload", ("fxv-a.json",)),
          ("goi_a_json_doi", A / "goi_a.json", gb + b" ", "goi", ("goi_a_sha256",)),
          ("sha256sums_doi", A / "SHA256SUMS", (A / "SHA256SUMS").read_bytes() + b"\n", "sha256sums", ("SHA256SUMS",)),
          ("marker_goi_hong", A / "goi_a.xong.json", (A / "goi_a.xong.json").read_bytes().replace(b"dong-goi-a", b"dong-goi-x"),
           "marker_goi", ("goi_a.xong.json",)),
          ("khoa_doi", A / "khoa" / "khoa.json", (A / "khoa" / "khoa.json").read_bytes() + b" ", "payload", ("khoa/khoa.json",)),
          ("spec_cau_hinh_doi_anh_khoa", yaml, yaml.read_bytes() + b"\n# fixture\n", "khoa", ("anh khoa",)),
          ("review_digest_doi", T.FX / "review_a.json", (T.FX / "review_a.json").read_bytes() + b" ", "review", ("sha256 review",)),
          # F9: JSON dau vao cat cut / sai kieu -> dung nhom, exit 2 (khong phai loi Python)
          ("json_goi_cat_cut", A / "goi_a.json", gb[:len(gb) // 2], "goi", ("khong phai JSON hop le",)),
          ("json_goi_sai_kieu", A / "goi_a.json", b"[]\n", "goi", ("sai kieu goc",)),
          ("json_kiem_goi_a_cat_cut", kq_kg, kq_kg.read_bytes()[:len(kq_kg.read_bytes()) // 2], "van_hanh",
           ("kiem_goi_a", "khong phai JSON hop le"))]
    if tuple(x[0] for x in ca) != AM_TEN:
        raise H.LoiHopDong("danh muc ca am lech AM_TEN")
    for ten, p, moi, nhom, can in ca:
        S = S_(f"am_{ten}")
        with sua_tam(p, moi):
            m = lenh(f"am_{ten}", b(S, "tien-kiem"), S)
        kiem_lenh(f"am_{ten}", m, 2, nhom, *can)
        kiem(f"am_{ten}_A_nguyen_byte_sau_khoi_phuc", nguyen_A() and plan.read_bytes() == pb)
    S = S_("am_review_verdict")
    m = lenh("am_review_verdict", b(S, "tien-kiem", hd=T.FX / "hop_dong_review_revise.json",
                                    pd=phe_duyet(S, "am_review_verdict_pd", hop_dong_cong_b_sha256=H.sha256_file(T.FX / "hop_dong_review_revise.json"),
                                                 review_a_sha256=H.sha256_file(T.FX / "review_a_revise.json"),
                                                 ma_b_sha256=KD.ma_b(KD.nap_hop_dong(T.FX / "hop_dong_review_revise.json"))["sha256"])), S)
    kiem_lenh("am_review_verdict_khong_approve", m, 2, "review", "verdict")
    S = S_("am_json_review_cat_cut")
    hrc = T.FX / "hop_dong_review_cat_cut.json"
    m = lenh("am_json_review_cat_cut", b(S, "tien-kiem", hd=hrc,
                                         pd=phe_duyet(S, "am_json_review_cat_cut_pd", hop_dong_cong_b_sha256=H.sha256_file(hrc),
                                                      review_a_sha256=H.sha256_file(T.FX / "review_a_cat_cut.json"),
                                                      ma_b_sha256=KD.ma_b(KD.nap_hop_dong(hrc))["sha256"])), S)
    kiem_lenh("am_json_review_cat_cut", m, 2, "review", "khong phai JSON hop le")
    for ten, f, can in (("json_hop_dong_cat_cut", "hop_dong_cat_cut.json", "khong phai JSON hop le"),
                        ("json_hop_dong_sai_kieu_truong", "hop_dong_sai_kieu.json", "a.so_file_goi sai kieu")):
        S = S_(f"am_{ten}")
        m = lenh(f"am_{ten}", b(S, "tien-kiem", hd=T.FX / f), S)
        kiem_lenh(f"am_{ten}", m, 2, "hop_dong", can)
    # B0-F9 (phan con lai): giao thuc cat cut / manifest no tro sai kieu / truong sai kieu; goi.json cua --con stt=null
    for ten, f, can in (("json_giao_thuc_cat_cut", "giao_thuc_cat_cut.json", "khong phai JSON hop le"),
                        ("json_giao_thuc_manifest_sai_kieu", "giao_thuc_manifest_sai_kieu.json", "videos phai la danh sach chuoi"),
                        ("json_giao_thuc_truong_sai_kieu", "giao_thuc_sai_kieu.json", "tracker.fps sai kieu"),
                        ("json_giao_thuc_fps_0", "giao_thuc_fps_0.json", "phai la so huu han > 0")):
        S = S_(f"am_{ten}")
        m = lenh(f"am_{ten}", b(S, "tien-kiem", "--giao-thuc", str(T.FX / f)), S)
        kiem_lenh(f"am_{ten}", m, 2, "giao_thuc", can)
        kiem(f"am_{ten}_khong_ghi_truoc_kiem", not S.exists())
    S = T.V / "phien_gia" / "am_con_stt_null"          # ngoai <vong>/phien/: lan chay gia khong vao kiem ket_thuc / so crash
    dc = S / "van_hanh" / "lan_chay" / "0001_tien-kiem"
    _ghi_neu_chua(dc / "goi.json", {"stt": None, "luot": VB.LUOT, "buoc": "tien-kiem", "phien": H.rel(S), "giao_thuc": str(T.GT),
                                    "hop_dong": str(T.HD), "phe_duyet": str(phe_duyet(S)), "han_mem": time.time() + 60,
                                    "lan_thu_truoc": {}, "tiem_loi": []})
    m = lenh("am_con_stt_null", ["--con", str(dc)])
    kiem_lenh("am_con_goi_stt_null", m, 2, "lan_chay", "stt sai kieu")
    kiem("am_con_goi_stt_null_khong_ghi", sorted(p.name for p in dc.iterdir()) == ["goi.json"], sorted(p.name for p in dc.iterdir()))
    for ten, doi, can in (("pd_duyet_false", {"duyet": False}, "duyet"), ("pd_sai_goi", {"goi_a_sha256": "0" * 64}, "goi_a_sha256"),
                          ("pd_sai_phien", {"phien_b": "results/phase2/r61_bdd/tests/cong_b_20261026/khac"}, "phien_b"),
                          ("pd_sai_ma_b", {"ma_b_sha256": "1" * 64}, "ma_b_sha256")):
        S = S_(f"am_{ten}")
        m = lenh(f"am_{ten}", b(S, "tien-kiem", pd=phe_duyet(S, f"am_{ten}", **doi)), S)
        kiem_lenh(f"am_{ten}", m, 2, "phe_duyet", can)
        kiem(f"am_{ten}_khong_ghi_truoc_kiem", not S.exists())
    S = S_("am_json_phe_duyet_cat_cut")
    pc = T.FX / "phe_duyet" / "am_json_phe_duyet_cat_cut.json"
    if not pc.is_file():
        pg = phe_duyet(S, "am_json_phe_duyet_cat_cut_goc").read_bytes()
        H.ghi_bytes(pc, pg[:len(pg) // 2], [T.V])
    m = lenh("am_json_phe_duyet_cat_cut", b(S, "tien-kiem", pd=pc), S)
    kiem_lenh("am_json_phe_duyet_cat_cut", m, 2, "phe_duyet", "khong phai JSON hop le")
    kiem("am_json_phe_duyet_cat_cut_khong_ghi_truoc_kiem", not S.exists())
    tt0 = H.VUNG_THUC_THI.exists()
    Sthat = H.VUNG_THUC_THI / "fixture_khong_duoc_tao"
    m = lenh("am_that_khong_phe_duyet", ["--cong", "B", "--phien", str(Sthat), "--buoc", "tien-kiem", "--so-doan", "1"])
    kiem_lenh("am_that_khong_phe_duyet", m, 2, "phe_duyet", "thieu --phe-duyet")
    m = lenh("am_that_tiem_loi", ["--cong", "B", "--phien", str(Sthat), "--buoc", "tien-kiem", "--so-doan", "1",
                                  *tl({"diem": "khoi_dong", "kieu": "native"})])
    kiem_lenh("am_that_tiem_loi_bi_chan", m, 2, None, "tiem loi chi cho fixture")
    kiem("am_that_khong_tao_thu_muc", H.VUNG_THUC_THI.exists() == tt0 and not Sthat.exists())
    S = S_("am_doi_so")
    for ten, them, can in (("so_doan_2", ["--so-doan", "2"], "--so-doan"), ("ngan_sach_421", ["--ngan-sach-giay", "421"], "ngan-sach"),
                           ("cong_A", ["--cong", "A"], "chi nhan --cong B")):
        m = lenh(f"am_{ten}", b(S, "tien-kiem", *them), S)
        kiem_lenh(f"am_{ten}", m, 2, None, can)
    # F6: traversal THUC SU thoat vung fixture (phien/../../../../ -> results/phase2/r61_bdd/tests/<ten>), kiem truoc
    ngoai = str(T.V / "phien") + "/../../../../ngoai_vung_traversal_b"
    ab = Path(os.path.abspath(ngoai))
    kiem("am_duong_traversal_thoat_vung_that", not H._trong(ab, KD.VUNG_FIXTURE_B) and ".." in ngoai,
         {"duong": ngoai, "chuan_hoa": H.rel(ab)})
    ton0 = nho("am_traversal_ton_tai_truoc", lambda: ab.exists())
    m = lenh("am_duong_traversal", b(Path(ngoai), "tien-kiem", pd=phe_duyet(S_("am_doi_so"), "am_traversal")))
    kiem_lenh("am_duong_traversal", m, 2, None, "phai nam duoi")
    kiem("am_duong_traversal_khong_ghi", ton0 is False and not ab.exists(), {"chuan_hoa": H.rel(ab)})
    import _winapi
    J = T.V / "phien" / "noi_junction"
    if not H.la_reparse(J):
        H.tao_thu_muc(T.V / "phien" / "dich_junction", [T.V])
        H.kiem_duong_ghi(J, [T.V])
        _winapi.CreateJunction(str(T.V / "phien" / "dich_junction"), str(J))
    m = lenh("am_duong_reparse", b(J / "con", "tien-kiem", pd=phe_duyet(S_("am_doi_so"), "am_reparse")))
    kiem_lenh("am_duong_reparse", m, 2, None, "reparse")
    kiem("am_A_nguyen_byte_cuoi", nguyen_A())


def ca_phuc_hoi_journal():
    """F3: journal phuc hoi bi sua (traversal / junction / khong rang buoc ten) -> chan TRUOC moi move/restore."""
    yeu_cau("ph_journal_traversal_chan", "ph_journal_traversal_khong_ghi", "ph_journal_reparse_chan",
            "ph_journal_reparse_khong_ghi", "ph_journal_khong_rang_buoc_duong_chan", "ph_sua_tam_ngoai_vung_chan_truoc_ghi",
            "ph_journal_hop_le_phuc_hoi_dat")
    goc = T.V / "khoi_phuc_ca"

    def journal(ten: str, duong: str, co: bool, sai_ten: bool = False) -> tuple[Path, Path]:
        d = goc / ten
        k = d / f"{'0' * 16 if sai_ten else hashlib.sha256(duong.encode()).hexdigest()[:16]}.json"
        H.ghi_bytes(k.with_suffix(".bytes"), b"x", [goc], ghi_de=True)
        H.ghi_json(k, {"duong": duong, "co": co}, [goc], ghi_de=True)
        return k, d

    def thu(k: Path, d: Path) -> str | None:
        try:
            phuc_hoi(k, d)
            return None
        except H.LoiHopDong as e:
            return str(e)
    duong = str(T.FX) + "/../../../../ngoai_phuc_hoi_traversal.txt"
    k, d = journal("traversal", duong, True)
    e = thu(k, d)
    kiem("ph_journal_traversal_chan", e is not None and "ngoai vung" in e, e)
    kiem("ph_journal_traversal_khong_ghi", not Path(os.path.abspath(duong)).exists() and k.is_file(), H.rel(Path(os.path.abspath(duong))))
    J, dich = T.V / "phien" / "noi_junction_ph", T.V / "phien" / "dich_junction_ph"
    if not H.la_reparse(J):
        import _winapi
        H.tao_thu_muc(dich, [T.V])
        H.kiem_duong_ghi(J, [T.V])
        _winapi.CreateJunction(str(dich), str(J))
    k, d = journal("reparse", str(J / "g.txt"), True)
    e = thu(k, d)
    kiem("ph_journal_reparse_chan", e is not None and "reparse" in e, e)
    kiem("ph_journal_reparse_khong_ghi", not (dich / "g.txt").exists() and k.is_file())
    k, d = journal("sai_ten", str(T.FX / "khong_rang_buoc.txt"), True, sai_ten=True)
    e = thu(k, d)
    kiem("ph_journal_khong_rang_buoc_duong_chan", e is not None and "khong rang buoc" in e
         and not (T.FX / "khong_rang_buoc.txt").exists(), e)
    p = T.V.parent / "ngoai_sua_tam.txt"                # trong lan_01 nhung NGOAI vung sua tam (fx/, phien/)
    kj = T.V / "khoi_phuc" / (hashlib.sha256(str(p).encode()).hexdigest()[:16] + ".json")
    try:
        with sua_tam(p, b"x"):
            pass
        e = None
    except H.LoiHopDong as x:
        e = str(x)
    kiem("ph_sua_tam_ngoai_vung_chan_truoc_ghi", e is not None and not p.exists() and not kj.exists(), e)
    p = S_("ph_hop_le") / "them_tam.txt"
    H.tao_thu_muc(p.parent, [T.V])
    with sua_tam(p, b"tam"):
        co_trong = p.is_file()
    kiem("ph_journal_hop_le_phuc_hoi_dat", co_trong and not p.exists() and any((T.V / "bo").glob(f"*_{p.name}")))


def ca_ma_b_doi():
    yeu_cau("ma_b_tien_kiem", "ma_b_hop_dong_doi_sau_tao_phien_chan", "ma_b_hop_dong_khoi_phuc")
    S = S_("ma_b_doi")
    kiem_lenh("ma_b_tien_kiem", lenh("mab_tien_kiem", b(S, "tien-kiem"), S), 0)
    goc = T.HD.read_bytes()
    moi = goc.replace(b'"phien_ban": 1', b'"phien_ban": 1, "fixture_doi": 1')
    with sua_tam(T.HD, moi):
        hd2 = KD.nap_hop_dong(T.HD)
        pd = phe_duyet(S, "ma_b_doi_pd2", hop_dong_cong_b_sha256=hd2["_sha256"], ma_b_sha256=KD.ma_b(hd2)["sha256"])
        m = lenh("mab_hop_dong_doi", b(S, "mo-phien", pd=pd), S)
    kiem_lenh("ma_b_hop_dong_doi_sau_tao_phien_chan", m, 2, None, "cau_hinh hop_dong_cong_b")
    kiem("ma_b_hop_dong_khoi_phuc", T.HD.read_bytes() == goc)


def tiem_lenh(S: Path, ten: str, buoc: str, *diem, them=()) -> dict:
    return lenh(ten, b(S, buoc, *them, *tl(*diem)), S)


def dv_trk(i: int) -> str:
    ctx = D.Ctx(T.GT)
    runs, vids = [r["run"] for r in ctx.runs()], ctx.videos
    return f"tracking:{runs[i // len(vids)]}:{vids[i % len(vids)]}"


def ca_crash_ghi():
    yeu_cau("cr1_mo_phien", "cr_truoc_ghi_exit_30", "cr_thu_lai_don_vi_sau_crash", "cr_sau_ghi_exit_30_artifact_khong_marker",
            "cr_sao_luu_artifact_do_truoc_phuc_hoi", "cr2_mo_phien", "cr_dang_ghi_marker_exit_30_con_dang_ghi",
            "cr_dang_ghi_duoc_sao_luu_roi_lam_lai", "cr_sau_marker_exit_30_marker_hop_le",
            "cr_sau_marker_khong_chay_lai_don_vi_hoan_tat", "cr_marker_don_vi_hong_exit_2_khong_thu_lai",
            "cr_bo_dem_crash_giu_qua_lan_goi")
    S = S_("crash_1")
    ok = mo(S, "cr1")
    kiem("cr1_mo_phien", ok)
    if not ok:
        return
    u1, u2 = dv_trk(0), dv_trk(1)
    m = tiem_lenh(S, "cr1_truoc_ghi", "tracking", {"diem": "truoc_ghi", "don_vi": u1, "kieu": "native"})
    kiem("cr_truoc_ghi_exit_30", m.get("exit") == 30 and (m.get("ket_thuc") or {}).get("don_vi") == u1, m.get("ket_thuc"))
    m = lenh("cr1_tiep_1", b(S, "tracking", "--gioi-han-don-vi", "1"), S)
    kiem("cr_thu_lai_don_vi_sau_crash", m.get("exit") == 10 and u1 in bat_dau_trong(S, stt_cua(m)), m.get("exit"))
    m = tiem_lenh(S, "cr1_sau_ghi", "tracking", {"diem": "sau_ghi", "don_vi": u2, "kieu": "native"})
    _, run, v = u2.split(":")
    rd = S / "tracking" / "BDDCOCOVID-val" / run
    qs = nho("cr1_sau_ghi", lambda: {"data": (rd / "data" / f"{v}.txt").is_file(), "marker": (rd / "xong" / f"{v}.xong.json").exists()})
    kiem("cr_sau_ghi_exit_30_artifact_khong_marker", m.get("exit") == 30 and qs["data"] and not qs["marker"], qs)
    m = lenh("cr1_tiep_2", b(S, "tracking", "--gioi-han-don-vi", "1"), S)
    sl = sorted((S / "van_hanh" / "artifact_do" / u2.replace(":", "__")).glob("lan_*/sao_luu.json"))
    kiem("cr_sao_luu_artifact_do_truoc_phuc_hoi", m.get("exit") == 10 and sl and dau_marker(S, run, v) is not None
         and all(x["sha256"] for p in sl for x in H.doc_json(p)["muc"]), [H.rel(p) for p in sl])
    S2 = S_("crash_2")
    ok = mo(S2, "cr2")
    kiem("cr2_mo_phien", ok)
    if not ok:
        return
    m = tiem_lenh(S2, "cr2_dang_ghi", "tracking", {"diem": "dang_ghi", "don_vi": u1, "kieu": "native"})
    _, run, v = u1.split(":")
    mk = S2 / "tracking" / "BDDCOCOVID-val" / run / "xong" / f"{v}.xong.json"
    qs = nho("cr2_dang_ghi", lambda: mk.with_name(mk.name + ".dang_ghi").is_file())
    kiem("cr_dang_ghi_marker_exit_30_con_dang_ghi", m.get("exit") == 30 and qs)
    m = lenh("cr2_tiep_1", b(S2, "tracking", "--gioi-han-don-vi", "1"), S2)
    sl = sorted((S2 / "van_hanh" / "artifact_do" / u1.replace(":", "__")).glob("lan_*/sao_luu.json"))
    kiem("cr_dang_ghi_duoc_sao_luu_roi_lam_lai", m.get("exit") == 10 and sl and mk.is_file()
         and not mk.with_name(mk.name + ".dang_ghi").exists() and
         any(".dang_ghi" in json.dumps(H.doc_json(p)) for p in sl))
    m = tiem_lenh(S2, "cr2_sau_marker", "tracking", {"diem": "sau_marker", "don_vi": u2, "kieu": "native"})
    _, run2, v2 = u2.split(":")
    d0 = nho("cr2_sau_marker", lambda: dau_marker(S2, run2, v2))
    kiem("cr_sau_marker_exit_30_marker_hop_le", m.get("exit") == 30 and d0 is not None)
    m = lenh("cr2_tiep_2", b(S2, "tracking", "--gioi-han-don-vi", "1"), S2)
    stt = stt_cua(m)
    x = [y for y in lan_chay(S2) if y["stt"] == stt]
    nk = VA.doc_nhat_ky(x[0]["dir"] / "tien_do.jsonl")[0] if x else []
    kiem("cr_sau_marker_khong_chay_lai_don_vi_hoan_tat", m.get("exit") == 10 and dau_marker(S2, run2, v2) == d0
         and u2 not in bat_dau_trong(S2, stt) and any(e.get("su_kien") == "bo_qua" and e.get("don_vi") == u2 for e in nk))
    p = S2 / "tracking" / "BDDCOCOVID-val" / run / "data" / f"{v}.txt"
    with sua_tam(p, lat_byte(p)):
        m = lenh("cr2_marker_hong", b(S2, "tracking"), S2)
    kiem_lenh("cr_marker_don_vi_hong_exit_2_khong_thu_lai", m, 2, None, "hong")
    dem = VA.dem_crash(lan_chay(S2))
    kiem("cr_bo_dem_crash_giu_qua_lan_goi", dem["tong"] == 2, dem)


def ca_cha_gian_doan():
    yeu_cau("cha1_mo_phien", "cha_crash_khi_con_chay_cay_duoc_thu_hoi", "cha_gian_doan_lan_sau_dung_40",
            "cha_dung_giu_chan_lan_goi_sau", "cha_sau_tao_con_crash_cay_thu_hoi", "cha_sau_tao_con_lan_sau_40_thieu_con_json")
    S = S_("cha_1")
    ok = mo(S, "cha1")
    kiem("cha1_mo_phien", ok)
    if ok:
        m = tiem_lenh(S, "cha1_khi_con_chay", "tracking", {"diem": "cha_khi_con_chay", "giay": 4, "kieu": "native"},
                      {"diem": "truoc_ghi", "kieu": "cho", "giay": 90})

        def qs_con():
            x = lan_chay(S)[-1] if lan_chay(S) else None
            con = H.doc_json(x["dir"] / "con.json") if x and (x["dir"] / "con.json").is_file() else None
            return VA.trang_thai_con(con["pid"], con.get("thoi_diem_tao")) if con else {}
        tt = nho("cha1_con", qs_con)
        kiem("cha_crash_khi_con_chay_cay_duoc_thu_hoi", m.get("exit") not in (0, 1, 2, 10, 30, 40, None)
             and (m.get("cay") or {}).get("active") == 0 and tt.get("con_song") is False,
             {"exit": m.get("exit_hex"), "cay": m.get("cay"), "tt": tt})
        m = lenh("cha1_lan_sau", b(S, "tracking"), S)
        kt = lan_chay(S)[-1]["ket_thuc"] if lan_chay(S) else {}
        kiem("cha_gian_doan_lan_sau_dung_40", m.get("exit") == 40 and kt.get("phan_loai") == "cha_gian_doan"
             and (kt.get("con_trang_thai") or {}).get("con_song") is False, kt)
        kiem_lenh("cha_dung_giu_chan_lan_goi_sau", lenh("cha1_giu_chan", b(S, "tracking"), S), 40, None, "DUNG HOI")
    S2 = S_("cha_2")
    m = tiem_lenh(S2, "cha2_sau_tao_con", "tien-kiem", {"diem": "cha_sau_tao_con", "kieu": "native"})
    kiem("cha_sau_tao_con_crash_cay_thu_hoi", m.get("exit") not in (0, 1, 2, 10, 30, 40, None) and (m.get("cay") or {}).get("active") == 0,
         {"exit": m.get("exit_hex"), "cay": m.get("cay")})
    m = lenh("cha2_lan_sau", b(S2, "tien-kiem"), S2)
    kiem_lenh("cha_sau_tao_con_lan_sau_40_thieu_con_json", m, 40, None, "thieu con.json")


def ca_job_pid_log():
    jp = (("job_loi", {"diem": "job_loi"}, "Job Object"), ("resume_loi", {"diem": "resume_loi"}, "NtResumeProcess"),
          ("loi_python", {"diem": "khoi_dong", "kieu": "python"}, "loi_python"),
          ("thieu_log", {"diem": "bo_faulthandler"}, "thieu log"))
    yeu_cau(*[f"jp_{t[0]}_dung_40" for t in jp], "jp_pid_khong_xac_dinh_dung_40", "jp_timeout_het_gio_40_khong_cap_lai_ngan_sach")
    for ten, diem, can in jp:
        S = S_(f"jp_{ten}")
        m = tiem_lenh(S, f"jp_{ten}", "tien-kiem", diem)
        kiem_lenh(f"jp_{ten}_dung_40", m, 40, None, can)
    S = S_("jp_pid")
    tiem_lenh(S, "jp_pid_cha_sau_con", "tien-kiem", {"diem": "cha_sau_con", "kieu": "native"})
    m = tiem_lenh(S, "jp_pid_khong_xac_dinh", "tien-kiem", {"diem": "pid_khong_xac_dinh"})
    kiem_lenh("jp_pid_khong_xac_dinh_dung_40", m, 40, None, "KHONG xac nhan duoc con da ket thuc")
    S = S_("jp_timeout")
    han = time.time() + 75
    m = lenh("jp_timeout", b(S, "tien-kiem", "--han-chot", f"{han:.3f}", *tl({"diem": "khoi_dong", "kieu": "cho", "giay": 300})),
             S, can=80)
    kt = m.get("ket_thuc") or {}
    con = kt.get("con") or {}
    kiem("jp_timeout_het_gio_40_khong_cap_lai_ngan_sach", m.get("exit") == 40 and kt.get("het_gio") is True
         and (con.get("timeout_giay") or 1e9) <= 75 - 15 + 1 and (m.get("cay") or {}).get("active") == 0,
         {"exit": m.get("exit"), "het_gio": kt.get("het_gio"), "timeout": con.get("timeout_giay"), "giay": m.get("giay")})


def ca_nguong():
    yeu_cau("ngdv_mo_phien", "ng_don_vi_crash_3_lan_dung", "ng_don_vi_giu_chan", "ngph_mo_phien", "ng_phien_crash_4_dung",
            "ng_phien_giu_chan", "ngkk_mo_phien", "ng_kiem_khoa_crash_2_dung", "ng_kiem_khoa_giu_chan")
    S = S_("ng_dv")
    ok = mo(S, "ngdv")
    kiem("ngdv_mo_phien", ok)
    if ok:
        u1 = dv_trk(0)
        ex = [tiem_lenh(S, f"ngdv_{k}", "tracking", {"diem": "truoc_ghi", "don_vi": u1, "kieu": "native"}).get("exit") for k in range(3)]
        kiem("ng_don_vi_crash_3_lan_dung", ex == [30, 30, 40], ex)
        kiem_lenh("ng_don_vi_giu_chan", lenh("ngdv_giu", b(S, "tracking"), S), 40, None, "DUNG HOI")
    S = S_("ng_phien")
    ok = mo(S, "ngph")
    kiem("ngph_mo_phien", ok)
    if ok:
        ex = [tiem_lenh(S, f"ngph_{k}", "tracking", {"diem": "truoc_ghi", "don_vi": dv_trk(k), "kieu": "native"}).get("exit")
              for k in range(4)]
        kiem("ng_phien_crash_4_dung", ex == [30, 30, 30, 40], ex)
        kiem_lenh("ng_phien_giu_chan", lenh("ngph_giu", b(S, "tracking"), S), 40, None, "DUNG HOI")
    S = S_("ng_kk")
    ok = mo(S, "ngkk")
    kiem("ngkk_mo_phien", ok)
    if ok:
        ex = [tiem_lenh(S, f"ngkk_{k}", "kiem-khoa", {"diem": "truoc_ghi", "don_vi": "kiem-khoa", "kieu": "native"}).get("exit")
              for k in range(2)]
        kiem("ng_kiem_khoa_crash_2_dung", ex == [30, 40], ex)
        kiem_lenh("ng_kiem_khoa_giu_chan", lenh("ngkk_giu", b(S, "kiem-khoa"), S), 40, None, "DUNG HOI")


def fixture_con(P: Path) -> list[str]:
    """Doi so cho mot lan goi FIXTURE con (cung entrypoint, phien con duoi vung fixture): ngan sach nho, gioi han du cho
    chup pyc + import (lan goi nay dung o ra soat so, TRUOC chon vong)."""
    return ["--phien", str(P), "--ngan-sach-giay", "20", "--gioi-han-giay", f"{gioi_han_con():.0f}"]


def gioi_han_con() -> float:
    return min(TRAN_TOI_DA, max(60.0, 3 * _PYC_TRUOC["giay"] + 60.0))


def _ghi_neu_chua(p: Path, obj) -> None:
    if not p.is_file():
        H.ghi_json(p, obj, [T.V])


def ca_khoi_dong_ngat():
    """F8a: journal khoi dong (thu vien chuan, truoc chup pyc / import) + lan goi bi ngat TRUOC lan_goi cua vong -> lan goi
    sau va sau doi snapshot van DUNG HOI."""
    yeu_cau("kd_journal_lan_goi_nay_truoc_chup_pyc_va_import", "kd_ngat_khoi_dong_crash_native_cay_thu_hoi",
            "kd_journal_thieu_ket_thuc_khong_vong_khong_lan_goi", "kd_ra_soat_phat_hien_va_dung", "kd_lan_goi_sau_exit_40",
            "kd_doi_snapshot_van_chan_khong_tao_vong", "kd_so_chi_them_giu_su_co")
    bd = H.doc_json(T.PH / "khoi_dong" / f"{KHOI_DONG['id']}.bat_dau.json")
    lg = H.doc_json(T.V / "lan_goi" / f"{T.LAN:02d}.bat_dau.json")
    kiem("kd_journal_lan_goi_nay_truoc_chup_pyc_va_import", bd.get("ghi_luc_epoch", 1e18) <= _PYC_TRUOC["t0"]
         and bd.get("module_du_an_da_nap") == [] and lg.get("khoi_dong") == KHOI_DONG["id"],
         {"ghi_luc": bd.get("ghi_luc_epoch"), "pyc_t0": _PYC_TRUOC["t0"], "module": bd.get("module_du_an_da_nap")})
    d = T.V / "tham_do" / "ngat_khoi_dong"
    P = d / "phien_kd"
    m = lenh("kd_ngat_khoi_dong", ["--tham-do", "ngat_khoi_dong", str(d), "30"], can=40, script=THIS)
    kiem("kd_ngat_khoi_dong_crash_native_cay_thu_hoi", m.get("exit") not in (0, 1, 2, 10, 40, None) and cay_tu_thu_hoi(m),
         {"exit": m.get("exit_hex"), "cay": m.get("cay")})

    def qs():
        kd = P / "khoi_dong"
        fh = sorted(kd.glob("*.faulthandler.log"))
        return {"bat_dau": sorted(p.name for p in kd.glob("*.bat_dau.json")), "ket_thuc": sorted(p.name for p in kd.glob("*.ket_thuc.json")),
                "vong": sorted(n for n in os.listdir(P) if n.startswith("vong_")) if P.is_dir() else None,
                "fh": fh[0].read_bytes().decode("utf-8", "replace")[-600:] if fh else ""}
    q = nho("kd_sau_ngat", qs)
    kiem("kd_journal_thieu_ket_thuc_khong_vong_khong_lan_goi", len(q["bat_dau"]) == 1 and not q["ket_thuc"] and q["vong"] == []
         and "Windows fatal exception" in q["fh"], q)
    so = nho("kd_ra_soat", lambda: (lambda s: {"muc": s["muc"], "ly_do": ly_do_dung(s)})(ra_soat(P, None)))
    kiem("kd_ra_soat_phat_hien_va_dung", any(x["khoa"].startswith("khoi_dong/") and not x["an_toan_xac_dinh"] for x in so["muc"])
         and bool(so["ly_do"]), so)
    m = lenh("kd_lan_sau", fixture_con(P), can=gioi_han_con() + 15, script=THIS)
    kiem_lenh("kd_lan_goi_sau_exit_40", m, 40, None, "DUNG HOI")
    _ghi_neu_chua(P / "vong_01" / "vong.json", {"snapshot": {"sha256": "0" * 64}, "gia_lap_doi_snapshot": True})
    dz = nho("kd_dung_sha", lambda: H.sha256_file(P / "dung_fixture.json") if (P / "dung_fixture.json").is_file() else None)
    m = lenh("kd_sau_doi_snapshot", fixture_con(P), can=gioi_han_con() + 15, script=THIS)
    kiem("kd_doi_snapshot_van_chan_khong_tao_vong", m.get("exit") == 40 and "DUNG HOI" in chu(m) and dz is not None
         and H.sha256_file(P / "dung_fixture.json") == dz and not (P / "vong_02").exists()
         and not (P / "vong_01" / "lan_goi").exists(), {"exit": m.get("exit"), "dz": dz})
    s2 = _doc(P / "so_crash_ngoai_du_kien.json") or {"muc": []}
    kd = [x for x in s2["muc"] if x["khoa"].startswith("khoi_dong/")]
    kiem("kd_so_chi_them_giu_su_co", len(kd) == 1 and kd[0]["khoa"] == f"khoi_dong/{q['bat_dau'][0][:-len('.bat_dau.json')]}"
         and len(list((P / "khoi_dong").glob("*.ket_thuc.json"))) == 2, kd)


KT_NATIVE = {"phan_loai": "crash_native", "exit": 3221225477, "exit_hex": "0xc0000005", "het_gio": False, "thieu_log": [],
             "job": {"dat": True}}
KT_NGOAI = {"het_gio": {"phan_loai": "het_gio", "exit": None, "het_gio": True, "thieu_log": [], "job": {"dat": True}},
            "traceback": {"phan_loai": "loi_python", "exit": 1, "het_gio": False, "thieu_log": [], "job": {"dat": True}},
            "thieu_log": {"phan_loai": "hoan_tat", "exit": 0, "het_gio": False, "thieu_log": ["faulthandler.log"],
                          "job": {"dat": True}}}
KIND_NGOAI = {"het_gio": "het_gio", "traceback": "loi_python", "thieu_log": "thieu_log"}


def gia_su_co(ph: Path, vong: str, k: int, kt: dict | None, ex: int, tiem=(), dung_json: bool = False) -> tuple[Path, Path]:
    """Mot su co TONG HOP = 2 nguon bang chung (ket qua lenh fixture + ket_thuc lan chay runner cung stt), dung cau truc
    lenh() / mot_doan() ghi. Phien gia duoi <vong>/so_gia/ (ngoai pham vi ra soat cua so chinh). kt=None: lan chay thieu
    ket_thuc (cha runner gian doan); dung_json: van_hanh/dung.json tro lan chay 1 (nhu runner dung() ghi)."""
    V = ph / vong
    _ghi_neu_chua(V / "vong.json", {"snapshot": {"sha256": f"{int(vong[5:]):064d}"}, "gia_lap": True})
    S = V / "phien" / f"s{k:02d}"
    d = S / "van_hanh" / "lan_chay" / "0001_tracking"
    _ghi_neu_chua(d / "goi.json", {"stt": 1, "buoc": "tracking", "tiem_loi": list(tiem)})
    if kt is not None:
        _ghi_neu_chua(d / "ket_thuc.json", {"stt": 1, "buoc": "tracking", "don_vi": None, **kt})
    if dung_json:
        _ghi_neu_chua(S / "van_hanh" / "dung.json", {"ly_do": ["gia lap"], "stt": 1, "gia_lap": True})
    argv = [H.PYTHON, "-B", str(RUNNER), "--cong", "B", "--phien", str(S), "--buoc", "tracking", "--so-doan", "1",
            *(tl(*tiem) if tiem else [])]
    p = V / "lenh" / f"s{k:02d}.json"
    _ghi_neu_chua(p, {"ten": f"s{k:02d}", "argv": argv, "exit": ex, "exit_hex": hex(ex & 0xFFFFFFFF), "het_gio_fixture": False,
                      "gan_job_fixture": True, "trong_job_goc": True, "cay": {"ok": True, "active": 0, "tong": 2},
                      "phien": str(S), "lan_chay_stt": [1], "lan_chay_moi": 1,
                      "ket_thuc": {"stt": 1, "don_vi": None, **kt} if kt is not None else None})
    return p, S


def thu_ap(ph: Path, p: Path, S: Path) -> dict:
    try:
        ap_so_sau_lenh(ph, None, p, S)
        dung = False
    except DungFixture:
        dung = True
    so = _doc(ph / "so_crash_ngoai_du_kien.json") or {}
    dz = _doc(ph / "dung_fixture.json") if (ph / "dung_fixture.json").is_file() else None
    return {"dung_ngay": dung, "muc": so.get("muc", []), "tiem": so.get("tiem", []), "dung_fixture": dz}


def ca_so_crash():
    """F8b/F8c tren bang chung TONG HOP (va mot bang chung that bo tiem): loi ngoai du kien khong phai native -> dung NGAY
    sau lenh, lan goi + snapshot sau van chan; crash native dem theo su co duy nhat qua nhieu lan ra soat / vong."""
    ten_b = list(KT_NGOAI)
    yeu_cau(*[f"f8b_{t}_ngoai_du_kien_dung_ngay_sau_lenh" for t in ten_b],
            *[f"f8b_{t}_lan_goi_va_snapshot_sau_van_chan" for t in ten_b],
            "f8b_tiem_khop_loai_khong_dung", "f8b_tiem_lech_loai_la_ngoai_du_kien_dung",
            "f8b_bang_chung_that_co_tiem_la_tiem", "f8b_bang_chung_that_bo_tiem_dung_ngay",
            "f8c_mot_su_co_dem_mot", "f8c_hai_su_co_dem_hai_khong_dung", "f8c_bon_su_co_dem_bon_dung",
            "f8c_ra_soat_lap_lai_khong_dem_trung", "f8c_qua_snapshot_cong_don_khong_reset")
    G = T.V / "so_gia"
    for t in ten_b:
        ph = G / f"ngoai_{t}"
        p, S = gia_su_co(ph, "vong_01", 0, KT_NGOAI[t], 40)
        r = nho(f"f8b_{t}", lambda: thu_ap(ph, p, S))
        kiem(f"f8b_{t}_ngoai_du_kien_dung_ngay_sau_lenh", r["dung_ngay"] and r["dung_fixture"] is not None
             and any(KIND_NGOAI[t] in x["loai"] and x.get("dung_ngay") for x in r["muc"]), r)
        _ghi_neu_chua(ph / "vong_02" / "vong.json", {"snapshot": {"sha256": "2" * 64}, "gia_lap_doi_snapshot": True})
        dz = nho(f"f8b_{t}_dung_sha", lambda: H.sha256_file(ph / "dung_fixture.json") if (ph / "dung_fixture.json").is_file() else None)
        m = lenh(f"f8b_{t}_lan_sau", fixture_con(ph), can=gioi_han_con() + 15, script=THIS)
        kiem(f"f8b_{t}_lan_goi_va_snapshot_sau_van_chan", m.get("exit") == 40 and "DUNG HOI" in chu(m) and dz is not None
             and H.sha256_file(ph / "dung_fixture.json") == dz and not (ph / "vong_03").exists(), {"exit": m.get("exit"), "dz": dz})
    u = "tracking:r:v0"
    ph = G / "tiem_khop"
    p, S = gia_su_co(ph, "vong_01", 0, {**KT_NATIVE, "don_vi": u}, 30, tiem=({"diem": "truoc_ghi", "don_vi": u, "kieu": "native"},))
    r = nho("f8b_tiem_khop", lambda: thu_ap(ph, p, S))
    kiem("f8b_tiem_khop_loai_khong_dung", not r["dung_ngay"] and not r["muc"] and len(r["tiem"]) == 2, r)
    ph = G / "tiem_lech"
    p, S = gia_su_co(ph, "vong_01", 0, KT_NGOAI["het_gio"], 40, tiem=({"diem": "khoi_dong", "kieu": "python"},))
    r = nho("f8b_tiem_lech", lambda: thu_ap(ph, p, S))
    kiem("f8b_tiem_lech_loai_la_ngoai_du_kien_dung", r["dung_ngay"] and any("het_gio" in x["loai"] for x in r["muc"]), r)
    # Bang chung THAT: lan chay loi_python cua ca job_pid_log (co tiem) -> 'tiem'; ban sao bo tiem_loi -> ngoai du kien, dung
    goc = [x for x in lan_chay(S_("jp_loi_python")) if (x["ket_thuc"] or {}).get("phan_loai") == "loi_python"]
    kiem("f8b_bang_chung_that_co_tiem_la_tiem", bool(goc) and bat_thuong_lan_chay(goc[-1]) == (["loi_python"], []),
         [bat_thuong_lan_chay(x) for x in goc])
    if goc and (T.V / "lenh" / "jp_loi_python.json").is_file():
        ph = G / "that_bo_tiem"
        S = ph / "vong_01" / "phien" / "s00"
        dd = S / "van_hanh" / "lan_chay" / goc[-1]["ten"]
        for f in sorted(goc[-1]["dir"].iterdir()):
            if f.is_file() and not (dd / f.name).is_file():
                b0 = f.read_bytes()
                if f.name == "goi.json":
                    b0 = (json.dumps({**json.loads(b0.decode("utf-8")), "tiem_loi": []}, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
                H.ghi_bytes(dd / f.name, b0, [T.V])
        mg = H.doc_json(T.V / "lenh" / "jp_loi_python.json")
        av = mg["argv"][:mg["argv"].index("--tiem-loi")] + mg["argv"][mg["argv"].index("--tiem-loi") + 2:] \
            if "--tiem-loi" in mg["argv"] else mg["argv"]
        p = ph / "vong_01" / "lenh" / "that.json"
        _ghi_neu_chua(p, {**mg, "argv": av, "phien": str(S), "lan_chay_stt": [goc[-1]["stt"]]})
        r = nho("f8b_that_bo_tiem", lambda: thu_ap(ph, p, S))
        kiem("f8b_bang_chung_that_bo_tiem_dung_ngay", r["dung_ngay"] and any("loi_python" in x["loai"] for x in r["muc"]), r)
    else:
        kiem("f8b_bang_chung_that_bo_tiem_dung_ngay", False, "thieu lan chay loi_python / ket qua lenh jp_loi_python cua ca job_pid_log")

    def dem(n: int) -> dict:
        ph = G / f"native_{n}"
        for k in range(n):
            gia_su_co(ph, "vong_01", k, {**KT_NATIVE, "don_vi": f"tracking:r:v{k}"}, 30)
        lan = [cap_nhat_so(ph, None) for _ in range(3)]
        so = _doc(ph / "so_crash_ngoai_du_kien.json")
        return {"bang_chung": len(so["muc"]), "su_co": su_co_native(so), "dung": lan}
    d = nho("f8c_dem", lambda: {str(n): dem(n) for n in (1, 2, 4)})
    kiem("f8c_mot_su_co_dem_mot", d["1"]["bang_chung"] == 2 and len(d["1"]["su_co"]) == 1 and d["1"]["dung"] == [False] * 3, d["1"])
    kiem("f8c_hai_su_co_dem_hai_khong_dung", d["2"]["bang_chung"] == 4 and len(d["2"]["su_co"]) == 2
         and d["2"]["dung"] == [False] * 3, d["2"])
    dz4 = _doc(G / "native_4" / "dung_fixture.json") or {}
    kiem("f8c_bon_su_co_dem_bon_dung", d["4"]["bang_chung"] == 8 and len(d["4"]["su_co"]) == 4 and d["4"]["dung"] == [True] * 3
         and any("4 crash native" in s for s in dz4.get("ly_do", [])), {"dem": d["4"], "dung": dz4})
    kiem("f8c_ra_soat_lap_lai_khong_dem_trung", all(len(d[n]["su_co"]) == int(n) for n in d), {n: d[n]["su_co"] for n in d})

    def qua_snapshot() -> dict:
        ph = G / "native_2"
        for k in (2, 3):                               # vong_02 (snapshot khac) them 2 su co moi
            gia_su_co(ph, "vong_02", k, {**KT_NATIVE, "don_vi": f"tracking:r:v{k}"}, 30)
        lan = [cap_nhat_so(ph, None) for _ in range(2)]
        so = _doc(ph / "so_crash_ngoai_du_kien.json")
        return {"bang_chung": len(so["muc"]), "su_co": su_co_native(so), "dung": lan,
                "vong": sorted({x.get("vong") for x in so["muc"]})}
    q = nho("f8c_qua_snapshot", qua_snapshot)
    kiem("f8c_qua_snapshot_cong_don_khong_reset", q["bang_chung"] == 8 and len(q["su_co"]) == 4 and q["dung"] == [True, True]
         and q["vong"] == ["vong_01", "vong_02"], q)


U_GIA = "tracking:r:v0"
TIEM_GIA_NATIVE = {"diem": "truoc_ghi", "don_vi": U_GIA, "kieu": "native"}
EX_NATIVE = 3221225477
QD_DUNG = {"exit_cha": 40, "ly_do_dung": ["gia lap"]}
#: B0-F8b-1 -- ten: (ket_thuc lan chay | None = thieu, exit lenh, tiem trong argv + goi, co dung.json, ky vong dung ngay)
F8B1 = {"cha_native_khong_tiem": (None, EX_NATIVE, (), False, True),
        "lan_chay_mo_lenh_exit_0": (None, 0, (), False, True),
        "dung40_khong_dung_json": ({"phan_loai": "hoan_tat", "exit": 0, "het_gio": False, "thieu_log": [], "job": {"dat": True}},
                                   40, (), False, True),
        "dung40_crash_khong_don_vi": ({**KT_NATIVE, "don_vi": None, "quyet_dinh": QD_DUNG}, 40, (), True, True),
        "dung40_nguong_crash_khong_tiem": ({**KT_NATIVE, "don_vi": U_GIA, "quyet_dinh": QD_DUNG}, 40, (), True, True),
        "doi_chung_dung40_nguong_co_tiem": ({**KT_NATIVE, "don_vi": U_GIA, "quyet_dinh": QD_DUNG}, 40, (TIEM_GIA_NATIVE,),
                                            True, False),
        "doi_chung_cha_native_co_tiem": (None, EX_NATIVE, ({"diem": "cha_khi_con_chay", "kieu": "native"},), False, False)}
#: B0-F8b-2 -- so JSON hop le nhung sai cau truc
SO_HONG = {"phan_tu_null": {"muc": [None], "tiem": []}, "muc_khong_phai_danh_sach": {"muc": {}, "tiem": []},
           "tiem_phan_tu_chuoi": {"muc": [], "tiem": ["x"]},
           "muc_thieu_bool": {"muc": [{"khoa": "k", "loai": "l", "su_co": "s", "native": "co"}], "tiem": []}}


def ca_so_crash_bo_sung():
    """B0-F8b-1: cha runner crash / lan chay thieu ket_thuc / exit 40 khong duoc tiem giai thich -> dung NGAY, khong dem vao
    nguong native, giu chan qua snapshot (ca lan goi fixture that). B0-F8b-2: so sai cau truc o khoi dong va ngay sau lenh ->
    DungFixture / exit 40, giu nguyen byte so, giu chan lan goi sau."""
    neg = [t for t, v in F8B1.items() if v[4]]
    pos = [t for t, v in F8B1.items() if not v[4]]
    yeu_cau(*[f"f8b1_{t}_dung_ngay_khong_dem_native" for t in neg], *[f"f8b1_{t}_giu_chan_sau_snapshot" for t in neg],
            *[f"f8b1_{t}_khong_dung" for t in pos], "f8b1_cha_native_lan_goi_that_sau_snapshot_van_40",
            "f8b2_dung_fixture_khong_bi_except_exception_nuot",
            *[f"f8b2_sau_lenh_{t}_dung_giu_nguyen_so" for t in SO_HONG],
            "f8b2_khoi_dong_phan_tu_null_exit_40_giu_nguyen_so", "f8b2_khoi_dong_phan_tu_null_lan_sau_van_40",
            "f8b2_khoi_dong_muc_khong_phai_danh_sach_exit_40", "f8b2_sau_lenh_lan_goi_that_sau_van_40")
    G = T.V / "so_gia"

    def thu_f8b1(ten: str) -> dict:
        kt, ex, tiem, dj, _ = F8B1[ten]
        ph = G / f"f8b1_{ten}"
        p, S = gia_su_co(ph, "vong_01", 0, kt, ex, tiem=tiem, dung_json=dj)
        r = thu_ap(ph, p, S)
        so = _doc(ph / "so_crash_ngoai_du_kien.json") or {"muc": []}
        dzp = ph / "dung_fixture.json"
        dz = H.sha256_file(dzp) if dzp.is_file() else None
        _ghi_neu_chua(ph / "vong_02" / "vong.json", {"snapshot": {"sha256": "2" * 64}, "gia_lap_doi_snapshot": True})
        giu = cap_nhat_so(ph, None)
        return {**r, "su_co_native": su_co_native(so), "dz": dz, "giu_sau_snapshot": giu,
                "dz_sau": H.sha256_file(dzp) if dzp.is_file() else None}
    for t in F8B1:
        r = nho(f"f8b1_{t}", lambda t=t: thu_f8b1(t))
        if F8B1[t][4]:
            kiem(f"f8b1_{t}_dung_ngay_khong_dem_native", r["dung_ngay"] and r["dung_fixture"] is not None and bool(r["muc"])
                 and all(x.get("dung_ngay") is True for x in r["muc"]) and r["su_co_native"] == [], r)
            kiem(f"f8b1_{t}_giu_chan_sau_snapshot", r["giu_sau_snapshot"] is True and r["dz"] is not None and r["dz_sau"] == r["dz"],
                 {k: r[k] for k in ("giu_sau_snapshot", "dz", "dz_sau")})
        else:
            kiem(f"f8b1_{t}_khong_dung", not r["dung_ngay"] and not r["muc"] and bool(r["tiem"]) and r["dz"] is None
                 and r["giu_sau_snapshot"] is False, r)
    ph = G / "f8b1_cha_native_khong_tiem"
    dz = nho("f8b1_cha_native_dz", lambda: H.sha256_file(ph / "dung_fixture.json") if (ph / "dung_fixture.json").is_file() else None)
    m = lenh("f8b1_cha_native_lan_sau", fixture_con(ph), can=gioi_han_con() + 15, script=THIS)
    kiem("f8b1_cha_native_lan_goi_that_sau_snapshot_van_40", m.get("exit") == 40 and "DUNG HOI" in chu(m) and dz is not None
         and H.sha256_file(ph / "dung_fixture.json") == dz and not (ph / "vong_03").exists(), {"exit": m.get("exit"), "dz": dz})
    kiem("f8b2_dung_fixture_khong_bi_except_exception_nuot", issubclass(DungFixture, BaseException)
         and not issubclass(DungFixture, Exception))

    def thu_f8b2_sau(ten: str) -> dict:
        ph = G / f"f8b2_sau_{ten}"
        p, S = gia_su_co(ph, "vong_01", 0, {**KT_NATIVE, "don_vi": U_GIA}, 30)     # bang chung tu no KHONG buoc dung
        sp = ph / "so_crash_ngoai_du_kien.json"
        _ghi_neu_chua(sp, SO_HONG[ten])
        h0 = H.sha256_file(sp)
        r = thu_ap(ph, p, S)
        return {"dung_ngay": r["dung_ngay"], "so_truoc": h0, "so_sau": H.sha256_file(sp), "dung_fixture": r["dung_fixture"]}
    for t in SO_HONG:
        r = nho(f"f8b2_sau_{t}", lambda t=t: thu_f8b2_sau(t))
        kiem(f"f8b2_sau_lenh_{t}_dung_giu_nguyen_so", r["dung_ngay"] and r["so_truoc"] == r["so_sau"]
             and any("khong dung duoc" in s for s in (r["dung_fixture"] or {}).get("ly_do", [])), r)
    ph = G / "f8b2_sau_phan_tu_null"
    h = nho("f8b2_sau_lan_sau_hash", lambda: [H.sha256_file(ph / n) if (ph / n).is_file() else None
                                              for n in ("so_crash_ngoai_du_kien.json", "dung_fixture.json")])
    m = lenh("f8b2_sau_lenh_lan_sau", fixture_con(ph), can=gioi_han_con() + 15, script=THIS)
    kiem("f8b2_sau_lenh_lan_goi_that_sau_van_40", m.get("exit") == 40 and "DUNG HOI" in chu(m) and None not in h
         and [H.sha256_file(ph / n) for n in ("so_crash_ngoai_du_kien.json", "dung_fixture.json")] == h
         and not (ph / "vong_02").exists(), {"exit": m.get("exit"), "hash": h})

    def chuan_bi_kd(ten: str) -> str:
        ph = G / f"f8b2_kd_{ten}"
        _ghi_neu_chua(ph / "so_crash_ngoai_du_kien.json", SO_HONG[ten])
        return H.sha256_file(ph / "so_crash_ngoai_du_kien.json")
    for t, lan_sau in (("phan_tu_null", True), ("muc_khong_phai_danh_sach", False)):
        ph = G / f"f8b2_kd_{t}"
        h0 = nho(f"f8b2_kd_{t}_so", lambda t=t: chuan_bi_kd(t))
        m = lenh(f"f8b2_kd_{t}", fixture_con(ph), can=gioi_han_con() + 15, script=THIS)
        dzp = ph / "dung_fixture.json"
        ok = m.get("exit") == 40 and "DUNG HOI" in chu(m) and "sai cau truc" in chu(m) \
            and H.sha256_file(ph / "so_crash_ngoai_du_kien.json") == h0 and dzp.is_file() \
            and not any(n.startswith("vong_") for n in os.listdir(ph))
        kiem(f"f8b2_khoi_dong_{t}_exit_40" + ("_giu_nguyen_so" if lan_sau else ""), ok,
             {"exit": m.get("exit"), "stderr": m.get("stderr", "")[-600:]})
        if lan_sau:
            dz = nho(f"f8b2_kd_{t}_dz", lambda: H.sha256_file(dzp) if dzp.is_file() else None)
            m = lenh(f"f8b2_kd_{t}_lan_sau", fixture_con(ph), can=gioi_han_con() + 15, script=THIS)
            kiem(f"f8b2_khoi_dong_{t}_lan_sau_van_40", m.get("exit") == 40 and "DUNG HOI" in chu(m) and dz is not None
                 and H.sha256_file(dzp) == dz and H.sha256_file(ph / "so_crash_ngoai_du_kien.json") == h0
                 and not any(n.startswith("vong_") for n in os.listdir(ph)), {"exit": m.get("exit"), "dz": dz})


def ca_ket_thuc():
    yeu_cau("PLAN_A_ma_cu_nguyen_byte_khong_tao_thuc_thi", "khong_dang_ghi_ngoai_ban_sao_luu", "moi_lenh_co_exit_khong_bi_gian_doan",
            "moi_lenh_trong_job_fixture_va_thu_hoi_ca_cay", "moi_lenh_con_trong_job_goc_tu_luc_tao",
            "moi_lenh_python_tuyet_doi_co_B", "chau_tham_do_python_tuyet_doi_co_B", "moi_con_khong_vuot_han_chot_va_co_B",
            "khong_crash_ngoai_du_kien_trong_vong")
    can_tg("ca ket thuc (bam / liet ke trong tien trinh)", 60)
    tl0 = H.doc_json(T.V / "thiet_lap" / "that_truoc.json")
    sau = trang_thai_that()
    kiem("PLAN_A_ma_cu_nguyen_byte_khong_tao_thuc_thi", tl0 == sau, {k: tl0[k] == sau[k] for k in sau})
    tep = [p for p in H.liet_ke_file(T.V) if p.is_file()]
    do = [H.rel(p) for p in tep if p.name.endswith(".dang_ghi") and "/artifact_do/" not in H.rel(p)]
    kiem("khong_dang_ghi_ngoai_ban_sao_luu", not do, do[:5])
    ls = [H.doc_json(p) for p in sorted((T.V / "lenh").glob("*.json")) if not p.name.endswith(".bat_dau.json")]
    kiem("moi_lenh_co_exit_khong_bi_gian_doan", all(x.get("exit") is not None and not x.get("bi_gian_doan") for x in ls),
         [x["ten"] for x in ls if x.get("exit") is None])
    kiem("moi_lenh_trong_job_fixture_va_thu_hoi_ca_cay",
         all(x.get("gan_job_fixture") and (x.get("cay") or {}).get("active") == 0 for x in ls if not x.get("bi_gian_doan")),
         [x["ten"] for x in ls if not x.get("gan_job_fixture") or (x.get("cay") or {}).get("active")])
    kiem("moi_lenh_con_trong_job_goc_tu_luc_tao", all(x.get("trong_job_goc") is True for x in ls if not x.get("bi_gian_doan")),
         [x["ten"] for x in ls if x.get("trong_job_goc") is not True])
    kiem("moi_lenh_python_tuyet_doi_co_B", all(x["argv"][:3] in ([H.PYTHON, "-B", str(RUNNER)], [H.PYTHON, "-B", str(THIS)])
                                               for x in ls if x.get("argv")))
    chau = [H.doc_json(p) for p in sorted((T.V / "tham_do").glob("*/chau.json"))] if (T.V / "tham_do").is_dir() else []
    kiem("chau_tham_do_python_tuyet_doi_co_B", bool(chau) and all(c["argv"][:3] == [H.PYTHON, "-B", str(THIS)] for c in chau),
         [c.get("argv") for c in chau])
    sai = []
    for d in sorted((T.V / "phien").iterdir()) if (T.V / "phien").is_dir() else []:
        for x in lan_chay(d) if not H.la_reparse(d) else []:
            g = x["goi"] or {}
            if g.get("han_chot_cha") is not None and g.get("han_cung", 0) > g["han_chot_cha"] - 15 + 1e-6:
                sai.append(x["ten"])
            if g.get("argv_con", [None, None])[:2] != [H.PYTHON, "-B"]:
                sai.append(f"{d.name}/{x['ten']} argv_con")
    kiem("moi_con_khong_vuot_han_chot_va_co_B", not sai, sai[:8])
    so = ra_soat(T.PH, (T.V.name, T.LAN))
    ngoai = [x for x in so["muc"] if x.get("vong") == T.V.name]
    kiem("khong_crash_ngoai_du_kien_trong_vong", not ngoai, ngoai[:5])


CA = [("thiet_lap", ca_thiet_lap), ("bao_ve_lan_goi", ca_bao_ve), ("thanh_cong", ca_thanh_cong),
      ("kiem_lai_buoc_xong", ca_kiem_lai_buoc_xong), ("am_dau_vao", ca_am_dau_vao), ("phuc_hoi_journal", ca_phuc_hoi_journal),
      ("ma_b_doi", ca_ma_b_doi), ("crash_ghi", ca_crash_ghi), ("cha_gian_doan", ca_cha_gian_doan),
      ("job_pid_log", ca_job_pid_log), ("nguong", ca_nguong), ("khoi_dong_ngat", ca_khoi_dong_ngat),
      ("so_crash", ca_so_crash), ("so_crash_bo_sung", ca_so_crash_bo_sung), ("ket_thuc", ca_ket_thuc)]


# ---------------------------------------------------------------------------
# So crash ngoai du kien (F8) -- xuyen vong
# ---------------------------------------------------------------------------
class DungFixture(BaseException):
    """F8b: so crash ngoai du kien yeu cau DUNG HOI -> dung NGAY sau lenh vua xong, truoc lenh ke tiep (BaseException: khong
    bi `except Exception` cua ca nuot; chay() bat rieng -> exit 40)."""


DIEM_CHA_NATIVE = ("cha_sau_tao_con", "cha_khi_con_chay", "cha_sau_con")
THAM_DO_EXIT = {"ngat_giua_popen_gan": "native", "loi_python_giua_popen_gan": 1, "tran_cung_trong_tien_trinh": 1,
                "thieu_du_phong": 10, "ngat_khoi_dong": "native"}
BAT_AN_TOAN = ("pid_khong_xac_dinh", "job_resume", "thieu_log")


def _doc(p: Path):
    return KD.doc_json_dv(p, p.name)[0]


def tiem_argv(argv: list) -> list[dict]:
    if "--tiem-loi" not in argv:
        return []
    try:
        t = json.loads(argv[argv.index("--tiem-loi") + 1])
    except (ValueError, IndexError, TypeError):
        return []
    return [x for x in t if isinstance(x, dict)] if isinstance(t, list) else []


def giai_thich_tiem(tiem: list[dict]) -> set[str]:
    """Bat thuong ma danh sach tiem GIAI THICH duoc theo LOAI (loi tiem DA XAC NHAN), khong chi 'co --tiem-loi'."""
    g = set()
    for t in tiem:
        d, k = t.get("diem"), t.get("kieu")
        if d in DIEM_CHA_NATIVE:
            g.add("cha_gian_doan")
            if d == "cha_sau_tao_con":
                g.add("thieu_con_json")
        elif d in ("job_loi", "resume_loi"):
            g.add("job_resume")
        elif d == "bo_faulthandler":
            g.add("thieu_log")
        elif d == "pid_khong_xac_dinh":
            g.add("pid_khong_xac_dinh")
        elif k == "python":
            g.add("loi_python")
        elif k == "cho":
            g.add("het_gio")
        else:
            g.add(f"crash_native@{t.get('don_vi')}")
    return g


def bat_thuong_lan_chay(x: dict) -> tuple[list[str], list[str]]:
    """(bat thuong cua lan chay runner, phan KHONG duoc tiem cua chinh lan chay do giai thich)."""
    kt, g = x.get("ket_thuc") or {}, x.get("goi") or {}
    gt = giai_thich_tiem([t for t in g.get("tiem_loi") or [] if isinstance(t, dict)] if isinstance(g.get("tiem_loi"), list) else [])
    lo, tt = kt.get("phan_loai"), kt.get("con_trang_thai") or {}
    bat = [lo] if lo in LOAI_CRASH_CON else []
    if kt.get("thieu_log"):
        bat.append("thieu_log")
    if not (kt.get("job") or {}).get("dat", True):
        bat.append("job_resume")
    if lo == "cha_gian_doan" and tt.get("con_song") is not False:
        bat.append("pid_khong_xac_dinh")

    def giai(b: str) -> bool:
        if "job_resume" in gt:                         # con bi cha giet khi con tam dung / khong tao con
            return True
        if b == "crash_native":
            return f"crash_native@{kt.get('don_vi')}" in gt or "crash_native@None" in gt
        if b == "pid_khong_xac_dinh":
            return tt.get("tiem_loi_fixture") == "pid_khong_xac_dinh" or ("thieu_con_json" in gt and kt.get("con") is None)
        return b in gt
    return bat, [b for b in bat if not giai(b)]


def giai_thich_dung_40(m: dict) -> tuple[bool, str]:
    """B0-F8b-1: exit 40 = tien trinh con YEU CAU DUNG HOI -- chi binh thuong khi bang chung dung khop loi tiem DA XAC NHAN.
    Fixture con: dung_fixture.json cua phien con. Runner B: van_hanh/dung.json tro lan chay DA ket_thuc, quyet dinh exit 40,
    moi bat thuong cua lan chay do duoc tiem cua chinh no giai thich (dung vi khong xac dinh don vi / nguong do crash KHONG
    tiem -> khong giai thich). Khong khop -> ngoai du kien, dung ngay."""
    argv = [str(a) for a in m.get("argv") or []]
    ph = argv[argv.index("--phien") + 1] if "--phien" in argv[:-1] else m.get("phien")
    if not ph:
        return False, "exit 40 nhung khong xac dinh duoc phien"
    P = Path(ph)
    if argv[2:3] == [str(THIS)]:
        ok = (P / "dung_fixture.json").is_file() and not H.la_reparse(P / "dung_fixture.json")
        return ok, f"fixture con exit 40: dung_fixture.json {'co' if ok else 'THIEU'}"
    p = P / "van_hanh" / "dung.json"
    dz = _doc(p) if p.is_file() and not H.la_reparse(p) else None
    stt = dz.get("stt") if isinstance(dz, dict) else None
    if not isinstance(stt, int) or isinstance(stt, bool):
        return False, "runner exit 40 nhung van_hanh/dung.json thieu / hong / khong tro lan chay"
    try:
        x = next((y for y in lan_chay(P) if y["stt"] == stt), None)
    except (ValueError, OSError) as e:
        return False, f"runner exit 40, khong doc duoc lan chay: {e}"
    kt = (x or {}).get("ket_thuc")
    if not isinstance(kt, dict) or not isinstance(x.get("goi"), (dict, type(None))) \
            or not isinstance(kt.get("quyet_dinh"), dict) or kt["quyet_dinh"].get("exit_cha") != VA.EXIT_DUNG:
        return False, f"dung.json -> lan chay {stt}: khong co ket_thuc hop le voi quyet dinh exit {VA.EXIT_DUNG}"
    bat, ko = bat_thuong_lan_chay(x)
    return bool(bat) and not ko, f"dung.json -> lan chay {stt}: bat thuong {bat}, khong duoc tiem giai thich {ko}"


def bat_thuong_lenh(m: dict) -> dict | None:
    """Lenh con cua fixture: None = binh thuong; khac -> {xac_nhan (loi tiem da xac nhan), an_toan, native, dung_ngay, ly_do}.
    B0-F8b-1: CHI exit 30 (runner bao con crash native, cay da thu hoi) la crash con duoc dem nguong; tien trinh truc tiep
    (cha runner / fixture con / tham do) tu crash native = cha gian doan -> dung ngay neu khong duoc tiem giai thich."""
    argv, ex, cay = m.get("argv") or [], m.get("exit"), m.get("cay") or {}
    an_toan = not (m.get("bi_gian_doan") or m.get("het_gio_fixture") or ex is None or cay.get("active") != 0)
    native = isinstance(ex, int) and ex not in (0, 1, 2, 10, 40)          # gom 30 (con cua runner crash native)
    cha_native = native and ex != 30
    dung_ngay = not (an_toan and ex == 30)
    if an_toan and ex in (0, 2, 10):
        return None
    if "--tham-do" in argv:
        i = argv.index("--tham-do")
        ky = THAM_DO_EXIT.get(argv[i + 1] if i + 1 < len(argv) else None)
        xac = an_toan and ((ky == "native" and cha_native) or (ky != "native" and ex == ky))
        ly = f"tham do {argv[i + 1:i + 2]} exit {ex} (ky vong {ky})"
    elif ex == 40 and an_toan:
        ok, ly = giai_thich_dung_40(m)
        if ok:
            return None
        xac = False
    else:
        tiem = tiem_argv(argv)
        gt = giai_thich_tiem(tiem)
        xac = an_toan and ((ex == 30 and any(g.startswith("crash_native@") for g in gt))
                           or (cha_native and "cha_gian_doan" in gt))
        ly = f"exit {ex} (tiem {[t.get('diem') for t in tiem]})"
    return {"xac_nhan": xac, "an_toan": an_toan, "native": native, "dung_ngay": dung_ngay, "ly_do": ly}


def _dung_ngay(x: dict) -> bool:
    if x.get("ke_thua") is not None:                   # Q9MG/QUVZ: chi muc ke thua DA xac minh trong lan goi nay duoc ngoai le
        return x.get("khoa") not in KE_THUA.get("khoa_hop_le", ()) or x.get("dung_ngay") is not False
    return x.get("dung_ngay", not x.get("an_toan_xac_dinh") or not x.get("native"))


def su_co_native(so: dict) -> list[str]:
    """F8c: crash native ngoai du kien dem theo SU CO duy nhat (nhieu nguon bang chung cung su_co = 1)."""
    return sorted({x.get("su_co") or x["khoa"] for x in so["muc"] if x.get("native") and not _dung_ngay(x)})


def _xet_lenh(them, v: str | None, f: Path) -> None:
    m = _doc(f)
    if not isinstance(m, dict) or not isinstance(m.get("argv") or [], list) or not isinstance(m.get("cay") or {}, dict) \
            or not isinstance(m.get("ket_thuc") or {}, dict):
        them("muc", f"lenh:{H.rel(f)}", su_co=f"lenh:{H.rel(f)}", vong=v, loai="ket_qua_lenh_hong", an_toan_xac_dinh=False,
             native=False, dung_ngay=True)
        return
    r = bat_thuong_lenh(m)
    if r is None:
        return
    stt = [s for s in m.get("lan_chay_stt") or [] if isinstance(s, int)] if isinstance(m.get("lan_chay_stt"), list) else []
    sc = f"lan_chay:{H.rel(Path(m['phien']))}/{stt[-1]:04d}" if isinstance(m.get("phien"), str) and stt else f"lenh:{H.rel(f)}"
    them("tiem" if r["xac_nhan"] else "muc", f"lenh:{H.rel(f)}", su_co=sc, vong=v, loai="lenh_con_bat_thuong",
         exit=m.get("exit"), exit_hex=m.get("exit_hex"), phan_loai=(m.get("ket_thuc") or {}).get("phan_loai"),
         an_toan_xac_dinh=r["an_toan"], native=r["native"], dung_ngay=r["dung_ngay"], ly_do=r["ly_do"])


def _xet_phien(them, v: str | None, S: Path) -> None:
    if H.la_reparse(S) or not S.is_dir():
        return
    try:
        ds = VA.doc_lan_chay(S)
    except (ValueError, OSError) as e:
        them("muc", f"lan_chay:{H.rel(S)}", su_co=f"lan_chay:{H.rel(S)}", vong=v, loai="lan_chay_doc_loi", loi=str(e),
             an_toan_xac_dinh=False, native=False, dung_ngay=True)
        return
    for x in ds:
        sc = f"lan_chay:{H.rel(S)}/{x['stt']:04d}"
        if not isinstance(x["ket_thuc"], (dict, type(None))) or not isinstance(x["goi"], (dict, type(None))):
            them("muc", f"lan_chay:{H.rel(x['dir'])}", su_co=sc, vong=v, loai="lan_chay_doc_loi", loi="goi/ket_thuc khong phai object",
                 an_toan_xac_dinh=False, native=False, dung_ngay=True)
            continue
        g = x["goi"] or {}
        if not x["ket_thuc"]:
            # B0-F8b-1: ra soat chi chay khi KHONG con runner song (sau khi lenh da thu hoi ca cay / khoi dong / tong ket) ->
            # lan chay thieu ket_thuc = cha runner gian doan. Khoa rieng (lan_chay_mo:) de ban ket_thuc ghi bo sung sau van
            # duoc xet doc lap; chi loi tiem cha (DIEM_CHA_NATIVE trong goi cua CHINH lan chay) giai thich duoc.
            tg = g.get("tiem_loi") if isinstance(g.get("tiem_loi"), list) else []
            if "cha_gian_doan" in giai_thich_tiem([t for t in tg if isinstance(t, dict)]):
                them("tiem", f"lan_chay_mo:{H.rel(x['dir'])}", su_co=sc, vong=v, loai="lan_chay_runner_dang_mo_cha_gian_doan",
                     tiem=tg)
            else:
                them("muc", f"lan_chay_mo:{H.rel(x['dir'])}", su_co=sc, vong=v, loai="lan_chay_runner_dang_mo_cha_gian_doan",
                     an_toan_xac_dinh=False, native=False, dung_ngay=True)
            continue
        bat, ko = bat_thuong_lan_chay(x)
        if not bat:
            continue
        if not ko:
            them("tiem", f"lan_chay:{H.rel(x['dir'])}", su_co=sc, vong=v, loai=f"lan_chay_runner_{'+'.join(bat)}",
                 tiem=g.get("tiem_loi"))
            continue
        kt = x["ket_thuc"]
        qd = kt.get("quyet_dinh") if isinstance(kt.get("quyet_dinh"), dict) else {}
        an_toan = not set(BAT_AN_TOAN) & set(ko)
        # B0-F8b-1: crash native KHONG tiem chi duoc dem nguong (thu lai) khi runner tu cho tiep tuc: co don vi va quyet
        # dinh khong phai exit 40 (runner da dung vi khong xac dinh don vi / cham nguong don vi / buoc / phien -> giu dung).
        cho_tiep = ko == ["crash_native"] and bool(kt.get("don_vi")) \
            and qd.get("exit_cha") != VA.EXIT_DUNG
        them("muc", f"lan_chay:{H.rel(x['dir'])}", su_co=sc, vong=v, loai=f"lan_chay_runner_{'+'.join(ko)}", bat_thuong=bat,
             an_toan_xac_dinh=an_toan, native=ko == ["crash_native"], dung_ngay=not cho_tiep,
             quyet_dinh_runner=qd.get("exit_cha"))


def loi_so(cu) -> list[str]:
    """B0-F8b-2: schema so crash ngoai du kien (dung cau truc ra_soat ghi) -- kiem TRUOC moi lan dung so cu."""
    if not isinstance(cu, dict):
        return [f"so phai la object, gap {type(cu).__name__}"]
    loi = [f"'{k}' thieu / khong phai danh sach" for k in ("muc", "tiem") if not isinstance(cu.get(k), list)]
    for k in ("muc", "tiem"):
        for i, x in enumerate(cu[k] if isinstance(cu.get(k), list) else []):
            if not isinstance(x, dict):
                loi.append(f"{k}[{i}] khong phai object ({type(x).__name__})")
                continue
            if not isinstance(x.get("khoa"), str) or not isinstance(x.get("loai"), str):
                loi.append(f"{k}[{i}] thieu khoa / loai kieu chuoi")
            if not all(x.get(t) is None or isinstance(x.get(t), str) for t in ("su_co", "vong")):
                loi.append(f"{k}[{i}] su_co / vong phai la chuoi hoac null")
            if k == "muc" and not all(isinstance(x.get(t), bool) for t in ("native", "dung_ngay", "an_toan_xac_dinh")):
                loi.append(f"muc[{i}] native / dung_ngay / an_toan_xac_dinh phai la bool")
    return loi[:8]


def dung_so_hong(ph: Path, loi: str) -> None:
    """B0-F8b-2: so khong dung duoc -> ghi dung_fixture.json MOT lan; KHONG sua / tao lai so (giu nguyen bang chung hong)."""
    p = ph / "dung_fixture.json"
    try:
        if not p.exists() and not H.la_reparse(p):
            H.ghi_json(p, {"ly_do": [f"so crash ngoai du kien khong dung duoc: {loi}"], "luc": time.strftime("%Y-%m-%dT%H:%M:%S"),
                           "huong_dan": "DUNG HOI nguoi dung; khong tu xoa/doi ten file nay, khong sua so de reset"}, [ph])
    except (H.LoiHopDong, OSError) as e:               # khong ghi duoc van dung (lan sau so van hong -> van 40)
        print(f"[DUNG HOI] khong ghi duoc dung_fixture.json: {e}", file=sys.stderr)


#: B0-F8b-2: loi khi doc / ra soat / ghi so -> trang thai an toan khong xac dinh -> DUNG HOI (khong de ca nuot, khong exit 1)
LOI_SO = (H.LoiHopDong, OSError, ValueError, TypeError, KeyError, AttributeError, IndexError)


def ra_soat(ph: Path, hien_tai: tuple[str, int] | None, chi: tuple[Path, Path | None] | None = None) -> dict:
    """chi=None: ra soat MOI vong + journal khoi dong (F8a: lan goi fixture thieu ket_thuc o BAT KY giai doan, ke ca truoc
    chup pyc / import; lan goi vong thieu ket_thuc). chi=(lenh, phien B): CHI bang chung cua lenh vua xong (F8b, ngay sau
    moi lenh). Lenh / lan chay bat thuong: loi tiem DA XAC NHAN (theo loai) -> 'tiem'; con lai -> 'muc' (ngoai du kien).
    Moi muc co su_co (F8c: lenh + lan chay cung stt = mot su co). Chi THEM vao so cu -> doi ma / vong moi khong reset."""
    so_p = ph / "so_crash_ngoai_du_kien.json"
    if H.la_reparse(so_p) or (so_p.exists() and not so_p.is_file()):
        raise H.LoiHopDong(f"so crash ngoai du kien {H.rel(so_p)} la reparse / khong phai file -- DUNG HOI")
    cu = _doc(so_p) if so_p.is_file() else None
    if so_p.is_file() and cu is None:
        raise H.LoiHopDong(f"so crash ngoai du kien {H.rel(so_p)} hong -- DUNG HOI (khong tu tao lai)")
    loi = loi_so(cu) if so_p.is_file() else []
    if loi:                                            # B0-F8b-2: JSON hop le nhung sai cau truc -> khong dung, khong sua so
        raise H.LoiHopDong(f"so crash ngoai du kien {H.rel(so_p)} sai cau truc: {loi} -- DUNG HOI (giu nguyen file)")
    so = {"muc": list((cu or {}).get("muc", [])), "tiem": list((cu or {}).get("tiem", []))}
    co = {x.get("khoa") for x in so["muc"] + so["tiem"]}
    luc = time.strftime("%Y-%m-%dT%H:%M:%S")

    def them(loai_so: str, khoa: str, **d):
        if khoa not in co:
            co.add(khoa)
            so[loai_so].append({"khoa": khoa, "ghi_nhan_luc": luc, **d})
    if KE_THUA.get("ph") == ph:                        # Q9MG/QUVZ: lich su lan_01 (da xac minh hash lan goi nay); chi them
        for x in KE_THUA["muc"]:
            them("muc", x["khoa"], **{k: v for k, v in x.items() if k != "khoa"})
    if chi is not None:
        f, S = chi
        _xet_lenh(them, f.parents[1].name, f)
        if S is not None:
            _xet_phien(them, f.parents[1].name, S)
        return so
    kd = ph / "khoi_dong"
    for bd in sorted(kd.glob("*.bat_dau.json")) if kd.is_dir() and not H.la_reparse(kd) else []:
        kid = bd.name[:-len(".bat_dau.json")]
        if kid == KHOI_DONG["id"] or (kd / f"{kid}.ket_thuc.json").is_file():
            continue
        j = _doc(bd) or {}
        txt = "".join(p.read_bytes().decode("utf-8", "replace") for p in
                      [kd / f"{kid}.faulthandler.log", *sorted(ph.glob(f"vong_*/faulthandler_lan_*_pid{j.get('pid')}.log"))]
                      if p.is_file())
        them("muc", f"khoi_dong/{kid}", su_co=f"khoi_dong:{kid}", vong=None,
             loai="tran_cung_watchdog" if "Timeout (" in txt else "fixture_gian_doan", an_toan_xac_dinh=False, native=False,
             dung_ngay=True, faulthandler_trich=txt[-1500:])
    for v in sorted(n for n in os.listdir(ph) if n.startswith("vong_") and (ph / n).is_dir() and not H.la_reparse(ph / n)):
        V = ph / v
        lg = V / "lan_goi"
        for bd in sorted(lg.glob("*.bat_dau.json")) if lg.is_dir() else []:
            n = int(bd.name.split(".")[0])
            j = _doc(bd) or {}
            if (v, n) == hien_tai or j.get("khoi_dong") == KHOI_DONG["id"] or (lg / f"{n:02d}.ket_thuc.json").is_file():
                continue
            fh = sorted(V.glob(f"faulthandler_lan_{n:02d}_pid*.log"))
            txt = fh[-1].read_bytes().decode("utf-8", "replace") if fh else ""
            them("muc", f"{v}/lan_goi/{n:02d}", su_co=f"khoi_dong:{j['khoi_dong']}" if j.get("khoi_dong") else f"{v}/lan_goi/{n:02d}",
                 vong=v, loai="tran_cung_watchdog" if "Timeout (" in txt else "fixture_gian_doan", an_toan_xac_dinh=False,
                 native=False, dung_ngay=True, faulthandler_trich=txt[-1500:])
        for f in sorted((V / "lenh").glob("*.json")) if (V / "lenh").is_dir() else []:
            if not f.name.endswith(".bat_dau.json"):
                _xet_lenh(them, v, f)
        PH = V / "phien"
        for S in sorted(PH.iterdir()) if PH.is_dir() else []:
            _xet_phien(them, v, S)
    return so


def ly_do_dung(so: dict) -> list[str]:
    """F8b: moi muc ngoai du kien KHONG phai crash native da thu hoi (timeout, traceback, exit 1, thieu log, Job/resume,
    PID khong xac dinh, cha gian doan, fixture bi ngat, cay chua thu hoi) -> dung NGAY; crash native ngoai du kien -> dung o
    su co thu NGUONG_NGOAI_NATIVE (dem theo su_co, F8c)."""
    ly = [f"{x['khoa']}: {x['loai']} -- loi ngoai du kien / trang thai an toan chua xac dinh" for x in so["muc"] if _dung_ngay(x)]
    n = su_co_native(so)
    if len(n) >= NGUONG_NGOAI_NATIVE:
        ly.append(f"{len(n)} crash native ngoai du kien (nguong {NGUONG_NGOAI_NATIVE}): {n}")
    return ly


def ap_so_sau_lenh(ph: Path, hien_tai, p_kq: Path, S: Path | None) -> None:
    """F8b: goi NGAY sau khi ghi ket qua moi lenh con: cap nhat so tu bang chung cua lenh do + ap quyet dinh dung."""
    try:
        dung = cap_nhat_so(ph, hien_tai, (p_kq, S))
    except LOI_SO as e:                                # so hong / sai cau truc -> khong biet trang thai -> dung (khong de ca nuot)
        dung_so_hong(ph, f"sau lenh {p_kq.stem}: {type(e).__name__}: {e}")
        raise DungFixture(f"so crash ngoai du kien khong dung duoc sau lenh {p_kq.stem}: {type(e).__name__}: {e}") from None
    if dung:
        raise DungFixture(f"so crash ngoai du kien yeu cau DUNG HOI sau lenh {p_kq.stem}")


def cap_nhat_so(ph: Path, hien_tai: tuple[str, int] | None, chi: tuple[Path, Path | None] | None = None) -> bool:
    """Ghi so (ghi de = chi them) + dung_fixture.json (ghi MOT lan, khong tu go). True = phai dung hoi."""
    so = ra_soat(ph, hien_tai, chi)
    H.ghi_json(ph / "so_crash_ngoai_du_kien.json", so, [ph], ghi_de=True)
    p, ly = ph / "dung_fixture.json", ly_do_dung(so)
    if ly and not p.exists():
        H.ghi_json(p, {"ly_do": ly, "luc": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "huong_dan": "DUNG HOI nguoi dung; khong tu xoa/doi ten file nay, khong mo vong/phien moi de reset"}, [ph])
    if p.exists() or H.la_reparse(p):
        d = _doc(p) or {}
        print(f"[DUNG HOI] fixture da dung ({H.rel(ph)}): {d.get('ly_do', 'dung_fixture.json khong doc duoc')}", file=sys.stderr)
        return True
    return False


# ---------------------------------------------------------------------------
# Q9MG / QUVZ: chuyen phien lan_01 -> lan_02 (quyet dinh nguoi dung Pbeca24e7 + P48bd9f29). lan_01 CHI DOC: manifest truoc
# (truoc ma tran) / sau (moi lan goi); xac minh cay tien trinh cu da ket thuc (PID + thoi diem tao + journal); ke thua lich
# su su co bang tham chieu duong / so byte / sha256; ngoai le CHI cho ly do dung hien co cua lan_01 ghim duoi day.
# ---------------------------------------------------------------------------
_L01 = "results/phase2/r61_bdd/tests/cong_b_20261026/lan_01"
_KD01 = "20261027_030727_163_pid12248"
_LC01 = f"{_L01}/vong_01/phien/thanh_cong/van_hanh/lan_chay/0002_mo-phien"
_LDN = "loi ngoai du kien / trang thai an toan chua xac dinh"
CHUYEN_PHIEN = {
    "results/phase2/r61_bdd/tests/cong_b_20261026/lan_02": {
        "tu": _L01,
        "quyet_dinh": [
            {"id": "Pbeca24e7", "ma": "Q9MG", "noi_dung": "lenh B1 sang --phien .../lan_02; lan_01 giu nguyen byte (khong ghi/xoa/doi "
             "ten/go dung_fixture.json, khong goi lai); xac minh cay cu da ket thuc truoc ma tran; ke thua lich su su co lan_01 "
             "bang tham chieu/hash bat bien, khong reset/dem trung; ngoai le chi cho ly do dung hien co cua lan_01; loi moi van chan"},
            {"id": "P48bd9f29", "ma": "QUVZ", "noi_dung": "buoc gop B1 + Q73M + Q9MG; cho phep chay fixture lan_02 o buoc nay"}],
        "ly_do_dung_bao_phu": [f"khoi_dong/{_KD01}: fixture_gian_doan -- {_LDN}",
                               f"vong_01/lan_goi/01: fixture_gian_doan -- {_LDN}",
                               f"lan_chay_mo:{_LC01}: lan_chay_runner_dang_mo_cha_gian_doan -- {_LDN}"],
        "khoa_bao_phu": [f"khoi_dong/{_KD01}", "vong_01/lan_goi/01", f"lan_chay_mo:{_LC01}"],
        "su_co_goc": f"khoi_dong:{_KD01}",
        "log_su_co_goc": f"{_L01}/vong_01/faulthandler_lan_01_pid12248.log",
    }}
KE_THUA: dict = {}


class ChuaXacMinh(Exception):
    """Chua xac minh duoc an toan de chuyen phien (truoc quyet dinh) -> exit 40, khong chay ma tran."""


def manifest_thu_muc(d: Path) -> dict:
    """CHI DOC: {duong tuong doi: [so byte, sha256]} moi file duoi d; reparse o bat ky dau -> LoiHopDong."""
    a = Path(os.path.abspath(d))
    if any(H.la_reparse(p) for p in [a, *a.parents]) or not a.is_dir():
        raise H.LoiHopDong(f"{H.rel(a)} khong phai thu muc thuong / di qua reparse")
    tep = {}
    for g, dirs, files in os.walk(a, followlinks=False):
        for n in dirs + files:
            if H.la_reparse(Path(g) / n):
                raise H.LoiHopDong(f"reparse trong {H.rel(a)}: {H.rel(Path(g) / n)}")
        for f in files:
            p = Path(g) / f
            tep[p.relative_to(a).as_posix()] = [os.lstat(p).st_size, H.sha256_file(p)]
    tep = dict(sorted(tep.items()))
    return {"goc": H.rel(a), "so_file": len(tep), "tong_byte": sum(v[0] for v in tep.values()),
            "sha256_tep": H.sha256_bytes(json.dumps(tep, sort_keys=True).encode("utf-8")), "luc": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tep": tep}


def _ft_epoch(t):
    return None if t is None else t / 1e7 - 11644473600.0


def mo_tien_trinh(pid: int) -> dict:
    import ctypes
    from ctypes import wintypes
    k = VA._k32()
    h = k.OpenProcess(0x1000, False, int(pid))            # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        e = ctypes.get_last_error()
        return {"mo_duoc": False, "loi_win": e, "khong_ton_tai": e == 87, "tao": None, "tao_epoch": None, "con_chay": None}
    try:
        ma = wintypes.DWORD()
        ok = k.GetExitCodeProcess(h, ctypes.byref(ma))
        t = VA.thoi_diem_tao(h)
        return {"mo_duoc": True, "loi_win": None, "khong_ton_tai": False, "tao": t, "tao_epoch": _ft_epoch(t),
                "con_chay": (ma.value == 259) if ok else None}
    finally:
        k.CloseHandle(h)


def liet_ke_tien_trinh() -> list[dict]:
    """Toolhelp32: MOI tien trinh dang co (pid, ppid, exe) + thoi diem tao / con chay (CHI doc, thu vien chuan)."""
    import ctypes
    from ctypes import wintypes

    class PE(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_wchar * 260)]
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PE)]
    k.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PE)]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k.CreateToolhelp32Snapshot(0x2, 0)                  # TH32CS_SNAPPROCESS
    if not h or h == ctypes.c_void_p(-1).value:
        raise OSError(f"CreateToolhelp32Snapshot loi {ctypes.get_last_error()}")
    out = []
    try:
        e = PE()
        e.dwSize = ctypes.sizeof(PE)
        ok = k.Process32FirstW(h, ctypes.byref(e))
        while ok:
            out.append({"pid": int(e.th32ProcessID), "ppid": int(e.th32ParentProcessID), "exe": e.szExeFile})
            ok = k.Process32NextW(h, ctypes.byref(e))
    finally:
        k.CloseHandle(h)
    if not out:
        raise OSError("Toolhelp32 khong liet ke duoc tien trinh nao")
    for x in out:
        x.update(mo_tien_trinh(x["pid"]) if x["pid"] else {"mo_duoc": False, "loi_win": None, "khong_ton_tai": False,
                                                         "tao": None, "tao_epoch": None, "con_chay": None})
    return out


def ung_vien_lan_01(src: Path) -> tuple[list[dict], list[str]]:
    """Moi PID cua lan_01 co trong journal / ban ghi (fixture, lan goi, lenh con, con runner, chau tham do, cha runner)."""
    ds, mo = [], []
    kd = src / "khoi_dong"
    for bd in sorted(kd.glob("*.bat_dau.json")):
        j, kid = H.doc_json(bd), bd.name[:-len(".bat_dau.json")]
        ds.append({"pid": int(j["pid"]), "vai": "fixture (journal khoi dong)", "nguon": H.rel(bd), "tao": None,
                   "bat_dau_epoch": float(j["bat_dau"]), "journal_ket_thuc": (kd / f"{kid}.ket_thuc.json").is_file()})
    for V in sorted(p for p in src.glob("vong_*") if p.is_dir()):
        for bd in sorted((V / "lan_goi").glob("*.bat_dau.json")):
            j, n = H.doc_json(bd), bd.name.split(".")[0]
            ds.append({"pid": int(j["pid"]), "vai": "fixture (lan goi vong)", "nguon": H.rel(bd), "tao": None,
                       "bat_dau_epoch": float(j["bat_dau"]), "journal_ket_thuc": (V / "lan_goi" / f"{n}.ket_thuc.json").is_file(),
                       "khoi_dong": j.get("khoi_dong")})
        for f in sorted((V / "lenh").glob("*.json")):
            if f.name.endswith(".bat_dau.json"):
                if not f.with_name(f.name[:-len(".bat_dau.json")] + ".json").is_file():
                    mo.append(H.rel(f))                # lenh dang mo: pid cha runner khong ghi -> phu bang log cha + quet cay
                continue
            m = H.doc_json(f)
            if isinstance(m.get("pid"), int):
                ds.append({"pid": m["pid"], "vai": f"lenh con {m.get('ten')}", "nguon": H.rel(f), "tao": m.get("thoi_diem_tao"),
                           "bat_dau_epoch": None, "exit": m.get("exit")})
        for f in sorted(V.glob("phien/*/van_hanh/lan_chay/*/con.json")) + sorted(V.glob("tham_do/*/chau.json")):
            c = H.doc_json(f)
            ds.append({"pid": int(c["pid"]), "vai": "con runner" if f.name == "con.json" else "chau tham do", "nguon": H.rel(f),
                       "tao": c.get("thoi_diem_tao"), "bat_dau_epoch": None,
                       "lan_chay_ket_thuc": (f.parent / "ket_thuc.json").is_file() if f.name == "con.json" else None})
        for f in sorted(V.glob("phien/*/van_hanh/cha/*_pid*.faulthandler.log")):
            ts, pid = f.name.split(".")[0].rsplit("_pid", 1)
            ds.append({"pid": int(pid), "vai": "cha runner (ten log faulthandler)", "nguon": H.rel(f), "tao": None,
                       "bat_dau_epoch": time.mktime(time.strptime(ts, "%Y%m%d_%H%M%S"))})
    return ds, mo


def _da_ket_thuc(u: dict, tt: dict) -> tuple[bool, str]:
    if tt["khong_ton_tai"]:
        return True, "khong con tien trinh nao mang PID nay (OpenProcess loi 87)"
    if not tt["mo_duoc"] or tt["tao"] is None:
        return False, f"PID dang ton tai nhung khong doc duoc thoi diem tao (loi {tt['loi_win']}) -- khong xac dinh"
    if u["tao"] is not None:
        if tt["tao"] != u["tao"]:
            return True, "PID tai dung: thoi diem tao khac ban ghi lan_01"
    elif u["bat_dau_epoch"] is not None:
        if tt["tao_epoch"] > u["bat_dau_epoch"] + 2:
            return True, "PID tai dung: tien trinh hien tai tao SAU moc bat dau ghi trong journal lan_01"
    else:
        return False, "khong co moc doi chieu"
    if tt["con_chay"] is False:
        return True, "cung tien trinh nhung da ket thuc (doi tuong con handle)"
    return False, "CUNG tien trinh (thoi diem tao khop) van chay / khong xac dinh"


def kiem_cay_cu(src: Path) -> dict:
    """Bang chung fixture pid12248 + cay lien quan cua lan_01 da ket thuc: tung PID ghi trong lan_01 doi chieu PID + thoi
    diem tao (hoac moc journal); quet MOI tien trinh: con / chau (ppid thuoc lan_01) va python.exe tao trong cua so lan_01."""
    ds, mo = ung_vien_lan_01(src)
    procs = liet_ke_tien_trinh()
    theo_pid = {x["pid"]: x for x in procs}
    for u in ds:
        tt = theo_pid.get(u["pid"]) or mo_tien_trinh(u["pid"])
        tt = {k: tt.get(k) for k in ("mo_duoc", "loi_win", "khong_ton_tai", "tao", "tao_epoch", "con_chay", "exe", "ppid")}
        u["hien_tai"] = tt
        u["da_ket_thuc"], u["ly_do"] = _da_ket_thuc(u, tt)
    pids = {u["pid"] for u in ds}
    moc = [u["bat_dau_epoch"] if u["bat_dau_epoch"] is not None else _ft_epoch(u["tao"]) for u in ds]
    mt = [os.lstat(Path(g) / f).st_mtime for g, _, fs in os.walk(src, followlinks=False) for f in fs]
    w0, w1 = min(m for m in moc if m is not None) - 5.0, max(mt) + 5.0
    lien_quan = []
    for x in procs:
        if x["pid"] in (0, os.getpid()) or not (x["pid"] in pids or x["ppid"] in pids or x["exe"].lower() == "python.exe"):
            continue
        if x["tao_epoch"] is None:
            if not x["khong_ton_tai"]:
                lien_quan.append({**x, "ly_do": "lien quan nhung khong doc duoc thoi diem tao"})
            continue
        if w0 <= x["tao_epoch"] <= w1 and x["con_chay"] is not False:
            lien_quan.append({**x, "ly_do": "tao trong cua so thoi gian lan_01 va con chay"})
    kd = src / "khoi_dong"
    fh = (H.ROOT / CHUYEN_PHIEN[next(iter(CHUYEN_PHIEN))]["log_su_co_goc"])
    jr = {"khoi_dong_12248_bat_dau": (kd / f"{_KD01}.bat_dau.json").is_file(),
          "khoi_dong_12248_ket_thuc": (kd / f"{_KD01}.ket_thuc.json").is_file(),
          "lan_goi_01_ket_thuc": (src / "vong_01" / "lan_goi" / "01.ket_thuc.json").is_file(),
          "log_12248_windows_fatal_exception": fh.is_file() and "Windows fatal exception" in fh.read_bytes().decode("utf-8", "replace"),
          "khoi_dong_khac": {b.name: (kd / b.name.replace(".bat_dau.json", ".ket_thuc.json")).is_file()
                             for b in sorted(kd.glob("*.bat_dau.json")) if _KD01 not in b.name}}
    co_12248 = any(u["pid"] == 12248 and u["nguon"].endswith(f"{_KD01}.bat_dau.json") for u in ds)
    dat = bool(ds) and co_12248 and all(u["da_ket_thuc"] for u in ds) and not lien_quan
    return {"dat": dat, "luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid_kiem": os.getpid(), "so_tien_trinh_he_thong": len(procs),
            "cua_so_lan_01_epoch": [round(w0, 3), round(w1, 3)], "ung_vien": ds, "lenh_dang_mo_khong_pid": mo,
            "lien_quan_con_song": lien_quan, "doi_chieu_journal": jr,
            "python_dang_chay": [{k: x[k] for k in ("pid", "ppid", "exe", "tao_epoch", "con_chay")} for x in procs
                                 if x["exe"].lower() == "python.exe"],
            "phuong_phap": "Toolhelp32 + OpenProcess(QUERY_LIMITED)/GetProcessTimes/GetExitCodeProcess; PID co thoi diem tao "
                           "ghi san -> so dung FILETIME; PID chi co moc journal -> tien trinh hien tai tao sau moc + 2 s = PID tai "
                           "dung; khong suy tu viec PID don le khong ton tai ma doi chieu ca cay (ppid) va python.exe trong cua so"}


def _ref(p: Path) -> dict:
    return {"duong": H.rel(p), "so_byte": p.stat().st_size, "sha256": H.sha256_file(p)}


def lien_ket_su_co(src: Path, cfg: dict) -> dict:
    """Nhan dien CUNG su co (khong dem trung): 3 ly do dung cua lan_01 deu do lan goi fixture pid12248 bi ngat."""
    V, kd = src / "vong_01", src / "khoi_dong"
    lg = H.doc_json(V / "lan_goi" / "01.bat_dau.json")
    mo = sorted(p.name for p in (V / "lenh").glob("*.bat_dau.json")
                if not p.with_name(p.name[:-len(".bat_dau.json")] + ".json").is_file())
    lc = H.ROOT / _LC01
    bd = H.doc_json(V / "lenh" / "tc_mo_phien.bat_dau.json")
    con = H.doc_json(lc / "con.json")
    t_lenh = time.mktime(time.strptime(bd["luc"], "%Y-%m-%dT%H:%M:%S"))
    t_con = _ft_epoch(con.get("thoi_diem_tao"))
    k2 = sorted(p for p in kd.glob("*.ket_thuc.json"))
    dk = {"lan_goi_01_thuoc_khoi_dong_pid12248": lg.get("khoi_dong") == _KD01 and lg.get("pid") == 12248,
          "vong_01_chi_co_lan_goi_01": sorted(p.name for p in (V / "lan_goi").glob("*.bat_dau.json")) == ["01.bat_dau.json"],
          "lenh_dang_mo_duy_nhat_la_tc_mo_phien": mo == ["tc_mo_phien.bat_dau.json"],
          "tc_mo_phien_bat_dau_sau_lan_goi_01": t_lenh >= float(lg["bat_dau"]) - 1,
          "lan_chay_0002_thieu_ket_thuc": not (lc / "ket_thuc.json").exists(),
          "con_0002_tao_sau_tc_mo_phien": t_con is not None and t_con >= t_lenh - 1,
          "con_0002_argv_tro_lan_chay_0002": os.path.normcase(str(con["argv"][-1])) == os.path.normcase(str(lc)),
          "lan_goi_sau_khong_chon_vong": bool(k2) and all(H.doc_json(p).get("vong") is None for p in k2)}
    return {"dat": all(dk.values()), "dieu_kien": dk, "su_co_goc": cfg["su_co_goc"]}


def chuyen_phien(ph: Path) -> dict | None:
    """Q9MG/QUVZ. Lan dau (chua co quyet_dinh.json): manifest lan_01 truoc -> kiem cay cu -> doi chieu noi dung dung cua
    lan_01 voi danh sach bao phu -> nhan dien cung su co -> ghi quyet_dinh.json (mot lan). MOI lan goi: xac minh lai moi
    tham chieu (so byte + sha256) va manifest lan_01 hien tai == truoc, roi nap KE_THUA."""
    cfg = CHUYEN_PHIEN.get(H.rel(ph))
    if cfg is None:
        return None
    src, cp, vung = H.ROOT / cfg["tu"], ph / "chuyen_phien", [ph]
    qd_p = cp / "quyet_dinh.json"
    m_now = manifest_thu_muc(src)
    if not qd_p.is_file():
        tr = cp / "manifest_lan_01_truoc.json"
        if not tr.is_file():
            H.ghi_json(tr, m_now, vung)
        if H.doc_json(tr)["tep"] != m_now["tep"]:
            raise ChuaXacMinh("lan_01 khac manifest truoc (ghi o lan goi truoc quyet dinh)")
        kc = kiem_cay_cu(src)
        kc_p = cp / f"kiem_cay_cu_{KHOI_DONG['id']}.json"
        H.ghi_json(kc_p, kc, vung)
        print(f"kiem cay lan_01: {'DAT' if kc['dat'] else 'KHONG DAT'} ({len(kc['ung_vien'])} PID, "
              f"{len(kc['lien_quan_con_song'])} lien quan con song) -> {H.rel(kc_p)}")
        if not kc["dat"]:
            raise ChuaXacMinh(f"chua xac minh cay tien trinh lan_01 da ket thuc: {H.rel(kc_p)}")
        so_p, dz_p = src / "so_crash_ngoai_du_kien.json", src / "dung_fixture.json"
        so, dz = H.doc_json(so_p), H.doc_json(dz_p)
        if loi_so(so) or dz.get("ly_do") != cfg["ly_do_dung_bao_phu"] \
                or sorted(x["khoa"] for x in so["muc"]) != sorted(cfg["khoa_bao_phu"]):
            raise ChuaXacMinh("so / dung_fixture.json cua lan_01 khac danh sach ly do dung duoc quyet dinh bao phu")
        lk = lien_ket_su_co(src, cfg)
        fh = H.ROOT / cfg["log_su_co_goc"]
        native = "Windows fatal exception" in fh.read_bytes().decode("utf-8", "replace")
        muc = [{"khoa": f"ke_thua:lan_01:{x['khoa']}", "loai": x["loai"],
                "su_co": cfg["su_co_goc"] if lk["dat"] else (x.get("su_co") or x["khoa"]),
                "vong": f"lan_01/{x.get('vong') or 'khoi_dong'}", "an_toan_xac_dinh": True, "native": native, "dung_ngay": False,
                "ke_thua": {"nguon": H.rel(so_p), "khoa_goc": x["khoa"], "su_co_goc_lan_01": x.get("su_co"),
                            "dung_ngay_goc": x.get("dung_ngay"), "native_goc": x.get("native")},
                "ngoai_le_chuyen_phien": "Pbeca24e7 (Q9MG) + P48bd9f29 (QUVZ)"} for x in so["muc"]]
        tc = [so_p, dz_p, *sorted((src / "khoi_dong").iterdir()), src / "vong_01" / "lan_goi" / "01.bat_dau.json",
              src / "vong_01" / "lenh" / "tc_mo_phien.bat_dau.json", H.ROOT / _LC01 / "con.json", H.ROOT / _LC01 / "goi.json", fh]
        qd = {"loai": "quyet_dinh_chuyen_phien", "tu": cfg["tu"], "den": H.rel(ph), "quyet_dinh": cfg["quyet_dinh"],
              "ly_do_dung_bao_phu": cfg["ly_do_dung_bao_phu"], "khoa_bao_phu": cfg["khoa_bao_phu"],
              "tham_chieu": [_ref(p) for p in tc], "manifest_truoc": {**_ref(tr), "so_file": m_now["so_file"],
                                                                      "sha256_tep": m_now["sha256_tep"]},
              "kiem_cay_cu": _ref(kc_p), "lien_ket_su_co": lk, "native_su_co_goc": native, "muc_ke_thua": muc,
              "dem_ke_thua": {"ban_ghi": len(muc), "su_co": sorted({m["su_co"] for m in muc}),
                              "su_co_native": sorted({m["su_co"] for m in muc}) if native else [],
                              "tiem_lan_01": len(so["tiem"])},
              "quy_tac": ["muc ke thua chi them vao so lan_02 (khong reset khi doi vong/snapshot); dem native theo su_co",
                          "ngoai le CHI cho khoa_bao_phu (ghim trong ma) va chi khi xac minh tham chieu + manifest DAT o lan goi",
                          "moi loi / ly do dung moi cua lan_02 theo chinh sach hien hanh (dung_fixture.json cua lan_02)"],
              "luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), "khoi_dong": KHOI_DONG["id"]}
        H.ghi_json(qd_p, qd, vung)
    qd = H.doc_json(qd_p)
    loi = []
    for r in qd["tham_chieu"] + [qd["manifest_truoc"], qd["kiem_cay_cu"]]:
        p = H.ROOT / r["duong"]
        if not p.is_file() or H.la_reparse(p) or p.stat().st_size != r["so_byte"] or H.sha256_file(p) != r["sha256"]:
            loi.append(f"tham chieu {r['duong']} lech so byte / sha256")
    m0 = H.doc_json(H.ROOT / qd["manifest_truoc"]["duong"])
    if m0["tep"] != m_now["tep"]:
        loi.append(f"lan_01 khac manifest truoc: {sorted(set(m0['tep']) ^ set(m_now['tep']))[:5]} / "
                   f"{[k for k in m0['tep'] if m_now['tep'].get(k) not in (None, m0['tep'][k])][:5]}")
    if loi:
        raise H.LoiHopDong(f"chuyen phien: {loi[:5]} -- DUNG HOI")
    hop_le = {m["khoa"] for m in qd["muc_ke_thua"]} & {f"ke_thua:lan_01:{k}" for k in cfg["khoa_bao_phu"]}
    KE_THUA.update(ph=ph, muc=qd["muc_ke_thua"], khoa_hop_le=hop_le, quyet_dinh=_ref(qd_p), manifest_truoc=m0)
    return {"quyet_dinh": _ref(qd_p), "manifest_lan_01_khop_truoc": True, "so_file_lan_01": m_now["so_file"],
            "khoa_hop_le": sorted(hop_le)}


def ghi_manifest_sau(ph: Path) -> dict | None:
    if KE_THUA.get("ph") != ph:
        return None
    m = manifest_thu_muc(H.ROOT / CHUYEN_PHIEN[H.rel(ph)]["tu"])
    p = ph / "chuyen_phien" / f"manifest_lan_01_sau_{KHOI_DONG['id']}.json"
    H.ghi_json(p, m, [ph])
    return {"duong": H.rel(p), "dat": m["tep"] == KE_THUA["manifest_truoc"]["tep"], "so_file": m["so_file"],
            "sha256_tep": m["sha256_tep"]}


def tong_ket_lan_01(ph: Path) -> dict | None:
    if H.rel(ph) not in CHUYEN_PHIEN:
        return None
    m0, cp = KE_THUA.get("manifest_truoc") or {}, ph / "chuyen_phien"
    sau = sorted(cp.glob("manifest_lan_01_sau_*.json"))
    lech = [p.name for p in sau if H.doc_json(p)["tep"] != m0.get("tep")]
    return {"dat": KE_THUA.get("ph") == ph and bool(m0) and bool(sau) and not lech, "so_file": m0.get("so_file"),
            "sha256_tep": m0.get("sha256_tep"), "manifest_sau": [H.rel(p) for p in sau], "lech": lech,
            "quyet_dinh": KE_THUA.get("quyet_dinh"), "kiem_cay_cu": [H.rel(p) for p in sorted(cp.glob("kiem_cay_cu_*.json"))],
            "muc_ke_thua": [m["khoa"] for m in KE_THUA.get("muc", [])],
            "su_co_ke_thua": sorted({m["su_co"] for m in KE_THUA.get("muc", [])}), "khoa_hop_le": sorted(KE_THUA.get("khoa_hop_le", ()))}


# ---------------------------------------------------------------------------
# Vong / snapshot / ket luan / main
# ---------------------------------------------------------------------------
def snapshot_fixture() -> dict:
    ma = {f"r61_bdd/cong_b/{n}": H.sha256_file(KD.CONG_B / n) for n in (*KD.FILE_MA_B, "test_van_hanh_cong_b.py")}
    ma.update({f"r61_bdd/{n}": H.sha256_file(H.SRC / "r61_bdd" / n) for n in sorted(os.listdir(H.SRC / "r61_bdd")) if n.endswith(".py")})
    ma.update({p.relative_to(H.SRC).as_posix(): H.sha256_file(p) for p in D.dong_import_tinh(H.SRC)})
    s = {"ma": dict(sorted(ma.items())), "hop_dong_cong_b": H.sha256_file(KD.HOP_DONG_B), "hop_dong_lenh": H.sha256_file(VA.HOP_DONG),
         "giao_thuc_that": H.sha256_file(D.GIAO_THUC), "python": sys.executable}
    return {**s, "sha256": H.sha256_bytes(json.dumps(s, sort_keys=True).encode("utf-8"))}


def chon_vong(ph: Path, snap: dict) -> Path:
    ds = sorted(n for n in os.listdir(ph) if n.startswith("vong_") and n[5:].isdigit() and (ph / n).is_dir())
    if ds and (ph / ds[-1] / "vong.json").is_file() and H.doc_json(ph / ds[-1] / "vong.json")["snapshot"]["sha256"] == snap["sha256"]:
        return ph / ds[-1]
    v = ph / f"vong_{max([int(n[5:]) for n in ds] + [0]) + 1:02d}"
    H.ghi_json(v / "vong.json", {"snapshot": snap, "tao_luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(),
                                 "vong_truoc": ds[-1] if ds else None}, [ph])
    return v


def gan_vong(V: Path) -> None:
    T.FX, T.A = V / "fx", V / "fx" / "A"
    T.HD, T.GT = T.FX / "hop_dong_cong_b_fixture.json", T.FX / "giao_thuc_cong_b.json"


def so_pyc(a: dict, b2: dict) -> dict:
    them = sorted(set(b2["pyc"]) - set(a["pyc"]))
    mat = sorted(set(a["pyc"]) - set(b2["pyc"]))
    doi = sorted(p for p in set(a["pyc"]) & set(b2["pyc"]) if a["pyc"][p] != b2["pyc"][p])
    return {"them": them, "mat": mat, "doi": doi, "dat": not (them or mat or doi)}


def main(lan: int) -> int:
    """Chay cac ca chua co ket qua. 0 = moi ca da co ket qua (ket luan tinh o tong_ket); 10 = het ngan sach."""
    for i, (ten, fn) in enumerate(CA):
        kq_p = T.V / "ca" / f"{i:02d}_{ten}.json"
        if kq_p.is_file():
            continue
        print(f"== ca {i:02d} {ten}")
        KQ.clear()
        KQ_CAN.clear()
        t1 = time.time()
        try:
            can_tg(f"ca {ten}", 15)
            fn()
        except HetGio as e:
            print(f"[CHUA XONG] {e}; chay lai DUNG lenh de tiep tuc (lenh / buoc da xong khong chay lai)")
            return H.EXIT_CHUA_XONG
        except Exception:                              # noqa: BLE001 -- loi trong ca: ghi KHONG DAT kem traceback, khong nuot
            import traceback
            kiem(f"{ten}_khong_loi_bat_ngo", False, traceback.format_exc()[-3000:])
            traceback.print_exc()
        can = list(dict.fromkeys(KQ_CAN))
        co = {k["kiem"] for k in KQ}
        for n in can:                                  # F7: kiem bat buoc khong chay -> KHONG DAT (khong im lang)
            if n not in co:
                kiem(n, False, "kiem BAT BUOC chua chay (dieu kien truoc khong dat / nhanh bi bo qua / loi giua ca)")
        s_sau = snapshot_fixture()["sha256"]
        kiem(f"{ten}_ma_khong_doi_trong_ca", s_sau == H.doc_json(T.V / "vong.json")["snapshot"]["sha256"])
        H.ghi_json(kq_p, {"ten": ten, "giay_lan_cuoi": round(time.time() - t1, 3), "lan_goi": lan, "can": can,
                          "kiem": list(KQ), "ben_thu_ba_cha": ben_thu_ba(f"sau_ca:{ten}", day_du=True)}, [T.V])
    return 0


def tong_ket(ph: Path, lan: int) -> dict:
    """F7: ket luan HIEN TAI tinh lai tu bang chung day du moi lan goi (khong dung 'dat' cu)."""
    ten_ca = [f"{i:02d}_{t}" for i, (t, _) in enumerate(CA)]
    kq = {n: H.doc_json(T.V / "ca" / f"{n}.json") for n in ten_ca if (T.V / "ca" / f"{n}.json").is_file()}
    la = sorted(p.stem for p in (T.V / "ca").glob("*.json") if p.stem not in ten_ca) if (T.V / "ca").is_dir() else []
    tat = [k for x in kq.values() for k in x["kiem"]]
    thieu_ca = [n for n in ten_ca if n not in kq]
    thieu_kiem = [f"{n}:{c}" for n, x in kq.items() for c in x.get("can", []) if c not in {k["kiem"] for k in x["kiem"]}]
    khong_dat = [k["kiem"] for k in tat if not k["dat"]]
    pyc = {}
    for k in range(1, lan + 1):
        p = T.V / "pyc" / f"lan_{k:02d}_so_sanh.json"
        pyc[f"lan_{k:02d}"] = H.doc_json(p)["dat"] if p.is_file() else None
    so = ra_soat(ph, (T.V.name, lan))
    ngoai = [x for x in so["muc"] if x.get("vong") == T.V.name]
    bc_loi = VB.kiem_bytecode()[1]
    ls = [H.doc_json(p) for p in sorted((T.V / "lenh").glob("*.json")) if not p.name.endswith(".bat_dau.json")]
    lgs = [H.doc_json(p) for p in sorted((T.V / "lan_goi").glob("*.bat_dau.json"))]
    kts = {p.name[:2]: H.doc_json(p) for p in sorted((T.V / "lan_goi").glob("*.ket_thuc.json"))}
    # P43284b78: sys.modules cua cha MOI lan goi trong vong (lan nay: hien tai; lan truoc: journal ket_thuc) -- khong co
    # ultralytics/torch, ngoai thu vien chuan chi numpy (tru goi nap san luc khoi dong)
    tb = {f"lan_{k:02d}": ben_thu_ba("tong_ket", day_du=True) if k == lan else (kts.get(f"{k:02d}") or {}).get("ben_thu_ba_cha")
          for k in range(1, lan + 1)}
    # Q73M: diem kiem SAU moi nhanh thuc thi -- moi lenh con, moi ca, moi buoc thiet lap (ban ghi thieu = khong dat)
    diem = [("lenh", x.get("ten"), x.get("ben_thu_ba_cha")) for x in ls]
    diem += [("ca", n, x.get("ben_thu_ba_cha")) for n, x in kq.items()]
    diem += [("thiet_lap", p.name, H.doc_json(p).get("ben_thu_ba_cha")) for p in sorted((T.V / "thiet_lap").glob("*.xong.json"))]
    tb_loi = [{"loai": a, "ten": n, "ban_ghi": x if isinstance(x, dict) else None} for a, n, x in diem if not tb_sach(x)]
    cha_sach = all(tb_sach(x) for x in tb.values()) and not tb_loi
    l01 = tong_ket_lan_01(ph)
    dat = bool(tat) and not (thieu_ca or la or thieu_kiem or khong_dat or ngoai or bc_loi) and all(v is True for v in pyc.values()) \
        and cha_sach and (l01 is None or l01["dat"])
    return {"dat": dat, "lan_goi_tinh": lan, "so_ca": len(kq), "thieu_ca": thieu_ca, "ca_la": la, "so_kiem": len(tat),
            "so_dat": sum(k["dat"] for k in tat), "khong_dat": khong_dat, "thieu_kiem_bat_buoc": thieu_kiem,
            "pyc_moi_lan": pyc, "crash_ngoai_du_kien_vong": ngoai, "crash_ngoai_du_kien_tat_ca": so["muc"],
            "so_loi_tiem": len(so["tiem"]), "bytecode_fixture": _BC, "bytecode_loi": bc_loi,
            "cha_khong_nap_ultralytics_torch": cha_sach, "ben_thu_ba_cha_moi_lan": tb,
            "ben_thu_ba_cha_diem_kiem": {"so_diem": len(diem), "theo_loai": {a: sum(d[0] == a for d in diem) for a in ("lenh", "ca", "thiet_lap")},
                                         "khong_sach_hoac_thieu": tb_loi,
                                         "pid_thoi_diem_tao": sorted({(x.get("pid"), x.get("thoi_diem_tao"), x.get("lan"))
                                                                      for x in [d[2] for d in diem] + list(tb.values())
                                                                      if isinstance(x, dict)}, key=str)},
            "chuyen_phien_lan_01": l01,
            "so_lenh_con": len(ls), "exit_lenh": {str(e): sum(x.get("exit") == e for x in ls) for e in sorted({x.get("exit") for x in ls}, key=str)},
            "lenh": [{k: x.get(k) for k in ("ten", "exit", "exit_hex", "giay", "cay", "trong_job_goc")} for x in ls],
            "giay_lenh_con": round(sum(x.get("giay") or 0 for x in ls), 3), "python": sys.executable,
            "lan_goi": [{"lan": i + 1, "bat_dau": x.get("bat_dau"), "pid": x.get("pid"),
                         "ket_thuc": kts.get(f"{i + 1:02d}")} for i, x in enumerate(lgs)],
            "vong": H.rel(T.V), "snapshot": H.doc_json(T.V / "vong.json")["snapshot"], "kiem": tat}


def chay(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phien", required=True)
    ap.add_argument("--ngan-sach-giay", type=float, required=True)
    ap.add_argument("--gioi-han-giay", type=float, required=True)
    ns = ap.parse_args(argv)
    ph = Path(os.path.abspath(H.ROOT / ns.phien))
    if not H._trong(ph, KD.VUNG_FIXTURE_B) or ph == KD.VUNG_FIXTURE_B or not 0 < ns.ngan_sach_giay <= 420 \
            or not ns.ngan_sach_giay < ns.gioi_han_giay <= TRAN_TOI_DA or VB.kiem_bytecode()[1]:
        print("--phien duoi results/phase2/r61_bdd/tests/cong_b_20261026/; 0 < ngan_sach <= 420 < ... ; gioi_han <= 540; "
              f"python -B ({VB.kiem_bytecode()[1]})", file=sys.stderr)
        return H.EXIT_HOP_DONG
    if KHOI_DONG["id"] is None or KHOI_DONG["dir"] != ph / "khoi_dong":   # F8a: khong co journal khoi dong -> khong chay
        print(f"[LOI HOP DONG] khong ghi duoc journal khoi dong truoc import: {KHOI_DONG['loi']}", file=sys.stderr)
        return H.EXIT_HOP_DONG
    H.kiem_duong_ghi(ph, [ph])
    H.tao_thu_muc(ph, [ph])
    T.PH = ph
    co_qd = (ph / "chuyen_phien" / "quyet_dinh.json").is_file()
    try:                                               # Q9MG/QUVZ: TRUOC ra soat so / chon vong / moi lenh con
        T.CP = chuyen_phien(ph)
    except ChuaXacMinh as e:                           # chua co quyet dinh -> phien moi chua co trang thai: dung, khong chay
        print(f"[DUNG HOI] chuyen phien chua xac minh duoc: {e}", file=sys.stderr)
        return VA.EXIT_DUNG
    except Exception as e:                             # noqa: BLE001 -- sau quyet dinh: lech tham chieu / loi moi -> dung, giu chan
        if co_qd and not (ph / "dung_fixture.json").exists():
            H.ghi_json(ph / "dung_fixture.json", {"ly_do": [f"chuyen phien: {type(e).__name__}: {e}"],
                                                  "luc": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                  "huong_dan": "DUNG HOI nguoi dung; khong tu xoa/doi ten file nay"}, [ph])
        print(f"[DUNG HOI] chuyen phien: {type(e).__name__}: {e}", file=sys.stderr)
        return VA.EXIT_DUNG
    try:
        if cap_nhat_so(ph, None):                      # F8: ra soat MOI vong + journal khoi dong TRUOC khi chon snapshot
            return VA.EXIT_DUNG
    except LOI_SO as e:                                # B0-F8b-2: so hong / sai cau truc -> dung, giu nguyen so
        dung_so_hong(ph, f"khoi dong: {type(e).__name__}: {e}")
        print(f"[DUNG HOI] {type(e).__name__}: {e}", file=sys.stderr)
        return VA.EXIT_DUNG
    T.V = chon_vong(ph, snapshot_fixture())
    T.SNAP = H.doc_json(T.V / "vong.json")["snapshot"]["sha256"]
    du_phong = max(45.0, 2 * _PYC_TRUOC["giay"] + 25.0)   # chup pyc cuoi + ghi ket thuc + cho cay, trong tran
    T.TRAN = T_BAT_DAU + ns.gioi_han_giay - DU_PHONG_TRAN
    T.HAN, T.MEM = T_BAT_DAU + ns.gioi_han_giay - du_phong, T_BAT_DAU + ns.ngan_sach_giay
    gan_vong(T.V)
    vung = [T.V]
    lg = T.V / "lan_goi"
    lan = len([p for p in lg.glob("*.bat_dau.json")]) + 1 if lg.is_dir() else 1
    T.LAN = lan
    H.ghi_json(lg / f"{lan:02d}.bat_dau.json", {"argv": _BC["orig_argv"], "bat_dau": T_BAT_DAU, "han": T.HAN, "mem": T.MEM,
                                                "tran": T.TRAN, "du_phong_ket_thuc": du_phong, "pid": os.getpid(),
                                                "khoi_dong": KHOI_DONG["id"], "chuyen_phien": T.CP,
                                                "ben_thu_ba_cha": ben_thu_ba("bat_dau_lan_goi", day_du=True)}, vung)
    H.ghi_json(T.V / "pyc" / f"lan_{lan:02d}_truoc.json", _PYC_TRUOC, vung)
    H.chuan_bi_moi_truong(T.V, vung)
    fh = open(H.kiem_duong_ghi(T.V / f"faulthandler_lan_{lan:02d}_pid{os.getpid()}.log", vung), "w", encoding="utf-8")
    faulthandler.enable(file=fh, all_threads=True)
    T.BV = bao_ve_lan_goi(T.TRAN, fh)                  # F1/F2: job goc TRUOC moi con + tran cung ghi vao log cua lan goi
    ma, loi_fx = H.EXIT_HOP_DONG, None
    try:
        if not T.BV.get("dat"):
            print(f"[LOI HOP DONG] khong gan duoc job goc cua fixture: {T.BV} -- khong khoi chay lenh con nao", file=sys.stderr)
        else:
            for k in sorted((T.V / "khoi_phuc").glob("*.json")) if (T.V / "khoi_phuc").is_dir() else []:
                phuc_hoi(k)                            # sua tam con do tu lan goi bi ngat (journal kiem nhu dau vao khong tin)
            if not (T.V / "thiet_lap" / "that_truoc.json").is_file():
                H.ghi_json(T.V / "thiet_lap" / "that_truoc.json", trang_thai_that(), vung)
            ma = main(lan)
    except DungFixture as e:                           # F8b: dung NGAY sau lenh co bang chung ngoai du kien
        print(f"[DUNG HOI] {e}", file=sys.stderr)
        ma = VA.EXIT_DUNG
    except H.LoiHopDong as e:
        print(f"[LOI HOP DONG] {e}", file=sys.stderr)
        ma = H.EXIT_HOP_DONG
    except BaseException:                            # noqa: BLE001 -- van chup pyc + ghi ket_thuc; cay da thu hoi trong lenh()
        import traceback
        loi_fx = traceback.format_exc()[-4000:]
        traceback.print_exc()
        ma = 1
    pyc_sau = chup_pyc()                               # SAU khi moi lenh da thu hoi ca cay (khoi_chay_trong_job doi active = 0)
    H.ghi_json(T.V / "pyc" / f"lan_{lan:02d}_sau.json", pyc_sau, vung)
    goc = H.doc_json(T.V / "pyc" / "lan_01_truoc.json") if (T.V / "pyc" / "lan_01_truoc.json").is_file() else _PYC_TRUOC
    ss = {"lan": lan, "trong_lan": so_pyc(_PYC_TRUOC, pyc_sau), "so_voi_dau_vong": so_pyc(goc, pyc_sau),
          "so_pyc": pyc_sau["so_pyc"], "giay_chup": [_PYC_TRUOC["giay"], pyc_sau["giay"]]}
    ss["dat"] = ss["trong_lan"]["dat"] and ss["so_voi_dau_vong"]["dat"]
    H.ghi_json(T.V / "pyc" / f"lan_{lan:02d}_so_sanh.json", ss, vung)
    print(f"pyc: {pyc_sau['so_pyc']} file trong pham vi; them/mat/doi trong lan {len(ss['trong_lan']['them'])}/"
          f"{len(ss['trong_lan']['mat'])}/{len(ss['trong_lan']['doi'])} -> {'DAT' if ss['dat'] else 'KHONG DAT'}")
    l01_sau = None
    try:
        l01_sau = ghi_manifest_sau(ph)                 # Q9MG: lan_01 nguyen byte SAU lan goi (truoc ket luan)
        if l01_sau is not None:
            print(f"lan_01 sau lan goi: {l01_sau['so_file']} file -> {'NGUYEN BYTE' if l01_sau['dat'] else 'LECH'}")
            if not l01_sau["dat"] and not (ph / "dung_fixture.json").exists():
                H.ghi_json(ph / "dung_fixture.json", {"ly_do": [f"lan_01 lech manifest truoc: {l01_sau['duong']}"],
                                                      "luc": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                      "huong_dan": "DUNG HOI nguoi dung; khong tu xoa/doi ten file nay"}, vung + [ph])
                ma = VA.EXIT_DUNG
    except LOI_SO as e:
        print(f"[DUNG HOI] manifest lan_01 sau: {type(e).__name__}: {e}", file=sys.stderr)
        ma = VA.EXIT_DUNG
    try:
        if ma == 0:                                    # moi ca da co ket qua: ket luan HIEN TAI tu bang chung day du
            tk = tong_ket(ph, lan)
            H.ghi_json(T.V / "ket_qua" / f"lan_{lan:02d}.json", tk, vung)
            ls = sorted((T.V / "ket_qua").glob("lan_*.json"))
            H.ghi_json(T.V / "ket_qua_test.json", {**tk, "lich_su": [{"tep": H.rel(p), "dat": H.doc_json(p)["dat"]} for p in ls]},
                       vung, ghi_de=True)
            print(f"KET QUA KIEM: {tk['so_dat']}/{tk['so_kiem']} dat; {tk['so_lenh_con']} lenh con; pyc {tk['pyc_moi_lan']}; "
                  f"thieu bat buoc {len(tk['thieu_kiem_bat_buoc'])}; ngoai du kien {len(tk['crash_ngoai_du_kien_vong'])} -> "
                  f"{'DAT' if tk['dat'] else 'KHONG DAT'}")
            ma = 0 if tk["dat"] else H.EXIT_HOP_DONG
        if cap_nhat_so(ph, (T.V.name, lan)):
            ma = VA.EXIT_DUNG
    except LOI_SO as e:                                # B0-F8b-2: ra soat / so loi o ket thuc -> dung, van ghi ket_thuc lan goi
        dung_so_hong(ph, f"ket thuc lan goi {lan}: {type(e).__name__}: {e}")
        print(f"[DUNG HOI] so crash ngoai du kien: {type(e).__name__}: {e}", file=sys.stderr)
        ma = VA.EXIT_DUNG
    H.ghi_json(lg / f"{lan:02d}.ket_thuc.json", {"exit": ma, "giay": round(time.time() - T_BAT_DAU, 3), "pyc_dat": ss["dat"],
                                                 "loi_fixture": loi_fx, "lan_01_sau": l01_sau,
                                                 "ben_thu_ba_cha": ben_thu_ba("ket_thuc_lan_goi", day_du=True)}, vung)
    faulthandler.cancel_dump_traceback_later()
    return ma


def ket_thuc_khoi_dong(ma: int) -> None:
    """F8a: dong journal khoi dong khi lan goi ket thuc CO KIEM SOAT (moi duong return cua chay())."""
    if KHOI_DONG["id"] is None:
        return
    H.ghi_json(KHOI_DONG["dir"] / f"{KHOI_DONG['id']}.ket_thuc.json",
               {"exit": ma, "giay": round(time.time() - T_BAT_DAU, 3), "vong": T.V.name if hasattr(T, "V") else None,
                "lan": getattr(T, "LAN", None), "luc": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "ben_thu_ba_cha": ben_thu_ba("ket_thuc_khoi_dong", day_du=True)}, [KHOI_DONG["dir"]])


if __name__ == "__main__":
    if CHE_DO == "tham_do":
        sys.exit(tham_do(sys.argv[2:]))
    if CHE_DO == "viec":
        sys.exit(viec(sys.argv[2:]))
    _ma = chay(sys.argv[1:])
    ket_thuc_khoi_dong(_ma)
    sys.exit(_ma)
