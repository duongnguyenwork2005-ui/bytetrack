"""
hop_dong_bdd.py -- Hop dong dung chung cua R61 (CHI thu vien chuan; import truoc moi thu vien ben thu ba).

  - Moi truong: sys.dont_write_bytecode, PYTHONDONTWRITEBYTECODE=1, YOLO_CONFIG_DIR / MPLCONFIGDIR trong
    <phien>/_cau_hinh_thu_vien/, TEMP/TMP/TMPDIR + tempfile.tempdir trong <phien>/_tam/.
  - Duong ghi: kiem LEXICAL (abspath nam trong vung) + RESOLVED (realpath nam trong vung) + KHONG co
    junction/symlink/reparse o BAT KY thanh phan nao (ke ca thu muc cha) truoc lan ghi/mkdir dau tien.
  - Ghi file: bytes nguyen ban, hash tren chinh bytes da ghi (khong chuan hoa LF/CRLF).
  - cli_log.json: argv, Python, cwd, exit, stdout/stderr cua tung lenh trong DUNG phien.
  - Marker don vi: <ten>.xong.json chua sha256 cac artifact; marker hop le chi khi bam lai khop.
  - Ma thoat: 0 hoan tat, 10 chua hoan tat (tiep tuc duoc), 2 loi hop dong/dau vao. Loi bat ngo KHONG bi nuot
    (traceback + exit 1 cua Python).
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PYTHON = "C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe"

EXIT_OK = 0
EXIT_CHUA_XONG = 10
EXIT_HOP_DONG = 2

FILE_ATTRIBUTE_REPARSE_POINT = 0x400

#: Vung goc cua R61 (tuong doi ROOT); cac lenh chi duoc ghi duoi cac vung con tuong minh cua chung.
VUNG_TESTS = ROOT / "results" / "phase2" / "r61_bdd" / "tests"
VUNG_CHUAN_BI = ROOT / "results" / "phase2" / "r61_bdd" / "chuan_bi"
VUNG_THUC_THI = ROOT / "results" / "phase2" / "r61_bdd" / "thuc_thi"


class LoiHopDong(Exception):
    """Vi pham hop dong / dau vao sai -> exit 2 (khong phai loi bat ngo)."""


class ChuaXong(Exception):
    """Doan da dung giua cac don vi (het ngan sach / gioi han) -> exit 10."""


# ---------------------------------------------------------------------------
# Hash
# ---------------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def chuan_json(o) -> str:
    return json.dumps(o, sort_keys=True, ensure_ascii=False, default=str)


def rel(p: Path) -> str:
    p = Path(os.path.abspath(p))
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


# ---------------------------------------------------------------------------
# Duong dan
# ---------------------------------------------------------------------------
def la_reparse(p: Path) -> bool:
    """True neu p la symlink / junction / reparse point bat ky (khong theo lien ket)."""
    try:
        st = os.lstat(p)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(st.st_mode):
        return True
    return bool(getattr(st, "st_file_attributes", 0) & FILE_ATTRIBUTE_REPARSE_POINT)


def _trong(p: Path, goc: Path) -> bool:
    try:
        p.relative_to(goc)
        return True
    except ValueError:
        return False


def kiem_duong_ghi(p, vung: list) -> Path:
    """Tra ve duong tuyet doi neu ghi duoc; nem LoiHopDong neu thoat vung hoac di qua reparse point.

    Kiem moi thanh phan tu o dia toi p (ca thu muc cha DA TON TAI) -- junction o bat ky dau cung bi chan.
    """
    if not vung:
        raise LoiHopDong("khong co vung ghi nao duoc khai bao")
    ap = Path(os.path.abspath(p))
    vung_abs = [Path(os.path.abspath(v)) for v in vung]
    goc = next((v for v in vung_abs if _trong(ap, v)), None)
    if goc is None:
        raise LoiHopDong(f"duong ghi {ap} nam ngoai vung {[rel(v) for v in vung_abs]}")
    cur = Path(ap.anchor)
    for part in ap.parts[1:]:
        cur = cur / part
        if la_reparse(cur):
            raise LoiHopDong(f"duong ghi {ap} di qua junction/symlink/reparse {cur}")
    real = Path(os.path.realpath(ap))
    if not any(_trong(real, Path(os.path.realpath(v))) for v in vung_abs):
        raise LoiHopDong(f"duong ghi {ap} resolve thanh {real} ngoai vung")
    return ap


def _kiem_to_tien_vung(d: Path, vung: list) -> None:
    """Thu muc cha CHUA co cua chinh vung (vd results/phase2/r61_bdd khi tao vung lan dau): chi duoc mkdir
    khi d la to tien cua mot vung khai bao va khong thanh phan nao da ton tai la reparse point."""
    if not any(_trong(Path(os.path.abspath(v)), d) for v in vung):
        raise LoiHopDong(f"thu muc {d} khong thuoc vung va khong phai to tien cua vung")
    cur = Path(d.anchor)
    for part in d.parts[1:]:
        cur = cur / part
        if la_reparse(cur):
            raise LoiHopDong(f"{d} di qua reparse {cur}")


def tao_thu_muc(p, vung: list) -> Path:
    ap = kiem_duong_ghi(p, vung)
    chua = []
    cur = ap
    while not cur.exists():
        chua.append(cur)
        cur = cur.parent
    vung_abs = [Path(os.path.abspath(v)) for v in vung]
    for d in reversed(chua):
        if any(_trong(d, v) for v in vung_abs):
            kiem_duong_ghi(d, vung)
        else:
            _kiem_to_tien_vung(d, vung)
        d.mkdir(exist_ok=False)
        if la_reparse(d):
            raise LoiHopDong(f"{d} thanh reparse sau mkdir")
    if not ap.is_dir():
        raise LoiHopDong(f"{ap} khong phai thu muc")
    return ap


def ghi_bytes(p, b: bytes, vung: list, ghi_de: bool = False) -> str:
    """Ghi nguyen bytes. ghi_de=False: file da ton tai -> LoiHopDong (khong ghi de). Tra ve sha256(b)."""
    ap = kiem_duong_ghi(p, vung)
    tao_thu_muc(ap.parent, vung)
    if ap.exists() and not ghi_de:
        raise LoiHopDong(f"khong ghi de file da co {rel(ap)}")
    tam = ap.with_name(ap.name + ".dang_ghi")
    kiem_duong_ghi(tam, vung)
    with open(tam, "wb") as f:
        f.write(b)
    os.replace(tam, ap)
    if sha256_file(ap) != sha256_bytes(b):
        raise LoiHopDong(f"bam lai {rel(ap)} khong khop bytes da ghi")
    return sha256_bytes(b)


def ghi_json(p, obj, vung: list, ghi_de: bool = False) -> str:
    b = (json.dumps(obj, indent=1, ensure_ascii=False, sort_keys=False, default=str) + "\n").encode("utf-8")
    return ghi_bytes(p, b, vung, ghi_de)


def doc_json(p) -> dict:
    return json.loads(Path(p).read_bytes().decode("utf-8"))


# ---------------------------------------------------------------------------
# Moi truong (goi TRUOC moi import ben thu ba)
# ---------------------------------------------------------------------------
def chuan_bi_moi_truong(phien, vung: list) -> dict:
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    phien = Path(os.path.abspath(phien))
    ch = tao_thu_muc(phien / "_cau_hinh_thu_vien", vung)
    tam = tao_thu_muc(phien / "_tam", vung)
    yolo = tao_thu_muc(ch / "ultralytics", vung)
    mpl = tao_thu_muc(ch / "matplotlib", vung)
    env = {"PYTHONDONTWRITEBYTECODE": "1", "YOLO_CONFIG_DIR": str(yolo), "MPLCONFIGDIR": str(mpl),
           "TEMP": str(tam), "TMP": str(tam), "TMPDIR": str(tam), "YOLO_OFFLINE": "1"}
    os.environ.update(env)
    tempfile.tempdir = str(tam)
    return env


def kiem_khong_pyc(thu_muc_src: Path = SRC / "r61_bdd") -> list[str]:
    return sorted(rel(p) for p in thu_muc_src.rglob("*.pyc"))


# ---------------------------------------------------------------------------
# cli_log
# ---------------------------------------------------------------------------
class _Tee(io.TextIOBase):
    def __init__(self, goc, buf):
        self.goc, self.buf = goc, buf

    def write(self, s):
        self.buf.write(s)
        try:
            self.goc.write(s)
        except Exception:  # noqa: BLE001 -- console dong khong lam hong log
            pass
        return len(s)

    def flush(self):
        with contextlib.suppress(Exception):
            self.goc.flush()


def chay_va_ghi_log(main_fn, argv: list[str], phien, vung: list, ten_lenh: str) -> int:
    """Chay main_fn(argv) -> exit; ghi them muc vao <phien>/cli_log.json (argv/Python/exit/stdout/stderr).

    LoiHopDong -> 2, ChuaXong -> 10; loi khac: ghi traceback vao log roi NEM LAI (khong nuot).
    """
    import traceback
    out, err = io.StringIO(), io.StringIO()
    t0 = time.time()
    ma, loi_khac = None, None
    with contextlib.redirect_stdout(_Tee(sys.stdout, out)), contextlib.redirect_stderr(_Tee(sys.stderr, err)):
        try:
            ma = int(main_fn(argv) or 0)
        except LoiHopDong as e:
            print(f"[LOI HOP DONG] {e}", file=sys.stderr)
            ma = EXIT_HOP_DONG
        except ChuaXong as e:
            print(f"[CHUA XONG] {e}")
            ma = EXIT_CHUA_XONG
        except BaseException as e:  # noqa: BLE001
            traceback.print_exc()
            loi_khac, ma = e, 1
    muc = {"lenh": ten_lenh, "argv": list(argv), "python": sys.executable, "python_version": sys.version,
           "cwd": os.getcwd(), "bat_dau": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
           "giay": round(time.time() - t0, 3), "exit": ma, "stdout": out.getvalue(), "stderr": err.getvalue()}
    p = Path(os.path.abspath(phien)) / "cli_log.json"
    cu = doc_json(p) if p.exists() else []
    cu.append(muc)
    ghi_json(p, cu, vung, ghi_de=True)
    if loi_khac is not None:
        raise loi_khac
    return ma


# ---------------------------------------------------------------------------
# Marker don vi + phien
# ---------------------------------------------------------------------------
def ghi_marker(p, artifact: list, vung: list, them: dict | None = None) -> dict:
    """Marker hoan tat: sha256 tung artifact (bam tren bytes da ghi). Khong ghi de marker da co."""
    goc = Path(os.path.abspath(p)).parent
    d = {"artifact": {Path(os.path.relpath(os.path.abspath(a), goc)).as_posix(): sha256_file(Path(a)) for a in artifact},
         "luc": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if them:
        d.update(them)
    ghi_json(p, d, vung, ghi_de=False)
    return d


def kiem_marker(p, artifact: list | None = None, danh_tinh: dict | None = None) -> tuple[str, list[str]]:
    """('xong'|'chua'|'hong', loi). 'hong' = marker ton tai nhung KHONG hop le: doc loi, schema sai, artifact rong,
    tap artifact khac ky vong (artifact), danh tinh don vi khac (danh_tinh), artifact thieu/bam lech.
    Marker 'hong' KHONG BAO GIO duoc coi la xong hay chua (xem da_xong)."""
    p = Path(p)
    if not p.exists() and not la_reparse(p):
        return "chua", []
    try:
        d = doc_json(p)
    except Exception as e:  # noqa: BLE001
        return "hong", [f"{rel(p)}: doc loi {e}"]
    if not isinstance(d, dict) or not isinstance(d.get("artifact"), dict) or not d["artifact"] \
            or not all(isinstance(h, str) and len(h) == 64 for h in d["artifact"].values()):
        return "hong", [f"{rel(p)}: schema marker sai / artifact rong"]
    loi = []
    if artifact is not None:
        ky = {Path(os.path.relpath(os.path.abspath(a), os.path.abspath(p.parent))).as_posix() for a in artifact}
        if set(d["artifact"]) != ky:
            loi.append(f"tap artifact {sorted(d['artifact'])} != ky vong {sorted(ky)}")
    for k, v in (danh_tinh or {}).items():
        if d.get(k) != v:
            loi.append(f"danh tinh {k}={d.get(k)!r} != {v!r}")
    for r, h in d["artifact"].items():
        a = p.parent / r                     # duong tuong doi theo thu muc marker
        if not a.is_file() or la_reparse(a):
            loi.append(f"thieu {r}")
        elif sha256_file(a) != h:
            loi.append(f"bam lech {r}")
    return ("hong" if loi else "xong"), loi


def da_xong(p, artifact: list | None = None, danh_tinh: dict | None = None) -> bool:
    """True = xong (bo qua); False = chua (duoc chay); marker hong -> LoiHopDong TRUOC moi ghi ket qua."""
    st, loi = kiem_marker(p, artifact, danh_tinh)
    if st == "hong":
        raise LoiHopDong(f"marker {rel(Path(p))} hong (khong chay de/ghi de): {loi[:4]}")
    return st == "xong"


def liet_ke_file(d) -> list[Path]:
    """Moi file duoi d qua os.walk (followlinks=False, khong vao reparse), sap xep. Thay Path.rglob: fixture
    lan_07 ghi access violation trong pathlib._iterate_directories (faulthandler.log)."""
    out = []
    for goc, dirs, files in os.walk(d, followlinks=False):
        dirs[:] = sorted(x for x in dirs if not la_reparse(Path(goc) / x))
        out += [Path(goc) / f for f in files]
    return sorted(out)


def bang_hash_thu_muc(d: Path, bo_qua=("cli_log.json",)) -> dict[str, str]:
    """{rel: sha256} moi file duoi d (khong theo reparse); dung cho SHA256SUMS va snapshot bat bien."""
    out = {}
    for p in liet_ke_file(d):
        if p.is_file() and not la_reparse(p) and p.name not in bo_qua:
            out[rel(p)] = sha256_file(p)
    return out


def liet_ke_nhe(d: Path, de_quy: bool = False) -> dict[str, list]:
    """Liet ke (size, mtime_ns) KHONG bam -- cho vung lon/co nguy co phat sinh ghi."""
    d = Path(d)
    if not d.exists():
        return {}
    it = d.rglob("*") if de_quy else d.iterdir()
    out = {}
    for p in sorted(it):
        try:
            st = os.lstat(p)
        except FileNotFoundError:
            continue
        out[rel(p)] = [st.st_size if p.is_file() else -1, st.st_mtime_ns]
    return out
