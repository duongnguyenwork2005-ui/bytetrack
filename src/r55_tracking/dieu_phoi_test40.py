"""
dieu_phoi_test40.py -- R55: bo dieu phoi test40 co TIEP TUC theo don vi cong viec, dung cache BAT BIEN.
Chi la lop DIEU PHOI: moi phep tinh (to xam + suy luan + ghi cache, tracker, TrackEval) goi ham da review
cua R54 (src/r54_tracking/chuan_bi_test40.py: tao_cache, mo_cache, chay_tracker, cham, tien_kiem...) va qua
do goi ham hien co cua repo (det_cache, run_tracking, run_trackeval). Phan moi: kiem dau vao, provenance,
marker hoan tat, chon lan thu, quan ly phien. Hop dong: configs/phase2/r55_tracking/hop_dong_dieu_phoi.json.

HAI DUONG TACH RIENG
  Chuan bi (chi doc du lieu, ghi results/phase2/r55_tracking/chuan_bi/):
    --buoc tien-kiem                                   doi chieu khoa R54 + tien_kiem R54 (manifest, checkpoint/
                                                       names CPU, bang hash anh + 256 anh mau, GT/seqinfo/ignored,
                                                       D4, cau hinh hieu luc) + backend/thu vien + dung luong dia
    --buoc khoa --tien-kiem <phien> --tests <phien>    khoa R55 (ten phien TUONG MINH, khong tu chon moi nhat)
    --buoc kiem-khoa --khoa <phien> [--luu chuan_bi]   bam lai khoa R55 + khoa R54; mac dinh CHI DOC
  Thuc thi (ghi results/phase2/r55_tracking/thuc_thi/<phien>/; duong that bat buoc --duyet <ma plan da duyet>):
    --buoc thuc-thi --cong-viec mo-phien --khoa <k> [--phien <ten>]
    --buoc thuc-thi --cong-viec cache|xac-nhan-cache --phien <p> --detector <d>
    --buoc thuc-thi --cong-viec tracking --phien <p> --run <r> [<r> ...] [--video <v>]
    --buoc thuc-thi --cong-viec cham --phien <p> --run <r> [<r> ...]
    --buoc thuc-thi --cong-viec lap-tracker --phien <p> --run <r> --video <v>
    --buoc thuc-thi --cong-viec cham-lai --phien <p> --run <r>
    --buoc thuc-thi --cong-viec hau-kiem --phien <p>
  Khong co mac dinh "chay tat ca"; import module khong chay gi.

FIXTURE: bien moi truong R55_FIXTURE = <...>/fixture.json NAM TRONG results/phase2/r55_tracking/tests/ -> moi vung
ghi doi vao thu muc chua fixture.json, nguon du lieu + giao thuc tong hop, detector GIA LAP (mo hinh gia cua fixture
R54). R55_NGAT=<diem>:<lan> (chi fixture) gia lap ngat tien trinh (os._exit 9), marker ghi bi cat
('<cache|tracking|cham|xac-nhan>-marker-cat') hoac loi cham.

Vong sua R55A-F1..F5: xac nhan cache cong bo bang HOAN_TAT.json ghi cuoi; marker hong = lan khong hop le, tim tiep;
kiem schema n_per_frame day du; kiem module (hien dien + sha256) SAU moi don vi truoc marker / ket luan; moi dich ghi
kiem doc lap voi goc vung ghi lexical (khong junction/symlink).

R56A (tai lieu handoff/logs/review56/01_bo_sung_chon_video.md):
  - tracking --video <v>: chi LOC danh sach don vi tracking con mot video (toan bo timeline video do). ctx.videos,
    manifest, kiem hash cache MOI video, rang buoc marker, dieu kien du video de cham/hau-kiem giu nguyen; --video o
    cong viec khac tracking/lap-tracker -> exit 2 (khong bo qua am tham).
  - khoa R56 (R56_LOCK) noi toi khoa R55 cha (duong dan + SHA256 trong hop dong): luc khoa, khoa cha phai nguyen ven,
    file khoa cha chi duoc khac o DUOC_THAY_KHOA_CHA (3 file ma/test/hop dong); bang khac biet cu/moi ghi vao khoa.
    Kiem khoa moi (kiem-khoa / mo-phien / moi cong viec) bam MOI file theo hash khoa moi, khong ngoai le.

    C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r55_tracking/dieu_phoi_test40.py --buoc tien-kiem
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

GOC = Path(__file__).resolve().parents[2]
SRC = GOC / "src"
if str(SRC / "r54_tracking") not in sys.path:
    sys.path.insert(0, str(SRC / "r54_tracking"))
import chuan_bi_test40 as W  # noqa: E402  (R54 da review; chi goi, khong sua)

VUNG_R55 = GOC / "results" / "phase2" / "r55_tracking"
HOP_DONG = GOC / "configs" / "phase2" / "r55_tracking" / "hop_dong_dieu_phoi.json"
THIET_KE = GOC / "handoff" / "logs" / "review55" / "01_thiet_ke_dieu_phoi.md"
THIET_KE_BO_SUNG = GOC / "handoff" / "logs" / "review56" / "01_bo_sung_chon_video.md"
FILE_TEST = Path(__file__).resolve().parent / "test_dieu_phoi_test40.py"
#: R56: 3 file DUY NHAT duoc phep khac hash so voi khoa R55 cha khi tao khoa R56 (chi mo ta chuyen tiep)
DUOC_THAY_KHOA_CHA = ("src/r55_tracking/dieu_phoi_test40.py", "src/r55_tracking/test_dieu_phoi_test40.py",
                      "configs/phase2/r55_tracking/hop_dong_dieu_phoi.json")
CONG_VIEC_VIDEO = ("tracking", "lap-tracker")
ENV_FIXTURE = "R55_FIXTURE"
ENV_NGAT = "R55_NGAT"
MARKER = "HOAN_TAT.json"
BUOC = ("tien-kiem", "khoa", "kiem-khoa", "thuc-thi")
CONG_VIEC = ("mo-phien", "cache", "xac-nhan-cache", "tracking", "cham", "lap-tracker", "cham-lai", "hau-kiem")
THU_TU = ["cache", "xac-nhan-cache", "tracking", "cham", "hau-kiem"]
KHOA_NPZ = {"frame_idx": "int32", "n_per_frame": "int32", "unreadable": "bool", "truncated": "bool",
            "time_s": "float64", "data": "float32", "orig_shape": "int32", "cls_goc": "int16"}
LOP_COCO_LOP5 = {1, 2, 3, 5, 7}
#: Module thuc thi (giong tap R54 khoa/kiem truoc suy luan) + boot_stats (cham). Nap o khoa VA o dau moi cong viec.
MODULE_THUC_THI = W.MODULE_SMOKE + ("boot_stats",) + W.MODULE_ULTRALYTICS + W.MODULE_ULTRALYTICS_SAU_PREDICTOR + \
    W.MODULE_TRACKEVAL
GIOI_HAN_TIEN_KIEM = W.GIOI_HAN_TIEN_KIEM + [
    "R55 KE THUA kiem anh cua R54: stat + size moi anh theo bang hash R47 va tu bam lai 256 anh mau co dinh; "
    "KHONG bam lai toan bo 56.340 anh o tien-kiem (cong viec cache bam tung anh truoc suy luan qua khung_mau R54).",
    "Du toan dung luong: kich thuoc file smoke R54 (stat, khong doc NPZ) x he so quy doi anh; kich ban, khong bao dam.",
]


class LoiChay(W.LoiR54):
    ma = 4


LoiDauVao, LoiGhiDe, LoiTienKiem = W.LoiDauVao, W.LoiGhiDe, W.LoiTienKiem
sha = W.sha256_file
rel = W.rel_goc
bay_gio = W.bay_gio


# ---------------------------------------------------------------------------
# Ngu canh: duong that / fixture
# ---------------------------------------------------------------------------
class Ctx:
    def __init__(self):
        self.hd = doc_json(HOP_DONG, "hop dong")
        t = os.environ.get(ENV_FIXTURE)
        self.fixture = bool(t)
        if os.environ.get(ENV_NGAT) and not self.fixture:
            raise LoiDauVao(f"{ENV_NGAT} chi dung trong fixture")
        if self.fixture:
            fx = Path(t).resolve()
            tests = VUNG_R55 / "tests"                 # goc LEXICAL (F5): khong resolve goc
            if fx.name != "fixture.json" or not trong_lexical(fx, tests) or fx.parent == tests or not fx.is_file():
                raise LoiDauVao(f"{ENV_FIXTURE}={t} phai la .../fixture.json ben trong {tests}")
            goc = fx.parent
            cfg = doc_json(fx, "fixture")
            self.fx_path = fx
            self.gt_path = W.kiem_trong_vung(goc / cfg["giao_thuc"], [goc])
            self.gt = doc_json(self.gt_path, "giao thuc fixture")
            self.nguon = W.Nguon(self.gt, W.kiem_trong_vung(goc / cfg["goc_nguon"], [goc]))
            self.mau = {"videos": tuple(cfg["mau"]["videos"]), "frame_dau": cfg["mau"]["frame_dau"],
                        "frame_cuoi": cfg["mau"]["frame_cuoi"]}
        else:
            goc = VUNG_R55
            self.fx_path = None
            self.gt_path = W.GIAO_THUC
            self.gt = W.doc_giao_thuc()
            self.nguon = W.Nguon(self.gt)
            self.mau = dict(W.MAU_SMOKE)
        self.goc = goc                                 # that: VUNG_R55 lexical; fixture: thu muc fixture.json
        self.vung = {k: goc / k for k in ("chuan_bi", "tests", "thuc_thi")}
        loi = kiem_hop_dong(self.hd, self.gt, self.fixture)
        if loi:
            raise LoiDauVao(f"hop dong / giao thuc lech: {loi[:6]}")
        self.runs = {r["run"]: r for r in W.ma_tran(self.gt)}
        self._videos = None

    @property
    def videos(self) -> list[str]:
        if self._videos is None:
            v = doc_json(self.nguon.manifest, "manifest")["videos"]
            if len(v) != self.gt["tap_du_lieu"]["so_video"] or len(set(v)) != len(v):
                raise LoiDauVao(f"manifest co {len(v)} video != {self.gt['tap_du_lieu']['so_video']}")
            self._videos = list(v)
        return self._videos

    def vung_ghi(self, k: str) -> Path:
        """Vung ghi k da kiem doc lap (F5) TRUOC khi tao phien / cau hinh thu vien."""
        return kiem_doc_lap(self.vung[k], self.goc)

    def mo_model(self, det: str):
        if self.fixture:
            import test_chuan_bi_test40 as T54       # mo hinh gia cua fixture R54 (tim hinh chu nhat trang)
            return T54.MoHinhGia(self.gt["detector"][det]["loai_checkpoint"])
        return W.mo_model_that(det, self.nguon.checkpoint[det])

    def doc_names(self):
        if self.fixture:
            import test_chuan_bi_test40 as T54
            return T54.names_gia({d: s["loai_checkpoint"] for d, s in self.gt["detector"].items()})
        return W.doc_names_checkpoint

    def tao_spec(self, det: str):
        ck = self.nguon.checkpoint[det]
        s = W.tao_spec(self.gt, det, ck, self.gt["detector"][det]["sha256"])
        if self.fixture:
            s.versions, s.created = {"fixture": "1"}, "fixture"
        else:
            s.fill_env()
        return s


def trong_lexical(p: Path, goc: Path) -> bool:
    """p (da resolve) nam trong goc LEXICAL (duong dan tuyet doi, KHONG resolve goc)."""
    try:
        Path(p).relative_to(Path(os.path.abspath(goc)))
        return True
    except ValueError:
        return False


def kiem_doc_lap(p: Path, goc: Path) -> Path:
    """F5: dich ghi phai nam trong goc vung ghi LEXICAL va resolve(p) == abspath(p), tuc khong thanh phan da ton
    tai nao (ke ca goc) la junction/symlink tro di cho khac. Khong dua vao W.tao_phien (no resolve ca goc lan dich)."""
    a = Path(os.path.abspath(p))
    r = Path(p).resolve()
    if r != a or not trong_lexical(a, goc):
        raise LoiDauVao(f"duong dan ghi {a} (resolve {r}) di qua junction/symlink hoac ngoai goc vung ghi {goc}")
    return a


def ghi_text(p: Path, s: str, vung: Path) -> None:
    """Ghi file van ban MOI trong vung (kiem doc lap + tu choi ghi de)."""
    p = kiem_doc_lap(p, vung)
    if p.exists():
        raise LoiGhiDe(f"{p} da ton tai - tu choi ghi de")
    p.write_text(s, encoding="utf-8")


def doc_json(p: Path, ten: str):
    if not Path(p).is_file():
        raise LoiDauVao(f"khong co {ten}: {p}")
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except ValueError as e:
        raise LoiDauVao(f"{ten} {p} khong doc duoc: {e}") from e


def kiem_hop_dong(hd: dict, gt: dict, fixture: bool) -> list[str]:
    """Hop dong phai tro dung giao thuc R54 + khoa R54 (hash) va dung 36 run / thu tu cong viec."""
    e = []
    g = hd.get("giao_thuc_r54", {})
    if (GOC / g.get("rel", "?")).resolve() != W.GIAO_THUC.resolve() or sha(W.GIAO_THUC) != g.get("sha256"):
        e.append("giao thuc R54 khac sha256 trong hop dong")
    k = hd.get("khoa_r54", {})
    kj = GOC / k.get("thu_muc", "?") / "khoa.json"
    if not kj.is_file() or sha(kj) != k.get("khoa_json_sha256"):
        e.append("khoa.json R54 khac sha256 trong hop dong")
    if hd.get("thu_tu_cong_viec") != THU_TU:
        e.append(f"thu_tu_cong_viec != {THU_TU}")
    e += [x for x in W.kiem_giao_thuc(gt) if not (fixture and x.startswith("tap_du_lieu"))]
    if not e:
        runs = [r["run"] for r in W.ma_tran(gt)]
        if hd.get("so_run") != W.SO_RUN or hd.get("danh_sach_run") != runs or len(set(runs)) != W.SO_RUN:
            e.append("danh_sach_run / so_run hop dong != ma tran giao thuc (36)")
    if not fixture and (hd.get("so_video"), gt["tap_du_lieu"]["so_video"]) != (W.SO_VIDEO, W.SO_VIDEO):
        e.append("duong that phai 40 video")
    kc = hd.get("khoa_cha_r55", {})
    if not kc.get("thu_muc") or not kc.get("khoa_json_sha256") or kc.get("duoc_thay") != list(DUOC_THAY_KHOA_CHA):
        e.append("khoa_cha_r55 trong hop dong thieu / duoc_thay khac 3 file R56")
    return e


# ---------------------------------------------------------------------------
# Marker / lan thu / ngat gia lap
# ---------------------------------------------------------------------------
_DEM_NGAT: dict[str, int] = {}


def ngat(ctx: Ctx, diem: str) -> bool:
    """Fixture: R55_NGAT='<diem>:<k>' -> lan thu k toi diem nay. Diem 'cham-loi' va '*-marker-cat' tra True (nguoi
    goi nem loi gia lap / ghi marker bi cat); diem khac: os._exit(9) gia lap tien trinh bi giet (khong don dep)."""
    v = os.environ.get(ENV_NGAT)
    if not ctx.fixture or not v:
        return False
    _DEM_NGAT[diem] = _DEM_NGAT.get(diem, 0) + 1
    if v != f"{diem}:{_DEM_NGAT[diem]}":
        return False
    if diem == "cham-loi" or diem.endswith("-marker-cat"):
        return True
    print(f"[ngat gia lap] {diem}:{_DEM_NGAT[diem]} -> exit 9", flush=True)
    os._exit(9)


def ghi_marker(ctx: Ctx, diem: str, d: Path, rang_buoc: dict, them: dict) -> str:
    """Marker GHI CUOI: rang buoc + sha256 moi file trong thu muc lan thu (tru marker, LOI.json).
    Fixture '<diem>-marker-cat': ghi NUA DAU byte marker roi os._exit(9) (gia lap ngat giua luc ghi marker)."""
    files = sorted(p for p in d.rglob("*") if p.is_file() and p.name not in (MARKER, "LOI.json")
                   and "_cau_hinh_thu_vien" not in p.parts)
    m = {"rang_buoc": rang_buoc, "file": {p.relative_to(d).as_posix(): sha(p) for p in files}, "luc": bay_gio(),
         **them}
    if ngat(ctx, f"{diem}-marker-cat"):
        b = (json.dumps(m, ensure_ascii=False, indent=1, default=str) + "\n").encode("utf-8")
        W.kiem_trong_vung(d / MARKER, [d]).write_bytes(b[: len(b) // 2])
        print(f"[ngat gia lap] {diem}-marker-cat -> marker cat {len(b) // 2}/{len(b)} byte, exit 9", flush=True)
        os._exit(9)
    return W.ghi_json(d / MARKER, m, [d])


def doc_marker(d: Path) -> dict | None:
    """Marker doc duoc (dict co rang_buoc) hoac None: khong co, hoac HONG (JSON cat / khong doc duoc). Chi DOC."""
    p = d / MARKER
    if not p.is_file():
        return None
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:                                   # gom UnicodeDecodeError
        return None
    return m if isinstance(m, dict) and isinstance(m.get("rang_buoc"), dict) else None


def danh_gia_lan(d: Path, rang_buoc: dict) -> tuple[str, str, dict | None]:
    """('hop_le'|'do_dang'|'hong'|'khong_khop'|'lech', ly do, marker). Chi DOC. 'hong' = marker co nhung khong doc
    duoc (ngat giua luc ghi marker): KHONG hoan tat, giu nguyen, nguoi goi bo qua va tim tiep."""
    p = d / MARKER
    if not p.is_file():
        return "do_dang", "khong co marker", None
    m = doc_marker(d)
    if m is None:
        return "hong", "marker khong doc duoc (cat / hong)", None
    a, b = W.chuan_json(m.get("rang_buoc")), W.chuan_json(rang_buoc)
    if a != b:
        khac = sorted(k for k in set(a or {}) | set(b) if (a or {}).get(k) != b.get(k))
        return "khong_khop", f"rang buoc khac: {khac}", m
    if not m.get("file"):
        return "lech", "marker khong liet ke file", m
    for f, s in m["file"].items():
        q = d / f
        if not W.trong(q, d):
            return "lech", f"{f} thoat thu muc lan thu", m
        if not q.is_file() or sha(q) != s:
            return "lech", f"{f} sha256 lech/thieu", m
    return "hop_le", "", m


def chon_lan(goc: Path, rang_buoc: dict) -> tuple[Path | None, dict | None, list[dict]]:
    """Lan thu hop le co ten nho nhat; cac lan khac chi ghi nhan (giu nguyen)."""
    ds, chon, mk = [], None, None
    if goc.is_dir():
        for d in sorted(x for x in goc.iterdir() if x.is_dir() and x.name.startswith("lan_")):
            tt, ly_do, m = danh_gia_lan(d, rang_buoc)
            ds.append({"lan": rel(d), "trang_thai": tt, "ly_do": ly_do})
            if tt == "hop_le" and chon is None:
                chon, mk = d, m
    return chon, mk, ds


_DEM_LAN = [0]


def ten_lan() -> str:
    """lan_<YYYYmmdd_HHMMSS>_<ns trong giay>_<dem>_pid<pid>: sap xep theo ten = thu tu tao (mot lan doc dong ho)."""
    t = time.time_ns()
    _DEM_LAN[0] += 1
    return (f"lan_{time.strftime('%Y%m%d_%H%M%S', time.localtime(t // 10**9))}_{t % 10**9:09d}_"
            f"{_DEM_LAN[0]:04d}_pid{os.getpid()}")


def lan_moi(goc: Path, vung: list[Path]) -> Path:
    """Thu muc lan thu MOI (ton tai -> LoiGhiDe); kiem doc lap voi vung[0] (F5) + sau resolve TRUOC khi tao."""
    kiem_doc_lap(goc, vung[0])
    W.kiem_trong_vung(goc, vung)
    return W.tao_phien(goc, ten_lan(), "lan")


# ---------------------------------------------------------------------------
# Khoa / moi truong
# ---------------------------------------------------------------------------
def nap_module_thuc_thi(gt: dict):
    import importlib
    W.kiem_env_backend(gt)
    for m in MODULE_THUC_THI:
        try:
            importlib.import_module(m)
        except ImportError as e:
            raise LoiDauVao(f"khong nap duoc module {m}: {e}") from e
    import run_tracking as rt
    rt.tracker_registry()


def kiem_khoa_r54(hd: dict) -> dict:
    k = GOC / hd["khoa_r54"]["thu_muc"]
    kq = W.kiem_khoa({"chuan_bi": k.parent}, k.name)          # chi doc
    if kq["khoa_sha256"] != hd["khoa_r54"]["khoa_json_sha256"]:
        kq["lech"].append("khoa.json R54 khac hop dong")
        kq["dat"] = False
    return kq


def kiem_khoa_r55(ctx: Ctx, ten: str | None) -> dict:
    if not ten:
        raise LoiDauVao("can --khoa <ten phien khoa R55> (khong tu chon khoa moi nhat)")
    if not W.TEN_HOP_LE.match(ten) or ".." in ten:
        raise LoiDauVao(f"ten khoa {ten!r} khong hop le")
    kq = W.kiem_khoa({"chuan_bi": ctx.vung["chuan_bi"]}, ten)  # chi doc; bam MOI file theo hash khoa nay
    k = kq["_khoa"]
    # R56: duong that chi nhan khoa R56 (co khoa cha); fixture nhan them khoa khong cha (mo phong khoa R55 cha)
    hop_le = ("R55_LOCK", "R56_LOCK") if ctx.fixture else ("R56_LOCK",)
    if k.get("buoc") not in hop_le or k.get("fixture") != ctx.fixture:
        kq["lech"].append(f"khoa {ten} ({k.get('buoc')}) khong phai khoa {'/'.join(hop_le)} cua che do "
                          f"{'fixture' if ctx.fixture else 'that'}")
    if k.get("hop_dong_sha256") != sha(HOP_DONG) or k.get("giao_thuc_sha256") != sha(ctx.gt_path):
        kq["lech"].append("hop dong / giao thuc khac khoa")
    if k.get("buoc") == "R56_LOCK":
        kq["lech"] += kiem_lien_ket_cha(ctx, k)
    kq["dat"] = not kq["lech"]
    return kq


def spec_khoa_cha_hop_dong(hd: dict) -> dict:
    """Khoa R55 cha theo hop dong (duong that): thu muc trong results/phase2/r55_tracking/chuan_bi/ + SHA256."""
    s = hd.get("khoa_cha_r55") or {}
    d = GOC / s.get("thu_muc", "?")
    if d.parent != VUNG_R55 / "chuan_bi" or not W.TEN_HOP_LE.match(d.name) or ".." in d.name:
        raise LoiDauVao(f"khoa_cha_r55.thu_muc {s.get('thu_muc')!r} khong nam trong {rel(VUNG_R55 / 'chuan_bi')}")
    return {"dir": d, "khoa_json_sha256": s.get("khoa_json_sha256")}


def spec_khoa_cha(ctx: Ctx) -> dict | None:
    """That: tu hop dong (bat buoc). Fixture: <thu muc fixture>/khoa_cha.json {"ten", "khoa_json_sha256"} neu co;
    khong co -> None (khoa fixture khong cha, mo phong khoa R55 cha)."""
    if not ctx.fixture:
        return spec_khoa_cha_hop_dong(ctx.hd)
    p = ctx.fx_path.parent / "khoa_cha.json"
    if not p.is_file():
        return None
    s = doc_json(p, "khoa_cha.json fixture")
    ten = str(s.get("ten", ""))
    if not W.TEN_HOP_LE.match(ten) or ".." in ten:
        raise LoiDauVao(f"khoa_cha.json fixture: ten {ten!r} khong hop le")
    return {"dir": ctx.vung["chuan_bi"] / ten, "khoa_json_sha256": s.get("khoa_json_sha256")}


def doc_khoa_cha(spec: dict, fixture: bool, hd: dict) -> tuple[dict | None, str | None, list[str]]:
    """khoa.json cha: (noi dung, sha256, lech nguyen ven). Chi DOC."""
    d = spec["dir"]
    kj = d / "khoa.json"
    if not kj.is_file():
        return None, None, [f"khong co khoa cha {rel(d)}/khoa.json"]
    lech = []
    s = sha(kj)
    if s != spec.get("khoa_json_sha256"):
        lech.append(f"khoa cha khoa.json {s} != {spec.get('khoa_json_sha256')} (hop dong / dac ta)")
    sums = (d / "SHA256SUMS").read_text(encoding="utf-8").split() if (d / "SHA256SUMS").is_file() else []
    if sums[:2] != [s, "khoa.json"]:
        lech.append("khoa cha: khoa.json khac SHA256SUMS")
    try:
        k = json.loads(kj.read_text(encoding="utf-8"))
    except ValueError as e:
        return None, s, lech + [f"khoa cha khong doc duoc: {e}"]
    if k.get("buoc") != "R55_LOCK" or k.get("fixture") != fixture:
        lech.append(f"khoa cha {k.get('buoc')} / fixture={k.get('fixture')} khong phai khoa R55 cua che do nay")
    if k.get("khoa_r54", {}).get("khoa_json_sha256") != hd["khoa_r54"]["khoa_json_sha256"]:
        lech.append("khoa cha noi toi khoa R54 khac hop dong")
    rels = {f.get("rel") for f in k.get("file", [])}
    lech += [f"khoa cha khong chua file duoc thay {r}" for r in DUOC_THAY_KHOA_CHA if r not in rels]
    return k, s, lech


def kiem_khoa_cha(spec: dict, fixture: bool, hd: dict) -> dict:
    """Nguyen ven khoa R55 cha (CHI DOC) + bam lai MOI file cua khoa cha: khac hash cha chi duoc o DUOC_THAY_KHOA_CHA
    ('thay'), con lai phai trung ('giu'), khac la 'lech'. Ngoai le 3 file CHI dung de tao khoa R56 (mo ta chuyen
    tiep), khong dung o kiem khoa thuc thi."""
    k, s, lech = doc_khoa_cha(spec, fixture, hd)
    bang = []
    for f in (k or {}).get("file", []):
        p = W.tuyet_doi(f["rel"])
        h = sha(p) if p.is_file() else "THIEU"
        tt = "giu" if h == f["sha256"] else ("thay" if f["rel"] in DUOC_THAY_KHOA_CHA and h != "THIEU" else "lech")
        if tt == "lech":
            lech.append(f"{f['rel']}: hien tai {h} != cha {f['sha256']} (ngoai 3 file duoc thay)")
        bang.append({"rel": f["rel"], "vai_tro": f["vai_tro"], "sha256_cha": f["sha256"], "sha256_hien_tai": h,
                     "trang_thai": tt})
    return {"khoa": rel(spec["dir"]), "khoa_sha256": s, "so_file": len(bang), "lech": lech, "bang": bang,
            "_khoa": k, "_dir": spec["dir"]}


def khac_biet_khoa_cha(cha: dict, files: list[dict]) -> tuple[dict, list[str]]:
    """Bang khac biet cu/moi giua file khoa cha va file khoa moi: tom tat {so_giu, thay, moi} + lech (file cha thieu
    trong khoa moi / khac hash ngoai 3 file duoc thay)."""
    moi = {f["rel"]: f for f in files}
    giu, thay, lech = 0, [], []
    for f in cha.get("file", []):
        m = moi.get(f["rel"])
        if m is None:
            lech.append(f"{f['rel']}: co trong khoa cha, thieu trong khoa moi")
        elif m["sha256"] == f["sha256"]:
            giu += 1
        elif f["rel"] in DUOC_THAY_KHOA_CHA:
            thay.append({"rel": f["rel"], "sha256_cha": f["sha256"], "sha256_moi": m["sha256"]})
        else:
            lech.append(f"{f['rel']}: khoa moi {m['sha256']} != cha {f['sha256']} (ngoai 3 file duoc thay)")
    cha_rels = {f["rel"] for f in cha.get("file", [])}
    them = [{"rel": r, "vai_tro": m["vai_tro"], "sha256_moi": m["sha256"]} for r, m in moi.items() if r not in cha_rels]
    return {"so_giu": giu, "thay": thay, "moi": them}, lech


def kiem_lien_ket_cha(ctx: Ctx, k: dict) -> list[str]:
    """Khoa R56: khoa cha ghi trong khoa = dac ta (hop dong / fixture); khoa.json cha nguyen ven; bang khac biet tinh
    lai tu khoa cha + danh sach file khoa moi trung ban ghi, chi 3 file duoc thay. Moi file cua khoa moi (ke ca 3 file
    'thay' va file ke thua tu khoa cha) da duoc W.kiem_khoa bam theo hash KHOA MOI - khong bo qua hash nao."""
    kc = k.get("khoa_cha_r55") or {}
    try:
        spec = spec_khoa_cha(ctx)
    except LoiDauVao as e:
        return [f"khoa cha: {e}"]
    if spec is None:
        return ["khoa R56 nhung khong co dac ta khoa cha"]
    lech = []
    if (kc.get("thu_muc"), kc.get("khoa_json_sha256")) != (rel(spec["dir"]), spec["khoa_json_sha256"]):
        lech.append(f"khoa cha trong khoa ({kc.get('thu_muc')}, {kc.get('khoa_json_sha256')}) khac dac ta")
    if kc.get("duoc_thay") != list(DUOC_THAY_KHOA_CHA):
        lech.append("duoc_thay trong khoa khac 3 file R56")
    cha, _, l0 = doc_khoa_cha({"dir": spec["dir"], "khoa_json_sha256": kc.get("khoa_json_sha256")}, ctx.fixture,
                              ctx.hd)
    lech += [f"khoa cha: {x}" for x in l0]
    if cha is not None:
        tom, l2 = khac_biet_khoa_cha(cha, k.get("file", []))
        lech += [f"khoa cha: {x}" for x in l2]
        if W.chuan_json(tom) != W.chuan_json(k.get("khac_biet_khoa_cha")):
            lech.append("bang khac biet khoa cha tinh lai khac ban ghi trong khoa")
    return lech


def kiem_moi_truong(ctx: Ctx, khoa: dict, bam: bool = True) -> list[str]:
    """Truoc moi cong viec (va truoc suy luan): module src/Ultralytics/TrackEval dang nap nam trong khoa (+ bam
    khop), backend, cau hinh tracker hieu luc, phien ban thu vien = khoa."""
    nap_module_thuc_thi(ctx.gt)
    da_khoa = {W.tuyet_doi(f["rel"]): f["sha256"] for f in khoa["file"]}
    nap = W.file_module_da_nap()
    lech = [f"module nap ngoai khoa: {rel(p)}" for p in nap if p not in da_khoa]
    if bam:
        lech += [f"{rel(p)}: sha256 khac khoa" for p in nap if p in da_khoa and sha(p) != da_khoa[p]]
    if W.chuan_json(W.thong_tin_backend(ctx.gt)) != W.chuan_json(khoa.get("backend_tracker")):
        lech.append("backend tracker khac khoa")
    cfg, loi = W.cau_hinh_hieu_luc(ctx.gt)
    lech += loi
    if W.chuan_json(cfg) != W.chuan_json(khoa.get("cau_hinh_hieu_luc")):
        lech.append("cau hinh tracker hieu luc khac khoa")
    if W.phien_ban_thu_vien() != khoa.get("thu_vien"):
        lech.append("phien ban thu vien khac khoa")
    return lech


_RESOLVE_FILE: dict[str, Path] = {}
_DA_KHOA: dict[int, tuple[dict, dict]] = {}


def file_module_dang_nap() -> list[Path]:
    """Cung tap file nhu W.file_module_da_nap (file nguon da resolve cua moi module dang nap thuoc src/, goi
    ultralytics, goi trackeval) nhung nho ket qua resolve theo chuoi __file__ (resolve tren Windows cham; goi sau
    MOI don vi). Khong nho sha256: kiem_module_sau van bam lai tung file moi lan."""
    goc = [SRC.resolve()] + [Path(sys.modules[g].__file__).resolve().parent for g in ("ultralytics", "trackeval")
                             if getattr(sys.modules.get(g), "__file__", None)]
    out = set()
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        p = _RESOLVE_FILE.get(f)
        if p is None:
            p = _RESOLVE_FILE[f] = Path(f).resolve()
        if any(p == g or g in p.parents for g in goc):
            out.add(p)
    return sorted(out)


def kiem_module_sau(khoa: dict) -> list[str]:
    """F4: SAU moi don vi, TRUOC marker / ket luan thanh cong: MOI module src/Ultralytics/TrackEval dang nap phai
    nam trong khoa VA bam lai khop khoa (khong chi su hien dien). Rong = dat."""
    k = _DA_KHOA.get(id(khoa))
    if k is None or k[0] is not khoa:                  # anh xa duong dan khoa (resolve) nho theo doi tuong khoa
        k = _DA_KHOA[id(khoa)] = (khoa, {W.tuyet_doi(f["rel"]): f["sha256"] for f in khoa["file"]})
    da_khoa = k[1]
    nap = file_module_dang_nap()
    return [f"module nap ngoai khoa: {rel(p)}" for p in nap if p not in da_khoa] + \
        [f"{rel(p)}: sha256 khac khoa" for p in nap if p in da_khoa and sha(p) != da_khoa[p]]


# ---------------------------------------------------------------------------
# Tien kiem / khoa / kiem-khoa (duong chuan bi)
# ---------------------------------------------------------------------------
def du_toan_dia(ctx: Ctx) -> dict:
    """Kich ban dung luong tu KICH THUOC file smoke R54 (stat, khong doc NPZ)."""
    if ctx.fixture:
        return {"bo_qua": "fixture"}
    sm = GOC / "results/phase2/r54_tracking/smoke/lan_20261021_002054_245100_pid16684"
    he_so = ctx.gt["tap_du_lieu"]["so_anh"] / (2 * 128)
    kt = {d: sum(p.stat().st_size for p in (sm / d).rglob("*") if p.is_file()) if (sm / d).is_dir() else None
          for d in ("cache", "trackers", "trackeval")}
    if any(v is None for v in kt.values()):
        return {"loi": f"thieu thu muc smoke {rel(sm)}", "kich_thuoc_smoke": kt}
    can = int((kt["cache"] + 2 * kt["trackers"] + 2 * kt["trackeval"]) * he_so)   # cham sao MOT -> x2
    trong_dia = shutil.disk_usage(GOC).free
    return {"loai": "kich ban theo gia dinh, KHONG bao dam", "smoke": rel(sm), "kich_thuoc_smoke": kt,
            "he_so_quy_doi_anh": round(he_so, 2), "uoc_can_byte": can, "trong_byte": trong_dia,
            "dat": trong_dia > 2 * can,
            "gia_dinh": ["kich thuoc tuyen tinh theo so anh", "128 frame dau dai dien mat do", "cham sao MOT (x2)",
                         "khong tinh lap-tracker / cham-lai / hau-kiem"]}


def buoc_tien_kiem(ctx: Ctx, args) -> int:
    phien = W.tao_phien(ctx.vung_ghi("chuan_bi"), None, "tien-kiem")
    W.dat_moi_truong(phien)
    t0 = time.time()
    lech = []
    k54 = kiem_khoa_r54(ctx.hd)
    lech += [f"khoa R54: {x}" for x in k54["lech"]]
    kq = W.tien_kiem(ctx.gt, ctx.nguon, doc_names=ctx.doc_names(), mau=ctx.mau)
    lech += kq["lech"]
    k = k54["_khoa"]
    if W.chuan_json(kq["cau_hinh_hieu_luc"]) != W.chuan_json(k.get("cau_hinh_hieu_luc")):
        lech.append("cau hinh tracker hieu luc khac khoa R54")
    tv = W.phien_ban_thu_vien()
    if tv != k.get("thu_vien"):
        lech.append(f"phien ban thu vien khac khoa R54: {tv}")
    try:
        nap_module_thuc_thi(ctx.gt)
        bk = W.thong_tin_backend(ctx.gt)
        if W.chuan_json(bk) != W.chuan_json(k.get("backend_tracker")):
            lech.append("backend tracker khac khoa R54")
    except W.LoiR54 as e:
        bk = None
        lech.append(f"backend: {e}")
    try:
        tt = ctx.vung_ghi("thuc_thi")
        if tt.exists() and not tt.is_dir():
            lech.append(f"vung thuc thi {tt} khong phai thu muc")
    except LoiDauVao as e:
        lech.append(f"vung thuc thi: {e}")
    dia = du_toan_dia(ctx)
    if not ctx.fixture and not dia.get("dat"):
        lech.append(f"dung luong dia: {dia}")
    if ctx.fixture:
        doan = {"bo_qua": "fixture (video doan co dinh khong co trong du lieu tong hop)"}
    else:
        doan = {**ctx.hd["doan_day"]["doan_co_dinh"], "lech": kiem_doan_co_dinh(ctx.hd, kq["bang"])}
        lech += doan["lech"]
    kq["doan_co_dinh"] = doan
    v = [phien]
    sha_bang = W.ghi_bang_csv(phien / "bang_tien_kiem.csv", kq.pop("bang"), v)
    sha_mau = W.ghi_json(phien / "mau_anh.json", kq["mau_smoke"] | {"anh": kq.pop("anh_tu_bam_lai")}, v)
    ma = 0 if not lech else LoiTienKiem.ma
    kq.update({"buoc": "tien-kiem", "r55": True, "fixture": ctx.fixture, "phien": rel(phien), "luc": bay_gio(),
               "lech": lech, "ket_luan": "dat" if not lech else "khong_dat",
               "khoa_r54": {k_: v_ for k_, v_ in k54.items() if not k_.startswith("_")},
               "thoi_gian_s": round(time.time() - t0, 1), "backend_tracker": bk, "thu_vien": tv,
               "ma_r55_sha256": ma_r55_sha(ctx), "giao_thuc_sha256": sha(ctx.gt_path),
               "backend_env": os.environ.get("BYTETRACK_BACKEND"), "dung_luong": dia,
               "bang_tien_kiem_sha256": sha_bang, "mau_smoke_sha256": sha_mau, "gioi_han": GIOI_HAN_TIEN_KIEM,
               "exit": ma})
    W.ghi_json(phien / "tien_kiem.json", kq, v)
    print(f"[tien-kiem] {kq['ket_luan']} ({len(lech)} lech) -> {rel(phien)}  exit {ma}")
    for x in lech[:30]:
        print("   -", x)
    return ma


def kiem_doan_co_dinh(hd: dict, rows: list[dict]) -> list[str]:
    """R56: doan do co dinh trong hop dong phai la TOAN BO mot video cua bang tien kiem (frame 1..seqLength) va so
    dong GT can cu khop bang. Chi doi chieu so lieu da co; KHONG do gi."""
    d = hd.get("doan_day", {}).get("doan_co_dinh", {})
    r = [x for x in rows if x.get("video") == d.get("video")]
    if len(r) != 1:
        return [f"doan co dinh: video {d.get('video')!r} co {len(r)} dong trong bang tien kiem (can 1)"]
    try:
        so = {k: int(r[0][k]) for k in ("so_anh", "seq_length", "frame_dau", "frame_cuoi", "so_dong_gt")}
    except (KeyError, TypeError, ValueError) as e:
        return [f"doan co dinh: bang tien kiem thieu / sai cot {e}"]
    loi = []
    if not (d.get("frame_dau") == so["frame_dau"] == 1 and d.get("frame_cuoi") == so["frame_cuoi"] == so["seq_length"]
            == so["so_anh"]) or d.get("toan_video") is not True:
        loi.append(f"doan co dinh {d.get('video')} frame {d.get('frame_dau')}-{d.get('frame_cuoi')} khong phai toan "
                   f"video (bang: {so})")
    if d.get("so_dong_gt_can_cu") != so["so_dong_gt"]:
        loi.append(f"doan co dinh: so dong GT can cu {d.get('so_dong_gt_can_cu')} != bang {so['so_dong_gt']}")
    return loi


def ma_r55_sha(ctx: Ctx) -> dict:
    return {rel(p): sha(p) for p in (Path(__file__).resolve(), FILE_TEST, HOP_DONG)} | (
        {rel(ctx.fx_path): sha(ctx.fx_path), rel(ctx.gt_path): sha(ctx.gt_path)} if ctx.fixture else {})


def buoc_khoa(ctx: Ctx, args) -> int:
    cb = ctx.vung_ghi("chuan_bi")
    if not args.tien_kiem or not args.tests:
        raise LoiDauVao("khoa can --tien-kiem <phien> va --tests <phien> TUONG MINH")
    for t in (args.tien_kiem, args.tests):
        if not W.TEN_HOP_LE.match(t) or ".." in t:
            raise LoiDauVao(f"ten phien {t!r} khong hop le")
    tk = ctx.vung["chuan_bi"] / args.tien_kiem
    tkj = doc_json(tk / "tien_kiem.json", "tien_kiem.json")
    if not (tkj.get("r55") and tkj.get("ket_luan") == "dat" and tkj.get("exit") == 0 and
            tkj.get("fixture") == ctx.fixture):
        raise LoiDauVao(f"tien-kiem {tk.name} khong dat / khong phai R55")
    tj_p = ctx.vung["tests"] / args.tests / "ket_qua_test.json"
    tj = doc_json(tj_p, "ket_qua_test.json")
    ht = ma_r55_sha(ctx)
    loi = []
    if not tj.get("tat_ca_dat") or tj.get("exit") != 0:
        loi.append("fixture chua dat")
    for k_, p in (("dieu_phoi_sha256", Path(__file__)), ("test_sha256", FILE_TEST), ("hop_dong_sha256", HOP_DONG)):
        if tj.get(k_) != sha(p):
            loi.append(f"ket qua test {k_} != ma hien tai")
    if not THIET_KE.is_file():
        loi.append("thieu tai lieu thiet ke")
    loi += W.doi_chieu_tien_kiem(tkj, W.dau_vao_sha(ctx.gt, ctx.nguon), ("dau_vao_sha256",), None)
    loi += [f"{r}: doi sau tien-kiem" for r, s in tkj.get("ma_r55_sha256", {}).items() if ht.get(r) != s]
    if set(tkj.get("ma_r55_sha256", {})) != set(ht):
        loi.append("tap ma R55 khac tien-kiem")
    for f, k_ in (("bang_tien_kiem.csv", "bang_tien_kiem_sha256"), ("mau_anh.json", "mau_smoke_sha256")):
        if not (tk / f).is_file() or sha(tk / f) != tkj.get(k_):
            loi.append(f"{f} khac tien_kiem.json")
    k54 = kiem_khoa_r54(ctx.hd)
    loi += [f"khoa R54: {x}" for x in k54["lech"]]
    spec = spec_khoa_cha(ctx)                          # that: bat buoc (hop dong); fixture: khoa_cha.json neu co
    kc = None
    if spec is not None:
        kc = kiem_khoa_cha(spec, ctx.fixture, ctx.hd)
        loi += [f"khoa cha: {x}" for x in kc["lech"]]
        if not THIET_KE_BO_SUNG.is_file():
            loi.append(f"thieu tai lieu bo sung {rel(THIET_KE_BO_SUNG)}")
    if loi:
        raise LoiDauVao(f"khong khoa ({len(loi)} lech): {loi[:8]}")
    phien = W.tao_phien(kiem_doc_lap(cb, ctx.goc), None, "khoa")
    W.dat_moi_truong(phien)
    if ctx.fixture:
        ctx.mo_model(next(iter(ctx.gt["detector"])))       # nap module mo hinh gia vao tap khoa
    nap_module_thuc_thi(ctx.gt)
    cfg, loi_cfg = W.cau_hinh_hieu_luc(ctx.gt)
    bk = W.thong_tin_backend(ctx.gt)
    if loi_cfg or W.chuan_json(cfg) != W.chuan_json(tkj.get("cau_hinh_hieu_luc")):
        raise LoiDauVao(f"cau hinh hieu luc khac tien-kiem: {loi_cfg}")
    items: list[tuple[Path, str]] = []
    items += [(W.tuyet_doi(f["rel"]), f"khoa_r54:{f['vai_tro']}") for f in k54["_khoa"]["file"]]
    items += [(Path(__file__).resolve(), "dieu_phoi_r55"), (FILE_TEST, "test_r55"), (HOP_DONG, "hop_dong_r55"),
              (THIET_KE, "thiet_ke_r55"), (tj_p, "ket_qua_test_r55"), (tk / "tien_kiem.json", "tien_kiem_r55"),
              (tk / "bang_tien_kiem.csv", "tien_kiem_r55"), (tk / "mau_anh.json", "tien_kiem_r55"),
              (W.GIAO_THUC, "giao_thuc_r54"), (ctx.gt_path, "giao_thuc")]
    if ctx.fixture:
        items.append((ctx.fx_path, "fixture"))
    items += [(W.tuyet_doi(r_), "dau_vao") for r_ in tkj["dau_vao_sha256"]]
    items += [(Path(r_["yaml_resolve"]), "yaml_tracker") for r_ in cfg["bo_loc"].values()]
    goc_ngoai = W.goc_ma_ngoai()
    items += [(p, W.vai_tro_ma(p, goc_ngoai)) for p in W.file_module_da_nap()]
    items += [(p, f"thu_vien_record:{pk}") for pk, p in W.file_record_thu_vien().items()]
    if kc is not None:                                 # R56: tai lieu bo sung + khoa cha + MOI file khoa cha
        items += [(THIET_KE_BO_SUNG, "bo_sung_r56")]
        items += [(kc["_dir"] / f, "khoa_cha_r55") for f in ("khoa.json", "SHA256SUMS", "thoi_diem_khoa.json")]
        items += [(W.tuyet_doi(f["rel"]), f"khoa_cha_r55:{f['vai_tro']}") for f in kc["_khoa"]["file"]]
    seen, files = {}, []
    for p, vai in items:
        p = Path(p).resolve()
        if p in seen:
            continue
        if not p.is_file():
            raise LoiDauVao(f"file can khoa khong ton tai: {p}")
        seen[p] = sha(p)
        files.append({"rel": rel(p), "vai_tro": vai, "size": p.stat().st_size, "sha256": seen[p]})
    lech54 = [f["rel"] for f in k54["_khoa"]["file"] if seen[W.tuyet_doi(f["rel"])] != f["sha256"]]
    lech_tk = [r_ for r_, s in tkj["dau_vao_sha256"].items() if seen[W.tuyet_doi(r_)] != s]
    if lech54 or lech_tk:
        raise LoiDauVao(f"doi trong luc khoa: R54 {lech54[:5]} / dau vao {lech_tk[:5]}")
    cha = {}
    if kc is not None:
        tom, lech_c = khac_biet_khoa_cha(kc["_khoa"], files)
        if lech_c or sha(kc["_dir"] / "khoa.json") != kc["khoa_sha256"]:
            raise LoiDauVao(f"doi trong luc khoa: khoa cha {lech_c[:5]}")
        cha = {"khoa_cha_r55": {"thu_muc": rel(kc["_dir"]), "khoa_json_sha256": kc["khoa_sha256"],
                                "so_file": kc["so_file"], "duoc_thay": list(DUOC_THAY_KHOA_CHA)},
               "khac_biet_khoa_cha": tom, "tai_lieu_bo_sung": rel(THIET_KE_BO_SUNG),
               "ghi_chu_khoa_cha": "khoa cha la bang chung lich su bat bien; ngoai le 3 file chi mo ta chuyen tiep "
                                   "luc tao khoa nay; kiem khoa nay bam MOI file theo hash khoa nay, khong ngoai le"}
    khoa = {"phien_ban": 2 if kc is not None else 1, "buoc": "R56_LOCK" if kc is not None else "R55_LOCK",
            "fixture": ctx.fixture, "tao_luc": bay_gio(), "phien": rel(phien), **cha,
            "tien_kiem": rel(tk), "ket_qua_test": rel(tj_p), "so_test": tj.get("so_test"),
            "khoa_r54": {"thu_muc": ctx.hd["khoa_r54"]["thu_muc"], "khoa_json_sha256": k54["khoa_sha256"],
                         "so_file": k54["so_file"]},
            "giao_thuc_r54_sha256": sha(W.GIAO_THUC), "giao_thuc": rel(ctx.gt_path),
            "giao_thuc_sha256": sha(ctx.gt_path), "hop_dong_sha256": sha(HOP_DONG),
            "ma_tran": list(ctx.runs), "thu_tu_cong_viec": THU_TU, "backend_tracker": bk, "cau_hinh_hieu_luc": cfg,
            "thu_vien": W.phien_ban_thu_vien(), "goc_ma_ngoai": {k_: str(v) for k_, v in goc_ngoai.items()},
            "so_file": len(files), "file": files,
            "gioi_han_import_luoi": "module Ultralytics nap luoi trong setup_model/stream_inference chi co trong khoa "
                                    "qua khoa R54 (ra soat tinh + smoke R54B); moi cong viec kiem lai module nap "
                                    "TRUOC va SAU tung don vi (ngoai khoa -> khong ghi marker, exit 4)",
            "ghi_chu": "khoa R55 noi toi khoa/giao thuc R54 (hash); sua sau khoa -> phien khoa moi, giu phien cu"}
    s = W.ghi_json(phien / "khoa.json", khoa, [phien])
    ghi_text(phien / "SHA256SUMS", f"{s}  khoa.json\n", phien)
    W.ghi_json(phien / "thoi_diem_khoa.json", {"tao_luc": khoa["tao_luc"], "khoa_sha256": s,
                                               "time_ns": time.time_ns()}, [phien])
    print(f"[khoa] {khoa['buoc']} {len(files)} file -> {rel(phien)}  khoa.json {s}")
    if kc is not None:
        t = khoa["khac_biet_khoa_cha"]
        print(f"   khoa cha {rel(kc['_dir'])} ({kc['khoa_sha256']}): giu {t['so_giu']}, thay "
              f"{[x['rel'] for x in t['thay']]}, moi {len(t['moi'])}")
    return 0


def buoc_kiem_khoa(ctx: Ctx, args) -> int:
    if args.luu:
        if args.luu != "chuan_bi":
            raise LoiDauVao("--luu chi nhan chuan_bi")
        ctx.vung_ghi("chuan_bi")                        # F5: kiem vung ghi TRUOC khi doc / ghi
    kq = kiem_khoa_r55(ctx, args.khoa)
    k54 = kiem_khoa_r54(ctx.hd)
    lech = kq["lech"] + [f"khoa R54: {x}" for x in k54["lech"]]
    ma = 0 if not lech else 2
    kc = kq["_khoa"].get("khoa_cha_r55")
    print(f"[kiem-khoa] {kq['khoa']} ({kq['_khoa'].get('buoc')}, khoa.json {kq['khoa_sha256']}): {kq['so_file']} "
          f"file; khoa R54 {k54['so_file']} file" + (f"; khoa cha {kc['thu_muc']} ({kc['khoa_json_sha256']})" if kc
                                                      else "") + f"; {len(lech)} lech -> exit {ma}")
    for x in lech[:20]:
        print("   -", x)
    if args.luu:
        phien = W.tao_phien(ctx.vung_ghi("chuan_bi"), None, "kiem-khoa")
        W.ghi_json(phien / "ket_qua.json", {k_: v for k_, v in kq.items() if not k_.startswith("_")} |
                   {"khoa_r54": {k_: v for k_, v in k54.items() if not k_.startswith("_")}, "lech": lech, "exit": ma},
                   [phien])
        print(f"   ket qua -> {rel(phien)}/ket_qua.json")
    return ma


# ---------------------------------------------------------------------------
# Thuc thi: phien
# ---------------------------------------------------------------------------
class Phien:
    def __init__(self, ctx: Ctx, ten: str):
        if not ten or not W.TEN_HOP_LE.match(ten) or ".." in ten:
            raise LoiDauVao(f"--phien {ten!r} khong hop le / thieu")
        self.ctx = ctx
        self.dir = kiem_doc_lap(ctx.vung_ghi("thuc_thi") / ten, ctx.goc)
        W.kiem_trong_vung(self.dir, [ctx.vung["thuc_thi"]])
        self.pj = doc_json(self.dir / "phien.json", "phien.json")
        self.vung = [self.dir]
        self.khoa = None

    def xac_minh(self) -> dict:
        kq = kiem_khoa_r55(self.ctx, self.pj.get("khoa"))
        if kq["khoa_sha256"] != self.pj.get("khoa_sha256"):
            kq["lech"].append("khoa.json khac khoa da gan voi phien")
        if self.pj.get("hop_dong_sha256") != sha(HOP_DONG) or self.pj.get("giao_thuc_sha256") != sha(self.ctx.gt_path):
            kq["lech"].append("hop dong / giao thuc khac phien")
        if kq["lech"]:
            raise LoiDauVao(f"kiem-khoa / rang buoc phien lech: {kq['lech'][:5]}")
        self.khoa = kq["_khoa"]
        self.file_khoa = {W.tuyet_doi(f["rel"]): f["sha256"] for f in self.khoa["file"]}
        return kq

    @property
    def co_so(self) -> dict:
        return {"khoa": self.pj["khoa"], "khoa_sha256": self.pj["khoa_sha256"],
                "hop_dong_sha256": self.pj["hop_dong_sha256"], "giao_thuc_sha256": self.pj["giao_thuc_sha256"]}


def cong_viec_mo_phien(ctx: Ctx, args) -> dict:
    tt = ctx.vung_ghi("thuc_thi")                       # F5: TRUOC moi doc / ghi
    kq = kiem_khoa_r55(ctx, args.khoa)
    k54 = kiem_khoa_r54(ctx.hd)
    if not kq["dat"] or not k54["dat"]:
        raise LoiDauVao(f"kiem-khoa lech: {(kq['lech'] + k54['lech'])[:5]}")
    d = W.tao_phien(kiem_doc_lap(tt, ctx.goc), args.phien, "phien")
    pj = {"khoa": Path(kq["khoa"]).name, "khoa_sha256": kq["khoa_sha256"], "hop_dong_sha256": sha(HOP_DONG),
          "giao_thuc_sha256": sha(ctx.gt_path), "fixture": ctx.fixture, "duyet": args.duyet, "tao_luc": bay_gio(),
          "so_video": len(ctx.videos), "ma_tran": list(ctx.runs), "thu_tu_cong_viec": THU_TU}
    W.ghi_json(d / "phien.json", pj, [d])
    return {"phien": rel(d), "khoa": pj["khoa"]}


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def rang_buoc_cache(ph: Phien, det: str, spec_sha: str, v: str, bang_v: list[dict]) -> dict:
    g = ph.ctx.gt
    return {"loai": "cache", **ph.co_so, "detector": det, "checkpoint_sha256": g["detector"][det]["sha256"],
            "loai_checkpoint": g["detector"][det]["loai_checkpoint"], "spec_sha256": spec_sha, "video": v,
            "so_anh": len(bang_v), "anh_sha256_ds": W.sha256_ds((r["rel"], r["sha256"]) for r in bang_v),
            "cache_cfg": g["cache"]}


def rang_buoc_xac_nhan(ph: Phien, det: str) -> dict:
    return {"loai": "xac-nhan-cache", **ph.co_so, "detector": det}


def manifest_cache(ph: Phien, det: str, nk: dict | None = None) -> tuple[Path | None, dict | None, list[str]]:
    """Manifest cache BAT BIEN cua detector trong phien. (thu muc, noi dung, loi).
    F1: chi lan co HOAN_TAT.json (ghi CUOI, sau manifest_cache.json + SHA256SUMS) hop le moi la DA CONG BO.
    Lan khong marker / marker hong = chua cong bo: giu nguyen, bo qua. Marker doc duoc nhung lech (hash /
    rang buoc) -> loi (chan). >1 ban cong bo -> loi."""
    goc = ph.dir / "xac_nhan_cache" / det
    ds = sorted(d for d in goc.iterdir() if d.is_dir()) if goc.is_dir() else []
    ok, loi = [], []
    for x in ds:
        tt, ly_do, _ = danh_gia_lan(x, rang_buoc_xac_nhan(ph, det))
        if nk is not None:
            nk["lan_thu"].append({"lan": rel(x), "trang_thai": tt, "ly_do": ly_do})
        if tt == "hop_le":
            ok.append(x)
        elif tt not in ("do_dang", "hong"):
            loi.append(f"{det}: manifest {rel(x)} {tt}: {ly_do}")
    if loi:
        return None, None, loi
    if not ok:
        return None, None, []
    if len(ok) > 1:
        return None, None, [f"{det}: {len(ok)} manifest cache da cong bo - chi duoc mot"]
    d = ok[0]
    s = sha(d / "manifest_cache.json")
    sums = (d / "SHA256SUMS").read_text(encoding="utf-8").split() if (d / "SHA256SUMS").is_file() else []
    if sums[:2] != [s, "manifest_cache.json"]:
        return d, None, [f"{det}: manifest_cache.json sha256 khac SHA256SUMS"]
    m = json.loads((d / "manifest_cache.json").read_text(encoding="utf-8"))
    m["_sha256"] = s
    return d, m, []


def manifest_chua_cong_bo(ph: Phien, det: str) -> list[tuple[Path, dict | None]]:
    """F1 (vong 3): lan xac nhan CHUA/KHONG CON hop le nhung da co manifest_cache.json hoac marker. Marker hong sau
    cong bo khong phan biet duoc voi ngat giua luc ghi marker -> coi la CO THE DA CONG BO: rang buoc cache cua no
    duoc giu khi phuc hoi (cache dong bang, xac nhan lai phai tro dung cac lan cache do). (thu muc, manifest doc duoc
    hoac None). Chi DOC."""
    goc = ph.dir / "xac_nhan_cache" / det
    out = []
    for x in sorted(d for d in goc.iterdir() if d.is_dir()) if goc.is_dir() else []:
        if danh_gia_lan(x, rang_buoc_xac_nhan(ph, det))[0] not in ("do_dang", "hong"):
            continue
        if not ((x / "manifest_cache.json").is_file() or (x / MARKER).is_file()):
            continue
        try:
            m = json.loads((x / "manifest_cache.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):                   # manifest chua ghi xong / khong co
            m = None
        out.append((x, m if isinstance(m, dict) else None))
    return out


def kiem_cache_theo_manifest(ph: Phien, det: str) -> dict:
    """Bam lai MOI npz/json/meta trong manifest; lech -> LoiDauVao (truoc moi tracking cua detector)."""
    d, m, loi = manifest_cache(ph, det)
    if m is None:
        raise LoiDauVao(f"chua co manifest cache hop le cho {det}: {loi or 'chua xac nhan'}")
    if W.chuan_json(m.get("rang_buoc")) != W.chuan_json(ph.co_so) or m.get("detector") != det:
        loi.append("rang buoc manifest khac phien")
    if sorted(m.get("video", {})) != sorted(ph.ctx.videos):
        loi.append(f"manifest co {len(m.get('video', {}))} video != {len(ph.ctx.videos)}")
    for v, x in m.get("video", {}).items():
        lan = W.tuyet_doi(x["lan"])
        if not W.trong(lan, ph.dir / "cache" / det):
            loi.append(f"{v}: lan cache ngoai phien")
            continue
        for f, k in ((f"{v}.npz", "npz_sha256"), (f"{v}.json", "json_sha256"), ("meta.json", "meta_sha256")):
            if not (lan / f).is_file() or sha(lan / f) != x[k]:
                loi.append(f"{det}/{v}: {f} sha256 lech manifest")
    if loi:
        raise LoiDauVao(f"cache {det} khong khop manifest ({len(loi)}): {loi[:5]}")
    return m


def cong_viec_cache(ph: Phien, args, nk: dict) -> dict:
    ctx, det = ph.ctx, args.detector
    if det not in ctx.gt["detector"]:
        raise LoiDauVao(f"--detector {det!r} khong thuoc giao thuc")
    d0, m0, loi0 = manifest_cache(ph, det)
    if d0 is not None or loi0:
        raise LoiDauVao(f"{det} da co manifest cache ({rel(d0) if d0 else loi0}) - BAT BIEN, tu choi tao cache moi")
    if sha(ctx.nguon.checkpoint[det]) != ctx.gt["detector"][det]["sha256"]:
        raise LoiDauVao(f"checkpoint {det} sha256 khac giao thuc")
    bang = W.doc_bang_hash(ctx.nguon.bang_hash)
    truoc = manifest_chua_cong_bo(ph, det)
    spec = ctx.tao_spec(det)
    ss = spec.spec_sha256()
    viec, sai = [], []
    for v in ctx.videos:
        rb = rang_buoc_cache(ph, det, ss, v, bang.get(v, []))
        chon, _, ds = chon_lan(ph.dir / "cache" / det / v, rb)
        nk["lan_thu"] += ds
        sai += [x["lan"] for x in ds if x["trang_thai"] == "lech"]
        if chon is None:
            viec.append((v, rb))
        else:
            nk["tai_su_dung"].append(rel(chon))
    # F1 (vong 3): TRUOC khi nap model. Cache da hoan tat ma sai hash -> chan (khong suy luan lai de thay); da co ban
    # manifest co the da cong bo -> cache dong bang: chi tai su dung, khong goi detector tao / thay cache.
    if sai:
        raise LoiDauVao(f"{det}: {len(sai)} lan cache da hoan tat nhung sai hash {sai[:3]} - chan, khong suy luan lai "
                        f"de thay cache")
    if viec and truoc:
        raise LoiDauVao(f"{det}: da co {len(truoc)} ban manifest cache chua/khong con cong bo ({rel(truoc[0][0])}) - "
                        f"cache dong bang, khong goi detector tao/thay cache cho {[v for v, _ in viec][:5]}")
    model = None
    for v, rb in viec:
        n = len(bang.get(v, []))
        si = W.doc_seqinfo(ctx.nguon.seqinfo(v))
        if n == 0 or int(si.get("seqLength", -1)) != n:
            raise LoiDauVao(f"{v}: so anh bang hash {n} != seqLength {si.get('seqLength')}")
        if model is None:
            lech = kiem_moi_truong(ctx, ph.khoa)
            if lech:
                raise LoiDauVao(f"kiem truoc suy luan lech: {lech[:5]}")
            model = ctx.mo_model(det)
            nk["goi_detector"] = True
        d = lan_moi(ph.dir / "cache" / det / v, ph.vung)
        t0 = time.time()
        try:
            out = W.tao_cache(ctx.gt, ctx.nguon, det, model, spec, d, {"videos": [v], "frame_dau": 1, "frame_cuoi": n},
                              bang, not ctx.fixture)
            ms = kiem_module_sau(ph.khoa)
            if ms:
                raise LoiChay(f"kiem module sau suy luan lech: {ms[:5]}")
        except Exception as e:
            W.ghi_json(d / "LOI.json", {"loi": f"{type(e).__name__}: {e}", "luc": bay_gio()}, [d])
            raise
        ngat(ctx, "cache-truoc-marker")
        ghi_marker(ctx, "cache", d, rb, {"giay": round(time.time() - t0, 2), "chi_tiet": out["video"][v],
                                         "anh_xa": out["anh_xa"], "meta_sha256": out["meta_sha256"]})
        nk["moi"].append(rel(d))
        print(f"  [cache] {det}/{v}: {out['video'][v]['n_boxes']} box, {round(time.time() - t0, 1)} s -> {rel(d)}")
        ngat(ctx, "cache-sau-marker")
    return {"detector": det, "spec_sha256": ss, "so_video": len(ctx.videos), "so_moi": len(viec)}


def kiem_npz(ctx: Ctx, det: str, d: Path, v: str, n: int, rb: dict) -> list[str]:
    """Schema / anh xa / frame-timeline / unreadable-truncated / spec cua MOT lan cache (chi doc)."""
    import numpy as np
    import det_cache
    loi = []
    with np.load(d / f"{v}.npz") as z:
        if set(z.files) != set(KHOA_NPZ):
            return [f"{v}: khoa npz {sorted(z.files)} != {sorted(KHOA_NPZ)}"]
        a = {k: z[k] for k in KHOA_NPZ}
    loi += [f"{v}: {k} dtype {a[k].dtype} != {t}" for k, t in KHOA_NPZ.items() if str(a[k].dtype) != t]
    if a["data"].ndim != 2 or a["data"].shape[1] != 6:
        return loi + [f"{v}: data shape {a['data'].shape} != (N, 6)"]
    N = len(a["data"])
    if a["cls_goc"].shape != (N,):
        loi.append(f"{v}: cls_goc shape {a['cls_goc'].shape} != ({N},)")
    npf = a["n_per_frame"]                     # F3: VideoDetections dung npf lam offsets -> shape/khong am/tong
    if npf.shape != (n,):
        loi.append(f"{v}: n_per_frame shape {npf.shape} != ({n},)")
    elif (npf < 0).any():
        loi.append(f"{v}: n_per_frame co gia tri am")
    if int(npf.sum()) != N:
        loi.append(f"{v}: tong n_per_frame {int(npf.sum())} != so box {N}")
    if a["frame_idx"].shape != (n,) or not np.array_equal(a["frame_idx"], np.arange(1, n + 1)):
        loi.append(f"{v}: frame_idx khac 1..{n}")
    from mot_datasets.detrac import DetracDataset
    dt = 1.0 / float(DetracDataset.fps)
    if a["time_s"].shape != (n,) or any(float(t) != (i - 1) * dt for i, t in enumerate(a["time_s"].tolist(), 1)):
        loi.append(f"{v}: time_s khac (i-1)*dt")
    if a["unreadable"].shape != (n,) or a["truncated"].shape != (n,) or a["unreadable"].any() or a["truncated"].any():
        loi.append(f"{v}: unreadable/truncated")
    if a["orig_shape"].tolist() != list(W.IMG_HW):
        loi.append(f"{v}: orig_shape {a['orig_shape'].tolist()}")
    info = json.loads((d / f"{v}.json").read_text(encoding="utf-8"))
    meta = det_cache.load_meta(d)
    names = {int(k): s for k, s in info.get("names", {}).items()}
    try:
        bang = W.bang_anh_xa_coco(names, rb["loai_checkpoint"])
        goc = a["data"].copy()
        goc[:, 5] = a["cls_goc"]
        if not np.array_equal(W.doi_id_sang_coco(goc, bang), a["data"]):
            loi.append(f"{v}: cls coco != anh xa(cls_goc)")
        if W.chuan_json(info.get("anh_xa_coco")) != W.chuan_json(None if bang is None else
                                                                {str(k): x for k, x in sorted(bang.items())}):
            loi.append(f"{v}: anh_xa_coco trong json khac names")
    except Exception as e:  # noqa: BLE001
        loi.append(f"{v}: anh xa lop loi {e}")
    if rb["loai_checkpoint"] == "lop5" and not set(int(c) for c in a["data"][:, 5]) <= LOP_COCO_LOP5:
        loi.append(f"{v}: lop COCO ngoai {sorted(LOP_COCO_LOP5)}")
    if (info.get("n_frames"), info.get("n_boxes"), info.get("n_unreadable"), info.get("n_truncated"),
            info.get("frames")) != (n, N, 0, 0, [1, n]):
        loi.append(f"{v}: json n_frames/n_boxes/unreadable/truncated/frames lech")
    if info.get("anh_sha256_ds") != rb["anh_sha256_ds"]:
        loi.append(f"{v}: anh_sha256_ds khac bang hash")
    if info.get("spec_sha256") != meta.get("spec_sha256") or meta.get("spec_sha256") != rb["spec_sha256"]:
        loi.append(f"{v}: spec_sha256 json/meta/rang buoc lech")
    c = ctx.gt["cache"]
    if meta.get("backend") != "ultralytics" or meta.get("weights_sha256") != rb["checkpoint_sha256"] or any(
            meta.get(k) != c[k] for k in ("imgsz", "conf", "iou", "max_det", "half", "agnostic_nms", "augment",
                                          "batch", "classes", "mask_ignored_regions")) or str(meta.get("device")) != str(c["device"]):
        loi.append(f"{v}: meta.json khac cau hinh cache giao thuc")
    for cmin in ctx.gt["tai"]["dieu_kien"].values():
        e = det_cache.check_compatible(meta, cmin, ctx.gt["tai"]["classes"], ctx.gt["tai"]["max_det"])
        if e:
            loi.append(f"{v}: check_compatible({cmin}) {e}")
    st, why = det_cache.sequence_spec_status(d, v, meta)
    if st != "ok":
        loi.append(f"{v}: sequence_spec_status {st} {why}")
    return loi


def cong_viec_xac_nhan_cache(ph: Phien, args, nk: dict) -> dict:
    ctx, det = ph.ctx, args.detector
    if det not in ctx.gt["detector"]:
        raise LoiDauVao(f"--detector {det!r} khong thuoc giao thuc")
    d0, _, loi0 = manifest_cache(ph, det, nk)
    if loi0:
        raise LoiDauVao(f"{det}: manifest cache da cong bo nhung lech {loi0[:3]} - chan, khong tao lai")
    if d0 is not None:
        raise LoiGhiDe(f"{det} da co manifest cache {rel(d0)} - bat bien, khong tao lai")
    bang = W.doc_bang_hash(ctx.nguon.bang_hash)
    truoc = manifest_chua_cong_bo(ph, det)
    thieu, chon, spec_ss, sai = [], {}, set(), []
    for v in ctx.videos:
        goc = ph.dir / "cache" / det / v
        ds = sorted(x for x in goc.iterdir() if x.is_dir()) if goc.is_dir() else []
        hop_le = None
        for x in ds:                           # spec_sha256 lay tu marker: KHONG dung model / fill_env de xac nhan
            mk = doc_marker(x)
            if mk is None:                     # F2: khong marker / marker hong -> lan khong hop le, tim tiep
                if (x / MARKER).is_file():
                    nk["lan_thu"].append({"lan": rel(x), "trang_thai": "hong", "ly_do": "marker khong doc duoc"})
                continue
            rb = rang_buoc_cache(ph, det, mk.get("rang_buoc", {}).get("spec_sha256"), v, bang.get(v, []))
            tt, ly_do, m = danh_gia_lan(x, rb)
            nk["lan_thu"].append({"lan": rel(x), "trang_thai": tt, "ly_do": ly_do})
            if tt == "lech":                   # F1 (vong 3): cache da hoan tat nhung sai hash -> chan ca detector
                sai.append(rel(x))
            if tt == "hop_le" and hop_le is None:
                hop_le = (x, m, rb)
        if hop_le is None:
            thieu.append(v)
        else:
            chon[v] = hop_le
            spec_ss.add(hop_le[2]["spec_sha256"])
    if sai:
        raise LoiDauVao(f"{det}: {len(sai)} lan cache da hoan tat nhung sai hash {sai[:3]} - chan, khong xac nhan, "
                        f"khong tracking")
    if thieu:
        raise LoiDauVao(f"{det}: thieu cache hoan tat cho {len(thieu)}/{len(ctx.videos)} video: {thieu[:5]} - "
                        f"khong xac nhan, khong tracking")
    if len(spec_ss) != 1:
        raise LoiDauVao(f"{det}: spec_sha256 khong dong nhat giua cac video: {sorted(spec_ss)}")
    lech_truoc = []                            # F1 (vong 3): giu rang buoc cache cua ban manifest co the da cong bo
    for x, mt in truoc:
        if mt is None:
            continue
        if mt.get("spec_sha256") not in spec_ss or sorted(mt.get("video", {})) != sorted(chon):
            lech_truoc.append(f"{rel(x)}: spec / danh sach video")
            continue
        for v, (cx, m, _) in chon.items():
            a = mt["video"][v]
            if (a.get("lan"), a.get("npz_sha256"), a.get("json_sha256"), a.get("meta_sha256"),
                    a.get("marker_sha256")) != (rel(cx), m["file"][f"{v}.npz"], m["file"][f"{v}.json"],
                                                m["file"]["meta.json"], sha(cx / MARKER)):
                lech_truoc.append(f"{rel(x)}/{v}")
    if lech_truoc:
        raise LoiDauVao(f"{det}: cache khac ban manifest truoc (co the da cong bo) {lech_truoc[:5]} - chan, khong "
                        f"xac nhan cache thay the")
    loi = []
    for v, (x, m, rb) in chon.items():
        loi += kiem_npz(ctx, det, x, v, len(bang[v]), rb)
    if loi:
        raise LoiDauVao(f"{det}: cache khong dat xac nhan ({len(loi)}): {loi[:6]}")
    ms = kiem_module_sau(ph.khoa)
    if ms:
        raise LoiChay(f"kiem module sau xac nhan lech: {ms[:5]}")
    d = lan_moi(ph.dir / "xac_nhan_cache" / det, ph.vung)
    man = {"detector": det, "rang_buoc": ph.co_so, "spec_sha256": spec_ss.pop(), "so_video": len(chon),
           "tao_luc": bay_gio(),
           "video": {v: {"lan": rel(x), "npz_sha256": m["file"][f"{v}.npz"], "json_sha256": m["file"][f"{v}.json"],
                         "meta_sha256": m["file"]["meta.json"], "marker_sha256": sha(x / MARKER),
                         "n_frames": len(bang[v]), "n_boxes": m["chi_tiet"]["n_boxes"]} for v, (x, m, rb) in chon.items()},
           "kiem": ["schema npz", "anh xa lop", "frame/timeline", "unreadable/truncated", "spec/meta = giao thuc",
                    "sha256 npz/json/meta = marker"]}
    s = W.ghi_json(d / "manifest_cache.json", man, [d])
    ngat(ctx, "xac-nhan-truoc-sums")
    ghi_text(d / "SHA256SUMS", f"{s}  manifest_cache.json\n", d)
    ngat(ctx, "xac-nhan-truoc-marker")
    ghi_marker(ctx, "xac-nhan", d, rang_buoc_xac_nhan(ph, det), {"manifest_sha256": s})   # CONG BO (ghi cuoi)
    nk["moi"].append(rel(d))
    print(f"  [xac-nhan-cache] {det}: {len(chon)} video -> {rel(d)} ({s[:12]})")
    return {"detector": det, "manifest": rel(d), "manifest_sha256": s, "so_video": len(chon)}


# ---------------------------------------------------------------------------
# Tracking / cham
# ---------------------------------------------------------------------------
def cau_hinh_run(ctx: Ctx, r: dict):
    import run_tracking as rt
    cfg = rt.load_tracker_cfg(W.yaml_arg(r["yaml"]), {"track_buffer": W.TRACK_BUFFER_BUOC})
    return cfg, rt.cfg_to_dict(cfg)


def rang_buoc_tracking(ph: Phien, r: dict, cfg_d: dict, man: dict, v: str) -> dict:
    import run_tracking as rt
    import ul_backend
    t = ph.ctx.gt["tai"]
    return {"loai": "tracking", **ph.co_so, "run": r["run"], "detector": r["detector"], "cond": r["cond"],
            "conf_min": r["conf_min"], "bo_loc": r["bo_loc"], "seed": r["seed"],
            "yaml_sha256": sha(Path(rt.resolve_tracker_yaml(W.yaml_arg(r["yaml"])))),
            "cfg_sha256": W.sha256_bytes(json.dumps(W.chuan_json(cfg_d), sort_keys=True).encode()),
            "backend": ul_backend.backend_name(), "manifest_cache_sha256": man["_sha256"], "video": v,
            "npz_sha256": man["video"][v]["npz_sha256"], "classes": t["classes"], "max_det": t["max_det"],
            "fps": W.FPS, "elapsed_mode": "skip"}


def lay_runs(ctx: Ctx, args) -> list[dict]:
    if not args.run:
        raise LoiDauVao("can --run <ten run> (mot hoac nhieu; khong co mac dinh tat ca)")
    la = [x for x in args.run if x not in ctx.runs]
    if la:
        raise LoiDauVao(f"run ngoai ma tran 36: {la[:5]}")
    return [ctx.runs[x] for x in dict.fromkeys(args.run)]


def lay_video(ctx: Ctx, args) -> list[str]:
    """R56: --video <v> (thuoc manifest) -> [v], xu ly TOAN BO timeline video do; khong co --video -> moi video
    manifest. Chi loc danh sach don vi; ctx.videos / manifest / tap danh gia khong doi."""
    if args.video is None:
        return list(ctx.videos)
    if args.video not in ctx.videos:
        raise LoiDauVao(f"--video {args.video!r} khong thuoc manifest ({len(ctx.videos)} video) - tu choi truoc khi "
                        f"tao dau ra")
    return [args.video]


def kiem_info(ctx: Ctx, r: dict, info: dict, n: int) -> list[str]:
    loi = []
    if (info["n_frames"], info["n_buoc_timing"], info["n_steps"], info["track_buffer"]) != (n, n, [1], W.TRACK_BUFFER_BUOC):
        loi.append(f"{r['run']}/{info['video']}: frame/buoc/track_buffer lech")
    if r["bo_loc"] == "pf500" and (info.get("seed_thuc_dung"), info.get("num_particles"), info.get("k1_optimal_proposal"),
                                   info.get("k2_roughening")) != (r["seed"], 500, False, 0.0):
        loi.append(f"{r['run']}/{info['video']}: seed/N/K1/K2 thuc dung lech")
    return loi


def cong_viec_tracking(ph: Phien, args, nk: dict) -> dict:
    import run_tracking as rt
    ctx = ph.ctx
    runs = lay_runs(ctx, args)
    vids = lay_video(ctx, args)                 # R56: loc TRUOC moi kiem/ghi; video la -> exit 2, khong dau ra
    nk["pham_vi"] = {"loc_video": args.video is not None, "video": vids, "so_video_manifest": len(ctx.videos)}
    nk["don_vi"] = []
    # Xac nhan cache CUA MOI detector can dung TRUOC bat ky tracking nao (thieu / sai hash -> dung). Kiem MOI video
    # trong manifest cache (ca khi --video): loc video KHONG giam kiem hash cache.
    mans = {d: kiem_cache_theo_manifest(ph, d) for d in dict.fromkeys(r["detector"] for r in runs)}
    lech = kiem_moi_truong(ctx, ph.khoa)
    if lech:
        raise LoiDauVao(f"kiem truoc tracking lech: {lech[:5]}")
    reg = rt.tracker_registry()
    tr = ctx.gt["tracker"]
    so_moi = 0
    for r in runs:
        cfg, cfg_d = cau_hinh_run(ctx, r)
        man = mans[r["detector"]]
        for v in vids:
            rb = rang_buoc_tracking(ph, r, cfg_d, man, v)   # rang buoc KHONG chua pham vi loc -> dung chung voi full
            chon, mk, ds = chon_lan(ph.dir / "tracking" / r["run"] / v, rb)
            nk["lan_thu"] += ds
            if chon is not None:
                nk["tai_su_dung"].append(rel(chon))
                nk["don_vi"].append({"run": r["run"], "video": v, "frame": [1, mk["info"]["n_frames"]],
                                     "trang_thai": "tai_su_dung", "lan": rel(chon)})
                continue
            d = lan_moi(ph.dir / "tracking" / r["run"] / v, ph.vung)
            lan_cache = W.tuyet_doi(man["video"][v]["lan"])
            try:
                if sha(lan_cache / f"{v}.npz") != man["video"][v]["npz_sha256"]:
                    raise LoiDauVao(f"{v}.npz doi sau xac nhan")
                vd = W.mo_cache(lan_cache, v, r["conf_min"], ctx.gt["tai"]["classes"], ctx.gt["tai"]["max_det"])
                info = W.chay_tracker(d, v, vd, cfg, r["seed"], reg, tr["assoc_log"], tr["pf_diag"])
                loi = kiem_info(ctx, r, info, man["video"][v]["n_frames"])
                ms = kiem_module_sau(ph.khoa)
                if loi or ms:
                    raise LoiChay(f"{loi[:3]} kiem module sau tracking {ms[:3]}")
            except Exception as e:
                W.ghi_json(d / "LOI.json", {"loi": f"{type(e).__name__}: {e}", "luc": bay_gio()}, [d])
                raise
            ngat(ctx, "tracking-truoc-marker")
            ghi_marker(ctx, "tracking", d, rb, {"info": info, "lan_cache": rel(lan_cache)})
            nk["moi"].append(rel(d))
            nk["don_vi"].append({"run": r["run"], "video": v, "frame": [1, info["n_frames"]], "trang_thai": "moi",
                                 "lan": rel(d)})
            so_moi += 1
            ngat(ctx, "tracking-sau-marker")
    for d_, man in mans.items():                # cache khong doi sau tracking
        kiem_cache_theo_manifest(ph, d_)
    fr = sorted({(x["video"], tuple(x["frame"])) for x in nk["don_vi"]})
    print(f"  [tracking] pham vi {'--video ' + vids[0] if args.video is not None else 'toan bo manifest'}: "
          f"{len(vids)} video, frame {', '.join(f'{v} {a}-{b}' for v, (a, b) in fr[:3])}"
          f"{' ...' if len(fr) > 3 else ''}; {len(runs)} run; moi {so_moi}, tai su dung {len(nk['don_vi']) - so_moi}")
    return {"so_run": len(runs), "so_lan_moi": so_moi, "video": vids, "loc_video": args.video is not None}


def mot_cua_run(ph: Phien, r: dict, man: dict) -> tuple[dict, list[str]]:
    _, cfg_d = cau_hinh_run(ph.ctx, r)
    out, thieu = {}, []
    for v in ph.ctx.videos:
        chon, mk, _ = chon_lan(ph.dir / "tracking" / r["run"] / v, rang_buoc_tracking(ph, r, cfg_d, man, v))
        if chon is None:
            thieu.append(v)
        else:
            out[v] = (chon, mk["file"][f"data/{v}.txt"])
    return out, thieu


def seq_info(ctx: Ctx) -> dict:
    return {v: int(W.doc_seqinfo(ctx.nguon.seqinfo(v))["seqLength"]) for v in ctx.videos}


def kiem_gt_khoa(ph: Phien) -> str:
    rows = []
    for v in ph.ctx.videos:
        for p in (ph.ctx.nguon.gt_txt(v), ph.ctx.nguon.seqinfo(v)):
            s = ph.file_khoa.get(p.resolve())
            if s is None or not p.is_file() or sha(p) != s:
                raise LoiDauVao(f"GT/seqinfo {rel(p)} khong co trong khoa hoac sha256 lech")
            rows.append((rel(p), s))
    return W.sha256_ds(rows)


def rang_buoc_cham(ph: Phien, r: dict, man: dict, mot: dict, gt_ds: str, si: dict) -> dict:
    """F6: rang buoc cham gom hash manifest cache + cac lan tracking da dung (khong chi hash MOT): manifest khac ->
    lan cham cu KHONG duoc tai su dung / chon du MOT trung byte."""
    return {"loai": "cham", **ph.co_so, "run": r["run"], "split": ph.ctx.nguon.thu_muc_gt.name, "gt_sha256_ds": gt_ds,
            "manifest_cache_sha256": man["_sha256"],
            "mot_lan_ds": W.sha256_ds((v, rel(lan)) for v, (lan, _) in mot.items()),
            "mot_sha256_ds": W.sha256_ds((v, s) for v, (_, s) in mot.items()),
            "danh_gia": ph.ctx.gt["danh_gia"], "seq_info": si}


def cong_viec_cham(ph: Phien, args, nk: dict) -> dict:
    ctx = ph.ctx
    runs = lay_runs(ctx, args)
    mans = {d: kiem_cache_theo_manifest(ph, d) for d in dict.fromkeys(r["detector"] for r in runs)}
    lech = kiem_moi_truong(ctx, ph.khoa)
    if lech:
        raise LoiDauVao(f"kiem truoc cham lech: {lech[:5]}")
    gt_ds = kiem_gt_khoa(ph)
    si = seq_info(ctx)
    split, gt_folder = ctx.nguon.thu_muc_gt.name, ctx.nguon.thu_muc_gt.parent
    bo = {}
    for r in runs:                               # du video cho MOI run truoc khi cham bat ky run nao
        mot, thieu = mot_cua_run(ph, r, mans[r["detector"]])
        if thieu:
            raise LoiDauVao(f"{r['run']}: chua du {len(ctx.videos)} video co tracking hoan tat (thieu {thieu[:5]}) - "
                            f"khong cham")
        bo[r["run"]] = mot
    kq = {}
    for r in runs:
        mot = bo[r["run"]]
        rb = rang_buoc_cham(ph, r, mans[r["detector"]], mot, gt_ds, si)
        chon, mk, ds = chon_lan(ph.dir / "cham" / r["run"], rb)
        nk["lan_thu"] += ds
        if chon is not None:
            nk["tai_su_dung"].append(rel(chon))
            kq[r["run"]] = {"lan": rel(chon), "tai_su_dung": True}
            continue
        d = lan_moi(ph.dir / "cham" / r["run"], ph.vung)
        try:
            trk = d / "trackers" / split / r["run"] / "data"
            trk.mkdir(parents=True)
            for v, (lan, s) in mot.items():
                b = (lan / "data" / f"{v}.txt").read_bytes()
                if W.sha256_bytes(b) != s:
                    raise LoiDauVao(f"{r['run']}/{v}: MOT lech marker")
                (trk / f"{v}.txt").write_bytes(b)
            if ngat(ctx, "cham-loi"):
                raise LoiChay("loi TrackEval gia lap (fixture)")
            for p in (d / "trackers", d / "trackeval"):
                W.kiem_trong_vung(p, [d])
            ev = W.cham(gt_folder, d / "trackers", split, r["run"], si, d / "trackeval", [d, gt_folder])
            if not ev["huu_han"]:
                raise LoiChay(f"{r['run']}: metric khong huu han")
            if kiem_gt_khoa(ph) != gt_ds or any(sha(trk / f"{v}.txt") != s for v, (_, s) in mot.items()):
                raise LoiChay(f"{r['run']}: GT/MOT doi trong luc cham")
            ms = kiem_module_sau(ph.khoa)
            if ms:
                raise LoiChay(f"{r['run']}: kiem module sau cham lech {ms[:5]}")
        except Exception as e:
            W.ghi_json(d / "LOI.json", {"loi": f"{type(e).__name__}: {e}", "luc": bay_gio()}, [d])
            raise
        ngat(ctx, "cham-truoc-marker")
        ghi_marker(ctx, "cham", d, rb, {"combined": ev["combined"], "file_metric": ev["file"],
                                        "cau_hinh": ev["cau_hinh"], "mot_lan": {v: rel(lan) for v, (lan, _) in mot.items()}})
        nk["moi"].append(rel(d))
        kq[r["run"]] = {"lan": rel(d), "HOTA": ev["combined"].get("HOTA")}
    return {"run": kq}


def lan_cham_da_chon(ph: Phien, r: dict) -> tuple[Path, dict]:
    man = kiem_cache_theo_manifest(ph, r["detector"])
    mot, thieu = mot_cua_run(ph, r, man)
    if thieu:
        raise LoiDauVao(f"{r['run']}: chua du video")
    # F6: so TOAN BO rang buoc (manifest cache, lan tracking, MOT, GT, danh gia); khong marker / hong / lech -> bo qua
    chon, mk, _ = chon_lan(ph.dir / "cham" / r["run"], rang_buoc_cham(ph, r, man, mot, kiem_gt_khoa(ph), seq_info(ph.ctx)))
    if chon is None:
        raise LoiDauVao(f"{r['run']}: chua co lan cham hoan tat khop bo MOT / manifest cache hien tai")
    return chon, mk


def cong_viec_lap_tracker(ph: Phien, args, nk: dict) -> dict:
    """Chay lai tracker tu CUNG cache/seed vao thu muc moi; so byte MOT voi lan da chon. Khong goi detector."""
    import run_tracking as rt
    ctx = ph.ctx
    r = lay_runs(ctx, args)[0]
    v = lay_video(ctx, args)[0]
    man = kiem_cache_theo_manifest(ph, r["detector"])
    lech = kiem_moi_truong(ctx, ph.khoa)
    if lech:
        raise LoiDauVao(f"kiem truoc lap tracker lech: {lech[:5]}")
    cfg, cfg_d = cau_hinh_run(ctx, r)
    chon, mk, _ = chon_lan(ph.dir / "tracking" / r["run"] / v, rang_buoc_tracking(ph, r, cfg_d, man, v))
    if chon is None:
        raise LoiDauVao(f"{r['run']}/{v}: chua co lan tracking hoan tat")
    d = lan_moi(ph.dir / "lap_tracker" / r["run"] / v, ph.vung)
    vd = W.mo_cache(W.tuyet_doi(man["video"][v]["lan"]), v, r["conf_min"], ctx.gt["tai"]["classes"], ctx.gt["tai"]["max_det"])
    info = W.chay_tracker(d, v, vd, cfg, r["seed"], rt.tracker_registry(), ctx.gt["tracker"]["assoc_log"],
                          ctx.gt["tracker"]["pf_diag"])
    moi = sha(d / "data" / f"{v}.txt")
    ms = kiem_module_sau(ph.khoa)
    kq = {"run": r["run"], "video": v, "lan_goc": rel(chon), "sha256_goc": mk["file"][f"data/{v}.txt"],
          "frame": [1, info["n_frames"]],
          "sha256_lap": moi, "trung_byte": moi == mk["file"][f"data/{v}.txt"], "seed_thuc_dung": info["seed_thuc_dung"],
          "kiem_module_sau": ms}
    W.ghi_json(d / "ket_qua_lap.json", kq, [d])
    if ms:
        raise LoiChay(f"lap tracker {r['run']}/{v}: kiem module sau lech {ms[:5]}")
    if not kq["trung_byte"]:
        raise LoiChay(f"lap tracker {r['run']}/{v} khong trung byte")
    return kq


def cong_viec_cham_lai(ph: Phien, args, nk: dict) -> dict:
    """Cham lai tu DUNG ban sao MOT da bam trong lan cham hoan tat; khong goi detector/tracker."""
    ctx = ph.ctx
    r = lay_runs(ctx, args)[0]
    goc_lan, mk = lan_cham_da_chon(ph, r)
    lech = kiem_moi_truong(ctx, ph.khoa)
    if lech:
        raise LoiDauVao(f"kiem truoc cham lai lech: {lech[:5]}")
    gt_ds = kiem_gt_khoa(ph)
    if gt_ds != mk["rang_buoc"]["gt_sha256_ds"]:
        raise LoiDauVao("GT khac lan cham goc")
    split, gt_folder = ctx.nguon.thu_muc_gt.name, ctx.nguon.thu_muc_gt.parent
    d = lan_moi(ph.dir / "cham_lai" / r["run"], ph.vung)
    ev = W.cham(gt_folder, goc_lan / "trackers", split, r["run"], seq_info(ctx), d / "trackeval",
                [d, goc_lan / "trackers", gt_folder])
    tt, ly_do, _ = danh_gia_lan(goc_lan, mk["rang_buoc"])
    ms = kiem_module_sau(ph.khoa)
    kq = {"run": r["run"], "lan_cham_goc": rel(goc_lan), "file_goc": mk["file_metric"], "file_lai": ev["file"],
          "file_trung_byte": ev["file"] == mk["file_metric"], "combined_trung": ev["combined"] == mk["combined"],
          "lan_goc_con_nguyen": tt == "hop_le", "kiem_module_sau": ms}
    W.ghi_json(d / "ket_qua_cham_lai.json", kq, [d])
    if ms:
        raise LoiChay(f"cham lai {r['run']}: kiem module sau lech {ms[:5]}")
    if not (kq["file_trung_byte"] and kq["combined_trung"] and kq["lan_goc_con_nguyen"]):
        raise LoiChay(f"cham lai {r['run']} lech: {kq} {ly_do}")
    return kq


def cong_viec_hau_kiem(ph: Phien, args, nk: dict) -> dict:
    ctx = ph.ctx
    chua, loi, dem = [], [], {"cache": 0, "tracking": 0, "cham": 0}
    mans = {}
    for det in ctx.gt["detector"]:
        try:
            mans[det] = kiem_cache_theo_manifest(ph, det)
            dem["cache"] += len(mans[det]["video"])
        except LoiDauVao as e:
            chua.append(f"cache {det}: {e}")
    for run, r in ctx.runs.items():
        if r["detector"] not in mans:
            chua.append(f"{run}: chua co cache")
            continue
        mot, thieu = mot_cua_run(ph, r, mans[r["detector"]])
        dem["tracking"] += len(mot)
        if thieu:
            chua.append(f"{run}: thieu tracking {len(thieu)} video")
            continue
        try:
            lan_cham_da_chon(ph, r)
            dem["cham"] += 1
        except LoiDauVao as e:
            chua.append(f"{run}: {e}")
    lan_all = [p for sub in ("cache", "xac_nhan_cache", "tracking", "cham") for p in (ph.dir / sub).rglob("lan_*")
               if p.is_dir()]
    do_dang = sorted(rel(p) for p in lan_all if not (p / MARKER).is_file())
    hong = sorted(rel(p) for p in lan_all if (p / MARKER).is_file() and doc_marker(p) is None)
    loi += kiem_module_sau(ph.khoa)
    kq = {"dem": dem, "chua_xong": chua, "loi": loi, "lan_khong_marker_giu_nguyen": do_dang,
          "lan_marker_hong_giu_nguyen": hong, "du_36_run": not chua and dem["cham"] == W.SO_RUN}
    d = lan_moi(ph.dir / "hau_kiem", ph.vung)
    W.ghi_json(d / "hau_kiem.json", kq, [d])
    if chua or loi:
        raise LoiTienKiem(f"hau-kiem: {len(chua)} muc chua xong / {len(loi)} loi -> {rel(d)}")
    return kq


CONG_VIEC_HAM = {"cache": cong_viec_cache, "xac-nhan-cache": cong_viec_xac_nhan_cache, "tracking": cong_viec_tracking,
                 "cham": cong_viec_cham, "lap-tracker": cong_viec_lap_tracker, "cham-lai": cong_viec_cham_lai,
                 "hau-kiem": cong_viec_hau_kiem}


def buoc_thuc_thi(ctx: Ctx, args) -> int:
    cv = args.cong_viec
    if cv not in CONG_VIEC:
        raise LoiDauVao(f"--cong-viec bat buoc, mot trong {CONG_VIEC}")
    if args.video is not None and cv not in CONG_VIEC_VIDEO:    # R56: khong bo qua am tham
        raise LoiDauVao(f"--video chi dung voi --cong-viec {'/'.join(CONG_VIEC_VIDEO)}, khong dung voi {cv}")
    if not ctx.fixture and not (args.duyet or "").strip():
        raise LoiDauVao("duong that can --duyet <ma plan thuc thi da duoc nguoi dung duyet>; R55 khong cap quyen")
    can = {"cache": ("detector",), "xac-nhan-cache": ("detector",), "tracking": ("run",), "cham": ("run",),
           "lap-tracker": ("run", "video"), "cham-lai": ("run",), "mo-phien": ("khoa",)}.get(cv, ())
    thieu = [a for a in can if not getattr(args, a)]
    if cv != "mo-phien" and not args.phien:
        thieu.append("phien")
    if thieu:
        raise LoiDauVao(f"--cong-viec {cv} thieu tham so: {thieu}")
    if cv in ("lap-tracker", "cham-lai") and len(args.run) != 1:
        raise LoiDauVao(f"{cv} nhan dung mot --run")
    if cv == "mo-phien":
        kq = cong_viec_mo_phien(ctx, args)
        print(f"[mo-phien] {kq['phien']} (khoa {kq['khoa']})")
        return 0
    ph = Phien(ctx, args.phien)
    nkd = lan_moi(ph.dir / "nhat_ky" / cv.replace("-", "_"), ph.vung)
    W.dat_moi_truong(nkd)
    nk = {"cong_viec": cv, "tham_so": dict(vars(args)), "bat_dau": bay_gio(), "pid": os.getpid(), "lan_thu": [],
          "tai_su_dung": [], "moi": [], "goi_detector": False}
    t0 = time.time()
    try:
        k = ph.xac_minh()
        nk["khoa"] = {"ten": k["khoa"], "sha256": k["khoa_sha256"], "so_file": k["so_file"]}
        nk["ket_qua"] = CONG_VIEC_HAM[cv](ph, args, nk)
        ma = 0
    except W.LoiR54 as e:
        nk["loi"] = f"{type(e).__name__}: {e}"
        ma = e.ma
    except Exception as e:  # noqa: BLE001
        nk["loi"] = f"{type(e).__name__}: {e}"
        ma = LoiChay.ma
    nk.update({"ket_thuc": bay_gio(), "giay": round(time.time() - t0, 2), "exit": ma})
    W.ghi_json(nkd / "ket_qua.json", nk, [nkd])
    out = sys.stdout if ma == 0 else sys.stderr
    print(f"[thuc-thi {cv}] moi {len(nk['moi'])}, tai su dung {len(nk['tai_su_dung'])} -> {rel(nkd)}  exit {ma}", file=out)
    if "loi" in nk:
        print("   -", nk["loi"][:1500], file=out)
    return ma


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R55 dieu phoi test40 (chuan bi tach rieng thuc thi)")
    ap.add_argument("--buoc", required=True, choices=BUOC)
    ap.add_argument("--tien-kiem")
    ap.add_argument("--tests")
    ap.add_argument("--khoa")
    ap.add_argument("--luu", choices=["chuan_bi"])
    ap.add_argument("--cong-viec", choices=CONG_VIEC)
    ap.add_argument("--phien")
    ap.add_argument("--detector")
    ap.add_argument("--run", nargs="+")
    ap.add_argument("--video")
    ap.add_argument("--duyet")
    args = ap.parse_args(argv)
    try:
        if args.buoc != "thuc-thi" and (args.cong_viec or args.phien or args.detector or args.run or args.video):
            raise LoiDauVao("--cong-viec/--phien/--detector/--run/--video chi dung voi --buoc thuc-thi")
        ctx = Ctx()
        return {"tien-kiem": buoc_tien_kiem, "khoa": buoc_khoa, "kiem-khoa": buoc_kiem_khoa,
                "thuc-thi": buoc_thuc_thi}[args.buoc](ctx, args)
    except W.LoiR54 as e:
        print(f"[{args.buoc}] {type(e).__name__}: {e}", file=sys.stderr)
        return e.ma


if __name__ == "__main__":
    raise SystemExit(main())
