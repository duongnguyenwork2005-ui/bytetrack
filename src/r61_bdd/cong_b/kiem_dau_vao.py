"""
kiem_dau_vao.py -- Kiem DAU VAO cong B cua R61 (plan chuan bi B0, 26/10/2026). Muc module CHI thu vien chuan + ma r61
(muc module cua hop_dong_bdd / chuan_bi_bdd / dieu_phoi_bdd / van_hanh_cong_a cung chi thu vien chuan).

Ghim (configs/phase2/r61_bdd/hop_dong_cong_b.json; fixture: ban sao che_do=fixture duoi tests/cong_b_20261026/):
  - goi A: goi_a.json (goi_a_sha256), SHA256SUMS (bytes = danh muc + sha256), marker goi_a.xong.json, MOI file payload +
    file thua tren pham vi payload cua A (van_hanh_cong_a.payload + goi/ + dong_goi/phuc_hoi/), khong reparse trong phien A;
  - khoa A: khoa.json (khoa_sha256) + khoa.xong.json + kiem_khoa/vong_01 (sha256, dat) + ANH KHOA HIEN TAI
    (dieu_phoi_bdd.so_khoa(khoa, anh_khoa) + van_hanh_cong_a.so_phu_luc(phu_luc_van_hanh, phu_luc)) -- KHONG doi phep so A;
  - PLAN nguyen byte: sha256 toan bo + truoc/sau muc 0 + byte_muc0 + so_byte (phep so 'nguyen_byte', khong chuan hoa);
  - review A: doc NGUYEN BAN, sha256 = digest da ghim, verdict = approve (khong sua review);
  - phu luc dung_da_go: DUNG hai file (duong, so byte, sha256); thieu / doi / thua -> chan;
  - van_hanh A: moi file thuoc goi A / phu luc / DUNG mot muc loai tru (mau + so_file + dieu kien + ly do); dung.json chan;
  - phe duyet B that: file rieng do orchestrator cung cap (duyet=true + rang buoc goi/khoa/review/hop dong/ma B/phien).
Loi -> LoiDauVao (con cua LoiHopDong -> exit 2, KHONG thu lai). Thong diep: "[<nhom>] [...]" voi nhom loi DAU TIEN theo
thu tu NHOM; ket qua tho moi nhom tra ve trong dict (ghi bang chung dat/loi tho).
JSON dau vao (hop dong, goi_a.json, review, phe duyet, khoa, kiem_khoa, kiem_goi_a, file phien B) doc qua doc_json_dv:
loi doc / giai ma / JSON cat cut / sai kieu / schema -> loi THO cua dung nhom (exit 2), khong de JSONDecodeError/KeyError
thoat thanh loi Python (B0-F9).
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hop_dong_bdd as H  # noqa: E402
import chuan_bi_bdd as CB  # noqa: E402  (muc module chi thu vien chuan)
import dieu_phoi_bdd as D  # noqa: E402  (muc module chi thu vien chuan)
import van_hanh_cong_a as VA  # noqa: E402  (muc module chi thu vien chuan; chi DOC ham, khong doi)

CONG_B = Path(__file__).resolve().parent
HOP_DONG_B = H.ROOT / "configs" / "phase2" / "r61_bdd" / "hop_dong_cong_b.json"
VUNG_FIXTURE_B = H.VUNG_TESTS / "cong_b_20261026"
FILE_MA_B = ("kiem_dau_vao.py", "van_hanh_cong_b.py", "thong_ke_tiep_tuc.py")
#: Thu tu nhom kiem (re -> nang); loi bao theo nhom DAU TIEN khong dat.
NHOM = ("duong", "plan", "review", "phe_duyet", "goi", "sha256sums", "marker_goi", "phu_luc", "van_hanh", "payload", "khoa")
PLAN_KHOA = ("sha256_toan_bo", "sha256_truoc_muc0", "sha256_sau_muc0", "byte_muc0", "so_byte")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class LoiDauVao(H.LoiHopDong):
    def __init__(self, nhom: str, loi: list, kq: dict | None = None):
        self.nhom, self.loi, self.kq = nhom, list(loi), kq
        super().__init__(f"[{nhom}] {self.loi[:6]}")


def duong(r) -> Path:
    """Duong trong hop dong: tuong doi ROOT (hoac tuyet doi, vd review ngoai repo)."""
    return Path(os.path.abspath(H.ROOT / r))


# ---------------------------------------------------------------------------
# Doc JSON dau vao co bao ve + schema (B0-F9)
# ---------------------------------------------------------------------------
def doc_json_dv(p, mo_ta: str, kieu: type = dict) -> tuple[object | None, list[str]]:
    """(obj, []) hoac (None, [loi tho]). Khong nem: loi doc, giai ma utf-8, JSON cat cut, sai kieu goc."""
    try:
        b = Path(p).read_bytes()
    except OSError as e:
        return None, [f"{mo_ta} khong doc duoc: {type(e).__name__}: {e}"]
    try:
        d = json.loads(b.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        return None, [f"{mo_ta} khong phai JSON hop le ({len(b)} byte): {type(e).__name__}: {e}"]
    if not isinstance(d, kieu):
        return None, [f"{mo_ta} sai kieu goc {type(d).__name__} (can {kieu.__name__})"]
    return d, []


def doc_json_hd(p, nhom: str, mo_ta: str) -> dict:
    """Doc JSON bat buoc hop le (file phien B / dau vao): loi -> LoiDauVao(nhom) giu nguyen nguyen nhan tho."""
    d, loi = doc_json_dv(p, mo_ta)
    if loi:
        raise LoiDauVao(nhom, loi)
    return d


def _dung_kieu(v, t) -> bool:
    if isinstance(v, bool) and t is not bool and (t is int or (isinstance(t, tuple) and bool not in t)):
        return False
    return isinstance(v, t)


def loi_schema(d: dict, mau: dict, tt: str = "") -> list[str]:
    """mau: {khoa: kieu | (kieu, ...) | {mau con}}; bool khong duoc tinh la int."""
    loi = []
    for k, t in mau.items():
        if k not in d:
            loi.append(f"thieu {tt}{k}")
        elif isinstance(t, dict):
            loi += loi_schema(d[k], t, f"{tt}{k}.") if isinstance(d[k], dict) else [f"{tt}{k} khong phai object"]
        elif not _dung_kieu(d[k], t):
            loi.append(f"{tt}{k} sai kieu {type(d[k]).__name__}")
    return loi


SO = (int, float)
SCHEMA_HOP_DONG = {
    "che_do": str, "python": str,
    "a": {"phien": str, "luot_a": str, "goi_a_sha256": str, "khoa_sha256": str, "sha256sums_sha256": str, "kiem_khoa": str,
          "kiem_khoa_sha256": str, "kiem_goi_a": str, "stt_dong_goi_dau": int, "so_file_goi": int},
    "plan": {"duong": str, "phep_so": str, "sha256_toan_bo": str, "sha256_truoc_muc0": str, "sha256_sau_muc0": str,
             "byte_muc0": list, "so_byte": int},
    "review_a": {"duong": str, "sha256": str, "verdict": str},
    "phu_luc_dung_da_go": list, "van_hanh_a": {"chan": list, "loai_tru": list}, "chinh_sach_crash": dict,
    "deadline": {"ngan_sach_toi_da": SO, "du_phong_cung_con": SO, "du_phong_ket_thuc": SO, "toi_thieu_khoi_chay": SO,
                 "du_phong_don_vi": SO},
    "bootstrap_khoi": {"kich_thuoc_khoi": int, "kich_thuoc_khoi_fixture": int}}
SCHEMA_GOI = {"luot": str, "phien_A": str, "khoa": str, "khoa_sha256": str, "stt_dong_goi_dau": int, "so_file": int,
              "file": dict}


def loi_schema_hop_dong(hd: dict) -> list[str]:
    loi = loi_schema(hd, SCHEMA_HOP_DONG)
    if loi:
        return loi
    for i, e in enumerate(hd["phu_luc_dung_da_go"]):
        loi += loi_schema(e, {"duong": str, "so_byte": int, "sha256": str}, f"phu_luc_dung_da_go[{i}].") \
            if isinstance(e, dict) else [f"phu_luc_dung_da_go[{i}] khong phai object"]
    for i, x in enumerate(hd["van_hanh_a"]["loai_tru"]):
        if not isinstance(x, dict):
            loi.append(f"van_hanh_a.loai_tru[{i}] khong phai object")
            continue
        loi += loi_schema(x, {"ten": str, "mau": str, "so_file": int, "dieu_kien": dict, "ly_do": str}, f"loai_tru[{i}].")
        try:
            re.compile(x.get("mau") if isinstance(x.get("mau"), str) else "")
        except re.error as e:
            loi.append(f"loai_tru[{i}].mau regex sai: {e}")
    if not all(isinstance(c, str) for c in hd["van_hanh_a"]["chan"]):
        loi.append("van_hanh_a.chan phai la danh sach chuoi")
    return loi


# ---------------------------------------------------------------------------
# Hop dong + ma B
# ---------------------------------------------------------------------------
def nap_hop_dong(p=None) -> dict:
    p = Path(os.path.abspath(p or HOP_DONG_B))
    if not p.is_file() or H.la_reparse(p):
        raise LoiDauVao("hop_dong", [f"khong doc duoc hop dong {H.rel(p)}"])
    hd, loi = doc_json_dv(p, f"hop dong {H.rel(p)}")
    if loi:
        raise LoiDauVao("hop_dong", loi)
    fx = hd.get("che_do") == "fixture"
    loi = loi_schema_hop_dong(hd)
    if hd.get("che_do") not in ("that", "fixture"):
        loi.append(f"che_do {hd.get('che_do')!r}")
    if not fx and p != HOP_DONG_B:
        loi.append("hop dong that chi duoc la configs/phase2/r61_bdd/hop_dong_cong_b.json")
    if fx and (not H._trong(p, VUNG_FIXTURE_B) or p == VUNG_FIXTURE_B):
        loi.append("hop dong fixture phai nam duoi results/phase2/r61_bdd/tests/cong_b_20261026/")
    if hd.get("python") != H.PYTHON:
        loi.append(f"python {hd.get('python')!r} != {H.PYTHON}")
    if loi:
        raise LoiDauVao("hop_dong", loi)
    A, plan = duong(hd["a"]["phien"]), duong(hd["plan"]["duong"])
    if fx:
        sai = [n for n, x in (("a", A), ("plan", plan), ("review", duong(hd["review_a"]["duong"])))
               if not H._trong(x, VUNG_FIXTURE_B)]
        if sai:
            raise LoiDauVao("hop_dong", [f"hop dong fixture tro ra ngoai vung fixture: {sai}"])
    elif A != VA.PHIEN_A_THAT or plan != Path(os.path.abspath(CB.PLAN)):
        raise LoiDauVao("hop_dong", ["hop dong that phai tro phien A co dinh va handoff/PLAN_DA_CHOT.md"])
    hd["_duong"], hd["_sha256"], hd["_fixture"] = p, H.sha256_file(p), fx
    return hd


SCHEMA_GIAO_THUC = {"tap_du_lieu": {"split": str, "gt_folder": str, "manifest": {"rel": str}},
                    "tracker": {"fps": SO, "bo_loc": list}, "thong_ke": {"bootstrap": {"so_video": int, "so_mau": int}},
                    "detector": dict, "tai": {"dieu_kien": dict}}


def nap_ctx(p) -> "D.Ctx":
    """B0-F9: giao thuc + manifest no tro toi doc/kiem JSON + schema TAI LOP B truoc dieu_phoi_bdd.Ctx (khong sua helper
    A); loi doc / schema / kieu con lai trong D.Ctx / runs() / detectors() -> LoiDauVao('giao_thuc') giu nguyen nhan tho
    (exit 2, khong de JSONDecodeError/TypeError thoat thanh exit 1)."""
    p = Path(os.path.abspath(p))
    if not p.is_file() or H.la_reparse(p):
        raise LoiDauVao("giao_thuc", [f"khong doc duoc giao thuc {H.rel(p)}"])
    gt, loi = doc_json_dv(p, f"giao thuc {H.rel(p)}")
    if loi:
        raise LoiDauVao("giao_thuc", loi)
    loi = loi_schema(gt, SCHEMA_GIAO_THUC)
    fx = gt.get("che_do") == "fixture"
    if fx and not isinstance(gt.get("goc"), str):
        loi.append("goc (fixture) thieu / sai kieu")
    if not loi:
        for i, bl in enumerate(gt["tracker"]["bo_loc"]):
            if not isinstance(bl, dict) or not isinstance(bl.get("ten"), str) \
                    or not (bl.get("seeds") is None or isinstance(bl.get("seeds"), list)):
                loi.append(f"tracker.bo_loc[{i}] sai kieu (can object co ten chuoi, seeds danh sach / null)")
        fps = gt["tracker"]["fps"]                     # B0-F9: D.Ctx tinh 1.0/fps -> fps=0 la ZeroDivisionError (exit 1)
        if isinstance(fps, bool) or not math.isfinite(fps) or fps <= 0:
            loi.append(f"tracker.fps {fps!r} phai la so huu han > 0")
    if loi:
        raise LoiDauVao("giao_thuc", [f"giao thuc {H.rel(p)}: {x}" for x in loi])
    mp = (Path(gt["goc"]) if fx else H.ROOT) / gt["tap_du_lieu"]["manifest"]["rel"]
    man, loi = doc_json_dv(mp, f"manifest giao thuc {H.rel(mp)}")
    if not loi and not (isinstance(man.get("videos"), list) and all(isinstance(v, str) for v in man["videos"])):
        loi = [f"manifest giao thuc {H.rel(mp)}: videos phai la danh sach chuoi"]
    if loi:
        raise LoiDauVao("giao_thuc", loi)
    try:
        ctx = D.Ctx(p)
        ctx.runs()
        ctx.detectors()
    except LoiDauVao:
        raise
    except H.LoiHopDong as e:
        raise LoiDauVao("giao_thuc", [str(e)]) from None
    except (ValueError, TypeError, KeyError, AttributeError, OSError, ArithmeticError) as e:
        raise LoiDauVao("giao_thuc", [f"giao thuc {H.rel(p)}: {type(e).__name__}: {e}"]) from None
    return ctx


def ma_b(hd: dict) -> dict:
    """Manifest ma B (ghim rieng): 3 file runner cong_b + dong import tinh DIEM_VAO_MA (doc, khong doi) + hop dong."""
    ma = {f"r61_bdd/cong_b/{n}": H.sha256_file(CONG_B / n) for n in FILE_MA_B}
    ma.update({p.relative_to(H.SRC).as_posix(): H.sha256_file(p) for p in D.dong_import_tinh(H.SRC)})
    s = {"ma": dict(sorted(ma.items())),
         "cau_hinh": {"hop_dong_cong_b": hd["_sha256"], "hop_dong_cong_b_duong": H.rel(hd["_duong"]),
                      "hop_dong_lenh": H.sha256_file(VA.HOP_DONG), "giao_thuc_that": H.sha256_file(D.GIAO_THUC)},
         "python": H.PYTHON}
    return {**s, "sha256": H.sha256_bytes(json.dumps(s, sort_keys=True).encode("utf-8"))}


# ---------------------------------------------------------------------------
# Tung nhom (tra ve loi tho; khong nem)
# ---------------------------------------------------------------------------
def reparse_trong(goc: Path) -> list[str]:
    """Moi muc reparse (thu muc hoac file) duoi goc va tren duong toi goc -- liet_ke_file bo qua im lang nen kiem rieng."""
    out = []
    cur = Path(goc.anchor)
    for part in goc.parts[1:]:
        cur = cur / part
        if H.la_reparse(cur):
            out.append(H.rel(cur))
    for g, dirs, files in os.walk(goc, followlinks=False):
        out += [H.rel(Path(g) / x) for x in dirs + files if H.la_reparse(Path(g) / x)]
    return out


def kiem_plan(hd: dict) -> tuple[dict, list[str]]:
    p = duong(hd["plan"]["duong"])
    if not p.is_file() or H.la_reparse(p):
        return {}, [f"thieu PLAN {H.rel(p)}"]
    b = p.read_bytes()
    try:
        t = CB.tach_plan(b)
    except H.LoiHopDong as e:                         # khong tach duoc muc 0 -> loi nhom plan (khong nem tran)
        return {"sha256_toan_bo": H.sha256_bytes(b), "so_byte": len(b)}, [f"PLAN: {e}"]
    t["sha256_muc0"] = H.sha256_bytes(b[t["byte_muc0"][0]:t["byte_muc0"][1]])
    if hd["plan"].get("phep_so") != "nguyen_byte":
        return t, [f"phep_so {hd['plan'].get('phep_so')!r} chua duoc duyet (chi 'nguyen_byte')"]
    return t, [f"{k}: {t[k]} != {hd['plan'][k]}" for k in PLAN_KHOA if t[k] != hd["plan"][k]]


def kiem_review(hd: dict) -> tuple[dict, list[str]]:
    p = duong(hd["review_a"]["duong"])
    if not p.is_file() or H.la_reparse(p):
        return {}, [f"thieu review {H.rel(p)}"]
    b = p.read_bytes()
    r = {"duong": H.rel(p), "sha256": H.sha256_bytes(b), "so_byte": len(b)}
    if r["sha256"] != hd["review_a"]["sha256"]:
        return r, [f"sha256 review {r['sha256']} != {hd['review_a']['sha256']}"]
    d, loi = doc_json_dv(p, "review")
    if loi:
        return r, loi
    r["verdict"] = d.get("verdict")
    return r, ([] if r["verdict"] == hd["review_a"]["verdict"] else [f"verdict {r['verdict']!r} != {hd['review_a']['verdict']!r}"])


def kiem_phe_duyet(hd: dict, p, phien_b: Path | None, sha_ma_b: str) -> tuple[dict, list[str]]:
    if p is None:
        return {}, ["thieu --phe-duyet (phe duyet B that do orchestrator cung cap sau khi nguoi dung duyet)"]
    p = Path(os.path.abspath(p))
    if not p.is_file() or H.la_reparse(p):
        return {}, [f"khong doc duoc phe duyet {H.rel(p)}"]
    if hd["_fixture"] != H._trong(p, VUNG_FIXTURE_B) or (not hd["_fixture"] and H._trong(p, H.VUNG_TESTS)):
        return {}, ["phe duyet fixture/that khong khop che do hop dong"]
    b = p.read_bytes()
    d, loi = doc_json_dv(p, "phe duyet")
    if loi:
        return {"duong": H.rel(p), "sha256": H.sha256_bytes(b)}, loi
    a = hd["a"]
    ky = {"loai": "phe_duyet_cong_b", "duyet": True, "goi_a_sha256": a["goi_a_sha256"], "khoa_sha256": a["khoa_sha256"],
          "review_a_sha256": hd["review_a"]["sha256"], "hop_dong_cong_b_sha256": hd["_sha256"], "ma_b_sha256": sha_ma_b,
          "phien_b": None if phien_b is None else H.rel(phien_b)}
    loi = [f"{k}: {d.get(k)!r} != {v!r}" for k, v in ky.items() if d.get(k) != v]
    if not isinstance(d.get("plan_thuc_thi_sha256"), str) or not HEX64.match(d["plan_thuc_thi_sha256"]):
        loi.append("plan_thuc_thi_sha256 khong phai 64 hex")
    if not isinstance(d.get("nguoi_duyet"), str) or not d["nguoi_duyet"].strip():
        loi.append("thieu nguoi_duyet")
    return {"duong": H.rel(p), "sha256": H.sha256_bytes(b), "noi_dung": d}, loi


def kiem_goi(hd: dict, A: Path) -> tuple[dict | None, list[str]]:
    gp = A / "goi_a.json"
    if not gp.is_file() or H.la_reparse(gp):
        return None, ["thieu goi_a.json"]
    a, loi = hd["a"], []
    sha = H.sha256_file(gp)
    if sha != a["goi_a_sha256"]:
        loi.append(f"goi_a_sha256 {sha} != {a['goi_a_sha256']}")
    g, l2 = doc_json_dv(gp, "goi_a.json")             # digest lech van doc CO BAO VE de bao nguyen nhan tho (F9)
    if l2:
        return None, loi + l2
    l2 = loi_schema(g, SCHEMA_GOI, "goi_a.json ")
    if not l2 and not all(isinstance(r, str) and isinstance(h, str) and HEX64.match(h) for r, h in g["file"].items()):
        l2.append("goi_a.json file: muc khong phai {duong: sha256 64 hex}")
    if l2:
        return None, loi + l2
    for k, v in (("khoa_sha256", a["khoa_sha256"]), ("phien_A", H.rel(A)), ("stt_dong_goi_dau", a["stt_dong_goi_dau"]),
                 ("so_file", a["so_file_goi"]), ("luot", a["luot_a"])):
        if g.get(k) != v:
            loi.append(f"goi_a.json {k} {g.get(k)!r} != {v!r}")
    if len(g.get("file", {})) != g.get("so_file"):
        loi.append("goi_a.json so_file khac so muc file")
    return g, loi


def kiem_sha256sums(hd: dict, A: Path, g: dict) -> list[str]:
    p = A / "SHA256SUMS"
    if not p.is_file() or H.la_reparse(p):
        return ["thieu SHA256SUMS"]
    b = p.read_bytes()
    loi = [] if b == "".join(f"{h}  {r}\n" for r, h in sorted(g["file"].items())).encode("utf-8") else \
        ["SHA256SUMS khac danh muc goi_a.json"]
    if H.sha256_bytes(b) != hd["a"]["sha256sums_sha256"]:
        loi.append(f"sha256 SHA256SUMS {H.sha256_bytes(b)} != {hd['a']['sha256sums_sha256']}")
    return loi


def kiem_marker_goi(hd: dict, A: Path) -> list[str]:
    gx = A / "goi_a.xong.json"
    st, loi = H.kiem_marker(gx, [A / "goi_a.json", A / "SHA256SUMS"], {"buoc": "dong-goi-a", "luot": hd["a"]["luot_a"],
                                                                         "khoa_sha256": hd["a"]["khoa_sha256"]})
    return [] if st == "xong" else [f"goi_a.xong.json {st}: {loi[:4]}"]


def kiem_phu_luc(hd: dict, A: Path) -> tuple[list[dict], list[str]]:
    d = A / "van_hanh" / "dung_da_go"
    ky = {e["duong"]: e for e in hd["phu_luc_dung_da_go"]}
    co = {p.relative_to(A).as_posix(): p for p in (H.liet_ke_file(d) if d.is_dir() else [])}
    out, loi = [], []
    for r, e in ky.items():
        p = A / r
        if r not in co or not p.is_file() or H.la_reparse(p):
            loi.append(f"thieu {r}")
            continue
        sz, sha = p.stat().st_size, H.sha256_file(p)
        out.append({"duong": H.rel(p), "so_byte": sz, "sha256": sha})
        if sz != e["so_byte"]:
            loi.append(f"{r}: so_byte {sz} != {e['so_byte']}")
        if sha != e["sha256"]:
            loi.append(f"{r}: sha256 {sha} != {e['sha256']}")
    loi += [f"thua dung_da_go: {r}" for r in sorted(set(co) - set(ky))]
    return out, loi


def kiem_van_hanh(hd: dict, A: Path, g: dict) -> tuple[dict, list[str]]:
    vh = A / "van_hanh"
    cfg = hd["van_hanh_a"]
    loi = [f"chan: van_hanh/{c} ton tai" for c in cfg["chan"] if (vh / c).exists() or H.la_reparse(vh / c)]
    goi_tep, phu = set(g["file"]), {e["duong"] for e in hd["phu_luc_dung_da_go"]}
    dem = {x["ten"]: 0 for x in cfg["loai_tru"]}
    bang = {}
    for p in (H.liet_ke_file(vh) if vh.is_dir() else []):
        r_vh = p.relative_to(vh).as_posix()
        r_a = f"van_hanh/{r_vh}"
        if r_vh in cfg["chan"]:
            continue
        sz = p.stat().st_size
        if H.rel(p) in goi_tep:
            loai = "goi_a"
        elif r_a in phu:
            loai = "phu_luc"
        else:
            m = [x for x in cfg["loai_tru"] if re.match(x["mau"], r_vh)]
            if len(m) != 1:
                loi.append(f"thua van_hanh: {r_a}" + (f" (khop {len(m)} muc loai tru)" if m else ""))
                continue
            x = m[0]
            dem[x["ten"]] += 1
            loai = f"loai_tru:{x['ten']}"
            if "so_byte" in x["dieu_kien"] and sz != x["dieu_kien"]["so_byte"]:
                loi.append(f"{r_a}: so_byte {sz} != {x['dieu_kien']['so_byte']} (dieu kien loai tru {x['ten']})")
        bang[r_a] = [loai, sz, H.sha256_file(p)]
    for x in cfg["loai_tru"]:
        if dem[x["ten"]] != x["so_file"]:
            loi.append(f"loai tru {x['ten']}: {dem[x['ten']]} file != so_file {x['so_file']}")
        if "ket_qua_dat_va_goi" in x["dieu_kien"]:
            kp = A / hd["a"]["kiem_goi_a"]
            kq, l2 = doc_json_dv(kp, hd["a"]["kiem_goi_a"]) if kp.is_file() and not H.la_reparse(kp) else \
                (None, [f"thieu {hd['a']['kiem_goi_a']}"])
            if l2:
                loi.append(f"dieu kien loai tru {x['ten']}: {l2}")
            elif (kq.get("dat"), kq.get("goi_a_sha256"), kq.get("khoa_sha256")) != \
                    (True, hd["a"]["goi_a_sha256"], hd["a"]["khoa_sha256"]):
                loi.append(f"dieu kien loai tru {x['ten']}: {hd['a']['kiem_goi_a']} khong dat/khong rang buoc goi")
    return dict(sorted(bang.items())), loi


def kiem_payload(A: Path, g: dict) -> dict:
    lech = sorted(r for r, h in g["file"].items()
                  if not (H.ROOT / r).is_file() or H.la_reparse(H.ROOT / r) or H.sha256_file(H.ROOT / r) != h)
    pv = set(VA.payload(A, int(g["stt_dong_goi_dau"]))) | set(VA.liet_ke_rel(A / "goi")) | \
        set(VA.liet_ke_rel(A / "dong_goi" / "phuc_hoi"))
    thua = sorted(r for r in pv if r not in g["file"])
    return {"so_file": len(g["file"]), "lech": lech, "thua": thua}


def kiem_khoa(hd: dict, A: Path, g: dict, ctx) -> tuple[dict, list[str]]:
    a, loi = hd["a"], []
    kp = A / "khoa" / "khoa.json"
    sha = H.sha256_file(kp) if kp.is_file() else None
    if sha != a["khoa_sha256"] or g.get("khoa") != H.rel(kp):
        loi.append(f"khoa.json sha256 {sha} != {a['khoa_sha256']} hoac goi khong tro khoa")
    st, l2 = H.kiem_marker(A / "khoa" / "khoa.xong.json", [kp], {"buoc": "khoa", "luot": a["luot_a"]})
    if st != "xong":
        loi.append(f"khoa.xong.json {st}: {l2[:3]}")
    kk = A / a["kiem_khoa"]
    kq, l2 = doc_json_dv(kk, a["kiem_khoa"]) if kk.is_file() and not H.la_reparse(kk) else (None, [f"thieu {a['kiem_khoa']}"])
    if l2:
        loi += l2
    elif H.sha256_file(kk) != a["kiem_khoa_sha256"] or kq.get("dat") is not True or kq.get("khoa_sha256") != a["khoa_sha256"]:
        loi.append(f"{a['kiem_khoa']} khong dung sha256 ghim / khong dat / khong rang buoc khoa")
    khoa, l2 = doc_json_dv(kp, "khoa/khoa.json") if sha is not None else (None, [])
    loi += l2
    if loi:
        return {}, loi
    pa = f"{g['phien_A']}/cache/"
    cache = {r: h for r, h in g["file"].items() if r.startswith(pa)}
    try:
        lk = D.so_khoa(khoa, D.anh_khoa(ctx, A, cache))
        lp = VA.so_phu_luc(khoa.get("phu_luc_van_hanh", {}), VA.phu_luc(ctx, A))
    except (H.LoiHopDong, KeyError, TypeError, ValueError, OSError) as e:   # nguyen nhan tho giu trong thong diep
        return {}, [f"anh khoa hien tai: {type(e).__name__}: {e}"]
    return {"so_khoa_lech": lk, "so_phu_luc_lech": lp}, [f"anh khoa: {x}" for x in lk + lp]


# ---------------------------------------------------------------------------
# Tong hop
# ---------------------------------------------------------------------------
def kiem_tat_ca(hd: dict, ctx, phe_duyet=None, phien_b: Path | None = None, nang: bool = True) -> dict:
    """Chay MOI nhom re; nhom nang (anh khoa) chi khi cac nhom re dat. Nem LoiDauVao theo nhom loi dau tien; dict tho
    (moi nhom: dat/loi) gan vao loi.kq va tra ve khi dat. Khong dung ket qua kiem cu."""
    A = duong(hd["a"]["phien"])
    plan_cfg = Path(os.path.abspath(VA.cfg_va(ctx)["plan"]))
    if plan_cfg != duong(hd["plan"]["duong"]) or ctx.fixture != hd["_fixture"]:
        raise LoiDauVao("hop_dong", ["giao thuc va hop dong khong cung che do / PLAN"])
    kq = {"nhom": {}, "luc": time.strftime("%Y-%m-%dT%H:%M:%S")}

    def ghi(n, loi, **them):
        kq["nhom"][n] = {"dat": not loi, "loi": list(loi), **them}

    ghi("duong", [f"reparse: {r}" for r in reparse_trong(A)] if A.is_dir() else [f"thieu phien A {H.rel(A)}"])
    t, l = kiem_plan(hd)
    ghi("plan", l, tho=t)
    r, l = kiem_review(hd)
    ghi("review", l, tho=r)
    mb = ma_b(hd)
    kq["ma_b"] = mb
    pd, l = kiem_phe_duyet(hd, phe_duyet, phien_b, mb["sha256"])
    ghi("phe_duyet", l, tho=pd)
    g, l = kiem_goi(hd, A) if A.is_dir() else (None, ["thieu phien A"])
    ghi("goi", l)
    if g is not None:
        ghi("sha256sums", kiem_sha256sums(hd, A, g))
        ghi("marker_goi", kiem_marker_goi(hd, A))
        pl, l = kiem_phu_luc(hd, A)
        ghi("phu_luc", l, tho=pl)
        vb, l = kiem_van_hanh(hd, A, g)
        ghi("van_hanh", l, so_file=len(vb))
        kq["van_hanh_bang"] = vb
        pv = kiem_payload(A, g)
        ghi("payload", [f"lech: {x}" for x in pv["lech"]] + [f"thua: {x}" for x in pv["thua"]], so_file=pv["so_file"])
    re_dat = all(x["dat"] for x in kq["nhom"].values())
    if g is not None and nang and re_dat:
        k, l = kiem_khoa(hd, A, g, ctx)
        ghi("khoa", l, tho=k)
    elif nang:
        kq["nhom"]["khoa"] = {"dat": False, "loi": ["bo qua: nhom re truoc chua dat"], "bo_qua": True}
    kq["dat"] = all(x["dat"] for x in kq["nhom"].values())
    if not kq["dat"]:
        n = next(x for x in NHOM if x in kq["nhom"] and not kq["nhom"][x]["dat"])
        kq["nhom_loi_dau"] = n
        raise LoiDauVao(n, kq["nhom"][n]["loi"], kq)
    kq["manifest"] = manifest(hd, kq, g)
    return kq


def manifest(hd: dict, kq: dict, g: dict) -> dict:
    """Manifest B: dinh danh goi/khoa + PLAN tho + review + phe duyet + phu luc (duong/so byte/sha256) + MOI file van_hanh A
    (phan loai + so byte + sha256) + ma B. So lai o moi loi goi sau (so_manifest)."""
    a = hd["a"]
    return {"goi_a": f"{a['phien']}/goi_a.json", "goi_a_sha256": a["goi_a_sha256"], "khoa_sha256": a["khoa_sha256"],
            "phien_A": a["phien"], "so_file_goi": len(g["file"]),
            "plan": {k: kq["nhom"]["plan"]["tho"][k] for k in PLAN_KHOA + ("sha256_muc0",)},
            "review_a": kq["nhom"]["review"]["tho"], "phe_duyet": {k: v for k, v in kq["nhom"]["phe_duyet"]["tho"].items()
                                                                   if k != "noi_dung"},
            "phu_luc_dung_da_go": kq["nhom"]["phu_luc"]["tho"], "van_hanh_a": kq["van_hanh_bang"],
            "ma_b_sha256": kq["ma_b"]["sha256"], "hop_dong_cong_b_sha256": hd["_sha256"]}


def so_manifest(cu: dict, moi: dict) -> list[str]:
    loi = []
    for k in sorted(set(cu) | set(moi)):
        a, b = cu.get(k), moi.get(k)
        if k == "van_hanh_a" and isinstance(a, dict) and isinstance(b, dict):
            loi += [f"van_hanh_a {r} doi/thieu/thua" for r in sorted(set(a) | set(b)) if a.get(r) != b.get(r)]
        elif H.chuan_json(a) != H.chuan_json(b):
            loi.append(f"manifest {k} lech")
    return loi
