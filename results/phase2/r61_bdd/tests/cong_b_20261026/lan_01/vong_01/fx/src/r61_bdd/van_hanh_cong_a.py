"""
van_hanh_cong_a.py -- Lop VAN HANH cong A cua R61 (plan A0, 24/10/2026). CHI nhan --cong A.

Tien trinh CHA (chi thu vien chuan, khong bam du lieu lon): kiem phien / hop dong / snapshot ma, roi chay MOT tien
trinh CON tuan tu cho moi doan. TRUOC moi lan chay ghi `van_hanh/lan_chay/<stt>_<buoc>/goi.json` (buoc, detector,
argv cha/con, snapshot ma + dau vao, so lan da bat dau cua tung don vi, bo dem crash truoc do). stdout/stderr cua con
ghi THANG ra file rieng tung lan, con bat faulthandler ra `faulthandler.log` rieng; exit tho + phan loai + don vi dang
do ghi vao `ket_thuc.json`. Bo dem crash tinh lai tu MOI lan chay da ghi (khong luu bien dem rieng -> khong reset).
Tien trinh CON: goi DUNG helper hien co (dieu_phoi_bdd / hop_dong_bdd / chuan_bi_bdd) theo don vi; nhat ky don vi
`tien_do.jsonl` (flush + fsync) truoc/sau moi don vi -> cha xac dinh duoc don vi dang do khi con crash native.

Buoc (moi loi goi DUNG MOT doan: --so-doan khac 1 bi tu choi; --ngan-sach-giay <= 420):
    tien-kiem -> cache --detector <det> --device 0 (x3) -> kiem-cache -> smoke -> khoa -> kiem-khoa -> dong-goi-a
    -> kiem-goi-a
Don vi: cache = detector-video; kiem-cache = detector-video; smoke = run-video (tracking) + run (cham);
khoa / kiem-khoa / dong-goi-a / kiem-goi-a = doan bam (danh muc co dinh luc bat dau) + buoc tong hop cuoi.
Doan dung GIUA don vi khi het ngan sach mem / gioi han don vi (moi doan lam it nhat 1 don vi) -> exit 10.

Exit cha: 0 xong buoc; 10 chua xong (chay lai DUNG lenh); 2 loi hop dong / marker hong / hash-spec-provenance lech
(KHONG tu thu lai); 30 crash native da ghi, duoi nguong (chay lai DUNG lenh = thu lai don vi dang do); 40 DUNG HOI
nguoi dung (vuot nguong crash, crash khong xac dinh duoc don vi, loi Python exit 1, het gio cung, cha gian doan,
thieu log stdout/stderr/faulthandler, khong gan duoc Job Object kill-on-close cho con) -- phien
bi chan bang van_hanh/dung.json, moi loi goi sau tra 40; 1 loi bat ngo cua chinh tien trinh cha. Exit khac (vd
0xC0000005, 3) = chinh tien trinh cha crash -> loi goi sau phat hien lan chay thieu ket_thuc.json va DUNG.
Chinh sach crash (ghi chu nguoi dung, plan A): toi da 2 thu lai moi don vi; dung khi mot don vi crash lan 3, tong crash
phien dat 4, hoac cung buoc khoa / kiem-khoa / dong-goi-a crash lan 2. Khong reset bang phien moi: cong A that chi
chay tren phien co dinh PHIEN_A_THAT.
Don vi chua co marker nhung co artifact do: CHUYEN nguyen byte (sha256 truoc/sau) vao van_hanh/artifact_do/ roi moi cho
ghi lai. Marker ton tai nhung hong / hash lech -> exit 2 (khong ghi de, khong chay de).
Tiem loi (--tiem-loi) CHI voi giao thuc fixture duoi results/phase2/r61_bdd/tests/cong_a_20261024/.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hop_dong_bdd as H  # noqa: E402
import chuan_bi_bdd as CB  # noqa: E402  (chi thu vien chuan)
import dieu_phoi_bdd as D  # noqa: E402  (muc module chi thu vien chuan)

LUOT = "R61_cong_A_20261024"
PHIEN_A_THAT = H.VUNG_CHUAN_BI / "cong_a_20261024_01"
VUNG_FIXTURE = H.VUNG_TESTS / "cong_a_20261024"
CHUP_DAU_THAT = H.VUNG_CHUAN_BI / "chuan_bi_cpu" / "lan_01" / "chup_dau"
HOP_DONG = H.ROOT / "configs" / "phase2" / "r61_bdd" / "hop_dong_lenh.json"
BUOC = ("tien-kiem", "cache", "kiem-cache", "smoke", "khoa", "kiem-khoa", "dong-goi-a", "kiem-goi-a")
BUOC_KHOA = ("khoa", "kiem-khoa", "dong-goi-a")
EXIT_CRASH = 30               # khong trung exit 3 cua CRT (abort / raise SIGSEGV) hay 0xC0000005
EXIT_DUNG = 40
NGUONG = {"don_vi": 3, "phien": 4, "buoc_khoa": 2}
LOAI_CRASH = ("crash_native", "cha_gian_doan")
NGAN_SACH_TOI_DA = 420.0
DU_PHONG_CUNG = 150.0          # het gio CUNG cua con = ngan sach + du phong (< 600 s); KHONG phai exit 10
KT_DOAN_BAM = 64
#: Thu muc payload goi A (bam toan bo); van_hanh/ chi gom phien.json, artifact_do/, lan chay truoc dong-goi-a.
GOC_PAYLOAD = ("tien_kiem", "tien_kiem_a", "cache", "kiem_cache", "smoke", "khoa", "kiem_khoa")
LOAI_TRU_GOI = {
    "_tam/, _cau_hinh_thu_vien/": "tam/cau hinh thu vien cua phien (thu vien co the ghi)",
    "dong_goi/ (tru dong_goi/phuc_hoi/)": "vung lam viec dong goi (danh muc + doan bam); RIENG dong_goi/phuc_hoi/ (sao "
                                          "luu phuc hoi sau khi danh muc co dinh) NAM TRONG file, bam rieng (A0-F2a)",
    "goi_a.json, SHA256SUMS, goi_a.xong.json": "chinh goi (marker goi_a.xong.json bam goi_a.json + SHA256SUMS)",
    "van_hanh/so_crash.json, van_hanh/dung.json, van_hanh/goi_cha.jsonl, van_hanh/cha/":
        "nhat ky dieu phoi dang mo / dan xuat (ban chot: goi/nhat_ky_crash_chot.json)",
    "van_hanh/lan_chay/<stt >= stt_dong_goi_dau>": "lan chay dong-goi-a va sau do (dang mo khi dong goi)",
    "van_hanh/kiem_khoa_sau_goi/, van_hanh/kiem_goi_a/": "kiem sau dong goi (chi doc payload; sao luu phuc hoi cua "
                                                         "kiem nam trong vong_NN/phuc_hoi/, khong vao payload -- A0-F2b)",
}
#: write_paths R61_PREP (giong chup-cuoi) va plan cong A -- doi chieu pham vi cua tien kiem A.
WP_PREP = ("src/r61_bdd/", "configs/phase2/r61_bdd/", "results/phase2/r61_bdd/tests/",
           "results/phase2/r61_bdd/chuan_bi/chuan_bi_cpu/", "handoff/logs/review61/01_quyet_dinh_va_giao_thuc.md",
           "handoff/logs/review61/02_ket_qua_chuan_bi.md", "handoff/PLAN_DA_CHOT.md")
WP_A = ("src/r61_bdd/van_hanh_cong_a.py", "src/r61_bdd/test_van_hanh_cong_a.py", "src/r61_bdd/dieu_phoi_bdd.py",
        "configs/phase2/r61_bdd/hop_dong_lenh.json", "results/phase2/r61_bdd/tests/cong_a_20261024/",
        "results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01/", "handoff/PLAN_DA_CHOT.md",
        "handoff/logs/review61/03_bang_chung_cong_a.md")
DAU_NAP = "[NAP_DETECTOR_MO_PHONG] "
#: Phien da qua kiem luot/giao thuc trong loi goi nay; chi khi co moi ghi goi_cha.jsonl (A0-F5).
PHIEN_CHAP_NHAN: Path | None = None
#: Job Object KILL_ON_JOB_CLOSE cua cha: cha chet (ke ca crash native) -> kernel dong handle -> con bi giet (A0-F4).
_JOB = None


def cfg_va(ctx) -> dict:
    """PLAN / baseline / kich thuoc doan bam: hang so cho cong A that; fixture khai bao trong giao thuc (van_hanh_a)."""
    if not ctx.fixture:
        return {"plan": CB.PLAN, "chup_dau": CHUP_DAU_THAT, "kt": KT_DOAN_BAM}
    v = ctx.gt["van_hanh_a"]
    return {"plan": ctx.duong(v["plan"]), "chup_dau": ctx.duong(v["chup_dau"]),
            "kt": int(v.get("kich_thuoc_doan_bam", KT_DOAN_BAM))}


def ghi_dong(p: Path, obj, vung) -> None:
    """Them MOT dong JSON (flush + fsync) -- nhat ky chi noi them."""
    H.kiem_duong_ghi(p, vung)
    H.tao_thu_muc(p.parent, vung)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def liet_ke_rel(d: Path) -> list[str]:
    return [H.rel(p) for p in H.liet_ke_file(d) if p.is_file()] if d.is_dir() else []


# ---------------------------------------------------------------------------
# Tien kiem A: chup + doi chieu TRUC TIEP baseline lan_01 (khong goi chup-cuoi cu)
# ---------------------------------------------------------------------------
def chup_a(plan_p: Path, thu_muc: Path, vung) -> dict:
    """Cung noi dung chuan_bi_bdd.chup (PLAN nguyen byte, git status/diff, nguon bao ve, vung nguy co, pyc, settings)
    nhung ghi vao vung cua phien A va doc PLAN tu plan_p."""
    H.tao_thu_muc(thu_muc, vung)
    plan_b = plan_p.read_bytes()
    H.ghi_bytes(thu_muc / "PLAN_DA_CHOT.md.bytes", plan_b, vung)
    st = CB.git("status", "--short", "--untracked-files=all")
    H.ghi_bytes(thu_muc / "git_status_short.txt", st, vung)
    df, dc = CB.git("diff"), CB.git("diff", "--cached")
    H.ghi_bytes(thu_muc / "git_diff.patch", df, vung)
    H.ghi_bytes(thu_muc / "git_diff_cached.patch", dc, vung)     # index so voi HEAD (noi dung staged)
    nguon = {r: (H.sha256_file(H.ROOT / r) if (H.ROOT / r).is_file() else None) for r in CB.NGUON_BAO_VE}
    nguy_co = {d: H.liet_ke_nhe(H.ROOT / d, de_quy) for d, de_quy in CB.VUNG_NGUY_CO}
    pyc = sorted(H.rel(p) for p in H.liet_ke_file(H.SRC) if p.suffix == ".pyc")
    out = {"head": CB.git("rev-parse", "HEAD").decode().strip(),
           "nhanh": CB.git("rev-parse", "--abbrev-ref", "HEAD").decode().strip(),
           "git_status_sha256": H.sha256_bytes(st), "git_diff_sha256": H.sha256_bytes(df),
           "git_diff_cached_sha256": H.sha256_bytes(dc), "plan": CB.tach_plan(plan_b), "plan_duong": H.rel(plan_p),
           "nguon_bao_ve": nguon, "appdata_ultralytics": CB.appdata_ultralytics(), "pyc_trong_src": pyc,
           "so_muc_nguy_co": {k: len(v) for k, v in nguy_co.items()}}
    H.ghi_json(thu_muc / "vung_nguy_co.json", nguy_co, vung)
    H.ghi_json(thu_muc / "chup.json", out, vung)
    return out


def pham_vi(r: str) -> str | None:
    if CB.duoc_phep(r, list(WP_A)):
        return "pham_vi_a"
    if CB.duoc_phep(r, list(WP_PREP)):
        return "pham_vi_prep"
    if any(w.startswith(r.rstrip("/") + "/") for w in WP_A + WP_PREP):
        return "to_tien_pham_vi"            # thu muc cha cua write path: mtime doi khi tao con
    return None


def _duong_status(ln: str) -> str:
    r = ln[3:].strip().strip('"')
    return r.split(" -> ")[-1] if " -> " in r else r


def tach_diff(b: bytes | None) -> dict[str, str]:
    """Patch git diff -> {duong: sha256 khoi 'diff --git' cua file do} (so noi dung tung file, khong chi ten)."""
    out, ten, khoi = {}, None, []
    for ln in (b or b"").splitlines(keepends=True):
        if ln.startswith(b"diff --git "):
            if ten is not None:
                out[ten] = H.sha256_bytes(b"".join(khoi))
            ten = ln.decode("utf-8", "replace").rstrip("\r\n").split(" b/", 1)[-1].strip('"')
            khoi = []
        khoi.append(ln)
    if ten is not None:
        out[ten] = H.sha256_bytes(b"".join(khoi))
    return out


LOAI_NGOAI_LE = {"pyc_moi": {"loai", "duong", "ghi_chu"}, "mtime_thu_muc": {"loai", "vung", "duong", "ghi_chu"}}


def ngoai_le_duyet(hd: dict | None = None) -> list[dict]:
    """Danh sach ngoai le baseline DA DUYET (Pe27e304a) tu hop_dong_lenh.json; cau truc sai -> LoiHopDong (khong doan)."""
    hd = H.doc_json(HOP_DONG) if hd is None else hd
    ds = hd["van_hanh_cong_a"]["ngoai_le_baseline_duyet"]["muc"]
    if not isinstance(ds, list):
        raise H.LoiHopDong("ngoai_le_baseline_duyet.muc khong phai danh sach")
    for m in ds:
        if not isinstance(m, dict) or m.get("loai") not in LOAI_NGOAI_LE or set(m) != LOAI_NGOAI_LE[m["loai"]] \
                or not isinstance(m["duong"], str) or m["duong"] != m["duong"].strip("/") or "*" in m["duong"]:
            raise H.LoiHopDong(f"ngoai_le_baseline_duyet: muc sai cau truc: {m}")
        if m["loai"] == "pyc_moi" and not (m["duong"].startswith("src/") and m["duong"].endswith(".pyc")):
            raise H.LoiHopDong(f"ngoai_le_baseline_duyet: pyc_moi phai la file .pyc duoi src/: {m['duong']}")
    if len({(m["loai"], m.get("vung"), m["duong"]) for m in ds}) != len(ds):
        raise H.LoiHopDong("ngoai_le_baseline_duyet: muc trung")
    return ds


def so_sanh_chup(dau: dict, cuoi: dict, nc_dau: dict, nc_cuoi: dict, st_dau: list[str], st_cuoi: list[str],
                 diff_dau: bytes = b"", diff_cuoi: bytes = b"", cached_dau: bytes | None = b"",
                 cached_cuoi: bytes = b"", duyet: list[dict] | None = None) -> dict:
    """Doi chieu THO voi baseline. Ngoai le: muc '.git' (vung '.') chi doi mtime, va DUNG cac muc `duyet`
    (pyc_moi: dung duong pyc; mtime_thu_muc: dung (vung, duong), thu muc o ca hai dau, chi mtime doi) -> ghi rieng
    (ngoai_le_duyet), khong mien noi dung Git (HEAD, nhanh, git status, noi dung git diff tung file, index = git diff
    --cached, .gitignore/.gitattributes ... van kiem). cached_dau=None: baseline khong ghi duoc index -> loi."""
    loi, ngoai_le, doi_nc, nl_duyet = [], [], [], []
    duyet = duyet or []
    pyc_duyet = {m["duong"] for m in duyet if m["loai"] == "pyc_moi"}
    mt_duyet = {(m["vung"], m["duong"]) for m in duyet if m["loai"] == "mtime_thu_muc"}
    da, dc = tach_diff(diff_dau), tach_diff(diff_cuoi)
    diff_doi = [{"file": r, "dau": da.get(r), "cuoi": dc.get(r), "pham_vi": pham_vi(r)}
                for r in sorted(set(da) | set(dc)) if da.get(r) != dc.get(r)]
    loi += [f"noi dung git diff doi ngoai pham vi: {x['file']}" for x in diff_doi if x["pham_vi"] is None]
    if cached_dau is None:
        loi.append("baseline khong co index (git diff --cached) va co dong staged -- khong doi chieu duoc index")
        ia, ic = {}, tach_diff(cached_cuoi)
    else:
        ia, ic = tach_diff(cached_dau), tach_diff(cached_cuoi)
    index_doi = sorted(r for r in set(ia) | set(ic) if ia.get(r) != ic.get(r))
    loi += [f"index Git (git diff --cached) doi: {r}" for r in index_doi]      # khong ai duoc stage: moi doi la loi
    for k in ("sha256_truoc_muc0", "sha256_sau_muc0"):
        if dau["plan"][k] != cuoi["plan"][k]:
            loi.append(f"PLAN ngoai muc 0 khac baseline: {k}")
    nguon_doi = sorted(r for r in set(dau["nguon_bao_ve"]) | set(cuoi["nguon_bao_ve"])
                       if dau["nguon_bao_ve"].get(r) != cuoi["nguon_bao_ve"].get(r))
    loi += [f"nguon bao ve doi: {r}" for r in nguon_doi]
    if dau["appdata_ultralytics"] != cuoi["appdata_ultralytics"]:
        loi.append("settings.json Ultralytics trong AppData doi")
    for p in sorted(set(cuoi["pyc_trong_src"]) - set(dau["pyc_trong_src"])):
        if p in pyc_duyet:
            nl_duyet.append({"loai": "pyc_moi", "muc": p, "dau": None, "cuoi": "co trong pyc_trong_src",
                             "phan_loai": "ngoai_le_duyet_pyc_moi", "loi_tho": f"pyc moi: {p}"})
        else:
            loi.append(f"pyc moi: {p}")
    for d in sorted(set(nc_dau) | set(nc_cuoi)):
        a, b = nc_dau.get(d, {}), nc_cuoi.get(d, {})
        for r in sorted(set(a) | set(b)):
            if a.get(r) == b.get(r):
                continue
            m = {"vung": d, "muc": r, "dau": a.get(r), "cuoi": b.get(r)}
            if d == "." and r == ".git" and a.get(r) and b.get(r) and a[r][0] == b[r][0]:
                m["phan_loai"] = "ngoai_le_mtime_thu_muc_git"
                ngoai_le.append(m)
            elif (d, r) in mt_duyet and a.get(r) and b.get(r) and a[r][0] == b[r][0] == -1:
                m["phan_loai"] = "ngoai_le_duyet_mtime_thu_muc"
                nl_duyet.append({**m, "loai": "mtime_thu_muc", "loi_tho": f"vung nguy co doi ngoai pham vi: {d}:{r}"})
            else:
                m["phan_loai"] = pham_vi(r) or "ngoai_pham_vi"
                if m["phan_loai"] == "ngoai_pham_vi":
                    loi.append(f"vung nguy co doi ngoai pham vi: {d}:{r}")
            doi_nc.append(m)
    moi, mat = sorted(set(st_cuoi) - set(st_dau)), sorted(set(st_dau) - set(st_cuoi))
    st_doi = [{"dong": ln, "loai": lo, "pham_vi": pham_vi(_duong_status(ln))} for lo, ds in (("moi", moi), ("mat", mat))
              for ln in ds]
    loi += [f"git status {x['loai']} ngoai pham vi: {x['dong']}" for x in st_doi if x["pham_vi"] is None]
    if dau["head"] != cuoi["head"] or dau["nhanh"] != cuoi["nhanh"]:
        loi.append("HEAD/nhanh doi")
    return {"dat": not loi, "loi": loi, "ngoai_le": ngoai_le, "ngoai_le_duyet": nl_duyet, "danh_sach_duyet": duyet,
            "vung_nguy_co_doi": doi_nc, "git_status_doi": st_doi,
            "nguon_bao_ve_doi": nguon_doi, "git_diff_doi": diff_doi, "index_doi": index_doi,
            "head": [dau["head"], cuoi["head"]], "nhanh": [dau["nhanh"], cuoi["nhanh"]],
            "plan_dau": dau["plan"], "plan_cuoi": cuoi["plan"], "so_nguon_bao_ve": len(dau["nguon_bao_ve"]),
            "write_paths_prep": list(WP_PREP), "write_paths_a": list(WP_A)}


def doi_chieu_baseline(chup_dau: Path, chup_cuoi: Path) -> dict:
    def st(p):
        return (p / "git_status_short.txt").read_bytes().decode("utf-8").splitlines()
    cd = chup_dau / "git_diff_cached.patch"
    if cd.is_file():
        cached_dau = cd.read_bytes()
    else:       # baseline lan_01 (chuan_bi_bdd.chup) khong ghi --cached: index rong <=> khong dong staged nao
        cached_dau = None if any(ln[:1] not in (" ", "?") for ln in st(chup_dau) if ln) else b""
    vao = (H.doc_json(chup_dau / "chup.json"), H.doc_json(chup_cuoi / "chup.json"),
           H.doc_json(chup_dau / "vung_nguy_co.json"), H.doc_json(chup_cuoi / "vung_nguy_co.json"),
           st(chup_dau), st(chup_cuoi), (chup_dau / "git_diff.patch").read_bytes(),
           (chup_cuoi / "git_diff.patch").read_bytes(), cached_dau, (chup_cuoi / "git_diff_cached.patch").read_bytes())
    duyet = ngoai_le_duyet()
    tho = so_sanh_chup(*vao)                                # ket qua THO: chi ngoai le .git
    out = so_sanh_chup(*vao, duyet=duyet)
    out["ket_qua_tho_khong_ngoai_le_duyet"] = {"dat": tho["dat"], "loi": tho["loi"]}
    out["hop_dong_lenh_sha256"] = H.sha256_file(HOP_DONG)
    for m in out["ngoai_le_duyet"]:                         # ban ghi tho cua file pyc tai thoi diem doi chieu
        if m["loai"] == "pyc_moi":
            p = H.ROOT / m["muc"]
            m["file"] = ({"sha256": H.sha256_file(p), "kich_thuoc": os.stat(p).st_size, "mtime_ns": os.stat(p).st_mtime_ns}
                         if p.is_file() and not H.la_reparse(p) else None)
    out["baseline_index_suy_ra_tu_status"] = not cd.is_file()
    out["baseline"] = H.rel(chup_dau)
    out["baseline_chup_sha256"] = H.sha256_file(chup_dau / "chup.json")
    return out


def kiem_plan(ctx) -> dict:
    """PLAN hien tai: phan NGOAI muc 0 phai trung baseline lan_01 (truoc moi nap detector / inference / khoa)."""
    cv = cfg_va(ctx)
    t = CB.tach_plan(cv["plan"].read_bytes())
    dau = H.doc_json(cv["chup_dau"] / "chup.json")["plan"]
    lech = [k for k in ("sha256_truoc_muc0", "sha256_sau_muc0") if t[k] != dau[k]]
    if lech:
        raise H.LoiHopDong(f"PLAN ngoai muc 0 khac baseline lan_01: {lech}")
    return t


# ---------------------------------------------------------------------------
# Tien trinh CON: nhat ky don vi, tiem loi (fixture), sao luu artifact do, bam theo doan
# ---------------------------------------------------------------------------
TIEM: list[dict] = []
T0 = time.time()


def tiem(diem: str, dv: str | None = None) -> None:
    """Tiem loi FIXTURE: chi khi goi.json co tiem_loi (cha chi chap nhan voi giao thuc fixture; con kiem lai)."""
    for t in TIEM:
        if t.get("diem") == diem and t.get("don_vi") in (None, dv):
            if t.get("kieu") == "python":
                raise RuntimeError(f"[TIEM LOI PYTHON] {diem} {dv}")
            if t.get("kieu") == "cho":                    # con dang chay khi cha bi tiem crash (A0-F4)
                print(f"[CHO] pid {os.getpid()} {diem}", flush=True)
                time.sleep(float(t.get("giay", 60)))
                continue
            crash_native()


def crash_native() -> None:
    """Access violation THAT (doc NULL, 0xC0000005 tren Windows), khong qua raise(SIGSEGV) cua CRT. CHI fixture."""
    import faulthandler
    faulthandler._read_null()


class NhatKy:
    """tien_do.jsonl: bat_dau (pha chuan_bi/chay/kiem_marker, lan_thu) -> xong | bo_qua; cha doc khi con crash."""

    def __init__(self, p: Path, vung, truoc: dict, ngan_sach: float, gioi_han: int | None, A: Path, stt: int):
        self.p, self.vung, self.truoc, self.A, self.stt = p, vung, truoc, A, stt
        self.ngan_sach, self.gioi_han = ngan_sach, gioi_han
        self.hien_tai, self.pha, self.lam, self.t0 = None, None, 0, time.time()
        #: Goc sao luu artifact do. Mac dinh van_hanh/artifact_do (thuoc payload, chi dung TRUOC khi danh muc goi co
        #: dinh); dong-goi-a sau danh muc / kiem-goi-a / kiem-khoa dat vao vung lam viec rieng (A0-F2a/F2b).
        self.goc_do: Path | None = None

    def ghi(self, **d):
        ghi_dong(self.p, {"t": round(time.time(), 3), **d}, self.vung)

    def bat_dau(self, dv: str, pha: str = "chay"):
        self.hien_tai, self.pha, self.t0 = dv, pha, time.time()
        self.ghi(su_kien="bat_dau", don_vi=dv, pha=pha,
                 lan_thu=self.truoc.get(dv, 0) + 1 if pha != "kiem_marker" else None)
        if pha != "kiem_marker":
            tiem("truoc_ghi", dv)

    def xong(self, dv: str, **info):
        tiem("sau_marker", dv)
        self.ghi(su_kien="xong", don_vi=dv, giay=round(time.time() - self.t0, 3), **info)
        if self.pha == "chay":
            self.lam += 1
        self.hien_tai = self.pha = None

    def bo_qua(self, dv: str):
        self.ghi(su_kien="bo_qua", don_vi=dv, giay=round(time.time() - self.t0, 3))
        self.hien_tai = self.pha = None

    def kiem_doan(self, mo_ta: str):
        """Dung GIUA don vi: da lam >= 1 don vi trong doan VA (het ngan sach mem HOAC dat gioi han don vi)."""
        if self.lam >= 1 and (time.time() - T0 > self.ngan_sach or
                              (self.gioi_han is not None and self.lam >= self.gioi_han)):
            raise H.ChuaXong(f"{mo_ta}: dung giua don vi sau {self.lam} don vi moi")


def sao_luu_do(nk: NhatKy, A: Path, vung, paths: list[Path], dv: str, stt: int) -> list[dict]:
    """Artifact do (don vi chua marker hop le): CHUYEN nguyen byte vao <goc_do>/<don vi>/lan_<stt>/ (goc_do mac dinh
    van_hanh/artifact_do), bam sha256 truoc va sau khi chuyen; chi sau do helper moi duoc ghi lai."""
    co = [p for p in paths if p.exists() or H.la_reparse(p)]
    if not co:
        return []
    goc = (nk.goc_do or A / "van_hanh" / "artifact_do") / dv.replace(":", "__")
    dich, k = goc / f"lan_{stt:04d}", 1
    while dich.exists() or H.la_reparse(dich):           # cung don vi sao luu nhieu lan trong mot lan chay: khong de
        k += 1
        dich = goc / f"lan_{stt:04d}_{k}"
    out = []
    for p in co:
        if H.la_reparse(p):
            raise H.LoiHopDong(f"artifact do {H.rel(p)} la reparse point")
        q = dich / p.relative_to(A)
        H.kiem_duong_ghi(p, vung)
        H.kiem_duong_ghi(q, vung)
        H.tao_thu_muc(q.parent, vung)
        tep = [p] if p.is_file() else [x for x in H.liet_ke_file(p) if x.is_file()]
        truoc = {x.relative_to(p).as_posix() if p.is_dir() else "": H.sha256_file(x) for x in tep}
        os.replace(p, q)
        sau = {k: H.sha256_file(q / k if k else q) for k in truoc}
        if sau != truoc:
            raise H.LoiHopDong(f"sao luu artifact do {H.rel(p)}: sha256 doi khi chuyen")
        out.append({"tu": H.rel(p), "den": H.rel(q), "sha256": truoc})
    H.ghi_json(dich / "sao_luu.json", {"don_vi": dv, "stt": stt, "muc": out}, vung)
    nk.ghi(su_kien="sao_luu_do", don_vi=dv, den=H.rel(dich), muc=out)
    return out


def kem_dang_ghi(paths: list[Path]) -> list[Path]:
    """Moi artifact kem file tam '<ten>.dang_ghi' cua H.ghi_bytes (ghi do dang khi crash) -> cung duoc sao luu."""
    return [x for p in paths for x in (p, p.with_name(p.name + ".dang_ghi"))]


def tiem_dang_ghi(nk: NhatKy, p: Path, b: bytes) -> None:
    """Tiem FIXTURE 'dang_ghi': mo phong crash GIUA H.ghi_bytes -- nua dau bytes da vao '<ten>.dang_ghi' (fsync), chua
    os.replace, roi access violation. Khong co tiem_loi (moi lan chay that) -> khong lam gi."""
    for t in TIEM:
        if t.get("diem") == "dang_ghi" and t.get("don_vi") in (None, nk.hien_tai) and t.get("tep") in (None, p.name):
            tam = H.kiem_duong_ghi(p.with_name(p.name + ".dang_ghi"), nk.vung)
            H.tao_thu_muc(tam.parent, nk.vung)
            with open(tam, "wb") as f:
                f.write(b[:max(1, len(b) // 2)])
                f.flush()
                os.fsync(f.fileno())
            crash_native()


def ghi_bao_toan(nk: NhatKy, p: Path, obj) -> str:
    """Ghi ket qua kiem/bam: ban cu chua marker va '.dang_ghi' -> sao luu nguyen byte TRUOC; khong ghi de (A0-F6)."""
    sao_luu_do(nk, nk.A, nk.vung, kem_dang_ghi([p]), nk.hien_tai or "ngoai_don_vi", nk.stt)
    tiem_dang_ghi(nk, p, (json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
    return H.ghi_json(p, obj, nk.vung)


def ghi_marker_bt(nk: NhatKy, p: Path, artifact: list[Path], them: dict) -> dict:
    """Ghi marker: '<marker>.dang_ghi' con sot (crash giua luc ghi marker) -> sao luu nguyen byte TRUOC, vi H.ghi_bytes
    mo file tam bang 'wb' se ghi de bang chung do (A0-F6; helper giu nguyen)."""
    sao_luu_do(nk, nk.A, nk.vung, [p.with_name(p.name + ".dang_ghi")], nk.hien_tai or "ngoai_don_vi", nk.stt)
    tiem_dang_ghi(nk, p, json.dumps({"artifact": {H.rel(a): None for a in artifact}, **them}).encode("utf-8"))
    return H.ghi_marker(p, artifact, nk.vung, them)


def bam_theo_doan(nk: NhatKy, ten: str, wd: Path, liet_ke, vung, kt: int, them: dict | None = None) -> dict:
    """Bam theo doan: danh muc CO DINH luc bat dau (wd/danh_muc.json), moi doan mot don vi (wd/doan/<k>.json ghi
    [sha256, size, mtime_ns]); tong hop cuoi kiem DU moi doan, dung tap file tung doan, va stat lai MOI file (doi sau
    khi bam -> loi). Tra {rel: sha256 | None (thieu)}."""
    dm_p = wd / "danh_muc.json"
    if not dm_p.is_file():
        nk.bat_dau(f"{ten}:danh_muc")
        files = sorted(liet_ke())
        doan = [[i, min(i + kt, len(files))] for i in range(0, len(files), kt)]
        ghi_bao_toan(nk, dm_p, {"buoc": ten, "file": files, "doan": doan, **(them or {})})
        nk.xong(f"{ten}:danh_muc", so_file=len(files), so_doan=len(doan))
    dm = H.doc_json(dm_p)
    for k, (i0, i1) in enumerate(dm["doan"]):
        p = wd / "doan" / f"{k:04d}.json"
        if p.is_file():
            continue
        nk.kiem_doan(f"{ten}: bam")
        dv = f"{ten}:bam:{k:04d}"
        nk.bat_dau(dv)
        out = {}
        for r in dm["file"][i0:i1]:
            a = H.ROOT / r
            if not a.is_file() or H.la_reparse(a):
                out[r] = [None, None, None]
                continue
            st = os.stat(a)
            out[r] = [H.sha256_file(a), st.st_size, st.st_mtime_ns]
        ghi_bao_toan(nk, p, out)          # ghi_bytes: .dang_ghi roi os.replace; .dang_ghi cu duoc sao luu truoc
        nk.xong(dv, so_file=len(out))
    dv = f"{ten}:tong_hop_bam"
    nk.bat_dau(dv, "chuan_bi")
    tong = {}
    for k, (i0, i1) in enumerate(dm["doan"]):
        d = H.doc_json(wd / "doan" / f"{k:04d}.json")
        if sorted(d) != dm["file"][i0:i1]:
            raise H.LoiHopDong(f"{ten}: doan bam {k} khong dung danh muc")
        tong.update(d)
    doi = []
    for r, (h, sz, mt) in tong.items():
        a = H.ROOT / r
        now = None if not a.is_file() else [os.stat(a).st_size, os.stat(a).st_mtime_ns]
        if now != ([sz, mt] if h is not None else None):
            doi.append(r)
    if doi:
        raise H.LoiHopDong(f"{ten}: {len(doi)} file doi/mat sau khi bam (vd {doi[:3]}) -- khong dung ket qua bam")
    nk.xong(dv, so_file=len(tong))
    return {r: v[0] for r, v in tong.items()}


# ---------------------------------------------------------------------------
# Con: dieu kien chung
# ---------------------------------------------------------------------------
def file_tien_kiem(A: Path) -> list[Path]:
    c = A / "tien_kiem_a" / "chup"
    return [A / "tien_kiem" / "tien_kiem.json", A / "tien_kiem_a" / "so_sanh.json", A / "tien_kiem_a" / "moi_truong.json",
            c / "chup.json", c / "PLAN_DA_CHOT.md.bytes", c / "git_status_short.txt", c / "git_diff.patch",
            c / "git_diff_cached.patch", c / "vung_nguy_co.json"]


def can_xong(p: Path, artifact: list[Path], danh_tinh: dict, ten: str) -> None:
    if not H.da_xong(p, artifact, danh_tinh):              # hong -> LoiHopDong trong da_xong
        raise H.LoiHopDong(f"chua co {ten} hop le ({H.rel(p)})")


def can_tien_kiem(ctx, A: Path):
    can_xong(A / "tien_kiem_a" / "xong.json", file_tien_kiem(A), {"buoc": "tien-kiem", "luot": LUOT}, "tien-kiem A")
    kiem_plan(ctx)
    return D.nap_tien_kiem(ctx, A)                         # nguon THUC TE (bam lai) == tien kiem + giao thuc


def can_kiem_cache(A: Path) -> dict:
    can_xong(A / "kiem_cache" / "xong.json", [A / "kiem_cache" / "ket_qua.json"], {"buoc": "kiem-cache"}, "kiem-cache")
    kq = H.doc_json(A / "kiem_cache" / "ket_qua.json")
    if not kq.get("dat"):
        raise H.LoiHopDong("kiem-cache khong dat")
    return kq


def marker_smoke(ctx, A: Path) -> dict:
    """{rel marker: tuple duong} cua DU 36 run x video smoke (tracking) + 36 run cham, theo giao thuc."""
    root, vids = A / "smoke", list(ctx.gt["smoke_cong_A"]["videos"])
    out = {}
    for r in ctx.runs():
        rd = root / "tracking" / ctx.split / r["run"]
        for v in vids:
            out[H.rel(rd / "xong" / f"{v}.xong.json")] = ("tracking", rd, r["run"], v)
        out[H.rel(root / "cham" / r["run"] / "cham.xong.json")] = ("cham", root / "cham" / r["run"], r["run"], None)
    return out


def can_smoke(ctx, A: Path) -> dict:
    """Smoke DU va HOP LE truoc khoa/dong goi (A0-F1): marker tong hop + MOI marker tracking/cham (bam artifact,
    danh tinh) + bam marker trung voi ban ghi trong ket_qua_smoke.json + so dem dung ky vong."""
    can_xong(A / "smoke" / "xong.json", [A / "smoke" / "ket_qua_smoke.json"], {"buoc": "smoke"}, "smoke")
    kq = H.doc_json(A / "smoke" / "ket_qua_smoke.json")
    ms = marker_smoke(ctx, A)
    thieu = []
    for r, (loai, d, run, v) in ms.items():               # marker hong / bam lech -> LoiHopDong trong da_xong
        if not (D.xong_don_vi(d, run, v) if loai == "tracking" else D.xong_cham(d, run)):
            thieu.append(r)
    if thieu:
        raise H.LoiHopDong(f"smoke thieu {len(thieu)} marker tracking/cham hop le (vd {thieu[:3]}) -- khong khoa/dong goi")
    n_t = sum(1 for x in ms.values() if x[0] == "tracking")
    if not kq.get("dat") or kq.get("so_tracking") != n_t or kq.get("so_cham") != len(ms) - n_t:
        raise H.LoiHopDong("smoke: ket_qua_smoke.json khong dat / so tracking-cham khac ky vong")
    lech = sorted(r for r in ms if (kq.get("marker_sha256") or {}).get(r) != H.sha256_file(H.ROOT / r))
    if lech or set(kq.get("marker_sha256") or {}) != set(ms):
        raise H.LoiHopDong(f"smoke: marker khac ban ghi trong ket_qua_smoke.json (vd {lech[:3]})")
    return kq


def can_khoa(A: Path) -> Path:
    kp = A / "khoa" / "khoa.json"
    can_xong(A / "khoa" / "khoa.xong.json", [kp], {"buoc": "khoa", "luot": LUOT}, "khoa")
    return kp


def phu_luc(ctx, A: Path) -> dict:
    """Phan khoa them cua lop van hanh: PLAN nguyen byte (+ ngoai muc 0), hop dong van hanh, giao thuc, ma r61_bdd/*.py,
    tien kiem A, kiem-cache, smoke."""
    r61 = H.SRC / "r61_bdd"
    return {"plan": CB.tach_plan(cfg_va(ctx)["plan"].read_bytes()), "hop_dong_lenh_sha256": H.sha256_file(HOP_DONG),
            "giao_thuc_sha256": H.sha256_file(ctx.gt_path),
            "ma_r61": {f"r61_bdd/{n}": H.sha256_file(r61 / n) for n in sorted(os.listdir(r61)) if n.endswith(".py")},
            "tien_kiem_a": H.bang_hash_thu_muc(A / "tien_kiem_a"), "kiem_cache": H.bang_hash_thu_muc(A / "kiem_cache"),
            "smoke": H.bang_hash_thu_muc(A / "smoke")}


def so_phu_luc(cu: dict, moi: dict) -> list[str]:
    loi = []
    for k in sorted(set(cu) | set(moi)):
        a, b = cu.get(k), moi.get(k)
        if isinstance(a, dict) and isinstance(b, dict):
            loi += [f"phu_luc {k}: {f} doi/thieu/thua" for f in sorted(set(a) | set(b))
                    if H.chuan_json(a.get(f)) != H.chuan_json(b.get(f))]
        elif H.chuan_json(a) != H.chuan_json(b):
            loi.append(f"phu_luc {k} lech")
    return loi


def ram() -> dict | None:
    try:
        import ctypes

        class M(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("tong", "trong", "a", "b", "c", "d", "e")]
        m = M()
        m.dwLength = ctypes.sizeof(M)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
            return None
        return {"tong_byte": int(m.tong), "trong_byte": int(m.trong)}
    except Exception:  # noqa: BLE001 -- khong do duoc -> None (ghi ro trong bao cao)
        return None


def bo_nho_tien_trinh() -> dict:
    out = {"dinh_working_set_byte": None, "cuda_dinh_cap_phat_byte": None}
    try:
        import ctypes
        from ctypes import wintypes

        class P(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + \
                       [(n, ctypes.c_size_t) for n in ("Peak", "WS", "a", "b", "c", "d", "PF", "PeakPF")]
        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        ps = ctypes.windll.psapi
        ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(P), wintypes.DWORD]
        p = P()
        p.cb = ctypes.sizeof(P)
        if ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(p), p.cb):
            out["dinh_working_set_byte"] = int(p.Peak)
    except Exception:  # noqa: BLE001
        pass
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
            out["cuda_dinh_cap_phat_byte"] = int(torch.cuda.max_memory_allocated())
    except Exception:  # noqa: BLE001
        pass
    return out


def moi_truong(ctx, A: Path) -> dict:
    import platform
    import shutil
    du = shutil.disk_usage(A)
    d = {"python": sys.version, "nen_tang": platform.platform(), "so_cpu": os.cpu_count(),
         "dia_byte": {"tong": du.total, "trong": du.free}, "ram": ram(), "thu_vien": D.thu_vien(),
         "ma_nguon": D.ma_nguon(ctx),
         "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "BYTETRACK_BACKEND", "YOLO_CONFIG_DIR", "TEMP")}}
    import torch
    cu = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "co_cuda": bool(torch.cuda.is_available())}
    if cu["co_cuda"]:
        cu["thiet_bi"] = [{"ten": torch.cuda.get_device_name(i), "bo_nho_byte": int(torch.cuda.get_device_properties(i).total_memory)}
                          for i in range(torch.cuda.device_count())]
        cu["cudnn"] = torch.backends.cudnn.version()
    d["cuda"] = cu
    if not ctx.fixture and not cu["co_cuda"]:
        raise H.LoiHopDong("tien-kiem A: khong co GPU CUDA cho cache --device 0")
    return d


# ---------------------------------------------------------------------------
# Con: cac buoc
# ---------------------------------------------------------------------------
def con_tien_kiem(ctx, A, vung, nk, g) -> int:
    dv, xk = "tien-kiem", A / "tien_kiem_a" / "xong.json"
    if xk.exists() or H.la_reparse(xk):
        nk.bat_dau(dv, "kiem_marker")
        can_xong(xk, file_tien_kiem(A), {"buoc": "tien-kiem", "luot": LUOT}, "tien-kiem A")
        nk.bo_qua(dv)
        print("tien-kiem: da xong (bo qua, khong ghi lai)")
        return 0
    nk.bat_dau(dv)
    sao_luu_do(nk, A, vung, [A / "tien_kiem", A / "tien_kiem_a"], dv, g["stt"])
    cv = cfg_va(ctx)
    chup_a(cv["plan"], A / "tien_kiem_a" / "chup", vung)
    ss = doi_chieu_baseline(cv["chup_dau"], A / "tien_kiem_a" / "chup")
    H.ghi_json(A / "tien_kiem_a" / "so_sanh.json", ss, vung)
    print(json.dumps({"dat": ss["dat"], "loi": ss["loi"][:10], "ngoai_le": [m["muc"] for m in ss["ngoai_le"]],
                      "ngoai_le_duyet": [m["muc"] for m in ss["ngoai_le_duyet"]],
                      "loi_tho": ss["ket_qua_tho_khong_ngoai_le_duyet"]["loi"][:10]}, ensure_ascii=False))
    if not ss["dat"]:
        raise H.LoiHopDong(f"tien-kiem A: doi chieu baseline lan_01 khong dat: {ss['loi'][:6]}")
    D.buoc_tien_kiem(ctx, A, vung)
    H.ghi_json(A / "tien_kiem_a" / "moi_truong.json", moi_truong(ctx, A), vung)
    ghi_marker_bt(nk, xk, file_tien_kiem(A), {"buoc": "tien-kiem", "luot": LUOT})
    nk.xong(dv)
    return 0


def con_cache(ctx, A, vung, nk, g) -> int:
    import det_cache
    det = g["detector"]
    nk.bat_dau(f"cache:{det}:chuan_bi", "chuan_bi")
    nguon, sha_tk = can_tien_kiem(ctx, A)                 # PLAN + nguon TRUOC moi ghi / nap detector
    nk.xong(f"cache:{det}:chuan_bi")
    cache_dir = A / "cache" / det
    cho = []
    for v in ctx.videos:                                   # quet marker TRUOC moi ghi; hong -> 2
        mk = cache_dir / f"{v}.xong.json"
        if mk.exists() or H.la_reparse(mk):
            nk.bat_dau(f"cache:{det}:{v}", "kiem_marker")
            D.xong_cache(cache_dir, det, v)
            nk.bo_qua(f"cache:{det}:{v}")
        else:
            cho.append(v)
    if not cho:
        print(f"cache {det}: da du {len(ctx.videos)} video")
        return 0
    nk.bat_dau(f"cache:{det}:nap", "chuan_bi")
    H.tao_thu_muc(cache_dir, vung)
    spec = D.spec_cho(ctx, det, g["device"]).fill_env()
    H.kiem_duong_ghi(cache_dir / "meta.json", vung)
    det_cache.write_meta(cache_dir, spec, mode="append" if (cache_dir / "meta.json").exists() else "create")
    model = D.nap_model(ctx, det)
    nk.xong(f"cache:{det}:nap")
    for v in cho:
        nk.kiem_doan(f"cache {det}")
        dv = f"cache:{det}:{v}"
        nk.bat_dau(dv)
        sao_luu_do(nk, A, vung, [cache_dir / n for n in sorted(os.listdir(cache_dir)) if n.startswith(v + ".")], dv,
                   g["stt"])
        r = D.tao_cache_video(ctx, det, model, spec, cache_dir, v, vung, nguon, sha_tk)
        nk.xong(dv, n_boxes=r["n_boxes"], giay_suy_luan=round(r["giay"], 3))
    print(f"cache {det}: xong ({nk.lam} video moi)")
    return 0


def con_kiem_cache(ctx, A, vung, nk, g) -> int:
    nk.bat_dau("kiem-cache:chuan_bi", "chuan_bi")
    nguon, sha_tk = can_tien_kiem(ctx, A)
    out = A / "kiem_cache"
    if (out / "xong.json").exists() or H.la_reparse(out / "xong.json"):
        can_kiem_cache(A)
        nk.xong("kiem-cache:chuan_bi")
        print("kiem-cache: da xong")
        return 0
    nk.xong("kiem-cache:chuan_bi")
    for det in ctx.detectors():
        cd = A / "cache" / det
        for v in ctx.videos:
            p = out / "don_vi" / det / f"{v}.json"
            if p.is_file():
                continue
            nk.kiem_doan("kiem-cache")
            dv = f"kiem-cache:{det}:{v}"
            nk.bat_dau(dv)
            loi = D.kiem_cache_video(ctx, det, cd, v)
            mk = H.doc_json(cd / f"{v}.xong.json") if not loi else {}
            if not loi and (mk.get("tien_kiem_sha256"), mk.get("gt_sha256"), mk.get("seqinfo_sha256")) != \
                    (sha_tk, nguon["gt"][v], nguon["seqinfo"][v]):
                loi.append(f"{v}: marker cache khong rang buoc tien kiem/nguon hien tai")
            f = {"npz": cd / f"{v}.npz", "json": cd / f"{v}.json", "meta": cd / "meta.json", "xong": cd / f"{v}.xong.json"}
            rec = {"detector": det, "video": v, "dat": not loi, "loi": loi,
                   "sha256": {k: H.sha256_file(x) if x.is_file() else None for k, x in f.items()},
                   "byte": {k: x.stat().st_size if x.is_file() else None for k, x in f.items()},
                   "n_frames": mk.get("n_frames"), "n_boxes": mk.get("n_boxes"), "giay_suy_luan": mk.get("giay_suy_luan")}
            ghi_bao_toan(nk, p, rec)
            nk.xong(dv, dat=not loi)
            if loi:
                raise H.LoiHopDong(f"kiem-cache {det}/{v}: {loi[:3]}")
    dv = "kiem-cache:tong_hop"
    nk.bat_dau(dv)
    recs = [H.doc_json(out / "don_vi" / d / f"{v}.json") for d in ctx.detectors() for v in ctx.videos]
    sai = [f"{r['detector']}/{r['video']}" for r in recs if not r["dat"]]
    tong = {d: {"so_video": sum(1 for r in recs if r["detector"] == d),
                "so_frame": sum(r["n_frames"] or 0 for r in recs if r["detector"] == d),
                "byte_npz_json": sum((r["byte"]["npz"] or 0) + (r["byte"]["json"] or 0) for r in recs if r["detector"] == d),
                "giay_suy_luan": round(sum(r["giay_suy_luan"] or 0 for r in recs if r["detector"] == d), 3)}
            for d in ctx.detectors()}
    kq = {"dat": not sai, "loi": sai, "so_don_vi": len(recs), "ky_vong": len(ctx.detectors()) * len(ctx.videos),
          "theo_detector": tong}
    ghi_bao_toan(nk, out / "ket_qua.json", kq)
    ghi_marker_bt(nk, out / "xong.json", [out / "ket_qua.json"], {"buoc": "kiem-cache"})
    nk.xong(dv)
    if sai:
        raise H.LoiHopDong(f"kiem-cache khong dat: {sai[:5]}")
    print(json.dumps(kq, ensure_ascii=False))
    return 0


def file_don_vi_do(run_dir: Path, v: str) -> list[Path]:
    return kem_dang_ghi(D.file_don_vi(run_dir, v)) + [run_dir / "xong" / f"{v}.xong.json.dang_ghi"]


def tong_hop_smoke(ctx, A, vids, runs) -> dict:
    root, kc = A / "smoke", A / "kiem_cache" / "don_vi"
    loi, rows, cham = [], [], {}
    sl = {v: D.doc_seqinfo(ctx.gt_root / ctx.split / v / "seqinfo.ini") for v in vids}
    for r in runs:
        for v in vids:
            m = H.doc_json(root / "tracking" / ctx.split / r["run"] / "xong" / f"{v}.xong.json")
            rec = H.doc_json(kc / r["detector"] / f"{v}.json")
            if m.get("npz_sha256") != rec["sha256"]["npz"] or H.sha256_file(A / "cache" / r["detector"] / f"{v}.npz") != \
                    rec["sha256"]["npz"]:
                loi.append(f"{r['run']}/{v}: npz khac kiem-cache (cache doi qua smoke)")
            if m.get("n_frames") != sl[v] or rec.get("n_frames") != sl[v]:
                loi.append(f"{r['run']}/{v}: n_frames {m.get('n_frames')} != seqLength {sl[v]}")
            if m.get("seed_truyen") != r["seed"] or (r["seed"] is not None and m.get("seed_thuc_dung") != r["seed"]):
                loi.append(f"{r['run']}/{v}: seed {m.get('seed_truyen')}/{m.get('seed_thuc_dung')} != {r['seed']}")
            if H.chuan_json(m.get("tham_so_bo_loc")) != H.chuan_json(ctx.gt["tracker"]["tham_so_bo_loc_du_kien"][r["bo_loc"]]):
                loi.append(f"{r['run']}/{v}: tham so bo loc khac giao thuc")
            rows.append({"run": r["run"], "video": v, "bo_loc": r["bo_loc"], "giay": m.get("giay"),
                         "n_frames": m.get("n_frames"), "n_steps": m.get("n_steps")})
        c = H.doc_json(root / "cham" / r["run"] / "cham.xong.json")
        if c.get("so_video") != len(vids) or c.get("evaluator") != ctx.gt["danh_gia"]["evaluator"]:
            loi.append(f"{r['run']}: cham so_video/evaluator lech")
        cham[r["run"]] = c.get("combined")
    theo_bl = {}
    for x in rows:
        t = theo_bl.setdefault(x["bo_loc"], {"giay": 0.0, "frame": 0})
        t["giay"] += x["giay"] or 0.0
        t["frame"] += x["n_frames"] or 0
    for t in theo_bl.values():
        t["ms_moi_frame"] = round(1000 * t["giay"] / t["frame"], 4) if t["frame"] else None
    ms = {r: H.sha256_file(H.ROOT / r) for r in marker_smoke(ctx, A)}
    return {"dat": not loi, "loi": loi, "videos": vids, "so_tracking": len(rows), "so_cham": len(cham), "marker_sha256": ms,
            "ky_vong": {"tracking": len(runs) * len(vids), "cham": len(runs)}, "tracking": rows,
            "thoi_gian_tracking_theo_bo_loc": theo_bl, "seqLength": sl, "cham_combined": cham,
            "ghi_chu": "diem smoke KHONG dung de chon cau hinh/tuning; thoi gian cham/run lay tu tien_do.jsonl"}


def con_smoke(ctx, A, vung, nk, g) -> int:
    import run_tracking as rt
    nk.bat_dau("smoke:chuan_bi", "chuan_bi")
    can_tien_kiem(ctx, A)
    can_kiem_cache(A)
    root, vids, runs = A / "smoke", list(ctx.gt["smoke_cong_A"]["videos"]), ctx.runs()
    if (root / "xong.json").exists() or H.la_reparse(root / "xong.json"):
        can_smoke(ctx, A)
        nk.xong("smoke:chuan_bi")
        print("smoke: da xong")
        return 0
    nk.xong("smoke:chuan_bi")
    trk, cho_t, cho_c = root / "tracking", [], []
    for r in runs:                                          # quet marker TRUOC moi ghi; hong -> 2
        for v in vids:
            rd = trk / ctx.split / r["run"]
            if (rd / "xong" / f"{v}.xong.json").exists() or H.la_reparse(rd / "xong" / f"{v}.xong.json"):
                nk.bat_dau(f"smoke:tracking:{r['run']}:{v}", "kiem_marker")
                D.xong_don_vi(rd, r["run"], v)
                nk.bo_qua(f"smoke:tracking:{r['run']}:{v}")
            else:
                cho_t.append((r, v))
        mc = root / "cham" / r["run"] / "cham.xong.json"
        if mc.exists() or H.la_reparse(mc):
            nk.bat_dau(f"smoke:cham:{r['run']}", "kiem_marker")
            D.xong_cham(root / "cham" / r["run"], r["run"])
            nk.bo_qua(f"smoke:cham:{r['run']}")
        else:
            cho_c.append(r)
    reg = rt.tracker_registry() if cho_t else None
    for r, v in cho_t:
        nk.kiem_doan("smoke tracking")
        dv, rd = f"smoke:tracking:{r['run']}:{v}", trk / ctx.split / r["run"]
        nk.bat_dau(dv)
        sao_luu_do(nk, A, vung, file_don_vi_do(rd, v), dv, g["stt"])
        m = D.chay_don_vi(ctx, r, v, A / "cache" / r["detector"], rd, reg, vung)
        nk.xong(dv, giay_tracking=m["giay"], n_frames=m["n_frames"])
    for r in cho_c:
        nk.kiem_doan("smoke cham")
        dv = f"smoke:cham:{r['run']}"
        nk.bat_dau(dv)
        sao_luu_do(nk, A, vung, [root / "cham" / r["run"], root / "cham" / "_trackeval" / r["run"]], dv, g["stt"])
        D.cham_run(ctx, r["run"], vids, trk, root / "cham", vung)
        nk.xong(dv)
    dv = "smoke:tong_hop"
    nk.bat_dau(dv)
    kq = tong_hop_smoke(ctx, A, vids, runs)
    sao_luu_do(nk, A, vung, kem_dang_ghi([root / "ket_qua_smoke.json"]), dv, g["stt"])
    H.ghi_json(root / "ket_qua_smoke.json", kq, vung)
    if not kq["dat"]:
        raise H.LoiHopDong(f"smoke: {kq['loi'][:5]}")
    ghi_marker_bt(nk, root / "xong.json", [root / "ket_qua_smoke.json"], {"buoc": "smoke"})
    nk.xong(dv)
    print(f"smoke: {kq['so_tracking']} don vi tracking, {kq['so_cham']} run cham")
    return 0


def doi_chieu_kiem_cache(ctx, A: Path, cache: dict) -> None:
    """Cache bam luc khoa/kiem khoa phai TRUNG byte cache da kiem o kiem-cache (khong doi qua smoke)."""
    lech = []
    for det in ctx.detectors():
        cd = A / "cache" / det
        for v in ctx.videos:
            rec = H.doc_json(A / "kiem_cache" / "don_vi" / det / f"{v}.json")
            for k, f in (("npz", f"{v}.npz"), ("json", f"{v}.json"), ("meta", "meta.json"), ("xong", f"{v}.xong.json")):
                if cache.get(H.rel(cd / f)) != rec["sha256"][k]:
                    lech.append(f"{det}/{f}")
    if lech:
        raise H.LoiHopDong(f"cache khac ket qua kiem-cache: {sorted(set(lech))[:5]}")


def con_khoa(ctx, A, vung, nk, g) -> int:
    nk.bat_dau("khoa:chuan_bi", "chuan_bi")
    can_tien_kiem(ctx, A)
    can_kiem_cache(A)
    can_smoke(ctx, A)
    xk = A / "khoa" / "khoa.xong.json"
    if xk.exists() or H.la_reparse(xk):
        can_khoa(A)
        nk.xong("khoa:chuan_bi")
        print("khoa: da co khoa hop le -- khong ghi de")
        return 0
    nk.xong("khoa:chuan_bi")
    tong = bam_theo_doan(nk, "khoa", A / "khoa" / "bam", lambda: liet_ke_rel(A / "cache"), vung, cfg_va(ctx)["kt"])
    dv = "khoa:ghi"
    nk.bat_dau(dv)
    if sorted(liet_ke_rel(A / "cache")) != sorted(tong) or any(h is None for h in tong.values()):
        raise H.LoiHopDong("khoa: danh muc cache doi so voi luc bam")
    doi_chieu_kiem_cache(ctx, A, tong)
    core = D.anh_khoa(ctx, A, tong)
    mt = H.doc_json(A / "tien_kiem_a" / "moi_truong.json")
    if H.chuan_json(core["ma_nguon"]) != H.chuan_json(mt["ma_nguon"]):
        raise H.LoiHopDong("khoa: ma nguon (dong import tinh) khac snapshot tien-kiem")
    khoa = {**core, "phu_luc_van_hanh": phu_luc(ctx, A), "luot": LUOT}
    sao_luu_do(nk, A, vung, kem_dang_ghi([A / "khoa" / "khoa.json"]), dv, g["stt"])
    H.ghi_json(A / "khoa" / "khoa.json", khoa, vung)
    ghi_marker_bt(nk, xk, [A / "khoa" / "khoa.json"], {"buoc": "khoa", "luot": LUOT})
    nk.xong(dv)
    print(f"khoa: {len(tong)} file cache, {len(core['ma_nguon'])} file ma src, {len(core['ma_thu_vien'])} file thu vien")
    return 0


def vong_kiem(goc: Path, vung) -> Path:
    """Vong kiem dang do (chua xong.json) -> tiep tuc; khong co -> vong moi."""
    ds = sorted(n for n in os.listdir(goc) if n.startswith("vong_")) if goc.is_dir() else []
    if ds and not (goc / ds[-1] / "xong.json").exists():
        return goc / ds[-1]
    return H.tao_thu_muc(goc / f"vong_{len(ds) + 1:02d}", vung)


def con_kiem_khoa(ctx, A, vung, nk, g) -> int:
    nk.bat_dau("kiem-khoa:chuan_bi", "chuan_bi")
    kp = can_khoa(A)
    kiem_plan(ctx)
    sau_goi = (A / "dong_goi" / "danh_muc.json").exists()
    R = vong_kiem(A / "van_hanh" / "kiem_khoa_sau_goi" if sau_goi else A / "kiem_khoa", vung)
    nk.goc_do = R / "phuc_hoi"          # sao luu cua kiem nam trong vong; sau goi -> ngoai payload da chot (A0-F2b)
    nk.xong("kiem-khoa:chuan_bi", vong=H.rel(R))
    tong = bam_theo_doan(nk, "kiem-khoa", R / "bam", lambda: liet_ke_rel(A / "cache"), vung, cfg_va(ctx)["kt"])
    dv = "kiem-khoa:so_sanh"
    nk.bat_dau(dv)
    loi = [] if sorted(liet_ke_rel(A / "cache")) == sorted(tong) else ["danh muc cache doi so voi luc bam"]
    khoa = H.doc_json(kp)
    loi += D.so_khoa(khoa, D.anh_khoa(ctx, A, tong))
    loi += so_phu_luc(khoa.get("phu_luc_van_hanh", {}), phu_luc(ctx, A))
    kq = {"khoa": H.rel(kp), "khoa_sha256": H.sha256_file(kp), "dat": not loi, "loi": loi, "vong": H.rel(R),
          "sau_dong_goi": sau_goi, "so_file_cache": len(tong)}
    ghi_bao_toan(nk, R / "ket_qua.json", kq)
    ghi_marker_bt(nk, R / "xong.json", [R / "ket_qua.json"], {"buoc": "kiem-khoa"})
    nk.xong(dv, dat=not loi)
    print(json.dumps({"dat": not loi, "loi": loi[:8]}, ensure_ascii=False))
    if loi:
        raise H.LoiHopDong(f"kiem-khoa lech: {loi[:6]}")
    return 0


def payload(A: Path, stt_dau: int) -> list[str]:
    out = []
    for d in GOC_PAYLOAD:
        out += liet_ke_rel(A / d)
    out.append(H.rel(A / "van_hanh" / "phien.json"))
    out += liet_ke_rel(A / "van_hanh" / "artifact_do")
    lc = A / "van_hanh" / "lan_chay"
    for n in sorted(os.listdir(lc)):
        if n[:4].isdigit() and int(n[:4]) < stt_dau:
            out += liet_ke_rel(lc / n)
    return sorted(out)


def tong_hop_crash(A: Path) -> dict:
    ds = doc_lan_chay(A)
    return {"dem": dem_crash(ds), "nguong": NGUONG,
            "lan_chay": [{"stt": x["stt"], "ten": x["ten"], **({k: x["ket_thuc"].get(k) for k in
                          ("buoc", "exit", "exit_hex", "phan_loai", "don_vi", "quyet_dinh", "giay")}
                         if x["ket_thuc"] else {"dang_mo": True})} for x in ds],
            "log_cha": H.bang_hash_thu_muc(A / "van_hanh" / "cha") if (A / "van_hanh" / "cha").is_dir() else {}}


def con_dong_goi_a(ctx, A, vung, nk, g) -> int:
    nk.bat_dau("dong-goi-a:chuan_bi", "chuan_bi")
    kp = can_khoa(A)
    can_tien_kiem(ctx, A)
    can_kiem_cache(A)
    can_smoke(ctx, A)
    vk = sorted(n for n in os.listdir(A / "kiem_khoa") if n.startswith("vong_")) if (A / "kiem_khoa").is_dir() else []
    R = A / "kiem_khoa" / vk[-1] if vk else None
    if R is None or not H.da_xong(R / "xong.json", [R / "ket_qua.json"], {"buoc": "kiem-khoa"}) or \
            not H.doc_json(R / "ket_qua.json")["dat"] or H.doc_json(R / "ket_qua.json")["khoa_sha256"] != H.sha256_file(kp):
        raise H.LoiHopDong("dong-goi-a: chua co kiem-khoa DAT cho khoa hien tai")
    gx = A / "goi_a.xong.json"
    goi_tep = [A / "goi_a.json", A / "SHA256SUMS"]
    if gx.exists() or H.la_reparse(gx):
        can_xong(gx, goi_tep, {"buoc": "dong-goi-a", "luot": LUOT}, "goi A")
        nk.xong("dong-goi-a:chuan_bi")
        print("dong-goi-a: goi A hop le da co -- khong ghi de")
        return 0
    nk.xong("dong-goi-a:chuan_bi")
    wd = A / "dong_goi"
    goi_do = kem_dang_ghi([A / "goi", *goi_tep, gx])          # gx chua ton tai o day; chi con '.dang_ghi' (A0-F6)
    if any(p.exists() or H.la_reparse(p) for p in goi_do):
        # Goi do dang (chua marker): sao luu goi do + vung bam cu TRUOC khi lap danh muc moi -> ban sao luu NAM TRONG
        # payload duoc bam (A0-F2); danh muc cu khong con dung vi artifact_do vua doi.
        dv = "dong-goi-a:goi_do"
        nk.bat_dau(dv)
        sao_luu_do(nk, A, vung, goi_do + [wd], dv, g["stt"])
        nk.xong(dv)
    # Tu day danh muc payload co the da co dinh: moi sao luu phuc hoi (danh muc / doan bam / marker do dang) vao
    # dong_goi/phuc_hoi/ -- NGOAI payload() nen khong thanh file thua; goi_a.json bam rieng tung file o day (A0-F2a).
    nk.goc_do = wd / "phuc_hoi"
    stt_dau = H.doc_json(wd / "danh_muc.json")["stt_dau"] if (wd / "danh_muc.json").is_file() else g["stt"]
    tong = bam_theo_doan(nk, "dong-goi-a", wd, lambda: payload(A, stt_dau), vung, cfg_va(ctx)["kt"], {"stt_dau": stt_dau})
    dv = "dong-goi-a:ghi"
    nk.bat_dau(dv)
    now = payload(A, stt_dau)
    if now != sorted(tong) or any(h is None for h in tong.values()):
        raise H.LoiHopDong(f"dong-goi-a: payload doi so voi danh muc (thua {sorted(set(now) - set(tong))[:3]}, "
                           f"thieu {sorted(set(tong) - set(now))[:3]})")
    pc = H.rel(A / "cache") + "/"
    cache = {r: h for r, h in tong.items() if r.startswith(pc)}
    khoa = H.doc_json(kp)
    loi = D.so_khoa(khoa, D.anh_khoa(ctx, A, cache)) + so_phu_luc(khoa.get("phu_luc_van_hanh", {}), phu_luc(ctx, A))
    if loi:
        raise H.LoiHopDong(f"goi A: khoa lech {loi[:5]}")
    if any(p.exists() or H.la_reparse(p) for p in goi_do):
        raise H.LoiHopDong("dong-goi-a: goi do xuat hien sau khi bam payload -- khong ghi")
    nkc = A / "goi" / "nhat_ky_crash_chot.json"
    tep = dict(tong)
    phuc_hoi = liet_ke_rel(wd / "phuc_hoi")                  # bang chung phuc hoi sau danh muc: nho, bam truc tiep
    tep.update({r: H.sha256_file(H.ROOT / r) for r in phuc_hoi})
    tep[H.rel(nkc)] = H.ghi_json(nkc, {**tong_hop_crash(A), "chot_tai_stt": g["stt"]}, vung)
    goi = {"luot": LUOT, "phien_A": H.rel(A), "khoa": H.rel(kp), "khoa_sha256": H.sha256_file(kp),
           "giao_thuc_sha256": H.sha256_file(ctx.gt_path), "hop_dong_lenh_sha256": H.sha256_file(HOP_DONG),
           "kiem_khoa_vong": H.rel(R), "kiem_khoa_ket_qua_sha256": H.sha256_file(R / "ket_qua.json"),
           "stt_dong_goi_dau": stt_dau, "loai_tru": LOAI_TRU_GOI, "phuc_hoi_dong_goi": phuc_hoi, "so_file": len(tep),
           "file": dict(sorted(tep.items()))}
    H.ghi_json(A / "goi_a.json", goi, vung)
    tiem("giua_goi", dv)
    H.ghi_bytes(A / "SHA256SUMS", "".join(f"{h}  {r}\n" for r, h in sorted(tep.items())).encode("utf-8"), vung)
    ghi_marker_bt(nk, gx, goi_tep, {"buoc": "dong-goi-a", "luot": LUOT, "khoa_sha256": goi["khoa_sha256"]})
    nk.xong(dv, so_file=len(tep))
    print(f"goi A: {len(tep)} file; goi_a_sha256 {H.sha256_file(A / 'goi_a.json')}; khoa_sha256 {goi['khoa_sha256']}")
    return 0


def con_kiem_goi_a(ctx, A, vung, nk, g) -> int:
    nk.bat_dau("kiem-goi-a:chuan_bi", "chuan_bi")
    gx, gp = A / "goi_a.xong.json", A / "goi_a.json"
    if not (gx.exists() or H.la_reparse(gx)):
        raise H.LoiHopDong("goi A chua hoan tat (thieu goi_a.xong.json) -- khong kiem/khong hop thuc hoa goi thieu")
    can_xong(gx, [gp, A / "SHA256SUMS"], {"buoc": "dong-goi-a", "luot": LUOT}, "goi A")
    goi = H.doc_json(gp)
    R = vong_kiem(A / "van_hanh" / "kiem_goi_a", vung)
    nk.goc_do = R / "phuc_hoi"          # phuc hoi kiem goi KHONG ghi vao payload da chot (A0-F2b)
    nk.xong("kiem-goi-a:chuan_bi", vong=H.rel(R))
    tong = bam_theo_doan(nk, "kiem-goi-a", R / "bam", lambda: list(goi["file"]), vung, cfg_va(ctx)["kt"])
    dv = "kiem-goi-a:so_sanh"
    nk.bat_dau(dv)
    lech = sorted(r for r, h in goi["file"].items() if tong.get(r) != h)
    # File thua: TOAN BO pham vi payload (ca van_hanh/artifact_do, phien.json, lan chay truoc dong goi) + goi/ +
    # dong_goi/phuc_hoi/ (A0-F2, F2a)
    thua = sorted(r for r in set(payload(A, int(goi["stt_dong_goi_dau"]))) | set(liet_ke_rel(A / "goi")) |
                  set(liet_ke_rel(A / "dong_goi" / "phuc_hoi")) if r not in goi["file"])
    loi = [f"lech/thieu: {r}" for r in lech] + [f"thua: {r}" for r in thua]
    sums = "".join(f"{h}  {r}\n" for r, h in sorted(goi["file"].items())).encode("utf-8")
    if (A / "SHA256SUMS").read_bytes() != sums:
        loi.append("SHA256SUMS khac danh muc goi_a.json")
    kp = H.ROOT / goi["khoa"]
    if not kp.is_file() or H.sha256_file(kp) != goi["khoa_sha256"]:
        loi.append("khoa.json khac khoa_sha256 trong goi")
    kq = {"dat": not loi, "loi": loi, "goi_a": H.rel(gp), "goi_a_sha256": H.sha256_file(gp),
          "khoa_sha256": goi["khoa_sha256"], "so_file": len(goi["file"]), "so_lech": len(lech), "so_thua": len(thua),
          "vong": H.rel(R)}
    ghi_bao_toan(nk, R / "ket_qua.json", kq)
    ghi_marker_bt(nk, R / "xong.json", [R / "ket_qua.json"], {"buoc": "kiem-goi-a"})
    nk.xong(dv, dat=not loi)
    print(json.dumps({k: kq[k] for k in ("dat", "goi_a_sha256", "khoa_sha256", "so_file")} | {"loi": loi[:8]},
                     ensure_ascii=False))
    if loi:
        raise H.LoiHopDong(f"kiem-goi-a: {loi[:6]}")
    return 0


BUOC_CON = {"tien-kiem": con_tien_kiem, "cache": con_cache, "kiem-cache": con_kiem_cache, "smoke": con_smoke,
            "khoa": con_khoa, "kiem-khoa": con_kiem_khoa, "dong-goi-a": con_dong_goi_a, "kiem-goi-a": con_kiem_goi_a}


def kiem_lan_con(d: Path) -> tuple[dict, Path, "D.Ctx"]:
    """--con: xac thuc DAY DU lan chay TRUOC moi ghi (A0-F5): duong <A>/van_hanh/lan_chay/<stt>_<buoc> khong qua
    reparse; A la phien A that co dinh hoac phien fixture; goi.json + phien.json cung luot/phien/giao thuc; giao thuc
    that <-> phien that; lan chay chua dung (khong co log con / ket thuc)."""
    d = Path(os.path.abspath(d))
    if d.parent.name != "lan_chay" or d.parent.parent.name != "van_hanh":
        raise H.LoiHopDong(f"--con {d}: khong phai <phien>/van_hanh/lan_chay/<lan>")
    A = d.parent.parent.parent
    if not (A == PHIEN_A_THAT or (H._trong(A, VUNG_FIXTURE) and A != VUNG_FIXTURE)):
        raise H.LoiHopDong(f"--con {d}: phien ngoai phien A that / vung fixture")
    cur = Path(d.anchor)
    for part in d.parts[1:]:
        cur = cur / part
        if H.la_reparse(cur):
            raise H.LoiHopDong(f"--con {d}: di qua reparse {cur}")
    if not d.is_dir() or not (d / "goi.json").is_file() or Path(os.path.realpath(d)) != Path(os.path.realpath(A)) / d.relative_to(A):
        raise H.LoiHopDong(f"--con {d}: thieu goi.json / duong resolve khac")
    da_dung = [f for f in ("faulthandler.log", "tien_do.jsonl", "ket_qua_con.json", "ket_thuc.json") if (d / f).exists()]
    if da_dung:
        raise H.LoiHopDong(f"--con {d}: lan chay da dung/da dong ({da_dung}) -- khong chay lai")
    g, pj = H.doc_json(d / "goi.json"), A / "van_hanh" / "phien.json"
    if g.get("luot") != LUOT or g.get("phien") != H.rel(A) or not d.name.startswith(f"{int(g.get('stt', -1)):04d}_") \
            or not pj.is_file():
        raise H.LoiHopDong(f"--con {d}: goi.json khong dung luot/phien/stt hoac phien chua mo")
    ctx = D.Ctx(Path(g["giao_thuc"]))
    ph = H.doc_json(pj)
    if ph.get("luot") != LUOT or ph.get("phien") != H.rel(A) or ph.get("giao_thuc") != H.rel(ctx.gt_path):
        raise H.LoiHopDong(f"--con {d}: phien.json khong dung luot/phien/giao thuc cua goi.json")
    if ctx.fixture == (A == PHIEN_A_THAT):
        raise H.LoiHopDong(f"--con {d}: giao thuc fixture/that khong khop loai phien")
    if g.get("tiem_loi") and not ctx.fixture:
        raise H.LoiHopDong("tiem loi chi cho giao thuc fixture duoi tests/cong_a_20261024/")
    return g, A, ctx


def con_main(d: Path) -> int:
    global TIEM
    import faulthandler
    try:
        g, A, ctx = kiem_lan_con(d)                   # KHONG ghi gi neu sai
    except H.LoiHopDong as e:
        print(f"[LOI HOP DONG] {e}", file=sys.stderr)
        return H.EXIT_HOP_DONG
    d = Path(os.path.abspath(d))
    vung = [A]
    TIEM = list(g.get("tiem_loi") or [])
    if not any(t.get("diem") == "bo_faulthandler" for t in TIEM):     # tiem "thieu log" CHI fixture (A0-F3)
        fh = open(H.kiem_duong_ghi(d / "faulthandler.log", vung), "x", encoding="utf-8")
        faulthandler.enable(file=fh, all_threads=True)
    H.chuan_bi_moi_truong(A, vung)
    tiem("khoi_dong")
    nk = NhatKy(d / "tien_do.jsonl", vung, g["lan_thu_truoc"], float(g["ngan_sach"]), g.get("gioi_han_don_vi"), A,
                int(g["stt"]))
    if ctx.fixture:                                   # detector MO PHONG chi voi giao thuc fixture (D.nap_model kiem lai)
        from test_chuan_bi_bdd import ModelGia

        def nap(det, cfg):
            print(DAU_NAP + det, flush=True)
            return ModelGia(cfg["loai_checkpoint"])
        D.NAP_MODEL = nap
    goc_marker = H.ghi_marker

    def ghi_marker(*a, **k):                         # diem tiem "sau_ghi": artifact da ghi, marker chua ghi
        tiem("sau_ghi", nk.hien_tai)
        return goc_marker(*a, **k)
    H.ghi_marker = ghi_marker
    ket = {"stt": g["stt"], "buoc": g["buoc"], "exit": None}
    try:
        ket["exit"] = int(BUOC_CON[g["buoc"]](ctx, A, vung, nk, g) or 0)
    except H.LoiHopDong as e:
        print(f"[LOI HOP DONG] {e}", file=sys.stderr)
        ket.update(exit=H.EXIT_HOP_DONG, loi=str(e), don_vi=nk.hien_tai)
    except H.ChuaXong as e:
        print(f"[CHUA XONG] {e}")
        ket.update(exit=H.EXIT_CHUA_XONG, ly_do=str(e))
    ket.update(so_don_vi_moi=nk.lam, giay=round(time.time() - T0, 3), bo_nho=bo_nho_tien_trinh())
    H.ghi_json(d / "ket_qua_con.json", ket, vung, ghi_de=True)
    nk.ghi(su_kien="ket_thuc_con", exit=ket["exit"])
    return ket["exit"]


# ---------------------------------------------------------------------------
# Tien trinh CHA
# ---------------------------------------------------------------------------
def doc_lan_chay(A: Path) -> list[dict]:
    lc = A / "van_hanh" / "lan_chay"
    out = []
    for n in (sorted(os.listdir(lc)) if lc.is_dir() else []):
        d = lc / n
        if not d.is_dir() or not n[:4].isdigit():
            continue
        kt, gp = d / "ket_thuc.json", d / "goi.json"
        out.append({"stt": int(n[:4]), "ten": n, "dir": d, "ket_thuc": H.doc_json(kt) if kt.is_file() else None,
                    "goi": H.doc_json(gp) if gp.is_file() else None})
    return out


def doc_nhat_ky(p: Path) -> tuple[list[dict], bool]:
    if not p.is_file():
        return [], True
    ev, ok = [], True
    for ln in p.read_bytes().decode("utf-8", "replace").splitlines():
        try:
            ev.append(json.loads(ln))
        except ValueError:
            ok = False
    return ev, ok


def don_vi_dang_do(ev: list[dict]) -> str | None:
    mo = None
    for e in ev:
        if e.get("su_kien") == "bat_dau":
            mo = e.get("don_vi")
        elif e.get("su_kien") in ("xong", "bo_qua") and e.get("don_vi") == mo:
            mo = None
        elif e.get("su_kien") == "ket_thuc_con":
            mo = None
    return mo


def dem_crash(ds: list[dict]) -> dict:
    tong, dv, bu = 0, {}, {}
    for x in ds:
        k = x["ket_thuc"]
        if not k or k.get("phan_loai") not in LOAI_CRASH:
            continue
        tong += 1
        if k.get("don_vi"):
            dv[k["don_vi"]] = dv.get(k["don_vi"], 0) + 1
        bu[k.get("buoc")] = bu.get(k.get("buoc"), 0) + 1
    return {"tong": tong, "don_vi": dv, "buoc": bu}


def dem_bat_dau(ds: list[dict]) -> dict:
    out = {}
    for x in ds:
        for e in doc_nhat_ky(x["dir"] / "tien_do.jsonl")[0]:
            if e.get("su_kien") == "bat_dau" and e.get("pha") != "kiem_marker":
                out[e["don_vi"]] = out.get(e["don_vi"], 0) + 1
    return out


def phan_loai(ma, het_gio: bool, d: Path) -> dict:
    ev, ok = doc_nhat_ky(d / "tien_do.jsonl")
    err_p, fh_p = d / "stderr.txt", d / "faulthandler.log"
    err = err_p.read_bytes().decode("utf-8", "replace") if err_p.is_file() else None
    fh = fh_p.read_bytes().decode("utf-8", "replace") if fh_p.is_file() else ""
    if het_gio:
        loai = "het_gio"
    elif ma in (0, 10, 2):
        loai = {0: "hoan_tat", 10: "chua_xong", 2: "loi_hop_dong"}[ma]
    elif ma == 1:
        loai = "loi_python" if err and "Traceback (most recent call last)" in err else "exit1_khong_traceback"
    else:
        loai = "crash_native"
    return {"phan_loai": loai, "don_vi": don_vi_dang_do(ev) if ok else None, "nhat_ky_doc_duoc": ok,
            "thieu_log": [f for f in ("stdout.txt", "stderr.txt", "faulthandler.log") if not (d / f).is_file()],
            "faulthandler_byte": len(fh), "faulthandler_bao_loi_native": any(s in fh for s in
                                                                               ("Windows fatal exception", "Fatal Python error",
                                                                                "Segmentation fault")),
            "ly_do_hop_dong": [ln for ln in (err or "").splitlines() if ln.startswith("[LOI HOP DONG]")]}


def quyet_dinh(kt: dict, dem: dict, buoc: str) -> tuple[int, list[str]]:
    lo = kt["phan_loai"]
    if not (kt.get("job") or {}).get("dat", True):
        return EXIT_DUNG, [f"Job Object kill-on-close cho con khong dat ({kt['job']}) -- khong bao dam con chet theo cha"]
    if kt.get("thieu_log"):                        # A0-F3: thieu log -> khong tin exit/phan loai, khong thu lai
        return EXIT_DUNG, [f"thieu log {kt['thieu_log']} (exit {kt.get('exit')}, {lo}) -- khong xac dinh duoc trang "
                           "thai an toan"]
    if lo in ("hoan_tat", "chua_xong", "loi_hop_dong"):
        return {"hoan_tat": 0, "chua_xong": 10, "loi_hop_dong": 2}[lo], []
    if lo != "crash_native":
        return EXIT_DUNG, [f"{lo}: khong tu thu lai, chua xac dinh trang thai an toan"]
    if not kt["don_vi"]:
        return EXIT_DUNG, ["crash native khong xac dinh duoc don vi dang do"]
    ly = []
    n = dem["don_vi"].get(kt["don_vi"], 0)
    if n >= NGUONG["don_vi"]:
        ly.append(f"don vi {kt['don_vi']} crash lan {n} (nguong {NGUONG['don_vi']})")
    if dem["tong"] >= NGUONG["phien"]:
        ly.append(f"tong crash phien {dem['tong']} (nguong {NGUONG['phien']})")
    if buoc in BUOC_KHOA and dem["buoc"].get(buoc, 0) >= NGUONG["buoc_khoa"]:
        ly.append(f"buoc {buoc} crash lan {dem['buoc'][buoc]} (nguong {NGUONG['buoc_khoa']})")
    return (EXIT_DUNG, ly) if ly else (EXIT_CRASH, [])


def _k32():
    import ctypes
    from ctypes import wintypes
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.OpenProcess.restype = wintypes.HANDLE
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    k.CreateJobObjectW.restype = wintypes.HANDLE
    k.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    k.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    k.GetCurrentProcess.restype = wintypes.HANDLE
    k.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    return k


def thoi_diem_tao(h) -> int | None:
    import ctypes
    from ctypes import wintypes
    ft = [wintypes.FILETIME() for _ in range(4)]
    if not _k32().GetProcessTimes(h, *[ctypes.byref(f) for f in ft]):
        return None
    return (ft[0].dwHighDateTime << 32) | ft[0].dwLowDateTime


def trang_thai_con(pid: int, tao: int | None) -> dict:
    """con_song True / False / None (khong xac dinh). Cung pid nhung thoi diem tao khac = tien trinh khac (pid tai dung)."""
    import ctypes
    from ctypes import wintypes
    k = _k32()
    h = k.OpenProcess(0x1000, False, int(pid))          # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        e = ctypes.get_last_error()
        return {"con_song": False if e == 87 else None, "mo_duoc": False, "loi_win": e}
    try:
        ma = wintypes.DWORD()
        ok = k.GetExitCodeProcess(h, ctypes.byref(ma))
        t = thoi_diem_tao(h)
        if t is None or not ok:
            return {"con_song": None, "mo_duoc": True}
        if tao is not None and t != tao:
            return {"con_song": False, "mo_duoc": True, "pid_tai_dung": True}
        return {"con_song": ma.value == 259, "mo_duoc": True, "exit": None if ma.value == 259 else ma.value}
    finally:
        k.CloseHandle(h)


def gan_job_cha() -> dict:
    """Gan CHINH tien trinh cha vao Job Object KILL_ON_JOB_CLOSE TRUOC khi tao con: con ke thua job ngay luc
    CreateProcess -> khong co khoang ho Popen -> Assign (A0-F4). Handle job (khong ke thua) giu trong _JOB suot doi
    cha; cha chet (ke ca crash native) -> kernel dong handle -> moi tien trinh con trong job bi giet."""
    global _JOB
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
    k = _k32()
    job = k.CreateJobObjectW(None, None)
    if not job:
        return {"dat": False, "buoc": "CreateJobObjectW", "loi_win": ctypes.get_last_error()}
    info = MoRong()
    info.Basic.LimitFlags = 0x2000                       # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not k.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        return {"dat": False, "buoc": "SetInformationJobObject", "loi_win": ctypes.get_last_error()}
    if not k.AssignProcessToJobObject(job, k.GetCurrentProcess()):
        return {"dat": False, "buoc": "AssignProcessToJobObject(cha)", "loi_win": ctypes.get_last_error()}
    _JOB = job
    return {"dat": True, "kill_on_job_close": True, "gan": "cha (con ke thua khi tao)"}


def trong_job(p) -> bool | None:
    import ctypes
    from ctypes import wintypes
    r = wintypes.BOOL()
    if _JOB is None or not _k32().IsProcessInJob(int(p._handle), _JOB, ctypes.byref(r)):
        return None
    return bool(r.value)


def tiep_tuc_con(p) -> dict:
    """Con tao voi CREATE_SUSPENDED (chua chay lenh nao) -> NtResumeProcess sau khi da xac nhan job + ghi con.json."""
    import ctypes
    from ctypes import wintypes
    nt = ctypes.WinDLL("ntdll")
    nt.NtResumeProcess.argtypes = [wintypes.HANDLE]
    nt.NtResumeProcess.restype = ctypes.c_long
    st = nt.NtResumeProcess(int(p._handle))
    return {"dat": st == 0, "ntstatus": hex(st & 0xFFFFFFFF)}


def snapshot(ctx) -> dict:
    return {"ma": {p.relative_to(H.SRC).as_posix(): H.sha256_file(p) for p in D.dong_import_tinh(H.SRC)},
            "hop_dong_lenh_sha256": H.sha256_file(HOP_DONG), "giao_thuc_sha256": H.sha256_file(ctx.gt_path)}


def dau_vao(ctx, A: Path) -> dict:
    out = {"plan": CB.tach_plan(cfg_va(ctx)["plan"].read_bytes())}
    for k, p in (("tien_kiem", A / "tien_kiem" / "tien_kiem.json"), ("khoa", A / "khoa" / "khoa.json"),
                 ("goi_a", A / "goi_a.json")):
        out[f"{k}_sha256"] = H.sha256_file(p) if p.is_file() else None
    return out


def dung(A: Path, vung, ly_do: list[str], stt) -> None:
    p = A / "van_hanh" / "dung.json"
    if not p.exists():
        H.ghi_json(p, {"ly_do": ly_do, "stt": stt, "luc": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "huong_dan": "DUNG HOI nguoi dung; khong tu xoa file nay, khong mo phien moi de reset bo dem"}, vung)


def mot_doan(ns, argv, ctx, A: Path, vung, snap: dict, tiem_loi: list) -> int:
    ds = doc_lan_chay(A)
    stt = (ds[-1]["stt"] if ds else 0) + 1
    ten = f"{stt:04d}_{ns.buoc}" + (f"_{ns.detector}" if ns.buoc == "cache" else "")
    d = H.tao_thu_muc(A / "van_hanh" / "lan_chay" / ten, vung)
    cmd = [H.PYTHON, str(Path(__file__).resolve()), "--con", str(d)]
    g = {"stt": stt, "luot": LUOT, "buoc": ns.buoc, "detector": ns.detector if ns.buoc == "cache" else None,
         "device": ns.device, "phien": H.rel(A), "giao_thuc": str(ctx.gt_path), "ngan_sach": ns.ngan_sach_giay,
         "gioi_han_don_vi": ns.gioi_han_don_vi, "argv_cha": [sys.executable, str(Path(__file__).resolve()), *argv],
         "argv_con": cmd, "pid_cha": os.getpid(), "bat_dau": time.strftime("%Y-%m-%dT%H:%M:%S"), "snapshot": snap,
         "dau_vao": dau_vao(ctx, A), "lan_thu_truoc": dem_bat_dau(ds), "dem_crash_truoc": dem_crash(ds),
         "tiem_loi": tiem_loi}
    job = gan_job_cha() if _JOB is None else {"dat": True, "kill_on_job_close": True, "gan": "cha (da gan)"}
    g["job_cha"] = job                                    # ghi TRUOC khi tao con (A0-F4)
    H.ghi_json(d / "goi.json", g, vung)
    env = dict(os.environ)
    env.update({"PYTHONFAULTHANDLER": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
    if ctx.fixture:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    t0, het_gio, ma, con = time.time(), False, None, None
    with open(H.kiem_duong_ghi(d / "stdout.txt", vung), "wb") as fo, open(H.kiem_duong_ghi(d / "stderr.txt", vung), "wb") as fe:
        if job["dat"]:                                    # khong co job -> KHONG tao con
            # CREATE_SUSPENDED: con chua chay lenh nao cho toi khi cha xac nhan job + ghi con.json (A0-F4)
            p = subprocess.Popen(cmd, cwd=str(H.ROOT), env=env, stdout=fo, stderr=fe, creationflags=0x4)
            tao = thoi_diem_tao(int(p._handle))
            for t in tiem_loi:                            # diem tiem CHA (fixture): crash ngay sau tao con, TRUOC con.json
                if t.get("diem") == "cha_sau_tao_con":
                    print(f"[TIEM CHA] con pid {p.pid} tao {tao}", file=sys.stderr, flush=True)
                    crash_native()
            tj = trong_job(p)
            job = {**job, "con_trong_job": tj} if tj else {**job, "dat": False, "buoc": "IsProcessInJob(con)",
                                                            "con_trong_job": tj}
            con = {"pid": p.pid, "thoi_diem_tao": tao, "tao_tam_dung": True, "job": job}
            H.ghi_json(d / "con.json", con, vung)               # lan sau doi chieu con da chet chua (A0-F4)
            if job["dat"]:
                tt = tiep_tuc_con(p)
                if not tt["dat"]:
                    job = {**job, "dat": False, "buoc": "NtResumeProcess(con)", "tiep_tuc": tt}
            if not job["dat"]:                            # khong bao dam con chet theo cha -> dung con, DUNG hoi
                p.kill()
                ma = p.wait()
            else:
                for t in tiem_loi:                        # diem tiem CHA (fixture): cha crash KHI con dang chay
                    if t.get("diem") == "cha_khi_con_chay":
                        time.sleep(float(t.get("giay", 2)))
                        crash_native()
                try:
                    ma = p.wait(timeout=ns.ngan_sach_giay + DU_PHONG_CUNG)
                except subprocess.TimeoutExpired:
                    p.kill()
                    ma, het_gio = p.wait(), True
    for t in tiem_loi:                                    # diem tiem cua CHA (fixture): con da ket thuc, chua ghi ket_thuc
        if t.get("diem") == "cha_sau_con":
            crash_native()
    pl = phan_loai(ma, het_gio, d) if con else {"phan_loai": "khong_tao_con", "don_vi": None, "thieu_log": []}
    kt = {"stt": stt, "buoc": ns.buoc, "exit": ma, "exit_hex": hex(ma & 0xFFFFFFFF) if ma is not None else None,
          "het_gio": het_gio, "giay": round(time.time() - t0, 3), "con": con, "job": job, **pl}
    dem = dem_crash(ds + [{"ket_thuc": kt}])
    ra, ly = quyet_dinh(kt, dem, ns.buoc)
    kt.update(dem_crash_sau=dem, quyet_dinh={"exit_cha": ra, "ly_do_dung": ly})
    H.ghi_json(d / "ket_thuc.json", kt, vung)
    print(f"[VAN HANH] lan {stt} {ns.buoc}: exit con {ma} ({kt['phan_loai']}, don vi {kt['don_vi']}) -> exit {ra}"
          + (f"; DUNG: {ly}" if ly else ""))
    if ra == EXIT_DUNG:
        dung(A, vung, ly, stt)
    return ra


def main_cha(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cong", required=True)
    ap.add_argument("--phien", required=True)
    ap.add_argument("--buoc", required=True, choices=BUOC)
    ap.add_argument("--so-doan", type=int, default=1)
    ap.add_argument("--ngan-sach-giay", type=float, default=NGAN_SACH_TOI_DA)
    ap.add_argument("--gioi-han-don-vi", type=int, default=None, help="toi da so don vi moi trong mot doan (>= 1)")
    ap.add_argument("--detector", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--giao-thuc", default=str(D.GIAO_THUC))
    ap.add_argument("--tiem-loi", default=None, help="CHI fixture: JSON danh sach diem tiem loi")
    ns = ap.parse_args(argv)
    # --- kiem doi so TRUOC moi ghi (khong tao thu muc nao neu sai) ---
    if ns.cong != "A":
        raise H.LoiHopDong("van_hanh_cong_a chi nhan --cong A")
    ctx = D.Ctx(Path(ns.giao_thuc))
    A = Path(os.path.abspath(ns.phien if Path(ns.phien).is_absolute() else H.ROOT / ns.phien))
    if ctx.fixture:
        if not H._trong(A, VUNG_FIXTURE) or A == VUNG_FIXTURE:
            raise H.LoiHopDong(f"phien fixture phai nam duoi {H.rel(VUNG_FIXTURE)}/")
    else:
        if A != PHIEN_A_THAT:
            raise H.LoiHopDong(f"cong A that chi chay tren phien co dinh {H.rel(PHIEN_A_THAT)} (khong mo phien moi)")
        if ns.tiem_loi:
            raise H.LoiHopDong("tiem loi chi cho giao thuc fixture -- khong bat tren du lieu that")
    vung = D.vung_phien(ctx, A, "A")
    if ns.so_doan != 1:                                   # A0-F9: moi loi goi DUNG mot doan (khong 2 x 210 s)
        raise H.LoiHopDong("moi loi goi dung MOT doan: --so-doan phai = 1")
    if not 0 < ns.ngan_sach_giay <= NGAN_SACH_TOI_DA:
        raise H.LoiHopDong(f"--ngan-sach-giay phai trong (0, {NGAN_SACH_TOI_DA:g}] giay")
    if ns.gioi_han_don_vi is not None and ns.gioi_han_don_vi < 1:
        raise H.LoiHopDong("--gioi-han-don-vi phai >= 1")
    if ns.buoc == "cache" and ns.detector not in ctx.detectors():
        raise H.LoiHopDong("--detector khong thuoc giao thuc")
    tiem_loi = json.loads(ns.tiem_loi) if ns.tiem_loi else []
    pj = A / "van_hanh" / "phien.json"
    if not pj.exists() and ns.buoc != "tien-kiem":
        raise H.LoiHopDong("phien A chua mo (chay --buoc tien-kiem truoc)")
    if pj.exists() or H.la_reparse(pj):                 # A0-F5: kiem luot/giao thuc TRUOC moi ghi
        ph = H.doc_json(pj)
        if ph.get("luot") != LUOT or ph.get("phien") != H.rel(A) or ph.get("giao_thuc") != H.rel(ctx.gt_path):
            raise H.LoiHopDong("phien A khong dung luot/giao thuc cua lop van hanh nay")
    elif A.exists() and any(A.iterdir()):
        raise H.LoiHopDong("thu muc phien da co noi dung nhung chua co van_hanh/phien.json -- khong nhan dien duoc luot")
    snap = snapshot(ctx)
    if not pj.exists():                                 # phien.json la file DAU TIEN cua phien moi
        H.ghi_json(pj, {"luot": LUOT, "phien": H.rel(A), "fixture": ctx.fixture, "giao_thuc": H.rel(ctx.gt_path),
                        "tao_luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "snapshot": snap, "nguong_crash": NGUONG}, vung)
    global PHIEN_CHAP_NHAN
    PHIEN_CHAP_NHAN = A
    H.chuan_bi_moi_truong(A, vung)
    import faulthandler
    fh_p = A / "van_hanh" / "cha" / f"{time.strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}.faulthandler.log"
    H.tao_thu_muc(fh_p.parent, vung)
    fh = open(H.kiem_duong_ghi(fh_p, vung), "w", encoding="utf-8")
    faulthandler.enable(file=fh, all_threads=True)
    ds = doc_lan_chay(A)
    for x in ds:                                       # cha truoc da gian doan (crash/bi giet) sau khi tao lan chay
        if x["ket_thuc"] is None:
            kq_con, cj = x["dir"] / "ket_qua_con.json", x["dir"] / "con.json"
            con = H.doc_json(cj) if cj.is_file() else None
            tt = trang_thai_con(con["pid"], con.get("thoi_diem_tao")) if con else {"con_song": None, "ly_do": "thieu con.json"}
            kt = {"stt": x["stt"], "buoc": (x["goi"] or {}).get("buoc"), "exit": None, "exit_hex": None,
                  "het_gio": None, "phan_loai": "cha_gian_doan", "don_vi": None,
                  "ket_qua_con": H.doc_json(kq_con) if kq_con.is_file() else None, "con": con, "con_trang_thai": tt,
                  "job_cha": (x["goi"] or {}).get("job_cha"),
                  "file_con": {f: (x["dir"] / f).is_file() for f in ("faulthandler.log", "tien_do.jsonl")},
                  "ghi_bo_sung_luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid_ghi_bo_sung": os.getpid(),
                  "ghi_chu": "tien trinh CHA khong ghi duoc ket_thuc (crash/bi giet); khong xac dinh don vi -> dung hoi"}
            ly = [f"cha gian doan o lan chay {x['stt']}"]
            ly.append("con da ket thuc (xac nhan qua pid + thoi diem tao)" if tt.get("con_song") is False else
                      f"KHONG xac nhan duoc con da ket thuc: {tt} -- khong chay tiep")
            if con is None and (kt["job_cha"] or {}).get("dat"):
                ly.append("thieu con.json: con (neu da tao) duoc tao TAM DUNG trong job kill-on-close cua cha -> chua chay "
                          f"lenh nao va bi giet cung cha; file cua con {kt['file_con']} (A0-F4)")
            dem = dem_crash([y if y is not x else {**y, "ket_thuc": kt} for y in ds])
            kt.update(dem_crash_sau=dem, quyet_dinh={"exit_cha": EXIT_DUNG, "ly_do_dung": ly})
            H.ghi_json(x["dir"] / "ket_thuc.json", kt, vung)
            dung(A, vung, kt["quyet_dinh"]["ly_do_dung"], x["stt"])
    ds = doc_lan_chay(A)
    H.ghi_json(A / "van_hanh" / "so_crash.json", tong_hop_crash(A), vung, ghi_de=True)
    if (A / "van_hanh" / "dung.json").exists():
        print(f"[DUNG HOI] phien A da dung: {H.doc_json(A / 'van_hanh' / 'dung.json')['ly_do']}", file=sys.stderr)
        return EXIT_DUNG
    ph = H.doc_json(pj)
    lech = [k for k in ("hop_dong_lenh_sha256", "giao_thuc_sha256") if ph["snapshot"][k] != snap[k]] + \
        [f"ma {f}" for f in sorted(set(ph["snapshot"]["ma"]) | set(snap["ma"])) if ph["snapshot"]["ma"].get(f) != snap["ma"].get(f)]
    if lech:
        raise H.LoiHopDong(f"ma/hop dong/giao thuc doi so voi luc mo phien A: {lech[:6]}")
    try:
        return mot_doan(ns, argv, ctx, A, vung, snap, tiem_loi)      # DUNG mot doan moi loi goi (A0-F9)
    finally:
        H.ghi_json(A / "van_hanh" / "so_crash.json", tong_hop_crash(A), vung, ghi_de=True)


def main(argv: list[str]) -> int:
    t0 = time.time()
    try:
        ma = main_cha(argv)
    except H.LoiHopDong as e:
        print(f"[LOI HOP DONG] {e}", file=sys.stderr)
        ma = H.EXIT_HOP_DONG
    A = PHIEN_CHAP_NHAN                    # A0-F5: chi ghi goi_cha.jsonl vao phien DA qua kiem luot/giao thuc
    if A is not None:
        ghi_dong(A / "van_hanh" / "goi_cha.jsonl", {"argv": argv, "exit": ma, "giay": round(time.time() - t0, 3),
                                                    "luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid()}, [A])
    return ma


if __name__ == "__main__":
    if sys.argv[1:2] == ["--con"]:
        sys.exit(con_main(Path(sys.argv[2])))
    sys.exit(main(sys.argv[1:]))
