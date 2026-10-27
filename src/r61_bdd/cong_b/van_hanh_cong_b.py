"""
van_hanh_cong_b.py -- Lop VAN HANH cong B cua R61 (plan chuan bi B0, 26/10/2026). CHI nhan --cong B.

Plan chuan bi KHONG mo B that: B that can file phe duyet rieng (--phe-duyet) do orchestrator cung cap sau khi nguoi dung
duyet ke hoach thuc thi; thieu / sai -> exit 2 TRUOC moi ghi. Fixture: hop dong + giao thuc + phe duyet fixture duoi
results/phase2/r61_bdd/tests/cong_b_20261026/.

    C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -B src/r61_bdd/cong_b/van_hanh_cong_b.py --cong B \
        --phien <phien_B> --buoc <buoc> --so-doan 1 --ngan-sach-giay 420 --phe-duyet <phe_duyet_b.json> \
        [--han-chot <epoch>] [--gioi-han-don-vi N] [--hop-dong <hd>] [--giao-thuc <gt>]
    ... van_hanh_cong_b.py --in-ma-b [--hop-dong <hd>]      # chi DOC: in manifest ma B (cho file phe duyet)

Buoc: tien-kiem -> mo-phien -> tracking -> cham -> lap-tracker -> cham-lai -> tong-hop -> kiem-khoa -> hau-kiem.
Don vi: tracking:<run>:<video>; cham:<run>; lap:<run>:<video>; cham-lai:<run>; tong-hop:khoi:<k>; hau-kiem:khoi:<k>.

CHA (thu vien chuan + ma r61 muc module thu vien chuan; khong bam du lieu lon): bat buoc chay bang interpreter da chi dinh
voi -B (sys.orig_argv); kiem doi so / duong (lexical + realpath + reparse) / phe duyet / phien / manifest ma B TRUOC moi
ghi; gan CHINH no vao Job Object KILL_ON_JOB_CLOSE truoc khi tao con; tao MOT con [python, -B, file, --con, <lan>] tam
dung, xac nhan con trong job, ghi con.json, roi moi resume. Deadline: han_cung = min(bat dau + ngan sach + 150 s,
--han-chot - du phong ket thuc); con nhan han_mem qua goi.json (khong cap lai ngan sach); con lai < toi thieu -> khong tao
lan chay, exit 10. Exit cha: 0 / 10 / 2 / 30 (crash native duoi nguong) / 40 (DUNG HOI, van_hanh/dung.json chan moi loi
goi sau) / 1; cha crash -> lan sau thay lan chay thieu ket_thuc.json -> cha_gian_doan -> 40.
CON: bat buoc -B; kiem lan chay truoc moi ghi; faulthandler rieng; moi truong (TEMP/cau hinh thu vien trong phien) TRUOC
import ben thu ba; kiem LAI toan bo dau vao (kiem_dau_vao.kiem_tat_ca + manifest B + dieu_phoi_bdd.nap_phien_b) truoc moi
buoc phu thuoc; nhat ky don vi tien_do.jsonl (fsync); artifact do (chua marker hop le, ke ca .dang_ghi) CHUYEN nguyen byte
vao van_hanh/artifact_do/ truoc khi ghi lai; marker hong -> exit 2. Chi goi helper hien co (dieu_phoi_bdd, thong_ke_bdd).
Buoc da xong (marker hop le) KHONG mien kiem: tien-kiem / mo-phien kiem lai dau vao hien tai + manifest truoc khi bo qua
(B0-F4); lap-tracker / cham-lai / hau-kiem kiem lai marker + bam + so byte artifact TUNG run, khong tin bang ket_qua (B0-F5).
Chinh sach crash: toi da 2 thu lai moi don vi; dung o crash thu 3 / don vi, thu 4 / phien, thu 2 cua kiem-khoa.
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
T0 = time.time()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hop_dong_bdd as H  # noqa: E402
import kiem_dau_vao as KD  # noqa: E402  (muc module chi thu vien chuan)
D, VA = KD.D, KD.VA

LUOT = "R61_cong_B_20261026"
SCRIPT = Path(__file__).resolve()
BUOC = ("tien-kiem", "mo-phien", "tracking", "cham", "lap-tracker", "cham-lai", "tong-hop", "kiem-khoa", "hau-kiem")
BUOC_KHOA = ("kiem-khoa",)
NGUONG = {"don_vi": 3, "phien": 4, "buoc_khoa": 2}
EXIT_CRASH, EXIT_DUNG = VA.EXIT_CRASH, VA.EXIT_DUNG
NGAN_SACH_TOI_DA = 420.0
#: Diem tiem FIXTURE cua CHA (con: khoi_dong, truoc_ghi, sau_ghi, dang_ghi, sau_marker, bo_faulthandler; kieu native/python/cho)
DIEM_CHA = ("cha_sau_tao_con", "cha_khi_con_chay", "cha_sau_con", "job_loi", "resume_loi", "pid_khong_xac_dinh")
PHIEN_CHAP_NHAN: Path | None = None
TIEM: list[dict] = []


# ---------------------------------------------------------------------------
# Chung
# ---------------------------------------------------------------------------
def tuy_chon_python() -> list[str]:
    out = []
    for x in list(getattr(sys, "orig_argv", []))[1:]:
        if not x.startswith("-"):
            break
        out.append(x)
    return out


def kiem_bytecode() -> tuple[dict, list[str]]:
    """Bang chung chong ghi bytecode: interpreter tuyet doi da chi dinh + co -B that su tren dong lenh (khong chi env)."""
    d = {"executable": sys.executable, "orig_argv": list(getattr(sys, "orig_argv", [])), "tuy_chon": tuy_chon_python(),
         "flags_dont_write_bytecode": sys.flags.dont_write_bytecode, "sys_dont_write_bytecode": sys.dont_write_bytecode,
         "PYTHONDONTWRITEBYTECODE": os.environ.get("PYTHONDONTWRITEBYTECODE")}
    loi = []
    if os.path.normcase(os.path.abspath(sys.executable)) != os.path.normcase(os.path.abspath(H.PYTHON)):
        loi.append(f"interpreter {sys.executable} != {H.PYTHON}")
    if "-B" not in d["tuy_chon"] or sys.flags.dont_write_bytecode != 1 or not sys.dont_write_bytecode:
        loi.append(f"thieu -B (tuy chon {d['tuy_chon']}, flags {sys.flags.dont_write_bytecode})")
    return d, loi


def cmd_con(d: Path) -> list[str]:
    return [H.PYTHON, "-B", str(SCRIPT), "--con", str(d)]


def tiem(diem: str, dv: str | None = None) -> None:
    for t in TIEM:
        if t.get("diem") == diem and t.get("don_vi") in (None, dv):
            if t.get("kieu") == "python":
                raise RuntimeError(f"[TIEM LOI PYTHON] {diem} {dv}")
            if t.get("kieu") == "cho":
                print(f"[CHO] pid {os.getpid()} {diem}", flush=True)
                time.sleep(float(t.get("giay", 60)))
                continue
            VA.crash_native()


def vung_b(hd: dict, P: Path) -> list[Path]:
    goc = KD.VUNG_FIXTURE_B if hd["_fixture"] else H.VUNG_THUC_THI
    if not H._trong(P, goc) or P == goc:
        raise H.LoiHopDong(f"phien B {'fixture' if hd['_fixture'] else 'that'} phai nam duoi {H.rel(goc)}/<phien>")
    H.kiem_duong_ghi(P, [P])                            # lexical + realpath + KHONG reparse o moi thanh phan
    return [P]


# ---------------------------------------------------------------------------
# CON: nhat ky don vi, sao luu artifact do, ghi bao toan
# ---------------------------------------------------------------------------
class NhatKyB:
    """tien_do.jsonl: bat_dau (lan_thu) -> xong | bo_qua; dung giua don vi khi qua han_mem hoac dat gioi han."""

    def __init__(self, p: Path, vung, truoc: dict, han_mem: float, gioi_han: int | None, P: Path, stt: int):
        self.p, self.vung, self.truoc, self.A, self.stt = p, vung, truoc, P, stt
        self.han_mem, self.gioi_han = han_mem, gioi_han
        self.hien_tai, self.pha, self.lam, self.t0 = None, None, 0, time.time()
        self.goc_do: Path | None = None              # None -> <phien>/van_hanh/artifact_do (van_hanh_cong_a.sao_luu_do)

    def ghi(self, **d):
        VA.ghi_dong(self.p, {"t": round(time.time(), 3), **d}, self.vung)

    def bat_dau(self, dv: str, pha: str = "chay"):
        self.hien_tai, self.pha, self.t0 = dv, pha, time.time()
        self.ghi(su_kien="bat_dau", don_vi=dv, pha=pha, lan_thu=self.truoc.get(dv, 0) + 1)
        if pha == "chay":
            tiem("truoc_ghi", dv)

    def xong(self, dv: str, **info):
        tiem("sau_marker", dv)
        self.ghi(su_kien="xong", don_vi=dv, giay=round(time.time() - self.t0, 3), **info)
        if self.pha == "chay":
            self.lam += 1
        self.hien_tai = self.pha = None

    def bo_qua(self, dv: str):
        self.ghi(su_kien="bo_qua", don_vi=dv)

    def kiem_doan(self, mo_ta: str):
        if time.time() > self.han_mem or (self.gioi_han is not None and self.lam >= self.gioi_han):
            raise H.ChuaXong(f"{mo_ta}: dung giua don vi sau {self.lam} don vi moi (han_mem/gioi han)")


def tiem_dang_ghi(nk: NhatKyB, p: Path, b: bytes) -> None:
    """FIXTURE 'dang_ghi': nua bytes vao '<file>.dang_ghi' (fsync) roi access violation -- mo phong crash giua ghi."""
    for t in TIEM:
        if t.get("diem") == "dang_ghi" and t.get("don_vi") in (None, nk.hien_tai) and t.get("tep") in (None, p.name):
            tam = H.kiem_duong_ghi(p.with_name(p.name + ".dang_ghi"), nk.vung)
            H.tao_thu_muc(tam.parent, nk.vung)
            with open(tam, "wb") as f:
                f.write(b[:max(1, len(b) // 2)])
                f.flush()
                os.fsync(f.fileno())
            VA.crash_native()


def sao_luu(nk: NhatKyB, paths: list[Path], dv: str) -> list[dict]:
    return VA.sao_luu_do(nk, nk.A, nk.vung, paths, dv, nk.stt)


def ghi_bao_toan(nk: NhatKyB, p: Path, obj) -> str:
    """Ban cu (ke ca .dang_ghi) -> sao luu nguyen byte TRUOC; khong ghi de."""
    sao_luu(nk, VA.kem_dang_ghi([p]), nk.hien_tai or "ngoai_don_vi")
    tiem_dang_ghi(nk, p, (json.dumps(obj, indent=1, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
    return H.ghi_json(p, obj, nk.vung)


def ghi_bytes_bao_toan(nk: NhatKyB, p: Path, b: bytes) -> str:
    sao_luu(nk, VA.kem_dang_ghi([p]), nk.hien_tai or "ngoai_don_vi")
    tiem_dang_ghi(nk, p, b)
    return H.ghi_bytes(p, b, nk.vung)


def ghi_marker_bt(nk: NhatKyB, p: Path, artifact: list[Path], them: dict) -> dict:
    sao_luu(nk, [p.with_name(p.name + ".dang_ghi")], nk.hien_tai or "ngoai_don_vi")
    return H.ghi_marker(p, artifact, nk.vung, them)


# ---------------------------------------------------------------------------
# CON: buoc
# ---------------------------------------------------------------------------
def kiem_vao(ctx, hd, P: Path, g: dict) -> dict:
    """Kiem LAI toan bo dau vao (khong dung ket qua cu); sau mo-phien so them manifest B."""
    kq = KD.kiem_tat_ca(hd, ctx, g["phe_duyet"], P)
    mp = P / "manifest_b.json"
    if mp.exists() or H.la_reparse(mp):
        loi = KD.so_manifest(_manifest_luu(mp, "manifest_b.json"), kq["manifest"])
        if loi:
            raise KD.LoiDauVao("manifest", loi)
    return kq


def _manifest_luu(p: Path, mo_ta: str) -> dict:
    d = KD.doc_json_hd(p, "manifest", mo_ta)
    if not isinstance(d.get("manifest"), dict):
        raise KD.LoiDauVao("manifest", [f"{mo_ta}: thieu / sai kieu truong manifest"])
    return d["manifest"]


def kiem_tho(ctx, hd, P: Path, vung, nk, g) -> dict:
    """kiem_tat_ca HIEN TAI; khong dat -> ghi ket qua THO cua lan nay (khong phai marker) roi nem LoiDauVao."""
    try:
        return KD.kiem_tat_ca(hd, ctx, g["phe_duyet"], P)
    except KD.LoiDauVao as e:
        H.ghi_json(P / "tien_kiem_b" / f"khong_dat_lan_{nk.stt:04d}.json", e.kq or {"nhom_loi_dau": e.nhom, "loi": e.loi}, vung)
        raise


def nap_b(ctx, hd, P: Path, nk: NhatKyB, g: dict) -> dict:
    dv = f"{g['buoc']}:chuan_bi"
    nk.bat_dau(dv, "chuan_bi")
    st, loi = H.kiem_marker(P / "mo_phien.xong.json", [P / "phien.json", P / "manifest_b.json"], {"buoc": "mo-phien", "luot": LUOT})
    if st != "xong":
        raise H.LoiHopDong(f"phien B chua mo / marker mo-phien {st}: {loi[:3]}")
    kiem_vao(ctx, hd, P, g)                          # noi dung goi A + anh khoa (= tap kiem cua D.nap_phien_b, mo rong)
    ph = KD.doc_json_hd(P / "phien.json", "phien_b", "phien.json")   # phan con lai cua D.nap_phien_b
    sai = KD.loi_schema(ph, {k: str for k in ("goi_a", "goi_a_sha256", "khoa", "khoa_sha256", "review_a", "review_a_sha256",
                                              "plan_b", "plan_b_sha256", "giao_thuc_sha256", "phien_A")})
    if sai:
        raise KD.LoiDauVao("phien_b", [f"phien.json {x}" for x in sai])
    for k in ("goi_a", "khoa", "review_a", "plan_b"):
        f = KD.duong(ph[k])
        if not f.is_file() or H.sha256_file(f) != ph[f"{k}_sha256"]:
            raise H.LoiHopDong(f"{k} doi sau mo-phien")
    if H.sha256_file(ctx.gt_path) != ph["giao_thuc_sha256"]:
        raise H.LoiHopDong("giao thuc doi sau mo-phien")
    ph["goi"] = KD.doc_json_hd(KD.duong(ph["goi_a"]), "goi", "goi_a.json")
    if ph["goi"].get("khoa") != ph["khoa"] or ph["goi"].get("phien_A") != ph["phien_A"]:
        raise H.LoiHopDong("phien.json khong khop goi A")
    nk.xong(dv)
    return ph


def con_tien_kiem(ctx, hd, P, vung, nk, g) -> int:
    dv, out = "tien-kiem", P / "tien_kiem_b" / "ket_qua.json"
    mk = P / "tien_kiem_b" / "xong.json"
    if H.da_xong(mk, [out], {"buoc": "tien-kiem", "luot": LUOT}):
        # B0-F4: marker cu hop le KHONG mien kiem: kiem LAI dau vao hien tai + so manifest voi ket qua da luu TRUOC khi bo
        # qua; artifact hoan tat giu nguyen (ket qua kiem lai ghi file rieng theo lan chay).
        nk.bat_dau("tien-kiem:kiem_lai", "chuan_bi")
        kq = kiem_tho(ctx, hd, P, vung, nk, g)
        loi = KD.so_manifest(_manifest_luu(out, "tien_kiem_b/ket_qua.json"), kq["manifest"])
        H.ghi_json(P / "tien_kiem_b" / f"kiem_lai_lan_{nk.stt:04d}.json",
                   {"dat": not loi, "loi": loi, "nhom": {k: v["dat"] for k, v in kq["nhom"].items()}, "luc": kq["luc"]}, vung)
        if loi:
            raise KD.LoiDauVao("manifest", loi)
        nk.xong("tien-kiem:kiem_lai")
        nk.bo_qua(dv)
        print("tien-kiem: da co ket qua hop le; kiem lai dau vao hien tai dat")
        return 0
    nk.bat_dau(dv)
    kq = kiem_tho(ctx, hd, P, vung, nk, g)            # ket qua THO cua lan kiem khong dat (khong phai marker)
    ghi_bao_toan(nk, out, kq)
    ghi_marker_bt(nk, mk, [out], {"buoc": "tien-kiem", "luot": LUOT})
    nk.xong(dv)
    print(f"tien-kiem: dat ({len(kq['van_hanh_bang'])} file van_hanh A, ma B {kq['ma_b']['sha256'][:12]})")
    return 0


def con_mo_phien(ctx, hd, P, vung, nk, g) -> int:
    dv, mk = "mo-phien", P / "mo_phien.xong.json"
    pj, mp = P / "phien.json", P / "manifest_b.json"
    if H.da_xong(mk, [pj, mp], {"buoc": "mo-phien", "luot": LUOT}):
        nap_b(ctx, hd, P, nk, g)                       # B0-F4: kiem LAI dau vao hien tai + manifest B + phien.json truoc khi bo qua
        nk.bo_qua(dv)
        print("mo-phien: phien B da mo; kiem lai dau vao hien tai dat")
        return 0
    tk = P / "tien_kiem_b"
    if not H.da_xong(tk / "xong.json", [tk / "ket_qua.json"], {"buoc": "tien-kiem", "luot": LUOT}):
        raise H.LoiHopDong("mo-phien: chua co tien-kiem dat")
    nk.bat_dau(dv)
    kq = kiem_tho(ctx, hd, P, vung, nk, g)            # kiem LAI, khong dung ket qua tien kiem
    loi = KD.so_manifest(_manifest_luu(tk / "ket_qua.json", "tien_kiem_b/ket_qua.json"), kq["manifest"])
    if loi:
        raise KD.LoiDauVao("manifest", loi)
    A = KD.duong(hd["a"]["phien"])
    gp, pd = A / "goi_a.json", kq["nhom"]["phe_duyet"]["tho"]
    goi = KD.doc_json_hd(gp, "goi", "goi_a.json")
    ghi_bao_toan(nk, mp, {"luot": LUOT, "manifest": kq["manifest"], "ma_b": kq["ma_b"]})
    ghi_bao_toan(nk, pj, {"luot": LUOT, "goi_a": H.rel(gp), "goi_a_sha256": H.sha256_file(gp), "khoa": goi["khoa"],
                          "khoa_sha256": goi["khoa_sha256"], "phien_A": goi["phien_A"],
                          "review_a": kq["nhom"]["review"]["tho"]["duong"], "review_a_sha256": hd["review_a"]["sha256"],
                          "plan_b": pd["duong"], "plan_b_sha256": pd["sha256"], "giao_thuc_sha256": H.sha256_file(ctx.gt_path),
                          "manifest_b_sha256": H.sha256_file(mp)})
    ghi_marker_bt(nk, mk, [pj, mp], {"buoc": "mo-phien", "luot": LUOT})
    nk.xong(dv)
    print("mo-phien B: ok")
    return 0


def _quet_marker(ctx, root: Path, runs: list[dict], videos: list[str]) -> None:
    for r in runs:                                     # marker hong o BAT KY dau -> 2 truoc moi ghi
        for v in videos:
            if (root / ctx.split / r["run"] / "xong" / f"{v}.xong.json").exists():
                D.xong_don_vi(root / ctx.split / r["run"], r["run"], v)


def _don_vi_tracking(ctx, ph, nk, vung, root: Path, r: dict, v: str, ten: str, reg) -> None:
    run_dir = root / ctx.split / r["run"]
    dv = f"{ten}:{r['run']}:{v}"
    if D.xong_don_vi(run_dir, r["run"], v):            # marker hong -> LoiHopDong
        if dv in nk.truoc:
            nk.bo_qua(dv)                              # bang chung: don vi da bat dau truoc do KHONG chay lai
        return
    nk.kiem_doan(ten)
    nk.bat_dau(dv)
    D.kiem_cache_tu_goi(ph)(r["detector"], v)          # cache A dung goi truoc MOI don vi
    mk = run_dir / "xong" / f"{v}.xong.json"
    sao_luu(nk, VA.kem_dang_ghi(D.file_don_vi(run_dir, v) + [mk]), dv)
    D.chay_don_vi(ctx, r, v, H.ROOT / ph["phien_A"] / "cache" / r["detector"], run_dir, reg, vung)
    nk.xong(dv)


def con_tracking(ctx, hd, P, vung, nk, g) -> int:
    ph = nap_b(ctx, hd, P, nk, g)
    import run_tracking as rt
    reg = rt.tracker_registry()
    runs = ctx.runs()
    _quet_marker(ctx, P / "tracking", runs, ctx.videos)
    for r in runs:
        for v in ctx.videos:
            _don_vi_tracking(ctx, ph, nk, vung, P / "tracking", r, v, "tracking", reg)
    print(f"tracking: xong ({nk.lam} don vi moi)")
    return 0


def _don_vi_cham(ctx, nk, vung, P: Path, run: str, goc: str, ten: str) -> None:
    out = P / goc / run
    dv = f"{ten}:{run}"
    if D.xong_cham(out, run):
        if dv in nk.truoc:
            nk.bo_qua(dv)
        return
    nk.kiem_doan(ten)
    nk.bat_dau(dv)
    sao_luu(nk, VA.kem_dang_ghi(D.file_cham(out) + [out / "cham.xong.json", out / "trackeval_stdout.txt",
                                                   out / "trackeval_error.log"]) + [P / goc / "_trackeval" / run], dv)
    D.cham_run(ctx, run, ctx.videos, P / "tracking", P / goc, vung)
    nk.xong(dv)


def con_cham(ctx, hd, P, vung, nk, g) -> int:
    nap_b(ctx, hd, P, nk, g)
    runs = ctx.runs()
    thieu = [(r["run"], v) for r in runs for v in ctx.videos
             if not D.xong_don_vi(P / "tracking" / ctx.split / r["run"], r["run"], v)]       # hong -> 2
    if thieu:                                          # chua du MOI don vi tracking -> khong cham run nao
        raise H.LoiHopDong(f"thieu {len(thieu)} don vi tracking (vd {thieu[:3]}) -- khong cham")
    for r in runs:
        D.xong_cham(P / "cham" / r["run"], r["run"])   # marker cham hong -> 2 truoc moi ghi
    for r in runs:
        _don_vi_cham(ctx, nk, vung, P, r["run"], "cham", "cham")
    return 0


def kiem_lap(ctx, P: Path) -> tuple[list[dict], list[str]]:
    """B0-F5: kiem LAI artifact tai lap tung run (marker + bam tung file) va so byte voi tracking goc -- khong tin bang
    ket_qua. Tra ve (bang {run, trung_byte} cung dang bang da ghi, loi)."""
    video, ket, loi = ctx.gt["tai_lap_cong_B"]["video"], [], []
    for r in ctx.runs():
        goc, lap = P / "tracking" / ctx.split / r["run"], P / "lap" / ctx.split / r["run"]
        dt = {"run": r["run"], "video": video}
        st_g, l_g = H.kiem_marker(goc / "xong" / f"{video}.xong.json", D.file_don_vi(goc, video), dt)
        st_l, l_l = H.kiem_marker(lap / "xong" / f"{video}.xong.json", D.file_don_vi(lap, video), dt)
        trung = st_g == "xong" and st_l == "xong" and all(D.so_byte(goc / s / f"{video}{e}", lap / s / f"{video}{e}")
                                                          for s, e in (("data", ".txt"), ("data_cls", ".csv")))
        ket.append({"run": r["run"], "trung_byte": trung})
        if st_g != "xong":
            loi.append(f"tracking goc {r['run']}/{video} marker {st_g}: {l_g[:2]}")
        if st_l != "xong":
            loi.append(f"tai lap {r['run']}/{video} marker {st_l}: {l_l[:2]}")
        elif not trung:
            loi.append(f"tai lap {r['run']}/{video} lech byte voi tracking goc")
    return ket, loi


def kiem_cham_lai(ctx, P: Path) -> tuple[list[dict], list[str]]:
    """B0-F5: kiem LAI marker + bam artifact cham lai / cham goc tung run va so byte 3 file -- khong tin bang ket_qua."""
    ket, loi = [], []
    for run in ctx.gt["tai_lap_cong_B"]["cham_lai"]:
        goc, lai = P / "cham" / run, P / "cham_lai" / run
        st_g, l_g = H.kiem_marker(goc / "cham.xong.json", D.file_cham(goc), {"run": run})
        st_l, l_l = H.kiem_marker(lai / "cham.xong.json", D.file_cham(lai), {"run": run})
        trung = st_g == "xong" and st_l == "xong" and all(D.so_byte(goc / f, lai / f) for f in
                                                          ("per_video_metrics.csv", "combined_metrics.csv", "raw_fields.json"))
        ket.append({"run": run, "trung_byte": trung})
        if st_g != "xong":
            loi.append(f"cham goc {run} marker {st_g}: {l_g[:2]}")
        if st_l != "xong":
            loi.append(f"cham lai {run} marker {st_l}: {l_l[:2]}")
        elif not trung:
            loi.append(f"cham lai {run} lech byte voi cham goc")
    return ket, loi


def _kiem_bang(p: Path, mk: Path, danh_tinh: dict, ket: list[dict]) -> list[str]:
    """Bang ket_qua da ghi: marker hop le VA noi dung = bang vua tinh lai tu artifact."""
    st, l = H.kiem_marker(mk, [p], danh_tinh)
    if st != "xong":
        return [f"{H.rel(p)} marker {st}: {l[:2]}"]
    d, l = KD.doc_json_dv(p, H.rel(p), list)
    return l if l else ([] if d == ket else [f"{H.rel(p)} khac bang tinh lai tu artifact"])


def con_lap_tracker(ctx, hd, P, vung, nk, g) -> int:
    ph = nap_b(ctx, hd, P, nk, g)
    video = ctx.gt["tai_lap_cong_B"]["video"]
    kq_p, mk = P / "lap" / f"ket_qua_{video}.json", P / "lap" / f"ket_qua_{video}.xong.json"
    if H.da_xong(mk, [kq_p], {"buoc": "lap-tracker", "video": video}):
        nk.bat_dau("lap:kiem_lai", "chuan_bi")        # B0-F5: kiem lai artifact tung run truoc khi bo qua
        ket, loi = kiem_lap(ctx, P)
        loi += _kiem_bang(kq_p, mk, {"buoc": "lap-tracker", "video": video}, ket)
        if loi:
            raise H.LoiHopDong(f"lap-tracker kiem lai: {loi[:6]}")
        nk.xong("lap:kiem_lai")
        nk.bo_qua("lap:so_sanh")
        return 0
    import run_tracking as rt
    reg = rt.tracker_registry()
    runs = ctx.runs()
    _quet_marker(ctx, P / "lap", runs, [video])
    for r in runs:
        if not D.xong_don_vi(P / "tracking" / ctx.split / r["run"], r["run"], video):
            raise H.LoiHopDong(f"lap-tracker: tracking goc {r['run']}/{video} chua xong")
    for r in runs:
        _don_vi_tracking(ctx, ph, nk, vung, P / "lap", r, video, "lap", reg)
    nk.bat_dau("lap:so_sanh")
    ket, loi = kiem_lap(ctx, P)
    ghi_bao_toan(nk, kq_p, ket)
    if loi:
        raise H.LoiHopDong(f"tai lap tracker lech: {loi[:6]}")
    ghi_marker_bt(nk, mk, [kq_p], {"buoc": "lap-tracker", "video": video})
    nk.xong("lap:so_sanh", so_run=len(ket))
    return 0


def con_cham_lai(ctx, hd, P, vung, nk, g) -> int:
    nap_b(ctx, hd, P, nk, g)
    runs = ctx.gt["tai_lap_cong_B"]["cham_lai"]
    kq_p, mk = P / "cham_lai" / "ket_qua.json", P / "cham_lai" / "ket_qua.xong.json"
    if H.da_xong(mk, [kq_p], {"buoc": "cham-lai"}):
        nk.bat_dau("cham-lai:kiem_lai", "chuan_bi")   # B0-F5: kiem lai artifact tung run truoc khi bo qua
        ket, loi = kiem_cham_lai(ctx, P)
        loi += _kiem_bang(kq_p, mk, {"buoc": "cham-lai"}, ket)
        if loi:
            raise H.LoiHopDong(f"cham-lai kiem lai: {loi[:6]}")
        nk.xong("cham-lai:kiem_lai")
        nk.bo_qua("cham-lai:so_sanh")
        return 0
    for run in runs:
        D.xong_cham(P / "cham_lai" / run, run)
        if not D.xong_cham(P / "cham" / run, run):
            raise H.LoiHopDong(f"cham-lai: cham goc {run} chua xong")
    for run in runs:
        _don_vi_cham(ctx, nk, vung, P, run, "cham_lai", "cham-lai")
    nk.bat_dau("cham-lai:so_sanh")
    ket, loi = kiem_cham_lai(ctx, P)
    ghi_bao_toan(nk, kq_p, ket)
    if loi:
        raise H.LoiHopDong(f"cham lai lech: {loi[:6]}")
    ghi_marker_bt(nk, mk, [kq_p], {"buoc": "cham-lai"})
    nk.xong("cham-lai:so_sanh", so_run=len(ket))
    return 0


def kt_khoi(hd, ctx) -> int:
    return int(hd["bootstrap_khoi"]["kich_thuoc_khoi_fixture" if ctx.fixture else "kich_thuoc_khoi"])


def _du_lieu_thong_ke(ctx, P: Path):
    import thong_ke_bdd as T
    import thong_ke_tiep_tuc as TT
    raw, comb, dau_vao = D.doc_cham(ctx, P)          # 36 marker cham hop le (hong -> 2) + sha256 dau vao
    sai = {run: T.sai_so_tai_to_hop(raw[run], ctx.videos, comb[run]) for run in raw}
    dung = ctx.gt["thong_ke"]["dung_sai_tai_to_hop"]
    vuot = [run for run, s in sai.items() if any(s[m] > dung[m] for m in T.METRIC)]
    if vuot:
        raise H.LoiHopDong(f"tai to hop lech evaluator vuot dung sai: {vuot[:3]}")
    nhom = TT.nhom_raws(ctx.runs(), raw)
    st, obs = TT.chuan_bi(nhom, ctx.videos)
    return raw, comb, dau_vao, sai, dung, nhom, st, obs


def _khoi(nk, ctx, hd, chi_so, st, thu_muc: Path, ten: str, dinh_danh: dict) -> list[Path]:
    """Moi khoi chi so mot don vi: <thu_muc>/<k>.npy + marker (danh tinh: khoi, hang, chi so, dau vao)."""
    import thong_ke_tiep_tuc as TT
    out = []
    for k, (i0, i1) in enumerate(TT.khoi(ctx.so_mau, kt_khoi(hd, ctx))):
        p, mk = thu_muc / f"{k:04d}.npy", thu_muc / f"{k:04d}.xong.json"
        dt = {"khoi": k, "i0": i0, "i1": i1, **dinh_danh}
        out.append(p)
        dv = f"{ten}:khoi:{k:04d}"
        if H.da_xong(mk, [p], dt):
            if dv in nk.truoc:
                nk.bo_qua(dv)
            continue
        nk.kiem_doan(ten)
        nk.bat_dau(dv)
        sao_luu(nk, VA.kem_dang_ghi([p, mk]), dv)
        ghi_bytes_bao_toan(nk, p, TT.npy_bytes(TT.mau_khoi(st, chi_so, i0, i1)))
        ghi_marker_bt(nk, mk, [p], dt)
        nk.xong(dv)
    return out


def con_tong_hop(ctx, hd, P, vung, nk, g) -> int:
    import thong_ke_bdd as T
    import thong_ke_tiep_tuc as TT
    nap_b(ctx, hd, P, nk, g)
    out = P / "tong_hop"
    tep = [out / f for f in ("ci_108.csv", "bang_chinh.csv", "kiem_chung.json", "chi_so_bootstrap.npy")]
    if H.da_xong(out / "xong.json", tep, {"buoc": "tong-hop"}):
        nk.bo_qua("tong-hop:ghi")
        return 0
    raw, comb, dau_vao, sai, dung, nhom, st, obs = _du_lieu_thong_ke(ctx, P)
    p_cs = out / "chi_so_bootstrap.npy"
    if p_cs.exists() or H.la_reparse(p_cs):
        chi_so = D.doc_chi_so(ctx, p_cs)             # tai dung: dtype/shape/mien/TRUNG default_rng(0) truoc moi ghi
    else:
        nk.bat_dau("tong-hop:chi_so")
        sao_luu(nk, [p_cs.with_name(p_cs.name + ".dang_ghi")], "tong-hop:chi_so")
        H.ghi_bytes(p_cs, TT.npy_bytes(T.tao_chi_so(len(ctx.videos), ctx.so_mau, 0)), vung)
        chi_so = D.doc_chi_so(ctx, p_cs)
        nk.xong("tong-hop:chi_so")
    dd = {"chi_so_sha256": H.sha256_file(p_cs), "dau_vao_sha256": H.sha256_bytes(H.chuan_json(dau_vao).encode("utf-8"))}
    ps = _khoi(nk, ctx, hd, chi_so, st, out / "khoi", "tong-hop", dd)
    nk.kiem_doan("tong-hop")
    nk.bat_dau("tong-hop:ghi")
    rows = TT.hang_ci(nhom, obs, TT.ghep([TT.doc_npy(p) for p in ps], ctx.so_mau))
    if len(rows) != ctx.gt["thong_ke"]["so_ci"]:
        raise H.LoiHopDong(f"so CI {len(rows)} != {ctx.gt['thong_ke']['so_ci']}")
    b_ci, b_bang = D.csv_bytes(rows), D.csv_bytes(T.bang_chinh(comb))
    ghi_bytes_bao_toan(nk, tep[0], b_ci)
    ghi_bytes_bao_toan(nk, tep[1], b_bang)
    ghi_bao_toan(nk, tep[2], {
        "so_ci": len(rows), "so_mau": ctx.so_mau, "so_video": len(ctx.videos), "rng": "numpy.random.default_rng(0)",
        "chi_so_sha256": dd["chi_so_sha256"], "chi_so_shape": list(chi_so.shape), "chi_so_dtype": str(chi_so.dtype),
        "tai_tinh_tu_chi_so_luu_trung": True, "ci_108_sha256": H.sha256_bytes(b_ci), "bang_chinh_sha256": H.sha256_bytes(b_bang),
        "dau_vao_sha256": dau_vao, "sai_so_tai_to_hop_max": {m: max(s[m] for s in sai.values()) for m in T.METRIC},
        "dung_sai": dung, "chia_khoi": {"kich_thuoc": kt_khoi(hd, ctx), "so_khoi": len(ps),
                                        "khoi_sha256": [H.sha256_file(p) for p in ps],
                                        "ghi_chu": "phan vi tren TOAN mau ghep; khong gop CI tung khoi"}})
    ghi_marker_bt(nk, out / "xong.json", tep, {"buoc": "tong-hop"})
    nk.xong("tong-hop:ghi", so_ci=len(rows))
    print(f"tong-hop: {len(rows)} CI, {ctx.so_mau} mau, {len(ps)} khoi")
    return 0


def con_kiem_khoa(ctx, hd, P, vung, nk, g) -> int:
    ph = nap_b(ctx, hd, P, nk, g)
    nk.bat_dau("kiem-khoa")
    D.buoc_kiem_khoa(ctx, H.ROOT / ph["khoa"], P, vung, H.ROOT / ph["phien_A"])     # ghi <phien_B>/kiem_khoa/<thoi diem>/
    nk.xong("kiem-khoa")
    return 0


def con_hau_kiem(ctx, hd, P, vung, nk, g) -> int:
    """Hau kiem KHONG tin tom tat: dem marker, tai tinh MOI khoi bootstrap tu raw + chi so luu (theo don vi), so byte khoi /
    ci_108.csv / bang_chinh.csv / kiem_chung, tai lap, cham lai, kiem-khoa B; dau vao A da kiem lai trong nap_b."""
    import thong_ke_bdd as T
    import thong_ke_tiep_tuc as TT
    ph = nap_b(ctx, hd, P, nk, g)
    runs, loi = ctx.runs(), []
    ca = H.ROOT / ph["phien_A"] / "cache"
    n_cache = sum(1 for d in ctx.detectors() for v in ctx.videos
                  if H.kiem_marker(ca / d / f"{v}.xong.json", D.file_marker_cache(ca / d, v), {"detector": d, "video": v})[0] == "xong")
    n_trk = sum(1 for r in runs for v in ctx.videos
                if H.kiem_marker(P / "tracking" / ctx.split / r["run"] / "xong" / f"{v}.xong.json",
                                 D.file_don_vi(P / "tracking" / ctx.split / r["run"], v), {"run": r["run"], "video": v})[0] == "xong")
    n_cham = sum(1 for r in runs if H.kiem_marker(P / "cham" / r["run"] / "cham.xong.json", D.file_cham(P / "cham" / r["run"]),
                                                  {"run": r["run"]})[0] == "xong")
    ky = {"cache": len(ctx.detectors()) * len(ctx.videos), "tracking": len(runs) * len(ctx.videos), "cham": len(runs)}
    for k, n in (("cache", n_cache), ("tracking", n_trk), ("cham", n_cham)):
        if n != ky[k]:
            loi.append(f"{k}: {n} != {ky[k]}")
    out = P / "tong_hop"
    tep_th = [out / f for f in ("ci_108.csv", "bang_chinh.csv", "kiem_chung.json", "chi_so_bootstrap.npy")]
    st_th, l_th = H.kiem_marker(out / "xong.json", tep_th, {"buoc": "tong-hop"})
    khoi_lech = []
    # Moi lan hau kiem HOAN TAT dong mot vong; lan sau mo vong MOI va tai tinh lai MOI khoi (khong dung ket qua kiem cu);
    # trong mot vong chua dong (exit 10 / crash) tiep tuc theo khoi.
    hk = P / "hau_kiem_b"
    vs = sorted(n for n in os.listdir(hk) if n.startswith("vong_")) if hk.is_dir() else []
    R = hk / vs[-1] if vs and not (hk / vs[-1] / "xong_vong.json").exists() else hk / f"vong_{len(vs) + 1:02d}"
    try:
        if st_th == "chua" or not all(p.is_file() for p in tep_th):
            raise H.LoiHopDong("thong ke chua tong hop")
        if st_th == "hong":                            # van TAI TINH de chi ro dau ra nao bi sua
            loi.append(f"tong_hop/xong.json hong: {l_th[:3]}")
        kc, l_kc = KD.doc_json_dv(out / "kiem_chung.json", "tong_hop/kiem_chung.json")
        loi += l_kc
        kc = kc or {}
        raw, comb, dau_vao, sai, dung, nhom, st, obs = _du_lieu_thong_ke(ctx, P)
        chi_so = D.doc_chi_so(ctx, out / "chi_so_bootstrap.npy")
        dd = {"chi_so_sha256": H.sha256_file(out / "chi_so_bootstrap.npy"),
              "dau_vao_sha256": H.sha256_bytes(H.chuan_json(dau_vao).encode("utf-8"))}
        ps = _khoi(nk, ctx, hd, chi_so, st, R / "khoi", "hau-kiem", {**dd, "vong": R.name})
        nk.kiem_doan("hau-kiem")
        nk.bat_dau("hau-kiem:ghi")
        for p in ps:
            q = out / "khoi" / p.name
            if not q.is_file() or q.read_bytes() != p.read_bytes():
                khoi_lech.append(p.name)
        rows = TT.hang_ci(nhom, obs, TT.ghep([TT.doc_npy(p) for p in ps], ctx.so_mau))
        b_ci, b_bang = D.csv_bytes(rows), D.csv_bytes(T.bang_chinh(comb))
        for f, b in (("ci_108.csv", b_ci), ("bang_chinh.csv", b_bang)):
            if (out / f).read_bytes() != b:
                loi.append(f"{f} khac tinh lai tu chi so luu")
        kyk = {"so_ci": ctx.gt["thong_ke"]["so_ci"], "so_mau": ctx.so_mau, "so_video": len(ctx.videos),
               "chi_so_sha256": dd["chi_so_sha256"], "ci_108_sha256": H.sha256_bytes(b_ci),
               "bang_chinh_sha256": H.sha256_bytes(b_bang), "dau_vao_sha256": dau_vao, "tai_tinh_tu_chi_so_luu_trung": True}
        loi += [f"kiem_chung {k} lech" for k, v in kyk.items() if H.chuan_json(kc.get(k)) != H.chuan_json(v)]
        loi += [f"khoi bootstrap {x} khac tai tinh" for x in khoi_lech]
    except H.LoiHopDong as e:
        if isinstance(e, KD.LoiDauVao):
            raise
        loi.append(f"thong ke: {e}")
        if nk.hien_tai is None:
            nk.bat_dau("hau-kiem:ghi")
    # B0-F5: tai lap / cham lai kiem LAI tu artifact tung run (marker + bam + so byte), roi doi chieu bang da ghi
    tl = ctx.gt["tai_lap_cong_B"]
    ket_l, l_l = kiem_lap(ctx, P)
    l_l += _kiem_bang(P / "lap" / f"ket_qua_{tl['video']}.json", P / "lap" / f"ket_qua_{tl['video']}.xong.json",
                      {"buoc": "lap-tracker", "video": tl["video"]}, ket_l)
    loi += [f"tai lap: {x}" for x in l_l]
    ket_c, l_c = kiem_cham_lai(ctx, P)
    l_c += _kiem_bang(P / "cham_lai" / "ket_qua.json", P / "cham_lai" / "ket_qua.xong.json", {"buoc": "cham-lai"}, ket_c)
    loi += [f"cham lai: {x}" for x in l_c]
    kk = sorted((P / "kiem_khoa").glob("*/ket_qua.json")) if (P / "kiem_khoa").exists() else []
    kq_kk, l_kk = KD.doc_json_dv(kk[-1], "kiem_khoa B") if kk else (None, ["thieu"])
    if l_kk or not kq_kk.get("dat"):
        loi.append(f"chua co kiem-khoa dat trong phien B {l_kk[:1]}")
    d = {"dat": not loi, "loi": loi, "dem": {"cache": n_cache, "tracking": n_trk, "cham": n_cham}, "ky_vong": ky,
         "khoi_lech": khoi_lech, "vong_tai_tinh": H.rel(R), "luc": time.strftime("%Y-%m-%dT%H:%M:%S")}
    ghi_bao_toan(nk, P / "hau_kiem.json", d)
    H.ghi_json(R / "xong_vong.json", {"dat": not loi, "hau_kiem_sha256": H.sha256_file(P / "hau_kiem.json")}, vung)
    print(json.dumps({k: d[k] for k in ("dat", "loi", "dem")}, ensure_ascii=False))
    if loi:
        raise H.LoiHopDong(f"hau-kiem: {loi}")
    nk.xong("hau-kiem:ghi")
    return 0


BUOC_CON = {"tien-kiem": con_tien_kiem, "mo-phien": con_mo_phien, "tracking": con_tracking, "cham": con_cham,
            "lap-tracker": con_lap_tracker, "cham-lai": con_cham_lai, "tong-hop": con_tong_hop, "kiem-khoa": con_kiem_khoa,
            "hau-kiem": con_hau_kiem}
#: goi.json cua lan chay (dau vao cua --con, kiem TRUOC moi dung; B0-F9)
SCHEMA_GOI_CON = {"stt": int, "luot": str, "buoc": str, "phien": str, "giao_thuc": str, "hop_dong": str, "phe_duyet": str,
                  "han_mem": KD.SO, "lan_thu_truoc": dict, "tiem_loi": list}


def doc_phien(pj: Path) -> dict:
    """van_hanh/phien.json: JSON hop le + snapshot_ma_b {ma, cau_hinh, sha256} dung kieu; sai -> exit 2 (F9)."""
    if H.la_reparse(pj):
        raise KD.LoiDauVao("phien_b", ["van_hanh/phien.json la reparse"])
    ph = KD.doc_json_hd(pj, "phien_b", "van_hanh/phien.json")
    sn = ph.get("snapshot_ma_b")
    if not isinstance(sn, dict) or KD.loi_schema(sn, {"ma": dict, "cau_hinh": dict, "sha256": str}):
        raise KD.LoiDauVao("phien_b", ["van_hanh/phien.json snapshot_ma_b thieu / sai kieu"])
    return ph


def kiem_lan_con(d: Path) -> tuple[dict, Path, dict, "D.Ctx"]:
    """--con: xac thuc DAY DU lan chay TRUOC moi ghi: <P>/van_hanh/lan_chay/<stt>_<buoc>, khong reparse, P dung vung
    theo che do, goi.json + phien.json cung luot/phien, lan chua dung, ma B = manifest ghim luc tao phien."""
    d = Path(os.path.abspath(d))
    if d.parent.name != "lan_chay" or d.parent.parent.name != "van_hanh":
        raise H.LoiHopDong(f"--con {d}: khong phai <phien>/van_hanh/lan_chay/<lan>")
    P = d.parents[2]
    if not d.is_dir() or not (d / "goi.json").is_file():
        raise H.LoiHopDong(f"--con {d}: thieu goi.json")
    g = KD.doc_json_hd(d / "goi.json", "lan_chay", "goi.json")
    loi = KD.loi_schema(g, SCHEMA_GOI_CON)              # B0-F9: schema TRUOC moi int()/float()/tra khoa (stt=null -> exit 2)
    if not loi and (g["buoc"] not in BUOC_CON or not all(isinstance(t, dict) for t in g["tiem_loi"])
                    or not (g.get("gioi_han_don_vi") is None or KD._dung_kieu(g["gioi_han_don_vi"], int))):
        loi.append("buoc / tiem_loi / gioi_han_don_vi sai kieu hoac ngoai danh muc")
    if loi:
        raise KD.LoiDauVao("lan_chay", [f"--con {d}: goi.json {x}" for x in loi])
    hd = KD.nap_hop_dong(g.get("hop_dong"))
    vung_b(hd, P)
    H.kiem_duong_ghi(d / "faulthandler.log", [P])
    if Path(os.path.realpath(d)) != Path(os.path.realpath(P)) / d.relative_to(P):
        raise H.LoiHopDong(f"--con {d}: duong resolve khac")
    da_dung = [f for f in ("faulthandler.log", "tien_do.jsonl", "ket_qua_con.json", "ket_thuc.json") if (d / f).exists()]
    if da_dung:
        raise H.LoiHopDong(f"--con {d}: lan chay da dung ({da_dung}) -- khong chay lai")
    pj = P / "van_hanh" / "phien.json"
    if g.get("luot") != LUOT or g.get("phien") != H.rel(P) or not d.name.startswith(f"{g['stt']:04d}_") \
            or not pj.is_file():
        raise H.LoiHopDong(f"--con {d}: goi.json khong dung luot/phien/stt hoac phien chua tao")
    ph = doc_phien(pj)
    ctx = KD.nap_ctx(g["giao_thuc"])                    # B0-F9: JSON/schema giao thuc + manifest -> exit 2
    if ph.get("luot") != LUOT or ph.get("phien") != H.rel(P) or ph.get("giao_thuc") != H.rel(ctx.gt_path) \
            or ph.get("hop_dong") != H.rel(hd["_duong"]) or ctx.fixture != hd["_fixture"]:
        raise H.LoiHopDong(f"--con {d}: phien.json / giao thuc / hop dong khong khop")
    if g.get("tiem_loi") and not hd["_fixture"]:
        raise H.LoiHopDong("tiem loi chi cho fixture")
    if KD.ma_b(hd)["sha256"] != ph["snapshot_ma_b"]["sha256"]:
        raise H.LoiHopDong("ma B / hop dong doi so voi luc tao phien")
    return g, P, hd, ctx


def con_main(d: Path) -> int:
    global TIEM
    import faulthandler
    bc, loi = kiem_bytecode()
    if loi:
        print(f"[LOI HOP DONG] {loi}", file=sys.stderr)
        return H.EXIT_HOP_DONG
    try:
        g, P, hd, ctx = kiem_lan_con(d)                # KHONG ghi gi neu sai
    except H.LoiHopDong as e:
        print(f"[LOI HOP DONG] {e}", file=sys.stderr)
        return H.EXIT_HOP_DONG
    d = Path(os.path.abspath(d))
    vung = [P]
    TIEM = list(g.get("tiem_loi") or [])
    if not any(t.get("diem") == "bo_faulthandler" for t in TIEM):
        fh = open(H.kiem_duong_ghi(d / "faulthandler.log", vung), "x", encoding="utf-8")
        faulthandler.enable(file=fh, all_threads=True)
    H.chuan_bi_moi_truong(P, vung)                     # TRUOC moi import ben thu ba
    tiem("khoi_dong")
    nk = NhatKyB(d / "tien_do.jsonl", vung, g["lan_thu_truoc"], float(g["han_mem"]), g.get("gioi_han_don_vi"), P, int(g["stt"]))
    if TIEM:                                           # diem tiem 'sau_ghi' / 'dang_ghi' cua marker (CHI fixture)
        goc_marker = H.ghi_marker

        def ghi_marker(p, artifact, vung_, them=None):
            tiem("sau_ghi", nk.hien_tai)
            tiem_dang_ghi(nk, Path(p), json.dumps({"artifact": [str(a) for a in artifact]}).encode("utf-8"))
            return goc_marker(p, artifact, vung_, them)
        H.ghi_marker = ghi_marker
    ket = {"stt": g["stt"], "buoc": g["buoc"], "exit": None, "bytecode": bc}
    try:
        ket["exit"] = int(BUOC_CON[g["buoc"]](ctx, hd, P, vung, nk, g) or 0)
    except H.LoiHopDong as e:
        print(f"[LOI HOP DONG] {e}", file=sys.stderr)
        ket.update(exit=H.EXIT_HOP_DONG, loi=str(e), nhom=getattr(e, "nhom", None), don_vi=nk.hien_tai)
    except H.ChuaXong as e:
        print(f"[CHUA XONG] {e}")
        ket.update(exit=H.EXIT_CHUA_XONG, ly_do=str(e))
    ket.update(so_don_vi_moi=nk.lam, giay=round(time.time() - T0, 3), con_lai_toi_han_mem=round(nk.han_mem - time.time(), 3))
    H.ghi_json(d / "ket_qua_con.json", ket, vung, ghi_de=True)
    nk.ghi(su_kien="ket_thuc_con", exit=ket["exit"])
    return ket["exit"]


# ---------------------------------------------------------------------------
# CHA
# ---------------------------------------------------------------------------
def quyet_dinh(kt: dict, dem: dict, buoc: str) -> tuple[int, list[str]]:
    lo = kt["phan_loai"]
    if not (kt.get("job") or {}).get("dat", True):
        return EXIT_DUNG, [f"Job Object / resume con khong dat ({kt['job']}) -- khong bao dam con chet theo cha"]
    if kt.get("thieu_log"):
        return EXIT_DUNG, [f"thieu log {kt['thieu_log']} (exit {kt.get('exit')}, {lo}) -- khong xac dinh trang thai an toan"]
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


def tong_hop_crash(P: Path) -> dict:
    ds = VA.doc_lan_chay(P)
    return {"dem": VA.dem_crash(ds), "nguong": NGUONG,
            "lan_chay": [{"stt": x["stt"], "ten": x["ten"], **({k: x["ket_thuc"].get(k) for k in
                          ("buoc", "exit", "exit_hex", "phan_loai", "don_vi", "quyet_dinh", "giay", "tiem_loi")}
                         if x["ket_thuc"] else {"dang_mo": True})} for x in ds]}


def dung(P: Path, vung, ly_do: list[str], stt) -> None:
    p = P / "van_hanh" / "dung.json"
    if not p.exists():
        H.ghi_json(p, {"ly_do": ly_do, "stt": stt, "luc": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "huong_dan": "DUNG HOI nguoi dung; khong tu xoa file nay, khong mo phien moi de reset bo dem"}, vung)


def phuc_hoi_cha_gian_doan(P: Path, vung, tiem_loi: list) -> None:
    ds = VA.doc_lan_chay(P)
    for x in ds:
        if x["ket_thuc"] is not None:
            continue
        cj = x["dir"] / "con.json"
        con = H.doc_json(cj) if cj.is_file() else None
        if any(t.get("diem") == "pid_khong_xac_dinh" for t in tiem_loi):
            tt = {"con_song": None, "tiem_loi_fixture": "pid_khong_xac_dinh"}
        else:
            tt = VA.trang_thai_con(con["pid"], con.get("thoi_diem_tao")) if con else {"con_song": None, "ly_do": "thieu con.json"}
        kq_con = x["dir"] / "ket_qua_con.json"
        kt = {"stt": x["stt"], "buoc": (x["goi"] or {}).get("buoc"), "exit": None, "exit_hex": None, "het_gio": None,
              "phan_loai": "cha_gian_doan", "don_vi": None, "ket_qua_con": H.doc_json(kq_con) if kq_con.is_file() else None,
              "con": con, "con_trang_thai": tt, "job_cha": (x["goi"] or {}).get("job_cha"),
              "file_con": {f: (x["dir"] / f).is_file() for f in ("faulthandler.log", "tien_do.jsonl")},
              "ghi_bo_sung_luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid_ghi_bo_sung": os.getpid()}
        ly = [f"cha gian doan o lan chay {x['stt']}",
              "con da ket thuc (xac nhan qua pid + thoi diem tao)" if tt.get("con_song") is False else
              f"KHONG xac nhan duoc con da ket thuc: {tt} -- khong chay tiep"]
        if con is None and (kt["job_cha"] or {}).get("dat"):
            ly.append("thieu con.json: con (neu da tao) tao TAM DUNG trong job kill-on-close cua cha -> chua chay lenh nao")
        dem = VA.dem_crash([y if y is not x else {**y, "ket_thuc": kt} for y in ds])
        kt.update(dem_crash_sau=dem, quyet_dinh={"exit_cha": EXIT_DUNG, "ly_do_dung": ly})
        H.ghi_json(x["dir"] / "ket_thuc.json", kt, vung)
        dung(P, vung, ly, x["stt"])


def mot_doan(ns, ctx, hd, P: Path, vung, snap: dict, tiem_loi: list, bc: dict) -> int:
    dl = hd["deadline"]
    han_cung = T0 + ns.ngan_sach_giay + float(dl["du_phong_cung_con"])
    if ns.han_chot is not None:
        han_cung = min(han_cung, float(ns.han_chot) - float(dl["du_phong_ket_thuc"]))
    con_lai = han_cung - time.time()
    if con_lai < float(dl["toi_thieu_khoi_chay"]):     # khong khoi chay khi thieu du phong (khong tao lan chay)
        print(f"[THIEU THOI GIAN] con {con_lai:.1f} s < {dl['toi_thieu_khoi_chay']} s -- khong tao lan chay; chay lai DUNG lenh")
        return H.EXIT_CHUA_XONG
    han_mem = min(T0 + ns.ngan_sach_giay, han_cung - float(dl["du_phong_don_vi"]))
    ds = VA.doc_lan_chay(P)
    stt = (ds[-1]["stt"] if ds else 0) + 1
    d = H.tao_thu_muc(P / "van_hanh" / "lan_chay" / f"{stt:04d}_{ns.buoc}", vung)
    cmd = cmd_con(d)
    g = {"stt": stt, "luot": LUOT, "buoc": ns.buoc, "phien": H.rel(P), "giao_thuc": str(ctx.gt_path),
         "hop_dong": str(hd["_duong"]), "phe_duyet": str(Path(os.path.abspath(ns.phe_duyet))), "ngan_sach": ns.ngan_sach_giay,
         "han_chot_cha": ns.han_chot, "han_cung": han_cung, "han_mem": han_mem, "gioi_han_don_vi": ns.gioi_han_don_vi,
         "argv_cha": bc["orig_argv"], "argv_con": cmd, "bytecode_cha": bc, "pid_cha": os.getpid(),
         "bat_dau": time.strftime("%Y-%m-%dT%H:%M:%S"), "snapshot_ma_b_sha256": snap["sha256"],
         "dau_vao": {"plan": KD.kiem_plan(hd)[0]}, "lan_thu_truoc": VA.dem_bat_dau(ds), "dem_crash_truoc": VA.dem_crash(ds),
         "tiem_loi": tiem_loi}
    if any(t.get("diem") == "job_loi" for t in tiem_loi):
        job = {"dat": False, "buoc": "tiem_loi_fixture job_loi"}
    else:
        job = VA.gan_job_cha() if VA._JOB is None else {"dat": True, "kill_on_job_close": True, "gan": "cha (da gan)"}
    g["job_cha"] = job
    H.ghi_json(d / "goi.json", g, vung)
    env = dict(os.environ)
    env.update({"PYTHONFAULTHANDLER": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"})
    if ctx.fixture:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    t0, het_gio, ma, con = time.time(), False, None, None
    with open(H.kiem_duong_ghi(d / "stdout.txt", vung), "wb") as fo, open(H.kiem_duong_ghi(d / "stderr.txt", vung), "wb") as fe:
        if job["dat"]:
            p = subprocess.Popen(cmd, cwd=str(H.ROOT), env=env, stdout=fo, stderr=fe, creationflags=0x4)  # CREATE_SUSPENDED
            tao = VA.thoi_diem_tao(int(p._handle))
            if any(t.get("diem") == "cha_sau_tao_con" for t in tiem_loi):
                print(f"[TIEM CHA] con pid {p.pid} tao {tao}", file=sys.stderr, flush=True)
                VA.crash_native()
            tj = VA.trong_job(p)
            job = {**job, "con_trong_job": tj} if tj else {**job, "dat": False, "buoc": "IsProcessInJob(con)", "con_trong_job": tj}
            con = {"pid": p.pid, "thoi_diem_tao": tao, "tao_tam_dung": True, "job": job, "argv": cmd,
                   "timeout_giay": round(han_cung - time.time(), 3)}
            H.ghi_json(d / "con.json", con, vung)
            if job["dat"]:
                tt = {"dat": False, "ntstatus": "tiem_loi_fixture"} if any(t.get("diem") == "resume_loi" for t in tiem_loi) \
                    else VA.tiep_tuc_con(p)
                if not tt["dat"]:
                    job = {**job, "dat": False, "buoc": "NtResumeProcess(con)", "tiep_tuc": tt}
            if not job["dat"]:
                p.kill()
                ma = p.wait()
            else:
                if any(t.get("diem") == "cha_khi_con_chay" for t in tiem_loi):
                    time.sleep(float(next(t for t in tiem_loi if t.get("diem") == "cha_khi_con_chay").get("giay", 2)))
                    VA.crash_native()
                try:
                    ma = p.wait(timeout=max(1.0, han_cung - time.time()))
                except subprocess.TimeoutExpired:
                    p.kill()
                    ma, het_gio = p.wait(), True
    if any(t.get("diem") == "cha_sau_con" for t in tiem_loi):
        VA.crash_native()
    pl = VA.phan_loai(ma, het_gio, d) if con else {"phan_loai": "khong_tao_con", "don_vi": None, "thieu_log": []}
    kt = {"stt": stt, "buoc": ns.buoc, "exit": ma, "exit_hex": hex(ma & 0xFFFFFFFF) if ma is not None else None,
          "het_gio": het_gio, "giay": round(time.time() - t0, 3), "con": con, "job": job, "tiem_loi": tiem_loi, **pl}
    dem = VA.dem_crash(ds + [{"ket_thuc": kt}])
    ra, ly = quyet_dinh(kt, dem, ns.buoc)
    kt.update(dem_crash_sau=dem, quyet_dinh={"exit_cha": ra, "ly_do_dung": ly})
    H.ghi_json(d / "ket_thuc.json", kt, vung)
    print(f"[VAN HANH B] lan {stt} {ns.buoc}: exit con {ma} ({kt['phan_loai']}, don vi {kt['don_vi']}) -> exit {ra}"
          + (f"; DUNG: {ly}" if ly else ""))
    if ra == EXIT_DUNG:
        dung(P, vung, ly, stt)
    return ra


def main_cha(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cong", required=True)
    ap.add_argument("--phien", required=True)
    ap.add_argument("--buoc", required=True, choices=BUOC)
    ap.add_argument("--so-doan", type=int, default=1)
    ap.add_argument("--ngan-sach-giay", type=float, default=NGAN_SACH_TOI_DA)
    ap.add_argument("--han-chot", type=float, default=None, help="epoch: han tuyet doi do loi goi ben ngoai truyen xuong")
    ap.add_argument("--gioi-han-don-vi", type=int, default=None)
    ap.add_argument("--hop-dong", default=str(KD.HOP_DONG_B))
    ap.add_argument("--giao-thuc", default=str(D.GIAO_THUC))
    ap.add_argument("--phe-duyet", default=None)
    ap.add_argument("--tiem-loi", default=None, help="CHI fixture: JSON danh sach diem tiem loi")
    ns = ap.parse_args(argv)
    # --- moi kiem duoi day TRUOC lan ghi dau tien (khong tao thu muc nao neu sai) ---
    bc, loi = kiem_bytecode()
    if loi:
        raise H.LoiHopDong(f"bytecode/interpreter: {loi}")
    if ns.cong != "B":
        raise H.LoiHopDong("van_hanh_cong_b chi nhan --cong B")
    if ns.so_doan != 1:
        raise H.LoiHopDong("moi loi goi dung MOT doan: --so-doan phai = 1")
    if not 0 < ns.ngan_sach_giay <= NGAN_SACH_TOI_DA:
        raise H.LoiHopDong(f"--ngan-sach-giay phai trong (0, {NGAN_SACH_TOI_DA:g}]")
    if ns.gioi_han_don_vi is not None and ns.gioi_han_don_vi < 1:
        raise H.LoiHopDong("--gioi-han-don-vi phai >= 1")
    hd = KD.nap_hop_dong(ns.hop_dong)
    ctx = KD.nap_ctx(ns.giao_thuc)                     # B0-F9: giao thuc / manifest hong -> LoiDauVao (exit 2), khong exit 1
    if ctx.fixture != hd["_fixture"]:
        raise H.LoiHopDong("giao thuc va hop dong khong cung che do (fixture/that)")
    P = Path(os.path.abspath(ns.phien if Path(ns.phien).is_absolute() else H.ROOT / ns.phien))
    vung = vung_b(hd, P)
    D.vung_phien(ctx, P, "B")
    try:
        tiem_loi = json.loads(ns.tiem_loi) if ns.tiem_loi else []
    except ValueError as e:
        raise H.LoiHopDong(f"--tiem-loi khong phai JSON: {e}") from None
    if not isinstance(tiem_loi, list) or not all(isinstance(t, dict) for t in tiem_loi):
        raise H.LoiHopDong("--tiem-loi phai la danh sach object")
    if tiem_loi and not hd["_fixture"]:
        raise H.LoiHopDong("tiem loi chi cho fixture -- khong bat tren B that")
    snap = KD.ma_b(hd)
    pd, loi = KD.kiem_phe_duyet(hd, ns.phe_duyet, P, snap["sha256"])
    if loi:
        raise KD.LoiDauVao("phe_duyet", loi)
    pj = P / "van_hanh" / "phien.json"
    if pj.exists() or H.la_reparse(pj):
        ph = doc_phien(pj)
        if ph.get("luot") != LUOT or ph.get("phien") != H.rel(P) or ph.get("giao_thuc") != H.rel(ctx.gt_path) \
                or ph.get("hop_dong") != H.rel(hd["_duong"]):
            raise H.LoiHopDong("phien B khong dung luot/giao thuc/hop dong cua lop van hanh nay")
        lech = [f"ma {f}" for f in sorted(set(ph["snapshot_ma_b"]["ma"]) | set(snap["ma"]))
                if ph["snapshot_ma_b"]["ma"].get(f) != snap["ma"].get(f)]
        lech += [f"cau_hinh {k}" for k in sorted(set(ph["snapshot_ma_b"]["cau_hinh"]) | set(snap["cau_hinh"]))
                 if ph["snapshot_ma_b"]["cau_hinh"].get(k) != snap["cau_hinh"].get(k)]
        if ph.get("phe_duyet_sha256") != pd["sha256"]:
            lech.append("phe duyet doi so voi luc tao phien")
        if lech:
            raise H.LoiHopDong(f"ma B / hop dong / phe duyet doi so voi luc tao phien: {lech[:6]}")
    elif P.exists() and any(P.iterdir()):
        raise H.LoiHopDong("thu muc phien da co noi dung nhung chua co van_hanh/phien.json")
    elif ns.buoc != "tien-kiem":
        raise H.LoiHopDong("phien B chua tao (chay --buoc tien-kiem truoc)")
    else:
        H.ghi_json(pj, {"luot": LUOT, "phien": H.rel(P), "fixture": hd["_fixture"], "giao_thuc": H.rel(ctx.gt_path),
                        "hop_dong": H.rel(hd["_duong"]), "phe_duyet": pd["duong"], "phe_duyet_sha256": pd["sha256"],
                        "tao_luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "snapshot_ma_b": snap, "nguong_crash": NGUONG}, vung)
    global PHIEN_CHAP_NHAN
    PHIEN_CHAP_NHAN = P
    H.chuan_bi_moi_truong(P, vung)
    import faulthandler
    fh_p = P / "van_hanh" / "cha" / f"{time.strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}.faulthandler.log"
    H.tao_thu_muc(fh_p.parent, vung)
    fh = open(H.kiem_duong_ghi(fh_p, vung), "w", encoding="utf-8")
    faulthandler.enable(file=fh, all_threads=True)
    phuc_hoi_cha_gian_doan(P, vung, tiem_loi)
    H.ghi_json(P / "van_hanh" / "so_crash.json", tong_hop_crash(P), vung, ghi_de=True)
    if (P / "van_hanh" / "dung.json").exists():
        print(f"[DUNG HOI] phien B da dung: {H.doc_json(P / 'van_hanh' / 'dung.json')['ly_do']}", file=sys.stderr)
        return EXIT_DUNG
    try:
        return mot_doan(ns, ctx, hd, P, vung, snap, tiem_loi, bc)
    finally:
        H.ghi_json(P / "van_hanh" / "so_crash.json", tong_hop_crash(P), vung, ghi_de=True)


def main(argv: list[str]) -> int:
    try:
        ma = main_cha(argv)
    except H.LoiHopDong as e:
        print(f"[LOI HOP DONG] {e}", file=sys.stderr)
        ma = H.EXIT_HOP_DONG
    if PHIEN_CHAP_NHAN is not None:
        VA.ghi_dong(PHIEN_CHAP_NHAN / "van_hanh" / "goi_cha.jsonl",
                    {"argv": list(getattr(sys, "orig_argv", [])), "exit": ma, "giay": round(time.time() - T0, 3),
                     "luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid()}, [PHIEN_CHAP_NHAN])
    return ma


def in_ma_b(argv: list[str]) -> int:
    hd_p = argv[argv.index("--hop-dong") + 1] if "--hop-dong" in argv else None
    bc, loi = kiem_bytecode()
    if loi:
        print(f"[LOI HOP DONG] {loi}", file=sys.stderr)
        return H.EXIT_HOP_DONG
    hd = KD.nap_hop_dong(hd_p)
    print(json.dumps({"ma_b_sha256": KD.ma_b(hd)["sha256"], "hop_dong_cong_b_sha256": hd["_sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--con"]:
        sys.exit(con_main(Path(sys.argv[2])))
    if sys.argv[1:2] == ["--in-ma-b"]:
        sys.exit(in_ma_b(sys.argv[2:]))
    sys.exit(main(sys.argv[1:]))
