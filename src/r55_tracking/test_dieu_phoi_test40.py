"""
test_dieu_phoi_test40.py -- Fixture cho bo dieu phoi R55 (khong suy luan that, khong GPU, khong setup_model).

Du lieu tong hop + detector GIA LAP lay tu fixture R54 (tao_fixture / MoHinhGia: 2 video x 20 anh, co frame khong GT
va anh D4). Moi cong viec di qua CLI (main) cua dieu_phoi_test40.py va ham cache/tracker/TrackEval THAT tren CPU.
Ngat gia lap (R55_NGAT, os._exit 9) chay bang tien trinh con THAT; cac ca khac goi main() trong tien trinh de
tiet kiem thoi gian nap thu vien. Moi lan chay ghi results/phase2/r55_tracking/tests/lan_<thoi diem>/.

    C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r55_tracking/test_dieu_phoi_test40.py
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR))
import dieu_phoi_test40 as D  # noqa: E402

W = D.W
GOC = D.GOC
PY = sys.executable
VUNG_TESTS = D.VUNG_R55 / "tests"
KET_QUA: list[dict] = []
V1, V2 = W.MAU_SMOKE["videos"]
DET = ("yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5")


def ca(ten: str):
    def deco(fn):
        def chay(*a, **k):
            t0 = time.time()
            try:
                ct = fn(*a, **k)
                KET_QUA.append({"ten": ten, "dat": True, "chi_tiet": ct, "giay": round(time.time() - t0, 2)})
                print(f"  [DAT ] {ten}", flush=True)
            except Exception as e:  # noqa: BLE001
                KET_QUA.append({"ten": ten, "dat": False, "loi": f"{type(e).__name__}: {e}",
                                "trace": traceback.format_exc()[-3000:], "giay": round(time.time() - t0, 2)})
                print(f"  [LOI ] {ten}: {type(e).__name__}: {str(e)[:400]}", flush=True)
        return chay
    return deco


# ---------------------------------------------------------------------------
# Tien ich
# ---------------------------------------------------------------------------
FX: dict = {}          # fixture.json, phien, khoa ...
CLI_LOG: list[dict] = []


def cli(args: list[str], fixture: bool = True, env_them: dict | None = None, timeout=600) -> subprocess.CompletedProcess:
    """Tien trinh con THAT: python src/r55_tracking/dieu_phoi_test40.py <args>."""
    env = {k: v for k, v in os.environ.items() if k not in (D.ENV_FIXTURE, D.ENV_NGAT, "BYTETRACK_BACKEND")}
    if fixture:
        env[D.ENV_FIXTURE] = str(FX["json"])
    env.update(env_them or {})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    t0 = time.time()
    r = subprocess.run([PY, str(DIR / "dieu_phoi_test40.py"), *args], cwd=GOC, env=env, capture_output=True,
                       text=True, encoding="utf-8", errors="replace", timeout=timeout)
    CLI_LOG.append({"args": args, "fixture": fixture, "env_them": env_them, "exit": r.returncode,
                    "giay": round(time.time() - t0, 1), "stdout": r.stdout[-1200:], "stderr": r.stderr[-1200:]})
    return r


def ip(args: list[str]) -> int:
    """main() trong tien trinh (cung CLI parser), che do fixture."""
    return D.main(args)


def tt(cv: str, *a) -> list[str]:
    return ["--buoc", "thuc-thi", "--cong-viec", cv, "--phien", FX["phien"], *a]


def lans(p: Path) -> list[Path]:
    return sorted(x for x in p.iterdir() if x.is_dir() and x.name.startswith("lan_")) if p.is_dir() else []


def anh_chup(p: Path) -> dict:
    return {q.relative_to(p).as_posix(): W.sha256_file(q) for q in sorted(p.rglob("*")) if q.is_file()}


def nk_cuoi(cv: str) -> dict:
    d = lans(FX["phien_dir"] / "nhat_ky" / cv.replace("-", "_"))[-1]
    return json.loads((d / "ket_qua.json").read_text(encoding="utf-8"))


def can(r, ma: int, chua: str = ""):
    out = (r.stdout or "") + (r.stderr or "")
    assert r.returncode == ma, (r.returncode, ma, out[-1500:])
    assert chua in out, (chua, out[-1500:])
    return {"exit": r.returncode, "dong": out.strip().splitlines()[-2:]}


def runs_cua(det: str) -> list[str]:
    return [r for r in D.Ctx().runs if f"-{det}-" in r]


def sua_byte(p: Path) -> bytes:
    b = p.read_bytes()
    p.write_bytes(b + b"x")
    return b


def snapshot() -> dict:
    out = {}

    def them(p: Path, de_quy: bool):
        if not p.exists():
            return
        for q in (p.rglob("*") if de_quy else p.iterdir()):
            try:
                st = q.stat()
                out[str(q)] = (st.st_size, st.st_mtime_ns if q.is_file() else 0)
            except OSError:
                pass
    them(GOC, False)
    for d in ("src", "src/r54_tracking", "src/r55_tracking", "configs/phase2/r55_tracking", "data/interim/det_cache",
              "data/processed", "data/processed/trackers", "results", "results/trackeval", "results/phase2",
              "results/phase2/r54_tracking", "results/phase2/r55_tracking", "handoff/logs/review55",
              "handoff/logs/review56", "results/phase2/r55_tracking/chuan_bi", "results/phase2/r55_tracking/tests"):
        them(GOC / d, False)
    them(GOC / D.doc_json(D.HOP_DONG, "hd")["khoa_cha_r55"]["thu_muc"], True)     # khoa R55 cha THAT (chi doc)
    for d in ("results/phase2/r54_tracking/chuan_bi", "results/phase2/r54_tracking/smoke", "data/processed/DETRAC-test"):
        them(GOC / d, False)
    for v in (V1, V2):
        them(GOC / "data/processed/DETRAC-test" / v, True)
    for d in (Path.home() / "AppData/Roaming/Ultralytics", Path.home() / ".config/Ultralytics"):
        them(d, True)
    return out


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------
def tao_fx(lan: Path):
    import test_chuan_bi_test40 as T54
    fx = lan / "fx"
    gt, ng, loai = T54.tao_fixture(fx / "nguon")
    (fx / "giao_thuc_fx.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1), encoding="utf-8")
    cfg = {"giao_thuc": "giao_thuc_fx.json", "goc_nguon": "nguon",
           "mau": {"videos": [V1, V2], "frame_dau": 1, "frame_cuoi": T54.N_FX}}
    (fx / "fixture.json").write_text(json.dumps(cfg), encoding="utf-8")
    FX.update({"dir": fx, "json": fx / "fixture.json", "nguon": ng, "gt": gt, "n": T54.N_FX})
    os.environ[D.ENV_FIXTURE] = str(fx / "fixture.json")


def ghi_test_gia(ten: str) -> str:
    """Ket qua fixture GIA (trong vung fixture) de khoa fixture co the doi chieu sha256 ma hien tai."""
    d = FX["dir"] / "tests" / ten
    d.mkdir(parents=True)
    W.ghi_json(d / "ket_qua_test.json", {"tat_ca_dat": True, "exit": 0, "so_test": 0, "gia_lap": True,
                                         "dieu_phoi_sha256": W.sha256_file(DIR / "dieu_phoi_test40.py"),
                                         "test_sha256": W.sha256_file(Path(__file__)),
                                         "hop_dong_sha256": W.sha256_file(D.HOP_DONG)}, [d])
    return ten


# ---------------------------------------------------------------------------
# Cac ca
# ---------------------------------------------------------------------------
@ca("A1 import khong chay gi; script khong tham so / buoc la -> exit 2, khong ghi")
def ca_import(lan: Path):
    truoc = sorted(p.name for p in D.VUNG_R55.iterdir())
    env = {k: v for k, v in os.environ.items() if k not in (D.ENV_FIXTURE, D.ENV_NGAT)}
    r = subprocess.run([PY, "-c", f"import sys; sys.path.insert(0, {str(DIR)!r}); import dieu_phoi_test40 as D; "
                        "print('ok', D.CONG_VIEC)"], cwd=GOC, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-800:]
    a = can(cli([], fixture=False), 2, "--buoc")
    b = can(cli(["--buoc", "full"], fixture=False), 2, "invalid choice")
    assert sorted(p.name for p in D.VUNG_R55.iterdir()) == truoc
    return {"import": r.stdout.strip()[:200], "khong_tham_so": a, "buoc_la": b}


@ca("A2 duong that: thieu --duyet / --cong-viec / --phien / ten khoa / R55_NGAT / env fixture ngoai tests -> exit 2, khong ghi")
def ca_cli_loi_that(lan: Path):
    truoc = sorted(p.name for p in D.VUNG_R55.iterdir())
    out = {}
    for ten, args, env, chua in (
            ("khong_duyet", ["--buoc", "thuc-thi", "--cong-viec", "tracking", "--phien", "x", "--run", "r"], None, "--duyet"),
            ("khong_cong_viec", ["--buoc", "thuc-thi", "--duyet", "x"], None, "--cong-viec"),
            ("thieu_phien", ["--buoc", "thuc-thi", "--cong-viec", "cham", "--duyet", "x", "--run", "r54-yolov8n_coco-c25-cv"],
             None, "phien"),
            ("khoa_khong_ten", ["--buoc", "khoa"], None, "TUONG MINH"),
            ("kiem_khoa_khong_ten", ["--buoc", "kiem-khoa"], None, "--khoa"),
            ("cong_viec_ngoai_thuc_thi", ["--buoc", "tien-kiem", "--run", "x"], None, "chi dung voi"),
            ("ngat_duong_that", ["--buoc", "kiem-khoa", "--khoa", "x"], {D.ENV_NGAT: "cache-truoc-marker:1"}, "chi dung trong fixture"),
            ("env_ngoai_tests", ["--buoc", "kiem-khoa", "--khoa", "x"],
             {D.ENV_FIXTURE: str(D.VUNG_R55 / "fixture.json")}, "phai la")):
        out[ten] = can(cli(args, fixture=False, env_them=env), 2, chua)
    assert sorted(p.name for p in D.VUNG_R55.iterdir()) == truoc, "CLI loi da ghi vao vung R55"
    return out


@ca("B1 CLI chuan bi fixture: tien-kiem -> khoa (ten tuong minh) -> kiem-khoa (chi doc / --luu)")
def ca_chuoi_chuan_bi(lan: Path):
    cb = FX["dir"] / "chuan_bi"
    r1 = can(cli(["--buoc", "tien-kiem"]), 0, "dat")
    tk = sorted(cb.glob("tien_kiem_*"))[-1]
    tkj = json.loads((tk / "tien_kiem.json").read_text(encoding="utf-8"))
    assert tkj["khoa_r54"]["dat"] and tkj["khoa_r54"]["khoa_sha256"].startswith("e2d1b0d5")
    assert any("KHONG bam lai toan bo" in g for g in tkj["gioi_han"])
    ten_t = ghi_test_gia("lan_gia_dat")
    r0 = can(cli(["--buoc", "khoa", "--tien-kiem", tk.name]), 2, "TUONG MINH")
    r2 = can(cli(["--buoc", "khoa", "--tien-kiem", tk.name, "--tests", ten_t]), 0, "khoa.json")
    k = sorted(cb.glob("khoa_*"))[-1]
    kj = json.loads((k / "khoa.json").read_text(encoding="utf-8"))
    rels = {f["rel"] for f in kj["file"]}
    for need in ("src/r55_tracking/dieu_phoi_test40.py", "configs/phase2/r55_tracking/hop_dong_dieu_phoi.json",
                 "handoff/logs/review55/01_thiet_ke_dieu_phoi.md", "src/r54_tracking/chuan_bi_test40.py",
                 "configs/phase2/r54_tracking/giao_thuc_test40.json"):
        assert need in rels, need
    assert any(r.replace("\\", "/").endswith("ultralytics/utils/events.py") for r in rels)   # ke thua khoa R54
    assert kj["khoa_r54"]["khoa_json_sha256"] == D.doc_json(D.HOP_DONG, "hd")["khoa_r54"]["khoa_json_sha256"]
    truoc = sorted(p.name for p in cb.iterdir())
    r3 = can(cli(["--buoc", "kiem-khoa", "--khoa", k.name]), 0, "0 lech")
    assert sorted(p.name for p in cb.iterdir()) == truoc, "kiem-khoa mac dinh da ghi"
    r4 = can(cli(["--buoc", "kiem-khoa", "--khoa", k.name, "--luu", "chuan_bi"]), 0, "ket_qua.json")
    FX.update({"khoa": k.name, "khoa_dir": k, "tk": tk.name})
    return {"tien_kiem": r1, "khoa_thieu_tests": r0, "khoa": r2, "so_file_khoa": kj["so_file"], "kiem_khoa": r3,
            "kiem_khoa_luu": r4}


@ca("C1 mo-phien; trung ten -> exit 1 (tu choi ghi de); ten thoat vung -> exit 2; tracking khi chua co cache -> exit 2")
def ca_mo_phien(lan: Path):
    r1 = can(cli(["--buoc", "thuc-thi", "--cong-viec", "mo-phien", "--khoa", FX["khoa"], "--phien", "p1"]), 0, "mo-phien")
    FX.update({"phien": "p1", "phien_dir": FX["dir"] / "thuc_thi" / "p1"})
    pj0 = W.sha256_file(FX["phien_dir"] / "phien.json")
    r2 = can(cli(["--buoc", "thuc-thi", "--cong-viec", "mo-phien", "--khoa", FX["khoa"], "--phien", "p1"]), 1, "ghi de")
    assert W.sha256_file(FX["phien_dir"] / "phien.json") == pj0
    r3 = can(cli(["--buoc", "thuc-thi", "--cong-viec", "mo-phien", "--khoa", FX["khoa"], "--phien", "..\\thoat"]), 2,
             "khong hop le")
    r4 = can(cli(["--buoc", "thuc-thi", "--cong-viec", "tracking", "--phien", "..", "--run", "r54-yolov8n_coco-c25-cv"]),
             2, "khong hop le")
    assert ip(tt("tracking", "--run", "r54-yolov8n_coco-c25-cv")) == 2
    assert "chua co manifest" in nk_cuoi("tracking")["loi"]
    assert not (FX["phien_dir"] / "tracking").exists()
    return {"mo_phien": r1, "trung_ten": r2, "ten_thoat": r3, "phien_thoat": r4}


@ca("C2 cache bi ngat SAU marker video 1 -> thieu 1 video: xac-nhan tu choi, tracking tu choi (truoc tracking)")
def ca_cache_thieu_video(lan: Path):
    det = "yolov8n_coco"
    r = cli(tt("cache", "--detector", det), env_them={D.ENV_NGAT: "cache-sau-marker:1"})
    assert r.returncode == 9, r.stdout[-800:] + r.stderr[-800:]
    c = FX["phien_dir"] / "cache" / det
    assert len(lans(c / V1)) == 1 and (lans(c / V1)[0] / D.MARKER).is_file() and not (c / V2).exists()
    assert ip(tt("xac-nhan-cache", "--detector", det)) == 2
    assert "thieu cache hoan tat cho 1/2" in nk_cuoi("xac-nhan-cache")["loi"]
    assert not (FX["phien_dir"] / "xac_nhan_cache").exists()
    assert ip(tt("tracking", "--run", "r54-yolov8n_coco-c25-cv")) == 2
    assert not (FX["phien_dir"] / "tracking").exists()
    return {"ngat": r.returncode, "xac_nhan": nk_cuoi("xac-nhan-cache")["loi"][:200]}


@ca("C3 cache ngat TRUOC marker roi ngat GIUA luc ghi marker (cat) o video 2 -> hai lan giu nguyen; tiep tuc: tai su "
    "dung video 1, lan moi video 2")
def ca_cache_tiep_tuc(lan: Path):
    det = "yolov8n_coco"
    c = FX["phien_dir"] / "cache" / det
    r = cli(tt("cache", "--detector", det), env_them={D.ENV_NGAT: "cache-truoc-marker:1"})
    assert r.returncode == 9, r.stdout[-800:] + r.stderr[-800:]
    dd = lans(c / V2)
    assert len(dd) == 1 and not (dd[0] / D.MARKER).is_file() and (dd[0] / f"{V2}.npz").is_file()
    chup = anh_chup(dd[0])
    r1 = cli(tt("cache", "--detector", det), env_them={D.ENV_NGAT: "cache-marker-cat:1"})
    assert r1.returncode == 9 and "marker cat" in r1.stdout, r1.stdout[-800:] + r1.stderr[-800:]
    d1 = lans(c / V2)[1]
    assert (d1 / D.MARKER).is_file() and D.doc_marker(d1) is None and D.danh_gia_lan(d1, {})[0] == "hong"
    chup1 = anh_chup(d1)
    r2 = can(cli(tt("cache", "--detector", det)), 0, "moi 1, tai su dung 1")
    nk = nk_cuoi("cache")
    tts = {x["trang_thai"] for x in nk["lan_thu"]}
    assert nk["goi_detector"] and len(nk["moi"]) == 1 and {"do_dang", "hong"} <= tts, nk["lan_thu"]
    assert len(lans(c / V2)) == 3 and anh_chup(dd[0]) == chup and not (dd[0] / D.MARKER).is_file()
    assert anh_chup(d1) == chup1
    assert ip(tt("cache", "--detector", det)) == 0           # lan nua: khong goi detector, khong lan moi
    nk = nk_cuoi("cache")
    assert not nk["goi_detector"] and not nk["moi"] and len(nk["tai_su_dung"]) == 2
    FX["do_dang_cache"] = (dd[0], chup)
    FX["hong_cache"] = (d1, chup1)
    return {"tiep_tuc": r2, "lan_v2": [p.name for p in lans(c / V2)], "marker_cat": r1.stdout.strip()[-160:]}


def sua_npf(p_npz: Path, kieu: str):
    """Ghi lai npz voi n_per_frame sai nhung GIU tong (kieu 'am': frame 1 = -1, bu vao frame 2; 'thieu': gop 2 frame
    cuoi -> shape n-1)."""
    import numpy as np
    a = dict(np.load(p_npz))
    f = a["n_per_frame"].astype(np.int64)
    if kieu == "am":
        f[1] += f[0] + 1
        f[0] = -1
    else:
        f = np.concatenate([f[:-2], [f[-2] + f[-1]]])
    a["n_per_frame"] = f.astype(np.int32)
    assert int(a["n_per_frame"].sum()) == len(a["data"])
    np.savez_compressed(p_npz, **a)


@ca("C4 xac-nhan-cache: ngat truoc SHA256SUMS / truoc marker / marker cat -> chua cong bo (tracking chan), lan dang giu "
    "nguyen, xac nhan lai vao thu muc moi KHONG goi detector; manifest BAT BIEN (lan 2 -> exit 1); cache sau manifest "
    "-> exit 2; n_per_frame sai giu tong -> chan truoc tracking")
def ca_xac_nhan(lan: Path):
    det = "yolov8n_coco"
    xn = FX["phien_dir"] / "xac_nhan_cache" / det
    run = "r54-yolov8n_coco-c25-cv"
    ngat_kq, chup_xn = {}, {}
    for diem in ("xac-nhan-truoc-sums", "xac-nhan-truoc-marker", "xac-nhan-marker-cat"):
        r = cli(tt("xac-nhan-cache", "--detector", det), env_them={D.ENV_NGAT: f"{diem}:1"})
        assert r.returncode == 9, (diem, r.stdout[-800:] + r.stderr[-800:])
        d = lans(xn)[-1]
        assert (d / "manifest_cache.json").is_file() and D.danh_gia_lan(d, {})[0] in ("do_dang", "hong"), diem
        assert ip(tt("tracking", "--run", run)) == 2 and "chua co manifest" in nk_cuoi("tracking")["loi"], diem
        assert not (FX["phien_dir"] / "tracking").exists()
        chup_xn[d] = anh_chup(d)
        ngat_kq[diem] = {"exit": r.returncode, "file": sorted(chup_xn[d])}
    assert not (lans(xn)[0] / "SHA256SUMS").is_file() and not (lans(xn)[1] / D.MARKER).is_file()
    assert ip(tt("xac-nhan-cache", "--detector", det)) == 0
    nk = nk_cuoi("xac-nhan-cache")
    assert not nk["goi_detector"] and len(nk["moi"]) == 1
    tts = [x["trang_thai"] for x in nk["lan_thu"] if "/xac_nhan_cache/" in x["lan"]]
    assert sorted(tts) == ["do_dang", "do_dang", "hong"], nk["lan_thu"]
    assert any(x["trang_thai"] == "hong" and "/cache/" in x["lan"] for x in nk["lan_thu"])   # marker cache cat: bo qua
    m = lans(xn)
    assert len(m) == 4 and all(anh_chup(d) == c for d, c in chup_xn.items())
    dm, man, loi = D.manifest_cache(D.Phien(D.Ctx(), FX["phien"]), det)
    assert dm == m[3] and not loi
    assert man["so_video"] == 2 and sorted(man["video"]) == [V1, V2]
    assert man["video"][V2]["lan"].endswith(lans(FX["phien_dir"] / "cache" / det / V2)[2].name)
    assert ip(tt("xac-nhan-cache", "--detector", det)) == 1
    chup = anh_chup(FX["phien_dir"] / "cache")
    assert ip(tt("cache", "--detector", det)) == 2
    nk = nk_cuoi("cache")
    assert "BAT BIEN" in nk["loi"] and not nk["goi_detector"] and anh_chup(FX["phien_dir"] / "cache") == chup
    FX["xn_do_dang"] = chup_xn
    assert ip(tt("cache", "--detector", DET[1])) == 0 and ip(tt("xac-nhan-cache", "--detector", DET[1])) == 0
    # F3: n_per_frame sai (am / thieu phan tu) nhung GIU tong, marker ghi lai khop hash -> xac nhan chan, tracking chan
    d3 = DET[2]
    assert ip(tt("cache", "--detector", d3)) == 0
    c1 = lans(FX["phien_dir"] / "cache" / d3 / V1)[0]
    p_npz, p_mk = c1 / f"{V1}.npz", c1 / D.MARKER
    goc_npz, goc_mk = p_npz.read_bytes(), p_mk.read_bytes()
    npf = {}
    for kieu in ("am", "thieu"):
        try:
            sua_npf(p_npz, kieu)
            mk = json.loads(goc_mk.decode("utf-8"))
            mk["file"][f"{V1}.npz"] = W.sha256_file(p_npz)
            p_mk.write_text(json.dumps(mk), encoding="utf-8")
            ma = ip(tt("xac-nhan-cache", "--detector", d3))
            loi = nk_cuoi("xac-nhan-cache")["loi"]
            ma_t = ip(tt("tracking", "--run", *runs_cua(d3)))
            loi_t = nk_cuoi("tracking")["loi"]
        finally:
            p_npz.write_bytes(goc_npz)
            p_mk.write_bytes(goc_mk)
        assert ma == 2 and "n_per_frame" in loi, (kieu, ma, loi)
        assert ma_t == 2 and "chua co manifest" in loi_t, (kieu, ma_t, loi_t)
        assert not (FX["phien_dir"] / "xac_nhan_cache" / d3).exists() and not (FX["phien_dir"] / "tracking").exists()
        npf[kieu] = loi[:220]
    assert ip(tt("xac-nhan-cache", "--detector", d3)) == 0
    return {"ngat_xac_nhan": ngat_kq, "manifest": man["video"][V1], "spec": man["spec_sha256"], "n_per_frame": npf}


@ca("C5 kiem_npz bat schema / timeline / anh xa / unreadable sai (ham xac nhan, ban sao trong vung test)")
def ca_kiem_npz(lan: Path):
    import shutil
    import numpy as np
    ctx = D.Ctx()
    ph = D.Phien(ctx, FX["phien"])
    ph.xac_minh()
    man = D.kiem_cache_theo_manifest(ph, "yolov8n_mixed5")
    goc = W.tuyet_doi(man["video"][V1]["lan"])
    mk = json.loads((goc / D.MARKER).read_text(encoding="utf-8"))
    rb = mk["rang_buoc"]
    assert D.kiem_npz(ctx, "yolov8n_mixed5", goc, V1, FX["n"], rb) == []
    out = {}
    for ten, sua in (("time_s", lambda a: a.update(time_s=a["time_s"] + 1e-9)),
                     ("frame_idx", lambda a: a.update(frame_idx=a["frame_idx"][::-1].copy())),
                     ("unreadable", lambda a: a.update(unreadable=np.ones_like(a["unreadable"]))),
                     ("cls", lambda a: a.update(cls_goc=(a["cls_goc"] * 0 + 1).astype(np.int16))),
                     ("thieu_khoa", lambda a: a.pop("cls_goc")), ("npf_am", None), ("npf_thieu", None)):
        d = lan / "kiem_npz" / ten
        shutil.copytree(goc, d)
        if sua is None:                                   # F3: n_per_frame sai nhung giu tong
            sua_npf(d / f"{V1}.npz", ten[4:])
        else:
            a = dict(np.load(d / f"{V1}.npz"))
            sua(a)
            np.savez_compressed(d / f"{V1}.npz", **a)
        loi = D.kiem_npz(ctx, "yolov8n_mixed5", d, V1, FX["n"], rb)
        assert loi, ten
        if sua is None:
            assert any("n_per_frame" in x for x in loi), (ten, loi)
        out[ten] = loi[:2]
    return out


@ca("D1 cache / meta.json sai hash sau xac nhan -> tracking bi chan TRUOC tracking, khong lan tracking nao")
def ca_cache_sai_hash(lan: Path):
    det = "yolov8n_mixed5"
    man = json.loads(next((FX["phien_dir"] / "xac_nhan_cache" / det).iterdir()).joinpath("manifest_cache.json")
                     .read_text(encoding="utf-8"))
    out = {}
    for ten, p in (("npz", W.tuyet_doi(man["video"][V2]["lan"]) / f"{V2}.npz"),
                   ("meta", W.tuyet_doi(man["video"][V1]["lan"]) / "meta.json"),
                   ("json", W.tuyet_doi(man["video"][V1]["lan"]) / f"{V1}.json")):
        b = sua_byte(p)
        try:
            ma = ip(tt("tracking", "--run", *runs_cua(det)))
            loi = nk_cuoi("tracking")["loi"]
        finally:
            p.write_bytes(b)
        assert ma == 2 and "khong khop manifest" in loi, (ma, loi)
        assert not (FX["phien_dir"] / "tracking").exists()
        out[ten] = loi[:200]
    # F1: manifest DA CONG BO (marker doc duoc) nhung manifest_cache.json sai hash -> tracking chan, xac nhan lai chan
    xn = FX["phien_dir"] / "xac_nhan_cache" / det
    p = next(xn.iterdir()) / "manifest_cache.json"
    truoc = sorted(x.name for x in xn.iterdir())
    b = sua_byte(p)
    try:
        ma = ip(tt("tracking", "--run", *runs_cua(det)))
        loi = nk_cuoi("tracking")["loi"]
        ma_x = ip(tt("xac-nhan-cache", "--detector", det))
        loi_x = nk_cuoi("xac-nhan-cache")["loi"]
    finally:
        p.write_bytes(b)
    assert ma == 2 and "chua co manifest cache hop le" in loi and "lech" in loi, (ma, loi)
    assert ma_x == 2 and "da cong bo nhung lech" in loi_x, (ma_x, loi_x)
    assert sorted(x.name for x in xn.iterdir()) == truoc and not (FX["phien_dir"] / "tracking").exists()
    out["manifest_cong_bo_sai_hash"] = {"tracking": loi[:200], "xac_nhan": loi_x[:200]}
    return out


@ca("E1 tracking ngat TRUOC marker -> cham tu choi (chua du video); tiep tuc tai su dung video 1, lan dang giu nguyen")
def ca_tracking_ngat(lan: Path):
    run = "r54-yolov8n_coco-c25-cv"
    r = cli(tt("tracking", "--run", run), env_them={D.ENV_NGAT: "tracking-truoc-marker:2"})
    assert r.returncode == 9, r.stdout[-800:] + r.stderr[-800:]
    t = FX["phien_dir"] / "tracking" / run
    assert (lans(t / V1)[0] / D.MARKER).is_file() and not (lans(t / V2)[0] / D.MARKER).is_file()
    assert (lans(t / V2)[0] / "data" / f"{V2}.txt").is_file()                   # dau ra do dang co ton tai
    chup = anh_chup(lans(t / V2)[0])
    assert ip(tt("cham", "--run", run)) == 2 and "chua du 2 video" in nk_cuoi("cham")["loi"]
    assert not (FX["phien_dir"] / "cham").exists()
    assert ip(tt("tracking", "--run", run)) == 0
    nk = nk_cuoi("tracking")
    assert len(nk["moi"]) == 1 and len(nk["tai_su_dung"]) == 1
    assert len(lans(t / V2)) == 2 and anh_chup(lans(t / V2)[0]) == chup
    FX["do_dang_trk"] = (lans(t / V2)[0], chup)
    return {"ngat": 9, "tiep_tuc": {"moi": nk["moi"], "tai_su_dung": nk["tai_su_dung"]}}


@ca("E2 tracking ngat SAU marker -> lan co marker duoc tai su dung (khong chay lai)")
def ca_tracking_sau_marker(lan: Path):
    run = "r54-yolov8n_coco-c25-ekf"
    r = cli(tt("tracking", "--run", run), env_them={D.ENV_NGAT: "tracking-sau-marker:1"})
    assert r.returncode == 9, r.stdout[-800:] + r.stderr[-800:]
    assert ip(tt("tracking", "--run", run)) == 0
    nk = nk_cuoi("tracking")
    t = FX["phien_dir"] / "tracking" / run
    assert len(nk["tai_su_dung"]) == 1 and len(nk["moi"]) == 1 and len(lans(t / V1)) == 1
    return {"tai_su_dung": nk["tai_su_dung"]}


@ca("E3 doi backend (BYTETRACK_BACKEND=np) -> exit 2 truoc tracking; doi khoa (phien.json / file da khoa) -> exit 2")
def ca_doi_backend_khoa(lan: Path):
    run = "r54-yolov8n_coco-c25-ukf"
    t = FX["phien_dir"] / "tracking" / run
    a = can(cli(tt("tracking", "--run", run), env_them={"BYTETRACK_BACKEND": "np"}), 2, "BYTETRACK_BACKEND")
    assert not t.exists()
    # phien thu 2 voi khoa_sha256 bi doi
    p2 = FX["dir"] / "thuc_thi" / "p_doi_khoa"
    p2.mkdir()
    pj = json.loads((FX["phien_dir"] / "phien.json").read_text(encoding="utf-8"))
    (p2 / "phien.json").write_text(json.dumps(pj | {"khoa_sha256": "0" * 64}), encoding="utf-8")
    assert ip(["--buoc", "thuc-thi", "--cong-viec", "tracking", "--phien", "p_doi_khoa", "--run", run]) == 2
    loi_p2 = json.loads((lans(p2 / "nhat_ky" / "tracking")[-1] / "ket_qua.json").read_text(encoding="utf-8"))["loi"]
    assert "khac khoa da gan" in loi_p2 and not (p2 / "tracking").exists()
    # file da khoa (D4 fixture) bi sua -> kiem-khoa + cong viec deu chan
    p = FX["nguon"].loai_d4
    b = sua_byte(p)
    try:
        k = can(cli(["--buoc", "kiem-khoa", "--khoa", FX["khoa"]]), 2, "lech")
        assert ip(tt("tracking", "--run", run)) == 2
        loi = nk_cuoi("tracking")["loi"]
    finally:
        p.write_bytes(b)
    assert "kiem-khoa" in loi and not t.exists()
    return {"backend": a, "phien_doi_khoa": loi_p2[:200], "file_khoa_doi": k}


@ca("E4 doi cau hinh / seed trong rang buoc -> lan cu KHONG duoc tai su dung, giu nguyen; lan moi duoc tao")
def ca_doi_cau_hinh(lan: Path):
    run = "r54-yolov8n_coco-c25-pf500-s1"
    assert ip(tt("tracking", "--run", run)) == 0
    t = FX["phien_dir"] / "tracking" / run / V1
    d0 = lans(t)[0]
    mk = json.loads((d0 / D.MARKER).read_text(encoding="utf-8"))
    rb = mk["rang_buoc"]
    out = {k: D.danh_gia_lan(d0, rb | {k: v})[:2] for k, v in (("seed", 2), ("cfg_sha256", "0" * 64),
                                                                ("backend", "np"), ("khoa_sha256", "1" * 64),
                                                                ("npz_sha256", "2" * 64))}
    assert all(x[0] == "khong_khop" for x in out.values()), out
    assert D.danh_gia_lan(d0, rb)[0] == "hop_le"
    # mo phong lan cu sinh voi cau hinh khac: sua rang buoc trong marker cua lan cu
    (d0 / D.MARKER).write_text(json.dumps(mk | {"rang_buoc": rb | {"cfg_sha256": "f" * 64}}), encoding="utf-8")
    chup = anh_chup(d0)
    assert ip(tt("tracking", "--run", run)) == 0
    nk = nk_cuoi("tracking")
    assert len(nk["moi"]) == 1 and any(x["trang_thai"] == "khong_khop" for x in nk["lan_thu"])
    assert anh_chup(d0) == chup and len(lans(t)) == 2
    a = (d0 / "data" / f"{V1}.txt").read_bytes()
    b = (lans(t)[1] / "data" / f"{V1}.txt").read_bytes()
    assert a == b and len(a) > 0                 # cung cache + seed -> MOT trung byte
    return {"danh_gia": out, "mot_trung": True}


@ca("F1 tracking 36 run tu cache da xac nhan; frame khong GT / D4 van qua tracker; PF seed/N/K1/K2")
def ca_tracking_du(lan: Path):
    ctx = D.Ctx()
    assert ip(tt("tracking", "--run", *ctx.runs)) == 0
    for run in ctx.runs:
        for v in (V1, V2):
            ds = [d for d in lans(FX["phien_dir"] / "tracking" / run / v) if (d / D.MARKER).is_file()]
            assert ds, (run, v)
    t = FX["phien_dir"] / "tracking"
    d = [x for x in lans(t / "r54-yolo11n_mixed5-c10-pf500-s2" / V2) if (x / D.MARKER).is_file()][0]
    info = json.loads((d / D.MARKER).read_text(encoding="utf-8"))["info"]
    assert (info["seed_thuc_dung"], info["num_particles"], info["k1_optimal_proposal"], info["k2_roughening"],
            info["n_buoc_timing"], info["n_steps"]) == (2, 500, False, 0.0, FX["n"], [1]), info
    import pandas as pd
    mot = pd.read_csv(d / "data" / f"{V2}.txt", header=None)
    assert set(mot[0]) >= {18, 19, 20}                                          # anh D4 (khong GT) van co track
    return {"so_run": len(ctx.runs), "info_pf": info}


@ca("F2 cham loi (gia lap TrackEval) -> exit 4, lan loi giu nguyen; cham lai -> lan moi; cham 36 run")
def ca_cham(lan: Path):
    run = "r54-yolov8n_coco-c25-cv"
    r = cli(tt("cham", "--run", run), env_them={D.ENV_NGAT: "cham-loi:1"})
    assert r.returncode == 4, r.stdout[-800:] + r.stderr[-800:]
    c = FX["phien_dir"] / "cham" / run
    d0 = lans(c)[0]
    assert (d0 / "LOI.json").is_file() and not (d0 / D.MARKER).is_file()
    chup = anh_chup(d0)
    r1 = cli(tt("cham", "--run", run), env_them={D.ENV_NGAT: "cham-marker-cat:1"})     # F2: marker cham bi cat
    assert r1.returncode == 9 and "marker cat" in r1.stdout, r1.stdout[-800:] + r1.stderr[-800:]
    d1 = lans(c)[1]
    assert (d1 / D.MARKER).is_file() and D.doc_marker(d1) is None
    chup1 = anh_chup(d1)
    ctx = D.Ctx()
    assert ip(tt("cham", "--run", *ctx.runs)) == 0
    assert any(x["trang_thai"] == "hong" and x["lan"].endswith(d1.name) for x in nk_cuoi("cham")["lan_thu"])
    assert anh_chup(d0) == chup and anh_chup(d1) == chup1 and len(lans(c)) == 3
    import pandas as pd
    import math
    hota = {}
    for run_ in ctx.runs:
        d = [x for x in lans(FX["phien_dir"] / "cham" / run_) if D.doc_marker(x) is not None]
        assert len(d) == 1, run_
        mk = D.doc_marker(d[0])
        assert all(math.isfinite(float(v)) for k, v in mk["combined"].items() if k != "video")
        hota[run_] = mk["combined"]["HOTA"]
    pv = pd.read_csv([x for x in lans(c) if D.doc_marker(x) is not None][0] / "trackeval" / run / "per_video_metrics.csv")
    assert sorted(pv["video"]) == [V1, V2] and pv["HOTA"].between(0, 1).all()
    assert ip(tt("cham", "--run", run)) == 0 and nk_cuoi("cham")["tai_su_dung"]       # khong cham lai khi da xong
    FX["hong_cham"] = (d1, chup1)
    FX["cham_hop_le"] = lans(c)[2]
    return {"loi_gia_lap": r.returncode, "marker_cat": r1.returncode, "so_run": len(hota), "hota_vi_du": hota[run]}


@ca("G1 lap tracker tu CUNG cache/seed (cv, pf500 s0/s1) -> MOT trung byte, khong goi detector")
def ca_lap_tracker(lan: Path):
    out = {}
    for run in ("r54-yolov8n_mixed5-c25-cv", "r54-yolov8n_mixed5-c25-pf500-s0", "r54-yolo11n_mixed5-c10-pf500-s1"):
        assert ip(tt("lap-tracker", "--run", run, "--video", V1)) == 0
        kq = nk_cuoi("lap-tracker")
        assert kq["ket_qua"]["trung_byte"] and not kq["goi_detector"]
        out[run] = kq["ket_qua"]["sha256_lap"][:16]
    return out


@ca("G2 cham lai tu CUNG bo MOT da bam -> 3 file metric trung byte; MOT ban sao bi sua -> chan")
def ca_cham_lai(lan: Path):
    out = {}
    for run in ("r54-yolov8n_coco-c25-cv", "r54-yolo11n_mixed5-c10-ukf", "r54-yolov8n_mixed5-c25-pf500-s2"):
        assert ip(tt("cham-lai", "--run", run)) == 0
        kq = nk_cuoi("cham-lai")["ket_qua"]
        assert kq["file_trung_byte"] and kq["combined_trung"] and kq["lan_goc_con_nguyen"] and kq["kiem_module_sau"] == []
        if run == "r54-yolov8n_coco-c25-cv":                 # F2: bo qua lan marker cat dung TRUOC lan hop le
            assert kq["lan_cham_goc"].endswith(FX["cham_hop_le"].name), kq["lan_cham_goc"]
            assert D.doc_marker(FX["hong_cham"][0]) is None
        out[run] = kq["file_lai"]["combined_metrics.csv"][:16]
    run = "r54-yolov8n_coco-c10-cv"
    d = [x for x in lans(FX["phien_dir"] / "cham" / run) if (x / D.MARKER).is_file()][0]
    p = d / "trackers" / FX["nguon"].thu_muc_gt.name / run / "data" / f"{V1}.txt"
    b = sua_byte(p)
    try:
        ma = ip(tt("cham-lai", "--run", run))
        loi = nk_cuoi("cham-lai")["loi"]
    finally:
        p.write_bytes(b)
    assert ma == 2 and "chua co lan cham hoan tat" in loi, (ma, loi)
    out["mot_sua"] = loi[:160]
    return out


@ca("G3 hau-kiem: du 36 run; lan do dang / loi duoc liet ke va giu nguyen byte")
def ca_hau_kiem(lan: Path):
    assert ip(tt("hau-kiem")) == 0
    kq = nk_cuoi("hau-kiem")["ket_qua"]
    assert kq["du_36_run"] and kq["dem"] == {"cache": 6, "tracking": 72, "cham": 36}, kq["dem"]
    assert kq["loi"] == [], kq["loi"]                                      # gom kiem module sau (F4)
    for k in ("do_dang_cache", "do_dang_trk"):
        d, chup = FX[k]
        assert anh_chup(d) == chup and W.rel_goc(d) in kq["lan_khong_marker_giu_nguyen"], k
    for k in ("hong_cache", "hong_cham"):                                  # F2: marker cat -> liet ke, giu nguyen
        d, chup = FX[k]
        assert anh_chup(d) == chup and W.rel_goc(d) in kq["lan_marker_hong_giu_nguyen"], k
    for d, chup in FX["xn_do_dang"].items():                               # F1: xac nhan chua cong bo
        assert anh_chup(d) == chup and (W.rel_goc(d) in kq["lan_khong_marker_giu_nguyen"] or
                                        W.rel_goc(d) in kq["lan_marker_hong_giu_nguyen"]), d
    return {"dem": kq["dem"], "lan_khong_marker": len(kq["lan_khong_marker_giu_nguyen"]),
            "lan_marker_hong": kq["lan_marker_hong_giu_nguyen"]}


@ca("G4 kiem module SAU cong viec (F4, muc ham): khop khoa sau 36 run; ban khoa trong bo nho voi sha256 module sai -> "
    "phat hien 'sha256 khac khoa' (khong tiem module gia - P6fb1cad7 tu choi ca do)")
def ca_kiem_module_sau(lan: Path):
    import copy
    ph = D.Phien(D.Ctx(), FX["phien"])
    ph.xac_minh()
    assert D.file_module_dang_nap() == W.file_module_da_nap()       # ban nho resolve = ham R54 (cung tap file)
    assert D.kiem_module_sau(ph.khoa) == []
    k2 = copy.deepcopy(ph.khoa)
    dich = W.rel_goc(Path(sys.modules["det_cache"].__file__))
    hit = [f for f in k2["file"] if f["rel"] == dich]
    assert len(hit) == 1, dich
    hit[0]["sha256"] = "0" * 64
    lech = D.kiem_module_sau(k2)
    assert lech == [f"{dich}: sha256 khac khoa"], lech
    return {"so_file_module": len(D.file_module_dang_nap()), "khop": [], "sha_sai": lech}


@ca("H1 duong dan thoat vung / fixture.json sai / phien khong ton tai -> LoiDauVao, khong ghi")
def ca_duong_dan(lan: Path):
    ctx = D.Ctx()
    out = {}
    try:
        D.lan_moi(GOC / "results" / "thoat", [FX["phien_dir"]])
        raise AssertionError("lan_moi ngoai vung khong loi")
    except W.LoiDauVao as e:
        out["lan_moi"] = str(e)[:120]
    for ten in ("..", "a/b", "khong_co"):
        try:
            D.Phien(ctx, ten)
            raise AssertionError(ten)
        except W.LoiDauVao as e:
            out[ten] = str(e)[:100]
    bad = FX["dir"] / "bad"
    bad.mkdir()
    (bad / "fixture.json").write_text(json.dumps({"giao_thuc": "../../../../x.json", "goc_nguon": "nguon",
                                                  "mau": {"videos": [V1], "frame_dau": 1, "frame_cuoi": 2}}))
    goc = os.environ[D.ENV_FIXTURE]
    os.environ[D.ENV_FIXTURE] = str(bad / "fixture.json")
    try:
        try:
            D.Ctx()
            raise AssertionError("giao thuc ngoai vung khong loi")
        except W.LoiDauVao as e:
            out["giao_thuc_thoat"] = str(e)[:120]
    finally:
        os.environ[D.ENV_FIXTURE] = goc
    assert sorted(p.name for p in bad.iterdir()) == ["fixture.json"]
    return out


@ca("H2 junction (F5): chuan_bi / thuc_thi / phien / thu muc lan thu la junction tro ra ngoai goc vung ghi -> exit 2 "
    "TRUOC khi tao phien / cau hinh thu vien; dich junction khong bi ghi")
def ca_junction(lan: Path):
    import _winapi
    import shutil
    ngoai = lan / "ngoai_goc_fixture"               # dich junction: ngoai goc fixture / ngoai phien, van trong tests/
    fx2 = lan / "fx2"
    shutil.copytree(FX["dir"] / "nguon", fx2 / "nguon")
    shutil.copy2(FX["dir"] / "giao_thuc_fx.json", fx2 / "giao_thuc_fx.json")
    shutil.copy2(FX["json"], fx2 / "fixture.json")
    links = []

    def jn(dich: Path, link: Path):
        dich.mkdir(parents=True)
        _winapi.CreateJunction(str(dich), str(link))
        links.append(link)
        assert link.resolve() == dich.resolve() and Path(os.path.abspath(link)) != link.resolve()

    out = {}
    try:
        jn(ngoai / "chuan_bi", fx2 / "chuan_bi")
        jn(ngoai / "thuc_thi", fx2 / "thuc_thi")
        env = {D.ENV_FIXTURE: str(fx2 / "fixture.json")}
        for ten, args in (("tien_kiem", ["--buoc", "tien-kiem"]),
                          ("khoa", ["--buoc", "khoa", "--tien-kiem", "x", "--tests", "y"]),
                          ("kiem_khoa_luu", ["--buoc", "kiem-khoa", "--khoa", FX["khoa"], "--luu", "chuan_bi"]),
                          ("mo_phien", ["--buoc", "thuc-thi", "--cong-viec", "mo-phien", "--khoa", FX["khoa"],
                                        "--phien", "pj"])):
            out[ten] = can(cli(args, env_them=env), 2, "junction")
        # phien la junction (trong vung thuc_thi that cua fixture)
        pj = FX["dir"] / "thuc_thi" / "p_junction"
        jn(ngoai / "p_junction", pj)
        shutil.copy2(FX["phien_dir"] / "phien.json", ngoai / "p_junction" / "phien.json")
        assert ip(["--buoc", "thuc-thi", "--cong-viec", "hau-kiem", "--phien", "p_junction"]) == 2
        # thu muc cha cua lan thu la junction (cham-lai cua run chua tung cham lai thanh cong)
        run = "r54-yolov8n_coco-c10-cv"
        cl = FX["phien_dir"] / "cham_lai" / run
        assert not cl.exists()
        jn(ngoai / "cham_lai", cl)
        assert ip(tt("cham-lai", "--run", run)) == 2
        loi = nk_cuoi("cham-lai")["loi"]
        assert "junction" in loi, loi
        out["cham_lai"] = loi[:200]
        try:
            D.kiem_doc_lap(cl / "lan_x", FX["phien_dir"])
            raise AssertionError("kiem_doc_lap khong chan junction")
        except W.LoiDauVao as e:
            out["kiem_doc_lap"] = str(e)[:160]
        con = {d.name: sorted(q.name for q in d.rglob("*")) for d in ngoai.iterdir()}
        assert con == {"chuan_bi": [], "thuc_thi": [], "p_junction": ["phien.json"], "cham_lai": []}, con
        out["dich_junction"] = con
    finally:
        for link in links:                            # chi go LIEN KET junction do ca nay tao; dich giu nguyen
            os.rmdir(link)
    assert all(not os.path.lexists(x) for x in links) and (ngoai / "p_junction" / "phien.json").is_file()
    return out


@ca("I1 (F1 vong 3) manifest DA CONG BO roi marker xac nhan hong + npz cache sai hash -> cache exit 2 (khong goi "
    "detector, khong lan moi), xac-nhan exit 2, tracking exit 2; marker cache hong -> cache dong bang exit 2; bang "
    "chung giu nguyen byte")
def ca_marker_cong_bo_hong(lan: Path):
    det = DET[2]
    ph = FX["phien_dir"]
    xn, cdir = ph / "xac_nhan_cache" / det, ph / "cache" / det
    d_xn = [x for x in lans(xn) if D.doc_marker(x) is not None]
    assert len(d_xn) == 1, lans(xn)
    d_xn = d_xn[0]
    man = json.loads((d_xn / "manifest_cache.json").read_text(encoding="utf-8"))
    p_mk = d_xn / D.MARKER
    b_mk = p_mk.read_bytes()
    p_mk.write_bytes(b_mk[: len(b_mk) // 2])                  # marker cong bo bi hong SAU cong bo (giu nguyen ve sau)
    assert D.danh_gia_lan(d_xn, {})[0] == "hong"
    run = runs_cua(det)[0]
    trk = {v: len(lans(ph / "tracking" / run / v)) for v in (V1, V2)}
    lan_c2 = W.tuyet_doi(man["video"][V2]["lan"])
    out = {}
    for ten, p in (("npz_sai_hash", lan_c2 / f"{V2}.npz"), ("marker_cache_hong", lan_c2 / D.MARKER)):
        goc = p.read_bytes()
        if ten == "npz_sai_hash":
            sua_byte(p)
        else:
            p.write_bytes(goc[: len(goc) // 2])
        chup_c, chup_x = anh_chup(cdir), anh_chup(xn)
        try:
            ma_c = ip(tt("cache", "--detector", det))
            nk_c = nk_cuoi("cache")
            ma_x = ip(tt("xac-nhan-cache", "--detector", det))
            loi_x = nk_cuoi("xac-nhan-cache")["loi"]
            ma_t = ip(tt("tracking", "--run", run))
            loi_t = nk_cuoi("tracking")["loi"]
            giu = anh_chup(cdir) == chup_c and anh_chup(xn) == chup_x
        finally:
            p.write_bytes(goc)
        can_c = "sai hash" if ten == "npz_sai_hash" else "dong bang"
        assert ma_c == 2 and can_c in nk_c["loi"] and not nk_c["goi_detector"] and not nk_c["moi"], (ten, ma_c, nk_c)
        assert ma_x == 2 and ("sai hash" if ten == "npz_sai_hash" else "thieu cache") in loi_x, (ten, ma_x, loi_x)
        assert ma_t == 2 and "chua co manifest" in loi_t, (ten, ma_t, loi_t)
        assert giu, f"{ten}: bang chung cache / xac nhan doi"
        assert {v: len(lans(ph / "tracking" / run / v)) for v in (V1, V2)} == trk and len(lans(xn)) == 1
        out[ten] = {"cache": nk_c["loi"][:200], "xac_nhan": loi_x[:160], "tracking": loi_t[:120]}
    FX["xn_hong_cong_bo"] = (d_xn, anh_chup(d_xn), man)
    return out


@ca("I2 (F6) manifest KHAC nhung MOT cung hash: xac nhan lai giu dung lan cache cu -> tracking lan moi trung byte MOT; "
    "cham KHONG tai su dung lan cham cu -> lan moi; cham-lai / hau-kiem khong nhan lan cham cu")
def ca_manifest_khac_mot_trung(lan: Path):
    det = DET[2]
    ph_dir = FX["phien_dir"]
    d_xn, chup_xn, man_cu = FX["xn_hong_cong_bo"]
    assert ip(tt("cache", "--detector", det)) == 0                       # dong bang: chi tai su dung
    nk = nk_cuoi("cache")
    assert not nk["goi_detector"] and not nk["moi"] and len(nk["tai_su_dung"]) == 2, nk
    assert ip(tt("xac-nhan-cache", "--detector", det)) == 0
    ctx = D.Ctx()
    ph = D.Phien(ctx, FX["phien"])
    ph.xac_minh()
    man = D.kiem_cache_theo_manifest(ph, det)
    dm, _, _ = D.manifest_cache(ph, det)
    assert dm != d_xn and man["_sha256"] != W.sha256_file(d_xn / "manifest_cache.json")
    for v in (V1, V2):
        assert {k: man["video"][v][k] for k in ("lan", "npz_sha256", "json_sha256", "meta_sha256", "marker_sha256")} == \
            {k: man_cu["video"][v][k] for k in ("lan", "npz_sha256", "json_sha256", "meta_sha256", "marker_sha256")}, v
    run = runs_cua(det)[0]
    r = ctx.runs[run]
    c = ph_dir / "cham" / run
    cu = [x for x in lans(c) if D.doc_marker(x) is not None]
    assert len(cu) == 1
    cu = cu[0]
    mk_cu, chup_cu = D.doc_marker(cu), anh_chup(cu)
    trk_cu = {v: [x for x in lans(ph_dir / "tracking" / run / v) if D.doc_marker(x) is not None] for v in (V1, V2)}
    assert ip(tt("tracking", "--run", run)) == 0
    nk = nk_cuoi("tracking")
    assert len(nk["moi"]) == 2 and not nk["tai_su_dung"], nk                # lan tracking cu: manifest khac -> khong dung
    for v in (V1, V2):
        moi = lans(ph_dir / "tracking" / run / v)[-1]
        assert D.doc_marker(moi)["file"][f"data/{v}.txt"] == D.doc_marker(trk_cu[v][0])["file"][f"data/{v}.txt"], v
    assert ip(tt("cham", "--run", run)) == 0
    nk = nk_cuoi("cham")
    assert len(nk["moi"]) == 1 and not nk["tai_su_dung"], nk
    moi = lans(c)[-1]
    mk_moi = D.doc_marker(moi)
    a, b = mk_cu["rang_buoc"], mk_moi["rang_buoc"]
    assert a["mot_sha256_ds"] == b["mot_sha256_ds"] and a["manifest_cache_sha256"] != b["manifest_cache_sha256"]
    assert a["mot_lan_ds"] != b["mot_lan_ds"] and anh_chup(cu) == chup_cu
    d_chon, _ = D.lan_cham_da_chon(ph, r)
    assert d_chon == moi
    assert ip(tt("cham-lai", "--run", run)) == 0
    kq = nk_cuoi("cham-lai")["ket_qua"]
    assert kq["lan_cham_goc"].endswith(moi.name) and kq["file_trung_byte"] and kq["combined_trung"], kq
    assert ip(tt("hau-kiem")) != 0                                       # 11 run cung detector chua tracking lai
    hk = json.loads((lans(ph_dir / "hau_kiem")[-1] / "hau_kiem.json").read_text(encoding="utf-8"))
    khac = sorted(x for x in runs_cua(det) if x != run)
    assert sorted(x.split(":")[0] for x in hk["chua_xong"]) == khac and hk["dem"]["cham"] == W.SO_RUN - len(khac), hk
    assert W.rel_goc(d_xn) in hk["lan_marker_hong_giu_nguyen"] and anh_chup(d_xn) == chup_xn
    return {"manifest_cu": man_cu["video"][V1]["lan"], "manifest_moi_sha": man["_sha256"][:16],
            "cham_cu": cu.name, "cham_moi": moi.name, "chua_xong": len(hk["chua_xong"])}


# ---------------------------------------------------------------------------
# R56A: khoa moi lien ket khoa cha, chon video, doan co dinh
# ---------------------------------------------------------------------------
def ghi_khoa_cha_fx(ten: str, s: str | None = None) -> None:
    """<fixture>/khoa_cha.json: dac ta khoa cha cua che do fixture (khong thuoc ma_r55_sha)."""
    k = FX["dir"] / "chuan_bi" / ten / "khoa.json"
    (FX["dir"] / "khoa_cha.json").write_text(json.dumps({"ten": ten, "khoa_json_sha256": s or W.sha256_file(k)}),
                                             encoding="utf-8")


def ban_sao_khoa(goc: Path, ten: str, sua) -> Path:
    """Ban sao mot khoa fixture trong chuan_bi fixture; sua(khoa dict) roi ghi lai khoa.json + SHA256SUMS khop."""
    import shutil
    d = FX["dir"] / "chuan_bi" / ten
    shutil.copytree(goc, d)
    k = json.loads((d / "khoa.json").read_text(encoding="utf-8"))
    sua(k)
    (d / "khoa.json").write_text(json.dumps(k, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "SHA256SUMS").write_text(f"{W.sha256_file(d / 'khoa.json')}  khoa.json\n", encoding="utf-8")
    return d


@ca("J1 (R56) khoa moi lien ket khoa cha: khoa cha sai sha256 / SHA256SUMS cha sai / file cha lech ngoai 3 file -> "
    "khoa exit 2, khong tao phien; khoa cha hop le (3 file mang hash R55 that) -> R56_LOCK ghi bang khac biet; kiem-khoa "
    "khoa moi 0 lech; khoa cha doi / dac ta khac / bang khac biet sua -> exit 2; kiem lien ket khoa R55 THAT (chi doc)")
def ca_khoa_cha(lan: Path):
    cb = FX["dir"] / "chuan_bi"
    hd = D.doc_json(D.HOP_DONG, "hd")
    that = json.loads((GOC / hd["khoa_cha_r55"]["thu_muc"] / "khoa.json").read_text(encoding="utf-8"))
    cu = {f["rel"]: f["sha256"] for f in that["file"] if f["rel"] in D.DUOC_THAY_KHOA_CHA}
    assert sorted(cu) == sorted(D.DUOC_THAY_KHOA_CHA), cu

    def dat_cu(k):                                    # mo phong khoa cha voi ma/test/hop dong R55 (hash THAT)
        for f in k["file"]:
            if f["rel"] in cu:
                f["sha256"] = cu[f["rel"]]

    def lech_ngoai(k):
        dat_cu(k)
        next(f for f in k["file"] if f["rel"] == "src/r54_tracking/chuan_bi_test40.py")["sha256"] = "0" * 64

    ok = ban_sao_khoa(FX["khoa_dir"], "cha_ma_cu", dat_cu)
    ngoai = ban_sao_khoa(FX["khoa_dir"], "cha_lech_ngoai", lech_ngoai)
    sums = ban_sao_khoa(FX["khoa_dir"], "cha_sums_cu", dat_cu)
    (sums / "khoa.json").write_bytes((sums / "khoa.json").read_bytes() + b"\n")       # SHA256SUMS khong cap nhat
    lenh = ["--buoc", "khoa", "--tien-kiem", FX["tk"], "--tests", "lan_gia_dat"]
    out = {}
    for ten, cha, s, chua in (("sha_cha_sai", ok, "0" * 64, "khoa cha khoa.json"),
                              ("sums_cha_sai", sums, None, "khac SHA256SUMS"),
                              ("lech_ngoai_3_file", ngoai, None, "ngoai 3 file")):
        ghi_khoa_cha_fx(cha.name, s)
        truoc = sorted(p.name for p in cb.iterdir())
        out[ten] = can(cli(lenh), 2, chua)
        assert sorted(p.name for p in cb.iterdir()) == truoc, ten
    ghi_khoa_cha_fx(ok.name)
    truoc = {p.name for p in cb.iterdir()}
    out["khoa"] = can(cli(lenh), 0, "R56_LOCK")
    moi = sorted({p.name for p in cb.iterdir()} - truoc)
    assert len(moi) == 1, moi
    k2 = cb / moi[0]
    kj = json.loads((k2 / "khoa.json").read_text(encoding="utf-8"))
    kc, kb = kj["khoa_cha_r55"], kj["khac_biet_khoa_cha"]
    assert kj["buoc"] == "R56_LOCK" and kc["khoa_json_sha256"] == W.sha256_file(ok / "khoa.json")
    assert kc["thu_muc"] == W.rel_goc(ok) and kc["duoc_thay"] == list(D.DUOC_THAY_KHOA_CHA)
    assert sorted(x["rel"] for x in kb["thay"]) == sorted(D.DUOC_THAY_KHOA_CHA), kb["thay"]
    assert all(x["sha256_cha"] == cu[x["rel"]] and x["sha256_moi"] == W.sha256_file(GOC / x["rel"]) for x in kb["thay"])
    cha = json.loads((ok / "khoa.json").read_text(encoding="utf-8"))
    assert kb["so_giu"] == len(cha["file"]) - 3
    rels = {f["rel"] for f in kj["file"]}
    assert {f["rel"] for f in cha["file"]} <= rels
    them = {x["rel"] for x in kb["moi"]}
    assert {"handoff/logs/review56/01_bo_sung_chon_video.md", W.rel_goc(ok / "khoa.json"),
            W.rel_goc(ok / "SHA256SUMS"), W.rel_goc(ok / "thoi_diem_khoa.json")} <= them, them
    out["kiem_khoa"] = can(cli(["--buoc", "kiem-khoa", "--khoa", k2.name]), 0, "0 lech")
    # khoa cha doi byte sau khi khoa moi -> kiem-khoa khoa moi chan
    p = ok / "khoa.json"
    b = p.read_bytes()
    p.write_bytes(b + b" ")
    try:
        out["cha_doi"] = can(cli(["--buoc", "kiem-khoa", "--khoa", k2.name]), 2, "khoa cha")
    finally:
        p.write_bytes(b)
    # dac ta khoa cha (fixture) tro khoa khac -> chan
    ghi_khoa_cha_fx(ngoai.name)
    try:
        out["dac_ta_khac"] = can(cli(["--buoc", "kiem-khoa", "--khoa", k2.name]), 2, "khac dac ta")
    finally:
        ghi_khoa_cha_fx(ok.name)
    # bang khac biet bi sua (ban sao khoa moi, SHA256SUMS khop) -> chan
    bs = ban_sao_khoa(k2, "khoa_r56_bang_sua", lambda k: k["khac_biet_khoa_cha"].update(so_giu=0))
    out["bang_sua"] = can(cli(["--buoc", "kiem-khoa", "--khoa", bs.name]), 2, "bang khac biet")
    out["kiem_khoa_lai"] = can(cli(["--buoc", "kiem-khoa", "--khoa", k2.name]), 0, "0 lech")
    # khoa R55 cha THAT: chi doc, kiem lien ket voi ma hien tai -> chi 3 file duoc thay khac hash
    kt = D.kiem_khoa_cha(D.spec_khoa_cha_hop_dong(hd), False, hd)
    thay = sorted(x["rel"] for x in kt["bang"] if x["trang_thai"] == "thay")
    assert kt["lech"] == [] and kt["khoa_sha256"] == hd["khoa_cha_r55"]["khoa_json_sha256"], kt["lech"][:5]
    assert kt["so_file"] == hd["khoa_cha_r55"]["so_file"] and thay == sorted(D.DUOC_THAY_KHOA_CHA), thay
    FX.update({"khoa2": k2.name})
    return {**out, "khoa_moi": k2.name, "so_file": kj["so_file"], "so_giu": kb["so_giu"], "thay": kb["thay"],
            "khoa_cha_that": {"khoa_sha256": kt["khoa_sha256"], "so_file": kt["so_file"], "thay": thay,
                              "so_giu": sum(1 for x in kt["bang"] if x["trang_thai"] == "giu")}}


@ca("J2 (R56) tracking --video qua CLI tren phien khoa R56: --video sai cong viec / video la / cache chua du / cache "
    "video KHONG chon sai hash -> exit 2, khong dau ra tracking; ngat roi tiep tuc (lan do giu nguyen); chi sinh dau ra "
    "video chon, frame 1-n; goi lai tai su dung; cham / hau-kiem chua du video bi chan; chay tiep khong --video chi lam "
    "phan thieu; MOT (CV/UKF/PF500 s0-s2, ca 2 video) trung byte voi lan tinh doc lap qua run_tracking.run_video_to_dir")
def ca_chon_video(lan: Path):
    import det_cache
    import run_tracking as rt
    tt_dir = FX["dir"] / "thuc_thi"
    out = {"mo_phien": can(cli(["--buoc", "thuc-thi", "--cong-viec", "mo-phien", "--khoa", FX["khoa2"], "--phien",
                                "p2"]), 0, "mo-phien")}
    p2 = tt_dir / "p2"

    def t2(cv, *a):
        return ["--buoc", "thuc-thi", "--cong-viec", cv, "--phien", "p2", *a]

    def nk2(cv):
        d = lans(p2 / "nhat_ky" / cv.replace("-", "_"))[-1]
        return json.loads((d / "ket_qua.json").read_text(encoding="utf-8"))

    det = DET[0]
    runs = [f"r54-{det}-c25-{x}" for x in ("cv", "ukf", "pf500-s0", "pf500-s1", "pf500-s2")]
    t = p2 / "tracking"
    # --video o cong viec khac tracking / lap-tracker -> exit 2, khong ghi gi
    for cv, a in (("cache", ["--detector", det]), ("xac-nhan-cache", ["--detector", det]), ("cham", ["--run", runs[0]]),
                  ("cham-lai", ["--run", runs[0]]), ("hau-kiem", [])):
        truoc = anh_chup(p2)
        out[f"video_{cv}"] = can(cli(t2(cv, *a, "--video", V1)), 2, "--video chi dung")
        assert anh_chup(p2) == truoc and sorted(x.name for x in p2.iterdir()) == ["phien.json"], cv
    out["video_mo_phien"] = can(cli(["--buoc", "thuc-thi", "--cong-viec", "mo-phien", "--khoa", FX["khoa2"], "--phien",
                                     "p3", "--video", V1]), 2, "--video chi dung")
    assert not (tt_dir / "p3").exists()
    # cache chua du (V1 xong, V2 chua) -> tracking --video V1 chan
    r = cli(t2("cache", "--detector", det), env_them={D.ENV_NGAT: "cache-sau-marker:1"})
    assert r.returncode == 9, r.stdout[-800:] + r.stderr[-800:]
    out["cache_chua_du"] = can(cli(t2("tracking", "--run", runs[1], "--video", V1)), 2, "chua co manifest")
    assert not t.exists()
    assert ip(t2("cache", "--detector", det)) == 0 and ip(t2("xac-nhan-cache", "--detector", det)) == 0
    # video la -> chan truoc dau ra
    out["video_la"] = can(cli(t2("tracking", "--run", runs[1], "--video", "MVI_99999")), 2, "khong thuoc manifest")
    assert not t.exists()
    man = json.loads(next(d for d in lans(p2 / "xac_nhan_cache" / det) if D.doc_marker(d))
                     .joinpath("manifest_cache.json").read_text(encoding="utf-8"))
    # cache cua video KHONG duoc chon sai hash -> tracking --video V1 van chan
    p = W.tuyet_doi(man["video"][V2]["lan"]) / f"{V2}.npz"
    b = sua_byte(p)
    try:
        ma = ip(t2("tracking", "--run", runs[1], "--video", V1))
        loi = nk2("tracking")["loi"]
    finally:
        p.write_bytes(b)
    assert ma == 2 and "khong khop manifest" in loi and V2 in loi, (ma, loi)
    assert not t.exists()
    out["cache_khong_chon_sai_hash"] = loi[:200]
    # ngat TRUOC marker (UKF / V1) roi tiep tuc
    r = cli(t2("tracking", "--run", *runs[1:], "--video", V1), env_them={D.ENV_NGAT: "tracking-truoc-marker:1"})
    assert r.returncode == 9, r.stdout[-800:] + r.stderr[-800:]
    dd = lans(t / runs[1] / V1)
    assert len(dd) == 1 and not (dd[0] / D.MARKER).is_file() and (dd[0] / "data" / f"{V1}.txt").is_file()
    chup = anh_chup(dd[0])
    out["chon_video"] = can(cli(t2("tracking", "--run", *runs, "--video", V1)), 0, "moi 5, tai su dung 0")
    nk = nk2("tracking")
    assert nk["pham_vi"] == {"loc_video": True, "video": [V1], "so_video_manifest": 2}, nk["pham_vi"]
    assert len(nk["don_vi"]) == 5 and all(x["video"] == V1 and x["frame"] == [1, FX["n"]] and x["trang_thai"] == "moi"
                                          for x in nk["don_vi"]), nk["don_vi"]
    assert any(x["trang_thai"] == "do_dang" and x["lan"].endswith(dd[0].name) for x in nk["lan_thu"])
    assert anh_chup(dd[0]) == chup and not (dd[0] / D.MARKER).is_file()
    assert sorted(x.name for x in t.iterdir()) == sorted(runs) and all(sorted(x.name for x in (t / r_).iterdir()) == [V1]
                                                                        for r_ in runs)
    out["goi_lai"] = can(cli(t2("tracking", "--run", *runs, "--video", V1)), 0, "moi 0, tai su dung 5")
    assert all(x["trang_thai"] == "tai_su_dung" for x in nk2("tracking")["don_vi"])
    # cham / hau-kiem chua du video -> chan
    assert ip(t2("cham", "--run", *runs)) == 2 and "chua du 2 video" in nk2("cham")["loi"]
    assert not (p2 / "cham").exists()
    assert ip(t2("hau-kiem")) != 0
    hk = json.loads((lans(p2 / "hau_kiem")[-1] / "hau_kiem.json").read_text(encoding="utf-8"))
    assert all(f"{r_}: thieu tracking 1 video" in hk["chua_xong"] for r_ in runs) and not hk["du_36_run"], hk
    # chay tiep KHONG --video -> chi lam V2, tai su dung V1
    out["chay_tiep"] = can(cli(t2("tracking", "--run", *runs)), 0, "moi 5, tai su dung 5")
    nk = nk2("tracking")
    assert nk["pham_vi"]["loc_video"] is False and nk["pham_vi"]["video"] == [V1, V2]
    assert sorted((x["video"], x["trang_thai"]) for x in nk["don_vi"]) == sorted(
        [(V1, "tai_su_dung")] * 5 + [(V2, "moi")] * 5), nk["don_vi"]
    assert [len(lans(t / r_ / V1)) for r_ in runs] == [1, 2, 1, 1, 1] and anh_chup(dd[0]) == chup
    # MOT: so byte voi lan tinh DOC LAP qua duong toan video hien co (run_tracking.run_video_to_dir), cung cache/cfg/seed
    ctx = D.Ctx()
    reg = rt.tracker_registry()
    so = {}
    for run in runs:
        r_ = ctx.runs[run]
        cfg = rt.load_tracker_cfg(W.yaml_arg(r_["yaml"]), {"track_buffer": W.TRACK_BUFFER_BUOC})
        for v in (V1, V2):
            vd = det_cache.VideoDetections(W.tuyet_doi(man["video"][v]["lan"]) / f"{v}.npz", r_["conf_min"],
                                           ctx.gt["tai"]["classes"], ctx.gt["tai"]["max_det"], None,
                                           allow_truncated=False)
            o = lan / "doc_lap" / run / v
            rt.run_video_to_dir(o, v, vd, cfg, 1.0 / W.FPS, "skip", r_["seed"], reg, ctx.gt["tracker"]["assoc_log"],
                                ctx.gt["tracker"]["pf_diag"])
            a = (o / "data" / f"{v}.txt").read_bytes()
            chon = [x for x in lans(t / run / v) if D.doc_marker(x) is not None]
            assert len(chon) == 1, (run, v)
            assert a == (chon[0] / "data" / f"{v}.txt").read_bytes() and len(a) > 0, (run, v)
            so[f"{run}/{v}"] = W.sha256_bytes(a)[:16]
    assert ip(t2("lap-tracker", "--run", runs[4], "--video", V1)) == 0
    lt = nk2("lap-tracker")["ket_qua"]
    assert lt["trung_byte"] and lt["frame"] == [1, FX["n"]], lt
    return {**out, "mot_doc_lap_trung_byte": so, "seed_pf": [ctx.runs[x]["seed"] for x in runs[2:]]}


@ca("J3 (R56) doan co dinh MVI_40864 1-1515 = toan video: kiem tren bang tien kiem R55 THAT (chi doc) khop 1515 anh / "
    "44.689 dong GT (~29,50/anh); bang sua -> bat; tien-kiem fixture ghi bo qua")
def ca_doan_co_dinh(lan: Path):
    import csv
    hd = D.doc_json(D.HOP_DONG, "hd")
    p = GOC / "results/phase2/r55_tracking/chuan_bi/tien_kiem_20261021_121436_943200_pid22720/bang_tien_kiem.csv"
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    d = hd["doan_day"]["doan_co_dinh"]
    assert (d["video"], d["frame_dau"], d["frame_cuoi"], d["toan_video"]) == ("MVI_40864", 1, 1515, True)
    assert len(rows) == 40 and D.kiem_doan_co_dinh(hd, rows) == []
    r = next(x for x in rows if x["video"] == "MVI_40864")
    ti_le = int(r["so_dong_gt"]) / int(r["seq_length"])
    assert round(ti_le, 2) == d["dong_gt_moi_anh_can_cu"] == 29.5, ti_le
    out = {}
    for ten, sua in (("seq_length", {"seq_length": "1516"}), ("so_dong_gt", {"so_dong_gt": "1"}),
                     ("frame_cuoi", {"frame_cuoi": "1000"})):
        loi = D.kiem_doan_co_dinh(hd, [x if x is not r else {**r, **sua} for x in rows])
        assert loi, ten
        out[ten] = loi[0][:160]
    assert D.kiem_doan_co_dinh(hd, [x for x in rows if x is not r])
    tkj = json.loads((FX["dir"] / "chuan_bi" / FX["tk"] / "tien_kiem.json").read_text(encoding="utf-8"))
    assert "bo_qua" in tkj["doan_co_dinh"], tkj.get("doan_co_dinh")
    return {**out, "dong_gt_moi_anh": round(ti_le, 4), "so_anh": int(r["so_anh"]), "so_dong_gt": int(r["so_dong_gt"])}


# ---------------------------------------------------------------------------
def main() -> int:
    lan = W.tao_phien(D.kiem_doc_lap(VUNG_TESTS, D.VUNG_R55), None, "lan")
    W.dat_moi_truong(lan)
    truoc = snapshot()
    t0 = time.time()
    ca_import(lan)
    ca_cli_loi_that(lan)
    tao_fx(lan)
    for f in (ca_chuoi_chuan_bi, ca_mo_phien, ca_cache_thieu_video, ca_cache_tiep_tuc, ca_xac_nhan, ca_kiem_npz,
              ca_cache_sai_hash, ca_tracking_ngat, ca_tracking_sau_marker, ca_doi_backend_khoa, ca_doi_cau_hinh,
              ca_tracking_du, ca_cham, ca_lap_tracker, ca_cham_lai, ca_hau_kiem, ca_kiem_module_sau, ca_duong_dan,
              ca_junction, ca_marker_cong_bo_hong, ca_manifest_khac_mot_trung, ca_khoa_cha, ca_chon_video,
              ca_doan_co_dinh):
        if ("khoa" not in FX and f is ca_mo_phien) or ("phien" not in FX and f not in (ca_chuoi_chuan_bi, ca_mo_phien)) \
                or ("khoa2" not in FX and f is ca_chon_video):
            KET_QUA.append({"ten": f.__name__, "dat": False, "loi": "bo qua: chuoi chuan bi/mo-phien/khoa R56 loi"})
            continue
        f(lan)
    sau = snapshot()
    khac = sorted(k for k in set(truoc) | set(sau) if truoc.get(k) != sau.get(k) and not W.trong(Path(k), lan))
    KET_QUA.append({"ten": "Z1 khong ghi ngoai vung (snapshot thu muc nhay cam, gom khoa R55 cha THAT)", "dat": not khac,
                    "chi_tiet": khac[:50]})
    print(f"  [{'DAT ' if not khac else 'LOI '}] Z1 khong ghi ngoai vung ({len(khac)} khac)")
    tt_that = D.VUNG_R55 / "thuc_thi"
    that_0 = [r for r in CLI_LOG if not r["fixture"] and r["exit"] == 0]
    z2 = not tt_that.exists() and not that_0
    KET_QUA.append({"ten": "Z2 khong thuc thi du lieu that: khong co results/phase2/r55_tracking/thuc_thi/, moi lenh CLI "
                    "duong that deu bi tu choi (exit != 0)", "dat": z2,
                    "chi_tiet": {"thuc_thi_that_ton_tai": tt_that.exists(),
                                 "lenh_that_exit_0": [r["args"] for r in that_0]}})
    print(f"  [{'DAT ' if z2 else 'LOI '}] Z2 khong thuc thi du lieu that")
    so_dat = sum(1 for k in KET_QUA if k["dat"])
    tat_ca = so_dat == len(KET_QUA)
    W.ghi_json(lan / "cli_log.json", CLI_LOG, [lan])
    kq = {"buoc": "fixture R56A (R55 F1-F6 + chon video + khoa R56 lien ket khoa cha)", "lan": W.rel_goc(lan), "luc": W.bay_gio(), "tong_giay": round(time.time() - t0, 1),
          "so_test": len(KET_QUA), "so_dat": so_dat, "tat_ca_dat": tat_ca, "so_lenh_cli_con": len(CLI_LOG),
          "dieu_phoi_sha256": W.sha256_file(DIR / "dieu_phoi_test40.py"), "test_sha256": W.sha256_file(Path(__file__)),
          "hop_dong_sha256": W.sha256_file(D.HOP_DONG), "thiet_ke_sha256": W.sha256_file(D.THIET_KE)
          if D.THIET_KE.is_file() else None, "bo_sung_r56_sha256_luc_chay": W.sha256_file(D.THIET_KE_BO_SUNG)
          if D.THIET_KE_BO_SUNG.is_file() else None, "giao_thuc_r54_sha256": W.sha256_file(W.GIAO_THUC),
          "thu_vien": W.phien_ban_thu_vien(),
          "gioi_han": ["detector GIA LAP (MoHinhGia fixture R54) qua det_cache.build_video_yolo; YOLO/spec.fill_env()/GPU "
                       "khong chay; khong setup_model/forward checkpoint that",
                       "du lieu tong hop 2 video x 20 anh (co frame khong GT va anh D4); duong that 40 video chua chay",
                       "tracker (CV/EKF/UKF/PF500) va TrackEval THAT tren CPU; mot so ca goi main() trong tien trinh, "
                       "ngat/backend/chuoi chuan bi dung tien trinh con THAT (cli_log.json)",
                       "khoa fixture ke thua file khoa R54 THAT (bam lai, chi doc); ket qua test dung cho khoa fixture la "
                       "ban GIA trong vung fixture",
                       "snapshot chi phu cac thu muc liet ke trong snapshot(), khong phai toan bo dia",
                       "marker cat = gia lap (ghi nua dau byte marker roi os._exit 9), khong phai mat dien / loi he "
                       "thong tep that; junction tao bang _winapi.CreateJunction trong tests/, go lien ket sau ca",
                       "F4 kiem module sau cong viec: chi kiem muc ham (khoa trong bo nho sai sha256) + duong duong "
                       "tinh qua moi cong viec; KHONG co ca tiem module gia ngoai khoa trong mot don vi (nguoi dung tu "
                       "choi P6fb1cad7)",
                       "R56: khoa cha fixture = ban sao khoa fixture B1 voi 3 file duoc thay mang hash R55 THAT; khoa R55 "
                       "cha THAT chi DOC (kiem lien ket voi ma hien tai); tien-kiem duong that (doi chieu doan co dinh) "
                       "chua chay - doan co dinh chi kiem ham tren CSV tien kiem R55 THAT (chi doc)",
                       "R56: so byte MOT --video voi run_tracking.run_video_to_dir tren CUNG cache da xac nhan (khong so "
                       "timing); khong do thoi gian / QJTF; Z2 chi dua tren cli_log + su ton tai thu muc thuc_thi that"],
          "ket_qua": KET_QUA, "exit": 0 if tat_ca else 1}
    W.ghi_json(lan / "ket_qua_test.json", kq, [lan])
    print(f"[fixture R55A] {so_dat}/{len(KET_QUA)} dat -> {W.rel_goc(lan)}  exit {kq['exit']}  ({kq['tong_giay']} s)")
    return kq["exit"]


if __name__ == "__main__":
    raise SystemExit(main())
