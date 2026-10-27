"""
test_van_hanh_cong_a.py -- Fixture CPU cua lop van hanh cong A (plan A0, 24/10/2026).

Du lieu TONG HOP + detector MO PHONG (dung lai the gioi / giao thuc fixture / ModelGia cua test_chuan_bi_bdd; khong GPU,
khong checkpoint that). Moi lenh van hanh chay qua DUNG CLI `van_hanh_cong_a.py` trong tien trinh rieng (cha -> con),
log tung lenh ghi vao <phien>/lenh/. Tiem loi CHI qua --tiem-loi voi giao thuc fixture.

    python src/r61_bdd/test_van_hanh_cong_a.py --phien results/phase2/r61_bdd/tests/cong_a_20261024/lan_01 \
        --so-doan 1 --ngan-sach-giay 420

Kich ban tuan tu, ket qua tung kich ban ghi <phien>/kich_ban/<NN>_<ten>.json; het ngan sach giua hai kich ban -> exit 10,
chay lai DUNG lenh de tiep tuc cung phien (kich ban da xong khong chay lai; kich ban da bat dau ma chua ghi ket qua
-> ghi KHONG DAT "bi gian doan", khong chay lai).
Exit: 0 moi kiem dat; 2 co kiem khong dat / doi so sai; 10 con kich ban chua chay.
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
import van_hanh_cong_a as V  # noqa: E402  (keo theo chuan_bi_bdd, dieu_phoi_bdd -- muc module chi thu vien chuan)
CB, D = V.CB, V.D

RUNNER = H.SRC / "r61_bdd" / "van_hanh_cong_a.py"
DET0, DET1, DET2 = "yolov8n_coco", "yolov8n_mixed5", "yolo11n_mixed5"
KQ: list[dict] = []


def kiem(ten: str, ok: bool, chi_tiet=None):
    KQ.append({"kiem": ten, "dat": bool(ok), "chi_tiet": chi_tiet})
    print(("[DAT] " if ok else "[KHONG DAT] ") + ten + ("" if ok else f" -- {json.dumps(chi_tiet, ensure_ascii=False, default=str)[:1500]}"))


# ---------------------------------------------------------------------------
# Goi CLI + doc trang thai phien A gia
# ---------------------------------------------------------------------------
class T:
    P: Path
    gtp: Path
    fx: Path


def chay(argv: list[str], ten: str) -> dict:
    cmd = [H.PYTHON, str(RUNNER), *argv]
    env = dict(os.environ)
    env.update({"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "-1"})
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(H.ROOT), env=env, capture_output=True, timeout=590)
        ma, out, err = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        ma, out, err = None, e.stdout or b"", e.stderr or b""
    d = T.P / "lenh"
    stt = len(os.listdir(d)) + 1 if d.is_dir() else 1
    m = {"stt": stt, "ten": ten, "argv": cmd, "exit": ma, "exit_hex": None if ma is None else hex(ma & 0xFFFFFFFF),
         "giay": round(time.time() - t0, 3), "stdout": out.decode("utf-8", "replace"), "stderr": err.decode("utf-8", "replace")}
    H.ghi_json(d / f"{stt:03d}_{ten}.json", m, [T.P])
    print(f"  [lenh {stt:03d}] {ten}: exit {ma} ({m['giay']} s)")
    return m


def a(S: Path, buoc: str, *them) -> list[str]:
    return ["--cong", "A", "--phien", str(S), "--buoc", buoc, "--giao-thuc", str(T.gtp), "--device", "cpu", *them]


def tl(*ds) -> list[str]:
    return ["--tiem-loi", json.dumps([{"diem": d, "don_vi": u, "kieu": k} for d, u, k in ds])]


def lan(S: Path) -> list[dict]:
    return V.doc_lan_chay(S)


def kt(S: Path) -> dict:
    return lan(S)[-1]["ket_thuc"] or {}


def nk(x: dict) -> list[dict]:
    return V.doc_nhat_ky(x["dir"] / "tien_do.jsonl")[0]


def dem(S: Path) -> dict:
    return V.dem_crash(lan(S))


def sm(p: Path):
    return (H.sha256_file(p), os.stat(p).st_mtime_ns) if p.is_file() else None


def mk_cache(S: Path, det: str, v: str) -> str:
    cd = S / "cache" / det
    return H.kiem_marker(cd / f"{v}.xong.json", D.file_marker_cache(cd, v), {"detector": det, "video": v})[0]


def ly_do(k: dict) -> list[str]:
    return k.get("ly_do_hop_dong", [])


def la_native(k: dict) -> bool:
    return k.get("exit") not in (None, 0, 1, 2, 10)


def chi_tiet(S: Path, m: dict | None = None) -> dict:
    k = kt(S)
    return {"exit_cha": None if m is None else m["exit"], "con": {x: k.get(x) for x in
            ("exit", "exit_hex", "phan_loai", "don_vi", "quyet_dinh", "ly_do_hop_dong", "faulthandler_bao_loi_native")},
            "dem": dem(S)}


def trang_thai_ngoai() -> dict:
    return {"appdata": CB.appdata_ultralytics(),
            "pyc_src": sorted(H.rel(p) for p in H.liet_ke_file(H.SRC) if p.suffix == ".pyc"),
            "interim": H.liet_ke_nhe(H.ROOT / "data" / "interim"), "processed": H.liet_ke_nhe(H.ROOT / "data" / "processed"),
            "trackers": H.liet_ke_nhe(H.ROOT / "data" / "processed" / "trackers"),
            "chuan_bi": H.liet_ke_nhe(H.VUNG_CHUAN_BI), "phien_A_that_ton_tai": V.PHIEN_A_THAT.exists(),
            "ma_r61": {n: H.sha256_file(H.SRC / "r61_bdd" / n) for n in sorted(os.listdir(H.SRC / "r61_bdd"))
                       if n.endswith((".py", ".json"))},
            "hop_dong_lenh": H.sha256_file(V.HOP_DONG), "giao_thuc_that": H.sha256_file(D.GIAO_THUC)}


# ---------------------------------------------------------------------------
# Kich ban
# ---------------------------------------------------------------------------
def kb_thiet_lap():
    import test_chuan_bi_bdd as F
    vung = [T.P]
    fx = T.P / "fx"
    sha = F.tao_the_gioi(fx, vung)
    g = H.doc_json(F.giao_thuc_fixture(H.doc_json(D.GIAO_THUC), fx, sha, vung))
    H.ghi_bytes(fx / "PLAN_DA_CHOT.md", CB.PLAN.read_bytes(), vung)       # ban sao PLAN: kich ban sua ban sao, khong sua PLAN that
    g["van_hanh_a"] = {"plan": "PLAN_DA_CHOT.md", "chup_dau": "chup_dau", "kich_thuoc_doan_bam": 8}
    gtp = fx / "giao_thuc_cong_a.json"
    H.ghi_json(gtp, g, vung)
    V.chup_a(fx / "PLAN_DA_CHOT.md", fx / "chup_dau", vung)
    H.ghi_json(T.P / "thiet_lap.json", {"gtp": str(gtp), "fx": str(fx), "truoc": trang_thai_ngoai()}, vung)
    kiem("thiet_lap_fixture", gtp.is_file() and len(D.Ctx(gtp).runs()) == 36 and not V.PHIEN_A_THAT.exists(),
         {"phien_A_that_ton_tai": V.PHIEN_A_THAT.exists()})
    vh = H.doc_json(V.HOP_DONG)["van_hanh_cong_a"]
    kiem("hop_dong_lenh_json_hop_le_co_dieu_khoan_A0", all(k in vh for k in (
        "mot_doan", "xac_thuc_truoc_ghi", "smoke_truoc_khoa", "doi_chieu_git", "dong_goi_tiep_tuc", "fixture_vong"))
        and all(k in vh["chinh_sach_crash"] for k in ("thieu_log", "cha_crash_khi_con_chay")), sorted(vh))


def kb_so_sanh_git():
    """So sanh baseline: ngoai le DUY NHAT la mtime thu muc .git, khong lan sang file/noi dung Git (ham thuan)."""
    dau = {"head": "h0", "nhanh": "phase2", "plan": {"sha256_truoc_muc0": "t", "sha256_sau_muc0": "s"},
           "nguon_bao_ve": {"src/config.py": "c"}, "appdata_ultralytics": {"sha256": None}, "pyc_trong_src": []}
    nc = {".": {".git": [-1, 100], ".gitignore": [10, 5], "src": [-1, 7]},
          "results": {"results/phase2": [-1, 3]}}
    st = ["?? results/phase2/r61_bdd/tests/lan_01/x.json"]

    def ss(cuoi=None, nc2=None, st2=None):
        return V.so_sanh_chup(dau, {**dau, **(cuoi or {})}, nc, nc2 or nc, st, st2 or st)

    git_mt = {".": {**nc["."], ".git": [-1, 200]}, "results": nc["results"]}
    r = ss(nc2=git_mt)
    kiem("git_chi_mtime_thu_muc_git_la_ngoai_le", r["dat"] and r["loi"] == [] and [m["muc"] for m in r["ngoai_le"]] == [".git"], r)
    r = ss(nc2={".": {**git_mt["."], ".gitignore": [10, 9]}, "results": nc["results"]})
    kiem("git_ngoai_le_khong_lan_sang_gitignore", r["loi"] == ["vung nguy co doi ngoai pham vi: .:.gitignore"]
         and len(r["ngoai_le"]) == 1, r["loi"])
    r = ss(cuoi={"head": "h1"}, nc2=git_mt)
    kiem("git_ngoai_le_khong_mien_HEAD", r["loi"] == ["HEAD/nhanh doi"], r["loi"])
    r = ss(nc2=git_mt, st2=st + [" M .gitattributes"])
    kiem("git_ngoai_le_khong_mien_git_status", r["loi"] == ["git status moi ngoai pham vi:  M .gitattributes"], r["loi"])
    r = V.so_sanh_chup(dau, dau, {".": {"src": [-1, 7]}}, {".": {"src": [-1, 7], ".git": [-1, 200]}}, st, st)
    kiem("git_thu_muc_git_moi_xuat_hien_khong_phai_ngoai_le", r["loi"] == ["vung nguy co doi ngoai pham vi: .:.git"], r["loi"])
    r = ss(cuoi={"plan": {"sha256_truoc_muc0": "t", "sha256_sau_muc0": "s2"}}, nc2=git_mt)
    kiem("plan_ngoai_muc0_doi_la_loi", r["loi"] == ["PLAN ngoai muc 0 khac baseline: sha256_sau_muc0"], r["loi"])
    r = ss(nc2={".": git_mt["."], "results": {**nc["results"], "results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01": [-1, 1]}},
           st2=st + ["?? results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01/a.json", "?? src/r61_bdd/van_hanh_cong_a.py"])
    kiem("thay_doi_trong_pham_vi_A_khong_loi", r["dat"] and {x["pham_vi"] for x in r["git_status_doi"]} == {"pham_vi_a"}, r)
    r = ss(nc2=git_mt, st2=st + ["?? data/interim/moi.npz"])
    kiem("git_status_ngoai_pham_vi_la_loi", r["loi"] == ["git status moi ngoai pham vi: ?? data/interim/moi.npz"], r["loi"])
    # A0-F7: noi dung git diff tung file + index (git diff --cached) duoc doi chieu; ngoai le .git khong mien
    def khoi(f: str, dong: bytes) -> bytes:
        return (f"diff --git a/{f} b/{f}\nindex 1111111..2222222 100644\n--- a/{f}\n+++ b/{f}\n@@ -1 +1 @@\n-a\n".encode()
                + dong + b"\n")
    cfg, cfg2, pl, pl2 = khoi("src/config.py", b"+b"), khoi("src/config.py", b"+c"), \
        khoi("handoff/PLAN_DA_CHOT.md", b"+x"), khoi("handoff/PLAN_DA_CHOT.md", b"+y")
    st_m = st + [" M src/config.py", " M handoff/PLAN_DA_CHOT.md"]
    r = V.so_sanh_chup(dau, dau, nc, git_mt, st_m, st_m, cfg + pl, cfg2 + pl, b"", b"")
    kiem("git_ngoai_le_khong_mien_noi_dung_diff", r["loi"] == ["noi dung git diff doi ngoai pham vi: src/config.py"]
         and [m["muc"] for m in r["ngoai_le"]] == [".git"], r["loi"])
    r = V.so_sanh_chup(dau, dau, nc, git_mt, st_m, st_m, cfg + pl, cfg + pl2, b"", b"")
    kiem("git_diff_doi_trong_pham_vi_khong_loi", r["dat"] and [x["file"] for x in r["git_diff_doi"]] ==
         ["handoff/PLAN_DA_CHOT.md"] and r["git_diff_doi"][0]["pham_vi"] == "pham_vi_a", r)
    r = V.so_sanh_chup(dau, dau, nc, git_mt, st_m, st_m, cfg + pl, cfg + pl, b"", cfg)
    kiem("git_ngoai_le_khong_mien_index", r["loi"] == ["index Git (git diff --cached) doi: src/config.py"]
         and r["index_doi"] == ["src/config.py"], r["loi"])
    r = V.so_sanh_chup(dau, dau, nc, git_mt, st_m, st_m, cfg + pl, cfg + pl, None, b"")
    kiem("baseline_thieu_index_co_staged_la_loi", len(r["loi"]) == 1 and "baseline khong co index" in r["loi"][0], r["loi"])
    # Thong tin (KHONG tinh dat/khong dat): doi chieu trang thai THAT hien tai voi baseline lan_01, chi doc, ghi trong phien test
    out = T.P / "thong_tin"
    V.chup_a(CB.PLAN, out / "chup_that", [T.P])
    r = V.doi_chieu_baseline(V.CHUP_DAU_THAT, out / "chup_that")
    H.ghi_json(out / "doi_chieu_that_lan_01.json", r, [T.P])
    print(f"  [thong tin] doi chieu that voi lan_01: dat={r['dat']} loi={r['loi'][:5]} ngoai_le={[m['muc'] for m in r['ngoai_le']]}")


PYC_D = "src/__pycache__/baseline_track.cpython-311.pyc"
LOI_3 = ["pyc moi: " + PYC_D, "vung nguy co doi ngoai pham vi: results:results/track_video",
         "vung nguy co doi ngoai pham vi: src:src/__pycache__"]


def kb_ngoai_le_duyet():
    """Pe27e304a: ngoai le baseline CHI dung 3 muc (pyc baseline_track, mtime src/__pycache__, mtime results/track_video),
    doc tu hop_dong_lenh.json; khong lan sang pyc / file / thu muc khac, Git, PLAN ngoai muc 0."""
    ds = V.ngoai_le_duyet()
    kiem("ngoai_le_duyet_dung_3_muc_tu_hop_dong", sorted((m["loai"], m.get("vung"), m["duong"]) for m in ds) == [
        ("mtime_thu_muc", "results", "results/track_video"), ("mtime_thu_muc", "src", "src/__pycache__"),
        ("pyc_moi", None, PYC_D)], ds)
    hd = H.doc_json(V.HOP_DONG)
    sai = []
    for m_sai in ({"loai": "mtime_thu_muc", "duong": "src/__pycache__", "ghi_chu": ""},         # thieu vung
                  {"loai": "pyc_moi", "duong": "src/__pycache__/*.pyc", "ghi_chu": ""},     # wildcard
                  {"loai": "file_moi", "duong": "data/x", "ghi_chu": ""},                    # loai la
                  {"loai": "pyc_moi", "duong": "data/x.pyc", "ghi_chu": ""}):                # ngoai src
        h2 = json.loads(json.dumps(hd))
        h2["van_hanh_cong_a"]["ngoai_le_baseline_duyet"]["muc"] = [m_sai]
        try:
            V.ngoai_le_duyet(h2)
            sai.append(m_sai)
        except H.LoiHopDong:
            pass
    kiem("ngoai_le_duyet_cau_truc_sai_bi_tu_choi", sai == [], sai)
    dau = {"head": "h0", "nhanh": "phase2", "plan": {"sha256_truoc_muc0": "t", "sha256_sau_muc0": "s"},
           "nguon_bao_ve": {"src/config.py": "c"}, "appdata_ultralytics": {"sha256": None}, "pyc_trong_src": []}
    nc = {".": {".git": [-1, 100], "src": [-1, 7], "src/__pycache__": [-1, 4]}, "src": {"src/__pycache__": [-1, 10], "src/r61_bdd": [-1, 11],
          "src/config.py": [50, 12]}, "results": {"results/track_video": [-1, 20], "results/phase2": [-1, 21]}}
    st = ["?? results/phase2/r61_bdd/tests/lan_01/x.json"]

    def bien(**doi):
        out = json.loads(json.dumps(nc))
        out["."][".git"] = [-1, 200]
        for k, v in doi.items():
            vg, r = k.split("|")
            if v is None:
                out[vg].pop(r, None)
            else:
                out[vg][r] = v
        return out
    d3 = {"src|src/__pycache__": [-1, 99], "results|results/track_video": [-1, 98]}
    c3 = {**dau, "pyc_trong_src": [PYC_D]}

    def ss(cuoi=None, nc2=None, st2=None, df=(b"", b""), ix=(b"", b"")):
        return V.so_sanh_chup(dau, cuoi or c3, nc, nc2 or bien(**d3), st, st2 or st, df[0], df[1], ix[0], ix[1], duyet=ds)
    r, tho = ss(), V.so_sanh_chup(dau, c3, nc, bien(**d3), st, st)
    kiem("ngoai_le_duyet_3_muc_dat_va_ghi_rieng_ket_qua_tho",
         r["dat"] and r["loi"] == [] and sorted(m["muc"] for m in r["ngoai_le_duyet"]) == sorted(
             [PYC_D, "src/__pycache__", "results/track_video"]) and [m["muc"] for m in r["ngoai_le"]] == [".git"]
         and sorted(m["loi_tho"] for m in r["ngoai_le_duyet"]) == sorted(LOI_3) and sorted(tho["loi"]) == sorted(LOI_3),
         {"r": r["loi"], "duyet": r["ngoai_le_duyet"], "tho": tho["loi"]})
    ca = {
        "pyc_khac_cung_thu_muc": (dict(cuoi={**dau, "pyc_trong_src": [PYC_D, "src/__pycache__/config.cpython-311.pyc"]}),
                                  ["pyc moi: src/__pycache__/config.cpython-311.pyc"]),
        "pyc_cung_ten_thu_muc_khac": (dict(cuoi={**dau, "pyc_trong_src": ["src/r61_bdd/__pycache__/baseline_track.cpython-311.pyc"]}),
                                      ["pyc moi: src/r61_bdd/__pycache__/baseline_track.cpython-311.pyc"]),
        "thu_muc_duyet_bi_xoa": (dict(nc2=bien(**{**d3, "src|src/__pycache__": None})),
                                 ["vung nguy co doi ngoai pham vi: src:src/__pycache__"]),
        "thu_muc_duyet_doi_thanh_file": (dict(nc2=bien(**{**d3, "results|results/track_video": [123, 98]})),
                                         ["vung nguy co doi ngoai pham vi: results:results/track_video"]),
        "duong_duyet_sai_vung": (dict(nc2=bien(**{**d3, ".|src/__pycache__": [-1, 5]})),
                                 ["vung nguy co doi ngoai pham vi: .:src/__pycache__"]),
        "thu_muc_khac_doi_mtime": (dict(nc2=bien(**{**d3, "results|results/track_video_2": [-1, 1]})),
                                   ["vung nguy co doi ngoai pham vi: results:results/track_video_2"]),
        "file_khac_cung_vung_doi": (dict(nc2=bien(**{**d3, "src|src/config.py": [51, 13]})),
                                    ["vung nguy co doi ngoai pham vi: src:src/config.py"]),
        "file_moi_khac": (dict(nc2=bien(**{**d3, "src|src/moi.py": [5, 1]})),
                          ["vung nguy co doi ngoai pham vi: src:src/moi.py"]),
        "git_status_pyc_duyet": (dict(st2=st + ["?? " + PYC_D]), ["git status moi ngoai pham vi: ?? " + PYC_D]),
        "git_HEAD": (dict(cuoi={**c3, "head": "h1"}), ["HEAD/nhanh doi"]),
        "plan_ngoai_muc0": (dict(cuoi={**c3, "plan": {"sha256_truoc_muc0": "t2", "sha256_sau_muc0": "s"}}),
                            ["PLAN ngoai muc 0 khac baseline: sha256_truoc_muc0"]),
        "noi_dung_diff_file_khac": (dict(df=(b"", b"diff --git a/src/config.py b/src/config.py\n+x\n")),
                                    ["noi dung git diff doi ngoai pham vi: src/config.py"]),
        "index_git": (dict(ix=(b"", b"diff --git a/src/config.py b/src/config.py\n+x\n")),
                      ["index Git (git diff --cached) doi: src/config.py"]),
    }
    sai_ca = {}
    for ten, (kw, ky_vong) in ca.items():
        r = ss(**kw)
        if r["dat"] or r["loi"] != ky_vong:
            sai_ca[ten] = {"loi": r["loi"], "ky_vong": ky_vong}
    kiem("ngoai_le_duyet_khong_lan_sang_muc_khac_git_plan", sai_ca == {}, sai_ca)
    r = V.so_sanh_chup(dau, c3, nc, bien(**d3), st, st)                    # khong truyen danh sach -> khong mien
    kiem("khong_truyen_danh_sach_duyet_thi_khong_mien", sorted(r["loi"]) == sorted(LOI_3), r["loi"])
    # Doi chieu THAT (chi doc) voi baseline lan_01: dat voi DUNG .git + 3 muc da duyet, ket qua tho van ghi 3 loi
    out = T.P / "thong_tin"
    V.chup_a(CB.PLAN, out / "chup_that_ngoai_le", [T.P])
    r = V.doi_chieu_baseline(V.CHUP_DAU_THAT, out / "chup_that_ngoai_le")
    H.ghi_json(out / "doi_chieu_that_ngoai_le_duyet.json", r, [T.P])
    kiem("doi_chieu_that_dat_voi_dung_git_va_3_muc_duyet",
         r["dat"] and [m["muc"] for m in r["ngoai_le"]] == [".git"]
         and sorted(m["muc"] for m in r["ngoai_le_duyet"]) == sorted([PYC_D, "src/__pycache__", "results/track_video"])
         and sorted(r["ket_qua_tho_khong_ngoai_le_duyet"]["loi"]) == sorted(LOI_3),
         {"loi": r["loi"], "ngoai_le": [m["muc"] for m in r["ngoai_le"]], "duyet": r["ngoai_le_duyet"],
          "tho": r["ket_qua_tho_khong_ngoai_le_duyet"]})


def kb_tu_choi_that():
    """Cong A that: phien co dinh, khong tiem loi; cong B / ngan sach / phien chua mo bi chan TRUOC moi ghi."""
    ton = V.PHIEN_A_THAT.exists()
    m = chay(["--cong", "A", "--phien", "results/phase2/r61_bdd/chuan_bi/khac_thu_cong_a", "--buoc", "tien-kiem"], "that_phien_khac")
    kiem("that_phien_khac_bi_chan", m["exit"] == 2 and "phien co dinh" in m["stderr"]
         and not (H.VUNG_CHUAN_BI / "khac_thu_cong_a").exists(), m["stderr"][-400:])
    m = chay(["--cong", "A", "--phien", H.rel(V.PHIEN_A_THAT), "--buoc", "tien-kiem",
              *tl(("khoi_dong", None, "native"))], "that_tiem_loi")
    kiem("that_tiem_loi_bi_chan_khong_tao_phien", not ton and m["exit"] == 2 and "tiem loi chi cho giao thuc fixture"
         in m["stderr"] and not V.PHIEN_A_THAT.exists(), m["stderr"][-400:])
    m = chay(["--cong", "B", "--phien", str(T.P / "S_cong_b"), "--buoc", "tien-kiem", "--giao-thuc", str(T.gtp)], "cong_b")
    kiem("cong_B_bi_chan", m["exit"] == 2 and "chi nhan --cong A" in m["stderr"] and not (T.P / "S_cong_b").exists())
    m = chay(a(T.P / "S_ngan_sach", "tien-kiem", "--so-doan", "2", "--ngan-sach-giay", "420"), "nhieu_doan_420")
    kiem("nhieu_doan_420_giay_bi_chan", m["exit"] == 2 and "dung MOT doan" in m["stderr"]
         and not (T.P / "S_ngan_sach").exists())
    m = chay(a(T.P / "S_hai_doan_210", "tien-kiem", "--so-doan", "2", "--ngan-sach-giay", "210"), "hai_doan_210")
    kiem("hai_doan_210_giay_bi_chan", m["exit"] == 2 and "dung MOT doan" in m["stderr"]               # A0-F9
         and not (T.P / "S_hai_doan_210").exists(), m["stderr"][-300:])
    m = chay(a(T.P / "S_chua_mo", "cache", "--detector", DET0), "chua_tien_kiem")
    kiem("buoc_truoc_tien_kiem_bi_chan", m["exit"] == 2 and "phien A chua mo" in m["stderr"] and not (T.P / "S_chua_mo").exists())
    # A0-F5: --con / --phien sai -> exit 2, KHONG ghi gi (ke ca faulthandler.log, _tam, goi_cha.jsonl)
    goc_ngoai = H.VUNG_TESTS / "khong_ton_tai_con_a0"
    m = chay(["--con", str(goc_ngoai / "van_hanh" / "lan_chay" / "0001_tien-kiem")], "con_ngoai_vung")
    kiem("con_duong_ngoai_vung_khong_ghi", m["exit"] == 2 and "phien ngoai" in m["stderr"] and not goc_ngoai.exists(),
         m["stderr"][-300:])
    S = T.P / "S_con_sai"
    dc = S / "van_hanh" / "lan_chay" / "0001_tien-kiem"
    H.ghi_json(dc / "goi.json", {"stt": 1, "luot": V.LUOT, "buoc": "tien-kiem", "giao_thuc": str(T.gtp),
                                 "phien": H.rel(V.PHIEN_A_THAT)}, [T.P])
    c0 = cay(S)
    m = chay(["--con", str(dc)], "con_goi_sai_phien")
    kiem("con_goi_sai_phien_khong_ghi", m["exit"] == 2 and "goi.json khong dung" in m["stderr"] and cay(S) == c0,
         {"stderr": m["stderr"][-300:], "moi": sorted(set(cay(S)) - set(c0))})
    S = T.P / "S_phien_la"
    H.ghi_json(S / "van_hanh" / "phien.json", {"luot": "LUOT_KHAC", "phien": H.rel(S), "giao_thuc": H.rel(T.gtp)}, [T.P])
    H.tao_thu_muc(S / "van_hanh" / "lan_chay", [T.P])
    c0 = cay(S)
    m = chay(a(S, "tien-kiem"), "phien_sai_luot")
    kiem("phien_sai_luot_khong_ghi_goi_cha", m["exit"] == 2 and "khong dung luot" in m["stderr"] and cay(S) == c0
         and not (S / "van_hanh" / "goi_cha.jsonl").exists(), {"stderr": m["stderr"][-300:], "moi": sorted(set(cay(S)) - set(c0))})


def cay(S: Path) -> list[str]:
    """Moi file VA thu muc duoi S (de chung minh khong tao gi)."""
    out = []
    for g, ds, fs in os.walk(S):
        out += [H.rel(Path(g) / x) + ("/" if x in ds else "") for x in ds + fs]
    return sorted(out)


def kb_thieu_log():
    """A0-F3: thieu log -> DUNG, khong nhan exit 0 la hoan tat, khong cho thu lai crash."""
    S = T.P / "S_thieu_log"
    m = chay(a(S, "tien-kiem", *tl(("bo_faulthandler", None, "khac"))), "thieu_log_exit0")
    k = kt(S)
    ly = (k.get("quyet_dinh") or {}).get("ly_do_dung", [])
    kiem("thieu_log_exit0_khong_nhan_hoan_tat", m["exit"] == V.EXIT_DUNG and k.get("exit") == 0 and k.get("phan_loai") ==
         "hoan_tat" and k.get("thieu_log") == ["faulthandler.log"] and len(ly) == 1 and ly[0].startswith("thieu log")
         and (S / "van_hanh" / "dung.json").is_file() and dem(S)["tong"] == 0, chi_tiet(S, m))
    S = T.P / "S_thieu_log_crash"
    m = chay(a(S, "tien-kiem", *tl(("bo_faulthandler", None, "khac"), ("truoc_ghi", "tien-kiem", "native"))),
             "thieu_log_crash")
    k = kt(S)
    ly = (k.get("quyet_dinh") or {}).get("ly_do_dung", [])
    kiem("thieu_log_crash_dung_khong_thu_lai", m["exit"] == V.EXIT_DUNG and la_native(k) and k.get("phan_loai") ==
         "crash_native" and k.get("don_vi") == "tien-kiem" and k.get("thieu_log") == ["faulthandler.log"] and len(ly) == 1
         and ly[0].startswith("thieu log") and dem(S)["tong"] == 1, chi_tiet(S, m))


def kb_cha_khi_con_chay():
    """A0-F4: cha crash KHI con dang chay -> con bi giet (Job Object); lan sau xac nhan con da chet roi DUNG."""
    S = T.P / "S_cha_khi_con"
    t = ["--tiem-loi", json.dumps([{"diem": "cha_khi_con_chay", "giay": 4},
                                   {"diem": "khoi_dong", "kieu": "cho", "giay": 90}])]
    m = chay(a(S, "tien-kiem", *t), "cha_crash_khi_con_chay")
    x = lan(S)[-1]
    con = H.doc_json(x["dir"] / "con.json") if (x["dir"] / "con.json").is_file() else {}
    t0 = time.time()
    tt = V.trang_thai_con(con["pid"], con.get("thoi_diem_tao")) if con else {}
    while con and tt.get("con_song") is not False and time.time() - t0 < 20:
        time.sleep(0.5)
        tt = V.trang_thai_con(con["pid"], con.get("thoi_diem_tao"))
    out = (x["dir"] / "stdout.txt").read_text(encoding="utf-8", errors="replace")
    kiem("cha_crash_khi_con_chay_con_bi_giet", m["exit"] not in (0, 1, 2, 10, V.EXIT_CRASH, V.EXIT_DUNG) and
         x["ket_thuc"] is None and (con.get("job") or {}).get("dat") and "[CHO]" in out and tt.get("con_song") is False
         and not (x["dir"] / "ket_qua_con.json").exists(),
         {"exit": m["exit"], "con": con, "trang_thai": tt, "cho_giay": round(time.time() - t0, 2), "stdout": out[-200:]})
    m = chay(a(S, "tien-kiem"), "sau_cha_crash_khi_con_chay")
    k = kt(S)
    ly = (k.get("quyet_dinh") or {}).get("ly_do_dung", [])
    kiem("tiep_tuc_xac_nhan_con_da_chet_roi_dung", m["exit"] == V.EXIT_DUNG and k.get("phan_loai") == "cha_gian_doan"
         and (k.get("con_trang_thai") or {}).get("con_song") is False and k.get("ket_qua_con") is None
         and any("con da ket thuc" in s for s in ly) and dem(S)["tong"] == 1, chi_tiet(S, m))


def kb_cha_sau_tao_con():
    """A0-F4: cha crash NGAY SAU khi tao con, TRUOC con.json / xac nhan job / cho con chay -> con (tao tam dung, ke thua
    job cua cha) khong chay lenh nao va khong song sot; lan sau DUNG voi ly do dung."""
    import re
    S = T.P / "S_cha_sau_tao_con"
    m = chay(a(S, "tien-kiem", "--tiem-loi", json.dumps([{"diem": "cha_sau_tao_con"}])), "cha_crash_sau_tao_con")
    x = lan(S)[-1]
    r = re.search(r"\[TIEM CHA\] con pid (\d+) tao (\S+)", m["stderr"])
    pid = int(r.group(1)) if r else None
    tao = int(r.group(2)) if r and r.group(2).isdigit() else None
    t0 = time.time()
    tt = V.trang_thai_con(pid, tao) if pid else {}
    while pid and tt.get("con_song") is not False and time.time() - t0 < 20:
        time.sleep(0.5)
        tt = V.trang_thai_con(pid, tao)
    out = (x["dir"] / "stdout.txt").read_bytes() if (x["dir"] / "stdout.txt").is_file() else None
    kiem("cha_crash_truoc_gan_job_con_khong_song_sot", m["exit"] not in (0, 1, 2, 10, V.EXIT_CRASH, V.EXIT_DUNG) and
         x["ket_thuc"] is None and ((x["goi"] or {}).get("job_cha") or {}).get("dat") is True and pid is not None and
         tao is not None and tt.get("con_song") is False and not (x["dir"] / "con.json").exists() and
         not (x["dir"] / "faulthandler.log").exists() and not (x["dir"] / "tien_do.jsonl").exists() and out == b"",
         {"exit": m["exit"], "pid": pid, "tao": tao, "trang_thai": tt, "cho_giay": round(time.time() - t0, 2),
          "job_cha": (x["goi"] or {}).get("job_cha"), "file": sorted(os.listdir(x["dir"])), "stderr": m["stderr"][-300:]})
    n = len(lan(S))
    m = chay(a(S, "tien-kiem"), "sau_cha_crash_sau_tao_con")
    k = kt(S)
    ly = (k.get("quyet_dinh") or {}).get("ly_do_dung", [])
    kiem("tiep_tuc_sau_cha_crash_truoc_con_json_dung", m["exit"] == V.EXIT_DUNG and len(lan(S)) == n and
         k.get("phan_loai") == "cha_gian_doan" and k.get("con") is None and
         (k.get("con_trang_thai") or {}).get("ly_do") == "thieu con.json" and any("TAM DUNG" in s for s in ly) and
         k.get("file_con") == {"faulthandler.log": False, "tien_do.jsonl": False} and dem(S)["tong"] == 1 and
         (S / "van_hanh" / "dung.json").is_file(), chi_tiet(S, m))


def kb_loi_python():
    S = T.P / "S_loi_python"
    m = chay(a(S, "tien-kiem", *tl(("truoc_ghi", "tien-kiem", "python"))), "loi_python")
    k, x = kt(S), lan(S)[-1]
    err = (x["dir"] / "stderr.txt").read_text(encoding="utf-8", errors="replace")
    kiem("exit1_traceback_khong_gan_crash_native", m["exit"] == V.EXIT_DUNG and k.get("exit") == 1 and k.get("phan_loai") == "loi_python"
         and k.get("don_vi") == "tien-kiem" and "Traceback (most recent call last)" in err and "[TIEM LOI PYTHON]" in err
         and dem(S)["tong"] == 0 and (S / "van_hanh" / "dung.json").is_file(), chi_tiet(S, m))
    n = len(lan(S))
    m = chay(a(S, "tien-kiem"), "loi_python_goi_lai")
    kiem("dung_hoi_ton_tai_qua_loi_goi", m["exit"] == V.EXIT_DUNG and "[DUNG HOI]" in m["stderr"] and len(lan(S)) == n)


def kb_crash_khong_don_vi():
    S = T.P / "S_crash_khoi_dong"
    m = chay(a(S, "tien-kiem", *tl(("khoi_dong", None, "native"))), "crash_khoi_dong")
    k = kt(S)
    kiem("crash_khong_xac_dinh_don_vi_thi_dung", m["exit"] == V.EXIT_DUNG and la_native(k) and k.get("phan_loai") == "crash_native"
         and k.get("don_vi") is None and k.get("faulthandler_bao_loi_native") and dem(S)["tong"] == 1
         and k["quyet_dinh"]["ly_do_dung"] == ["crash native khong xac dinh duoc don vi dang do"], chi_tiet(S, m))


def kb_cha_crash():
    S = T.P / "S_cha_crash"
    m = chay(a(S, "tien-kiem", *tl(("cha_sau_con", None, "native"))), "cha_crash")
    x = lan(S)[-1]
    kq_con = H.doc_json(x["dir"] / "ket_qua_con.json") if (x["dir"] / "ket_qua_con.json").is_file() else {}
    fh = "".join(p.read_text(encoding="utf-8", errors="replace") for p in (S / "van_hanh" / "cha").iterdir())
    kiem("cha_crash_giu_trang_thai", m["exit"] not in (0, 1, 2, 10, V.EXIT_CRASH, V.EXIT_DUNG) and x["ket_thuc"] is None and kq_con.get("exit") == 0
         and "Windows fatal exception" in fh, {"exit": m["exit"], "kq_con": kq_con})
    n = len(lan(S))
    m = chay(a(S, "cache", "--detector", DET0), "sau_cha_crash")
    k = kt(S)
    kiem("cha_gian_doan_phat_hien_va_dung", m["exit"] == V.EXIT_DUNG and k.get("phan_loai") == "cha_gian_doan"
         and k.get("ket_qua_con", {}).get("exit") == 0 and len(lan(S)) == n and dem(S)["tong"] == 1
         and (S / "van_hanh" / "dung.json").is_file(), chi_tiet(S, m))


def kb_nguong_don_vi():
    S = T.P / "S_nguong_don_vi"
    kiem("S3_tien_kiem", chay(a(S, "tien-kiem"), "S3_tien_kiem")["exit"] == 0)
    dv = f"cache:{DET0}:fxv-a"
    ex = [chay(a(S, "cache", "--detector", DET0, *tl(("truoc_ghi", dv, "native"))), f"S3_crash_{i}")["exit"] for i in (1, 2, 3)]
    ds = lan(S)[-3:]
    lt = [e["lan_thu"] for x in ds for e in nk(x) if e.get("su_kien") == "bat_dau" and e.get("don_vi") == dv]
    truoc = [x["goi"]["dem_crash_truoc"]["tong"] for x in ds]
    kiem("nguong_don_vi_crash_lan_3_dung", ex == [V.EXIT_CRASH, V.EXIT_CRASH, V.EXIT_DUNG] and kt(S)["quyet_dinh"]["ly_do_dung"] ==
         [f"don vi {dv} crash lan 3 (nguong 3)"] and dem(S)["don_vi"].get(dv) == 3, {"exit": ex, **chi_tiet(S)})
    kiem("bo_dem_khong_reset_qua_loi_goi", lt == [1, 2, 3] and truoc == [0, 1, 2], {"lan_thu": lt, "dem_truoc": truoc})
    kiem("crash_truoc_ghi_khong_de_artifact", not any(n.startswith("fxv-a.") for n in os.listdir(S / "cache" / DET0)))
    n = len(lan(S))
    m = chay(a(S, "cache", "--detector", DET0), "S3_goi_lai")
    kiem("sau_nguong_moi_loi_goi_bi_chan", m["exit"] == V.EXIT_DUNG and len(lan(S)) == n)


def kb_nguong_phien():
    S = T.P / "S_nguong_phien"
    cd = S / "cache" / DET0
    kiem("S4_tien_kiem", chay(a(S, "tien-kiem"), "S4_tien_kiem")["exit"] == 0)
    m = chay(a(S, "cache", "--detector", DET0, *tl(("truoc_ghi", f"cache:{DET0}:fxv-a", "native"))), "S4_truoc_ghi")
    x = lan(S)[-1]
    kiem("gian_doan_truoc_ghi_artifact", m["exit"] == V.EXIT_CRASH and kt(S)["don_vi"] == f"cache:{DET0}:fxv-a"
         and not any(n.startswith("fxv-a.") for n in os.listdir(cd))
         and all((x["dir"] / f).is_file() for f in ("goi.json", "stdout.txt", "stderr.txt", "faulthandler.log"))
         and kt(S)["faulthandler_bao_loi_native"], chi_tiet(S, m))
    m = chay(a(S, "cache", "--detector", DET0, *tl(("sau_marker", f"cache:{DET0}:fxv-a", "native"))), "S4_sau_marker")
    a_truoc = sm(cd / "fxv-a.npz")
    kiem("gian_doan_sau_marker_don_vi_hop_le", m["exit"] == V.EXIT_CRASH and kt(S)["don_vi"] == f"cache:{DET0}:fxv-a"
         and mk_cache(S, DET0, "fxv-a") == "xong", chi_tiet(S, m))
    m = chay(a(S, "cache", "--detector", DET0, *tl(("sau_ghi", f"cache:{DET0}:fxv-b", "native"))), "S4_sau_ghi")
    ev = nk(lan(S)[-1])
    b_do = {n: H.sha256_file(cd / n) for n in sorted(os.listdir(cd)) if n.startswith("fxv-b.")}
    kiem("gian_doan_sau_ghi_artifact_truoc_marker", m["exit"] == V.EXIT_CRASH and kt(S)["don_vi"] == f"cache:{DET0}:fxv-b"
         and "fxv-b.npz" in b_do and "fxv-b.xong.json" not in b_do, {"b_do": b_do, **chi_tiet(S, m)})
    kiem("tiep_tuc_bo_qua_don_vi_hoan_tat_khong_doi_byte", sm(cd / "fxv-a.npz") == a_truoc and
         any(e.get("su_kien") == "bo_qua" and e.get("don_vi") == f"cache:{DET0}:fxv-a" for e in ev))
    m = chay(a(S, "cache", "--detector", DET0, *tl(("truoc_ghi", f"cache:{DET0}:fxv-b", "native"))), "S4_crash_4")
    kiem("nguong_tong_crash_phien_4_dung", m["exit"] == V.EXIT_DUNG and kt(S)["quyet_dinh"]["ly_do_dung"] ==
         ["tong crash phien 4 (nguong 4)"] and dem(S)["tong"] == 4, chi_tiet(S, m))
    kiem("artifact_do_giu_nguyen_sau_crash", {n: H.sha256_file(cd / n) for n in sorted(os.listdir(cd))
                                              if n.startswith("fxv-b.")} == b_do and sm(cd / "fxv-a.npz") == a_truoc)
    n = len(lan(S))
    kiem("S4_sau_nguong_bi_chan", chay(a(S, "cache", "--detector", DET0), "S4_goi_lai")["exit"] == V.EXIT_DUNG and len(lan(S)) == n)


def kb_nguong_buoc_khoa():
    S = T.P / "S_nguong_khoa"
    kiem("K_tien_kiem", chay(a(S, "tien-kiem"), "K_tien_kiem")["exit"] == 0)
    t = tl(("truoc_ghi", "khoa:chuan_bi", "native"))
    ex = [chay(a(S, "khoa", *t), f"K_khoa_crash_{i}")["exit"] for i in (1, 2)]
    kiem("nguong_buoc_khoa_crash_lan_2_dung", ex == [V.EXIT_CRASH, V.EXIT_DUNG] and kt(S)["quyet_dinh"]["ly_do_dung"] ==
         ["buoc khoa crash lan 2 (nguong 2)"] and dem(S)["tong"] == 2, {"exit": ex, **chi_tiet(S)})
    kiem("K_sau_nguong_bi_chan", chay(a(S, "kiem-khoa"), "K_goi_lai")["exit"] == V.EXIT_DUNG)


def M() -> Path:
    return T.P / "M"


def kb_m1_cache():
    S = M()
    cd = S / "cache" / DET0
    m = chay(a(S, "tien-kiem"), "M_tien_kiem")
    ss = H.doc_json(S / "tien_kiem_a" / "so_sanh.json") if (S / "tien_kiem_a" / "so_sanh.json").is_file() else {}
    xk = H.kiem_marker(S / "tien_kiem_a" / "xong.json", V.file_tien_kiem(S), {"buoc": "tien-kiem", "luot": V.LUOT})[0]
    kiem("M_tien_kiem_doi_chieu_baseline", m["exit"] == 0 and ss.get("dat") is True and xk == "xong",
         {"loi": ss.get("loi"), "ngoai_le": [x["muc"] for x in ss.get("ngoai_le", [])]})
    m = chay(a(S, "cache", "--detector", DET0, *tl(("sau_ghi", f"cache:{DET0}:fxv-b", "native"))), "M_cache_sau_ghi")
    a_truoc = sm(cd / "fxv-a.npz")
    b_do = {n: H.sha256_file(cd / n) for n in sorted(os.listdir(cd)) if n.startswith("fxv-b.")}
    kiem("M_crash_sau_ghi", m["exit"] == V.EXIT_CRASH and "fxv-b.xong.json" not in b_do and "fxv-b.npz" in b_do, chi_tiet(S, m))
    m = chay(a(S, "cache", "--detector", DET0), "M_cache_tiep")
    x = lan(S)[-1]
    sl = H.doc_json(next((S / "van_hanh" / "artifact_do" / f"cache__{DET0}__fxv-b").glob("lan_*/sao_luu.json")))
    sao = {Path(it["tu"]).name: it["sha256"][""] for it in sl["muc"]}
    den_ok = all(H.sha256_file(H.ROOT / it["den"]) == it["sha256"][""] for it in sl["muc"])
    lt = [e["lan_thu"] for e in nk(x) if e.get("su_kien") == "bat_dau" and e.get("don_vi") == f"cache:{DET0}:fxv-b"]
    kiem("M_artifact_do_sao_luu_nguyen_byte_roi_ghi_lai", m["exit"] == 0 and sao == b_do and den_ok and lt == [2]
         and all(mk_cache(S, DET0, v) == "xong" for v in ("fxv-a", "fxv-b", "fxv-c")), {"sao": sao, "b_do": b_do, "lt": lt})
    kiem("M_don_vi_hoan_tat_khong_doi_byte", sm(cd / "fxv-a.npz") == a_truoc)


def kb_m2_marker_hong():
    S = M()
    cd = S / "cache" / DET0
    mkr = H.rel(cd / "fxv-c.xong.json")

    def thu(p: Path, sua, ten: str, *can: str):
        goc, n0, d0 = p.read_bytes(), len(lan(S)), dem(S)["tong"]
        p.write_bytes(sua(goc))
        try:
            m = chay(a(S, "cache", "--detector", DET0), ten)
        finally:
            p.write_bytes(goc)
        k, ev = kt(S), nk(lan(S)[-1])
        chay_moi = [e["don_vi"] for e in ev if e.get("su_kien") == "bat_dau" and e.get("pha") == "chay"]
        ld = ly_do(k)
        kiem(ten, m["exit"] == 2 and k.get("phan_loai") == "loi_hop_dong" and len(ld) == 1 and all(c in ld[0] for c in can)
             and len(lan(S)) == n0 + 1 and dem(S)["tong"] == d0 and not chay_moi, {"ly_do": ld, "chay_moi": chay_moi})
    thu(cd / "fxv-c.npz", lambda b: b + b"x", "marker_hash_npz_lech_dung_khong_retry", f"marker {mkr} hong", "bam lech fxv-c.npz")
    thu(cd / "fxv-c.json", lambda b: b + b" ", "metadata_provenance_lech_dung", f"marker {mkr} hong", "bam lech fxv-c.json")
    thu(cd / "fxv-c.xong.json", lambda b: b.replace(f'"detector": "{DET0}"'.encode(), f'"detector": "{DET1}"'.encode()),
        "marker_danh_tinh_lech_dung", f"marker {mkr} hong", "danh tinh detector")
    thu(cd / "fxv-c.xong.json", lambda b: b[:20], "marker_hong_doc_loi_dung", f"marker {mkr} hong", "doc loi")
    m = chay(a(S, "cache", "--detector", DET0), "M_cache_sau_khoi_phuc")
    kiem("M_sau_khoi_phuc_byte_tiep_tuc", m["exit"] == 0 and all(mk_cache(S, DET0, v) == "xong" for v in ("fxv-a", "fxv-b", "fxv-c")))


def kb_m3_plan_chan():
    S = M()
    pf = T.fx / "PLAN_DA_CHOT.md"
    goc = pf.read_bytes()
    pf.write_bytes(goc + "\nthem ngoai muc 0 (fixture)\n".encode("utf-8"))
    try:
        m = chay(a(S, "cache", "--detector", DET1), "M_plan_ngoai_muc0")
    finally:
        pf.write_bytes(goc)
    x, k = lan(S)[-1], kt(S)
    out = (x["dir"] / "stdout.txt").read_text(encoding="utf-8", errors="replace")
    ld = ly_do(k)
    kiem("plan_ngoai_muc0_chan_truoc_inference", m["exit"] == 2 and len(ld) == 1 and
         "PLAN ngoai muc 0 khac baseline lan_01" in ld[0] and "sha256_sau_muc0" in ld[0]
         and not (S / "cache" / DET1).exists() and V.DAU_NAP not in out, {"ly_do": ld, "stdout": out[-300:]})
    for det in (DET1, DET2):
        m = chay(a(S, "cache", "--detector", det), f"M_cache_{det}")
        out = (lan(S)[-1]["dir"] / "stdout.txt").read_text(encoding="utf-8", errors="replace")
        kiem(f"M_cache_{det}", m["exit"] == 0 and V.DAU_NAP + det in out and
             all(mk_cache(S, det, v) == "xong" for v in ("fxv-a", "fxv-b", "fxv-c")), chi_tiet(S, m))


def kb_m4_kiem_cache_smoke():
    S = M()
    kc = S / "kiem_cache"
    # A0-F6: tan du ghi do dang (.dang_ghi) + ket qua cu chua marker -> phai duoc sao luu nguyen byte truoc khi ghi
    gia = {kc / "don_vi" / DET0 / "fxv-a.json.dang_ghi": b'{"do_dang": 1', kc / "ket_qua.json": b'{"cu_chua_marker": 1}\n',
           kc / "ket_qua.json.dang_ghi": b'{"do'}
    for p, b in gia.items():
        H.ghi_bytes(p, b, [T.P])
    m = chay(a(S, "kiem-cache"), "M_kiem_cache")
    kiem("M_kiem_cache", m["exit"] == 0 and H.doc_json(kc / "ket_qua.json")["so_don_vi"] == 9, chi_tiet(S, m))
    sl = {it["tu"]: it for p in sorted((S / "van_hanh" / "artifact_do").glob("kiem-cache__*/lan_*/sao_luu.json"))
          for it in H.doc_json(p)["muc"]}
    kiem("bao_toan_dang_ghi_va_ket_qua_cu_khi_ghi_lai", all(
        H.rel(p) in sl and sl[H.rel(p)]["sha256"][""] == H.sha256_bytes(b) and
        H.sha256_file(H.ROOT / sl[H.rel(p)]["den"]) == H.sha256_bytes(b) for p, b in gia.items())
        and not any(p.exists() for p in gia if p.name.endswith(".dang_ghi")), {"sao_luu": sorted(sl)})
    cham = S / "smoke" / "cham"

    def xong_cham():
        return sorted(r["run"] for r in D.Ctx(T.gtp).runs() if (cham / r["run"] / "cham.xong.json").is_file())

    def snap(runs):
        return {f: sm(f) for r in runs for f in sorted((cham / r).iterdir()) if f.is_file()}
    e1 = chay(a(S, "smoke", "--gioi-han-don-vi", "40"), "M_smoke_1")["exit"]
    n_trk = len(list((S / "smoke" / "tracking").glob("*/*/xong/*.xong.json")))
    c1 = xong_cham()
    s1 = snap(c1)
    e2 = chay(a(S, "smoke", "--gioi-han-don-vi", "10"), "M_smoke_2")["exit"]
    c2 = xong_cham()
    e3 = chay(a(S, "smoke"), "M_smoke_3")["exit"]
    c3 = xong_cham()
    kq = H.doc_json(S / "smoke" / "ket_qua_smoke.json") if (S / "smoke" / "ket_qua_smoke.json").is_file() else {}
    kiem("smoke_cham_tiep_tuc_du_run", (e1, e2, e3) == (10, 10, 0) and n_trk == 36 and len(c1) == 4 and len(c2) == 14
         and len(c3) == 36 and snap(c1) == s1 and kq.get("dat") and kq.get("so_tracking") == 36 and kq.get("so_cham") == 36,
         {"exit": [e1, e2, e3], "n_trk": n_trk, "cham": [len(c1), len(c2), len(c3)], "loi": kq.get("loi")})
    bl = kq.get("thoi_gian_tracking_theo_bo_loc", {})
    kiem("smoke_tham_so_seed_timeline_khop", kq.get("loi") == [] and set(bl) == {"cv", "ekf", "ukf", "pf500"}, bl)


def kb_m4b_smoke_truoc_khoa():
    """A0-F1: khoa kiem DU marker/hash tracking + cham cua smoke (khong chi marker tong hop)."""
    S = M()
    g = D.Ctx(T.gtp)
    r0, v0 = g.runs()[0]["run"], g.gt["smoke_cong_A"]["videos"][0]
    mk = S / "smoke" / "tracking" / g.split / r0 / "xong" / f"{v0}.xong.json"
    b = mk.read_bytes()
    os.remove(mk)
    try:
        m = chay(a(S, "khoa"), "F1_khoa_thieu_marker_tracking")
    finally:
        H.ghi_bytes(mk, b, [T.P])
    ld = ly_do(kt(S))
    kiem("khoa_chan_smoke_thieu_marker_tracking", m["exit"] == 2 and len(ld) == 1 and "smoke thieu 1 marker" in ld[0]
         and H.rel(mk) in ld[0] and not (S / "khoa").exists(), ld)
    cf = S / "smoke" / "cham" / r0 / "per_video_metrics.csv"
    b = cf.read_bytes()
    cf.write_bytes(b + b"x")
    try:
        m = chay(a(S, "khoa"), "F1_khoa_cham_hash_lech")
    finally:
        cf.write_bytes(b)
    ld = ly_do(kt(S))
    kiem("khoa_chan_smoke_cham_hash_lech", m["exit"] == 2 and len(ld) == 1 and "cham.xong.json" in ld[0] and "hong" in ld[0]
         and "bam lech per_video_metrics.csv" in ld[0] and not (S / "khoa").exists(), ld)


def kb_m5_khoa():
    S = M()
    e1 = chay(a(S, "khoa", "--gioi-han-don-vi", "2"), "M_khoa_1")["exit"]
    dg = S / "khoa" / "bam" / "doan" / "0001.json.dang_ghi"            # A0-F6: doan bam do dang
    H.ghi_bytes(dg, b'{"do', [T.P])
    # A0-F6: crash THAT giua luc ghi marker khoa.xong.json (nua bytes trong .dang_ghi) -> lan sau sao luu roi ghi lai
    xk_dg = S / "khoa" / "khoa.xong.json.dang_ghi"
    e2 = chay(a(S, "khoa", "--tiem-loi", json.dumps([{"diem": "dang_ghi", "don_vi": "khoa:ghi", "tep": "khoa.xong.json"}])),
              "M_khoa_2_crash_marker")["exit"]
    x2, k2 = lan(S)[-1], kt(S)
    b_mk = xk_dg.read_bytes() if xk_dg.is_file() else None
    kiem("crash_giua_ghi_marker_khoa_de_dang_ghi", e2 == V.EXIT_CRASH and la_native(k2) and k2.get("don_vi") == "khoa:ghi"
         and b_mk and (S / "khoa" / "khoa.json").is_file() and not (S / "khoa" / "khoa.xong.json").exists(), chi_tiet(S))
    kj_do = H.sha256_file(S / "khoa" / "khoa.json") if (S / "khoa" / "khoa.json").is_file() else None
    e2b = chay(a(S, "khoa"), "M_khoa_2_tiep")["exit"]
    sl = [it for p in (S / "van_hanh" / "artifact_do" / "khoa__bam__0001").glob("lan_*/sao_luu.json")
          for it in H.doc_json(p)["muc"]]
    kiem("bao_toan_dang_ghi_doan_bam", [it["tu"] for it in sl] == [H.rel(dg)] and sl[0]["sha256"][""] ==
         H.sha256_bytes(b'{"do') and not dg.exists(), sl)
    slg = {it["tu"]: it for p in (S / "van_hanh" / "artifact_do" / "khoa__ghi").glob("lan_*/sao_luu.json")
           for it in H.doc_json(p)["muc"]}
    lt = [e["lan_thu"] for e in nk(lan(S)[-1]) if e.get("su_kien") == "bat_dau" and e.get("don_vi") == "khoa:ghi"]
    kiem("bao_toan_dang_ghi_marker_khoa", e2b == 0 and b_mk is not None and
         slg.get(H.rel(xk_dg), {}).get("sha256", {}).get("") == H.sha256_bytes(b_mk) and
         H.sha256_file(H.ROOT / slg[H.rel(xk_dg)]["den"]) == H.sha256_bytes(b_mk) and
         slg.get(H.rel(S / "khoa" / "khoa.json"), {}).get("sha256", {}).get("") == kj_do and lt == [2] and
         not xk_dg.exists() and H.kiem_marker(S / "khoa" / "khoa.xong.json", [S / "khoa" / "khoa.json"])[0] == "xong",
         {"exit": e2b, "sao_luu": sorted(slg), "lan_thu": lt})
    ev2 = nk(x2)
    bam2 = sorted(e["don_vi"] for e in ev2 if e.get("su_kien") == "bat_dau" and ":bam:" in str(e.get("don_vi")))
    kp = S / "khoa" / "khoa.json"
    k = H.doc_json(kp) if kp.is_file() else {}
    pl = k.get("phu_luc_van_hanh", {})
    kiem("khoa_chia_doan_tiep_tuc_khong_bam_lai", (e1, e2, e2b) == (10, V.EXIT_CRASH, 0) and
         bam2 == ["khoa:bam:0001", "khoa:bam:0002", "khoa:bam:0003"], {"exit": [e1, e2, e2b], "bam2": bam2})
    kiem("khoa_bao_ma_runner_va_hop_dong", "r61_bdd/van_hanh_cong_a.py" in k.get("ma_nguon", {}) and
         pl.get("hop_dong_lenh_sha256") == H.sha256_file(V.HOP_DONG) and
         pl.get("plan", {}).get("sha256_toan_bo") == H.sha256_file(T.fx / "PLAN_DA_CHOT.md") and
         "r61_bdd/test_van_hanh_cong_a.py" in pl.get("ma_r61", {}) and len(k.get("cache", {})) == 30,
         {"so_cache": len(k.get("cache", {})), "phu_luc": sorted(pl)})
    t = sm(kp)
    e3 = chay(a(S, "khoa"), "M_khoa_lai")["exit"]
    kiem("khoa_hop_le_khong_ghi_de", e3 == 0 and sm(kp) == t)
    e4 = chay(a(S, "kiem-khoa"), "M_kiem_khoa")["exit"]
    kk = S / "kiem_khoa" / "vong_01" / "ket_qua.json"
    kiem("M_kiem_khoa_dat", e4 == 0 and kk.is_file() and H.doc_json(kk)["dat"], chi_tiet(S))


def kb_m6_dong_goi():
    S = M()
    gp = S / "goi_a.json"
    m = chay(a(S, "dong-goi-a", *tl(("giua_goi", "dong-goi-a:ghi", "native"))), "M_dong_goi_crash")
    do = sm(gp)
    kiem("dong_goi_gian_doan_de_goi_thieu", m["exit"] == V.EXIT_CRASH and do is not None and not (S / "SHA256SUMS").exists()
         and not (S / "goi_a.xong.json").exists(), chi_tiet(S, m))
    m = chay(a(S, "kiem-goi-a"), "M_kiem_goi_thieu")
    ld = ly_do(kt(S))
    kiem("goi_thieu_khong_duoc_hop_thuc_hoa", m["exit"] == 2 and len(ld) == 1 and "goi A chua hoan tat" in ld[0], ld)
    m = chay(a(S, "dong-goi-a"), "M_dong_goi_tiep")
    sl = list((S / "van_hanh" / "artifact_do" / "dong-goi-a__goi_do").glob("lan_*/sao_luu.json"))
    sao = {Path(it["tu"]).name: it["sha256"].get("") for it in H.doc_json(sl[0])["muc"]} if sl else {}
    goi = H.doc_json(gp) if gp.is_file() else {"file": {}}
    st_dau = goi.get("stt_dong_goi_dau", 0)
    f = goi["file"]
    pa = H.rel(S) + "/"
    co = [pa + x for x in ("khoa/khoa.json", "kiem_khoa/vong_01/ket_qua.json", "tien_kiem/tien_kiem.json",
                           "kiem_cache/ket_qua.json", "smoke/ket_qua_smoke.json", "van_hanh/phien.json",
                           "goi/nhat_ky_crash_chot.json", f"cache/{DET0}/fxv-a.npz")]
    lc = [r for r in f if r.startswith(pa + "van_hanh/lan_chay/")]
    kiem("dong_goi_tiep_tuc_sao_luu_goi_do", m["exit"] == 0 and sao.get("goi_a.json") == do[0] and
         H.kiem_marker(S / "goi_a.xong.json", [gp, S / "SHA256SUMS"])[0] == "xong", {"sao": sao, **chi_tiet(S, m)})
    kiem("goi_payload_dung_pham_vi", all(c in f for c in co) and lc and
         all(int(r[len(pa) + len("van_hanh/lan_chay/"):][:4]) < st_dau for r in lc) and
         not any((r.startswith(pa + "dong_goi/") and not r.startswith(pa + "dong_goi/phuc_hoi/")) or
                 r.endswith("van_hanh/so_crash.json") or "/van_hanh/cha/" in r
                 for r in f) and H.doc_json(S / "goi" / "nhat_ky_crash_chot.json")["dem"]["tong"] == 3,
         {"thieu": [c for c in co if c not in f], "st_dau": st_dau, "so_file": len(f)})
    # A0-F2: manifest bao DU artifact phuc hoi (ca ban sao luu goi do va sao_luu.json cua no)
    ad = V.liet_ke_rel(S / "van_hanh" / "artifact_do")
    goi_do = [r for r in ad if "/artifact_do/dong-goi-a__goi_do/" in r]
    kiem("manifest_bao_du_artifact_phuc_hoi", ad and set(ad) <= set(f) and
         any(r.endswith("/goi_a.json") for r in goi_do) and any(r.endswith("/sao_luu.json") for r in goi_do) and
         any(r.endswith("/goi/nhat_ky_crash_chot.json") for r in goi_do),
         {"thieu_trong_manifest": sorted(set(ad) - set(f))[:10], "goi_do": goi_do})
    thua = S / "van_hanh" / "artifact_do" / "thua_fixture.txt"
    H.ghi_bytes(thua, b"thua\n", [T.P])
    try:
        m = chay(a(S, "kiem-goi-a"), "M_kiem_goi_file_thua")
    finally:
        os.remove(thua)
    ld = ly_do(kt(S))
    kiem("kiem_goi_phat_hien_file_thua_trong_artifact_do", m["exit"] == 2 and len(ld) == 1 and
         f"thua: {H.rel(thua)}" in ld[0], ld)
    truoc = {r: H.sha256_file(H.ROOT / r) for r in f}
    m = chay(a(S, "kiem-goi-a"), "M_kiem_goi")
    vg = sorted((S / "van_hanh" / "kiem_goi_a").glob("vong_*/ket_qua.json"))
    kq = H.doc_json(vg[-1]) if vg else {}
    kiem("kiem_goi_a_dat_chi_doc_payload", m["exit"] == 0 and kq.get("dat") and kq.get("so_file") == len(f) and
         kq.get("goi_a_sha256") == H.sha256_file(gp) and {r: H.sha256_file(H.ROOT / r) for r in f} == truoc, chi_tiet(S, m))
    t = sm(gp)
    m = chay(a(S, "dong-goi-a"), "M_dong_goi_lai")
    kiem("goi_hop_le_khong_ghi_de", m["exit"] == 0 and sm(gp) == t)


def kb_m7_plan_sau_khoa():
    S = M()
    pf = T.fx / "PLAN_DA_CHOT.md"
    goc = pf.read_bytes()
    pf.write_bytes(goc.replace(CB.DAU_MUC0, CB.DAU_MUC0 + " (sua sau khoa, fixture)".encode("utf-8"), 1))
    try:
        m = chay(a(S, "kiem-khoa"), "M_kiem_khoa_plan_doi")
    finally:
        pf.write_bytes(goc)
    ld = ly_do(kt(S))
    vk = S / "van_hanh" / "kiem_khoa_sau_goi" / "vong_01" / "ket_qua.json"
    kiem("kiem_khoa_phat_hien_PLAN_doi_sau_khoa", m["exit"] == 2 and len(ld) == 1 and
         "phu_luc plan: sha256_toan_bo doi/thieu/thua" in ld[0] and vk.is_file() and not H.doc_json(vk)["dat"], ld)
    m = chay(a(S, "kiem-khoa"), "M_kiem_khoa_khoi_phuc")
    m2 = chay(a(S, "kiem-goi-a"), "M_kiem_goi_cuoi")
    kiem("sau_khoi_phuc_kiem_khoa_va_goi_dat", m["exit"] == 0 and m2["exit"] == 0 and dem(S)["tong"] == 3 and
         not (S / "van_hanh" / "dung.json").exists(), chi_tiet(S, m2))


def N() -> Path:
    """Phien rieng cho phuc hoi dong goi / kiem sau goi (A0-F2a/F2b/F6): M da dung 3 crash, them crash se cham nguong."""
    return T.P / "N"


def chay_toi_xong(S: Path, buoc: str, ten: str, *them, toi_da: int = 4) -> dict:
    """Exit 10 -> chay lai DUNG lenh (khong tiem loi)."""
    for i in range(toi_da):
        m = chay(a(S, buoc, *them), f"{ten}_{i + 1}")
        if m["exit"] != 10:
            break
    return m


def dang_ghi(dv: str, tep: str | None = None) -> list[str]:
    return ["--tiem-loi", json.dumps([{"diem": "dang_ghi", "don_vi": dv, "tep": tep}])]


def sao_luu_trong(goc: Path) -> dict:
    return {it["tu"]: it for p in sorted(goc.glob("*/lan_*/sao_luu.json")) for it in H.doc_json(p)["muc"]}


def kb_n1_chuan_bi():
    S = N()
    ex = [chay_toi_xong(S, "tien-kiem", "N_tien_kiem")["exit"]]
    ex += [chay_toi_xong(S, "cache", f"N_cache_{d}", "--detector", d)["exit"] for d in (DET0, DET1, DET2)]
    ex += [chay_toi_xong(S, b, f"N_{b}")["exit"] for b in ("kiem-cache", "smoke", "khoa", "kiem-khoa")]
    kiem("N_chuan_bi_den_kiem_khoa", ex == [0] * 8 and dem(S)["tong"] == 0, {"exit": ex})


def kb_n2_dong_goi_doan_bam():
    """A0-F2a: crash THAT giua luc ghi dong_goi/doan/<k>.json (de .dang_ghi) SAU khi danh muc co dinh -> tiep tuc hoan
    tat, khong file thua, manifest bao du bang chung phuc hoi; kiem-goi-a dat."""
    S = N()
    wd, ad = S / "dong_goi", S / "van_hanh" / "artifact_do"
    ad0 = V.liet_ke_rel(ad)
    m = chay(a(S, "dong-goi-a", *dang_ghi("dong-goi-a:bam:0001")), "N_dong_goi_crash_doan_bam")
    k = kt(S)
    dg = wd / "doan" / "0001.json.dang_ghi"
    b_do = dg.read_bytes() if dg.is_file() else None
    dm0 = (wd / "danh_muc.json").read_bytes() if (wd / "danh_muc.json").is_file() else None
    d0 = sm(wd / "doan" / "0000.json")
    kiem("dong_goi_crash_giua_ghi_doan_bam_de_dang_ghi", m["exit"] == V.EXIT_CRASH and la_native(k) and
         k.get("don_vi") == "dong-goi-a:bam:0001" and bool(b_do) and dm0 is not None and d0 is not None and
         not (wd / "doan" / "0001.json").exists() and not (S / "goi_a.json").exists(), chi_tiet(S, m))
    n0 = len(lan(S))
    m = chay_toi_xong(S, "dong-goi-a", "N_dong_goi_tiep")
    gp = S / "goi_a.json"
    goi = H.doc_json(gp) if gp.is_file() else {"file": {}}
    f = goi["file"]
    muc = sao_luu_trong(wd / "phuc_hoi")
    ph = V.liet_ke_rel(wd / "phuc_hoi")
    bam = sorted(e["don_vi"] for x in lan(S)[n0:] for e in nk(x) if e.get("su_kien") == "bat_dau" and
                 str(e.get("don_vi")).startswith("dong-goi-a:bam:"))
    kiem("dong_goi_tiep_tuc_sau_gian_doan_doan_bam_hoan_tat", m["exit"] == 0 and
         H.kiem_marker(S / "goi_a.xong.json", [gp, S / "SHA256SUMS"])[0] == "xong" and
         (wd / "danh_muc.json").read_bytes() == dm0 and sm(wd / "doan" / "0000.json") == d0 and
         list(muc) == [H.rel(dg)] and muc[H.rel(dg)]["sha256"][""] == H.sha256_bytes(b_do) and
         H.sha256_file(H.ROOT / muc[H.rel(dg)]["den"]) == H.sha256_bytes(b_do) and not dg.exists() and
         V.liet_ke_rel(ad) == ad0 and bam and bam[0] == "dong-goi-a:bam:0001" and "dong-goi-a:bam:0000" not in bam,
         {"exit": m["exit"], "sao_luu": sorted(muc), "bam_dau": bam[:2], "artifact_do_doi": V.liet_ke_rel(ad) != ad0})
    dm = H.doc_json(wd / "danh_muc.json")
    nkc = H.rel(S / "goi" / "nhat_ky_crash_chot.json")
    kiem("manifest_bao_du_phuc_hoi_dong_goi", bool(ph) and set(ph) <= set(f) and goi.get("phuc_hoi_dong_goi") == ph and
         all(f[r] == H.sha256_file(H.ROOT / r) for r in ph) and sorted(set(f) - set(ph) - {nkc}) == dm["file"] and
         sorted(V.payload(S, goi["stt_dong_goi_dau"])) == dm["file"],
         {"phuc_hoi": ph, "thieu": sorted(set(ph) - set(f)), "so_file": len(f)})
    m = chay_toi_xong(S, "kiem-goi-a", "N_kiem_goi_sau_phuc_hoi")
    vg = sorted((S / "van_hanh" / "kiem_goi_a").glob("vong_*/ket_qua.json"))
    kq = H.doc_json(vg[-1]) if vg else {}
    kiem("kiem_goi_dat_sau_phuc_hoi_doan_bam", m["exit"] == 0 and kq.get("dat") is True and kq.get("so_thua") == 0 and
         kq.get("so_file") == len(f), {"exit": m["exit"], "loi": kq.get("loi"), **chi_tiet(S, m)})


def trang_thai_payload(S: Path) -> dict:
    goi = H.doc_json(S / "goi_a.json")
    ds = sorted(set(V.payload(S, goi["stt_dong_goi_dau"])) | set(V.liet_ke_rel(S / "goi")) |
                set(V.liet_ke_rel(S / "dong_goi" / "phuc_hoi")) | set(goi["file"]))
    return {"ds": ds, "sha": {r: H.sha256_file(H.ROOT / r) if (H.ROOT / r).is_file() else None for r in ds},
            "goi": [sm(S / n) for n in ("goi_a.json", "SHA256SUMS", "goi_a.xong.json")],
            "artifact_do": V.liet_ke_rel(S / "van_hanh" / "artifact_do")}


def kb_n3_kiem_sau_goi():
    """A0-F2b + F6: gian doan kiem-goi-a (giua ghi doan bam) va kiem-khoa sau goi (giua ghi marker xong.json) -> phuc
    hoi trong vong kiem, payload da chot giu NGUYEN byte va danh muc."""
    S = N()
    p0 = trang_thai_payload(S)
    m = chay(a(S, "kiem-goi-a", *dang_ghi("kiem-goi-a:bam:0000")), "N_kiem_goi_crash_doan_bam")
    k = kt(S)
    R = sorted((S / "van_hanh" / "kiem_goi_a").glob("vong_*"))[-1]
    dg = R / "bam" / "doan" / "0000.json.dang_ghi"
    b_do = dg.read_bytes() if dg.is_file() else None
    kiem("kiem_goi_crash_giua_ghi_doan_bam_de_dang_ghi", m["exit"] == V.EXIT_CRASH and la_native(k) and
         k.get("don_vi") == "kiem-goi-a:bam:0000" and bool(b_do) and not (R / "xong.json").exists(), chi_tiet(S, m))
    m = chay_toi_xong(S, "kiem-goi-a", "N_kiem_goi_tiep")
    muc = sao_luu_trong(R / "phuc_hoi")
    kq = H.doc_json(R / "ket_qua.json") if (R / "ket_qua.json").is_file() else {}
    kiem("kiem_goi_tiep_tuc_giu_nguyen_payload", m["exit"] == 0 and kq.get("dat") is True and kq.get("so_thua") == 0 and
         H.kiem_marker(R / "xong.json", [R / "ket_qua.json"])[0] == "xong" and list(muc) == [H.rel(dg)] and
         muc[H.rel(dg)]["sha256"][""] == H.sha256_bytes(b_do) and not dg.exists() and trang_thai_payload(S) == p0,
         {"exit": m["exit"], "vong": H.rel(R), "sao_luu": sorted(muc), "loi": kq.get("loi"),
          "payload_giu_nguyen": trang_thai_payload(S) == p0})
    m = chay(a(S, "kiem-khoa", *dang_ghi("kiem-khoa:so_sanh", "xong.json")), "N_kiem_khoa_crash_marker")
    k = kt(S)
    R = sorted((S / "van_hanh" / "kiem_khoa_sau_goi").glob("vong_*"))[-1]
    xd = R / "xong.json.dang_ghi"
    b_mk = xd.read_bytes() if xd.is_file() else None
    kq_do = H.sha256_file(R / "ket_qua.json") if (R / "ket_qua.json").is_file() else None
    kiem("kiem_khoa_sau_goi_crash_giua_ghi_marker", m["exit"] == V.EXIT_CRASH and la_native(k) and
         k.get("don_vi") == "kiem-khoa:so_sanh" and bool(b_mk) and kq_do is not None and not (R / "xong.json").exists(),
         chi_tiet(S, m))
    m = chay_toi_xong(S, "kiem-khoa", "N_kiem_khoa_tiep")
    muc = sao_luu_trong(R / "phuc_hoi")
    kiem("kiem_khoa_sau_goi_phuc_hoi_marker_giu_nguyen_payload", m["exit"] == 0 and
         H.kiem_marker(R / "xong.json", [R / "ket_qua.json"])[0] == "xong" and H.doc_json(R / "ket_qua.json")["dat"] and
         muc.get(H.rel(xd), {}).get("sha256", {}).get("") == H.sha256_bytes(b_mk) and
         H.sha256_file(H.ROOT / muc[H.rel(xd)]["den"]) == H.sha256_bytes(b_mk) and
         muc.get(H.rel(R / "ket_qua.json"), {}).get("sha256", {}).get("") == kq_do and not xd.exists() and
         trang_thai_payload(S) == p0, {"exit": m["exit"], "vong": H.rel(R), "sao_luu": sorted(muc)})
    m = chay_toi_xong(S, "kiem-goi-a", "N_kiem_goi_cuoi")
    vg = sorted((S / "van_hanh" / "kiem_goi_a").glob("vong_*/ket_qua.json"))
    kq = H.doc_json(vg[-1]) if vg else {}
    kiem("N_kiem_goi_cuoi_dat_khong_thua_khong_dung", m["exit"] == 0 and kq.get("dat") is True and
         kq.get("so_thua") == 0 and trang_thai_payload(S) == p0 and dem(S)["tong"] == 3 and
         not (S / "van_hanh" / "dung.json").exists(), {"loi": kq.get("loi"), **chi_tiet(S, m)})


def kb_ket_thuc():
    tl0 = H.doc_json(T.P / "thiet_lap.json")["truoc"]
    sau = trang_thai_ngoai()
    kiem("khong_ghi_ngoai_pham_vi_pyc_settings", all(tl0[k] == sau[k] for k in sau), {k: tl0[k] == sau[k] for k in sau})
    tep = [p for p in H.liet_ke_file(T.P) if p.is_file()]
    do_dang = [H.rel(p) for p in tep if p.name.endswith(".dang_ghi") and "/van_hanh/artifact_do/" not in H.rel(p)
               and "/phuc_hoi/" not in H.rel(p)]
    kiem("khong_file_tam_do_dang_ngoai_ban_sao_luu", not do_dang, do_dang[:5])
    lenh = [H.doc_json(p) for p in sorted((T.P / "lenh").glob("*.json"))]
    kiem("khong_lenh_het_gio", all(x["exit"] is not None for x in lenh))


KICH_BAN = [("thiet_lap", kb_thiet_lap), ("so_sanh_git", kb_so_sanh_git), ("ngoai_le_duyet", kb_ngoai_le_duyet),
            ("tu_choi_that", kb_tu_choi_that),
            ("loi_python", kb_loi_python), ("crash_khong_don_vi", kb_crash_khong_don_vi), ("cha_crash", kb_cha_crash),
            ("cha_khi_con_chay", kb_cha_khi_con_chay), ("cha_sau_tao_con", kb_cha_sau_tao_con), ("thieu_log", kb_thieu_log),
            ("nguong_don_vi", kb_nguong_don_vi), ("nguong_phien", kb_nguong_phien), ("nguong_buoc_khoa", kb_nguong_buoc_khoa),
            ("m1_cache", kb_m1_cache), ("m2_marker_hong", kb_m2_marker_hong), ("m3_plan_chan", kb_m3_plan_chan),
            ("m4_kiem_cache_smoke", kb_m4_kiem_cache_smoke), ("m4b_smoke_truoc_khoa", kb_m4b_smoke_truoc_khoa),
            ("m5_khoa", kb_m5_khoa), ("m6_dong_goi", kb_m6_dong_goi),
            ("m7_plan_sau_khoa", kb_m7_plan_sau_khoa), ("n1_chuan_bi", kb_n1_chuan_bi),
            ("n2_dong_goi_doan_bam", kb_n2_dong_goi_doan_bam), ("n3_kiem_sau_goi", kb_n3_kiem_sau_goi),
            ("ket_thuc", kb_ket_thuc)]


def snapshot_fixture() -> dict:
    """Snapshot MA THUC CHAY (A0-F8): dong import tinh src + moi r61_bdd/*.py (ca fixture) + hop dong + giao thuc + Python."""
    ma = {p.relative_to(H.SRC).as_posix(): H.sha256_file(p) for p in D.dong_import_tinh(H.SRC)}
    ma.update({f"r61_bdd/{n}": H.sha256_file(H.SRC / "r61_bdd" / n) for n in sorted(os.listdir(H.SRC / "r61_bdd"))
               if n.endswith(".py")})
    s = {"ma": dict(sorted(ma.items())), "hop_dong_lenh": H.sha256_file(V.HOP_DONG),
         "giao_thuc_that": H.sha256_file(D.GIAO_THUC), "python": sys.executable}
    return {**s, "sha256": H.sha256_bytes(json.dumps(s, sort_keys=True).encode("utf-8"))}


def chon_vong(ph: Path, snap: dict) -> Path:
    """Vong cuoi cung snapshot -> tiep tuc; khac (hoac chua co) -> vong MOI. Ket qua o goc lan_01 (truoc A0-F8) va vong
    cua ma khac KHONG BAO GIO duoc dung cho snapshot hien tai."""
    ds = sorted(n for n in os.listdir(ph) if n.startswith("vong_") and n[5:].isdigit() and (ph / n).is_dir())
    if ds and (ph / ds[-1] / "vong.json").is_file() and \
            H.doc_json(ph / ds[-1] / "vong.json")["snapshot"]["sha256"] == snap["sha256"]:
        return ph / ds[-1]
    v = ph / f"vong_{max([int(n[5:]) for n in ds] + [1]) + 1:02d}"
    H.ghi_json(v / "vong.json", {"snapshot": snap, "tao_luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(),
                                 "vong_truoc": ds[-1] if ds else None,
                                 "ghi_chu": "vong 01 = ket qua o goc lan_01 do ma truoc A0-F1..F9; khong chung nhan ma nay"},
               [ph])
    return v


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phien", required=True)
    ap.add_argument("--so-doan", type=int, default=1)
    ap.add_argument("--ngan-sach-giay", type=float, default=420.0)
    ns = ap.parse_args(argv)
    t0 = time.time()
    vung = [T.P]
    snap = H.doc_json(T.P / "vong.json")["snapshot"]
    if snapshot_fixture()["sha256"] != snap["sha256"]:
        raise H.LoiHopDong("ma doi so voi snapshot cua vong -- chay lai de mo vong moi")
    if (T.P / "thiet_lap.json").is_file():
        tl0 = H.doc_json(T.P / "thiet_lap.json")
        T.gtp, T.fx = Path(tl0["gtp"]), Path(tl0["fx"])
    da_chay = 0
    for i, (ten, fn) in enumerate(KICH_BAN):
        kq_p, bd_p = T.P / "kich_ban" / f"{i:02d}_{ten}.json", T.P / "kich_ban" / f"{i:02d}_{ten}.bat_dau.json"
        if kq_p.is_file():
            continue
        if bd_p.is_file():
            H.ghi_json(kq_p, {"ten": ten, "kiem": [{"kiem": f"{ten}_khong_bi_gian_doan", "dat": False,
                                                    "chi_tiet": "kich ban da bat dau o loi goi truoc nhung khong ghi ket qua"}]},
                       vung)
            continue
        if da_chay and time.time() - t0 > ns.ngan_sach_giay - 150:
            print(f"[CHUA XONG] het ngan sach sau {da_chay} kich ban; chay lai DUNG lenh de tiep tuc")
            return H.EXIT_CHUA_XONG
        H.ghi_json(bd_p, {"ten": ten, "luc": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid()}, vung)
        print(f"== kich ban {i:02d} {ten}")
        KQ.clear()
        t1 = time.time()
        fn()
        if ten == "thiet_lap":
            tl0 = H.doc_json(T.P / "thiet_lap.json")
            T.gtp, T.fx = Path(tl0["gtp"]), Path(tl0["fx"])
        s_sau = snapshot_fixture()["sha256"]
        kiem(f"{ten}_ma_khong_doi_trong_kich_ban", s_sau == snap["sha256"], {"truoc": snap["sha256"], "sau": s_sau})
        H.ghi_json(kq_p, {"ten": ten, "giay": round(time.time() - t1, 3), "snapshot_sha256": snap["sha256"],
                          "snapshot_sau_sha256": s_sau, "kiem": list(KQ)}, vung)
        da_chay += 1
    kq = [H.doc_json(p) for p in sorted((T.P / "kich_ban").glob("[0-9][0-9]_*.json")) if not p.name.endswith(".bat_dau.json")]
    tat = [k for x in kq for k in x["kiem"]]
    hien_tai = snapshot_fixture()["sha256"]
    sai_snap = [x["ten"] for x in kq if x.get("snapshot_sha256") != snap["sha256"] or x.get("snapshot_sau_sha256") != snap["sha256"]]
    dat = all(k["dat"] for k in tat) and len(kq) == len(KICH_BAN) and not sai_snap and hien_tai == snap["sha256"]
    lenh = [H.doc_json(p) for p in sorted((T.P / "lenh").glob("*.json"))]
    p_kq = T.P / "ket_qua_test.json"
    if p_kq.exists():                      # khong ghi de ket qua tong hop cua vong (chi ghi mot lan)
        print(f"ket qua vong da co: {H.rel(p_kq)}")
        return 0 if H.doc_json(p_kq)["dat"] and dat else H.EXIT_HOP_DONG
    H.ghi_json(p_kq, {
        "dat": dat, "so_kiem": len(tat), "so_dat": sum(k["dat"] for k in tat), "so_kich_ban": len(kq),
        "kiem": tat, "so_lenh": len(lenh), "exit_lenh": {str(e): sum(x["exit"] == e for x in lenh)
                                                       for e in sorted({x["exit"] for x in lenh}, key=str)},
        "lenh": [{k: x[k] for k in ("stt", "ten", "exit", "exit_hex", "giay")} for x in lenh],
        "python": sys.executable, "vong": H.rel(T.P), "snapshot_vong": snap, "snapshot_hien_tai_sha256": hien_tai,
        "kich_ban_sai_snapshot": sai_snap,
        "ma_sha256": {n: h for n, h in snap["ma"].items() if n.startswith("r61_bdd/")}}, vung)
    print(f"KET QUA: {sum(k['dat'] for k in tat)}/{len(tat)} dat")
    return 0 if dat else H.EXIT_HOP_DONG


if __name__ == "__main__":
    argv = sys.argv[1:]
    ph = Path(os.path.abspath(H.ROOT / argv[argv.index("--phien") + 1])) if "--phien" in argv else None
    nsd = float(argv[argv.index("--ngan-sach-giay") + 1]) if "--ngan-sach-giay" in argv else 420.0
    sd = int(argv[argv.index("--so-doan") + 1]) if "--so-doan" in argv else 1
    if ph is None or not H._trong(ph, V.VUNG_FIXTURE) or ph == V.VUNG_FIXTURE or sd != 1 or not 0 < nsd <= 420:
        print("--phien phai duoi results/phase2/r61_bdd/tests/cong_a_20261024/; --so-doan 1; 0 < ngan_sach <= 420",
              file=sys.stderr)
        sys.exit(H.EXIT_HOP_DONG)
    H.tao_thu_muc(ph, [ph])
    T.P = chon_vong(ph, snapshot_fixture())          # moi vong gan voi snapshot ma thuc chay (A0-F8)
    print(f"vong fixture: {H.rel(T.P)}")
    H.chuan_bi_moi_truong(T.P, [T.P])
    import faulthandler
    _fh = open(H.kiem_duong_ghi(T.P / f"faulthandler_{time.strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}.log", [T.P]), "w",
               encoding="utf-8")
    faulthandler.enable(file=_fh, all_threads=True)
    sys.exit(H.chay_va_ghi_log(main, argv, T.P, [T.P], "test_van_hanh_cong_a"))
