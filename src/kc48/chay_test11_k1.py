"""
chay_test11_k1.py -- R50A / P64dfb0d8: chay CHI test 11 (runner seed) cua src/test_pf_ctrv.py tren ma PF co K1.

Khong sua src/test_pf_ctrv.py: import module va goi dung ham `test_11_runner_seeds` (khong goi main(), khong chay
test 1-10). Ghi vao results/phase2/kc48_pf_code/test11_k1/:
    fixture_truoc.json  -- sha256 tung file cua 3 thu muc fixture + (byte, mtime_ns) moi file khac trong
                           data/interim, data/processed, TRUOC khi chay
    fixture_sau.json    -- nhu tren, SAU khi chay, kem so sanh voi truoc
    ket_qua.json        -- commit, sha256 ma / cau hinh, seed, lenh (ca lenh con test goi), PASS/FAIL tung phep
                           kiem, exit, thoi gian, so fixture truoc/sau, doi chieu voi ban fixture R48C

Test 11 tu ghi fixture (xoa + ghi lai cache tong hop, ghi 2 thu muc tracker -s0/-s1); nguoi dung cho phep 12/10.
PYTHONDONTWRITEBYTECODE=1 de khong ghi __pycache__ (ca tien trinh con).

    C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/chay_test11_k1.py
"""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import contextlib  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

OUT = ROOT / "results/phase2/kc48_pf_code/test11_k1"
FIXTURE = ["data/interim/det_cache/fixture_pf_seed_test",
           "data/processed/trackers/DETRAC-sample/p2-pftest-s0",
           "data/processed/trackers/DETRAC-sample/p2-pftest-s1"]
VUNG_NGOAI = ["data/interim", "data/processed"]
MA = ["src/pf_ctrv.py", "src/tracker_ctrv.py", "src/test_pf_ctrv.py", "src/run_tracking.py", "src/det_cache.py",
      "src/kc48/chay_test11_k1.py"]
R48C_SAU = ROOT / "results/phase2/kc48_pf_code/fixture_sau.json"
SEEDS = [0, 1]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    return r.stdout.strip() if r.returncode == 0 else f"<git exit {r.returncode}: {r.stderr.strip()}>"


def git_blob_sha256(rel: str) -> str | None:
    r = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True)
    return hashlib.sha256(r.stdout).hexdigest() if r.returncode == 0 else None


def thong_tin_ma(rel: str) -> dict:
    """sha256 tho cua file tren dia + so voi HEAD theo id blob git (`git hash-object` ap dung chuan hoa CRLF/LF
    nhu khi commit), vi sha256 dia CRLF != sha256 blob LF du noi dung khong doi."""
    blob_head = git("rev-parse", f"HEAD:{rel}")
    blob_head = blob_head if not blob_head.startswith("<git") else None
    blob_cay = git("hash-object", "--", rel)
    return {"sha256_cay_lam_viec": sha256(ROOT / rel), "sha256_blob_head": git_blob_sha256(rel),
            "git_blob_head": blob_head, "git_hash_object_cay": blob_cay,
            "trang_thai": ("chua_theo_doi" if blob_head is None
                           else "giong_head" if blob_cay == blob_head else "khac_head")}


def chup() -> dict:
    fx = {}
    for d in FIXTURE:
        p = ROOT / d
        files = {}
        if p.exists():
            for f in sorted(x for x in p.rglob("*") if x.is_file()):
                st = f.stat()
                files[f.relative_to(p).as_posix()] = {"sha256": sha256(f), "byte": st.st_size,
                                                      "mtime_ns": st.st_mtime_ns}
        fx[d] = {"ton_tai": p.exists(), "file": files}
    fx_abs = [(ROOT / d).resolve() for d in FIXTURE]
    ngoai = {}
    for v in VUNG_NGOAI:
        for f in (ROOT / v).rglob("*"):
            if not f.is_file():
                continue
            fr = f.resolve()
            if any(fr.is_relative_to(a) for a in fx_abs):
                continue
            st = f.stat()
            ngoai[f.relative_to(ROOT).as_posix()] = [st.st_size, st.st_mtime_ns]
    ngoai = dict(sorted(ngoai.items()))
    return {"thoi_diem": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "fixture": fx,
            "ngoai_fixture": {"vung": VUNG_NGOAI, "ghi_chu": "chi (byte, mtime_ns), khong hash noi dung",
                              "n_file": len(ngoai), "file_byte_mtime_ns": ngoai}}


def so_sanh(truoc: dict, sau: dict) -> dict:
    out = {"fixture": {}, "ngoai_fixture": {}}
    for d in FIXTURE:
        a, b = truoc["fixture"][d]["file"], sau["fixture"][d]["file"]
        out["fixture"][d] = {
            "them": sorted(set(b) - set(a)), "mat": sorted(set(a) - set(b)),
            "doi_noi_dung": sorted(k for k in set(a) & set(b) if a[k]["sha256"] != b[k]["sha256"]),
            "chi_doi_mtime": sorted(k for k in set(a) & set(b)
                                    if a[k]["sha256"] == b[k]["sha256"] and a[k]["mtime_ns"] != b[k]["mtime_ns"]),
            "giu_nguyen": sorted(k for k in set(a) & set(b) if a[k] == b[k])}
    a, b = truoc["ngoai_fixture"]["file_byte_mtime_ns"], sau["ngoai_fixture"]["file_byte_mtime_ns"]
    out["ngoai_fixture"] = {"n_truoc": len(a), "n_sau": len(b), "them": sorted(set(b) - set(a)),
                            "mat": sorted(set(a) - set(b)),
                            "doi": sorted(k for k in set(a) & set(b) if a[k] != b[k])}
    out["ngoai_fixture"]["khong_doi"] = not (out["ngoai_fixture"]["them"] or out["ngoai_fixture"]["mat"]
                                             or out["ngoai_fixture"]["doi"])
    return out


def doi_chieu_r48c(sau: dict) -> dict:
    """So sha256 fixture sau lan chay nay voi fixture_sau.json cua R48C (ma truoc K1, 2026-10-11)."""
    if not R48C_SAU.exists():
        return {"co_ban_r48c": False}
    ref = json.loads(R48C_SAU.read_text(encoding="utf-8"))["fixture"]
    rows = {}
    for d in FIXTURE:
        a = ref.get(d, {}).get("file", {})
        b = sau["fixture"][d]["file"]
        for k in sorted(set(a) | set(b)):
            rows[f"{d}/{k}"] = {"r48c": a.get(k, {}).get("sha256"), "r50a": b.get(k, {}).get("sha256"),
                                "trung": a.get(k, {}).get("sha256") == b.get(k, {}).get("sha256")}
    return {"co_ban_r48c": True, "nguon": R48C_SAU.relative_to(ROOT).as_posix(),
            "nguon_sha256": sha256(R48C_SAU), "theo_file": rows,
            "ghi_chu": "meta.json/synth.json cua cache va run_meta/summary/timing/video_meta chua thoi diem hoac "
                       "thoi gian chay nen co the khac; file dau ra tracking la data/synth.txt va data_cls/synth.csv"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t_bat_dau = time.time()
    lenh = [sys.executable, "src/kc48/chay_test11_k1.py", *sys.argv[1:]]

    truoc = chup()
    (OUT / "fixture_truoc.json").write_text(json.dumps(truoc, ensure_ascii=False, separators=(",", ":")),
                                            encoding="utf-8")
    git_status_truoc = git("status", "--porcelain=v1", "--untracked-files=all", "--", "src", "configs")

    import test_pf_ctrv as T  # noqa: E402  (chi dinh nghia ham, khong chay test khi import)
    import tracker_ctrv  # noqa: E402
    from run_tracking import cfg_to_dict, load_tracker_cfg, resolve_tracker_yaml  # noqa: E402

    tracker_ctrv.register()          # giong T.main() truoc khi chay test

    lenh_con = []
    goc_run = subprocess.run

    def run_ghi(cmd, *a, **kw):
        t0 = time.time()
        r = goc_run(cmd, *a, **kw)
        lenh_con.append({"cmd": [str(c) for c in cmd], "cwd": str(kw.get("cwd")), "returncode": r.returncode,
                         "giay": round(time.time() - t0, 3),
                         "stdout_cuoi": (r.stdout or "")[-1500:] if kw.get("capture_output") else None,
                         "stderr_cuoi": (r.stderr or "")[-1500:] if kw.get("capture_output") else None})
        return r

    buf = io.StringIO()
    loi = None
    T._RESULTS.clear()
    t0 = time.time()
    T.subprocess.run = run_ghi       # chi ghi lai lenh con; hanh vi khong doi
    try:
        with contextlib.redirect_stdout(buf):
            print("--- test_11_runner_seeds ---")
            T.test_11_runner_seeds()
    except AssertionError as e:
        loi = {"loai": "AssertionError", "thong_diep": str(e)}
    except Exception as e:  # noqa: BLE001
        loi = {"loai": type(e).__name__, "thong_diep": str(e), "traceback": traceback.format_exc()}
    finally:
        T.subprocess.run = goc_run
    giay_test = time.time() - t0
    log_test = buf.getvalue()
    sys.stdout.write(log_test)

    sau = chup()
    ss = so_sanh(truoc, sau)
    sau_ghi = dict(sau)
    sau_ghi["so_voi_truoc"] = ss
    (OUT / "fixture_sau.json").write_text(json.dumps(sau_ghi, ensure_ascii=False, separators=(",", ":")),
                                          encoding="utf-8")
    git_status_sau = git("status", "--porcelain=v1", "--untracked-files=all", "--", "src", "configs")

    phep_kiem = [{"ten": n, "ket_qua": "PASS" if ok else "FAIL"} for n, ok in T._RESULTS]
    n_kiem_du_kien = 5               # so lan goi _check trong test_11_runner_seeds
    dat = loi is None and len(phep_kiem) == n_kiem_du_kien and all(x["ket_qua"] == "PASS" for x in phep_kiem)
    exit_code = 0 if dat else 1

    yaml = Path(T.PF_YAML)
    yaml_thuc = Path(resolve_tracker_yaml(yaml))
    cfg = cfg_to_dict(load_tracker_cfg(yaml))
    run_meta = {}
    for s in SEEDS:
        p = ROOT / f"data/processed/trackers/DETRAC-sample/p2-pftest-s{s}/run_meta.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            tc = m.get("tracker_cfg") or {}
            run_meta[f"s{s}"] = {"seed": m.get("seed"), "git_commit": m.get("git_commit"),
                                 "created": m.get("created"), "tracker_yaml": m.get("tracker_yaml"),
                                 "pf_optimal_proposal_trong_cfg": tc.get("pf_optimal_proposal", "<khong co khoa>"),
                                 "pf_roughening": tc.get("pf_roughening"),
                                 "pf_num_particles": tc.get("pf_num_particles")}

    ket_qua = {
        "buoc": "R50A_TEST11_THIET_KE_DETA_TRANG_THAI / P64dfb0d8",
        "chi_chay": "src/test_pf_ctrv.py::test_11_runner_seeds (import + goi ham; khong goi main, khong chay test 1-10)",
        "commit_head": git("rev-parse", "HEAD"),
        "nhanh": git("rev-parse", "--abbrev-ref", "HEAD"),
        "ma": {p: thong_tin_ma(p) for p in MA},
        "ghi_chu_ma": "K1 nam trong src/pf_ctrv.py, src/tracker_ctrv.py cua CAY LAM VIEC (chua commit) -> "
                      "trang_thai khac_head; HEAD chi la diem goc. sha256_cay_lam_viec (byte tren dia, co the CRLF) "
                      "co the khac sha256_blob_head (LF) ca khi noi dung giong HEAD; so HEAD theo git_blob.",
        "git_status_src_configs": {"truoc": git_status_truoc, "sau": git_status_sau,
                                   "trung": git_status_truoc == git_status_sau},
        "cau_hinh_tracker": {"duong_dan": yaml.relative_to(ROOT).as_posix(), "sha256": sha256(yaml),
                             "duong_dan_nap_thuc": str(yaml_thuc), "sha256_nap_thuc": sha256(yaml_thuc),
                             "pf_optimal_proposal_hieu_luc": cfg.get("pf_optimal_proposal", "<khong co khoa -> mac dinh tat>"),
                             "cfg_hieu_luc": cfg},
        "seed": SEEDS,
        "lenh": " ".join(lenh),
        "lenh_con_test_goi": lenh_con,
        "run_meta_fixture": run_meta,
        "python": sys.version,
        "phep_kiem": phep_kiem,
        "n_phep_kiem": {"du_kien": n_kiem_du_kien, "da_chay": len(phep_kiem),
                        "pass": sum(x["ket_qua"] == "PASS" for x in phep_kiem)},
        "loi": loi,
        "dat": dat,
        "exit": exit_code,
        "thoi_gian": {"bat_dau": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t_bat_dau)),
                      "giay_test11": round(giay_test, 2), "giay_tong": round(time.time() - t_bat_dau, 2)},
        "so_fixture_truoc_sau": {
            "fixture": {d: {k: (v if k != "giu_nguyen" else len(v)) for k, v in x.items()}
                        for d, x in ss["fixture"].items()},
            "ngoai_fixture": {k: (v if not isinstance(v, list) else v[:50]) for k, v in ss["ngoai_fixture"].items()},
            "file": ["fixture_truoc.json", "fixture_sau.json"]},
        "doi_chieu_fixture_r48c": doi_chieu_r48c(sau),
        "log_test": log_test,
    }
    (OUT / "ket_qua.json").write_text(json.dumps(ket_qua, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[chay_test11_k1] {ket_qua['n_phep_kiem']['pass']}/{len(phep_kiem)} PASS (du kien {n_kiem_du_kien}); "
          f"ngoai fixture khong doi: {ss['ngoai_fixture']['khong_doi']}; exit {exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
