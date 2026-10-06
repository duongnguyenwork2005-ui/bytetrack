# -*- coding: utf-8 -*-
"""Vong 44 (R43-B1): kiem GOI / CAY SACH NGOAI repo -- test chay bang MA THAT trong cay, KHONG import tu repo goc.

Dung cay trong thu muc MOI (--thu-muc, ngoai repo; da ton tai -> tu choi):
  --zip GOI.zip                 giai nen goi delta; moi file cua cay (payload `files/`, ban trung noi dung, `support/`)
                                phai co sha256 == `sha256_target` trong MANIFEST.json; cay = <thu-muc>/cay/<duong dan repo>.
  --tu-git COMMIT --duong-dan P...   cay = blob Git cua cac duong dan (dung bytes packager se gui);
                                --thay P=FILE thay mot duong dan bang file cuc bo (vd. entry point CU -> chung minh
                                regression bat ma cu).
Moi lenh kiem (`--chay 'src/x.py [tham so]'`, lap lai duoc) chay bang `python <cay>/src/x.py ...` voi:
  cwd = <thu-muc> (ngoai repo; `python script.py` khong them cwd vao sys.path);
  moi truong: BO PYTHONPATH / PYTHONHOME / PYTHONSTARTUP / PYTHONUSERBASE cua nguoi dung, PYTHONNOUSERSITE=1,
  PYTHONDONTWRITEBYTECODE=1; PYTHONPATH CHI = <thu-muc>/shim (chua DUY NHAT sitecustomize.py sinh ra o day).
  sitecustomize ghi lai, cho MOI tien trinh Python (ke ca tien trinh con cua test), file cua moi module da nap:
  module DU AN (file trong cay hoac trong repo goc) phai NAM TRONG CAY; mot module tu repo goc -> KHONG DAT.
  --chan-pyarrow : sitecustomize lam `import pyarrow` THAT BAI co chu y (Smart App Control tren may nay chan DLL
                   pyarrow._compute -> `import pandas` loi). pandas chay nhu khi khong cai pyarrow; ma du an khong dung
                   pyarrow; KHONG thay module du an nao.
  --models-dir D : dat DET5_MODELS_DIR=D (weights NGOAI goi, chi cho test_detector_5lop); ghi sha256 moi .pt trong D.
Sau khi chay: kiem lai sha256 moi file cua cay (test khong duoc sua ma). Dem [PASS] / [FAIL] / [SKIP] tu stdout.
Ghi --out (JSON) + in tom tat. Exit 0: moi lenh exit 0 + kiem nguon module dat + cay khong doi; 1: khac; 2: loi dung cay."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata as im
import json
import os
import re
import shlex
import subprocess
import sys
import time
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]

SITECUSTOMIZE = r'''# Sinh boi handoff/logs/review44/kiem_cay_sach.py -- CHI cho tien trinh kiem; KHONG thuoc ma du an.
import atexit
import json
import os
import sys

if os.environ.get("KIEM_CHAN_PYARROW") == "1":
    import importlib.abc

    class _ChanPyarrow(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "pyarrow" or name.startswith("pyarrow."):
                raise ImportError("pyarrow bi chan co chu y trong tien trinh kiem (kiem_cay_sach.py --chan-pyarrow)")
            return None

    sys.meta_path.insert(0, _ChanPyarrow())


def _ghi_nguon():
    log = os.environ.get("KIEM_NGUON_LOG")
    if not log:
        return
    cay = os.path.normcase(os.path.realpath(os.environ["KIEM_CAY"]))
    repo = os.path.normcase(os.path.realpath(os.environ["KIEM_REPO"]))
    du_an, ngoai = {}, set()
    for ten, m in list(sys.modules.items()):
        f = getattr(m, "__file__", None)
        if not f:
            continue
        rf = os.path.normcase(os.path.realpath(f))
        if rf.startswith(cay + os.sep) or rf.startswith(repo + os.sep):
            du_an[ten] = rf
        elif "site-packages" in rf or "dist-packages" in rf:
            ngoai.add(ten.split(".")[0])
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"pid": os.getpid(), "argv": [str(x) for x in sys.argv[:4]], "du_an": du_an,
                             "ngoai": sorted(ngoai)}) + "\n")


atexit.register(_ghi_nguon)
'''


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for k in iter(lambda: f.read(1 << 20), b""):
            h.update(k)
    return h.hexdigest()


def git_blob(commit: str, path: str) -> bytes:
    return subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"], capture_output=True, check=True).stdout


def dung_tu_zip(zp: Path, thu_muc: Path) -> dict:
    cay = thu_muc / "cay"
    z = zipfile.ZipFile(zp)
    if z.testzip() is not None:
        raise SystemExit("ZIP hong")
    mf = json.loads(z.read("MANIFEST.json"))
    ten = set(z.namelist())
    ghi, lech, bo = [], [], []
    hang = [(r, r["trong_goi"]) for r in mf["file_trong_goi"] + mf.get("file_ho_tro", [])]
    hang += [(r, r["trung_noi_dung_voi"]) for r in mf["file_bo_qua"] if r.get("trung_noi_dung_voi")]
    for r, thanh_vien in hang:
        if thanh_vien not in ten:
            lech.append(f"{r['path']}: thieu thanh vien {thanh_vien}")
            continue
        b = z.read(thanh_vien)
        if sha(b) != r["sha256_target"]:
            lech.append(f"{r['path']}: sha256 {sha(b)[:12]} != MANIFEST {r['sha256_target'][:12]}")
            continue
        p = cay / r["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
        ghi.append({"path": r["path"], "tu": thanh_vien, "sha256": r["sha256_target"],
                    "loai": "ho_tro" if r.get("status") == "ho_tro" else "thay_doi"})
    bo = [r["path"] for r in mf["file_bo_qua"] if not r.get("trung_noi_dung_voi")]
    return {"che_do": "zip", "zip": str(zp), "zip_bytes": zp.stat().st_size, "zip_sha256": sha_file(zp),
            "base": mf["base_commit"], "target": mf["target_commit"], "so_file_cay": len(ghi), "lech": lech,
            "file_cay": ghi, "file_bi_loai_khong_vao_cay": bo}


def dung_tu_git(commit: str, duong_dan: list[str], thay: list[str], thu_muc: Path) -> dict:
    cay = thu_muc / "cay"
    ghi = []
    thay_map = dict(t.split("=", 1) for t in thay)
    for p in duong_dan:
        if p in thay_map:
            b = Path(thay_map[p]).read_bytes()
            nguon = f"file cuc bo {thay_map[p]}"
        else:
            b = git_blob(commit, p)
            nguon = f"git {commit}:{p}"
        (cay / p).parent.mkdir(parents=True, exist_ok=True)
        (cay / p).write_bytes(b)
        ghi.append({"path": p, "tu": nguon, "sha256": sha(b)})
    return {"che_do": "git", "commit": commit, "so_file_cay": len(ghi), "lech": [], "file_cay": ghi}


def moi_truong(thu_muc: Path, cay: Path, log_nguon: Path, chan_pyarrow: bool, models_dir: str | None) -> dict:
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONSAFEPATH",
                        "PYTHONINSPECT", "DET5_MODELS_DIR")}
    env.update({"PYTHONPATH": str(thu_muc / "shim"), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONIOENCODING": "utf-8", "KIEM_CAY": str(cay), "KIEM_REPO": str(REPO),
                "KIEM_NGUON_LOG": str(log_nguon), "KIEM_CHAN_PYARROW": "1" if chan_pyarrow else "0"})
    if models_dir:
        env["DET5_MODELS_DIR"] = models_dir
    return env


def chay_mot(py: str, lenh: str, thu_muc: Path, cay: Path, env: dict, so: int) -> dict:
    parts = shlex.split(lenh)
    log_nguon = Path(env["KIEM_NGUON_LOG"])
    if log_nguon.exists():
        log_nguon.unlink()
    argv = [py, str(cay / parts[0]), *parts[1:]]
    t0 = time.time()
    r = subprocess.run(argv, cwd=thu_muc, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tg = round(time.time() - t0, 1)
    (thu_muc / f"stdout_{so:02d}.txt").write_text(r.stdout + "\n----- STDERR -----\n" + r.stderr, encoding="utf-8")
    dem = {k: len(re.findall(rf"^\s*\[{k}\]", r.stdout, re.M)) for k in ("PASS", "FAIL", "SKIP")}
    ket_qua = [d for d in r.stdout.splitlines() if d.startswith("KET QUA")]
    ban_ghi = [json.loads(d) for d in log_nguon.read_text(encoding="utf-8").splitlines()] if log_nguon.exists() else []
    cay_n, repo_n = os.path.normcase(str(cay.resolve())), os.path.normcase(str(REPO.resolve()))
    du_an, tu_repo, ngoai = {}, [], set()
    for bg in ban_ghi:
        ngoai.update(bg["ngoai"])
        for ten, f in bg["du_an"].items():
            if f.startswith(repo_n + os.sep):
                tu_repo.append(f"{ten} <- {f} (pid {bg['pid']})")
            elif f.startswith(cay_n + os.sep):
                rel = os.path.relpath(f, cay_n).replace("\\", "/")
                du_an[rel] = sha_file(Path(f))
    return {"lenh": lenh, "argv": argv, "cwd": str(thu_muc), "exit": r.returncode, "thoi_gian_s": tg, "dem": dem,
            "dong_ket_qua": ket_qua, "so_tien_trinh_python_ghi_nguon": len(ban_ghi),
            "module_du_an_tu_cay": dict(sorted(du_an.items())), "module_tu_repo_goc": tu_repo,
            "goi_ngoai_da_nap": sorted(ngoai), "stdout": f"stdout_{so:02d}.txt",
            "nguon_dat": bool(ban_ghi) and not tu_repo}


def phien_ban_goi(ten_ds: set[str]) -> dict:
    pd = im.packages_distributions()
    ra = {}
    for t in sorted(ten_ds):
        for d in pd.get(t, []):
            try:
                ra[t] = f"{d} {im.version(d)}"
            except im.PackageNotFoundError:
                pass
    return ra


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thu-muc", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--zip")
    g.add_argument("--tu-git", metavar="COMMIT")
    ap.add_argument("--duong-dan", nargs="*", default=[])
    ap.add_argument("--thay", nargs="*", default=[], metavar="P=FILE")
    ap.add_argument("--chay", action="append", required=True, metavar="'src/x.py [tham so]'")
    ap.add_argument("--chan-pyarrow", action="store_true")
    ap.add_argument("--models-dir")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    thu_muc = Path(a.thu_muc).resolve()
    if thu_muc.exists():
        print(f"[LOI] {thu_muc} da ton tai -- can thu muc MOI")
        return 2
    if os.path.normcase(str(thu_muc)).startswith(os.path.normcase(str(REPO.resolve())) + os.sep):
        print(f"[LOI] {thu_muc} nam TRONG repo -- can thu muc ngoai repo")
        return 2
    thu_muc.mkdir(parents=True)
    try:
        cay_info = dung_tu_zip(Path(a.zip), thu_muc) if a.zip else dung_tu_git(a.tu_git, a.duong_dan, a.thay, thu_muc)
    except (subprocess.CalledProcessError, KeyError, OSError) as e:
        print(f"[LOI DUNG CAY] {e}")
        return 2
    if cay_info["lech"]:
        print(f"[LOI DUNG CAY] {cay_info['lech']}")
        return 2
    cay = thu_muc / "cay"
    (thu_muc / "shim").mkdir()
    (thu_muc / "shim" / "sitecustomize.py").write_text(SITECUSTOMIZE, encoding="utf-8")
    truoc = {p.relative_to(cay).as_posix(): sha_file(p) for p in cay.rglob("*") if p.is_file()}
    weights = ({p.name: sha_file(p) for p in sorted(Path(a.models_dir).rglob("*.pt"))} if a.models_dir else {})
    env = moi_truong(thu_muc, cay, thu_muc / "nguon_module.jsonl", a.chan_pyarrow, a.models_dir)
    ket = [chay_mot(a.python, lenh, thu_muc, cay, env, i + 1) for i, lenh in enumerate(a.chay)]
    sau = {p.relative_to(cay).as_posix(): sha_file(p) for p in cay.rglob("*") if p.is_file()}
    doi = sorted(k for k in truoc if sau.get(k) != truoc[k])
    moi = sorted(k for k in sau if k not in truoc)
    ngoai = set().union(*(set(k["goi_ngoai_da_nap"]) for k in ket)) if ket else set()
    tong = {"tao_luc": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
            "python": a.python, "phien_ban_python": sys.version.split()[0], "nen_tang": sys.platform,
            "thu_muc": str(thu_muc), "repo_goc": str(REPO), "cay": cay_info, "chan_pyarrow": a.chan_pyarrow,
            "models_dir": a.models_dir, "weights_ngoai_goi": weights,
            "goi_ngoai_phien_ban": phien_ban_goi(ngoai), "lenh": ket,
            "cay_sau_khi_chay": {"file_doi": doi, "file_moi": moi},
            "dat": all(k["exit"] == 0 and k["nguon_dat"] for k in ket) and not doi}
    Path(a.out).write_text(json.dumps(tong, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"cay: {cay_info['che_do']} {cay_info.get('zip_sha256', cay_info.get('commit', ''))[:12]} -> "
          f"{cay_info['so_file_cay']} file trong {cay}")
    for k in ket:
        print(f"[{'DAT' if k['exit'] == 0 and k['nguon_dat'] else 'KHONG DAT'}] {k['lenh']}: exit {k['exit']}, "
              f"PASS {k['dem']['PASS']} FAIL {k['dem']['FAIL']} SKIP {k['dem']['SKIP']}, {k['thoi_gian_s']} s; "
              f"{len(k['module_du_an_tu_cay'])} module du an tu cay, {len(k['module_tu_repo_goc'])} tu repo goc, "
              f"{k['so_tien_trinh_python_ghi_nguon']} tien trinh ghi nguon; {' | '.join(k['dong_ket_qua'])}")
    print(f"goi ngoai da nap: {tong['goi_ngoai_phien_ban']}")
    print(f"cay sau khi chay: {len(doi)} file doi {doi[:5]}, {len(moi)} file moi {moi[:5]}")
    print("KET LUAN:", "DAT" if tong["dat"] else "KHONG DAT")
    return 0 if tong["dat"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
