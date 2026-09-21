"""
test_review_delta.py -- Kiem tra goi DELTA (`src/make_review_delta.py`) khong gui lai ca repository va
khong lot file khong duoc phep.

Chay tren mot REPO GIA LAP trong thu muc tam (khong dung cham repo that): tao base commit, roi mot "vong
sua" gom file code, test, log, ket qua, file rieng cua nguoi dung, mot file nhi phan lon va mot ZIP.

  1. Base doc tu `handoff/review_base.json` va KHONG tu dong chay theo HEAD sau khi dung goi.
  2. File rieng cua nguoi dung (`.gitignore`, `src/demo/**`, `live.bat`, `outputs/**`) KHONG vao goi.
  3. Du lieu lon / nhi phan (`*.npz`, `*.pt`, `*.zip`, `*.diff`, `models/`, `data/raw/`) KHONG vao goi.
  4. File code/test/config them hoac sua trong vong nay CO trong goi, ban day du tai target.
  5. File KHONG doi giua base va target KHONG vao goi.
  6. MANIFEST ghi du base/target + them/sua/xoa + sha256; hash khop noi dung thuc trong ZIP.
  7. Patch trong goi ap duoc len DUNG cay base va cho ra noi dung target.
  8. File bi bo qua van duoc liet ke (kem ly do) trong MANIFEST va DEPENDENCIES.md.
  9. Thieu REVIEW_BRIEF.md -> tu choi dung goi.

Ba phep kiem hoi quy cho hai loi packager (review vong 3), moi phep dung REPO TAM RIENG:
 10. P1 - file rieng bi `git rm --cached` (bo theo doi, GIU tren o dia): goi KHONG chua diff hay
     marker noi dung cua file do; file lam viec van con nguyen.
 11. P2a - doi ten file code: ap patch len base phai cho ra DUNG target (ten moi co, ten cu BIEN MAT).
 12. P2b - hai file code moi TRUNG noi dung: ap patch len base phai co DU ca hai.
Hai phep 11-12 so TOAN BO duong dan code/test/config voi target, khong chi exit code hay mot file.
 13. --support: file KHONG doi so voi base duoc gui kem o `support/` voi status `ho_tro` (hash base == target,
     khong vao patch, khong tinh vao so file thay doi, co muc rieng trong DEPENDENCIES.md); file CO doi -> tu choi.

    python src/test_review_delta.py
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines()[-6:]:
            print(f"         {line}")
    assert ok, name


def run(cmd, cwd):
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", env=env)


def write(base: Path, rel: str, text: str) -> None:
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def make_repo(d: Path) -> str:
    """Repo gia lap: base commit + mot vong sua. Tra ve base commit."""
    run(["git", "init", "-q", "-b", "main"], d)
    run(["git", "config", "user.email", "t@t"], d)
    run(["git", "config", "user.name", "t"], d)
    (d / "src").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "src" / "make_review_delta.py", d / "src" / "make_review_delta.py")

    # ----- trang thai BASE -----
    write(d, "src/tracker.py", "x = 1\n")
    write(d, "src/khong_doi.py", "GIU NGUYEN\n")
    write(d, "configs/exp.yaml", "conf: 0.25\n")
    write(d, ".gitignore", "outputs/\n")
    write(d, "src/demo/live_view.py", "# cua nguoi dung\n")
    write(d, "handoff/logs/cu.log", "log cu\n")
    write(d, "results/phase2/cu.csv", "a,b\n1,2\n")
    run(["git", "add", "-A"], d)
    run(["git", "commit", "-qm", "base"], d)
    base = run(["git", "rev-parse", "HEAD"], d).stdout.strip()

    # ----- mot VONG SUA -----
    write(d, "src/tracker.py", "x = 2\nnew = True\n")                    # sua code
    write(d, "src/test_moi.py", "assert True\n")                         # them test
    write(d, "configs/exp.yaml", "conf: 0.30\n")                         # sua config
    write(d, "handoff/logs/moi.log", "log moi\nexit=0\n")                # log moi
    write(d, "results/phase2/moi.csv", "metric,value\nHOTA,0.58\n")      # ket qua moi
    write(d, "src/demo/buffer90_moi.py", "# cua nguoi dung\n")           # file rieng
    write(d, "live.bat", "@echo off\n")                                  # file rieng
    write(d, ".gitignore", "outputs/\n.venv/\n")                         # file rieng (bi SUA)
    (d / "models").mkdir(exist_ok=True)
    (d / "models" / "w.pt").write_bytes(b"\x00" * 2048)                  # weights
    (d / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (d / "data" / "interim" / "cache.npz").write_bytes(b"\x01" * 4096)   # cache
    (d / "handoff" / "phase2_review_latest.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    write(d, "handoff/patches/moc_full.diff", "diff --git a/x b/x\n" * 50)
    run(["git", "add", "-A", "-f"], d)
    run(["git", "commit", "-qm", "vong sua"], d)
    return base


def dung_goi(d: Path, base: str) -> zipfile.ZipFile:
    """Dat base roi dung goi delta trong repo tam `d`; tra ve ZipFile da mo."""
    (d / "handoff").mkdir(parents=True, exist_ok=True)   # fixture tu tao, khong phu thuoc packager
    run([PY, "src/make_review_delta.py", "--set-base", base, "--round", "1",
         "--base-snapshot", "handoff/archive/base.zip"], d)
    write(d, "handoff/REVIEW_BRIEF.md", "# brief\n")
    r = run([PY, "src/make_review_delta.py"], d)
    assert r.returncode == 0, r.stdout + r.stderr
    return zipfile.ZipFile(d / "handoff" / "review_delta_latest.zip")


def cay_code_sau_khi_ap(d: Path, base: str, patch: bytes) -> dict[str, str]:
    """Clone repo tam ve base, ap patch, tra ve {duong dan code/test/config: noi dung}."""
    w = Path(tempfile.mkdtemp(prefix="delta_apply_"))
    run(["git", "clone", "-q", str(d), str(w / "r")], w)
    r = w / "r"
    run(["git", "checkout", "-q", base], r)
    (r / "p.diff").write_bytes(patch)
    ap = run(["git", "apply", "p.diff"], r)
    assert ap.returncode == 0, ap.stdout + ap.stderr
    out = {}
    for q in sorted(r.rglob("*")):
        rel = q.relative_to(r).as_posix()
        if q.is_file() and (rel.startswith(("src/", "configs/")) and "make_review_delta" not in rel):
            out[rel] = q.read_text(encoding="utf-8")
    shutil.rmtree(w, ignore_errors=True)
    return out


def cay_code_tai(d: Path, commit: str) -> dict[str, str]:
    """{duong dan code/test/config: noi dung} tai mot commit cua repo tam."""
    ls = run(["git", "ls-tree", "-r", "--name-only", commit], d).stdout.split()
    out = {}
    for rel in ls:
        if rel.startswith(("src/", "configs/")) and "make_review_delta" not in rel:
            out[rel] = run(["git", "show", f"{commit}:{rel}"], d).stdout
    return out


def test_p1_file_rieng_bi_bo_theo_doi() -> None:
    """P1: `git rm --cached` file rieng -> noi dung KHONG duoc lot vao patch."""
    d = Path(tempfile.mkdtemp(prefix="delta_p1_"))
    try:
        MARKER = "MARKER_NOI_DUNG_RIENG_KHONG_DUOC_RO_RI"
        run(["git", "init", "-q", "-b", "main"], d)
        run(["git", "config", "user.email", "t@t"], d)
        run(["git", "config", "user.name", "t"], d)
        (d / "src").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "src" / "make_review_delta.py", d / "src" / "make_review_delta.py")
        write(d, "src/tracker.py", "x = 1\n")
        write(d, "src/demo/private.py", f"# {MARKER}\nbi_mat = 1\n")
        run(["git", "add", "-A", "-f"], d)
        run(["git", "commit", "-qm", "base"], d)
        base = run(["git", "rev-parse", "HEAD"], d).stdout.strip()

        run(["git", "rm", "--cached", "-q", "src/demo/private.py"], d)   # bo theo doi, GIU tren o dia
        write(d, "src/tracker.py", "x = 2\n")
        run(["git", "add", "src/tracker.py"], d)
        run(["git", "commit", "-qm", "bo theo doi file rieng"], d)

        with dung_goi(d, base) as z:
            names = z.namelist()
            patch = z.read(next(n for n in names if n.startswith("patch/"))).decode("utf-8", "replace")
            toan_bo = b"".join(z.read(n) for n in names).decode("utf-8", "replace")
        con_tren_dia = (d / "src" / "demo" / "private.py").read_text(encoding="utf-8")
        _check("P1: file rieng bi `git rm --cached` -> KHONG co diff/marker trong goi, file van con tren dia",
               MARKER not in toan_bo and "src/demo/private.py" not in patch
               and MARKER in con_tren_dia,
               f"marker trong goi={MARKER in toan_bo}; duong dan trong patch="
               f"{'src/demo/private.py' in patch}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_p2a_doi_ten() -> None:
    """P2a: doi ten file code -> ap patch phai cho DUNG target (ten cu BIEN MAT)."""
    d = Path(tempfile.mkdtemp(prefix="delta_p2a_"))
    try:
        run(["git", "init", "-q", "-b", "main"], d)
        run(["git", "config", "user.email", "t@t"], d)
        run(["git", "config", "user.name", "t"], d)
        (d / "src").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "src" / "make_review_delta.py", d / "src" / "make_review_delta.py")
        write(d, "src/old_util.py", "def f():\n    return 1\n")
        write(d, "configs/exp.yaml", "conf: 0.25\n")
        run(["git", "add", "-A"], d)
        run(["git", "commit", "-qm", "base"], d)
        base = run(["git", "rev-parse", "HEAD"], d).stdout.strip()

        run(["git", "mv", "src/old_util.py", "src/new_util.py"], d)
        run(["git", "commit", "-qm", "doi ten"], d)
        target = run(["git", "rev-parse", "HEAD"], d).stdout.strip()

        with dung_goi(d, base) as z:
            patch = z.read(next(n for n in z.namelist() if n.startswith("patch/")))
        sau = cay_code_sau_khi_ap(d, base, patch)
        mong = cay_code_tai(d, target)
        _check("P2a: doi ten -> ap patch cho DUNG target (ten moi co, ten cu bien mat)",
               sau == mong,
               f"thua={sorted(set(sau) - set(mong))} thieu={sorted(set(mong) - set(sau))}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_p2b_hai_file_trung_noi_dung() -> None:
    """P2b: hai file code moi trung noi dung -> ap patch phai co DU ca hai."""
    d = Path(tempfile.mkdtemp(prefix="delta_p2b_"))
    try:
        run(["git", "init", "-q", "-b", "main"], d)
        run(["git", "config", "user.email", "t@t"], d)
        run(["git", "config", "user.name", "t"], d)
        (d / "src").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "src" / "make_review_delta.py", d / "src" / "make_review_delta.py")
        write(d, "src/goc.py", "x = 0\n")
        run(["git", "add", "-A"], d)
        run(["git", "commit", "-qm", "base"], d)
        base = run(["git", "rev-parse", "HEAD"], d).stdout.strip()

        GIONG = "def chung():\n    return 42\n"
        write(d, "src/one.py", GIONG)
        write(d, "src/two.py", GIONG)                      # TRUNG noi dung voi one.py
        write(d, "handoff/logs/a.log", "cung log\n")
        write(d, "handoff/logs/b.log", "cung log\n")      # log trung -> duoc phep dedup payload
        run(["git", "add", "-A"], d)
        run(["git", "commit", "-qm", "them hai file trung noi dung"], d)
        target = run(["git", "rev-parse", "HEAD"], d).stdout.strip()

        with dung_goi(d, base) as z:
            names = z.namelist()
            patch = z.read(next(n for n in names if n.startswith("patch/")))
        sau = cay_code_sau_khi_ap(d, base, patch)
        mong = cay_code_tai(d, target)
        ca_hai_payload = {"files/src/one.py", "files/src/two.py"} <= set(names)
        log_dedup = sum(1 for n in names if n.startswith("logs/")) == 1
        _check("P2b: hai file code trung noi dung -> ap patch co DU ca hai; dedup chi ap cho log",
               sau == mong and ca_hai_payload and log_dedup,
               f"thieu sau khi ap={sorted(set(mong) - set(sau))}; "
               f"ca hai trong files/={ca_hai_payload}; so log trong goi="
               f"{sum(1 for n in names if n.startswith('logs/'))}")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    failed = 0
    tmp = Path(tempfile.mkdtemp(prefix="delta_test_"))
    try:
        base = make_repo(tmp)
        target = run(["git", "rev-parse", "HEAD"], tmp).stdout.strip()

        # --- 9. thieu REVIEW_BRIEF.md -> tu choi ---
        run([PY, "src/make_review_delta.py", "--set-base", base, "--round", "1",
             "--base-snapshot", "handoff/archive/base.zip"], tmp)
        r = run([PY, "src/make_review_delta.py"], tmp)
        _check("thieu REVIEW_BRIEF.md -> TU CHOI dung goi",
               r.returncode != 0 and "REVIEW_BRIEF" in (r.stdout + r.stderr), r.stdout + r.stderr)

        write(tmp, "handoff/REVIEW_BRIEF.md", "# brief\nbase->target\n")
        r = run([PY, "src/make_review_delta.py"], tmp)
        z_path = tmp / "handoff" / "review_delta_latest.zip"
        _check("dung goi delta thanh cong", r.returncode == 0 and z_path.exists(), r.stdout + r.stderr)

        with zipfile.ZipFile(z_path) as z:
            names = set(z.namelist())
            mj = json.loads(z.read("MANIFEST.json"))
            rows = list(csv.DictReader(io.StringIO(z.read("MANIFEST.csv").decode("utf-8"))))
            deps = z.read("DEPENDENCIES.md").decode("utf-8")
            patch_name = next(n for n in names if n.startswith("patch/"))
            patch_bytes = z.read(patch_name)
            noi_dung = {n: z.read(n) for n in names}

        # --- 1. base doc tu review_base.json, khong chay theo HEAD ---
        sau = json.loads((tmp / "handoff" / "review_base.json").read_text(encoding="utf-8"))
        _check("base lay tu review_base.json va KHONG tu chuyen sang HEAD sau khi dung goi",
               mj["base_commit"] == base and sau["base_commit"] == base and base != target,
               f"base={base[:7]} target={target[:7]} trong file={sau['base_commit'][:7]}")

        # --- 2. file rieng cua nguoi dung ---
        rieng = ["src/demo/live_view.py", "src/demo/buffer90_moi.py", "live.bat", ".gitignore"]
        lot = [p for p in rieng if any(p in n for n in names)]
        _check("file rieng cua nguoi dung KHONG co trong goi", not lot, lot)

        # --- 3. du lieu lon / nhi phan ---
        lon = [n for n in names if n.endswith((".npz", ".pt", ".zip")) or "models/" in n
               or n.endswith("moc_full.diff")]
        _check("du lieu lon / nhi phan / patch toan bo lich su KHONG co trong goi", not lon, lon)

        # --- 4. code/test/config duoc them hoac sua CO trong goi, ban day du tai target ---
        co = {"files/src/tracker.py", "files/src/test_moi.py", "files/configs/exp.yaml",
              "logs/handoff/logs/moi.log", "data/results/phase2/moi.csv"}
        thieu = co - names
        dung_target = noi_dung.get("files/src/tracker.py", b"") == b"x = 2\nnew = True\n"
        _check("code/test/config/log/ket qua moi CO trong goi, dung ban target",
               not thieu and dung_target, f"thieu={thieu}")

        # --- 5. file KHONG doi -> khong vao goi ---
        khong_doi = [n for n in names if "khong_doi" in n or "cu.log" in n or "cu.csv" in n]
        _check("file KHONG doi giua base va target KHONG vao goi", not khong_doi, khong_doi)

        # --- 6. MANIFEST day du + hash khop ---
        lech = [r_["trong_goi"] for r_ in rows if r_["trong_goi"]
                and hashlib.sha256(noi_dung[r_["trong_goi"]]).hexdigest() != r_["sha256_target"]]
        co_status = {r_["status"] for r_ in rows}
        _check("MANIFEST ghi du base/target + them/sua + sha256 khop noi dung thuc trong ZIP",
               not lech and mj["target_commit"] == target and {"them", "sua"} <= co_status,
               f"lech={lech} status={sorted(co_status)}")

        # --- 7. patch ap len DUNG cay base -> ra target ---
        w = Path(tempfile.mkdtemp(prefix="delta_apply_"))
        run(["git", "clone", "-q", str(tmp), str(w)], w.parent)
        run(["git", "checkout", "-q", base], w)
        (w / "p.diff").write_bytes(patch_bytes)
        ap = run(["git", "apply", "p.diff"], w)
        sau_ap = (w / "src" / "tracker.py").read_text(encoding="utf-8")
        _check("patch trong goi ap duoc len DUNG cay base va cho ra noi dung target",
               ap.returncode == 0 and sau_ap == "x = 2\nnew = True\n", ap.stdout + ap.stderr)
        shutil.rmtree(w, ignore_errors=True)

        # --- 8. file bo qua van duoc liet ke kem ly do ---
        bo = {r_["path"]: r_["ly_do_khong_gom"] for r_ in rows if not r_["trong_goi"]}
        # (src/demo/live_view.py KHONG doi trong vong nay -> khong nam trong diff, dung nhu ky vong)
        _check("file bi bo qua van duoc liet ke kem ly do trong MANIFEST va DEPENDENCIES.md",
               {"live.bat", ".gitignore", "src/demo/buffer90_moi.py", "models/w.pt",
                "data/interim/cache.npz"} <= set(bo)
               and all(bo[p] for p in bo) and "live.bat" in deps,
               "; ".join(f"{k} -> {v[:26]}" for k, v in sorted(bo.items())))

        # --- 10-12. hoi quy hai loi packager (review vong 3) ---
        test_p1_file_rieng_bi_bo_theo_doi()
        test_p2a_doi_ten()
        test_p2b_hai_file_trung_noi_dung()

        # --- 13. --support: file khong doi gui kem, khong gia ghi thanh thay doi ---
        r = run([PY, "src/make_review_delta.py", "--support", "src/khong_doi.py"], tmp)
        with zipfile.ZipFile(z_path) as z:
            names2 = set(z.namelist())
            mj2 = json.loads(z.read("MANIFEST.json"))
            rows2 = list(csv.DictReader(io.StringIO(z.read("MANIFEST.csv").decode("utf-8"))))
            deps2 = z.read("DEPENDENCIES.md").decode("utf-8")
            noi_dung = z.read("support/src/khong_doi.py").decode("utf-8") if "support/src/khong_doi.py" in names2 else ""
            patch2 = z.read(next(n for n in names2 if n.startswith("patch/"))).decode("utf-8")
        row = next((x for x in rows2 if x["path"] == "src/khong_doi.py"), None)
        r_doi = run([PY, "src/make_review_delta.py", "--support", "src/tracker.py"], tmp)   # file CO doi
        _check("--support file KHONG doi: nam o support/, status ho_tro, hash base == target, KHONG trong patch, "
               "khong tinh vao so_file_trong_goi, co muc rieng trong DEPENDENCIES.md; file CO doi -> TU CHOI",
               r.returncode == 0 and noi_dung == "GIU NGUYEN\n" and row is not None and row["status"] == "ho_tro"
               and row["sha256_base"] == row["sha256_target"] != "" and row["trong_patch"] == "False"
               and "src/khong_doi.py" not in patch2 and mj2["so_file_ho_tro"] == 1
               and mj2["so_file_trong_goi"] == mj["so_file_trong_goi"] and "ho tro review" in deps2
               and r_doi.returncode != 0 and "CO doi so voi base" in (r_doi.stdout + r_doi.stderr),
               f"rc={r.returncode} row={row} rc_doi={r_doi.returncode} {(r_doi.stdout + r_doi.stderr)[-120:]}")
    except AssertionError:
        failed += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
