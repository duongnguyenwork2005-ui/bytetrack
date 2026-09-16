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
    except AssertionError:
        failed += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(ok for _, ok in _RESULTS)
    print(f"\nKET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    return 1 if failed or n_ok != len(_RESULTS) else 0


if __name__ == "__main__":
    raise SystemExit(main())
