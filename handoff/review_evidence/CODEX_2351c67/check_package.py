#!/usr/bin/env python3
"""Verify round-35 ZIP payloads and changed paths against saved base preimages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--base-preimages", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    assert a.zip.stat().st_size == 378096
    assert sha(a.zip) == "76e7d2addaad29fdc3a42e91fb0a7fd2851864dcf47fc233ea52dbd688b68c1d"
    with tempfile.TemporaryDirectory(prefix="codex_r35_") as td:
        pkg = Path(td) / "package"
        tree = Path(td) / "tree"
        tree.mkdir()
        with zipfile.ZipFile(a.zip) as z:
            assert z.testzip() is None
            for n in z.namelist():
                assert not Path(n).is_absolute() and ".." not in Path(n).parts
            z.extractall(pkg)
            count = len(z.namelist())
        m = json.loads((pkg / "MANIFEST.json").read_text())
        rows = m["file_trong_goi"] + m["file_bo_qua"]
        preimages = 0
        for r in rows:
            f = pkg / (r.get("trong_goi") or r["trung_noi_dung_voi"])
            assert sha(f) == r["sha256_target"] and f.stat().st_size == r["bytes"]
            name = r["path"]
            assert not name.startswith(("src/", "data/", "outputs/", ".git/"))
            assert name not in (".gitignore", "live.bat")
            if r["sha256_base"]:
                old = a.base_preimages / name
                assert sha(old) == r["sha256_base"], name
                dest = tree / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(old, dest)
                preimages += 1
        subprocess.run(["git", "init", "-q", str(tree)], check=True)
        patch = next((pkg / "patch").glob("*.diff"))
        r = subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "apply",
                            "--whitespace=nowarn", str(patch)], cwd=tree, capture_output=True, text=True)
        (a.out / "git_apply.log").write_text(r.stdout + r.stderr + "\nexit=" + str(r.returncode) + "\n")
        assert r.returncode == 0, r.stderr
        for row in rows:
            assert sha(tree / row["path"]) == row["sha256_target"], row["path"]
        ref = tree / "handoff/review_evidence/CODEX_ccd801f"
        checks = (ref / "SHA256SUMS.txt").read_text().splitlines()
        for line in checks:
            h, path = line.split(maxsplit=1)
            assert sha(ref / path) == h
    result = {"payloads":len(m["file_trong_goi"]), "aliases":len(m["file_bo_qua"]),
              "base_preimages":preimages, "patch_paths":len(rows), "prior_evidence":len(checks),
              "zip_members":count, "whole_git_tree_verified":False, "zip_sha256":sha(a.zip)}
    (a.out / "package_verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
