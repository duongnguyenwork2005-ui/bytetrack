"""Verify the exact uploaded delta, its payloads and the changed-path roundtrip.

python check_package.py --zip review_delta_latest.zip --out package_check.json
This never claims to reconstruct the whole Git tree. Base files reconstructed
by reverse patch are checked against manifest hashes, not an independent Git
checkout. The review separately compares available preimages to prior accepted snapshots.
"""
import argparse
import hashlib
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

SHA = "b7674449fca6a7b6b0ee42d6c245601dd15b4e594ba8674a6e520c7c50d8b397"


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--zip", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data = args.zip.read_bytes()
    assert len(data) == 1607478 and hashlib.sha256(data).hexdigest() == SHA
    with tempfile.TemporaryDirectory(prefix="codex_delta39_") as td:
        root = Path(td)
        package, tree = root / "package", root / "tree"
        with zipfile.ZipFile(args.zip) as z:
            assert z.testzip() is None
            assert len(z.namelist()) == len(set(z.namelist())) == 209
            for info in z.infolist():
                p = PurePosixPath(info.filename)
                assert not p.is_absolute() and ".." not in p.parts and "\\" not in info.filename
                assert (info.external_attr >> 16) & 0o170000 != 0o120000
            z.extractall(package)
        manifest = json.loads((package / "MANIFEST.json").read_text())
        assert manifest["base_commit_short"] == "b83dc12"
        assert manifest["target_commit_short"] == "1454b3d"
        all_rows = manifest["file_trong_goi"] + manifest["file_bo_qua"]
        forbidden = []
        for row in all_rows:
            path = row["path"]
            if path in (".gitignore", "live.bat") or path.startswith(("src/demo/", "outputs/", ".git/")):
                forbidden.append(path)
            assert not path.endswith((".pt", ".engine", ".mp4", ".npz", ".zip"))
            src = package / (row.get("trong_goi") or row["trung_noi_dung_voi"])
            b = src.read_bytes()
            assert len(b) == row["bytes"] and hashlib.sha256(b).hexdigest() == row["sha256_target"], path
            dest = tree / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b)
        assert not forbidden
        permanent = args.out.parent.parent / "tree"
        permanent.mkdir(parents=True, exist_ok=True)
        for row in all_rows:
            d = permanent / row["path"]
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_bytes((tree / row["path"]).read_bytes())
        patch = next((package / "patch").glob("*.diff"))
        subprocess.run(["git", "init", "-q", str(tree)], check=True)
        reverse = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "--reverse", str(patch)],
                                 cwd=tree, capture_output=True)
        assert reverse.returncode == 0, reverse.stderr.decode(errors="replace")
        preimages = args.out.parent.parent / "base_preimages"
        base_hashes = 0
        for row in all_rows:
            p = tree / row["path"]
            if row.get("sha256_base"):
                assert hashlib.sha256(p.read_bytes()).hexdigest() == row["sha256_base"], row["path"]
                base_hashes += 1
                d = preimages / row["path"]
                d.parent.mkdir(parents=True, exist_ok=True)
                d.write_bytes(p.read_bytes())
            elif row["status"] == "them":
                assert not p.exists(), row["path"]
        forward = subprocess.run(["git", "-c", "core.autocrlf=false", "apply", str(patch)], cwd=tree, capture_output=True)
        assert forward.returncode == 0, forward.stderr.decode(errors="replace")
        for row in all_rows:
            assert hashlib.sha256((tree / row["path"]).read_bytes()).hexdigest() == row["sha256_target"], row["path"]
        result = {"zip_sha256": SHA, "bytes": len(data), "members": 209, "payloads": len(manifest["file_trong_goi"]),
                  "dedup_aliases": len(manifest["file_bo_qua"]), "roundtrip_paths": len(all_rows),
                  "reverse_base_hashes": base_hashes, "forbidden_paths": forbidden,
                  "scope": "changed paths only, not an independent full Git checkout", "pass": True}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
