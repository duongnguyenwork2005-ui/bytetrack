#!/usr/bin/env python3
"""Independent verification of round-34 delta and imported-label receipts.

Does not read the Windows dataset or assert that a complete Git tree was supplied.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--zip", type=Path, required=True)
    p.add_argument("--base-tree", type=Path, required=True)
    p.add_argument("--old-manifest", type=Path, required=True)
    p.add_argument("--old-native-hashes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    m = read_json(a.package / "MANIFEST.json")
    expected_zip = "4880e54fc98c0de6749d81294c2e93c5bf716718b07cac2e3131b58e147dfd39"
    assert a.zip.stat().st_size == 805621 and digest(a.zip) == expected_zip
    with zipfile.ZipFile(a.zip) as z:
        assert z.testzip() is None
        member_count = len(z.namelist())
        for name in z.namelist():
            assert not Path(name).is_absolute() and ".." not in Path(name).parts
    records = m["file_trong_goi"] + m["file_bo_qua"]
    assert len(records) == 77 and len({r["path"] for r in records}) == 77
    for r in records:
        payload = a.package / (r.get("trong_goi") or r["trung_noi_dung_voi"])
        assert digest(payload) == r["sha256_target"], r["path"]
        assert payload.stat().st_size == r["bytes"], r["path"]
        name = r["path"]
        assert not name.startswith(("src/", "configs/", "data/", "outputs/", ".git/")), name
        assert name not in (".gitignore", "live.bat"), name
        assert Path(name).suffix.lower() not in (".zip", ".npz", ".pt", ".engine", ".jpg", ".mp4"), name
    with tempfile.TemporaryDirectory(prefix="codex_ccd_patch_") as td:
        tree = Path(td)
        preimages = 0
        for r in records:
            if r["sha256_base"]:
                old = a.base_tree / r["path"]
                assert digest(old) == r["sha256_base"], r["path"]
                dest = tree / r["path"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(old, dest)
                preimages += 1
        subprocess.run(["git", "init", "-q", str(tree)], check=True)
        patch = next((a.package / "patch").glob("*.diff")).resolve()
        command = ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf", "apply", "--whitespace=nowarn", str(patch)]
        result = subprocess.run(command, cwd=tree, capture_output=True, text=True)
        (a.out / "git_apply.log").write_text(result.stdout + result.stderr + "\nexit=" + str(result.returncode) + "\n")
        assert result.returncode == 0
        for r in records:
            assert digest(tree / r["path"]) == r["sha256_target"], r["path"]
        evidence = tree / "handoff/review_evidence/CODEX_4e71d32"
        sums = (evidence / "SHA256SUMS.txt").read_text().splitlines()
        for line in sums:
            sha, path = line.split(maxsplit=1)
            assert digest(evidence / path.lstrip("*")) == sha, path
        assert len(sums) == 56
    package_result = dict(zip_bytes=a.zip.stat().st_size, zip_sha256=expected_zip,
                          zip_members=member_count, payloads_verified=len(m["file_trong_goi"]),
                          aliases_verified=len(m["file_bo_qua"]), base_preimages_verified=preimages,
                          patch_target_paths_verified=77, carried_review_evidence_verified=56,
                          active_src_or_config_changes=0, complete_git_tree_verified=False)
    (a.out / "package_verification.json").write_text(json.dumps(package_result, indent=2) + "\n")

    logs = a.package / "logs/handoff/logs/review34"
    old_sha = "a2e416f26d2ba44a50c7f3ec8578deec39f364cd2381c36dc12fe45a905fcf39"
    assert digest(a.old_manifest) == old_sha
    with a.old_manifest.open(encoding="utf-8", newline="") as f:
        old_rows = list(csv.DictReader(f, delimiter="\t"))
    native_prefix = "bdd100k/labels/box_track_20/train/"
    accepted_rows = {r["duong_dan"]: r for r in old_rows if r["duong_dan"].startswith(native_prefix)}
    independent_rows = {r["path"]: r for r in read_json(a.old_native_hashes) if r["path"].startswith(native_prefix)}
    receipt = logs / "03_manifest_nhap_nhan_train.tsv"
    with receipt.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert len(rows) == len(accepted_rows) == len(independent_rows) == 1400
    seen = set()
    for r in rows:
        dest = r["duong_dan_dich"]
        prefix = "data/raw/bdd100k/labels/box_track_20/train/"
        assert dest.startswith(prefix) and dest.endswith(".json")
        filename = dest[len(prefix):]
        assert "/" not in filename and "\\" not in filename and filename not in seen
        seen.add(filename)
        key = native_prefix + filename
        source_prefix = "data/raw/ung_vien_nhan/hf_jaffe03195_bdd100k_sot/"
        assert r["nguon_cach_ly"] == source_prefix + key
        assert r["sha256"] == accepted_rows[key]["sha256"] == independent_rows[key]["sha256"]
        assert int(r["byte"]) == int(accepted_rows[key]["byte"]) == independent_rows[key]["bytes"]
    before = read_json(logs / "02_kiem_truoc_nhap.json")
    after = read_json(logs / "03_nhap_nhan_train.json")
    assert before["exit"] == after["exit"] == 0
    assert before["manifest"]["sha256"] == after["manifest"]["sha256"] == old_sha
    assert before["kiem_dich_truoc"]["dich_ton_tai"] is False
    assert after["nhap"] == {"so_ghi_moi": 1400, "so_da_co_giu_nguyen": 0}
    assert after["manifest_nhap"]["sha256"] == digest(receipt)
    assert after["kiem_sau"]["so_khop_manifest"] == 1400
    assert after["kiem_sau"]["so_lech"] == 0
    coco = "074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d"
    assert before["kiem_truoc"]["coco_val_sha256_truoc"] == coco
    assert after["kiem_sau"]["coco_val_sha256_sau"] == coco
    imported = dict(import_rows=1400, unique_train_files=1400, accepted_manifest_matches=1400,
                    prior_independent_native_hash_matches=1400, differences=0,
                    manifest_sha256=digest(receipt), source_manifest_sha256=old_sha,
                    total_imported_bytes=sum(int(r["byte"]) for r in rows),
                    coco_val_recorded_sha256=coco, physical_windows_dataset_read=False)
    (a.out / "import_manifest_verification.json").write_text(json.dumps(imported, indent=2) + "\n")
    print(json.dumps({"package": package_result, "import_manifest": imported}, indent=2))


if __name__ == "__main__":
    main()
