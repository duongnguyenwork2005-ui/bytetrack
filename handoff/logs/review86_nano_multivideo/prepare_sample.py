"""Create a fixed, read-only subset of two locked test splits for Nano speed measurement."""
from __future__ import annotations

import configparser
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
N = 150


def seq_length(dataset: str, name: str) -> int:
    ini = ROOT / "data/processed" / dataset / name / "seqinfo.ini"
    cfg = configparser.ConfigParser()
    if not cfg.read(ini, encoding="utf-8"):
        raise FileNotFoundError(ini)
    return cfg.getint("Sequence", "seqLength")


def names() -> list[tuple[str, str]]:
    bdd = json.loads((ROOT / "results/phase2/manifests/bdd100k_transfer_cocovid_val.json").read_text(encoding="utf-8"))["videos"]
    assert len(bdd) == 200
    ua = (ROOT / "data/processed/seqmaps/DETRAC-test.txt").read_text(encoding="utf-8").splitlines()[1:]
    assert len(ua) == 40
    return [("bdd", bdd[i]) for i in range(2, 200, 5)] + [("ua", s.strip()) for s in ua]


def main() -> None:
    rows = []
    archive = OUT / "sample80_middle150.tar"
    if archive.exists():
        raise SystemExit(f"archive already exists: {archive}")
    with tarfile.open(archive, "w") as tar:
        for dataset, name in names():
            seq = "BDDCOCOVID-val" if dataset == "bdd" else "DETRAC-test"
            length = seq_length(seq, name)
            if length < N:
                raise ValueError((dataset, name, length))
            first = (length - N) // 2 + 1
            files = []
            for i in range(first, first + N):
                if dataset == "bdd":
                    p = ROOT / "data/raw/bdd100k/images/track/val" / name / f"{name}-{i:07d}.jpg"
                else:
                    p = ROOT / "data/extracted/DETRAC-Images" / name / f"img{i:05d}.jpg"
                if not p.is_file():
                    raise FileNotFoundError(p)
                tar.add(p, arcname=f"{dataset}/{name}/{p.name}", recursive=False)
                files.append(p.name)
            if dataset == "ua":
                region = ROOT / "data/processed/DETRAC-test" / name / "gt/ignored_regions.txt"
                if not region.is_file():
                    raise FileNotFoundError(region)
                tar.add(region, arcname=f"{dataset}/{name}/ignored_regions.txt", recursive=False)
            rows.append({"dataset": dataset, "name": name, "seq_length": length, "first": first, "last": first + N - 1, "n_frames": N, "first_file": files[0], "last_file": files[-1]})
        payload = json.dumps({"sample_rule": "BDD manifest index 2,7,...,197; all UA test; middle 150 consecutive frames", "rows": rows}, ensure_ascii=False, indent=2).encode("utf-8")
        from io import BytesIO
        info = tarfile.TarInfo("sample_manifest.json")
        info.size = len(payload)
        tar.addfile(info, BytesIO(payload))
    sha = hashlib.sha256()
    with archive.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    (OUT / "sample80_middle150.sha256").write_text(f"{sha.hexdigest()}  {archive.name}\n", encoding="ascii")
    (OUT / "sample_manifest.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(json.dumps({"videos": len(rows), "bdd": sum(r["dataset"] == "bdd" for r in rows), "ua": sum(r["dataset"] == "ua" for r in rows), "bytes": archive.stat().st_size, "sha256": sha.hexdigest()}))


if __name__ == "__main__":
    main()
