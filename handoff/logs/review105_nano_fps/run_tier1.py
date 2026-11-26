"""Run the final R105 CPU checks synchronously and record actual exit codes."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("host", type=Path)
    parser.add_argument("stage", type=Path)
    args = parser.parse_args()
    records = []
    for name, argv, log in (
        ("cpu_tests", [sys.executable, "-B", str(HERE / "test_r105.py")], "cpu_tests_final_verified.log"),
        ("real_evidence_audit", [sys.executable, "-B", str(HERE / "audit_benchmark.py"), str(args.host), str(args.stage)], "audit_final_verified.log"),
    ):
        result = subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (HERE / log).write_bytes(result.stdout)
        records.append(dict(check=name, command=argv, exit_code=result.returncode, evidence=str(HERE / log)))
    try:
        sources = sorted(HERE.glob("*.py")) + sorted((HERE / "drivers").glob("*.py"))
        for path in sources:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        for name in ("bench_nano.py", "prepare_nano.py", "cadence.py"):
            ast.parse((HERE / name).read_text(encoding="utf-8"), filename=name, feature_version=(3, 6))
        records.append(dict(check="syntax", exit_code=0, files=len(sources), command="compile all local R105 sources; ast.parse(feature_version=(3,6)) bench/prepare/cadence"))
        manifest = json.loads((HERE / "bundle_manifest.json").read_text())
        assert digest(HERE / "r105_deploy.tar.gz") == manifest["archive_sha256"]
        with tarfile.open(HERE / "r105_deploy.tar.gz") as archive:
            assert set(archive.getnames()) == {r["path"] for r in manifest["files"]}
            for row in manifest["files"]:
                assert digest(HERE / row["path"]) == row["sha256"]
                assert hashlib.sha256(archive.extractfile(row["path"]).read()).hexdigest() == row["sha256"]
        records.append(dict(check="existing_offline_bundle", exit_code=0, files=len(manifest["files"]),
                            command="verify existing archive and each payload against current files; no rebuild or deployment"))
    except Exception as error:
        records.append(dict(check="syntax_or_bundle", exit_code=1, error=repr(error)))
    output = dict(checks=records, source_sha256={str(p.relative_to(HERE)): digest(p) for p in sources},
                  scope="CPU Tier 1 and collected real Nano evidence only; separate Codex review pending",
                  benchmark_source="host_evidence/executed_sources; current offline bundle was not the executed deployment")
    (HERE / "tier1_checks_current.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(records))
    raise SystemExit(0 if all(r["exit_code"] == 0 for r in records) else 1)


if __name__ == "__main__":
    main()
