"""Keep the experimental frozen inputs separate from later integration hashes."""

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    frozen = json.loads((HERE / "robust/inputs.json").read_text())["sha256"]
    changed = {p: {"before": h, "after": sha(p)} for p, h in frozen.items() if sha(p) != h}
    # The historical R107 manifest included run_tracking, but not fingerprint.
    expected = {Path(p).name for p in frozen
                if Path(p).name in {"run_tracking.py", "run_fingerprint.py"}}
    assert {Path(p).name for p in changed} == expected
    paths = [ROOT / "src/rbpf_ctrv.py", ROOT / "src/run_tracking.py", ROOT / "src/run_fingerprint.py",
             ROOT / "configs/phase2/trackers/r112/rbpf500.yaml", HERE / "test_registered.py"]
    checks = [json.loads((HERE / f"registered_{b}.json").read_text()) for b in ["np", "ultralytics"]]
    assert all(row["canonical_source_sha256"] == sha(ROOT / "src/rbpf_ctrv.py") for row in checks)
    result = {"preintegration_verification": str(HERE / "robust/verification.json"),
              "expected_source_changes_after_experiments": changed,
              "remaining_frozen_hashes_unchanged": len(frozen) - len(changed),
              "fingerprint_module_was_in_old_manifest": any(Path(p).name == "run_fingerprint.py" for p in frozen),
              "integration_sha256": {str(p): sha(p) for p in paths}, "registry_checks": checks,
              "selection": "research backend only; Gaussian transfer IDSW gate failed; default unchanged"}
    (HERE / "postintegration_provenance.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
