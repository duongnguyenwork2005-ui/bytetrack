"""Verify the pre-handoff file snapshot without modifying experiment inputs."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent


def main():
    with (OUT / "protected_files_before.csv").open(encoding="utf-8", newline="") as f:
        before = list(csv.DictReader(f))
    changed = []
    for row in before:
        path = ROOT / row["path"]
        if not path.is_file():
            changed.append({"path": row["path"], "reason": "missing"})
            continue
        with path.open("rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if path.stat().st_size != int(row["bytes"]) or digest != row["sha256"]:
            changed.append({"path": row["path"], "reason": "content_changed"})
    saved_report = ROOT / ".git/test_eval_FIX_REPORT_before.md"
    report_preserved = None
    if saved_report.is_file():
        report_preserved = saved_report.read_bytes() in (ROOT / "FIX_REPORT.md").read_bytes()
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "protected_files": len(before),
        "changed_or_missing": changed,
        "original_uncommitted_FIX_REPORT_retained_verbatim": report_preserved,
        "scope": "All preexisting files under results, data/processed, data/interim, src, configs, plus detector weights; excludes __pycache__ and this new provenance directory. Image files are independently audited.",
    }
    (OUT / "preservation_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if changed or report_preserved is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
