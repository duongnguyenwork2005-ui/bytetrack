"""R39-F1: exercise the real journal reader/writer, without torch or training.

python REPRO_REVIEW_1454b3d.py --repo D:/LABOTARY/bytetrack --out repro.json
Exit 1 means at least one journal roundtrip is broken; 0 means all cases pass.
This is a focused reproduction, not a full checkpoint/Ultralytics resume test.
If repair is implemented only in the actual resume path, add an integration
test there instead of treating this writer-level script as the acceptance gate.
"""
import argparse
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    src = args.repo / "src/tiep_tuc_huan_luyen.py"
    spec = importlib.util.spec_from_file_location("resume_under_review", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cases = []
    with tempfile.TemporaryDirectory(prefix="codex_r39_journal_") as tmp:
        for name, suffix in [
            ("clean_control", b""),
            ("partial_tail_no_newline", b'{"loai":"epoch_da_luu","epoch":2'),
            ("partial_tail_with_newline", b'{"loai":"epoch_da_luu","epoch":2\n'),
            ("valid_record_no_newline", b'{"loai":"ket_thuc","trang_thai":"bi_ngat"}'),
        ]:
            run = Path(tmp) / name
            run.mkdir()
            mod.ghi_nhat_ky(run, "bat_dau", che_do_luot="moi", lan=0, ten="fixture")
            mod.ghi_nhat_ky(run, "epoch_da_luu", epoch=1, lan=0)
            p = run / mod.TEN_NHAT_KY
            prefix = p.read_bytes()
            p.write_bytes(prefix + suffix)
            before, warnings = mod.doc_nhat_ky(run)
            mod.ghi_nhat_ky(run, "bat_dau", che_do_luot="tiep_tuc", lan=1, ten="fixture")
            mod.ghi_nhat_ky(run, "epoch_da_luu", epoch=2, lan=1)
            error, after = None, []
            try:
                after, _ = mod.doc_nhat_ky(run)
            except mod.LoiTiepTuc as exc:
                error = str(exc).replace(str(run), "<fixture>")
            expected = len(before) + 2
            ok = error is None and len(after) == expected and after[:len(before)] == before
            cases.append({"case": name, "accepted_events_before": len(before),
                          "warnings_before": warnings, "expected_events_after": expected,
                          "actual_events_after": len(after), "error_after": error,
                          "original_valid_prefix_unchanged": p.read_bytes().startswith(prefix),
                          "pass": ok, "journal_after": p.read_text(encoding="utf-8")})
    result = {"source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
              "scope": "real doc_nhat_ky + ghi_nhat_ky; no torch, checkpoint, dataset, GPU or training",
              "bug_count": sum(not c["pass"] for c in cases), "cases": cases}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(result["bug_count"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
