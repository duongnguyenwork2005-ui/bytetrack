"""Validate all 320 archived Nano run metadata records against the fixed R86 protocol."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
expected = json.loads((HERE / "run_inputs.json").read_text())
manifest = json.loads((HERE / "sample_manifest.json").read_text())
wanted = {(r["dataset"], r["name"], tr, v) for r in manifest["rows"] for tr in ("cv", "ekf") for v in "AB"}
seen = set()
errors = []
max_temp = 0.0
max_swap = 0.0
therm_at = None
swap_at = None
fps_values = Counter()
regions_count = Counter()

with tarfile.open(HERE / "meta320.tar.gz", "r:gz") as tar:
    metas = [m for m in tar if m.name.endswith("/run_meta.json")]
    for member in metas:
        parts = Path(member.name).parts
        if len(parts) != 6 or parts[0] != "runs":
            errors.append((member.name, "path"))
            continue
        _, ds, name, tr, ver, _ = parts
        key = (ds, name, tr, ver)
        if key in seen or key not in wanted:
            errors.append((key, "unexpected or duplicate"))
        seen.add(key)
        x = json.load(tar.extractfile(member))
        d, cfg, h = x["dem"], x["cau_hinh"], x["hello_con"]
        if not (x["hoan_tat"] and x["ma_thoat"] == 0 and d["n_frames_khai_bao"] == d["n_frame_da_xu_ly"] == 150
                and d["n_unreadable"] == d["n_truncated"] == 0):
            errors.append((key, "completion"))
        if h["engine_sha256"] != expected["engines"][ver] or h["onnx_sha256"] != expected["onnx"][ver]:
            errors.append((key, "engine/onnx hash"))
        if (cfg["fps"], cfg["elapsed_mode"], cfg["track_buffer_frames"], cfg["classes"]) != (
                (5.0, "steps", 6, [1, 2, 3, 5, 7]) if ds == "bdd" else
                (25.0, "skip", 30, [2, 3, 5, 7])):
            errors.append((key, "dataset tracker config"))
        if cfg["conf_min"] != 0.25 or cfg["det_conf"] != 0.05 or cfg["det_iou"] != 0.7:
            errors.append((key, "detection thresholds"))
        if (h["regions"] is None) != (ds == "bdd"):
            errors.append((key, "ignored regions"))
        regions_count[(ds, h["n_regions"])] += 1
        fps_values[cfg["fps"]] += 1
        telem = x["thiet_bi"].get("tegrastats", {})
        temp = max((float(v) for v in telem.get("nhiet_max_C", {}).values()), default=0.0)
        if temp > max_temp:
            max_temp, therm_at = temp, key
        swap = float(telem.get("swap_dung_MB_max") or 0)
        if swap > max_swap:
            max_swap, swap_at = swap, key

if seen != wanted:
    errors.append(("coverage", {"missing": len(wanted - seen), "unexpected": len(seen - wanted)}))
progress = [json.loads(s) for s in (HERE / "progress.jsonl").read_text().splitlines() if s]
boots = {r.get("boot") for r in progress if not r.get("skipped_complete")}
if boots != {expected["boot"]}:
    errors.append(("boot", list(boots)))
if max_temp >= 75 or max_swap - expected["swap_mb_start"] > 100:
    errors.append(("resource guard", {"max_temp": max_temp, "max_swap": max_swap}))
report = {"n_expected": len(wanted), "n_meta": len(metas), "n_unique": len(seen), "n_progress": len(progress),
          "boots": sorted(boots), "max_temp_C": max_temp, "max_temp_at": therm_at,
          "max_swap_MB": max_swap, "max_swap_at": swap_at,
          "fps_counts": dict(fps_values), "regions_counts": {str(k): v for k, v in regions_count.items()},
          "errors": errors, "ok": not errors}
(HERE / "audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
if errors:
    raise SystemExit(1)
