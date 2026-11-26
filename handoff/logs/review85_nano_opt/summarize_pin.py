"""Collect R85 paired pinned-buffer smoke runs on Jetson Nano."""
import hashlib
import json
from pathlib import Path


ROOT = Path("/home/jetson/nano_run/r63_20261030")
rows = []
for name in ("base1", "out", "inout", "base2"):
    path = ROOT / ("r85_pin_{}_20261106".format(name))
    run = path / "luot"
    result = json.loads((run / "result.json").read_text())
    meta = json.loads((run / "live/run_meta.json").read_text())
    timing = meta["thoi_gian"]["tom_tat_sau_warmup"]
    guard = json.loads((path / "giam_sat.json").read_text())
    rows.append({"variant": name, "fps": result["fps"], "parity_replay": result["parity_byte"],
                 "n_frames": meta["dem"]["n_frame_da_xu_ly"],
                 "decode_ms": timing["decode_ms"]["trung_vi"],
                 "infer_ms": timing["infer_ms"]["trung_vi"],
                 "post_ms": timing["post_ms"]["trung_vi"],
                 "p90_ms": timing["tre_ms"]["p90"],
                 "guard": guard["vi_pham_cong"],
                 "mot_sha256": hashlib.sha256((run / "live/data/MVI_39031.txt").read_bytes()).hexdigest(),
                 "class_sha256": hashlib.sha256((run / "live/data_cls/MVI_39031.csv").read_bytes()).hexdigest()})
assert all(x["n_frames"] == 300 and x["parity_replay"] and not x["guard"] for x in rows)
assert len({x["mot_sha256"] for x in rows}) == len({x["class_sha256"] for x in rows}) == 1
print(json.dumps({"rows": rows, "all_parity": True, "all_guards_clear": True}, indent=2))
