"""Freeze R72 source/config/cache/GT inputs before the full run."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
old=json.loads((ROOT/"handoff/logs/review70_pf/input_lock.json").read_text(encoding="utf-8"))
paths={Path(p) for p in old}
paths.update([ROOT/"handoff/logs/review72_pf/00_protocol.json",ROOT/"src/r72_pf/run_dev.py",ROOT/"src/r72_pf/analyze.py",ROOT/"configs/phase2/experiments/kc48_r72_pf_dev.yaml",ROOT/"configs/phase2/experiments/kc48_r72_pf_smoke.yaml"])
paths.update((ROOT/"configs/phase2/trackers/r72").glob("*.yaml"))
lock={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
out=ROOT/"handoff/logs/review72_pf/input_lock.json"
if out.exists() and json.loads(out.read_text(encoding="utf-8")) != lock:
    raise ValueError("Frozen inputs changed")
out.write_text(json.dumps(lock,indent=2)+"\n",encoding="utf-8")
print(len(lock))
