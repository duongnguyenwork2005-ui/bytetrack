"""Run only supplied monitor tests meaningful on this POSIX reviewer host."""
import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--repo", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
p = a.repo / "src/test_chay_giam_sat.py"
s = importlib.util.spec_from_file_location("monitor_tests", p.resolve())
t = importlib.util.module_from_spec(s)
s.loader.exec_module(t)
with tempfile.TemporaryDirectory(prefix="codex_r39_monitor_") as d:
    root = Path(d)
    for name in ("m1", "m2", "m3", "m7", "m8", "m9", "m10", "m11"):
        t._chay(name.upper(), lambda name=name: getattr(t, name)(root))
r = {"scope": "unmodified supplied tests M1,M2,M3,M7,M8,M9,M10,M11 on POSIX",
     "not_run": ["M4 Windows kill exit semantics", "M5 Windows NTSTATUS", "M6 Windows CtrlBreak"],
     "checks": [{"name": n, "pass": ok} for n, ok in t.KQ],
     "pass": sum(ok for _, ok in t.KQ), "fail": sum(not ok for _, ok in t.KQ)}
a.out.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(r, ensure_ascii=False, indent=2))
raise SystemExit(int(r["fail"] > 0))
