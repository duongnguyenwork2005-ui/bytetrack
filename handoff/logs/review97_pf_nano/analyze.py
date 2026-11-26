"""Paired, per-video speed summary for R97; never bootstrap frames."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE / "nano_multi.json").read_text(encoding="utf-8"))
assert len(rows) == 32
assert len({r["boot"] for r in rows}) == 1
assert {r["mmc_errors"] for r in rows} == {"0"}
out = {}
for ds in ("bdd", "ua"):
    selected = [r for r in rows if r["dataset"] == ds]
    assert len(selected) == 16
    names = list(dict.fromkeys(r["name"] for r in selected))
    assert len(names) == 8
    pairs = []
    for name in names:
        pair = {r["tracker"]: r for r in selected if r["name"] == name}
        assert set(pair) == {"cv", "pf"}
        cv, pf = pair["cv"], pair["pf"]
        t = lambda r, k: float(r["timing"][k]["trung_vi"])
        pairs.append({"name": name,
                      "fps_cv": cv["fps_steady"], "fps_pf": pf["fps_steady"],
                      "fps_rel_pct": 100 * (pf["fps_steady"] / cv["fps_steady"] - 1),
                      "update_delta_ms": t(pf, "t_update_ms") - t(cv, "t_update_ms"),
                      "decode_delta_ms": t(pf, "decode_ms") - t(cv, "decode_ms"),
                      "infer_delta_ms": t(pf, "infer_ms") - t(cv, "infer_ms"),
                      "latency_p90_delta_ms": (pf["timing"]["tre_ms"]["p90"] -
                                                cv["timing"]["tre_ms"]["p90"])})
    rng = np.random.default_rng(9710)
    summary = {"n_video": 8, "pf_faster_count": sum(p["fps_rel_pct"] > 0 for p in pairs)}
    for key in ("fps_rel_pct", "update_delta_ms", "decode_delta_ms", "infer_delta_ms",
                "latency_p90_delta_ms"):
        x = np.asarray([p[key] for p in pairs], dtype=float)
        bs = np.median(x[rng.integers(0, len(x), size=(5000, len(x)))], axis=1)
        summary[key] = {"median": float(np.median(x)),
                        "min": float(np.min(x)), "max": float(np.max(x)),
                        "bootstrap95": np.quantile(bs, [.025, .975]).tolist()}
    out[ds] = {"summary": summary, "pairs": pairs}
    print(ds, json.dumps(summary, ensure_ascii=False), flush=True)
(HERE / "multi_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
