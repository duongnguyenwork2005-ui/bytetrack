"""Verify official metric combination, fixed gates, lineage, and output structure."""
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from trackeval.metrics import HOTA, Identity

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT / "src"))
import boot_stats


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


frozen = read(OUT / "inputs.json")
gate = read(OUT / "dev_gate.json")
dev = list(frozen["frames"])
comparisons, inputs = [], {}
for case in ("base", "candidate"):
    for seed in range(3):
        path = (ROOT / f"results/trackeval/DETRAC-all/p2-kc48_r72_pf_dev-A-motion200-s{seed}/raw_fields.json"
                if case == "base" else OUT / f"dev/evaluation/candidate-s{seed}/raw_fields.json")
        raw = read(path)
        assert set(raw) == set(dev)
        # Raw history omits LocA: a neutral dummy affects only that unused field.
        hota_input = {s: {**{k: np.array(v) if isinstance(v, list) else v for k, v in row.items()},
                          "LocA": np.zeros(19)} for s, row in raw.items()}
        hota = HOTA().combine_sequences(hota_input)
        ident = Identity({"PRINT_CONFIG": False}).combine_sequences(raw)
        measured = {"HOTA": float(np.mean(hota["HOTA"])), "IDF1": float(ident["IDF1"]),
                    "IDSW": sum(r["IDSW"] for r in raw.values()),
                    "FN": sum(r["CLR_FN"] for r in raw.values()), "FP": sum(r["CLR_FP"] for r in raw.values())}
        combined = boot_stats.combine(raw, dev)
        assert all(abs(measured[k] - combined[k]) < 1e-12 for k in measured)
        if case == "candidate":
            assert all(abs(measured[k] - gate["candidate_seeds"][seed][k]) < 1e-12 for k in measured)
            folder = OUT / f"dev/tracking/DETRAC-all/candidate-s{seed}"
            meta = read(folder / "run_meta.json")
            assert meta["cfg"] == frozen["cfg"] and meta["seed"] == seed
            assert meta["plugin_enabled"] and meta["cutoff"] == .25 and meta["max_det"] == 300
            assert meta["inputs_sha256"] == sha(OUT / "inputs.json")
            assert meta["plugin_sha256"] == sha(OUT / "pf_postgap.py")
            assert meta["runner_sha256"] == sha(OUT / "run_r107.py")
            assert meta["protocol_sha256"] == sha(OUT / "00_protocol.md")
            for video in dev:
                mot = pd.read_csv(folder / f"data/{video}.txt", header=None)
                timing = pd.read_csv(folder / f"timing/{video}.csv")
                assert len(timing) == frozen["frames"][video] and timing.frame.is_unique
                assert not mot.duplicated([0, 1]).any()
                assert set(mot[0]) <= set(timing.frame)
                assert (mot[[4, 5]].to_numpy() > 0).all() and np.isfinite(mot.to_numpy()).all()
                events = read(folder / f"events/{video}.json")
                assert all(e["frame"] in set(timing.frame) for e in events)
                inputs[str(folder / f"data/{video}.txt")] = sha(folder / f"data/{video}.txt")
        inputs[str(path)] = sha(path)
        comparisons.append({"case": case, "seed": seed, **measured})

base = gate["baseline"]
cand = gate["candidate"]
expected_conditions = [cand["IDSW"] <= .9 * base["IDSW"], cand["HOTA"] >= base["HOTA"] - .005,
                       cand["FN"] <= 1.005 * base["FN"],
                       all(c["IDSW"] <= b["IDSW"] for c, b in zip(gate["candidate_seeds"], frozen["baseline_seed_combined"])),
                       sum(v < 0 for v in gate["per_video_IDSW_delta"].values()) >
                       sum(v > 0 for v in gate["per_video_IDSW_delta"].values())]
assert gate["gate_pass"] == all(expected_conditions)
assert sorted(gate["conditions"].values()) == sorted(expected_conditions)
for p, digest in frozen["sha256"].items():
    assert sha(Path(p)) == digest, p
diagnosis = read(OUT / "diagnosis_verified.json")
for p, digest in diagnosis["sha256"].items():
    assert sha(Path(p)) == digest, p
smoke = read(OUT / "smoke_checks.json")
assert len(smoke) == 6 and all(r["exact_disabled_mot"] for r in smoke)
assert sum(r["repeat_candidate"] for r in smoke) == 2
result = {"official_trackeval_combinations": comparisons, "gate_verified": True,
          "unchanged_frozen_hashes": len(frozen["sha256"]), "complete_video_seed_units": 36,
          "smoke_disabled_exact": 6, "candidate_repeat_exact": 2,
          "no_external_domain_opened": not (OUT / "transfer").exists(),
          "inputs_sha256": inputs,
          "limits": "Baseline from stored R72 raw metrics, validated combinations and disabled prefix parity; no full baseline replay. Review is still pending."}
result["environment"] = {"python": platform.python_version(), "executable": sys.executable,
                         "packages": {p: importlib.metadata.version(p) for p in
                                      ("numpy", "scipy", "lap", "ultralytics", "trackeval", "pandas", "torch")}}
import inspect
from trackeval.datasets import MotChallenge2DBox
result["metric_sources_sha256"] = {inspect.getfile(cls): sha(Path(inspect.getfile(cls)))
                                    for cls in (HOTA, Identity, MotChallenge2DBox)}
(OUT / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print("VERIFIED: six official TrackEval combinations; 36 units; smoke parity/repeat; gate; frozen hashes")
