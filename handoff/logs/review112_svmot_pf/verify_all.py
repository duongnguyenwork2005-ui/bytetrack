"""Independent metric recombination and SVMOT transfer visibility checks."""

import hashlib
import json
from pathlib import Path
import run_pf as setup
import numpy as np
import pandas as pd
import trackeval
from diagnose_svmot import reconstruct

OUT, ROOT, runner = setup.OUT, setup.ROOT, setup.runner


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def combine(path):
    raw = read(path)
    h = {
        key: {**{k: np.asarray(v) if isinstance(v, list) else v for k, v in row.items()},
              "LocA": np.zeros(19)}
        for key, row in raw.items()
    }
    hota = trackeval.metrics.HOTA().combine_sequences(h)
    identity = trackeval.metrics.Identity({"PRINT_CONFIG": False}).combine_sequences(raw)
    result = {"HOTA": float(np.mean(hota["HOTA"])), "IDF1": float(identity["IDF1"]),
              "IDSW": sum(row["IDSW"] for row in raw.values()),
              "FN": sum(row["CLR_FN"] for row in raw.values()),
              "FP": sum(row["CLR_FP"] for row in raw.values())}
    recombined = runner.boot_stats.combine(runner.boot_stats.load_raw_fields(path), list(raw))
    assert all(abs(result[key] - recombined[key]) < 1e-12 for key in result)
    return result, raw


def main():
    inputs = read(OUT / "inputs.json")
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected
               for p, expected in inputs["sha256"].items())
    diagnosis = read(OUT / "svmot_diagnosis.json")
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == expected
               for p, expected in diagnosis["input_sha256"].items())
    checked = []
    for seed in range(3):
        old = ROOT / f"results/trackeval/DETRAC-all/p2-kc48_r72_pf_dev-A-motion200-s{seed}/raw_fields.json"
        checked.append(combine(old)[0])
        path = OUT / f"dev/evaluation/candidate-s{seed}/raw_fields.json"
        value, raw = combine(path)
        expected = read(OUT / "dev_gate.json")["candidate_seeds"][seed]
        assert all(abs(value[k] - expected[k]) < 1e-12 for k in value)
        checked.append(value)
    transfer = OUT / "transfer_svmot"
    summary = read(transfer / "verified_summary.json")
    rows = []
    for case in ["base", "candidate"]:
        for seed in range(3):
            name = f"{case}-s{seed}"
            path = transfer / "evaluation" / name / "raw_fields.json"
            value, raw_metrics = combine(path)
            assert all(abs(value[k] - summary["per_seed"][name][k]) < 1e-12 for k in value)
            checked.append(value)
            ec, dc, mc = runner.rte.build_configs("SVMOTSYS-test", name, summary["videos"])
            dc["TRACKERS_FOLDER"] = str(transfer / "tracking")
            dataset = trackeval.datasets.MotChallenge2DBox(dc)
            for video in summary["videos"]:
                gt = np.loadtxt(ROOT / "data/processed/SVMOTSYS-test" / video / "gt/gt.txt", delimiter=",")
                ids = np.unique(gt[:, 1]).astype(int)
                frames = dataset.seq_lengths[video]
                present = np.zeros((frames, len(ids)), bool)
                low = present.copy()
                at = gt[:, 0].astype(int) - 1, np.searchsorted(ids, gt[:, 1])
                present[at] = True
                low[at] = gt[:, 8] <= .5
                data = dataset.get_preprocessed_seq_data(dataset.get_raw_seq_data(name, video), "pedestrian")
                matched, events = reconstruct(data, present.shape)
                assert len(events) == raw_metrics[video]["IDSW"]
                missed = present & ~matched
                assert missed.sum() == raw_metrics[video]["CLR_FN"]
                rows.append({"case": case, "seed": seed, "video": video,
                             "low_gt": int(low.sum()),
                             "low_IDSW": sum(bool(low[frame, identifier]) for frame, identifier, _ in events),
                             "low_FN": int((missed & low).sum())})
    pd.DataFrame(rows).to_csv(transfer / "visibility.csv", index=False)
    grouped = pd.DataFrame(rows).groupby(["case", "seed"]).sum(numeric_only=True).reset_index()
    low_means = {case: {"IDSW": float(grouped[grouped.case == case].low_IDSW.mean()),
                        "FN": float(grouped[grouped.case == case].low_FN.mean())}
                 for case in ["base", "candidate"]}
    base, candidate = summary["summary"]["base"], summary["summary"]["candidate"]
    gate = {"IDSW": candidate["IDSW"] <= base["IDSW"],
            "HOTA": candidate["HOTA"] >= base["HOTA"] - .005,
            "FN": candidate["FN"] <= 1.005 * base["FN"],
            "all_seeds": all(summary["per_seed"][f"candidate-s{s}"]["IDSW"]
                             <= summary["per_seed"][f"base-s{s}"]["IDSW"] for s in range(3))}
    pilot = read(OUT / "meteor_pilot/verified_summary.json")
    for name, expected in pilot["quality"].items():
        if name.endswith("_mean"):
            continue
        value, _ = combine(OUT / "meteor_pilot" / f"{name}_raw.json")
        assert all(abs(value[k] - expected[k]) < 1e-12 for k in value)
        checked.append(value)
    result = {"official_combinations_verified": len(checked),
              "frozen_preintegration_hashes_verified": len(inputs["sha256"]),
              "svmot_diagnosis_hashes_verified": len(diagnosis["input_sha256"]),
              "transfer_clip_seed_visibility_checks": len(rows),
              "transfer_low_visibility": low_means,
              "transfer_conditions": gate, "transfer_gate_pass": all(gate.values()),
              "limits": "LocA absent in raw history and filled only for unused field; no LocA conclusions"}
    (OUT / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
