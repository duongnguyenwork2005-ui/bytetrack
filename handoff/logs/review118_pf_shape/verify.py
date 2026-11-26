"""Verify both Cartesian experiments and any planned SVMOT transfer."""

from pathlib import Path
import json
import run as job


def run():
    verified = []
    inputs_checked = set()
    for mode, root in (("two_observations", job.HERE), ("automatic", job.HERE / "automatic")):
        frozen = json.loads((root / "inputs.json").read_text())
        for path, digest in frozen["sha256"].items():
            assert job.base.sha(Path(path)) == digest, path
            inputs_checked.add(path)
        for seed in range(3):
            folder = root / f"dev/tracking/DETRAC-all/cartesian-s{seed}"
            raw = job.base.boot_stats.load_raw_fields(root / f"dev/evaluation/cartesian-s{seed}/raw_fields.json")
            combined = job.base.boot_stats.combine(raw, job.base.DEV)
            recorded = json.loads((root / f"dev/evaluation/cartesian-s{seed}/combined.json").read_text())
            assert all(abs(combined[k] - recorded[k]) < 1e-10 for k in combined)
            for video in job.base.DEV:
                row = json.loads((folder / "meta" / f"{video}.json").read_text())
                output = folder / "data" / f"{video}.txt"
                timing = job.base.pd.read_csv(folder / "timing" / f"{video}.csv")
                assert row["frames"] == frozen["frames"][video] == len(timing)
                assert job.base.sha(output) == row["mot_sha256"]
                verified.append({"stage": mode, "seed": seed, "video": video, **row})
    if (job.HERE / "transfer_summary.json").is_file():
        from transfer import VIDEOS
        for mode in ("parent", "candidate"):
            for seed in range(3):
                folder = job.HERE / f"transfer/tracking/SVMOTSYS-test/{mode}-s{seed}"
                raw = job.base.boot_stats.load_raw_fields(job.HERE / f"transfer/evaluation/{mode}-s{seed}/raw_fields.json")
                combined = job.base.boot_stats.combine(raw, VIDEOS)
                recorded = json.loads((job.HERE / f"transfer/evaluation/{mode}-s{seed}/combined.json").read_text())
                assert all(abs(combined[k] - recorded[k]) < 1e-10 for k in combined)
                for video in VIDEOS:
                    row = json.loads((folder / "meta" / f"{video}.json").read_text())
                    output = folder / "data" / f"{video}.txt"
                    assert job.base.sha(output) == row["mot_sha256"]
                    verified.append({"stage": "transfer", "mode": mode, "seed": seed, "video": video, **row})
                    if mode == "parent":
                        reference = job.HERE.parent / f"review112_svmot_pf/transfer_svmot/tracking/SVMOTSYS-test/candidate-s{seed}/data/{video}.txt"
                        old = job.np.loadtxt(reference, delimiter=",")
                        current = job.np.loadtxt(output, delimiter=",")
                        assert job.np.allclose(current, old, rtol=0, atol=1e-9)
    job.write(job.HERE / "verification.json", {"status": "PASS", "inputs_checked": len(inputs_checked),
                                            "units": len(verified), "runs": verified})
    print("PASS", len(verified), "tracking units;", len(inputs_checked), "input/source hashes", flush=True)


if __name__ == "__main__":
    run()
