"""Verify the archived first-run code, dev runs and both transfer experiments."""

from pathlib import Path
import json

from run import HERE, ROOT, previous, write


def run():
    frozen = json.loads((HERE / "inputs.json").read_text())
    checks = []
    for path, digest in frozen["sha256"].items():
        source = Path(path)
        if source.parent == HERE and source.name in ("apf.py", "run.py"):
            source = HERE / "initial_source" / source.name
        assert previous.sha(source) == digest, source
        checks.append(str(source))
    amended = json.loads((HERE / "ess_inputs.json").read_text())
    for path, digest in amended["sha256"].items():
        assert previous.sha(Path(path)) == digest, path
        checks.append(path)
    units = []
    for mode in ("apf", "apf_k2", "apf_ess_k2"):
        for seed in range(3):
            folder = HERE / "dev/tracking" / previous.SPLIT / f"{mode}-s{seed}"
            raw_path = HERE / f"dev/evaluation/{mode}-s{seed}/raw_fields.json"
            raw = previous.boot_stats.load_raw_fields(raw_path)
            assert set(raw) == set(previous.DEV)
            combined = previous.boot_stats.combine(raw, previous.DEV)
            saved = json.loads(raw_path.with_name("combined.json").read_text())
            assert all(abs(combined[k] - saved[k]) < 1e-10 for k in combined)
            for video in previous.DEV:
                metadata = json.loads((folder / "meta" / f"{video}.json").read_text())
                mot = folder / "data" / f"{video}.txt"
                assert previous.sha(mot) == metadata["mot_sha256"]
                assert metadata["frames"] == frozen["frames"][video]
                units.append({"stage": "dev", "variant": mode, "seed": seed, "video": video,
                              "frames": metadata["frames"], "mot_sha256": metadata["mot_sha256"]})
    manifest = json.loads((HERE.parent / "review116_fasttrack_full/sources.json").read_text())
    old = {"task_day_occlusion", "task_day_right_turn_occlusion", "task_tunnel"}
    for directory, work in (("fasttrack", "evaluation_r117"), ("fasttrack_ess", "evaluation_r117_ess")):
        for seed in range(3):
            for item in manifest["archives"]:
                name = Path(item["path"]).stem
                root = Path("F:/LABOTARY_data/goc_tai_ve/FastTrack-R115") if name in old else Path("D:/LABOTARY/test dataset/FastTrack-R116")
                data = json.loads((HERE / directory / f"seed{seed}/{name}.json").read_text())
                assert previous.sha(root / f"{name}_yolov8n.npz") == data["cache_sha256"]
                folder = Path("D:/LABOTARY/test dataset/FastTrack-R116") / work / f"seed{seed}"
                assert previous.sha(folder / "gt" / name / "gt/gt.txt") == data["gt_sha256"]
                row = data["variants"][f"pf-s{seed}"]
                assert previous.sha(folder / "trackers" / f"pf-s{seed}/data/{name}.txt") == row["mot_sha256"]
                units.append({"stage": directory, "seed": seed, "video": name, "frames": data["frames"],
                              "mot_sha256": row["mot_sha256"]})
    assert len(units) == 180
    write(HERE / "verification.json", {"frozen_inputs_checked": len(checks), "units": units,
                                      "run_count": len(units), "status": "PASS"})
    print("PASS", len(units), "completed tracking units,", len(checks), "frozen input hashes", flush=True)


if __name__ == "__main__":
    run()
