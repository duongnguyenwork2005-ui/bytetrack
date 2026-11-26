"""Run RBPF against the frozen R110 detector and compare with stored scores."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "review110_nano_filters"))
import replay_filters as replay


def configure():
    replay.CONFIGS.update(rbpf=replay.ROOT / "configs/phase2/trackers/r112/rbpf500.yaml")
    replay.METHODS = tuple((f"rbpf_s{seed}", "rbpf", seed) for seed in range(3))


def worker(task):
    # The historical worker only returns a KF parity field. Its replay body
    # cannot be called with RBPF alone, so preserve its input checks here.
    row, source, previous, out = task
    name, dataset = row["name"], row["dataset"]
    path = source / "cache" / name / f"{name}.npz"
    assert replay.sha256(path.read_bytes()).hexdigest() == row["cache_sha256"]
    condition = replay.read(previous / "overlap/runs/full/pipeline" / dataset / name / "B/run_meta.json")["cau_hinh"]
    mapping = {int(k): int(v) for k, v in condition["class_map"].items()}
    vd = replay.det_cache.VideoDetections(path, condition["conf_min"], condition["classes"], condition["max_det"], mapping)
    fps = 5 if dataset == "bdd" else 25
    np = replay.np
    assert np.array_equal(vd.frame_idx, np.arange(1, 151)) and not vd.unreadable.any()
    assert np.allclose(vd.time_s, np.arange(150) / fps, rtol=0, atol=1e-9)
    cfg = replay.tracking.load_tracker_cfg(replay.CONFIGS["rbpf"], {"track_buffer": condition["track_buffer_frames"]})
    for method, _, seed in replay.METHODS:
        mot, cls, timing = replay.tracking.track_video(vd, cfg, 1 / fps, condition["elapsed_mode"], seed=seed, video=name)
        assert len(timing) == 150 and not mot.duplicated(["frame", "id"]).any()
        assert np.isfinite(mot.to_numpy(float)).all() and (mot.iloc[:, 4:6] > 0).all().all()
        base = out / "trackers" / replay.evaluation.SPLITS[dataset][0] / method
        for kind, table in [("data", mot), ("data_cls", cls)]:
            (base / kind).mkdir(parents=True, exist_ok=True)
            table.to_csv(base / kind / f"{name}.{'txt' if kind == 'data' else 'csv'}", index=False, header=kind != "data")
    return name


def main():
    source = replay.HERE / "nano_capture/results"
    previous = HERE.parent / "review109_nano80/results"
    out = HERE / "nano80"
    old = replay.read(replay.HERE / "replay_full/scores.json")
    captures = replay.read(source / "captures.json")
    assert len(captures) == 80 and old["capture"] == captures
    out.mkdir(exist_ok=False)
    with replay.ProcessPoolExecutor(max_workers=4, initializer=configure) as pool:
        for i, name in enumerate(pool.map(worker, [(row, source, previous, out) for row in captures]), 1):
            print(i, name, flush=True)
    configure()
    # Historical aggregation ends by averaging pf_s0/s1/s2. These aliases
    # apply only during scoring and resolve to the three new output folders.
    replay.METHODS = tuple((f"rbpf_s{s}", "rbpf", s) for s in range(3))
    for dataset, (split, _) in replay.evaluation.SPLITS.items():
        for s in range(3):
            path = out / "trackers" / split
            (path / f"rbpf_s{s}").rename(path / f"pf_s{s}")
    replay.METHODS = tuple((f"pf_s{s}", "rbpf", s) for s in range(3))
    scored = replay.score(out, captures)
    summary = {}
    for dataset, (split, _) in replay.evaluation.SPLITS.items():
        for s in range(3):
            path = out / "trackers" / split
            (path / f"pf_s{s}").rename(path / f"rbpf_s{s}")
        summary[dataset] = {key: row["combined"] if "combined" in row else row
                            for key, row in old["quality"][dataset].items()}
        summary[dataset].update({key.replace("pf_", "rbpf_", 1): row["combined"] if "combined" in row else row
                                 for key, row in scored[dataset].items()})
    (out / "scores.json").write_text(json.dumps({"clips": 80, "runs": 240, "summary": summary, "rbpf": scored}, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
