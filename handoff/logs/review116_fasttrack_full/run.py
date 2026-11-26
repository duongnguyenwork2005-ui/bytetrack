"""Reuse the R115 experiment implementations on the entire FastTrack release."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import configparser
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np
from download import DATA, OLD_DATA

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / "review115_new_datasets"
OUT = HERE / "scores"
sys.path.insert(0, str(PREVIOUS))
import detect
import score

OLD_WORK = score.WORK
detect.DATA = DATA
score.DATA = DATA
score.WORK = detect.DATA / "evaluation_r116"


def sources():
    manifest = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))
    return {Path(item["path"]).stem: item for item in manifest["archives"]}


def fasttrack_info(archive, name):
    config = configparser.ConfigParser()
    config.read_string(archive.read(f"{name}/seqinfo.ini").decode())
    seq = config["Sequence"]
    fps = seq.getint("frameRate")
    length = seq.getint("seqLength")
    files = [f"{name}/img1/{i:06d}.jpg" for i in range(1, length + 1)]
    assert seq["name"] == name and seq["imDir"] == "img1" and fps > 0
    assert sorted(p for p in archive.namelist() if p.startswith(f"{name}/img1/")
                  and p.endswith(".jpg")) == files
    video = dict(line.split(": ") for line in
                 archive.read(f"{name}/video/video_info.txt").decode().splitlines())
    assert int(video["num_frames"]) == length
    return files, fps, (seq.getint("imHeight"), seq.getint("imWidth"))


def prepare(name, item):
    root = OLD_DATA if name in detect.FASTTRACK else DATA
    path = root / f"{name}.zip"
    assert path.stat().st_size == item["size"]
    assert detect.sha256(path) == item["lfs"]["oid"]
    with ZipFile(path) as archive:
        files, fps, shape = fasttrack_info(archive, name)
        secondary = dict(line.split(": ") for line in
                         archive.read(f"{name}/video/video_info.txt").decode().splitlines())
        gt = np.loadtxt(archive.open(f"{name}/gt/gt.txt"), delimiter=",")
    assert gt.shape[1] == 9 and np.isfinite(gt).all()
    assert (gt[:, 0] >= 1).all() and (gt[:, 0] <= len(files)).all()
    detail = {"frames": len(files), "fps": fps, "reconstructed_video_fps": int(secondary["fps"]),
              "shape": list(shape),
              "gt_counts": dict(Counter(map(str, gt[:, 7].astype(int)))),
              "vis_counts": dict(Counter(map(str, gt[:, 8]))),
              "archive_sha256": item["lfs"]["oid"]}
    score.write_json(HERE / f"{name}_audit.json", detail)
    return detail


def verify_record(name, record_path, work):
    record = json.loads(record_path.read_text(encoding="utf-8"))
    cache_path = work.parent / f"{name}_yolov8n.npz"
    gt_path = work / "gt" / name / "gt" / "gt.txt"
    assert detect.sha256(cache_path) == record["cache_sha256"], cache_path
    assert detect.sha256(gt_path) == record["gt_sha256"], gt_path
    assert set(record["variants"]) == {"kf", "ekf", "ukf", "pf-s0", "pf-s1", "pf-s2"}
    for method, result in record["variants"].items():
        mot = work / "trackers" / method / "data" / f"{name}.txt"
        assert detect.sha256(mot) == result["mot_sha256"], mot
    return record


def run_track(name):
    path = HERE / f"{name}.log"
    with path.open("w", encoding="utf-8") as output:
        subprocess.run([sys.executable, "-u", __file__, "track", name],
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    print("SCORED", name, flush=True)


def evaluate():
    from ultralytics.utils import LOGGER
    LOGGER.setLevel("ERROR")
    manifest = sources()
    detect.FASTTRACK_SHA.update({name: item["lfs"]["oid"] for name, item in manifest.items()})
    pending = dict(manifest)
    tasks = []
    deadline = time.monotonic() + 30 * 60
    with ThreadPoolExecutor(2) as worker:
        while pending:
            ready = [name for name in pending if
                     ((OLD_DATA if name in detect.FASTTRACK else DATA) / f"{name}.zip").is_file()]
            ready.sort(key=lambda n: (n in detect.FASTTRACK, pending[n]["size"]))
            if not ready:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Waiting for archives {list(pending)}")
                time.sleep(5)
                continue
            for name in ready:
                item = pending.pop(name)
                info = prepare(name, item)
                if name in detect.FASTTRACK:
                    record = verify_record(name, PREVIOUS / "scores" / f"{name}.json", OLD_WORK)
                    assert record["frames"] == info["frames"]
                    print("VERIFIED prior scores", name, info["frames"], flush=True)
                    continue
                output = OUT / f"{name}.json"
                if output.exists() and len(json.loads(output.read_text())["variants"]) == 6:
                    verify_record(name, output, score.WORK)
                    print("VERIFIED scores", name, flush=True)
                    continue
                print("DETECT", name, info["frames"], "frames", flush=True)
                detect.run([name], fasttrack_info)
                tasks.append(worker.submit(run_track, name))
        for task in tasks:
            task.result()


def combined(records):
    totals = {}
    for method in ("kf", "ekf", "ukf", "pf-s0", "pf-s1", "pf-s2"):
        runs = {name: record["variants"][method] for name, record in records.items()}
        scores = {}
        for metric in score.METRICS:
            key = metric.get_name()
            fields = {name: {k: np.asarray(v) if isinstance(v, list) else v
                             for k, v in result["raw"][key].items()} for name, result in runs.items()}
            scores[key] = metric.combine_sequences(fields)
        concise = score.concise(scores)
        for key in ("IDSW", "FN", "FP"):
            assert concise[key] == sum(run["metrics"][key] for run in runs.values())
        groups = {}
        for run in runs.values():
            for key, counts in run["groups"].items():
                groups.setdefault(key, Counter()).update(counts)
        totals[method] = {"metrics": concise, "groups": groups}
        if method in ("ekf", "ukf"):
            deltas = np.array([record["variants"][method]["metrics"]["IDSW"]
                               - record["variants"]["kf"]["metrics"]["IDSW"]
                               for record in records.values()])
            rng = np.random.default_rng(116)
            samples = rng.choice(deltas, size=(10000, len(deltas))).sum(axis=1)
            totals[method]["idsw_delta_vs_kf"] = {"per_clip": dict(zip(records, map(int, deltas))),
                                                   "bootstrap_clip_95": np.percentile(samples, [2.5, 97.5])}
    return {"clips": list(records), "frames": sum(row["frames"] for row in records.values()),
            "car_ids_per_sequence": sum(row["gt_ids"] for row in records.values()),
            "car_boxes": sum(row["gt_boxes"] for row in records.values()), "results": totals}


def aggregate():
    records = {}
    alternate = {}
    for name in sources():
        prior = name in detect.FASTTRACK
        directory, work = (PREVIOUS / "scores", OLD_WORK) if prior else (OUT, score.WORK)
        records[name] = verify_record(name, directory / f"{name}.json", work)
        audit = json.loads((HERE / f"{name}_audit.json").read_text())
        assert records[name]["frames"] == audit["frames"]
        if audit["fps"] != audit["reconstructed_video_fps"]:
            path = OUT / "alternative_fps" / f"{name}.json"
            alternate[name] = verify_record(name, path, DATA / "evaluation_r116_alternative_fps")
            assert alternate[name]["fps"] == audit["reconstructed_video_fps"]
            for key in ("frames", "gt_sha256", "cache_sha256"):
                assert alternate[name][key] == records[name][key]
    new = {name: row for name, row in records.items() if name not in detect.FASTTRACK}
    result = {"all12": combined(records), "new9": combined(new),
              "all12_alternate_fps": combined(records | alternate), "alternate_clips": list(alternate)}
    score.write_json(OUT / "scores.json", result)


if __name__ == "__main__":
    os.environ["YOLO_CONFIG_DIR"] = str(detect.DATA / "ultralytics_settings")
    OUT.mkdir(exist_ok=True)
    command = sys.argv[1]
    if command == "track":
        name = sys.argv[2]
        detail = json.loads((HERE / f"{name}_audit.json").read_text())
        score.score_sequence(name, OUT, fps=detail["fps"], fasttrack_metadata=fasttrack_info)
    elif command == "track-alt-fps":
        name = sys.argv[2]
        detail = json.loads((HERE / f"{name}_audit.json").read_text())
        assert detail["fps"] != detail["reconstructed_video_fps"]
        score.WORK = DATA / "evaluation_r116_alternative_fps"
        output = OUT / "alternative_fps"
        output.mkdir(exist_ok=True)
        score.score_sequence(name, output, fps=detail["reconstructed_video_fps"],
                             fasttrack_metadata=fasttrack_info)
    elif command == "run":
        evaluate()
    elif command == "aggregate":
        aggregate()
    else:
        raise ValueError(command)
