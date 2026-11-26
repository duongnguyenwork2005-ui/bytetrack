"""Audit the full SVMOT release and score the 21 remaining sequences as a separate split."""
import configparser
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from helmet182 import HERE, ROOT, YOLO, cv2, np, r64, rt, rte, trackeval

ARCHIVE = Path("F:/LABOTARY_data/goc_tai_ve/svmot_data_v1.0.zip")
RELEASE_MD5 = "0f8ef94e30e2b492411fadee8085ee10"
OUT = HERE / "svmot30"
TEST = set(r64.TAP["svmot"]["videos"])
SPLIT = "SVMOT-release21"


def catalog(archive):
    videos = {}
    for name in archive.namelist():
        parts = name.split("/")
        if len(parts) != 3 or parts[0] != "data" or parts[2] != "seqinfo.ini":
            continue
        video = parts[1]
        ini = configparser.ConfigParser()
        ini.read_string(archive.read(name).decode("utf-8-sig"))
        seq = ini["Sequence"]
        n = seq.getint("seqLength")
        assert [seq.getint(k) for k in ("frameRate", "imHeight", "imWidth")] == [30, 1080, 1920], video
        frames = [f"data/{video}/img/{index:04d}.jpg" for index in range(1, n + 1)]
        for frame in frames:
            archive.getinfo(frame)
        gt = np.loadtxt(io.BytesIO(archive.read(f"data/{video}/gt/gt.txt")), delimiter=",", ndmin=2)
        r64.kiem_gt(gt, n, video)
        gt[:, 7] = 1
        videos[video] = frames, gt
    assert len(videos) == 30 and sum(len(frames) for frames, _ in videos.values()) == 26900
    assert TEST <= set(videos) and len(TEST) == 9
    return videos


class Cache:
    orig_shape = (1080, 1920)

    def __init__(self, detections):
        self.detections = detections

    def iter_frames(self):
        for index, boxes in enumerate(self.detections):
            keep = (boxes[:, 4] > .25) & np.isin(boxes[:, 5], r64.LOP_TRACKER)
            yield index + 1, index / 30, False, boxes[keep][:300]


def model_detections(model, archive, frames):
    detections = []
    for start in range(0, len(frames), 4):
        images = [cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), cv2.IMREAD_COLOR)
                  for name in frames[start:start + 4]]
        assert all(img is not None and img.shape[:2] == Cache.orig_shape for img in images)
        results = model.predict(images, imgsz=640, conf=.05, iou=.7, max_det=3000,
                                device=0, half=False, rect=True, verbose=False)
        assert len(results) == len(images)
        for result in results:
            boxes = result.boxes.data.cpu().numpy().astype(np.float32)
            assert len(boxes) < 3000 and np.isfinite(boxes).all()
            detections.append(boxes)
    return detections


def score(clips, names):
    sys.path.insert(0, str(HERE.parent / "review99_roundabout_turns"))
    from audit_turns import trace

    ec, dc, mc = rte.build_configs(SPLIT, names[0], None)
    ec.update(PRINT_RESULTS=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
              LOG_ON_ERROR=str(OUT / "error.log"))
    dc.update(GT_FOLDER=str(OUT / "ground_truth"), TRACKERS_FOLDER=str(OUT / "trackers"),
              TRACKERS_TO_EVAL=names, SEQ_INFO={v: len(frames) for v, (frames, _) in clips.items()})
    metrics = [trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)]
    dataset = trackeval.datasets.MotChallenge2DBox(dc)
    result, messages = trackeval.Evaluator(ec).evaluate([dataset], metrics)
    assert all(m == "Success" for m in next(iter(messages.values())).values())
    visibility = {}
    for video, (frames, gt) in clips.items():
        ids = np.unique(gt[:, 1]).astype(int)
        at = gt[:, 0].astype(int) - 1, np.searchsorted(ids, gt[:, 1])
        assert np.all((0 <= gt[:, 8]) & (gt[:, 8] <= 1))
        present = np.zeros((len(frames), len(ids)), dtype=bool)
        low = present.copy()
        present[at] = True
        low[at] = gt[:, 8] <= .5
        visibility[video] = ids, present, low

    rows = []
    for name in names:
        scores = rte.flatten_results(result, name)
        scores.to_csv(OUT / f"release21_{name}.csv", index=False)
        expected = scores.set_index("video")
        for video, (ids, present, low) in visibility.items():
            raw = dataset.get_raw_seq_data(name, video)
            data = dataset.get_preprocessed_seq_data(raw, "pedestrian")
            assert all(np.array_equal(raw_ids, ids[compact]) for raw_ids, compact in zip(raw["gt_ids"], data["gt_ids"]))
            matched, swapped = trace(data, present.shape)
            missed = present & ~matched
            assert swapped.sum() == expected.loc[video, "IDSW"]
            assert missed.sum() == expected.loc[video, "FN"]
            for label, mask in (("visibility_le50", low), ("visibility_gt50", present & ~low)):
                rows.append({"video": video, "filter": name, "group": label, "gt_boxes": int(mask.sum()),
                             "IDSW": int((swapped & mask).sum()), "FN": int((missed & mask).sum())})
    with (OUT / "release21_visibility.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    with ARCHIVE.open("rb") as stream:
        assert hashlib.file_digest(stream, "md5").hexdigest() == RELEASE_MD5
    with ZipFile(ARCHIVE) as archive:
        videos = catalog(archive)
        summary = {v: {"frames": len(frames), "gt_boxes": len(gt), "ids": len(np.unique(gt[:, 1])),
                      "split": "official_test" if v in TEST else "release_train_descriptive"}
                   for v, (frames, gt) in videos.items()}
        (OUT / "manifest.json").write_text(json.dumps({"archive_md5": RELEASE_MD5, "videos": summary}, indent=2),
                                           encoding="utf-8")
        print("SVMOT_ARCHIVE_VERIFIED", len(videos), flush=True)
        config = r64.detector_r54("yolov8n_coco")
        checkpoint = ROOT / config["checkpoint"]
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == config["sha256"]
        model = YOLO(str(checkpoint))
        modes = {kind if seed is None else f"{kind}-s{seed}": (rt.load_tracker_cfg(path, {"track_buffer": 36}), seed)
                 for kind, (path, seeds) in r64.BO_LOC.items() for seed in (seeds or [None])}
        others = {v: videos[v] for v in sorted(set(videos) - TEST)}
        for i, (video, (frames, gt)) in enumerate(others.items(), 1):
            gt_folder = OUT / "ground_truth" / SPLIT / video / "gt"
            gt_folder.mkdir(parents=True, exist_ok=True)
            np.savetxt(gt_folder / "gt.txt", gt[np.lexsort((gt[:, 1], gt[:, 0]))], delimiter=",",
                       fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%d", "%d", "%.3f"])
            cache = OUT / "cache" / f"{video}.npz"
            cache.parent.mkdir(exist_ok=True)
            if cache.exists():
                with np.load(cache) as data:
                    boxes = [data[str(j)] for j in range(len(frames))]
            else:
                boxes = model_detections(model, archive, frames)
                temporary = cache.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **{str(j): box for j, box in enumerate(boxes)})
                temporary.replace(cache)
            for name, (cfg, seed) in modes.items():
                dest = OUT / "trackers" / SPLIT / name / "data" / f"{video}.txt"
                dest.parent.mkdir(parents=True, exist_ok=True)
                if dest.exists():
                    continue
                mot, _, timing = rt.track_video(Cache(boxes), cfg, dt_nominal=1/30, elapsed_mode="steps",
                                                seed=seed, video=video)
                assert len(timing) == len(frames) and not mot.duplicated(["frame", "id"]).any()
                temporary = dest.with_suffix(".tmp")
                mot.to_csv(temporary, index=False, header=False, float_format="%.10f")
                temporary.replace(dest)
            print(i, len(others), video, flush=True)
        score(others, list(modes))
        print("SVMOT30_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
