"""Score the new VNTraffic sample, including predeclared turn and straight strata."""
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

from helmet182 import HERE, YOLO, cv2, np, r64, rt, rte, trackeval

sys.path.insert(0, str(HERE.parent / "review99_roundabout_turns"))
from audit_turns import labels_for_camera, trace

ROOT_DATA = Path("F:/LABOTARY_data/goc_tai_ve/VehicleTracking-R108")
OUT = HERE / "vntraffic"
NAME = "VNTraffic"
SPLIT = "VNTRAFFIC-pilot"


class Cache:
    orig_shape = (1080, 1920)

    def __init__(self, boxes):
        self.boxes = boxes

    def iter_frames(self):
        for index, boxes in enumerate(self.boxes):
            yield index + 1, index / 30, False, boxes


def main():
    import torch
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    path = ROOT_DATA / "Vehicle_Tracking.zip"
    assert hashlib.md5(path.read_bytes()).hexdigest() == "a1a511436e55b8168928a6a278ae2ba7"
    with ZipFile(path) as archive:
        source = archive.read(f"{NAME}/{NAME}_GroundTruth.txt")
    gt = np.loadtxt(io.BytesIO(source), delimiter=",", ndmin=2)
    assert gt[:, 0].min() == 0 and gt[:, 0].max() == 500
    gt[:, 0] += 1
    duplicated = r64.dong_trung(gt)
    assert duplicated.sum() == 4
    gt = gt[~duplicated, :9]
    gt[:, 7:9] = 1
    folder = OUT / "ground_truth" / SPLIT / NAME / "gt"
    folder.mkdir(parents=True, exist_ok=True)
    gt_path = folder / "gt.txt"
    np.savetxt(gt_path, gt, delimiter=",", fmt=["%d"]*9)
    ids, present, groups, episodes = labels_for_camera(gt_path, NAME, 501, 30)
    frozen = {"frames": 501, "source_frames": 510, "ids": len(ids), "boxes": len(gt),
              "dropped_conflicting_rows": int(duplicated.sum()), "episodes": episodes,
              "turn25_boxes": int((groups["turn25"] >= 0).sum()),
              "straight8_boxes": int((groups["straight8"] >= 0).sum()),
              "gt_sha256": hashlib.sha256(gt_path.read_bytes()).hexdigest()}
    (OUT / "gt_audit.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print({k: v for k, v in frozen.items() if k != "episodes"}, flush=True)
    checkpoint = r64.detector_r54("yolov8n_coco")
    path = HERE.parents[2] / checkpoint["checkpoint"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint["sha256"]
    model = YOLO(str(path))
    cap = cv2.VideoCapture(str(ROOT_DATA / f"{NAME}_Original-video.mp4"))
    boxes = []
    try:
        assert cap.isOpened() and cap.get(cv2.CAP_PROP_FPS) == 30 and cap.get(cv2.CAP_PROP_FRAME_COUNT) == 510
        while len(boxes) < 501:
            frames = []
            for _ in range(min(4, 501 - len(boxes))):
                ok, frame = cap.read()
                assert ok and frame.shape[:2] == Cache.orig_shape
                frames.append(frame)
            for result in model.predict(frames, imgsz=640, conf=.05, iou=.7, max_det=3000,
                                        rect=True, device="cpu", verbose=False):
                raw = result.boxes.data.numpy().astype(np.float32)
                assert len(raw) < 3000 and np.isfinite(raw).all()
                boxes.append(raw[(raw[:, 4] > .25) & np.isin(raw[:, 5], r64.LOP_TRACKER)][:300])
    finally:
        cap.release()
    assert len(boxes) == 501
    output = {}
    for mode in ("cv", "ekf", "ukf"):
        cfg = rt.load_tracker_cfg(r64.BO_LOC[mode][0], {"track_buffer": 36})
        mot, _, timing = rt.track_video(Cache(boxes), cfg, dt_nominal=1/30, elapsed_mode="steps", video=NAME)
        assert len(timing) == 501 and not mot.duplicated(["frame", "id"]).any()
        destination = OUT / "trackers" / SPLIT / mode / "data" / f"{NAME}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        mot.to_csv(destination, index=False, header=False, float_format="%.10f")
        _, dc, mc = rte.build_configs(SPLIT, mode, None)
        dc.update(GT_FOLDER=str(OUT / "ground_truth"), TRACKERS_FOLDER=str(OUT / "trackers"), SEQ_INFO={NAME: 501})
        dataset = trackeval.datasets.MotChallenge2DBox(dc)
        data = dataset.get_preprocessed_seq_data(dataset.get_raw_seq_data(mode, NAME), "pedestrian")
        matched, changed = trace(data, present.shape)
        clear = trackeval.metrics.CLEAR(mc).eval_sequence(data)
        assert changed.sum() == clear["IDSW"] and (present & ~matched).sum() == clear["CLR_FN"]
        row = {"HOTA": float(trackeval.metrics.HOTA(mc).eval_sequence(data)["HOTA"].mean()),
               "IDF1": float(trackeval.metrics.Identity(mc).eval_sequence(data)["IDF1"])}
        for tag, mask in {"all": present, **{key: values >= 0 for key, values in groups.items()}}.items():
            row[tag] = {"boxes": int(mask.sum()), "IDSW": int((changed & mask).sum()),
                        "FN": int((~matched & mask).sum())}
        output[mode] = row
    (OUT / "metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output), flush=True)


if __name__ == "__main__":
    main()
