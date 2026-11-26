"""R85: materialize every hardware decoded MP4 frame as a CPU BGR NumPy array."""
import json
import time

import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402


SRC = "/home/jetson/nano_run/v16/nano_transfer/data/smoke/MVI_39031_f0001-0300.mp4"
Gst.init(None)
pipeline = Gst.parse_launch(
    "filesrc location={} ! qtdemux ! mpeg4videoparse ! nvv4l2decoder ! "
    "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! "
    "video/x-raw,format=BGR ! appsink name=frames sync=false emit-signals=false "
    "max-buffers=4 drop=false".format(SRC))
sink = pipeline.get_by_name("frames")
if sink is None:
    raise RuntimeError("appsink absent")
pull_ms, copy_ms, checksums = [], [], []
t0 = time.perf_counter()
status = pipeline.set_state(Gst.State.PLAYING)
if status == Gst.StateChangeReturn.FAILURE:
    raise RuntimeError("pipeline failed to start")
try:
    for frame in range(300):
        p0 = time.perf_counter()
        sample = sink.emit("try-pull-sample", 10 * Gst.SECOND)
        p1 = time.perf_counter()
        if sample is None:
            raise RuntimeError("no frame {}".format(frame + 1))
        caps = sample.get_caps().get_structure(0)
        w, h = int(caps.get_value("width")), int(caps.get_value("height"))
        if (w, h) != (960, 540):
            raise RuntimeError("wrong shape {}x{}".format(w, h))
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("map failed")
        try:
            stride = len(info.data) // h
            if stride < w * 3 or stride * h != len(info.data):
                raise RuntimeError("unexpected BGR stride {} length {}".format(stride, len(info.data)))
            image = np.frombuffer(info.data, dtype=np.uint8).reshape(h, stride)[:, :w * 3].copy().reshape(h, w, 3)
            if frame in (0, 149, 299):
                checksums.append([frame + 1, int(image.sum())])
        finally:
            buf.unmap(info)
        p2 = time.perf_counter()
        pull_ms.append((p1 - p0) * 1000)
        copy_ms.append((p2 - p1) * 1000)
finally:
    pipeline.set_state(Gst.State.NULL)
elapsed = time.perf_counter() - t0
print(json.dumps({"frames": len(pull_ms), "wall_seconds": elapsed,
                  "ms_per_frame": elapsed * 1000 / len(pull_ms),
                  "pull_p50_ms": float(np.median(pull_ms)),
                  "copy_p50_ms": float(np.median(copy_ms)),
                  "checksums": checksums}))
