"""R85 decode all 300 MP4 frames to BGR with the current OpenCV path."""
import json
import time

import cv2


src = "/home/jetson/nano_run/v16/nano_transfer/data/smoke/MVI_39031_f0001-0300.mp4"
t0 = time.perf_counter()
cap = cv2.VideoCapture(src)
if not cap.isOpened():
    raise RuntimeError("VideoCapture failed")
n = 0
checksums = []
while True:
    ok, image = cap.read()
    if not ok:
        break
    n += 1
    if image is None or image.shape != (540, 960, 3):
        raise RuntimeError("Wrong frame shape")
    if n in (1, 150, 300):
        checksums.append([n, int(image.sum())])
cap.release()
elapsed = time.perf_counter() - t0
if n != 300:
    raise RuntimeError("Expected 300 frames, got {}".format(n))
print(json.dumps({"frames": n, "wall_seconds": elapsed, "ms_per_frame": elapsed * 1000 / n,
                  "opencv": cv2.__version__, "checksums": checksums}))
