"""Regression for N23-F1. Run from the project root with OpenCV available.

    python /path/to/CODEX_REPRO_1d5ffaa_render_counter.py

Uses generated images only. Writes evidence to a fresh temporary directory.
Expected: correct video accepted; video with 27 wrong counters rejected.
On target 1d5ffaa both videos are accepted, so this regression exits 1.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import cv2

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import render_mot_video as rmv
import test_render_mot_video as fixture

output = Path(tempfile.mkdtemp(prefix="codex_n23_counter_"))
images = fixture.tao(output, cv2)
boxes = rmv.doc_mot(output / "mot.txt")
classes = rmv.doc_lop(output / "cls.csv")
results = []
for mode in ("correct", "wrong_counter"):
    video = output / (mode + ".mp4")
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (320, 240))
    assert writer.isOpened()
    changed = 0
    for frame_id, path in enumerate(rmv.liet_ke(images), 1):
        original = cv2.imread(str(path))
        frame = original.copy()
        rmv.ve_frame(cv2, frame, boxes.get(frame_id, []), classes, frame_id)
        if mode == "wrong_counter" and boxes.get(frame_id):
            # Counter region only; object labels start below this rectangle.
            frame[:24, :120] = original[:24, :120]
            cv2.putText(frame, "f999", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2, cv2.LINE_AA)
            changed += 1
        writer.write(frame)
    writer.release()
    report = output / (mode + "_check.json")
    command = [sys.executable, "src/kiem_video_render.py", "--video", str(video),
               "--images", str(images), "--mot", str(output / "mot.txt"),
               "--cls", str(output / "cls.csv"), "--fps", "25", "--buoc-lop", "3",
               "--json", str(report)]
    process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    (output / (mode + "_check.log")).write_text(process.stdout + process.stderr)
    results.append({"mode": mode, "wrong_counter_frames": changed,
                    "exit": process.returncode,
                    "checker": json.loads(report.read_text())})
print("Evidence directory:", output)
print(json.dumps(results, ensure_ascii=False, indent=2))
(output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
passed = results[0]["exit"] == 0 and results[1]["exit"] == 1
print("REGRESSION:", "PASS" if passed else "FAIL: wrong counter video was not rejected")
sys.exit(0 if passed else 1)
