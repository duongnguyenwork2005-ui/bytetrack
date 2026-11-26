"""Check bounded prefetch preserves source order/content and can stop early."""
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent / "v19/src"))
from nano_detect_stream import Nguon  # noqa: E402


with tempfile.TemporaryDirectory() as tmp:
    for i in range(1, 11):
        im = np.full((32, 48, 3), i * 10, np.uint8)
        assert cv2.imwrite(str(Path(tmp) / ("img%05d.jpg" % i)), im)
    os.environ["NANO_PREFETCH"] = "0"
    src = Nguon(tmp, 25, None)
    expected = [(idx, t, im.copy()) for idx, t, im, _, _ in src]
    src.dong()
    os.environ["NANO_PREFETCH"] = "1"
    src = Nguon(tmp, 25, None)
    actual = [(idx, t, im.copy()) for idx, t, im, _, _ in src]
    src.dong()
    assert len(actual) == len(expected) == 10
    assert all((x[0], x[1]) == (y[0], y[1]) and np.array_equal(x[2], y[2])
               for x, y in zip(expected, actual))
    src = Nguon(tmp, 25, None)
    it = iter(src)
    next(it)
    it.close()
    src.dong()
print("PASS: 10 ordered frames, pixel equality, early stop")
