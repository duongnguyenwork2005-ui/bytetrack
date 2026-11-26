"""Pixel-exact uint8 preprocessing check for the C1+LUT candidate."""
import importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


base = load("base", HERE.parent / "review63/v17_c/src/nano_detector.py")
lut = load("lut", HERE / "v20/src/nano_detector.py")
combo = load("combo", HERE / "v21/src/nano_detector.py")
rng = np.random.default_rng(83)
for shape in ((384, 640, 3), (640, 640, 3), (32, 48, 3)):
    img = rng.integers(0, 256, shape, dtype=np.uint8)
    a, b = base.preprocess(img), lut.preprocess(img)
    assert a.dtype == b.dtype == np.float32
    assert np.array_equal(a, b), shape
    assert np.array_equal(a, combo.preprocess(img)), shape
print("PASS: tensor preprocessing byte-equal in 3 shapes")
