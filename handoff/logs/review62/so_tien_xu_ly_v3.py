#!/usr/bin/env python3
"""R62 v3: compare ONLY uint8 BGR HWC -> float32 RGB NCHW conversion.

One-off investigation; Python 3.6 / NumPy 1.19.5 compatible. Tests OpenCV LUT
support explicitly. No letterbox, engine, postprocessing, threads, device
settings or production edits.
Print JSON to stdout (redirect to a NEW file). Timing data do not establish
an improvement in full-pipeline FPS. Default workload: 4 methods, 8 balanced
rounds, 8 calls per method/round, 640x640 input; stop after ~45 s if incomplete.
"""
from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import platform
import sys
import time

import numpy as np
import cv2


# Compute with exactly the same float32 division as the legacy conversion.
LUT = np.arange(256, dtype=np.float32) / 255.0


def legacy(img):
    x = img[:, :, ::-1].transpose(2, 0, 1)
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return x[None]


def float32_inplace(img):
    x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32)
    x /= 255.0
    return x[None]


def uint8_first(img):
    x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    x = x.astype(np.float32) / 255.0
    return x[None]


def uint8_first_inplace(img):
    x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    x = x.astype(np.float32)
    x /= 255.0
    return x[None]


def lookup(img):
    x = img[:, :, ::-1].transpose(2, 0, 1)
    # uint8 indices are always in [0,255]; clip changes no valid index.
    # ascontiguousarray makes layout part of the tested contract.
    return np.ascontiguousarray(np.take(LUT, x, mode="clip"))[None]


def lookup_uint8_contiguous(img):
    # Advanced indexing accepts uint8 directly, unlike take's index conversion.
    # Materialize the compact uint8 CHW input before the float32 gather.
    x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    return np.ascontiguousarray(LUT[x])[None]


def lookup_strided(img):
    # Gather directly from the view, then enforce the output layout explicitly.
    x = img[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(LUT[x])[None]


def lookup_cv2_hwc(img):
    # cv2.LUT must retain float32 table values exactly, including on OpenCV 3.2.
    # Validate rather than assume compatibility or byte equality.
    x = cv2.LUT(img, LUT)
    return np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1))[None]


def lookup_cv2_flat_chw(img):
    x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))
    # OpenCV sees a single channel column, not a 640-channel CHW tensor.
    y = cv2.LUT(x.reshape(-1, 1), LUT)
    return np.ascontiguousarray(y).reshape(1, 3, img.shape[0], img.shape[1])


METHODS = [("legacy", legacy), ("lookup", lookup),
           ("lookup_cv2_hwc", lookup_cv2_hwc), ("lookup_cv2_flat_chw", lookup_cv2_flat_chw)]


def digest(a):
    return hashlib.sha256(a.tobytes(order="C")).hexdigest()


def validation():
    rng = np.random.RandomState(62028)
    ramp = np.arange(16 * 16 * 3).astype(np.uint8).reshape(16, 16, 3)
    big = rng.randint(0, 256, size=(640, 640, 3)).astype(np.uint8)
    shapes = [(1, 1, 3), (1, 257, 3), (257, 1, 3), (13, 17, 3), (360, 640, 3)]
    cases = [("all_256_values_each_channel", ramp), ("square_640", big),
             ("strided_hw", big[::2, ::2, :]), ("reversed_hwc", big[::-1, ::-1, ::-1]),
             ("transposed_hw", big.transpose(1, 0, 2)),
             ("fortran_hwc", np.asfortranarray(big))]
    cases += [("shape_%dx%d" % s[:2], rng.randint(0, 256, size=s).astype(np.uint8))
              for s in shapes]
    assert all(len(np.unique(ramp[:, :, c])) == 256 for c in range(3))
    records = []
    for name, img in cases:
        initial = digest(img)
        ref = legacy(img)
        ref_bytes = ref.tobytes(order="C")
        row = {"case": name, "input_shape": list(img.shape),
               "input_strides": list(img.strides), "input_c_contiguous": bool(img.flags.c_contiguous),
               "reference_sha256": digest(ref), "methods": {}}
        for method, fn in METHODS:
            try:
                got = fn(img)
            except Exception as exc:
                row["methods"][method] = {"supported": False,
                                          "error": "%s: %s" % (type(exc).__name__, exc)}
                continue
            checks = {"dtype_float32": got.dtype == np.dtype("float32"),
                      "shape_nchw": got.shape == (1, 3, img.shape[0], img.shape[1]),
                      "c_contiguous": bool(got.flags.c_contiguous),
                      "bytes_identical": got.tobytes(order="C") == ref_bytes,
                      "input_unchanged": digest(img) == initial}
            row["methods"][method] = checks
        records.append(row)
    passed = all(all(checks.values()) for row in records for checks in row["methods"].values())
    return passed, records, big


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    if not len(a):
        return None
    med = float(np.median(a))
    return {"n": len(a), "median": med, "min": float(a.min()), "max": float(a.max()),
            "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)), "mad": float(np.median(np.abs(a - med)))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-calls", type=int, default=8)
    ap.add_argument("--cycles", type=int, default=1, help="each cycle has 2*method_count balanced rounds")
    ap.add_argument("--warmup-calls", type=int, default=3)
    ap.add_argument("--max-seconds", type=float, default=45.0)
    a = ap.parse_args()
    if min(a.batch_calls, a.cycles, a.warmup_calls) < 1 or not (0 < a.max_seconds <= 55):
        ap.error("positive counts required; --max-seconds must be in (0,55]")
    started = time.perf_counter()
    passed, validation_rows, img = validation()
    report = {"created_utc": datetime.datetime.utcnow().isoformat() + "Z",
              "scope": "tensor conversion only; no letterbox/decode/engine/post; not pipeline FPS",
              "python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__,
              "opencv_threads": cv2.getNumThreads(), "platform": platform.platform(),
              "script_sha256": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
              "validation_passed": passed, "validation": validation_rows,
              "parameters": vars(a), "input_shape": list(img.shape), "input_sha256": digest(img),
              "allocation": "each call returns its own float32 tensor; LUT initialized once before validation",
              "timing_unit": "ms per call = full timed batch wall / batch calls; includes allocation",
              "raw_batches": [], "balanced_complete": False}
    if not passed:
        report["status"] = "validation_failed"
        print(json.dumps(report, indent=2, sort_keys=True))
        return 2
    # Each method occupies every position twice per cycle: cyclic forward then
    # cyclic reversed orders. Keep every measured batch, even if a time limit hits.
    n = len(METHODS)
    orders = []
    for _ in range(a.cycles):
        for reverse in (False, True):
            base = list(range(n))
            if reverse:
                base.reverse()
            orders.extend([base[k:] + base[:k] for k in range(n)])
    report["planned_orders"] = [[METHODS[i][0] for i in order] for order in orders]
    timed_out = False
    for _, fn in METHODS:
        for _ in range(a.warmup_calls):
            if time.perf_counter() - started >= a.max_seconds:
                timed_out = True
                break
            fn(img)
        if timed_out:
            break
    report["seconds_before_timed_batches"] = time.perf_counter() - started
    if not timed_out:
        for round_idx, order in enumerate(orders, 1):
            for position, method_idx in enumerate(order, 1):
                if time.perf_counter() - started >= a.max_seconds:
                    timed_out = True
                    break
                name, fn = METHODS[method_idx]
                t0 = time.perf_counter()
                for _ in range(a.batch_calls):
                    fn(img)
                batch_ms = (time.perf_counter() - t0) * 1000.0
                report["raw_batches"].append({"round": round_idx, "position": position, "method": name,
                                               "batch_calls": a.batch_calls, "batch_ms": batch_ms,
                                               "ms_per_call": batch_ms / a.batch_calls})
            if timed_out:
                break
    rows = report["raw_batches"]
    report["balanced_complete"] = len(rows) == len(orders) * n
    report["summary_ms_per_call"] = {
        name: stats([r["ms_per_call"] for r in rows if r["method"] == name]) for name, _ in METHODS}
    baseline = {r["round"]: r["ms_per_call"] for r in rows if r["method"] == "legacy"}
    report["within_round_comparison_exploratory"] = {}
    for name, _ in METHODS[1:]:
        paired = [r for r in rows if r["method"] == name and r["round"] in baseline]
        report["within_round_comparison_exploratory"][name] = {
            "baseline_minus_candidate_ms": stats([baseline[r["round"]] - r["ms_per_call"] for r in paired]),
            "baseline_over_candidate_ratio": stats([baseline[r["round"]] / r["ms_per_call"] for r in paired])}
    report["total_seconds"] = time.perf_counter() - started
    report["status"] = "complete" if report["balanced_complete"] else "time_limit_incomplete"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["balanced_complete"] else 3


if __name__ == "__main__":
    sys.exit(main())
