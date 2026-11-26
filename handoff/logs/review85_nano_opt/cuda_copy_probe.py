"""Nano CUDA transfer microprobe: pageable vs pinned host memory, no TRT."""
import ctypes as ct
import json
import time

import numpy as np


CUDA = ct.CDLL("/usr/local/cuda/lib64/libcudart.so")
CUDA.cudaMalloc.argtypes = (ct.POINTER(ct.c_void_p), ct.c_size_t)
CUDA.cudaMallocHost.argtypes = (ct.POINTER(ct.c_void_p), ct.c_size_t)
CUDA.cudaMemcpy.argtypes = (ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int)
CUDA.cudaFree.argtypes = (ct.c_void_p,)
CUDA.cudaFreeHost.argtypes = (ct.c_void_p,)


def check(rc, name):
    if rc:
        raise RuntimeError("{}: CUDA error {}".format(name, rc))


def measure(dst, src, size, direction, reps=300):
    for _ in range(20):
        check(CUDA.cudaMemcpy(dst, src, size, direction), "warmup memcpy")
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        check(CUDA.cudaMemcpy(dst, src, size, direction), "memcpy")
        samples.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": float(np.percentile(samples, 50)), "p90_ms": float(np.percentile(samples, 90))}


def main():
    sizes = {"input": 1 * 3 * 384 * 640 * 4, "output": 1 * 84 * 5040 * 4}
    out = {}
    for label, size in sizes.items():
        dev = ct.c_void_p()
        host = ct.c_void_p()
        check(CUDA.cudaMalloc(ct.byref(dev), size), "cudaMalloc")
        check(CUDA.cudaMallocHost(ct.byref(host), size), "cudaMallocHost")
        page = np.arange(size, dtype=np.uint8)
        try:
            page_ptr = page.ctypes.data_as(ct.c_void_p)
            row = {}
            for kind in ("pageable", "pinned", "pinned", "pageable"):
                ptr = page_ptr if kind == "pageable" else host
                h2d = measure(dev, ptr, size, 1)
                d2h = measure(ptr, dev, size, 2)
                row.setdefault(kind, []).append({"h2d": h2d, "d2h": d2h})
            out[label] = {"bytes": size, "passes": row}
        finally:
            check(CUDA.cudaFreeHost(host), "cudaFreeHost")
            check(CUDA.cudaFree(dev), "cudaFree")
    print(json.dumps(out, sort_keys=True))


if __name__ == "__main__":
    main()
