"""Serial tracker-only timings on the same prefixes, with output repeat checks."""

import numpy as np

from run import HERE, previous, replay, write


def run():
    modes = ("base", "apf_k2", "apf_ess_k2")
    result = []
    reference = {}
    for repeat in range(3):
        order = modes if repeat % 2 == 0 else tuple(reversed(modes))
        for video in previous.SMOKE:
            for mode in order:
                mot, _, timing, events = replay(video, 0, mode, 512)
                values = mot.to_numpy(float)
                key = video, mode
                if key in reference:
                    assert np.array_equal(reference[key], values)
                else:
                    reference[key] = values
                measured = timing.t_update_ms.to_numpy()[32:]
                row = {"video": video, "mode": mode, "repeat": repeat, "mean_update_ms": float(measured.mean()),
                       "p90_update_ms": float(np.quantile(measured, .9)), "measured_frames": len(measured)}
                result.append(row)
                print(row, flush=True)
    summary = {mode: float(np.mean([r["mean_update_ms"] for r in result if r["mode"] == mode])) for mode in modes}
    write(HERE / "serial_timing.json", {"units": result, "mode_mean_ms": summary,
                                       "scope": "PC tracker update only, two prefixes, 32 warmup frames, serial alternating order"})
    print(summary, flush=True)


if __name__ == "__main__":
    run()
