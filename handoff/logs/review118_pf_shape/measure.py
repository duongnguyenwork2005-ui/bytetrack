"""Serial timing comparison on identical 512-frame prefixes."""

import numpy as np
import run as job
from automatic import installed as auto
from cartesian import installed as two


def run():
    rows, signatures = [], {}
    for repeat in range(3):
        order = ("parent", "two_observations", "automatic")
        if repeat % 2:
            order = tuple(reversed(order))
        for video in job.base.SMOKE:
            for mode in order:
                job.installed = auto if mode == "automatic" else two
                mot, timing = job.replay(video, 0, mode != "parent", length=512)
                tag = video, mode
                values = mot.to_numpy(float)
                if tag in signatures:
                    assert np.array_equal(values, signatures[tag])
                else:
                    signatures[tag] = values
                measured = timing.t_update_ms.to_numpy()[32:]
                rows.append({"repeat": repeat, "video": video, "mode": mode,
                             "mean_update_ms": float(measured.mean()),
                             "p90_update_ms": float(np.quantile(measured, .9)), "frames": len(measured)})
                print(rows[-1], flush=True)
    summary = {mode: float(np.mean([r["mean_update_ms"] for r in rows if r["mode"] == mode]))
               for mode in ("parent", "two_observations", "automatic")}
    job.write(job.HERE / "serial_timing.json", {"units": rows, "mean_update_ms": summary,
                                             "scope": "PC tracker update on two 512-frame prefixes, serial, 32 warmup"})
    print(summary, flush=True)


if __name__ == "__main__":
    run()
