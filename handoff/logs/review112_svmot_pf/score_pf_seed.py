"""Score one completed seed while independent replay seeds continue."""

import argparse
import json
from pathlib import Path
import run_pf as setup
import trackeval

runner, OUT = setup.runner, setup.OUT


def main(seed):
    tracker = f"candidate-s{seed}"
    target = OUT / "preview" / tracker
    target.mkdir(parents=True, exist_ok=True)
    assert all((OUT / "dev/tracking" / runner.SPLIT / tracker / "data" / f"{v}.txt").is_file()
               for v in runner.DEV)
    ec, dc, mc = runner.rte.build_configs(runner.SPLIT, tracker, runner.DEV)
    ec.update(PRINT_RESULTS=False, PRINT_CONFIG=False, OUTPUT_SUMMARY=False, OUTPUT_DETAILED=False,
              LOG_ON_ERROR=str(target / "error.log"))
    dc.update(TRACKERS_FOLDER=str(OUT / "dev/tracking"), OUTPUT_FOLDER=str(target))
    metrics = [trackeval.metrics.HOTA(mc), trackeval.metrics.CLEAR(mc), trackeval.metrics.Identity(mc)]
    dataset = trackeval.datasets.MotChallenge2DBox(dc)
    result, messages = trackeval.Evaluator(ec).evaluate([dataset], metrics)
    assert all(value == "Success" for value in next(iter(messages.values())).values())
    raw = runner.boot_stats.save_raw_fields(result, tracker, target / "raw_fields.json")
    combined = runner.boot_stats.combine(raw, runner.DEV)
    (target / "combined.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(json.dumps(combined, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, choices=(0, 1, 2))
    main(parser.parse_args().seed)
