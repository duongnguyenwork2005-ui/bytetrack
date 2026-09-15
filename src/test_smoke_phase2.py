"""
test_smoke_phase2.py -- Smoke test end-to-end Giai doan 2 tren fixture BDD100K (khong anh, khong GPU):
    fixture JSON -> check_bdd100k (manifest) -> export MOTChallenge (vehicle) -> det_cache --source gt
    -> run_experiment (cv, ekf, pf50 x 2 seed, sort_ekf30; n_steps tu timestamp; track_buffer 1.2 s = 6 buoc)
    -> TrackEval crowd-ignore -> occlusion_eval --gaps -> aggregate_phase2.
Kiem tra file ket qua ton tai, chi so huu han, PF gop 2 seed, chay lai -> bit-identical.

    python src/test_smoke_phase2.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from mot_datasets.bdd100k_mot import BDD100KMOTDataset, export_motchallenge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "data/fixtures/bdd100k_mini"
PY = sys.executable
EXP = ROOT / "configs/phase2/experiments/smoke_bddfx.yaml"
_RESULTS: list[tuple[str, bool]] = []


def _check(name, ok, detail=""):
    _RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in str(detail).strip().splitlines():
            print(f"         {line}")
    assert ok, f"{name}\n{detail}"


def run(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr)[-1500:]


def main() -> int:
    failed = 0
    try:
        rc, out = run([PY, "src/mot_datasets/make_bdd100k_fixture.py"]); _check("make fixture", rc == 0, out)
        rc, out = run([PY, "src/check_bdd100k.py", "--part", "val", "--data-root", str(FIX), "--no-images", "--rebuild"])
        _check("check_bdd100k tren fixture (manifest fixture_bdd100k_final_val.json)", rc == 0 and
               (ROOT / "results/phase2/manifests/fixture_bdd100k_final_val.json").exists(), out)
        ds = BDD100KMOTDataset(FIX); df = ds.gt_dataframe("val")
        for split in list(config.PROCESSED_DIR.glob("BDDFX-*")) + list((config.PROCESSED_DIR / "trackers").glob("BDDFX-*")):
            shutil.rmtree(split)
        r = export_motchallenge(df, "val", "vehicle", benchmark="BDDFX")
        _check("export MOT vehicle BDDFX-val", r["n_videos"] == 3)
        cdir = config.INTERIM_DIR / "det_cache" / "fixture_bddfx_gt"
        if cdir.exists():
            shutil.rmtree(cdir)
        rc, out = run([PY, "src/det_cache.py", "--dataset", "bdd100k", "--data-root", str(FIX), "--part", "val", "--all",
                       "--cache-id", "fixture_bddfx_gt", "--source", "gt", "--gt-jitter", "0.02", "--gt-dropout", "0.1", "--gt-seed", "0"])
        _check("det_cache gt-synthetic", rc == 0, out)
        shutil.rmtree(ROOT / "results/phase2/smoke_bddfx", ignore_errors=True)
        for d in (config.RESULTS_DIR / "trackeval" / "BDDFX-val").glob("p2-smoke_bddfx-*"):
            shutil.rmtree(d)
        rc, out = run([PY, "src/run_experiment.py", "--exp", str(EXP), "--no-skip"])
        _check("run_experiment smoke_bddfx (track + eval + aggregate)", rc == 0, out)
        out_dir = ROOT / "results/phase2/smoke_bddfx/A"
        summ = pd.read_csv(out_dir / "summary.csv")
        _check("summary.csv: 4 tracker, HOTA huu han, pf50 co n_seeds = 2",
               len(summ) == 4 and np.isfinite(summ.HOTA).all() and int(summ[summ.tracker == "pf50"].n_seeds.iloc[0]) == 2,
               summ.to_string())
        boot = pd.read_csv(out_dir / "paired_bootstrap.csv")
        _check("paired_bootstrap.csv: 3 tracker x 5 metric vs baseline cv", len(boot) == 15 and set(boot.baseline) == {"cv"})
        _check("timing.csv + event_outcomes (gap) ton tai", (out_dir / "timing.csv").exists()
               and (out_dir / "event_outcomes_BDDFX-val.csv").exists())
        eo = pd.read_csv(out_dir / "event_outcomes_BDDFX-val.csv")
        _check("occlusion_eval: co su kien gap va outcome cho moi tracker (5 dir: cv, ekf, pf50-s0, pf50-s1, sortekf)",
               eo.has_gap.any() and eo.tracker.nunique() == 5, str(eo.tracker.unique()))
        meta = pd.read_json(config.PROCESSED_DIR / "trackers/BDDFX-val/p2-smoke_bddfx-A-ekf/run_meta.json", typ="series")
        _check("run_meta: track_buffer 1.2 s @5fps = 6 buoc, elapsed steps", int(meta["track_buffer_frames"]) == 6
               and meta["elapsed_mode"] == "steps")
        f = config.PROCESSED_DIR / "trackers/BDDFX-val/p2-smoke_bddfx-A-pf50-s0/data/b1c66a42-6f7d68ca.txt"
        t0 = f.read_text()
        rc, out = run([PY, "src/run_experiment.py", "--exp", str(EXP), "--stage", "track", "--no-skip"])
        _check("chay lai: PF seed 0 bit-identical", rc == 0 and f.read_text() == t0)
    except AssertionError as e:
        failed += 1
        print(f"  !! {e}")
    n_ok = sum(ok for _, ok in _RESULTS)
    print("\n" + "=" * 78)
    print(f"KET QUA: {n_ok}/{len(_RESULTS)} phep kiem tra PASS")
    print("=" * 78)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
