"""
check_identical_runs.py -- So sanh output tracking cua hai lan chay (thu muc tracker) tren cung tap video:
bit-identical (van ban) + song anh ID toan sequence (track_compare.id_bijection). Dung de chung minh
mot thay doi "khong doi hanh vi" (vd bat assoc_log, doi ten thi nghiem, mixin tat) tren du lieu that.

    python src/check_identical_runs.py --split DETRAC-all \
        --a p2-groupA_detrac_dev-A-cv --b p2-expA_detrac_dev-c25-cv --manifest results/phase2/manifests/detrac_dev.json
    python src/check_identical_runs.py --pairs-from configs/phase2/experiments/expA_detrac_dev.yaml \
        --map "groupA_detrac_dev:A=c25,B=c10"      # so moi tracker cua exp voi ban moc 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from track_compare import id_bijection, read_mot  # noqa: E402


def compare_dirs(split: str, a: str, b: str, videos: list[str], tol: float = 0.0) -> pd.DataFrame:
    root = config.PROCESSED_DIR / "trackers" / split
    rows = []
    for v in videos:
        fa, fb = root / a / "data" / f"{v}.txt", root / b / "data" / f"{v}.txt"
        ta = fa.read_bytes() if fa.exists() else None
        tb = fb.read_bytes() if fb.exists() else None
        r = id_bijection(read_mot(fa), read_mot(fb), tol=tol)
        rows.append({"video": v, "a": a, "b": b, "bit_identical": ta is not None and ta == tb,
                     "id_bijection": r.is_bijection, "id_identity": r.is_identity, "rows_a": r.n_rows_a, "rows_b": r.n_rows_b,
                     "unmatched": r.n_unmatched_a + r.n_unmatched_b, "ids_a": r.n_ids_a, "ids_b": r.n_ids_b})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="DETRAC-all")
    ap.add_argument("--a"); ap.add_argument("--b")
    ap.add_argument("--manifest", default="results/phase2/manifests/detrac_dev.json")
    ap.add_argument("--pairs-from", help="exp yaml: so p2-<exp>-<cond>-<tracker> voi p2-<old>-<cond_old>-<tracker>")
    ap.add_argument("--map", default="groupA_detrac_dev:A=c25,B=c10", help="<old_exp>:<old_cond>=<new_cond>,...")
    ap.add_argument("--tol", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    videos = json.loads(Path(a.manifest).read_text(encoding="utf-8"))["videos"]
    dfs = []
    if a.pairs_from:
        from run_experiment import load_exp, plan
        exp = load_exp(a.pairs_from)
        old_exp, m = a.map.split(":")
        cmap = {kv.split("=")[1]: kv.split("=")[0] for kv in m.split(",")}
        for r in plan(exp):
            old_cond = cmap.get(r["cond"])
            if old_cond is None:
                continue
            old = f"p2-{old_exp}-{old_cond}-{r['tracker']}" + (f"-s{r['seed']}" if r["seed"] is not None else "")
            if not (config.PROCESSED_DIR / "trackers" / exp["split_name"] / old).is_dir():
                continue
            dfs.append(compare_dirs(exp["split_name"], old, r["dir"], videos, a.tol))
    else:
        dfs.append(compare_dirs(a.split, a.a, a.b, videos, a.tol))
    df = pd.concat(dfs, ignore_index=True)
    pd.set_option("display.width", 250)
    g = df.groupby(["a", "b"]).agg(n_video=("video", "size"), bit_identical=("bit_identical", "sum"),
                                   id_bijection=("id_bijection", "sum"), id_identity=("id_identity", "sum")).reset_index()
    print(g.to_string(index=False))
    n = len(df)
    print(f"\nTONG (don vi dem = cap (lan chay, video): {g.shape[0]} cap thu muc x {df.video.nunique()} video = {n}):")
    print(f"  bit-identical (so sanh BYTE cua file, khong chuan hoa): {int(df.bit_identical.sum())}/{n}")
    print(f"  song anh ID toan sequence (tol={a.tol}): {int(df.id_bijection.sum())}/{n}"
          f"  (trong do anh xa dong nhat: {int(df.id_identity.sum())}/{n})")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); df.to_csv(a.out, index=False)
    return 0 if df.bit_identical.all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
