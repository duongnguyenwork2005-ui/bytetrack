"""R63 G4: cham TrackEval mot tracker (goi run_trackeval.run + flatten_results nhu CLI) va LUU them raw_fields.json
(boot_stats.save_raw_fields) de bootstrap ghep cap theo video bang dung cong thuc COMBINED_SEQ.

    python handoff/logs/review63/cong_cu/g4_cham.py --split-name DETRAC-all --tracker <ten> --videos v1 v2 ...
Ghi results/trackeval/<split>/<ten>/{per_video_metrics,combined_metrics}.csv + raw_fields.json. Tu choi neu
thu muc dau ra da co.
"""
import argparse
import sys
from pathlib import Path

GOC = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(GOC / "src"))

import boot_stats  # noqa: E402
import config  # noqa: E402
import run_trackeval as rte  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-name", required=True)
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--dataset", default="detrac")
    a = ap.parse_args()
    out = config.RESULTS_DIR / "trackeval" / a.split_name / a.tracker
    if out.exists():
        print("[LOI] da ton tai", out)
        return 2
    lop = rte.dataset_class_for(a.dataset, a.split_name)
    results, _ = rte.run(a.split_name, a.tracker, a.videos, lop)
    df = rte.flatten_results(results, a.tracker)
    out.mkdir(parents=True)
    df[df["video"] != "COMBINED_SEQ"].sort_values("video").to_csv(out / "per_video_metrics.csv", index=False)
    comb = df[df["video"] == "COMBINED_SEQ"]
    comb.to_csv(out / "combined_metrics.csv", index=False)
    boot_stats.save_raw_fields(results, a.tracker, out / "raw_fields.json")
    c = comb.iloc[0]
    print("[g4_cham] %s (%s) HOTA %.4f IDF1 %.4f IDSW %d -> %s" % (a.tracker, lop, c["HOTA"], c["IDF1"], int(c["IDSW"]), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
