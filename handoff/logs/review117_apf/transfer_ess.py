"""Fixed-parameter transfer for the ESS-triggered APF variant."""

import json
import sys
import transfer as job


def selected():
    result = json.loads((job.HERE / "dev_summary_ess.json").read_text())
    row = result["candidates"]["apf_ess_k2"]
    assert row["mean"]["IDSW"] < result["baseline"]["IDSW"]
    assert all(row["gate"][key] for key in ("HOTA", "FN", "seeds"))
    return "apf_ess_k2"


job.WORK_TAG = "evaluation_r117_ess"
job.SCORE_ROOT = job.HERE / "fasttrack_ess"
job.selection = selected

if __name__ == "__main__":
    if sys.argv[1] == "summary":
        job.summarize()
    else:
        job.seed_run(int(sys.argv[1]))
