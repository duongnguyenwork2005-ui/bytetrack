"""Audit a completed HELMET matrix from published MOT, clip markers and evaluation artifacts.

Run only after the human/agent has inspected smoke overlays and recorded the
semantic finding. Any differences fail before a full-review JSON is produced.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import helmet_eval as h
from sher_metric import combine

def audit(phase):
    p=h.prep.LOG/phase
    idx=h.read(p/"phase_inputs.json")
    summary=h.read(p/"tong_hop_trang_thai.json")
    df=pd.read_csv(p/"combined_tung_run.csv")
    h.require(len(df)==summary["runs"]==36 and df.run.nunique()==36 and
              h.prep.sha256(p/"combined_tung_run.csv")==summary["combined_sha256"],"combined coverage")
    n_units=0
    for row in df.itertuples():
        nm=row.run
        tr=h.config.PROCESSED_DIR/"trackers"/idx["split"]/nm
        res=h.config.RESULTS_DIR/"trackeval"/idx["split"]/nm
        cls=h.read(tr/"run_meta.json"); c=h.read(res/"cong_r66.json")
        h.require(c["units"]==cls["units"] and set(c["units"])==set(idx["clips"]),"score/clip seals")
        h.check_files(res,["combined_metrics.csv","per_video_metrics.csv","raw_fields.json"],c["sha256"])
        raw=h.read(res/"raw_fields.json")
        h.require(set(raw)==set(idx["clips"]), "raw clip set")
        recombined=combine(raw,idx["clips"])
        h.require(all(abs(getattr(row,k)-v)<1e-9 for k,v in recombined.items()),"score differs from raw")
        for clip,marker in c["units"].items():
            directory=p/"units"/nm/clip
            cert=h.read(directory/"unit.json")
            h.require(h.prep.sha256(directory/"unit.json")==marker and cert["protocol_hash"]==idx["protocol_hash"],
                      "clip seal differs")
            h.check_files(tr,h.clip_paths(clip),cert["sha256"])
            n_units+=1
    return {"runs":len(df),"units":n_units,"summary_sha256":h.prep.sha256(p/"tong_hop_trang_thai.json")}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase",choices=["smoke","full"])
    ap.add_argument("--semantic-note",type=Path)
    args=ap.parse_args()
    result=audit(args.phase)
    if args.phase=="smoke":
        h.require(args.semantic_note and args.semantic_note.is_file(), "visual review is required")
        p=h.prep.LOG/"smoke"
        recall_path=h.prep.LOG/"smoke_review/recall.json"
        recall=h.read(recall_path)
        h.require(set(recall["detectors"])==set(h.r64.DETECTOR) and len(recall["clips"])==12,"recall missing")
        rep=h.read(p/"repeat_pf.json"); h.require(rep["status"]=="PASS","PF repeat failed")
        result.update(decision="PASS",repeat_pf_sha256=h.prep.sha256(p/"repeat_pf.json"),
            note_sha256=h.prep.sha256(args.semantic_note),recall_sha256=h.prep.sha256(recall_path))
    h.write(h.prep.LOG/args.phase/"final_review.json",result)
    print(json.dumps(result))
if __name__=="__main__":
    main()

