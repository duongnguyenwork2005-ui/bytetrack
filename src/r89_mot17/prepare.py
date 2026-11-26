"""Prepare seven unique MOT17 training sequences with publisher GT unchanged."""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import config

ARCHIVE=Path(r"F:\LABOTARY_data\goc_tai_ve\MOT17-R89\MOT17Labels.zip")
SPLIT="MOT17R89-train"
EVIDENCE=ROOT/"handoff/logs/review89_mot17"
SEQS=[f"MOT17-{x}-FRCNN" for x in ("02","04","05","09","10","11","13")]


def digest(b):return hashlib.sha256(b).hexdigest()


def main():
    out=config.PROCESSED_DIR/SPLIT
    seqmap=config.PROCESSED_DIR/"seqmaps"/f"{SPLIT}.txt"
    if out.exists() or seqmap.exists():raise RuntimeError("R89 processed namespace exists")
    manifest={}
    with zipfile.ZipFile(ARCHIVE) as z:
        for seq in SEQS:
            pre=seq[:-6]
            variants=[f"MOT17-{pre[-2:]}-{s}" for s in ("DPM","FRCNN","SDP")]
            gts=[z.read(f"train/{v}/gt/gt.txt") for v in variants]
            assert len({digest(b) for b in gts})==1,"GT differs among detector copies"
            ini=z.read(f"train/{seq}/seqinfo.ini")
            dest=out/seq
            (dest/"gt").mkdir(parents=True)
            (dest/"gt"/"gt.txt").write_bytes(gts[1])
            (dest/"seqinfo.ini").write_bytes(ini)
            manifest[seq]={"source_gt":f"train/{seq}/gt/gt.txt","source_seqinfo":f"train/{seq}/seqinfo.ini",
                           "gt_sha256":digest(gts[1]),"seqinfo_sha256":digest(ini),"duplicate_gt_variants":variants}
    seqmap.write_text("name\n"+"\n".join(SEQS)+"\n",encoding="utf-8")
    EVIDENCE.mkdir(parents=True,exist_ok=True)
    (EVIDENCE/"gt_manifest.json").write_text(json.dumps({"split":SPLIT,"sequences":manifest},indent=2),encoding="utf-8")
    print("PREPARED",len(manifest),"unique sequences",SPLIT)


if __name__=="__main__":main()
