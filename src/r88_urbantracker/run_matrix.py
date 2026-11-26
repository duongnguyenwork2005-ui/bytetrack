"""Resumeable R88 two-scene, two-confidence tracker matrix."""
from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path

from run import EVIDENCE, FILTERS, SCENES, evaluate, track

PROGRESS = EVIDENCE / "matrix_progress.json"


def main():
    state=json.loads(PROGRESS.read_text(encoding="utf-8")) if PROGRESS.exists() else {}
    for scene in SCENES:
        for conf in ("c25","c10"):
            for filt in FILTERS:
                key=f"{scene}/{conf}/{filt}"
                if state.get(key,{}).get("status")=="done":
                    print("REUSE",key,flush=True)
                    continue
                print("START",key,datetime.now().isoformat(),flush=True)
                try:
                    track(scene,filt,conf)
                    metrics=evaluate(scene,filt,conf)
                    state[key]={"status":"done","metrics":metrics,"finished":datetime.now().isoformat()}
                except Exception as exc:
                    state[key]={"status":"failed","error":repr(exc),"traceback":traceback.format_exc(),
                                "finished":datetime.now().isoformat()}
                    PROGRESS.write_text(json.dumps(state,indent=2),encoding="utf-8")
                    raise
                PROGRESS.write_text(json.dumps(state,indent=2),encoding="utf-8")
                print("DONE",key,flush=True)
    print("ALL DONE",len(state),flush=True)


if __name__=="__main__":main()
