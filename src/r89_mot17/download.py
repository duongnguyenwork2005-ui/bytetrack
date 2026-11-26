"""Resumable verified downloads from the official MOTChallenge archive."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import requests

URLS={"labels":"https://motchallenge.net/data/MOT17Labels.zip",
      "images":"https://motchallenge.net/data/MOT17.zip"}
ROOT=Path(r"F:\LABOTARY_data\goc_tai_ve\MOT17-R89")


def sha256(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(4<<20),b""):h.update(chunk)
    return h.hexdigest()


def run(kind):
    ROOT.mkdir(parents=True,exist_ok=True)
    url=URLS[kind]
    final=ROOT/Path(url).name
    part=final.with_name(final.name+".part")
    head=requests.head(url,allow_redirects=True,timeout=60)
    head.raise_for_status()
    expected=int(head.headers["Content-Length"])
    if final.exists():
        if final.stat().st_size!=expected:raise RuntimeError("existing final size mismatch")
    else:
        for attempt in range(12):
            start=part.stat().st_size if part.exists() else 0
            if start>expected:raise RuntimeError("part larger than remote")
            if start==expected:break
            headers={"Range":f"bytes={start}-"} if start else {}
            try:
                with requests.get(url,headers=headers,stream=True,timeout=(30,120)) as resp:
                    resp.raise_for_status()
                    if start and resp.status_code!=206:raise RuntimeError("server ignored Range; leaving partial unchanged")
                    if start and not resp.headers.get("Content-Range","").startswith(f"bytes {start}-"):
                        raise RuntimeError("Content-Range mismatch")
                    mode="ab" if start else "wb"
                    with part.open(mode) as out:
                        last_report=start
                        for chunk in resp.iter_content(4<<20):
                            if not chunk:continue
                            out.write(chunk)
                            if out.tell()-last_report>=256<<20:
                                last_report=out.tell()
                                print(kind,out.tell(),"/",expected,flush=True)
                if part.stat().st_size==expected:break
                print("incomplete HTTP body, retry",part.stat().st_size,flush=True)
            except Exception as e:
                print("retry",attempt+1,repr(e),flush=True)
                time.sleep(min(30,2**min(attempt,5)))
        if part.stat().st_size!=expected:raise RuntimeError(f"download incomplete {part.stat().st_size}/{expected}")
        os.replace(part,final)
    with zipfile.ZipFile(final) as z:
        bad=z.testzip()
        if bad:raise RuntimeError(f"CRC failure {bad}")
        members=z.namelist()
    result={"kind":kind,"url":url,"path":str(final),"bytes":expected,"sha256_local":sha256(final),
            "zip_crc":"pass","members":len(members),"first_members":members[:8]}
    (ROOT/f"{kind}_manifest.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("kind",choices=URLS);a=p.parse_args();run(a.kind)
