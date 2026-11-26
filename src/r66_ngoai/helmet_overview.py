from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
import argparse

root=Path(__file__).resolve().parents[2]/"handoff/logs/external_eval/r66/helmet/smoke_review"
def make(subfolder):
    files=sorted((root/subfolder).glob("*.jpg"))
    assert len(files)==12
    out=Image.new("RGB",(3*430,4*266),"white")
    for i,p in enumerate(files):
        with Image.open(p) as src:
            tile=ImageOps.contain(src,(426,241))
        x=i%3*430; y=i//3*266
        out.paste(tile,(x,y))
        ImageDraw.Draw(out).text((x+4,y+241),p.stem,fill="black")
    path=root/f"overview_{subfolder}.jpg"
    out.save(path,quality=93)
    print(path)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("folder")
    make(ap.parse_args().folder)

