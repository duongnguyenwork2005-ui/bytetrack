"""Extract only official-train SVMOT frames 600..629 for detector suitability check."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = Path(r"D:\LABOTARY\svmot_data_v1.0.zip")
SOURCE = "seq_07_morning"
NAME = "seq_07_morning_600_629"
FRAMES = range(600, 630)


def main() -> None:
    images = ROOT / "data" / "extracted" / "svmot_dev_smoke" / NAME
    images.mkdir(parents=True, exist_ok=True)
    processed = ROOT / "data" / "processed" / "SVMOT-dev-smoke" / NAME
    (processed / "gt").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as archive:
        for number in FRAMES:
            filename = f"{number:04d}.jpg"
            target = images / filename
            if target.exists():
                raise FileExistsError(target)
            target.write_bytes(archive.read(f"data/{SOURCE}/img/{filename}"))
        gt = pd.read_csv(io.BytesIO(archive.read(f"data/{SOURCE}/gt/gt.txt")), header=None)
    subset = gt[gt[0].between(min(FRAMES), max(FRAMES))].copy()
    subset[0] -= min(FRAMES) - 1
    subset[7] = 1
    subset.to_csv(processed / "gt" / "gt.txt", index=False, header=False)
    (processed / "seqinfo.ini").write_text(
        f"[Sequence]\nname={NAME}\nimDir={images.resolve()}\nframeRate=30\nseqLength=30\n"
        "imWidth=1920\nimHeight=1080\nimExt=.jpg\n", encoding="utf-8")
    seqmap = ROOT / "data" / "processed" / "seqmaps" / "SVMOT-dev-smoke.txt"
    seqmap.write_text("name\n" + NAME + "\n", encoding="utf-8")
    print(f"{NAME}: {len(FRAMES)} images, {len(subset)} GT boxes, "
          f"{int((gt[gt[0].between(min(FRAMES), max(FRAMES))][7] == 3).sum())} motorcycle GT")


if __name__ == "__main__":
    main()
