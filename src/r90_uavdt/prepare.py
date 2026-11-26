"""Convert selected publisher vehicle GT to the project's MOT one-class convention."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import config
from r90_uavdt.audit import EVIDENCE, GT_PREFIX, SOURCE

SPLIT = "UAVDT-R90train8"


def main() -> None:
    selected = json.loads((EVIDENCE / "selection.json").read_text(encoding="utf-8"))["all"]
    audit = json.loads((EVIDENCE / "label_audit.json").read_text(encoding="utf-8"))
    dest = config.PROCESSED_DIR / SPLIT
    seqmap = config.PROCESSED_DIR / "seqmaps" / f"{SPLIT}.txt"
    if dest.exists() or seqmap.exists():
        raise FileExistsError("R90 processed namespace already exists")
    manifest = {"split": SPLIT, "source_archive": str(SOURCE / "UAV-benchmark-MOTD_v1.0.zip"),
                "mot_class_1_semantics": "vehicle (car/truck/bus); evaluator numerical alias only",
                "mot_class_8_semantics": "publisher gt_ignore box", "sequences": {}}
    with zipfile.ZipFile(SOURCE / "UAV-benchmark-MOTD_v1.0.zip") as z:
        for seq in selected:
            raw_gt = z.read(f"{GT_PREFIX}{seq}_gt.txt")
            raw_ignore = z.read(f"{GT_PREFIX}{seq}_gt_ignore.txt")
            assert hashlib.sha256(raw_gt).hexdigest() == audit[seq]["sha256"]["gt"]
            assert hashlib.sha256(raw_ignore).hexdigest() == audit[seq]["sha256"]["ignore"]
            gt = np.loadtxt(io.BytesIO(raw_gt), delimiter=",", ndmin=2)
            assert set(np.unique(gt[:, 7])) == {1}, seq
            assert set(np.unique(gt[:, 6])) == {1}, seq
            lines = [",".join(str(int(x)) for x in row[:8]) + ",-1" for row in gt]
            ignored = np.loadtxt(io.BytesIO(raw_ignore), delimiter=",", ndmin=2) if raw_ignore else np.empty((0, 9))
            lines += [",".join(str(int(x)) for x in row[:7]) + ",8,-1" for row in ignored]
            pair = np.array([(int(r[0]), int(r[1])) for r in np.vstack((gt, ignored))])
            assert len(np.unique(pair, axis=0)) == len(pair), seq
            n = audit[seq]["frames"]
            assert int(gt[:, 0].max()) <= n
            lines.sort(key=lambda line: tuple(int(x) for x in line.split(",")[:2]))
            gt_bytes = ("\n".join(lines) + "\n").encode()
            info = (f"[Sequence]\nname={seq}\nimDir=img1\nframeRate=30\nseqLength={n}\n"
                    "imWidth=1024\nimHeight=540\nimExt=.jpg\n").encode()
            folder = dest / seq
            (folder / "gt").mkdir(parents=True)
            (folder / "gt/gt.txt").write_bytes(gt_bytes)
            (folder / "seqinfo.ini").write_bytes(info)
            manifest["sequences"][seq] = {"frames": n, "vehicle_rows": len(gt),
                "ignore_rows": len(ignored), "source_gt_sha256": audit[seq]["sha256"]["gt"],
                "source_ignore_sha256": audit[seq]["sha256"]["ignore"],
                "processed_gt_sha256": hashlib.sha256(gt_bytes).hexdigest(),
                "seqinfo_sha256": hashlib.sha256(info).hexdigest()}
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(selected) + "\n", encoding="utf-8")
    (EVIDENCE / "gt_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("PREPARED", len(selected), "vehicle sequences", SPLIT)


if __name__ == "__main__":
    main()
