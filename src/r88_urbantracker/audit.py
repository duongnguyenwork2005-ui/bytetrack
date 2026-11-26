"""Inspect Urban Tracker source annotations and frame alignment without extraction."""
from __future__ import annotations

import json
import sqlite3
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"F:\LABOTARY_data\goc_tai_ve\UrbanTracker-R69")
OUT = Path(__file__).resolve().parents[2] / "handoff/logs/review88_urbantracker"
OUT.mkdir(parents=True, exist_ok=True)


def db_rows(z: zipfile.ZipFile, name: str):
    data = z.read(name)
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "gt.sqlite"
        p.write_bytes(data)
        con = sqlite3.connect(str(p))
        try:
            types = list(con.execute("SELECT road_user_type,type_string FROM objects_type"))
            objects = list(con.execute("SELECT object_id,road_user_type FROM objects"))
            boxes = list(con.execute("SELECT object_id,frame_number,x_top_left,y_top_left,x_bottom_right,y_bottom_right FROM bounding_boxes"))
        finally:
            con.close()
    return types, objects, boxes


def main():
    result = {}
    tiles = []
    for scene, first in (("rene", 7200), ("stmarc", 1000)):
        with zipfile.ZipFile(ROOT / f"{scene}_annotations.zip") as az, zipfile.ZipFile(ROOT / f"{scene}_frames.zip") as fz:
            member = next(n for n in az.namelist() if n.endswith(f"{scene}_gt.sqlite"))
            types, objs, boxes = db_rows(az, member)
            obj_type = {i: t for i, t in objs}
            classes = Counter(obj_type[i] for i, *_ in boxes)
            ids = Counter(t for _, t in objs)
            frames = sorted({f for _, f, *_ in boxes})
            dups = len(boxes) - len({(i, f) for i, f, *_ in boxes})
            wh = [(x2-x1, y2-y1) for _, _, x1, y1, x2, y2 in boxes]
            result[scene] = {"types": types, "ids_per_type": dict(ids), "boxes_per_type": dict(classes),
                             "first_frame":frames[0], "last_frame":frames[-1], "annotated_frames":len(frames),
                             "duplicates":dups, "min_box_wh":[min(w for w,h in wh),min(h for w,h in wh)]}
            for offset in (0, 1):
                n = first + offset
                member = f"{scene}_frames/{n:08d}.jpg"
                image = cv2.imdecode(np.frombuffer(fz.read(member), dtype=np.uint8), cv2.IMREAD_COLOR)
                assert image is not None
                # Same source annotation drawn on neighboring images to audit zero/one based indexing.
                for oid, frame, x1, y1, x2, y2 in boxes:
                    if frame != first: continue
                    cv2.rectangle(image, (round(x1),round(y1)), (round(x2),round(y2)), (0,255,255), 2)
                    cv2.putText(image, str(oid), (round(x1),round(y1)-3), cv2.FONT_HERSHEY_SIMPLEX, .35,(0,0,255),1)
                cv2.putText(image, f"{scene} GT {first} vs JPG {n}", (20,35), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
                tiles.append(cv2.resize(image,(640,360)))
    cv2.imwrite(str(OUT / "alignment_first_frame.jpg"), np.vstack(tiles))
    (OUT / "source_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
