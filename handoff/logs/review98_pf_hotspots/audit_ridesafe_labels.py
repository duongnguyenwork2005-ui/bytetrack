"""Stream RideSafe videoset1 CVAT XML; do not extract 1.2 GB XML tree."""
from collections import Counter
import json
from pathlib import Path
import zipfile

from lxml import etree

ARCHIVE = Path("F:/LABOTARY_data/goc_tai_ve/RideSafe-400-R98/videoset1_xml_annots_with_rider_motor_poly.zip")
OUT = Path(__file__).resolve().parent / "ridesafe_videoset1_audit.json"
rows = []
with zipfile.ZipFile(ARCHIVE) as z:
    names = [n for n in z.namelist() if n.lower().endswith(".xml")]
    assert len(names) == 100
    for index, name in enumerate(names):
        counts = Counter()
        tracks = Counter()
        filters = set()
        motorcycle_frames = set()
        motorcycle_track_ids = set()
        motorcycle_attribute_ids = set()
        motorcycle_attribute_frame_pairs = set()
        errors = Counter()
        track_label = None
        track_id = None
        track_visible_boxes = 0
        track_any_boxes = 0
        track_visible_frames = []
        with z.open(name) as file:
            for event, elem in etree.iterparse(file, events=("start", "end"), huge_tree=False):
                tag = elem.tag
                if event == "start" and tag == "track":
                    track_label = elem.get("label")
                    track_id = elem.get("id")
                    track_visible_boxes = 0
                    track_any_boxes = 0
                    track_visible_frames = []
                    tracks[track_label] += 1
                    if track_label == "motorcycle":
                        motorcycle_track_ids.add(track_id)
                elif event == "end" and tag == "frame_filter":
                    if elem.text:
                        filters.add(elem.text.strip())
                elif event == "end" and tag == "box":
                    track_any_boxes += 1
                    counts[(track_label, "box")] += 1
                    outside = elem.get("outside", "0")
                    counts[(track_label, "outside_" + outside)] += 1
                    counts[(track_label, "keyframe_" + elem.get("keyframe", "missing"))] += 1
                    counts[(track_label, "occluded_" + elem.get("occluded", "missing"))] += 1
                    if track_label == "motorcycle" and outside == "0":
                        track_visible_boxes += 1
                        frame = int(elem.get("frame"))
                        motorcycle_frames.add(frame)
                        if frame in track_visible_frames:
                            errors["duplicate_frame_cvat_track_id"] += 1
                        track_visible_frames.append(frame)
                        attr = [a.text for a in elem.findall("attribute")
                                if a.get("name") == "motor_track_id"]
                        if len(attr) != 1 or not attr[0] or attr[0] == "-1":
                            errors["missing_or_default_motor_track_id"] += 1
                        else:
                            motorcycle_attribute_ids.add(attr[0])
                            key = (frame, attr[0])
                            if key in motorcycle_attribute_frame_pairs:
                                errors["duplicate_frame_motor_track_id"] += 1
                            motorcycle_attribute_frame_pairs.add(key)
                        if frame % 5:
                            errors["motorcycle_frame_not_multiple_5"] += 1
                        coords = [float(elem.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
                        if not (coords[2] > coords[0] and coords[3] > coords[1]):
                            errors["nonpositive_motorcycle_box"] += 1
                    elem.clear()
                    parent = elem.getparent()
                    while elem.getprevious() is not None:
                        del parent[0]
                elif event == "end" and tag in ("polyline", "polygon", "points", "track"):
                    if tag == "track" and track_label == "motorcycle":
                        if track_any_boxes:
                            counts[("motorcycle", "box_tracks")] += 1
                        if track_visible_boxes:
                            counts[("motorcycle", "visible_box_tracks")] += 1
                        if track_visible_boxes >= 2:
                            counts[("motorcycle", "multiframe_visible_tracks")] += 1
                            frames_sorted = sorted(track_visible_frames)
                            for a, b in zip(frames_sorted, frames_sorted[1:]):
                                counts[("motorcycle", "track_gap_5" if b - a == 5 else "track_gap_other")] += 1
                    elem.clear()
                    parent = elem.getparent()
                    while parent is not None and elem.getprevious() is not None:
                        del parent[0]
                    if tag == "track":
                        track_label = None
                        track_id = None
        row = {"file": name, "tracks": dict(tracks),
               "box_counts": {"|".join(k): v for k, v in counts.items()},
               "frame_filters": sorted(filters), "motorcycle_frames": len(motorcycle_frames),
               "first_motorcycle_frame": min(motorcycle_frames) if motorcycle_frames else None,
               "last_motorcycle_frame": max(motorcycle_frames) if motorcycle_frames else None,
               "motorcycle_tracks": len(motorcycle_track_ids), "errors": dict(errors)}
        row["motorcycle_attribute_ids"] = len(motorcycle_attribute_ids)
        rows.append(row)
        if (index + 1) % 20 == 0:
            print("audited", index + 1, flush=True)
totals = Counter()
filter_count = Counter()
for r in rows:
    totals["motorcycle_tracks"] += r["motorcycle_tracks"]
    totals["motorcycle_frames"] += r["motorcycle_frames"]
    for key, value in r["box_counts"].items():
        totals[key] += value
    for value in r["frame_filters"]:
        filter_count[value] += 1
summary = {"archive": str(ARCHIVE), "xml_files": len(rows),
           "frame_filters": dict(filter_count), "totals": dict(totals), "videos": rows}
OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps({k: summary[k] for k in ("xml_files", "frame_filters", "totals")}, indent=2), flush=True)
