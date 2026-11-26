"""Select mixed-vehicle dashcam scenes from ground truth alone."""
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "review112_svmot_pf"
sys.path.insert(0, str(PRIOR))
from meteor_remote_zip import SplitRemoteZip
from meteor_pilot import labels_for_camera, np, write

LABELS = "METEOR_Dataset/Frame XML Annotations/"
VIDEOS = "METEOR_Dataset/Raw Videos/"
CLASSES = {"car": {"Car"}, "motorcycle": {"MotorBike", "Scooter"}}
VIDEO_ROOT = Path("F:/LABOTARY_data/goc_tai_ve/METEOR-R113")
AUDIT = HERE / "meteor_labels"


def inspect(archive, member, label_content):
    name = Path(member).stem
    rows = {key: [] for key in CLASSES}
    raw_classes = Counter()
    with ZipFile(io.BytesIO(label_content)) as labels:
        xml = sorted(item for item in labels.namelist() if item.lower().endswith(".xml"))
        if len(xml) != 1800:
            # The sampled releases contain compact annotation indices even for
            # 150-frame label sets. Their mapping to 30 fps video is unknown.
            return {"name": name, "member": member, "xml_frames": len(xml),
                    "excluded": "annotation cadence not verified against 30 fps video",
                    "label_sha256": hashlib.sha256(label_content).hexdigest()}
        for frame, path in enumerate(xml, 1):
            assert Path(path).stem == f"frame_{frame - 1:06}"
            for obj in ET.fromstring(labels.read(path)).findall("object"):
                category = obj.findtext("name")
                raw_classes[category] += 1
                for key, group in CLASSES.items():
                    if category not in group:
                        continue
                    attrs = {a.findtext("name"): a.findtext("value") for a in obj.findall("attributes/attribute")}
                    x, y, right, bottom = [float(obj.findtext("bndbox/" + k)) for k in ["xmin", "ymin", "xmax", "ymax"]]
                    occ = int(obj.findtext("occluded"))
                    assert 0 <= x < right <= 1920 and 0 <= y < bottom <= 1080 and occ in (0, 1)
                    rows[key].append([frame, int(attrs["track_id"]), x, y, right - x, bottom - y, 1, 1, 1 - occ])
    stats = {}
    for key, values in rows.items():
        if not values:
            stats[key] = {"boxes": 0, "ids": 0, "occ": 0, "turn_frames": 0, "turns": 0, "turn_occ": 0}
            continue
        gt = np.array(values)
        assert len(np.unique(gt[:, :2], axis=0)) == len(gt)
        path = AUDIT / f"{name}_{key}.txt"
        np.savetxt(path, gt, delimiter=",", fmt="%.10g")
        ids, present, groups, episodes = labels_for_camera(path, name, 1800, 30)
        occluded = np.zeros_like(present)
        occluded[gt[:, 0].astype(int) - 1, np.searchsorted(ids, gt[:, 1])] = gt[:, 8] == 0
        turn = groups["turn25"] >= 0
        stats[key] = {"boxes": len(gt), "ids": len(ids), "occ": int(occluded.sum()),
                      "turn_frames": int(turn.sum()), "turns": sum(e["level"] == "turn25" for e in episodes),
                      "turn_occ": int((turn & occluded).sum()), "gt": str(path),
                      "gt_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    info = archive.getinfo(f"{VIDEOS}{name}.MP4")
    return {"name": name, "member": member, "label_sha256": hashlib.sha256(label_content).hexdigest(),
            "image_box_classes": dict(raw_classes), "classes": stats, "video_member": info.filename,
            "video_size": info.file_size}


def audit():
    AUDIT.mkdir(exist_ok=True)
    remote = SplitRemoteZip()
    assert remote.revision == "ecedf70e3d2d44c7e6e794246bb4bd1e4020662c"
    with ZipFile(remote) as archive:
        names = sorted(name for name in archive.namelist() if name.startswith(LABELS + "REC_2020_")
                       and name.endswith("_F.zip") and f"{VIDEOS}{Path(name).stem}.MP4" in archive.NameToInfo)
        prior = json.loads((PRIOR / "meteor_label_audit.json").read_text())["archive_member"]
        names.remove(prior)
        sample = [prior] + [names[len(names) * (2 * i + 1) // 16] for i in range(8)]
        checks = []
        for name in sample:
            path = AUDIT / Path(name).name
            if not path.exists():
                content = archive.read(name)
                path.write_bytes(content)
            content = path.read_bytes()
            import zlib
            assert zlib.crc32(content) == archive.getinfo(name).CRC
            row = inspect(archive, name, content)
            checks.append(row)
            print(row["name"], row.get("classes", row.get("excluded")), flush=True)
    selected = []
    for category in CLASSES:
        for metric in ["turns", "occ"]:
            eligible = [row for row in checks if "classes" in row and all(c["ids"] > 0 for c in row["classes"].values())
                        and row["classes"][category][metric] > 0]
            if eligible:
                best = min(eligible, key=lambda r: (-r["classes"][category][metric], r["name"]))
                if best["name"] not in selected:
                    selected.append(best["name"])
    write(HERE / "meteor_selection.json", {"revision": remote.revision, "scope": "front 30 fps; image plane only",
                                           "candidate_pool_without_r112": len(names), "sample": checks,
                                           "selected": selected, "bytes_fetched": sum(r["bytes"] for r in remote.transfers)})


def download():
    manifest = json.loads((HERE / "meteor_selection.json").read_text(encoding="utf-8"))
    VIDEO_ROOT.mkdir(exist_ok=True)
    remote = SplitRemoteZip()
    assert remote.revision == manifest["revision"]
    saved = {}
    with ZipFile(remote) as archive:
        for row in manifest["sample"]:
            if row["name"] not in manifest["selected"]:
                continue
            info = archive.getinfo(row["video_member"])
            target = VIDEO_ROOT / Path(info.filename).name
            if not target.exists():
                partial = target.with_suffix(".partial")
                with archive.open(info) as source, partial.open("wb") as out:
                    while chunk := source.read(8 * 1024 * 1024):
                        out.write(chunk)
                partial.replace(target)
            import zlib
            crc = 0
            with target.open("rb") as stream:
                while chunk := stream.read(8 * 1024 * 1024):
                    crc = zlib.crc32(chunk, crc)
            assert target.stat().st_size == info.file_size and crc == info.CRC
            with target.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            saved[row["name"]] = {"video": str(target), "sha256": digest, "bytes": info.file_size, "crc32": crc}
            write(HERE / "meteor_downloads.json", saved)
            print("READY", row["name"], info.file_size, flush=True)


if __name__ == "__main__":
    {"audit": audit, "download": download}[sys.argv[1]]()
