"""Download one front-facing clip's labels before assessing MOT suitability."""

from collections import Counter
import hashlib
import io
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile
from meteor_remote_zip import SplitRemoteZip

OUT = Path(__file__).resolve().parent


def main():
    remote = SplitRemoteZip()
    with ZipFile(remote) as archive:
        selected = next(item for item in archive.infolist()
                        if item.filename.startswith("METEOR_Dataset/Frame XML Annotations/REC_2020_")
                        and item.filename.endswith("_F.zip"))
        assert selected.file_size < 8 * 1024 * 1024
        content = archive.read(selected)
    path = OUT / ("meteor_labels_" + Path(selected.filename).name)
    path.write_bytes(content)
    with ZipFile(io.BytesIO(content)) as labels:
        assert labels.testzip() is None
        xmls = sorted(name for name in labels.namelist() if name.endswith(".xml"))
        classes, tags, attributes = Counter(), Counter(), Counter()
        sample = []
        for name in xmls:
            document = ET.fromstring(labels.read(name))
            for obj in document.findall("object"):
                classes[obj.findtext("name")] += 1
                tags.update(child.tag for child in obj)
                for child in obj.findall("attributes/attribute"):
                    attributes[child.findtext("name")] += 1
                if len(sample) < 3:
                    sample.append({child.tag: (child.text or "").strip() for child in obj
                                   if not list(child)})
        result = {"source_revision": remote.revision, "archive_member": selected.filename,
                  "downloaded_label_archive": str(path), "bytes": len(content),
                  "sha256": hashlib.sha256(content).hexdigest(), "selected_member_crc": True,
                  "all_inner_member_crc": True, "xml_frames": len(xmls),
                  "classes": dict(classes), "object_fields": dict(tags),
                  "attributes": dict(attributes), "example_scalar_fields": sample,
                  "bytes_fetched": sum(r["bytes"] for r in remote.transfers)}
    (OUT / "meteor_label_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
