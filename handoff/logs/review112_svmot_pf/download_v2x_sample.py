"""Download the author-linked V2X-Seq example to the existing dataset drive."""

import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin
from zipfile import ZipFile
import requests
from check_v2x_download import Form

OUT = Path(__file__).resolve().parent
DEST = Path("F:/LABOTARY_data/goc_tai_ve/V2X-Seq-R112")
URL = "https://drive.google.com/uc?export=download&id=1gjOmGEBMcipvDzu2zOrO9ex_OscUZMYY"
EXPECTED_BYTES = 585329481


def main():
    DEST.mkdir(exist_ok=True)
    target = DEST / "V2X-Seq-SPD-Example.zip"
    if not target.exists():
        partial = target.with_suffix(".zip.part")
        assert not partial.exists(), "Keep partial evidence and resume explicitly"
        with requests.Session() as session:
            page = session.get(URL, timeout=25)
            form = Form()
            form.feed(page.text)
            address = urljoin(page.url, form.action)
            assert address.startswith("https://drive.usercontent.google.com/")
            with session.get(address, params=form.fields, stream=True, timeout=(20, 60)) as response:
                response.raise_for_status()
                assert int(response.headers["Content-Length"]) == EXPECTED_BYTES
                downloaded = 0
                with partial.open("xb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                        downloaded += len(chunk)
                        if downloaded // (100 * 1024 * 1024) != (downloaded - len(chunk)) // (100 * 1024 * 1024):
                            print("DOWNLOADED", downloaded, flush=True)
                assert downloaded == EXPECTED_BYTES
        partial.replace(target)
    assert target.stat().st_size == EXPECTED_BYTES
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with ZipFile(target) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        labels = [name for name in names if "/label/camera/" in name and name.endswith(".json")]
        infos = [name for name in names if name.endswith("data_info.json")]
        classes = {}
        label_boxes = 0
        valid_2d = 0
        for name in labels:
            data = json.loads(archive.read(name))
            for obj in data:
                classes[obj["type"]] = classes.get(obj["type"], 0) + 1
                label_boxes += 1
                box = obj.get("2d_box")
                valid_2d += bool(box and box["xmax"] > box["xmin"] and box["ymax"] > box["ymin"])
        result = {
            "source": "https://github.com/AIR-THU/DAIR-V2X-Seq",
            "file_id": "1gjOmGEBMcipvDzu2zOrO9ex_OscUZMYY", "path": str(target),
            "bytes": target.stat().st_size, "sha256": digest, "all_member_crc": True,
            "zip_members": len(names), "camera_label_frames": len(labels),
            "camera_label_boxes": label_boxes, "nonempty_2d_boxes": valid_2d,
            "classes": classes, "data_info_files": infos,
        }
        (OUT / "v2x_sample_inventory.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (OUT / "v2x_sample_members.json").write_text(json.dumps(names, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
