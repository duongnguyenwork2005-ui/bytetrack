"""Check the official alternate example and compact train/val metadata archive."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import requests

HERE = Path(__file__).resolve().parent
DEST = Path("F:/LABOTARY_data/goc_tai_ve/V2X-Seq-R113")
FILES = {
    "example": "1lyl1gdrbfDI-sTQq7EoU-SkdBcWJqFn7",
    "trainval_metadata": "1SuJOyI3G8MstCY-EHS5tbWvaMh2iZuef",
    "trainval_fixed_images": "12B4FNZM9G3Q-kRr-h9vdt-p2c5bhGBcB",
}


def fetch(item):
    name, file_id = item
    params = {"id": file_id, "export": "download", "confirm": "t"}
    url = "https://drive.usercontent.google.com/download"
    result = {"name": name, "source_id": file_id}
    try:
        with requests.get(url, params=params, stream=True, timeout=(10, 25)) as response:
            chunks = response.iter_content(1024 * 1024)
            first = next(chunks, b"")
            length = int(response.headers.get("Content-Length") or 0)
            result.update(status=response.status_code, content_type=response.headers.get("Content-Type"),
                          content_length=length, zip=first.startswith(b"PK"), quota=b"quota" in first[:4096].lower())
            print(result, flush=True)
            if response.status_code != 200 or not result["zip"] or not 0 < length <= 700_000_000:
                return result
            target = DEST / f"{name}.zip"
            partial = target.with_suffix(".partial")
            if not target.exists():
                with partial.open("wb") as stream:
                    stream.write(first)
                    for block in chunks:
                        stream.write(block)
                        assert stream.tell() <= length
                assert partial.stat().st_size == length
                with ZipFile(partial) as archive:
                    assert archive.testzip() is None
                partial.replace(target)
            assert target.stat().st_size == length
            with target.open("rb") as stream:
                result["sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            with ZipFile(target) as archive:
                result["members"] = len(archive.infolist())
                result["first_members"] = archive.namelist()[:20]
                result["path"] = str(target)
            print(name, "downloaded", length, flush=True)
    except (requests.RequestException, OSError, ValueError) as error:
        result["error"] = str(error)
    return result


if __name__ == "__main__":
    DEST.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as executor:
        report = list(executor.map(fetch, FILES.items()))
    (HERE / "v2x_access.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
