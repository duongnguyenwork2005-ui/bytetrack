"""Fetch the author-published FastTrack archives from one immutable revision."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from time import sleep

import requests

HERE = Path(__file__).resolve().parent
DATA = Path("D:/LABOTARY/test dataset/FastTrack-R116")
OLD_DATA = Path("F:/LABOTARY_data/goc_tai_ve/FastTrack-R115")
PRIOR = {"task_day_occlusion", "task_day_right_turn_occlusion", "task_tunnel"}
HF = "https://huggingface.co"
REPO = "Hamidreza-Hashemp/FastTracker-Benchmark"
REVISION = "7e16fad855e64fae3fe7754f1bec8c8fbb50fd87"
CHUNK_BYTES = 64 * 2**20


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(item):
    target = DATA / Path(item["path"]).name
    expected_size, expected_sha = item["size"], item["lfs"]["oid"]
    old = OLD_DATA / target.name
    if old.is_file() and old.stem in PRIOR:
        print("REUSE prior", old.name, flush=True)
        return
    if target.is_file():
        assert target.stat().st_size == expected_size and digest(target) == expected_sha, target
        print("VERIFIED existing", target.name, expected_size, flush=True)
        return
    partial = target.with_suffix(".zip.partial")
    url = f"{HF}/datasets/{REPO}/resolve/{REVISION}/{item['path']}?download=true"
    with requests.Session() as session, partial.open("ab") as stream:
        start = stream.tell()
        print("FETCH", target.name, start, expected_size, flush=True)
        while start < expected_size:
            end = min(start + CHUNK_BYTES, expected_size) - 1
            header = {"Range": f"bytes={start}-{end}"}
            for attempt in range(5):
                try:
                    with session.get(f"{url}&span={start}-{end}", headers=header, stream=True,
                                     timeout=(15, 28)) as response:
                        response.raise_for_status()
                        expected = f"bytes {start}-{end}/{expected_size}"
                        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                            raise OSError(f"incorrect byte range status {response.status_code}")
                        block = response.content
                    if len(block) != end - start + 1:
                        raise OSError(f"incomplete range at {start}")
                    break
                except (requests.RequestException, OSError) as exc:
                    print("RETRY", target.name, start, type(exc).__name__, attempt + 1, flush=True)
                    if attempt == 4:
                        raise
                    sleep(3 * (attempt + 1))
            stream.write(block)
            stream.flush()
            start = end + 1
            if start % (CHUNK_BYTES * 8) == 0 or start == expected_size:
                print("BYTES", target.name, start, expected_size, flush=True)
    assert partial.stat().st_size == expected_size and digest(partial) == expected_sha, target
    partial.replace(target)
    print("VERIFIED downloaded", target.name, expected_size, flush=True)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    response = requests.get(f"{HF}/api/datasets/{REPO}/tree/{REVISION}/train", timeout=20)
    response.raise_for_status()
    sources = [item for item in response.json() if item["path"].endswith(".zip")]
    assert len(sources) == 12
    manifest = {"repo": REPO, "revision": REVISION, "archives": sources}
    (HERE / "sources.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with ThreadPoolExecutor(4) as workers:
        list(workers.map(fetch, sorted(sources, key=lambda row: row["size"])))


if __name__ == "__main__":
    main()
