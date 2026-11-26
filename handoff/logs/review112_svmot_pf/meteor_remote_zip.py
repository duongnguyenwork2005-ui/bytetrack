"""Read the author-linked split ZIP using verified HTTP byte ranges."""

import bisect
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
import requests

OUT = Path(__file__).resolve().parent


class SplitRemoteZip(io.RawIOBase):
    def __init__(self):
        self.session = requests.Session()
        result = self.session.get("https://huggingface.co/api/datasets/XijunWang/METEOR", timeout=25)
        result.raise_for_status()
        self.revision = result.json()["sha"]
        listing = json.loads((OUT / "meteor_files.json").read_text(encoding="utf-8"))
        self.parts = [item for item in listing if item["path"].startswith("chunk_")]
        self.parts.sort(key=lambda item: item["path"])
        self.starts = [0]
        for item in self.parts:
            self.starts.append(self.starts[-1] + item["size"])
        self.position = 0
        self.transfers = []

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0 if whence == 0 else self.position if whence == 1 else self.starts[-1]) + offset
        if self.position < 0:
            raise ValueError("Negative archive offset")
        return self.position

    def read(self, length=-1):
        if length < 0:
            length = self.starts[-1] - self.position
        assert length <= 32 * 1024 * 1024, "Use streaming reads for large archive members"
        result = []
        while length and self.position < self.starts[-1]:
            part = bisect.bisect_right(self.starts, self.position) - 1
            offset = self.position - self.starts[part]
            amount = min(length, self.starts[part + 1] - self.position)
            item = self.parts[part]
            url = (f"https://huggingface.co/datasets/XijunWang/METEOR/resolve/"
                   f"{self.revision}/{item['path']}?range_start={offset}&range_size={amount}")
            response = self.session.get(url, headers={"Range": f"bytes={offset}-{offset + amount - 1}"},
                                        stream=True, timeout=(20, 40))
            with response:
                response.raise_for_status()
                expected = f"bytes {offset}-{offset + amount - 1}/{item['size']}"
                assert response.status_code == 206 and response.headers.get("Content-Range") == expected
                data = response.raw.read(amount + 1)
                assert len(data) == amount
            self.transfers.append({"part": item["path"], "offset": offset, "bytes": amount,
                                   "sha256": hashlib.sha256(data).hexdigest()})
            result.append(data)
            self.position += amount
            length -= amount
        return b"".join(result)


def main():
    remote = SplitRemoteZip()
    with ZipFile(remote) as archive:
        members = [{"name": item.filename, "bytes": item.file_size,
                    "compressed_bytes": item.compress_size, "crc32": item.CRC}
                   for item in archive.infolist()]
        candidates = [item for item in members if not item["name"].endswith("/")
                      and (item["name"].endswith(".xml") or "annotation" in item["name"].lower())]
        print("MEMBERS", len(members), flush=True)
        print(json.dumps(members[:12], indent=2), flush=True)
        print("LABEL_CANDIDATES", json.dumps(candidates[:6], indent=2), flush=True)
    result = {"author_repo": "https://github.com/GAMMA-UMD/METEOR", "hf_revision": remote.revision,
              "archive_bytes": remote.starts[-1], "members": members,
              "bytes_fetched": sum(row["bytes"] for row in remote.transfers),
              "range_sha256": remote.transfers}
    (OUT / "meteor_remote_catalog.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
