"""Fetch the preselected public archive member, with CRC checking on streaming read."""

import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile
from meteor_remote_zip import SplitRemoteZip

OUT = Path(__file__).resolve().parent
DEST = Path("F:/LABOTARY_data/goc_tai_ve/METEOR-R112")


def main():
    audit = json.loads((OUT / "meteor_label_audit.json").read_text())
    stem = Path(audit["archive_member"]).stem
    remote = SplitRemoteZip()
    assert remote.revision == audit["source_revision"]
    DEST.mkdir(exist_ok=True)
    target = DEST / f"{stem}.MP4"
    with ZipFile(remote) as archive:
        info = archive.getinfo(f"METEOR_Dataset/Raw Videos/{stem}.MP4")
        assert not target.exists()
        partial = target.with_suffix(".MP4.part")
        with archive.open(info) as stream, partial.open("xb") as output:
            while chunk := stream.read(8 * 1024 * 1024):
                output.write(chunk)
        assert partial.stat().st_size == info.file_size
        partial.replace(target)
    shutil.copy2(audit["downloaded_label_archive"], DEST / Path(audit["downloaded_label_archive"]).name)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    result = {"source_revision": remote.revision, "video": str(target),
              "bytes": target.stat().st_size, "sha256": digest, "selected_member_crc": True,
              "fetched_compressed_bytes": sum(r["bytes"] for r in remote.transfers)}
    (OUT / "meteor_video_source.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
