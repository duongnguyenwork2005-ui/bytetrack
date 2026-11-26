"""Build a CPU-only deployment archive. Does not connect to or modify Nano."""
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
FILES = ["bench_nano.py", "prepare_nano.py", "cadence.py", "driver_manifest.json",
         "drivers/r86_driver.py", "drivers/r101_driver.py", "drivers/r102_driver.py"]


def main():
    archive_path = HERE / "r105_deploy.tar.gz"
    rows = []
    with tarfile.open(archive_path, "w:gz") as archive:
        for name in FILES:
            data = (HERE / name).read_bytes()
            item = tarfile.TarInfo(name)
            item.size = len(data)
            item.mode = 0o644
            archive.addfile(item, io.BytesIO(data))
            rows.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    with tarfile.open(archive_path, "r:gz") as archive:
        assert archive.getnames() == FILES
        for row in rows:
            assert hashlib.sha256(archive.extractfile(row["path"]).read()).hexdigest() == row["sha256"]
    result = dict(files=rows, archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                  bytes=archive_path.stat().st_size, deployed=False)
    (HERE / "bundle_manifest.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
