"""Keep primary-source snapshots and bounded public download checks."""

from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
from pathlib import Path
import requests

OUT = Path(__file__).resolve().parent
SOURCES = {
    "v2x_seq_readme": "https://raw.githubusercontent.com/AIR-THU/DAIR-V2X-Seq/main/README.md",
    "v2x_seq_labels": "https://raw.githubusercontent.com/AIR-THU/DAIR-V2X-Seq/main/dataset/v2x-seq-spd/README.md",
    "trafficmot": "https://raw.githubusercontent.com/lihaoliu-cambridge/trafficmot/main/README.md",
    "tumtraf": "https://innovation-mobility.com/en/project-providentia/a9-dataset/",
    "gram": "https://gram.web.uah.es/data/datasets/rtm/index.html",
    "kitti": "https://www.cvlibs.net/datasets/kitti/eval_tracking.php",
    "cityflow": "https://www.aicitychallenge.org/2022-data-and-evaluation/",
}


def fetch(item):
    name, url = item
    try:
        result = requests.get(url, timeout=20)
        assert len(result.content) < 2_000_000
        path = OUT / f"source_{name}.txt"
        path.write_bytes(result.content)
        return {"name": name, "url": url, "status": result.status_code,
                "bytes": len(result.content), "file": path.name,
                "sha256": hashlib.sha256(result.content).hexdigest()}
    except Exception as error:
        return {"name": name, "url": url, "error": str(error)}


def main():
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = list(pool.map(fetch, SOURCES.items()))
    url = "https://drive.google.com/uc?export=download&id=1gjOmGEBMcipvDzu2zOrO9ex_OscUZMYY"
    try:
        with requests.get(url, stream=True, timeout=20) as response:
            check = {"url": url, "status": response.status_code,
                     "content_type": response.headers.get("Content-Type"),
                     "content_length": response.headers.get("Content-Length"),
                     "content_disposition": response.headers.get("Content-Disposition"),
                     "downloaded_archive": False}
    except requests.RequestException as error:
        check = {"url": url, "error": str(error), "downloaded_archive": False}
    result = {"checked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "sources": sources, "v2x_example_download_check": check}
    (OUT / "dataset_sources.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
