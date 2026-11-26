"""Inspect the public example archive handshake without downloading the archive."""

from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import urljoin
import requests


class Form(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.fields = {}

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if tag == "form":
            self.action = values.get("action")
        if tag == "input" and values.get("name"):
            self.fields[values["name"]] = values.get("value", "")


def main():
    path = Path(__file__).resolve().parent / "v2x_example_access.json"
    url = "https://drive.google.com/uc?export=download&id=1gjOmGEBMcipvDzu2zOrO9ex_OscUZMYY"
    result = {"official_parent": "https://github.com/AIR-THU/DAIR-V2X-Seq",
              "example_file_id": "1gjOmGEBMcipvDzu2zOrO9ex_OscUZMYY",
              "archive_downloaded": False}
    with requests.Session() as session:
        response = session.get(url, timeout=20)
        form = Form()
        form.feed(response.text)
        if form.action:
            target = urljoin(response.url, form.action)
            assert target.startswith("https://drive.usercontent.google.com/")
            with session.get(target, params=form.fields, stream=True, timeout=25) as archive:
                first = next(archive.iter_content(4096), b"")
                result.update(status=archive.status_code,
                              content_type=archive.headers.get("Content-Type"),
                              content_length=archive.headers.get("Content-Length"),
                              content_disposition=archive.headers.get("Content-Disposition"),
                              zip_header=first[:4].hex(), bytes_read=len(first),
                              binary_zip=first.startswith(b"PK"))
                if not first.startswith(b"PK"):
                    (path.parent / "v2x_example_download_response.txt").write_bytes(first)
        else:
            result.update(status=response.status_code, form_found=False)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
