"""Download the public DAIR sample using the official Google Drive confirmation form."""
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import urlsplit
from zipfile import ZipFile, is_zipfile

import requests

DEFAULT_ID = "1gjOmGEBMcipvDzu2zOrO9ex_OscUZMYY"
LIMIT = 5 * 1024 ** 3


class DownloadForm(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.action = None
        self.fields = {}

    def handle_starttag(self, tag, attributes):
        values = dict(attributes)
        if tag == "form" and values.get("id") == "download-form":
            self.active = True
            self.action = values.get("action")
        if tag == "input" and self.active and values.get("type") == "hidden":
            self.fields[values["name"]] = values.get("value", "")

    def handle_endtag(self, tag):
        if tag == "form":
            self.active = False


def main():
    path = Path(sys.argv[1])
    file_id = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ID
    partial = path.with_suffix(".zip.partial")
    with requests.Session() as session:
        response = session.get("https://drive.usercontent.google.com/uc", params={"id": file_id, "export": "download"},
                               timeout=(15, 30), stream=True)
        response.raise_for_status()
        form = DownloadForm()
        if "text/html" in response.headers.get("Content-Type", ""):
            form.feed(response.text)
            response.close()
            if not form.action or urlsplit(form.action).hostname != "drive.usercontent.google.com":
                raise ValueError("Google Drive did not return the expected public confirmation form")
            response = session.get(form.action, params=form.fields, timeout=(20, 90), stream=True)
        with response as download:
            download.raise_for_status()
            size = int(download.headers.get("Content-Length", 0))
            if "text/html" in download.headers.get("Content-Type", ""):
                if "Too many users have viewed or downloaded" in download.text:
                    raise ValueError("Google Drive download quota for this public file has been reached")
                raise ValueError("Google Drive returned an HTML page in place of the archive")
            if size > LIMIT:
                raise ValueError(f"Download response type={download.headers.get('Content-Type')}, bytes={size}")
            print(f"DAIR response {download.status_code}, bytes advertised {size}", flush=True)
            received = 0
            with partial.open("wb") as output:
                for chunk in download.iter_content(1024 * 1024):
                    received += len(chunk)
                    if received > LIMIT:
                        raise ValueError("DAIR sample exceeded the transfer limit")
                    output.write(chunk)
            if size and received != size:
                raise ValueError(f"Incomplete DAIR sample: received {received} of {size}")
    if not is_zipfile(partial):
        raise ValueError("DAIR download did not contain a ZIP archive")
    with ZipFile(partial) as archive:
        print(f"DAIR archive: {received} bytes, {len(archive.infolist())} entries", flush=True)
    partial.replace(path)


if __name__ == "__main__":
    main()
