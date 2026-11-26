"""Check a newly found author dataset; download only a bounded public annotation kit."""
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

HERE = Path(__file__).resolve().parent
BASE = 'https://gram.web.uah.es/data/datasets/rtm/'


def main():
    result = dict(checked_utc=datetime.now(timezone.utc).isoformat(),
                  camera='Urban1: traffic surveillance camera at busy intersection, according to author',
                  rights='Public scientific research; test only, no tuning/training, according to author page',
                  tracking_status='not run: no matched detector cache/images in this R106 sample',
                  prior_head=dict(status=403, exit=1, note='Earlier plain urllib HEAD returned HTTPError 403'))
    for name, resource, cap in [('page', 'index.html', 1024*1024), ('annotation_kit', 'GRAM-RTMv4.zip', 24*1024*1024)]:
        url = BASE + resource
        entry = dict(url=url)
        try:
            try:
                response = urlopen(Request(url, headers={'User-Agent': 'R106-research-source-check/1.0'}), timeout=15)
            except HTTPError as exc:
                response = exc
            with response:
                entry.update(status=response.code, final_url=response.url)
                content = response.read(cap+1)
                if len(content) > cap:
                    entry['blocker'] = 'Response exceeds bounded sample cap; no dataset saved'
                elif response.code == 200:
                    filename = 'gram_' + ('page.txt' if name == 'page' else 'annotation_kit.zip')
                    (HERE / filename).write_bytes(content)
                    entry.update(file=filename, bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
                    if name == 'annotation_kit':
                        with zipfile.ZipFile(io.BytesIO(content)) as archive:
                            entry['manifest'] = [dict(name=i.filename, bytes=i.file_size, compressed=i.compress_size,
                                                      crc=i.CRC) for i in archive.infolist()]
                        entry['note'] = 'Central-directory manifest only; no kit code executed, no tracking result'
                else:
                    entry.update(blocker='Public GET rejected; no login attempted',
                                 response_preview=content[:180].decode('utf-8', errors='replace'))
        except Exception as exc:
            entry['blocker'] = type(exc).__name__ + ': ' + str(exc)
        result[name] = entry
        print(name, entry.get('status'), entry.get('bytes', 0), entry.get('blocker', ''), flush=True)
    (HERE / 'gram_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
