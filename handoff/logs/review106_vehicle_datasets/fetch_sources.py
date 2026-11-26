"""Capture small public author pages, with a hard response limit; no authentication."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

OUT = Path(__file__).resolve().parent
SOURCES = [
    ('helmet_author', 'https://raw.githubusercontent.com/LinHanhe/Helmet_use_detection/master/README.md'),
    ('helmet_osf', 'https://osf.io/4pwj8/'),
    ('svmot_record', 'https://zenodo.org/records/19468203'),
    ('svmot_readme', 'https://zenodo.org/records/19468203/files/README_RELEASE_v1.0.md?download=1'),
    ('roundabout_author', 'https://raw.githubusercontent.com/siri-rouser/RoundaboutHD/main/README.md'),
    ('roundabout_release', 'https://huggingface.co/datasets/yl4300/RoundaboutHD/raw/main/README.md'),
    ('ridesafe_author', 'https://raw.githubusercontent.com/dash-cop/DashCop/main/README.md'),
    ('ridesafe_release', 'https://huggingface.co/datasets/DeepBug/RideSafe-400/raw/main/README.md'),
    ('sherbrooke_author', 'https://www.jpjodoin.com/urbantracker/dataset.html'),
    ('urbantracker_author_alt', 'https://jpjodoin.com/urbantracker/dataset.html'),
    ('kitti_tracking', 'https://www.cvlibs.net/datasets/kitti/eval_tracking.php'),
    ('kitti_setup', 'https://www.cvlibs.net/datasets/kitti/setup.php'),
]


def main():
    result = []
    for name, url in SOURCES:
        record = dict(name=name, url=url, checked_utc=datetime.now(timezone.utc).isoformat())
        try:
            request = Request(url, headers={'User-Agent': 'R106-research-source-check/1.0'})
            try:
                response = urlopen(request, timeout=12)
            except HTTPError as exc:
                response = exc
            with response:
                record.update(status=response.code, final_url=response.url,
                              content_type=response.headers.get('Content-Type'))
                body = response.read(1024 * 1024 + 1)
                if len(body) > 1024 * 1024:
                    record['error'] = 'response exceeds 1 MiB cap; not saved'
                else:
                    filename = 'source_' + name + '.txt'
                    (OUT / filename).write_bytes(body)
                    record.update(file=filename, bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        except Exception as exc:
            record['error'] = type(exc).__name__ + ': ' + str(exc)
        result.append(record)
        print(name, record.get('status', record.get('error')), record.get('bytes', 0), flush=True)
    (OUT / 'sources_manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
