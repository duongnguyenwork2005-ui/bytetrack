"""Bounded read-only directory metadata from the author's public Nextcloud share."""
from pathlib import Path
import argparse
import hashlib
import json
from datetime import datetime, timezone
from urllib.parse import quote
import xml.etree.ElementTree as ET
import requests

HERE = Path(__file__).resolve().parent
BASE = 'https://nx21496.your-storageshare.de/public.php/dav/files/Zd4rwsMwwXMaXiK/'
BODY = ('<?xml version="1.0"?><d:propfind xmlns:d="DAV:"><d:prop>'
        '<d:displayname/><d:getcontentlength/><d:resourcetype/>'
        '<d:getlastmodified/><d:getetag/></d:prop></d:propfind>')
LIMIT = 2 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('name')
    args = parser.parse_args()
    assert args.path == '' or (not args.path.startswith('/') and '..' not in args.path.split('/'))
    assert args.name.replace('_', '').isalnum()
    output = HERE / (args.name + '.json')
    raw_path = HERE / (args.name + '.xml')
    assert not output.exists() and not raw_path.exists()
    url = BASE + quote(args.path, safe='/')
    with requests.request('PROPFIND', url, data=BODY.encode(),
                          headers={'Depth': '1', 'Content-Type': 'application/xml'},
                          stream=True, timeout=(10, 20), allow_redirects=False) as response:
        content = response.raw.read(LIMIT + 1)
        assert len(content) <= LIMIT, 'Metadata limit exceeded; do not download dataset bodies'
        result = {'time_UTC': datetime.now(timezone.utc).isoformat(), 'url': url,
                  'method': 'PROPFIND', 'Depth': 1, 'status': response.status_code,
                  'response_bytes': len(content),
                  'headers': {key: value for key, value in response.headers.items()
                              if key.lower() in {'date', 'content-type', 'content-length',
                                                 'content-range', 'last-modified', 'etag', 'dav'}},
                  'sha256': hashlib.sha256(content).hexdigest(), 'entries': []}
    raw_path.write_bytes(content)
    if result['status'] == 207:
        ns = {'d': 'DAV:'}
        for node in ET.fromstring(content).findall('d:response', ns):
            row = {key: node.findtext('.//d:' + key, default='', namespaces=ns)
                   for key in ('href', 'displayname', 'getcontentlength', 'getlastmodified', 'getetag')}
            row['directory'] = node.find('.//d:collection', ns) is not None
            result['entries'].append(row)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    assert result['status'] == 207, 'Keep the failed probe; do not request credentials'


if __name__ == '__main__':
    main()
