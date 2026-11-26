"""Inspect a public static ZIP through strictly bounded HTTP Range reads."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile
import requests

HERE = Path(__file__).resolve().parent
BASE = 'https://nx21496.your-storageshare.de/public.php/dav/files/Zd4rwsMwwXMaXiK/accid3nd_dataset/'
LIMIT = 2 * 1024 * 1024


class RangeFile(io.RawIOBase):
    def __init__(self, url, size, etag):
        self.url, self.size, self.etag = url, size, etag
        self.position = 0
        self.requests = []
        self.downloaded = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        point = offset if whence == io.SEEK_SET else (
            self.position + offset if whence == io.SEEK_CUR else self.size + offset)
        if point < 0:
            raise ValueError('Negative offset')
        self.position = point
        return point

    def read(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b''
        assert count <= 10 * LIMIT and self.downloaded + count <= 10 * LIMIT
        chunks = []
        while count:
            chunk = self._read_part(min(count, LIMIT))
            assert chunk
            chunks.append(chunk)
            count -= len(chunk)
        return b''.join(chunks)

    def _read_part(self, count=-1):
        count = self.size - self.position if count < 0 else min(count, self.size - self.position)
        if count <= 0:
            return b''
        assert count <= LIMIT and self.downloaded + count <= 10 * LIMIT
        assert len(self.requests) < 20
        start, end = self.position, self.position + count - 1
        with requests.get(self.url, headers={'Range': 'bytes=%d-%d' % (start, end)},
                          stream=True, timeout=(10, 20), allow_redirects=False) as response:
            record = {'start': start, 'end': end, 'status': response.status_code,
                      'Content-Range': response.headers.get('Content-Range'),
                      'ETag': response.headers.get('ETag')}
            self.requests.append(record)
            assert response.status_code == 206, 'Server ignored Range; body was not downloaded'
            assert response.headers['Content-Range'] == 'bytes %d-%d/%d' % (start, end, self.size)
            assert response.headers.get('ETag') == self.etag, 'Public file version changed'
            body = response.raw.read(count + 1)
            assert len(body) == count
            record['sha256'] = hashlib.sha256(body).hexdigest()
        self.downloaded += count
        self.position += count
        return body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', choices=['train%d.zip' % i for i in range(1, 6)])
    args = parser.parse_args()
    path = HERE / (args.archive + '.index_v2.json')
    assert not path.exists()
    directory = json.loads((HERE / 'dataset_root.json').read_text())
    entry = next(row for row in directory['entries'] if row['displayname'] == args.archive)
    stream = RangeFile(BASE + args.archive, int(entry['getcontentlength']), entry['getetag'])
    result = {'archive': args.archive, 'url': stream.url, 'size': stream.size,
              'directory_metadata_sha256': hashlib.sha256((HERE / 'dataset_root.json').read_bytes()).hexdigest(),
              'status': 'NOT_FINISHED', 'entries': []}
    try:
        with zipfile.ZipFile(stream) as archive:
            result['entries'] = [{'name': row.filename, 'bytes': row.file_size,
                                  'compressed_bytes': row.compress_size, 'CRC32': '%08x' % row.CRC,
                                  'header_offset': row.header_offset, 'directory': row.is_dir()}
                                 for row in archive.infolist()]
        result['status'] = 'PASS'
    except Exception as error:
        result['status'] = 'STOPPED'
        result['error'] = type(error).__name__ + ': ' + str(error)
    result.update(requests=stream.requests, downloaded_bytes=stream.downloaded)
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'entries'}, indent=2), flush=True)
    print('ENTRIES', len(result['entries']), flush=True)
    for row in result['entries'][:12]:
        print(row, flush=True)
    assert result['status'] == 'PASS', 'Keep failed metadata probe; no full download fallback'


if __name__ == '__main__':
    main()
