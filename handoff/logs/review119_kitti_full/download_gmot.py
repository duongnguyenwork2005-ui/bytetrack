"""Read author ZIP through HTTP Range; download only the four car sequences."""
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_STORED
import hashlib
import requests

HERE = Path(__file__).resolve().parent
OUT = Path('F:/LABOTARY_data/goc_tai_ve/GMOT40-R119')
URL = 'https://github.com/Spritea/GMOT40/releases/download/v0.1/GenericMOT_JPEG_Sequence.zip'
SIZE = 1498450771


class Remote(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.blocks = []
        self.received = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, pos, whence=0):
        self.pos = pos if whence == 0 else self.pos + pos if whence == 1 else SIZE + pos
        return self.pos
    def block(self, start, end):
        r = requests.get(URL, headers={'Range': f'bytes={start}-{end-1}'}, timeout=90, stream=True)
        assert r.status_code == 206, (r.status_code, dict(r.headers))
        assert r.headers['content-range'] == f'bytes {start}-{end-1}/{SIZE}'
        data = r.content
        assert len(data) == end - start
        self.received += len(data)
        self.blocks.append((start, end, data))
        return data
    def read(self, n=-1):
        n = SIZE - self.pos if n < 0 else n
        end = min(SIZE, self.pos + n)
        for first, last, data in self.blocks:
            if first <= self.pos and end <= last:
                result = data[self.pos-first:end-first]
                break
        else:
            assert n <= 4000000, ('unexpected unbounded ZIP read', n)
            result = self.block(self.pos, end)
        self.pos = end
        return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    remote = Remote()
    records = {}
    with ZipFile(remote) as source:
        members = source.infolist()
        print('METADATA', [x.filename for x in members if x.filename.endswith(('.ini', '.txt', '.json', '.mp4'))][:100], flush=True)
        for i in range(4):
            key = f'car-{i}'
            selected = [x for x in members if f'/{key}/' in '/' + x.filename and not x.is_dir()]
            print(key, len(selected), [x.filename for x in selected[:2]], flush=True)
            if sys.argv[1:] == ['inspect']:
                continue
            first = min(x.header_offset for x in selected)
            last_info = max(selected, key=lambda x: x.header_offset)
            last = min(SIZE, last_info.header_offset + 30 + len(last_info.filename.encode()) + len(last_info.extra) + last_info.compress_size + 1024)
            block = remote.block(first, last)
            part = OUT / f'{key}.partial.zip'
            content = {}
            with ZipFile(part, 'w', compression=ZIP_STORED) as target:
                for x in selected:
                    data = source.read(x)  # source checks CRC
                    target.writestr(x.filename, data)
                    content[x.filename] = {'sha256': hashlib.sha256(data).hexdigest(), 'source_crc': x.CRC, 'size': len(data)}
            path = part.with_name(f'{key}.zip')
            assert not path.exists()
            part.replace(path)
            records[key] = {'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'source_range': [first, last], 'range_sha256': hashlib.sha256(block).hexdigest(),
                            'members': content}
            (HERE / 'source_probes/gmot_downloads.json').write_text(json.dumps(
                {'url': URL, 'source_size': SIZE, 'received_bytes': remote.received, 'sequences': records}, indent=2))
            print('DOWNLOADED', key, path.stat().st_size, flush=True)
    print('NETWORK_BYTES', remote.received, flush=True)


if __name__ == '__main__': main()
