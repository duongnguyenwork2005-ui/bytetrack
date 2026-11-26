"""Fetch the first filename-matched image/label only, with ZIP CRC and version checks."""
from pathlib import Path
import hashlib
import json
import zipfile
from list_zip_v2 import RangeFile, BASE, LIMIT

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = HERE / 'sample'
    out.mkdir(exist_ok=False)
    inventory = json.loads((HERE / 'archive_inventory.json').read_text())
    pair = inventory['first_pairs'][0]
    assert pair['label_archives'] == pair['image_archives'] and len(pair['label_archives']) == 1
    name = pair['label_archives'][0]
    index_path = HERE / (name + '.index_v2.json')
    index = json.loads(index_path.read_text())
    directory = json.loads((HERE / 'dataset_root.json').read_text())
    entry = next(row for row in directory['entries'] if row['displayname'] == name)
    stream = RangeFile(BASE + name, int(entry['getcontentlength']), entry['getetag'])
    records = []
    with zipfile.ZipFile(stream) as archive:
        for key, output_name in (('label', 'label.json'), ('image', 'frame.jpg')):
            info = archive.getinfo(pair[key])
            frozen = next(row for row in index['entries'] if row['name'] == pair[key])
            assert info.file_size <= LIMIT and info.compress_size <= LIMIT
            assert info.file_size == frozen['bytes'] and info.compress_size == frozen['compressed_bytes']
            assert '%08x' % info.CRC == frozen['CRC32'] and info.header_offset == frozen['header_offset']
            data = archive.read(info)  # zipfile validates decompression and CRC before returning.
            assert len(data) == info.file_size
            path = out / output_name
            path.write_bytes(data)
            records.append({'kind': key, 'ZIP_entry': pair[key], 'CRC32': frozen['CRC32'],
                            'bytes': len(data), 'path': str(path), 'sha256': sha(path)})
    label = json.loads((out / 'label.json').read_text())
    result = {'status': 'PASS', 'criterion': 'First lexical filename-matched image/label; no tracker points used',
              'records': records, 'range_requests': stream.requests, 'downloaded_bytes': stream.downloaded,
              'label_root_keys': list(label),
              'source_sha256': {str(p): sha(p) for p in (Path(__file__), HERE / 'list_zip_v2.py',
                                                       index_path, HERE / 'archive_inventory.json')}}
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print('SAMPLE VERIFIED', result['downloaded_bytes'], result['label_root_keys'], flush=True)


if __name__ == '__main__':
    main()
