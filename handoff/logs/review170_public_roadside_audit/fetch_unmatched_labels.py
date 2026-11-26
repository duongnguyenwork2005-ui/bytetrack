"""Read one JSON from each mismatched group using local ZIP headers and CRC."""
from pathlib import Path
import hashlib
import json
import struct
import zlib
from list_zip_v2 import RangeFile, BASE, LIMIT

HERE = Path(__file__).resolve().parent


def main():
    out = HERE / 'unmatched_samples'
    out.mkdir(exist_ok=False)
    directory = json.loads((HERE / 'dataset_root.json').read_text())
    audit = json.loads((HERE / 'inventory_verified.json').read_text())
    groups = {name for name, row in audit['groups'].items()
              if row['labels']['frames'] and not row['paired']['frames']}
    selected = {}
    by_archive = {}
    for path in sorted(HERE.glob('train*.zip.index_v2.json')):
        index = json.loads(path.read_text())
        by_archive[index['archive']] = index
        for row in index['entries']:
            parts = row['name'].split('/')
            if len(parts) != 4 or parts[1] != 'labels' or not row['name'].endswith('.json'):
                continue
            group = parts[0] + '/' + parts[2]
            if group in groups and (group not in selected or row['name'] < selected[group]['entry']['name']):
                selected[group] = {'archive': index['archive'], 'entry': row}
    assert set(selected) == groups
    streams = {}
    records = []
    for number, (group, source) in enumerate(sorted(selected.items())):
        name, entry = source['archive'], source['entry']
        if name not in streams:
            remote = next(row for row in directory['entries'] if row['displayname'] == name)
            streams[name] = RangeFile(BASE + name, int(remote['getcontentlength']), remote['getetag'])
        stream = streams[name]
        assert entry['bytes'] <= LIMIT and entry['compressed_bytes'] <= LIMIT
        stream.seek(entry['header_offset'])
        header = struct.unpack('<4s5H3I2H', stream.read(30))
        signature, version, flags, method, mtime, mdate, crc, compressed, size, name_length, extra_length = header
        assert signature == b'PK\x03\x04' and not flags & 1 and method in (0, 8)
        body = stream.read(name_length + extra_length + entry['compressed_bytes'])
        archive_name = body[:name_length].decode('utf-8' if flags & 0x800 else 'cp437')
        assert archive_name == entry['name']
        payload = body[name_length + extra_length:]
        if method == 8:
            decoder = zlib.decompressobj(-15)
            data = decoder.decompress(payload, LIMIT + 1)
            assert len(data) <= LIMIT and decoder.eof and not decoder.unconsumed_tail
        else:
            data = payload
        assert len(data) == entry['bytes'] and '%08x' % (zlib.crc32(data) & 0xffffffff) == entry['CRC32']
        label = json.loads(data)['openlabel']
        assert len(label['frames']) == 1
        frame = next(iter(label['frames'].values()))
        path = out / ('%02d.json' % number)
        path.write_bytes(data)
        prefix = group.split('/')[0] + '/images/' + group.split('/')[1] + '/'
        image_rows = sorted(row['name'] for item in by_archive.values() for row in item['entries']
                            if row['name'].startswith(prefix) and row['name'].endswith('.jpg'))
        references = frame.get('frame_properties', {}).get('image_file_names', [])
        records.append({'group': group, 'archive': name, 'label_name': entry['name'],
                        'frame_properties': frame.get('frame_properties'),
                        'explicit_image_reference_exists': bool(references) and all(
                            prefix + Path(ref).name in image_rows for ref in references),
                        'first_image_names': image_rows[:2], 'objects': len(frame.get('objects', {})),
                        'path': str(path), 'sha256': hashlib.sha256(data).hexdigest(), 'CRC32': entry['CRC32']})
        print(group, records[-1]['explicit_image_reference_exists'], references, flush=True)
    result = {'status': 'PASS', 'meaning': 'Transport/JSON integrity, not correctness of dataset pairing',
              'criterion': 'First lexical label in every group with zero exact filename pairs',
              'records': records,
              'ranges': {key: value.requests for key, value in streams.items()},
              'downloaded_bytes': sum(value.downloaded for value in streams.values()),
              'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), HERE / 'list_zip_v2.py', HERE / 'inventory_verified.json')}}
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print('MISMATCH SAMPLE AUDIT COMPLETE', len(records), result['downloaded_bytes'], flush=True)


if __name__ == '__main__':
    main()
