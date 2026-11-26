"""Check names/timestamp gaps without inferring annotations or tracker performance."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timestamp(name):
    fields = Path(name).stem.split('_')
    seconds, nanos = int(fields[0]), int(fields[1])
    assert 0 <= nanos < 1000000000
    return seconds * 1000000000 + nanos


def main():
    target = HERE / 'inventory_verified.json'
    assert not target.exists()
    files = sorted(HERE.glob('train*.zip.index_v2.json'))
    assert len(files) == 5
    lookup = {}
    by_group = defaultdict(lambda: {'images': [], 'labels': [], 'paired': []})
    compressed = metadata = 0
    for path in files:
        record = json.loads(path.read_text())
        assert record['status'] == 'PASS'
        assert record['directory_metadata_sha256'] == sha(HERE / 'dataset_root.json')
        compressed += record['size']
        metadata += record['downloaded_bytes']
        assert all(row['status'] == 206 and row['end'] - row['start'] + 1 <= 2 * 1024 * 1024
                   for row in record['requests'])
        for row in record['entries']:
            if row['directory']:
                continue
            name = row['name']
            assert name not in lookup
            lookup[name] = {'archive': record['archive'], **row}
    images = {name for name in lookup if name.endswith('.jpg')}
    labels = {name for name in lookup if name.endswith('.json')}
    matched = set()
    unmatched_labels = []
    for name in sorted(labels):
        parts = name.split('/')
        group = parts[0] + '/' + parts[2]
        by_group[group]['labels'].append(timestamp(name))
        image = name.replace('/labels/', '/images/').removesuffix('.json') + '.jpg'
        if image in images:
            matched.add(image)
            by_group[group]['paired'].append(timestamp(name))
        else:
            unmatched_labels.append({'name': name, 'archive': lookup[name]['archive']})
    for name in images:
        parts = name.split('/')
        by_group[parts[0] + '/' + parts[2]]['images'].append(timestamp(name))
    groups = {}
    for group, data in sorted(by_group.items()):
        row = {}
        for kind, times in data.items():
            times = sorted(times)
            assert len(times) == len(set(times))
            gaps = [b - a for a, b in zip(times, times[1:])]
            assert all(gap > 0 for gap in gaps)
            row[kind] = {
                'frames': len(times),
                'span_seconds': (times[-1] - times[0]) / 1e9 if times else None,
                'delta_ns_most_common': Counter(gaps).most_common(12),
                'delta_gt_42ms': sum(gap > 42000000 for gap in gaps),
                'delta_gt_1second': sum(gap > 1000000000 for gap in gaps),
                'max_delta_seconds': max(gaps) / 1e9 if gaps else None,
            }
        groups[group] = row
    sample = json.loads((HERE / 'sample/label.json').read_text())['openlabel']
    assert list(sample['frames']) == ['0']
    frame = sample['frames']['0']
    first_pair = json.loads((HERE / 'archive_inventory.json').read_text())['first_pairs'][0]
    assert frame['frame_properties']['image_file_names'] == [Path(first_pair['image']).name]
    assert abs(frame['frame_properties']['timestamp'] - timestamp(first_pair['image']) / 1e9) < .000001
    objects = frame['objects']
    result = {
        'status': 'PASS', 'meaning': 'Metadata integrity only; no dataset converter or tracking experiment validated.',
        'compressed_archive_bytes': compressed, 'ZIP_index_downloaded_bytes': metadata,
        'images': len(images), 'JSON_files': len(labels), 'filename_matched_pairs': len(matched),
        'images_without_same_name_JSON': len(images - matched),
        'JSON_without_same_name_image': len(unmatched_labels),
        'first_unmatched_labels': unmatched_labels[:3],
        'groups': groups,
        'sample': {'objects': len(objects),
                   'classes': dict(Counter(obj['object_data']['type'] for obj in objects.values())),
                   'every_object_has_bbox': all('bbox' in obj['object_data'] for obj in objects.values()),
                   'frame_properties_reference_matches_filename': True,
                   'visibility_attribute_present_in_this_sample': 'occlu' in json.dumps(frame).lower()},
        'limitations': [
            'Filename matching is provisional; unmatched JSON may contain an alternate explicit image reference.',
            '40ms/42ms is a diagnostic relative to nominal 25Hz, not an assumption of uniform released cadence.',
            'One label does not establish all class counts, ID continuity, occlusion semantics or annotation quality.',
            'Website/paper release sizes differ; record this specific public archive version separately.',
        ],
        'sha256': {str(path): sha(path) for path in [Path(__file__), HERE / 'dataset_root.json',
                                                    HERE / 'sample/verification.json', HERE / 'sample/label.json',
                                                    HERE / 'sample/frame.jpg', *files]},
    }
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('groups', 'sha256')}, indent=2))
    print('GROUPS', len(groups), flush=True)


if __name__ == '__main__':
    main()
