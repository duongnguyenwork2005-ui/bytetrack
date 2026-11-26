"""Restore release ZIP parts beside a Git checkout; refuse conflicting existing files."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import os
import zipfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--dest', type=Path, required=True)
    args = parser.parse_args()
    base, root = args.assets.resolve(), args.dest.resolve()
    with gzip.open(base / 'payload-manifest.json.gz', 'rt', encoding='utf-8') as stream:
        manifest = json.load(stream)
    assert manifest['version'] == 1
    for asset in manifest['assets']:
        path = base / asset['name']
        assert path.stat().st_size == asset['bytes'] and sha(path) == asset['sha256'], path
    archives = {}
    try:
        for index, row in enumerate(manifest['files']):
            relative = Path(row['path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            target = root / relative
            assert target.resolve().is_relative_to(root)
            if target.exists():
                assert target.is_file() and sha(target) == row['sha256'], 'Conflicting existing file: ' + str(target)
                continue
            obj = manifest['objects'][row['sha256']]
            name = obj['asset']
            if name not in archives:
                archives[name] = zipfile.ZipFile(base / name)
            target.parent.mkdir(parents=True, exist_ok=True)
            actual = hashlib.sha256()
            with archives[name].open(obj['member']) as source, target.open('xb') as dest:
                for chunk in iter(lambda: source.read(1024 * 1024), b''):
                    actual.update(chunk)
                    dest.write(chunk)
            assert actual.hexdigest() == row['sha256'] and target.stat().st_size == row['bytes'], target
            os.utime(target, ns=(row['mtime_ns'], row['mtime_ns']))
            if (index + 1) % 5000 == 0:
                print('RESTORED', index + 1, flush=True)
    finally:
        for archive in archives.values():
            archive.close()
    print('RESTORE_VERIFIED', len(manifest['files']), 'files')


if __name__ == '__main__':
    main()
