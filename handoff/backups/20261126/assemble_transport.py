"""Assemble bounded GitHub upload pieces into the original verified ZIP files."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import time


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--manifest', default='transport-manifest.json.gz')
    args = parser.parse_args()
    root = args.assets.resolve()
    assert Path(args.manifest).name == args.manifest
    with gzip.open(root/args.manifest, 'rt', encoding='utf-8') as stream:
        manifest = json.load(stream)
    assert manifest['version'] == 1
    for original in manifest['archives']:
        name = original['name']
        assert Path(name).name == name and name.endswith(('.zip', '.tar.gz'))
        target = root/name
        if target.exists():
            assert target.stat().st_size == original['bytes'] and sha(target) == original['sha256'], target
            continue
        temporary = root/(name+'.assembling-'+str(time.time_ns()))
        actual, total = hashlib.sha256(), 0
        with temporary.open('xb') as dest:
            for piece in original['parts']:
                assert Path(piece['name']).name == piece['name']
                source = root/piece['name']
                h, count = hashlib.sha256(), 0
                with source.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(1024*1024), b''):
                        h.update(chunk);actual.update(chunk);count += len(chunk);total += len(chunk)
                        dest.write(chunk)
                assert count == piece['bytes'] and h.hexdigest() == piece['sha256'], source
        assert total == original['bytes'] and actual.hexdigest() == original['sha256'], target
        assert not target.exists()
        temporary.rename(target)
        print('ASSEMBLED_SHA256_VERIFIED', name, total, flush=True)


if __name__ == '__main__':
    main()
