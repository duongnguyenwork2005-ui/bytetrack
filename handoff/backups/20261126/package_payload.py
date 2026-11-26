"""Create byte-verified deduplicated ZIP parts for a private draft-release backup."""
from pathlib import Path
import argparse
import gzip
import hashlib
import json
import subprocess
import time
import zipfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not list(out.glob('payload-*.zip')), 'Preserve any partial backup; select a new output directory.'
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    rows = {r['path']: r for r in selection['release_payload']}
    ignored = subprocess.run(['git', '-c', 'core.longpaths=true', 'ls-files', '--others', '--ignored',
                              '--exclude-standard', '-z', '--', 'models/'], cwd=root,
                             stdout=subprocess.PIPE, check=True).stdout
    added_models = []
    for raw in ignored.split(b'\0'):
        if not raw:
            continue
        rel = raw.decode('utf-8')
        path = root / rel
        if path.suffix.lower() in {'.pt', '.onnx', '.engine', '.so'} and not path.is_symlink():
            rows[rel] = {'path': rel, 'bytes': path.stat().st_size, 'ignored_model_added': True}
            added_models.append(rel)
    objects, files, assets = {}, [], []
    archive, current, raw_bytes = None, None, 0
    start = time.monotonic()

    def close_part():
        nonlocal archive, current, raw_bytes
        if archive is None:
            return
        archive.close()
        assert current.stat().st_size < 2 * 1024**3
        assets.append({'name': current.name, 'bytes': current.stat().st_size, 'sha256': digest(current)})
        print('PART_CLOSED', assets[-1], flush=True)
        archive, current, raw_bytes = None, None, 0

    try:
        for index, rel in enumerate(sorted(rows)):
            path = root / rel
            assert path.resolve().is_relative_to(root) and not path.is_symlink(), rel
            before = path.stat()
            sha = digest(path)
            assert path.stat().st_mtime_ns == before.st_mtime_ns and path.stat().st_size == before.st_size, rel
            if sha not in objects:
                if archive is not None and raw_bytes + before.st_size > 512 * 1024**2:
                    close_part()
                if archive is None:
                    current = out / ('payload-%03d.zip' % (len(assets) + 1))
                    archive = zipfile.ZipFile(current, 'x', zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True)
                member = 'objects/' + sha
                with path.open('rb') as source, archive.open(member, 'w', force_zip64=True) as dest:
                    actual = hashlib.sha256()
                    for chunk in iter(lambda: source.read(1024 * 1024), b''):
                        actual.update(chunk)
                        dest.write(chunk)
                assert actual.hexdigest() == sha and path.stat().st_mtime_ns == before.st_mtime_ns, rel
                objects[sha] = {'asset': current.name, 'member': member, 'bytes': before.st_size}
                raw_bytes += before.st_size
            files.append({'path': rel, 'sha256': sha, 'bytes': before.st_size, 'mtime_ns': before.st_mtime_ns})
            if (index + 1) % 5000 == 0:
                print('PACKED', index + 1, '/', len(rows), 'unique', len(objects),
                      'elapsed_s', round(time.monotonic() - start, 1), flush=True)
    finally:
        close_part()
    manifest = {'version': 1, 'repository': 'duongnguyenwork2005-ui/bytetrack',
                'files': files, 'objects': objects, 'assets': assets,
                'ignored_model_paths_added': added_models,
                'excluded_payload_trees': ['data/', 'tmp/', 'results/track_video/', 'handoff/archive/'],
                'not_a_backup_of_all_ignored_files_external_drives_or_Nano': True,
                'git_source_restored_separately': True}
    target = out / 'payload-manifest.json.gz'
    with gzip.open(target, 'wt', encoding='utf-8', compresslevel=6) as stream:
        json.dump(manifest, stream, ensure_ascii=False, separators=(',', ':'))
    result = {'status': 'PASS', 'files': len(files), 'unique_objects': len(objects),
              'logical_bytes': sum(r['bytes'] for r in files), 'ignored_models_added': len(added_models),
              'assets': assets + [{'name': target.name, 'bytes': target.stat().st_size, 'sha256': digest(target)}]}
    (out / 'package-complete.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('PACKAGE_COMPLETE', len(files), 'files', len(assets), 'parts', sum(a['bytes'] for a in assets), flush=True)


if __name__ == '__main__':
    main()
