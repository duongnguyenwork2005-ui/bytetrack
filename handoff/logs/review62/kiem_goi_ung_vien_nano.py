"""Extract candidate ZIP after baseline, verify every manifest entry and exact code delta."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path('/home/jetson/nano_run/r62_20261028')
EXPECTED_ZIP = 'e1e155b0e37316e341343e502e48d329ae363f073cf57dfc83c90941e80d6bd1'
EXPECTED_CODE = '6a1165f06ad5f36e48c95a43a1e1c96022c0355058c9669830055dee0e057afe'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            data = f.read(1024 * 1024)
            if not data:
                return h.hexdigest()
            h.update(data)


def main():
    assert (ROOT / 'baseline1470_lan02/exit_code').read_text().strip() == '0'
    zpath = ROOT / 'nano_r62_candidate_v17.zip'
    assert sha(zpath) == EXPECTED_ZIP
    dest = ROOT / 'candidate'
    dest.mkdir()
    with zipfile.ZipFile(str(zpath)) as z:
        manifest = z.read('nano_transfer/MANIFEST_SHA256.txt').decode('utf8')
        names = set(z.namelist())
        for name in names:
            assert name.startswith('nano_transfer/') and '..' not in Path(name).parts and not name.startswith('/')
        assert len(names) == 563 and 'nano_transfer/MANIFEST_SHA256.txt' in names
        z.extractall(str(dest))
    pkg = dest / 'nano_transfer'
    entries = []
    for line in manifest.splitlines():
        digest, name = line.split('  ', 1)
        path = pkg / name
        assert sha(path) == digest, name
        entries.append(name)
    assert len(entries) == 562 and len(set(entries)) == 562
    assert sha(pkg / 'src/nano_detector.py') == EXPECTED_CODE
    base = ROOT / 'baseline/nano_transfer'
    differences = [name for name in entries if sha(base / name) != sha(pkg / name)]
    assert sorted(differences) == ['PACKAGE_INFO.json', 'src/nano_detector.py']
    result = {'status': 'verified_for_benchmark', 'zip_sha256': EXPECTED_ZIP,
              'manifest_entries': len(entries), 'changed_content': differences,
              'candidate_detector_sha256': EXPECTED_CODE,
              'baseline_detector_sha256': sha(base / 'src/nano_detector.py')}
    (ROOT / 'candidate_verified.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
