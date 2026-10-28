"""Build an isolated R62 candidate from the byte-verified v16 ZIP, on the PC."""
import hashlib
import json
from pathlib import Path
import zipfile

REPO = Path(__file__).resolve().parents[3]
BASE_SHA = '302f6af73d97aea009066416aaaa202db81311e60121676a75dacbe204e9cbf1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    source = REPO / 'handoff/nano_transfer_latest.zip'
    target = REPO / 'handoff/archive/nano_r62_candidate_v17.zip'
    report = REPO / 'handoff/logs/review62/27_goi_ung_vien.json'
    assert not target.exists(), str(target)
    assert sha(source.read_bytes()) == BASE_SHA
    with zipfile.ZipFile(source) as z:
        manifest = z.read('nano_transfer/MANIFEST_SHA256.txt').decode('utf8')
        files = {}
        for line in manifest.splitlines():
            digest, name = line.split('  ', 1)
            data = z.read('nano_transfer/' + name)
            assert sha(data) == digest, name
            files[name] = data
    assert len(files) == 562
    old_hashes = {name: sha(data) for name, data in files.items()}
    files['src/nano_detector.py'] = (REPO / 'src/nano_detector.py').read_bytes()
    assert sha(files['src/nano_detector.py']) == '6a1165f06ad5f36e48c95a43a1e1c96022c0355058c9669830055dee0e057afe'
    info = json.loads(files['PACKAGE_INFO.json'])
    info['r62_candidate'] = {
        'label': 'v17-candidate', 'status': 'UNACCEPTED; benchmark only',
        'baseline_zip_sha256': BASE_SHA,
        'change': 'uint8 BGR HWC to float32 RGB NCHW via cv2.LUT; fallback unchanged',
        'changed_code': {'src/nano_detector.py': sha(files['src/nano_detector.py'])},
        'date': '2026-10-29',
        'note': 'Other package fields describe original v16; no launcher deployment yet.'}
    files['PACKAGE_INFO.json'] = (json.dumps(info, ensure_ascii=False, indent=2) + '\n').encode('utf8')
    new_hashes = {name: sha(data) for name, data in files.items()}
    changed = [name for name in files if old_hashes[name] != new_hashes[name]]
    assert sorted(changed) == ['PACKAGE_INFO.json', 'src/nano_detector.py']
    files['MANIFEST_SHA256.txt'] = ''.join('%s  %s\n' % (new_hashes[name], name)
                                          for name in sorted(new_hashes)).encode('utf8')
    with zipfile.ZipFile(target, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, data in sorted(files.items()):
            z.writestr('nano_transfer/' + name, data)
    result = {'zip': str(target), 'sha256': sha(target.read_bytes()),
              'bytes': target.stat().st_size, 'baseline_sha256': BASE_SHA,
              'verified_files': 562, 'changed_files': {name: {'before': old_hashes[name],
              'after': new_hashes[name]} for name in changed},
              'manifest_sha256': sha(files['MANIFEST_SHA256.txt'])}
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
