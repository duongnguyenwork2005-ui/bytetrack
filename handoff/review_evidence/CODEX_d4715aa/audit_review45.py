"""Read-only review of the supplied round-45 package; synthetic CPU probes only.

Usage: python -B audit_review45.py --package EXTRACTED_ZIP --out NEW_DIR
Requires Python, NumPy and Git. No training, model loading, GPU or network calls.
Exit 0 means the audit completed, not that the reviewed draft passed.
"""
import argparse
import ast
import collections
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--package', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    p = args.package.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((p / 'MANIFEST.json').read_text())
    entries = manifest['file_trong_goi'] + manifest['file_bo_qua'] + manifest['file_ho_tro']
    csv_manifest = {r['path']: r for r in read_csv(p / 'MANIFEST.csv')}
    inventory = []
    for f in sorted(p.rglob('*')):
        if not f.is_file():
            continue
        b = f.read_bytes()
        text = b.decode('utf-8')
        row = {'path': f.relative_to(p).as_posix(), 'bytes': len(b), 'sha256': sha(b), 'lines': len(text.splitlines())}
        if f.suffix == '.py':
            tree = ast.parse(text, filename=str(f))
            row['python_ast_parsed'] = True
            row['functions'] = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        elif f.suffix == '.json':
            json.loads(text)
            row['json_parsed'] = True
        elif f.suffix == '.csv':
            row['csv_rows'] = len(read_csv(f))
        inventory.append(row)
    assert len(inventory) == 80
    for e in entries:
        f = p / (e['trong_goi'] or e['trung_noi_dung_voi'])
        b = f.read_bytes()
        assert len(b) == e['bytes'] and sha(b) == e['sha256_target'], e['path']
        for k, v in e.items():
            if k in csv_manifest[e['path']]:
                assert csv_manifest[e['path']][k] == str(v), (e['path'], k)
    patch = p / 'patch/delta_3e94b21..d4715aa.diff'
    changed_paths = re.findall(r'^diff --git a/.* b/(.*)$', patch.read_text(), re.M)
    assert len(changed_paths) == len(entries) == 75
    assert set(changed_paths) == {e['path'] for e in entries}
    patch_result = []
    with tempfile.TemporaryDirectory(prefix='r45_patch_') as td:
        root = Path(td)
        for e in entries:
            q = root / e['path']
            q.parent.mkdir(parents=True, exist_ok=True)
            q.write_bytes((p / (e['trong_goi'] or e['trung_noi_dung_voi'])).read_bytes())
        for reverse in (True, False):
            flags = ['--reverse'] if reverse else []
            for check in (True, False):
                cmd = ['git', '-c', 'core.autocrlf=false', 'apply', '--whitespace=nowarn', *flags]
                if check:
                    cmd += ['--check']
                cp = subprocess.run([*cmd, str(patch)], cwd=root, capture_output=True, text=True)
                patch_result.append({'reverse': reverse, 'check_only': check, 'exit': cp.returncode, 'stderr': cp.stderr})
                assert cp.returncode == 0
            for e in entries:
                q = root / e['path']
                expected = e['sha256_base'] if reverse else e['sha256_target']
                assert sha(q.read_bytes()) == expected if expected else not q.exists()
    ev = p / 'files/handoff/review_evidence/CODEX_3e94b21'
    legacy_count = 0
    for line in (ev / 'SHA256SUMS.txt').read_text().splitlines():
        expected, name = line.split(None, 1)
        f = ev / name.lstrip('*')
        if not f.exists() and name.lstrip('*') == 'CODEX_REVIEW_3e94b21.md':
            f = p / 'files/handoff/CODEX_REVIEW_3e94b21.md'
        assert sha(f.read_bytes()) == expected
        legacy_count += 1
    author = json.loads((ev / 'AUTHOR_MANIFEST.json').read_text())
    old_rows = {e['path']: e for key in ['file_trong_goi', 'file_bo_qua', 'file_ho_tro'] for e in author[key]}
    for e in entries:
        if e['sha256_base']:
            assert e['sha256_base'] == old_rows[e['path']]['sha256_target']
    inv = json.loads((ev / 'input_inventory.json').read_text())
    for e in inv['fixture_source_files']:
        b = (ev / e['path']).read_bytes()
        assert len(b) == e['bytes'] and sha(b) == e['sha256']
        assert old_rows[e['path'].removeprefix('input/')]['sha256_target'] == e['sha256']
    integrity = {'base': manifest['base_commit'], 'target': manifest['target_commit'], 'members': len(inventory),
                 'payloads': len(manifest['file_trong_goi']), 'aliases': len(manifest['file_bo_qua']),
                 'manifest_rows': len(entries), 'patch_paths': len(changed_paths), 'patch_commands': patch_result,
                 'modified_base_hashes_matched': sum(bool(e['sha256_base']) for e in entries),
                 'added_paths_absent_after_reverse': sum(not e['sha256_base'] for e in entries),
                 'target_hashes_after_apply': len(entries), 'legacy_checksums': legacy_count,
                 'legacy_inputs_match_author_manifest': len(inv['fixture_source_files']),
                 'scope': 'Internal package consistency; no independent full Git checkout or real-data verification.'}
    save(out / 'inventory.json', inventory)
    save(out / 'integrity.json', integrity)

    rd = p / 'logs/handoff/logs/review45'
    d = rd / 'phan_tich'
    tables = {f.name: read_csv(f) for f in d.glob('*.csv')}
    by_video = {r['video']: r for r in tables['theo_video.csv']}
    summary = json.loads((d / 'tong_hop.json').read_text())
    ignore = json.loads((d / 'vung_bo_qua.json').read_text())
    options = json.loads((d / 'dem_theo_phuong_an.json').read_text())
    assert len(by_video) == summary['so_video'] == 40
    totals = {k: sum(int(r[k]) for r in by_video.values()) for k in
              ['so_frame_xml', 'so_anh', 'anh_khong_frame', 'frame_khong_anh', 'so_box', 'so_vung_bo_qua']}
    assert totals == {'so_frame_xml': 56167, 'so_anh': 56340, 'anh_khong_frame': 173,
                      'frame_khong_anh': 0, 'so_box': 675774, 'so_vung_bo_qua': 163}
    assert sum(x['so_box'] for x in summary['loai'].values()) == totals['so_box']
    missing = tables['anh_khong_frame.csv']
    assert len(missing) == len({(r['video'], int(r['frame'])) for r in missing}) == 173
    by_gap = collections.defaultdict(list)
    for r in missing:
        num = int(r['frame'])
        a, b = map(int, r['doan'].split('-'))
        assert a <= num <= b and int(r['do_dai_doan']) == b - a + 1
        assert r['anh'] == f"data/extracted/DETRAC-Images/{r['video']}/img{num:05d}.jpg"
        by_gap[(r['video'], r['doan'])].append(num)
    for (video, gap), nums in by_gap.items():
        a, b = map(int, gap.split('-'))
        assert nums == list(range(a, b + 1))
    assert dict(collections.Counter(r['video'] for r in missing)) == summary['anh_khong_frame_theo_video']
    assert dict(collections.Counter(r['vi_tri'] for r in missing)) == summary['anh_khong_frame_theo_vi_tri']
    for video, r in by_video.items():
        assert int(r['so_anh']) - int(r['so_frame_xml']) == int(r['anh_khong_frame'])
        assert r['kich_thuoc'] == r['kich_thuoc_anh_cuoi'] == '960x540'
        assert re.fullmatch('[0-9a-f]{64}', r['xml_sha256'])
    for c, x in ignore['theo_loai'].items():
        assert x['tong'] == summary['loai'][c]['so_box']
        assert x['hop_pixel_ge_0.5'] == summary['loai'][c]['so_box_ioa_ge_0.5_voi_vung_bo_qua']
        assert x['hop_pixel_ge_0.5'] == (x['max_don_ge_0.5'] + x['hop_ge_0.5_nhung_max_don_lt_0.5']
                                         - x['max_don_ge_0.5_nhung_hop_lt_0.5'])
        assert x['hop_pixel_ge_0.5'] == sum(v.get(c, 0) for v in summary['video_co_gt_ioa_ge_0.5'].values())
    for fname, rs in tables.items():
        if fname.startswith(('mau_track_', 'mau_box_')):
            groups = collections.defaultdict(list)
            for r in rs:
                groups[r['loai']].append(int(r['stt']))
                assert r['video'] in by_video
                num = int(r.get('frame_giua', r.get('frame')))
                assert 1 <= num <= int(by_video[r['video']]['anh_max'])
                assert float(r['width']) > 0 and float(r['height']) > 0
            for inds in groups.values():
                assert inds == list(range(len(inds)))
    classification = {}
    for c in ['others', 'van']:
        source = tables[f'mau_track_{c}.csv']
        classified = tables[f'phan_loai_bang_mat_{c}.csv']
        assert len(source) == len(classified)
        groups = collections.defaultdict(lambda: {'tracks': 0, 'boxes': 0})
        for a, b in zip(source, classified):
            assert all(a[k] == b[k] for k in a)
            groups[b['nhom_bang_mat']]['tracks'] += 1
            groups[b['nhom_bang_mat']]['boxes'] += int(b['so_box_track'])
        classification[c] = dict(groups)
    label_writer = load_module('r45_label_writer', rd / 'ghi_phan_loai_bang_mat.py')
    with tempfile.TemporaryDirectory(prefix='r45_labels_') as td:
        root = Path(td)
        for c in ['others', 'van']:
            shutil.copyfile(d / f'mau_track_{c}.csv', root / f'mau_track_{c}.csv')
        argv = sys.argv[:]
        sys.argv = ['ghi_phan_loai_bang_mat.py', '--thu-muc', str(root)]
        with contextlib.redirect_stdout(io.StringIO()):
            assert label_writer.main() == 0
        sys.argv = argv
        for c in ['others', 'van']:
            assert (root / f'phan_loai_bang_mat_{c}.csv').read_bytes() == (d / f'phan_loai_bang_mat_{c}.csv').read_bytes()
    contributions = [{'video': r['video'], 'images_without_frame': int(r['anh_khong_frame']),
                      'static_regions': int(r['so_vung_bo_qua']),
                      'omitted_instances': int(r['anh_khong_frame']) * int(r['so_vung_bo_qua'])}
                     for r in by_video.values() if int(r['anh_khong_frame'])]
    omitted = sum(r['omitted_instances'] for r in contributions)
    assert omitted == 265
    static_annotated = sum(int(r['so_frame_xml']) * int(r['so_vung_bo_qua']) for r in by_video.values())
    corrected = []
    for name, values in options.items():
        extra_van = 38519 if 'D1_van=bo_qua' in name else 0
        extra_others = 16915 if 'D2_others=bo_qua' in name else 0
        keep = 'D4=giu_173_am_tinh' in name
        assert values['vung_bo_qua_cong_moi_anh'] == static_annotated + extra_van + extra_others
        assert values['gt_theo_lop']['car'] == (548555 if extra_van else 587074)
        assert values['gt_theo_lop']['bus'] == 71785
        assert values['gt_theo_lop'].get('truck', 0) == (0 if extra_others else 16915)
        assert values['so_anh_cham'] == (56340 if keep else 56167)
        if keep:
            corrected.append({'option': name, 'reported': values['vung_bo_qua_cong_moi_anh'],
                              'expected_with_static_ignore_on_retained_images': values['vung_bo_qua_cong_moi_anh'] + omitted})
    save(out / 'tables.json', {'totals': totals, 'csv_rows': {n: len(rs) for n, rs in tables.items()},
                             'classification': classification, 'classification_regenerated_byte_equal': True,
                             'missing_image_static_regions': contributions, 'omitted_total': omitted,
                             'corrected_retained_image_options': corrected,
                             'blank_camera_weather_rows': sum(not r['camera_state'] and not r['weather'] for r in by_video.values()),
                             'scope': 'All supplied CSV/JSON rows parsed and cross-checked; not recounted from original XML/images.'})

    sys.path.insert(0, str(ev / 'input/src'))
    evaluator = load_module('r45_evaluator', ev / 'input/src/eval_detector_5lop.py')
    parser = load_module('r45_parser', rd / 'phan_tich_detrac_test.py')
    counter = load_module('r45_counter', rd / 'dem_theo_phuong_an.py')
    def gt(boxes, classes, ign=()):
        return {'xyxy': np.asarray(boxes, float).reshape(-1, 4), 'cls': np.asarray(classes, int),
                'ign': np.asarray(ign, float).reshape(-1, 4)}
    def pred(boxes, classes, conf):
        return {'xyxy': np.asarray(boxes, float).reshape(-1, 4), 'cls': np.asarray(classes, int), 'conf': np.asarray(conf, float)}
    g = gt([[0, 0, 10, 10], [20, 0, 30, 10]], [0, 2])
    pr = pred([[0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]], [0, 2, 1], [.9, .8, .7])
    result = evaluator._json_safe(evaluator.danh_gia([g], [pr]))
    assert result['theo_lop']['truck']['AP50'] is None
    assert result['theo_lop']['truck']['van_hanh']['FP'] == 1
    assert result['theo_lop']['truck']['van_hanh']['precision'] == 0
    assert result['macro']['so_lop_co_gt'] == 2 and result['macro']['mAP50'] == 1
    probes = {'D3_no_gt_class': result}
    ign = [[0, 0, 10, 10]]
    pr = pred(ign, [0], [.9])
    before = evaluator._json_safe(evaluator.danh_gia([gt([[0, 0, 10, 10], [20, 0, 30, 10]], [0, 0], ign)], [pr]))
    after = evaluator._json_safe(evaluator.danh_gia([gt([[20, 0, 30, 10]], [0], ign)], [pr]))
    probes['D5_AP_direction_counterexample'] = {'keep_gt_in_ignore': before['theo_lop']['car'],
                                               'move_gt_in_ignore_to_ignored': after['theo_lop']['car']}
    assert before['theo_lop']['car']['AP50'] > after['theo_lop']['car']['AP50'] == 0
    with tempfile.TemporaryDirectory(prefix='r45_probe_') as td:
        root = Path(td)
        xml = root / 'data/extracted/DETRAC-Test-Annotations-XML/fixture.xml'
        xml.parent.mkdir(parents=True)
        xml.write_text('<sequence><sequence_attribute camera_state="stable" sence_weather="cloudy"/>'
                       '<ignored_region><box left="1" top="1" width="10" height="10"/>'
                       '<box left="20" top="20" width="10" height="10"/></ignored_region>'
                       '<frame num="1" density="1"><target_list><target id="1">'
                       '<box left="40" top="40" width="10" height="10"/>'
                       '<attribute vehicle_type="car"/></target></target_list></frame></sequence>')
        attrs = parser.doc_xml(xml)['attr']
        assert attrs == {}
        probes['XML_metadata_loss'] = {'expected': {'camera_state': 'stable', 'sence_weather': 'cloudy'}, 'actual': attrs}
        mf = root / 'results/phase2/manifests/detrac_test.json'
        mf.parent.mkdir(parents=True)
        mf.write_text('{"videos":["fixture"]}')
        imgs = root / 'data/extracted/DETRAC-Images/fixture'
        imgs.mkdir(parents=True)
        for i in [1, 2]:
            (imgs / f'img{i:05d}.jpg').write_bytes(b'filename-only counting fixture; never decoded')
        counter.REPO = root
        argv = sys.argv[:]
        sys.argv = ['dem_theo_phuong_an.py', '--out', str(root / 'counts.json')]
        with contextlib.redirect_stdout(io.StringIO()):
            assert counter.main() == 0
        sys.argv = argv
        counts = json.loads((root / 'counts.json').read_text())
        values = counts['D1_van=bo_qua|D2_others=bo_qua|D4=giu_173_am_tinh']
        assert values['so_anh_cham'] == 2 and values['vung_bo_qua_cong_moi_anh'] == 2
        probes['D4_synthetic'] = {'retained_images': 2, 'static_regions_per_image': 2,
                                  'reported_region_instances': 2, 'expected_region_instances': 4}
        npz = root / 'predictions.npz'
        empty = pred([], [], [])
        evaluator.luu_du_doan(npz, [Path('MVI_A/img00001.jpg'), Path('MVI_B/img00001.jpg')], [empty, empty])
        with np.load(npz, allow_pickle=False) as data:
            probes['NPZ_basename_design_note'] = {'input_paths': ['MVI_A/img00001.jpg', 'MVI_B/img00001.jpg'],
                                                  'saved_anh': data['anh'].tolist(),
                                                  'note': 'idx preserves positional association only; retain explicit ordered video/frame mapping for replay.'}
    probes['environment'] = {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__,
                             'torch_imported': 'torch' in sys.modules, 'ultralytics_imported': 'ultralytics' in sys.modules}
    assert not probes['environment']['torch_imported'] and not probes['environment']['ultralytics_imported']
    save(out / 'probes.json', probes)
    final = {'integrity_pass': True, 'findings_reproduced': ['D3_FP_despite_no_GT', 'D4_missing_static_ignore',
                                                           'D5_AP_not_guaranteed_to_increase', 'XML_metadata_loss'],
             'design_note': 'NPZ_basename_collision', 'package_unchanged': True,
             'training': False, 'model_inference': False, 'gpu': False, 'ssh': False}
    for row in inventory:
        assert sha((p / row['path']).read_bytes()) == row['sha256']
    assert len([f for f in p.rglob('*') if f.is_file()]) == len(inventory)
    save(out / 'summary.json', final)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
