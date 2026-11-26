"""Recompute every diagnostic from captured arrays against the frozen FP32 fixtures."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / 'evidence'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    target = HERE / 'verification.json'
    assert not target.exists()
    completed = read(EVIDENCE / 'complete.json')
    assert completed['status'] == 'PASS' and completed['fixtures'] == 10
    assert completed['before']['boot'] == completed['after']['boot']
    assert completed['before']['SD_errors'] == completed['after']['SD_errors'] == '0'
    assert completed['existing_R167_FAIL_unchanged'] is True and completed['Nano_FPS_measured'] is False
    assert sha(EVIDENCE / 'inputs.json') == completed['input_sha256']
    assert sha(EVIDENCE / 'native.py') == sha(HERE / 'native.py')
    assert sha(EVIDENCE / '00_protocol.md') == sha(HERE / '00_protocol.md')
    prior = read(HERE.parent / 'review167_fused_silu/final_summary.json')
    for mode in completed['models']:
        assert completed['engines'][mode]['engine_sha256'] == prior['detection'][mode]['engine']['engine_sha256']
    summary = {}
    bound = [Path(__file__), EVIDENCE / 'complete.json', EVIDENCE / 'inputs.json', EVIDENCE / 'native.py',
             HERE.parent / 'review167_fused_silu/final_summary.json']
    for index, row in enumerate(completed['rows']):
        path = EVIDENCE / 'arrays' / Path(row['array']).name
        fixture = HERE.parent / 'review167_fused_silu/parity' / Path(row['fixture']['path']).name
        assert sha(path) == row['array_sha256'] and sha(fixture) == row['fixture']['sha256']
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        with np.load(fixture, allow_pickle=False) as archive:
            assert np.array_equal(arrays['ORT_FP32'], archive['expected'])
        assert all(value.shape == (1, 9, 5040) and np.isfinite(value).all() for value in arrays.values())
        for mode in completed['models']:
            a = arrays[mode][0]
            for reference in ('ORT_FP32', 'deployed_FP16'):
                b = arrays['ORT_FP32' if reference == 'ORT_FP32' else 'original'][0]
                for bucket, cutoff in (('all', None), ('c001', .001), ('c25', .25)):
                    mask = np.ones(5040, dtype=bool) if cutoff is None else np.maximum(a[4:].max(0), b[4:].max(0)) >= cutoff
                    error_box = float(abs(a[:4, mask] - b[:4, mask]).max()) if mask.any() else 0.0
                    error_score = float(abs(a[4:, mask] - b[4:, mask]).max()) if mask.any() else 0.0
                    count = int(((a[4:].argmax(0) != b[4:].argmax(0)) & mask).sum())
                    saved = row['comparisons'][mode][reference][bucket]
                    assert math.isclose(error_box, saved['max_box_error_network_pixels'], abs_tol=1e-12)
                    assert math.isclose(error_score, saved['max_score_error'], abs_tol=1e-12)
                    assert saved['anchors'] == int(mask.sum()) and saved['argmax_class_changes'] == count
                    passed = error_box <= 6 and error_score <= .02
                    assert saved['within_6px_02'] == passed
                    item = summary.setdefault(mode, {}).setdefault(reference, {}).setdefault(bucket, {
                        'max_box_error_network_pixels': 0.0, 'max_score_error': 0.0,
                        'anchors': 0, 'argmax_class_changes': 0, 'fixtures_passing_6px_02': 0})
                    item['max_box_error_network_pixels'] = max(item['max_box_error_network_pixels'], error_box)
                    item['max_score_error'] = max(item['max_score_error'], error_score)
                    item['anchors'] += int(mask.sum())
                    item['argmax_class_changes'] += count
                    item['fixtures_passing_6px_02'] += int(passed)
        bound.extend((path, fixture))
    target.write_text(json.dumps({'status': 'PASS', 'meaning': 'Posthoc raw-reference diagnostic integrity only',
        'fixtures': 10, 'summary': summary, 'R167_raw_FAIL_unchanged': True,
        'not_bit_exact_or_a_quality_gate_override': True, 'Nano_FPS_measured': False,
        'sha256': {str(path): sha(path) for path in bound}}, indent=2) + '\n', encoding='utf-8')
    print('R173 RAW REFERENCE AUDIT VERIFIED', flush=True)
    for mode in completed['models']:
        print(mode, summary[mode], flush=True)


if __name__ == '__main__':
    main()
