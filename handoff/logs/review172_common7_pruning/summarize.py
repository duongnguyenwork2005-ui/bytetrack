"""Recheck the common seven-epoch checkpoint study and report compression, speed and quality separately."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
FULL = HERE / 'full80_evidence/runs_absolute/full'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify_bound(record, key='sha256'):
    for path, digest in record[key].items():
        assert sha(path) == digest, path


def main():
    target = HERE / 'final_summary.json'
    assert not target.exists(), 'Preserve an existing final comparison'
    frozen = read(HERE / 'training_inputs.json')
    assert frozen['status'] == 'PASS'
    verify_bound(frozen)
    assert frozen['accepted_common_epochs'] == 7 and frozen['primary10_complete'] is False
    proof_path = HERE / 'common_checkpoint_proof.json'
    proof = read(proof_path)
    assert proof['status'] == 'PASS' and proof['primary10_complete'] is False
    assert proof['new_training_performed'] is False
    provenance = {}
    bound = [Path(__file__), HERE / 'training_inputs.json', proof_path]
    for mode in ('control', 'joint50'):
        record = proof['models'][mode]
        assert record['accepted_epochs'] == 7
        assert sha(record['parent_checkpoint']) == record['parent_checkpoint_sha256']
        assert sha(record['copied_checkpoint']) == record['parent_checkpoint_sha256']
        selection_path = HERE / 'monitor' / mode / 'selection.json'
        selection = read(selection_path)
        assert selection['status'] == 'PASS' and selection['selected']['epoch_zero_based'] == 6
        assert sha(selection['selected']['checkpoint']) == record['parent_checkpoint_sha256']
        provenance[mode] = {
            'accepted_epochs': 7, 'checkpoint_was_resumed': False,
            'selected': selection['selected'],
            'parameters': frozen['models'][mode]['parameters'],
            'ConvMAC': frozen['models'][mode]['ConvMAC'],
        }
        bound.append(selection_path)
    exports = read(HERE / 'recovered_exports.json')
    assert exports['status'] == 'PASS'
    verify_bound(exports, 'source_sha256')
    for mode in provenance:
        onnx_path = Path(exports['models'][mode]['path'])
        assert sha(onnx_path) == exports['models'][mode]['onnx_sha256']
        provenance[mode]['ONNX_bytes'] = onnx_path.stat().st_size
    preflight = read(HERE / 'speed_preflight.json')
    assert preflight['status'] == 'PASS'
    verify_bound(preflight)
    score_path = FULL / 'score_verified.json'
    score = read(score_path)
    assert score['status'] == 'PASS' and score['clips'] == 80 and score['runs'] == 240
    assert score['verified_inference_frames'] == 36000
    assert set(score['models']) == {'original', 'control', 'joint50'}
    verify_bound(score)
    packages = {m['runtime_package'] for m in score['models'].values()}
    assert packages == {'/home/jetson/nano_run/r162_cuda_pre_20261120/package'}
    bound.extend((HERE / 'recovered_exports.json', HERE / 'speed_preflight.json', score_path))
    detection = score['detection']
    comparisons = {}
    for reference in ('original', 'control'):
        a, b = detection[reference], detection['joint50']
        gates = {key: b[key] >= a[key] - .005 for key in ('mAP50', 'mAP50_95')}
        class_deltas = {}
        for name in a['classes']:
            delta = b['classes'][name]['AP50'] - a['classes'][name]['AP50']
            gates[name + '_AP50'] = delta >= -.02
            class_deltas[name] = 100 * delta
        tracking = {}
        for dataset in ('bdd', 'ua'):
            old = score['quality'][dataset][reference]['combined']
            new = score['quality'][dataset]['joint50']['combined']
            tracking[dataset] = {
                'reference': old, 'candidate': new,
                'IDSW_delta': new['idsw'] - old['idsw'],
                'FN_delta': new['fn'] - old['fn'],
                'FP_delta': new['fp'] - old['fp'],
                'HOTA_delta_pp': 100 * (new['hota'] - old['hota']),
            }
        comparisons[reference] = {
            'AP_gates': gates,
            'all_AP_gates_pass': all(gates.values()),
            'mAP50_delta_pp': 100 * (b['mAP50'] - a['mAP50']),
            'mAP50_95_delta_pp': 100 * (b['mAP50_95'] - a['mAP50_95']),
            'class_AP50_delta_pp': class_deltas,
            'tracking': tracking,
            'mean_FPS_gain_pct': {
                dataset: 100 * (
                    score['speed'][dataset]['modes']['joint50']['mean_FPS'] /
                    score['speed'][dataset]['modes'][reference]['mean_FPS'] - 1
                ) for dataset in ('bdd', 'ua', 'all')
            },
        }
    control, student = provenance['control'], provenance['joint50']
    result = {
        'status': 'PASS',
        'meaning': 'Evidence and comparison verified; quality/raw failures remain failures.',
        'common7_training_checkpoints': provenance,
        'primary10_complete': False,
        'secondary_analysis': True,
        'new_training_performed': False,
        'control_tail7_9_and_all_resumed_tails_excluded': True,
        'compression': {
            key + '_reduction_pct': 100 * (1 - student[key] / control[key])
            for key in ('parameters', 'ConvMAC', 'ONNX_bytes')
        },
        'comparisons': comparisons,
        'raw_parity': preflight['raw_parity'],
        'clips': 80, 'runs': 240, 'inference_frames': 36000,
        'speed': score['speed'], 'detection': detection, 'quality': score['quality'],
        'stage_median_of_clip_medians_ms': score['stage_median_of_clip_medians_ms'],
        'launcher_or_defaults_changed': False,
        'limitations': [
            '150 JPEG/clip, 40 BDD + 40 UA, warm25, KF, display off; not full-video/camera FPS.',
            'Development data and repeated monitoring, not blind detection/tracking evaluation.',
            'Only uninterrupted R164 epoch6 per arm; the planned primary10 study remains incomplete.',
            'Both arms use R162 CUDA preprocessing; SiLU fusion is a separate R167 experiment.',
        ],
        'sha256': {str(path): sha(path) for path in bound},
    }
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('R172 FINAL COMPARISON VERIFIED', flush=True)
    for reference, row in comparisons.items():
        print(reference, row['mean_FPS_gain_pct'], row['mAP50_delta_pp'], row['tracking'], flush=True)


if __name__ == '__main__':
    main()
