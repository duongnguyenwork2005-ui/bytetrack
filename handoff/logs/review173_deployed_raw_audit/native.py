"""Nano3.6: compare the four existing engines on identical frozen input tensors."""
from pathlib import Path
import ctypes
import importlib.util
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'r167_fused_silu_20261121'
spec = importlib.util.spec_from_file_location('r173_frozen_helpers', HERE.parent / 'r164_latency_pruning_20261121/build_screen.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def compare(actual, reference, cutoff):
    scores_a = actual[:, 4:].max(axis=1)
    scores_b = reference[:, 4:].max(axis=1)
    mask = np.ones(scores_a.shape, dtype=bool) if cutoff is None else np.maximum(scores_a, scores_b) >= cutoff
    boxes = np.abs(actual[:, :4] - reference[:, :4]).transpose(0, 2, 1)[mask]
    scores = np.abs(actual[:, 4:] - reference[:, 4:]).transpose(0, 2, 1)[mask]
    box = float(boxes.max()) if len(boxes) else 0.0
    score = float(scores.max()) if len(scores) else 0.0
    return {'anchors': int(mask.sum()), 'max_box_error_network_pixels': box,
            'max_score_error': score, 'within_6px_02': box <= 6 and score <= .02,
            'argmax_class_changes': int(((actual[:, 4:].argmax(axis=1) != reference[:, 4:].argmax(axis=1)) & mask).sum())}


def main():
    assert not (HERE / 'complete.json').exists()
    other = HERE.parent / 'r172_common7_pruning_20261121/recovered/before.json'
    assert not other.exists(), 'Do not overlap R172 builds/captures'
    own = base.read(HERE / 'inputs.json')
    for path, digest in own['sha256'].items():
        assert base.sha(path) == digest, path
    config = base.read(PARENT / 'runtime_models.json')
    base.check(config)
    before = base.state()
    base.health(before['boot'])
    handles = [ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL) for path in config['libraries']]
    sys.path.insert(0, str(Path(config['package']) / 'src'))
    from nano_detector import TRTEngine
    models = {'original': config['original']}
    models.update(config['models'])
    engines = {}
    for mode, record in models.items():
        assert base.sha(record['engine']) == record['engine_sha256']
        engines[mode] = TRTEngine(record['engine'])
        assert engines[mode].in_shape == (1, 3, 384, 640) and engines[mode].out_shape == (1, 9, 5040)
        for tick in range(10):
            engines[mode](np.zeros((1, 3, 384, 640), dtype=np.float32))
    (HERE / 'arrays').mkdir(exist_ok=False)
    rows = []
    for index, fixture in enumerate(config['models']['control']['parity']):
        assert base.sha(fixture['path']) == fixture['sha256']
        with np.load(fixture['path'], allow_pickle=False) as archive:
            value, expected = archive['input'], archive['expected']
        outputs = {mode: np.array(engine(value), copy=True) for mode, engine in engines.items()}
        assert all(output.shape == expected.shape == (1, 9, 5040) and np.isfinite(output).all()
                   for output in outputs.values())
        array = HERE / 'arrays' / ('%02d.npz' % index)
        np.savez_compressed(array, ORT_FP32=expected, **outputs)
        row = {'fixture': fixture, 'array': str(array), 'array_sha256': base.sha(array), 'comparisons': {}}
        for mode, output in outputs.items():
            row['comparisons'][mode] = {
                reference: {name: compare(output, tensor, cutoff)
                            for name, cutoff in (('all', None), ('c001', .001), ('c25', .25))}
                for reference, tensor in (('ORT_FP32', expected), ('deployed_FP16', outputs['original']))
            }
        rows.append(row)
        base.write(HERE / 'partial.json', rows)
    base.check(config)
    assert not other.exists(), 'R172 started during diagnostic; retain results as invalid overlap'
    after = base.health(before['boot'])
    base.write(HERE / 'complete.json', {'status': 'PASS', 'fixtures': len(rows), 'models': list(models),
               'rows': rows, 'before': before, 'after': after, 'engines': models,
               'existing_R167_FAIL_unchanged': True, 'Nano_FPS_measured': False,
               'input_sha256': base.sha(HERE / 'inputs.json')})
    print('R173 RAW REFERENCE DIAGNOSTIC COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
