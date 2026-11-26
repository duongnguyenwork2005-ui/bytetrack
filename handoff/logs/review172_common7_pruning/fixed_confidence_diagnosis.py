"""Summarize existing actual Nano c25 counts; no inference or threshold search."""
from pathlib import Path
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    source = HERE / 'actual_quality_comparison.json'
    data = json.loads(source.read_text(encoding='utf-8'))
    assert data['status'] == 'PASS' and data['images_per_model'] == 7190
    for path, digest in data['sha256'].items():
        assert sha(path) == digest, path
    gt = {'car': 53204, 'truck': 3054, 'bus': 1469, 'motorcycle': 226, 'bicycle': 407}
    modes = {}
    for mode, quality in data['actual_quality'].items():
        assert quality['images'] == 7190
        assert quality['precision_recall_at_fixed_confidence'] == .25
        assert set(quality['classes']) == set(gt)
        classes = {}
        for name, row in quality['classes'].items():
            assert row['GT_boxes'] == gt[name]
            tp, det = row['TP_c25_IoU50'], row['detections_c25']
            assert 0 <= tp <= min(det, gt[name]) and det > 0
            assert math.isclose(tp / det, row['precision_c25'], abs_tol=1e-14)
            assert math.isclose(tp / gt[name], row['recall_c25'], abs_tol=1e-14)
            classes[name] = {
                'GT': gt[name], 'detections': det, 'TP': tp,
                'FP': det - tp, 'FN': gt[name] - tp,
                'precision': tp / det, 'recall': tp / gt[name],
                'AP50': row['AP50'],
            }
        total_gt = sum(r['GT'] for r in classes.values())
        total_tp = sum(r['TP'] for r in classes.values())
        total_det = sum(r['detections'] for r in classes.values())
        modes[mode] = {
            'classes': classes, 'GT': total_gt, 'TP': total_tp,
            'detections': total_det, 'FP': total_det - total_tp,
            'FN': total_gt - total_tp,
            'micro_precision': total_tp / total_det,
            'micro_recall': total_tp / total_gt,
            'macro_precision': sum(r['precision'] for r in classes.values()) / 5,
            'macro_recall': sum(r['recall'] for r in classes.values()) / 5,
        }
    deltas = {}
    for reference in ('original', 'control'):
        old, new = modes[reference], modes['joint50']
        deltas[reference] = {
            k: new[k] - old[k] for k in ('TP', 'detections', 'FP', 'FN')
        }
        deltas[reference]['class_recall_delta_pp'] = {
            name: 100 * (new['classes'][name]['recall'] - old['classes'][name]['recall'])
            for name in gt
        }
    target = HERE / 'fixed_confidence_diagnosis.json'
    assert not target.exists(), 'Preserve an existing diagnosis'
    result = {
        'status': 'PASS', 'meaning': 'Stored detection counts and derivations verified, not a quality pass.',
        'images': 7190, 'confidence': .25, 'IoU_TP': .5,
        'modes': modes, 'joint50_minus_reference': deltas,
        'inference_or_threshold_tuning_performed': False,
        'limitations': [
            'Detection FN/FP here are not TrackEval tracking FN/FP.',
            'AP integrates confidence rankings; fixed-c25 precision/recall are a distinct operating point.',
            'Bike recall can rise while AP/precision fall; this is not a uniform recall-only loss.',
            'Common7 secondary analysis; repeated development data and only 226 motorcycle GT boxes.',
        ],
        'sha256': {str(p): sha(p) for p in (source, Path(__file__))},
    }
    target.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'modes': {k: {f: v[f] for f in ('TP', 'FP', 'FN', 'micro_precision', 'micro_recall')} for k, v in modes.items()}, 'deltas': deltas}, indent=2))


if __name__ == '__main__':
    main()
