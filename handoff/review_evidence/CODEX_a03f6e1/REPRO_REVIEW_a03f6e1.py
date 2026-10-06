"""Probe the new CLI's configuration metadata, not the AP implementation.
Uses the exact supplied entrypoint, with explicit dependency/test doubles.
No actual GT reader, predictor, scoring implementation, model or dataset is tested.
Usage: python REPRO_REVIEW_a03f6e1.py TREE --out repro.json
Exit 1: CLI metadata discrepancy reproduces; 0: resolved.
"""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
import types
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('tree', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    source = a.tree / 'src/danh_gia_detector_bdd_val.py'
    with tempfile.TemporaryDirectory(prefix='codex_r43_cli_') as td:
        root = Path(td)
        conf = types.ModuleType('config'); conf.PROJECT_ROOT = root
        bx = types.ModuleType('bdd_yolo_export')
        bx.LoiDuLieu = ValueError
        bx.NguonCocoVid = bx._hop_le_box = bx.vai_tro = lambda *a, **kw: None
        dc = types.ModuleType('detector_classes'); dc.LOP_5 = ['car']; dc.ID_LOP_5 = {'car': 0}
        ev = types.ModuleType('eval_detector_5lop')
        ev.QUY_TAC = {'du_doan': {'conf': .001}, 'van_hanh': {'conf': .25, 'iou': .5}}
        ev._json_safe = lambda x: x
        ev.sha256_file = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        ev.kich_thuoc_anh = lambda p: (64, 48)
        ev.luu_du_doan = lambda *a: None
        ev.du_doan_ultralytics = lambda *a: ([{'conf': np.array([.6])}], {'test_double': True})
        observed = []
        def score(gt, dd, threshold):
            observed.append(threshold)
            return {'theo_lop': {'car': {'n_gt': 1, 'AP50': 1., 'AP50_95': 1.,
                     'van_hanh': {'precision': 1., 'recall': 1.}}},
                    'macro': {'mAP50': 1., 'mAP50_95': 1.}, 'van_hanh': {'conf': threshold, 'iou': .5}}
        ev.danh_gia = score
        for n, m in [('config', conf), ('bdd_yolo_export', bx), ('detector_classes', dc), ('eval_detector_5lop', ev)]:
            sys.modules[n] = m
        spec = importlib.util.spec_from_file_location('subject_r43', source)
        subject = importlib.util.module_from_spec(spec); spec.loader.exec_module(subject)
        subject.kiem_va_doc_gt = lambda *args, **kw: (
            {'so_video': 1, 'dem': {'gt_theo_lop': {'car': 1}, 'vung_bo_qua': 0}},
            [root/'fixture.jpg'], [{'cls': np.array([0])}])
        ck = root/'placeholder.pt'; ck.write_bytes(b'not a model; hash-only CLI fixture')
        cases = []
        for threshold in [.25, .5]:
            out = root/str(threshold)
            sys.argv = [str(source), '--model', 'probe='+str(ck), '--sha256', 'probe='+ev.sha256_file(ck),
                        '--out', str(out), '--conf-van-hanh', str(threshold)]
            with contextlib.redirect_stdout(io.StringIO()):
                rc = subject.main()
            result = json.loads((out/'ket_qua.json').read_text())
            cases.append({'requested_conf': threshold, 'scorer_received': observed[-1], 'exit': rc,
                          'top_level_rules_conf': result['quy_tac']['van_hanh']['conf'],
                          'model_rules_conf': result['mo_hinh']['probe']['van_hanh']['conf']})
    bad = sum(c['scorer_received'] != c['top_level_rules_conf'] for c in cases)
    result = {'scope': __doc__.split('Usage:')[0].strip(),
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'cases': cases, 'bug_count': bad}
    a.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    return int(bad > 0)


if __name__ == '__main__':
    raise SystemExit(main())
