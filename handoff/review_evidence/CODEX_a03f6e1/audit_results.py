"""Independent arithmetic / cross-file audit of handed-over round-43 results.
Does NOT recompute AP from predictions, read real GT, or load checkpoints.
Usage: python audit_results.py TREE --out audit_results.json
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('tree', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    root = a.tree / 'handoff/logs/review43'
    out = root / 'ket_qua'
    checks = []

    def check(name, value):
        checks.append({'name': name, 'pass': bool(value)})

    def eq(x, y):
        return math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)

    def js(p):
        return json.loads(p.read_text(encoding='utf-8'))

    def rows(p):
        with p.open(encoding='utf-8', newline='') as f:
            return list(csv.DictReader(f))

    models = ['yolov8n_coco', 'yolov8n_ft', 'yolo11n_ft']
    weights = [
        'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36',
        '5f66776d3202389ca484f3d9e2f107dddb354d21a9f1762d19401d690a4a0d8a',
        'd7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923',
    ]
    cls = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
    gt = [341436, 27343, 9087, 902, 4135]
    macro_rows = {r['mo_hinh']: r for r in rows(out / 'bang_macro.csv')}
    class_rows = {(r['mo_hinh'], r['lop']): r for r in rows(out / 'bang_theo_lop.csv')}
    first = None
    summaries = []
    checkpoint_log = {r['ten']: r for r in js(root / '01_kiem_checkpoint.json')['checkpoint']}
    for name, sha in zip(models, weights):
        kq = js(out / name / 'ket_qua.json')
        m = kq['mo_hinh'][name]
        gs = js(out / name / 'giam_sat/tom_tat.json')
        d = kq['dataset']['dataset_da_xac_minh']
        check(name + ': report checkpoint hash consistent', m['sha256'] == sha == checkpoint_log[name]['sha256_file'])
        check(name + ': monitor exit zero', gs['exit_code'] == 0 and gs['ctrl_c'] == [])
        check(name + ': complete image count', kq['dataset']['n_anh'] == m['n_anh'] == m['so_anh_da_du_doan'] == 39973)
        check(name + ': no image limit', kq['dataset']['gioi_han_anh'] == 0)
        check(name + ': 200 videos and image hash checks reported', d['so_video'] == 200 and d['doi_chieu_hash_anh']['khop'] == 39973 and not d['doi_chieu_hash_anh']['lech'])
        check(name + ': annotations partition', sum(d['dem']['gt_theo_lop'].values()) + d['dem']['vung_bo_qua'] + d['dem']['annotation_lop_khac'] == 465512)
        check(name + ': effective conf agrees with rules', m['van_hanh'] == kq['quy_tac']['van_hanh'] == {'conf': .25, 'iou': .5})
        if first is None:
            first = kq
        else:
            for field in ['quy_tac', 'quy_tac_gt']:
                check(name + ': common ' + field, kq[field] == first[field])
            check(name + ': identical dataset identity/counts', d == first['dataset']['dataset_da_xac_minh'])
        for c, n in zip(cls, gt):
            r = m['theo_lop'][c]
            v = r['van_hanh']
            check(name + '/' + c + ': GT and FN', r['n_gt'] == n and v['TP'] + v['FN'] == n)
            check(name + '/' + c + ': P/R/F1 arithmetic',
                  eq(v['precision'], v['TP']/(v['TP']+v['FP'])) and eq(v['recall'], v['TP']/n)
                  and eq(v['f1'], 2*v['TP']/(2*v['TP']+v['FP']+v['FN'])))
            rr = class_rows[(name, c)]
            for key in ['AP50', 'AP75', 'AP50_95', 'n_gt', 'n_anh_co_lop']:
                check(name + '/' + c + ': table ' + key, eq(float(rr[key]), r[key]))
            check(name + '/' + c + ': AP ordered/range', 0 <= r['AP50_95'] <= r['AP50'] <= 1 and 0 <= r['AP75'] <= r['AP50'])
        for key, source in [('mAP50', 'AP50'), ('mAP50_95', 'AP50_95')]:
            mean = sum(m['theo_lop'][c][source] for c in cls)/5
            check(name + ': macro ' + key, eq(mean, m['macro'][key]) and eq(mean, float(macro_rows[name][key])))
        for key, source in [('precision_vh', 'precision'), ('recall_vh', 'recall')]:
            check(name + ': macro ' + key, eq(sum(m['theo_lop'][c]['van_hanh'][source] for c in cls)/5, m['macro'][key]))
        baseline = first['mo_hinh']['yolov8n_coco']
        if name != 'yolov8n_coco':
            for c in cls:
                rr = class_rows[(name, c)]
                for score in ['AP50', 'AP50_95']:
                    expected = 100*(m['theo_lop'][c][score]-baseline['theo_lop'][c][score])
                    check(name + '/' + c + ': delta ' + score, eq(float(rr['chenh_'+score+'_diem_vs_baseline']), expected))
        summaries.append({'model': name, **m['macro'], 'wall_seconds_monitor': gs['thoi_gian_s'],
                          'detections_reported': sum(r['n_det_conf_thap'] for r in m['theo_lop'].values())})
    own = a.tree / 'src/danh_gia_detector_bdd_val.py'
    check('entrypoint bytes match pre-run code hash', hashlib.sha256(own.read_bytes()).hexdigest() == 'd1e2fd22a01a5fa117e64d909b13434a1f5226b27b5c0ed0f768ddfb6cc84ba1')
    result = {'scope': __doc__.split('Usage:')[0].strip(), 'checks': checks,
              'passed': sum(x['pass'] for x in checks), 'failed': sum(not x['pass'] for x in checks), 'models': summaries}
    a.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, indent=2))
    return int(result['failed'] != 0)


if __name__ == '__main__':
    raise SystemExit(main())
