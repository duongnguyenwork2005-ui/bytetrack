"""Independent review probes; no real dataset or project files are changed.

Usage: python REPRO_REVIEW_c4d64a1.py --repo PATH --out DIR
Exit 1 means one or more reported issues still reproduce; exit 0 means none do.
The validator under review is executed through its real CLI in temporary dirs.
"""
import argparse
import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    script = (args.repo / 'handoff/logs/review32/kiem_nhan_ung_vien.py').resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    mutations = ['clean', 'missing_root', 'missing_train', 'missing_coco',
                 'nan_train_box', 'labels_object', 'duplicate_track_in_frame',
                 'val_bbox_diff', 'val_class_diff', 'val_track_merge']
    for case in mutations:
        with tempfile.TemporaryDirectory(prefix='review32_') as td:
            d = Path(td)
            label = {'id': '1', 'category': 'car',
                     'attributes': {'occluded': False, 'truncated': False, 'crowd': False},
                     'box2d': {'x1': 10., 'y1': 20., 'x2': 40., 'y2': 60.}}
            coco = {'categories': [{'id': 1, 'name': 'car'}, {'id': 2, 'name': 'truck'}],
                    'videos': [{'id': 1, 'name': 'va'}], 'images': [], 'annotations': []}
            for split, video in [('train', 'tr'), ('val', 'va')]:
                frames = []
                for k in (1, 2):
                    name = f'{video}-{k:07d}.jpg'
                    img = d / 'images' / split / video / name
                    img.parent.mkdir(parents=True, exist_ok=True)
                    img.write_bytes(b'filename-inventory-only')
                    labs = [copy.deepcopy(label)]
                    if case == 'val_track_merge' and split == 'val':
                        labs.append(copy.deepcopy(label))
                        labs[-1]['id'] = '2'
                        labs[-1]['box2d'] = {'x1': 100., 'y1': 20., 'x2': 130., 'y2': 60.}
                    if split == 'train' and k == 1:
                        if case == 'nan_train_box': labs[0]['box2d']['x1'] = float('nan')
                        if case == 'labels_object': labs = {}
                        if case == 'duplicate_track_in_frame': labs.append(copy.deepcopy(label))
                    frames.append({'name': name, 'videoName': video, 'frameIndex': k-1, 'labels': labs})
                    if split == 'val':
                        coco['images'].append({'id': k, 'video_id': 1, 'file_name': video+'/'+name})
                        for n in labs:
                            b = n['box2d']
                            coco['annotations'].append({'id': len(coco['annotations'])+1, 'image_id': k,
                                'scalabel_id': n['id'], 'instance_id': 1, 'category_id': 1,
                                'ignore': 0, 'iscrowd': 0, 'bbox': [b['x1'], b['y1'], 31., 41.]})
                if case == 'missing_root' or (case == 'missing_train' and split == 'train'):
                    continue
                f = d/'labels'/split/(video+'.json')
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(json.dumps(frames))
            if case == 'val_bbox_diff': coco['annotations'][0]['bbox'][0] += 5.
            if case == 'val_class_diff': coco['annotations'][0]['category_id'] = 2
            if case != 'missing_coco': (d/'coco.json').write_text(json.dumps(coco))
            out = args.out/(case+'.json')
            cmd = [sys.executable, str(script), '--nhan', str(d/'labels'), '--anh', str(d/'images'),
                   '--coco-val', str(d/'coco.json'), '--ra', str(out)]
            p = subprocess.run(cmd, text=True, capture_output=True)
            (args.out/(case+'.log')).write_text(p.stdout+p.stderr)
            report = json.loads(out.read_text()) if out.exists() else {}
            result = {'case': case, 'exit': p.returncode, 'report_keys': list(report)}
            if case == 'clean':
                result['control_pass'] = p.returncode == 0
            elif case in ('missing_root', 'missing_train', 'nan_train_box', 'labels_object', 'duplicate_track_in_frame'):
                result['reproduced'] = p.returncode == 0
            else:
                # A comparison-only command may legitimately return 0 while
                # reporting differences. Do not force its exit policy here.
                result['observation_only'] = True
                result['success_exit'] = p.returncode == 0
            if 'train' in report: result['train_schema_errors'] = report['train']['loi_schema']
            if 'doi_chieu_val_coco' in report:
                c = report['doi_chieu_val_coco']
                result['val_box_diff'] = c['lech_box_lon_nhat']['w=x2-x1+1']
                result['val_id_inconsistency'] = c['so_track_id_anh_xa_bat_nhat_quan']
                result['val_class_map'] = c['anh_xa_lop_scalabel_coco_ignore']
            results.append(result)
            print(json.dumps(result, ensure_ascii=False))
    data = {'probes': results, 'bug_count': sum(x.get('reproduced', False) for x in results),
            'note': 'COCO difference cases expose an informational-only exit policy; source identity is not tested.'}
    (args.out/'summary.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return int(data['bug_count'] > 0 or not results[0]['control_pass'])


if __name__ == '__main__':
    sys.exit(main())
