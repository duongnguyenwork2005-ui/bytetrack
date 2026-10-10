"""Independent read-only audit of pinned native labels vs supplied COCO val."""
import collections as C
import csv
import hashlib
import json
import math
import argparse
from pathlib import Path

P = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument('--nhan', type=Path, default=P/'remote/bdd100k/labels/box_track_20')
ap.add_argument('--coco', type=Path, default=P.parent/'upload/01-bdd_box_track_val_cocofmt.json')
ap.add_argument('--out', type=Path, default=P/'evidence')
args = ap.parse_args()
NATIVE, COCO, OUT = args.nhan, args.coco, args.out
OUT.mkdir(parents=True, exist_ok=True)
co = json.loads(COCO.read_bytes())
cats = {a['id']:a['name'] for a in co['categories']}
ims = {a['id']:a['file_name'] for a in co['images']}
lookup = {}; duplicates = 0
for a in co['annotations']:
    key = (ims[a['image_id']], a['scalabel_id'])
    duplicates += key in lookup
    lookup[key] = a
summary = {'coco_sha256':hashlib.sha256(COCO.read_bytes()).hexdigest(), 'coco_duplicate_keys':duplicates}
mapping = {'other vehicle':'car', 'other person':'pedestrian', 'trailer':'truck'}
class_diff=[];ignore_diff=[]; val_matched=set();id_forward={};id_reverse={};id_bad=[];box_bad=[]
for split in ('train','val'):
    count=C.Counter();classes=C.Counter();err=C.Counter();attributes=C.Counter();multi=[]
    for p in sorted((NATIVE/split).glob('*.json')):
        count['files']+=1
        fs=json.loads(p.read_bytes()); frames=set();trackcats=C.defaultdict(set)
        for f in fs:
            count['frames']+=1
            err['duplicate_frame']+=f['name'] in frames;frames.add(f['name'])
            labs=f.get('labels',[])
            count['labels_absent']+='labels' not in f
            if not isinstance(labs,list):err['labels_not_list']+=1;continue
            count['empty_frames']+=not labs
            ids=set()
            for a in labs:
                count['annotations']+=1;classes[a['category']]+=1
                err['duplicate_id_same_frame']+=a['id'] in ids;ids.add(a['id'])
                trackcats[a['id']].add(a['category'])
                b=a['box2d'];at=a['attributes'];attributes.update(at.keys())
                err['nonfinite_box']+=not all(type(b[k]) in (int,float) and math.isfinite(b[k]) for k in ('x1','y1','x2','y2'))
                err['invalid_attributes']+=not all(type(at.get(k)) is bool for k in ('occluded','truncated','crowd'))
                if split!='val':continue
                key=(p.stem+'/'+f['name'],a['id']);v=lookup.get(key)
                if v is None:err['val_only_native']+=1;continue
                val_matched.add(key)
                x,y,w,h=v['bbox']; expected=[b['x1'],b['y1'],b['x2']-b['x1']+1,b['y2']-b['y1']+1]
                if [x,y,w,h]!=expected:box_bad.append(key)
                detail={'video':p.stem,'frame':f['name'],'scalabel_id':a['id'],'instance_id':v['instance_id'],
                        'native_class':a['category'],'coco_class':cats[v['category_id']],
                        'native_crowd':at['crowd'],'coco_iscrowd':v.get('iscrowd'),
                        'native_ignore':at.get('ignored'),'coco_ignore':v.get('ignore')}
                if mapping.get(a['category'],a['category'])!=detail['coco_class']:class_diff.append(detail)
                if (a['category'] in mapping)!=bool(v.get('ignore')):ignore_diff.append(detail)
                fkey=(p.stem,a['id']);rkey=(p.stem,v['instance_id'])
                if id_forward.setdefault(fkey,v['instance_id'])!=v['instance_id']:id_bad.append(('one_to_many',key))
                if id_reverse.setdefault(rkey,a['id'])!=a['id']:id_bad.append(('many_to_one',key))
        count['tracks']+=len(trackcats)
        multi.extend({'video':p.stem,'id':k,'categories':sorted(v)} for k,v in trackcats.items() if len(v)>1)
    summary[split]={'counts':dict(count),'classes':dict(classes),'errors':dict(err),'attribute_keys':dict(attributes),
                    'multi_category_track_count':len(multi),'multi_category_examples':multi[:20]}
summary['val']|={'matched':len(val_matched),'coco_only':len(set(lookup)-val_matched), 'box_mismatch':len(box_bad),
                'id_bijection_conflicts':len(id_bad),'class_mismatch':len(class_diff),'ignore_exceptions':len(ignore_diff),
                'mismatch_videos':len({x['video'] for x in class_diff}), 'mismatch_tracks':len({(x['video'],x['scalabel_id']) for x in class_diff})}
for name,rows in [('class_mismatches',class_diff),('ignore_exceptions',ignore_diff)]:
    with (OUT/(name+'.csv')).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['empty']);w.writeheader();w.writerows(rows)
(OUT/'real_labels_audit.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
