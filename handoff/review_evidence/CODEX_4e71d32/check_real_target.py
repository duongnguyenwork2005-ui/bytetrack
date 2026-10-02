"""Read pinned native labels, verify hashes, exercise corrected parser/COCO comparison.

No image inventory is synthesized. Image completeness on the Windows PC is NOT checked.
Requires pip's bundled requests for HTTP and the unchanged supplied range reader.
Only one temporary native JSON is on disk at a time.
"""
import collections
import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path
from pip._vendor import requests

P=Path(__file__).resolve().parent
ROOT=P/'tree/handoff/logs/review32'
OUT=P/'evidence'
sys.modules.setdefault('requests', requests)


def mod(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


def counters():
    d=collections.defaultdict(int)
    for k in ('json_hong','khong_phai_list'):d[k]=[]
    for k in ('theo_lop','crowd_theo_lop'):d[k]=collections.Counter()
    return d


def main():
    global ROOT, OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--snapshot',type=Path,default=P/'tree')
    ap.add_argument('--coco',type=Path,default=P.parent/'upload/01-bdd_box_track_val_cocofmt.json')
    ap.add_argument('--out',type=Path,default=P/'evidence')
    args=ap.parse_args()
    ROOT=args.snapshot/'handoff/logs/review32';OUT=args.out
    if not args.coco.is_file():ap.error('COCO val file is missing; no network download started')
    OUT.mkdir(parents=True,exist_ok=True)
    r=mod('range_reader',ROOT/'liet_ke_zip_tu_xa.py')
    checker=mod('checker',ROOT/'kiem_nhan_ung_vien.py')
    manifest={a['duong_dan']:a for a in csv.DictReader((ROOT/'09_manifest_nhan_ung_vien.tsv').open(),delimiter='\t')}
    url='https://huggingface.co/datasets/jaffe03195/bdd100k_sot/resolve/694197099692fe1254e93215e73e68ece8b1f41f/bdd100k.zip'
    http=r.FileHttp(url);z=zipfile.ZipFile(http)
    infos=sorted([i for i in z.infolist() if i.filename in manifest and ('/train/' in i.filename or '/val/' in i.filename)],key=lambda i:i.header_offset)
    assert len(infos)==1600
    counts={'train':counters(),'val':counters()};val={};hashes=[]
    with tempfile.TemporaryDirectory(prefix='review33_labels_') as td:
        for k,i in enumerate(infos,1):
            b=z.read(i);h=hashlib.sha256(b).hexdigest()
            assert h==manifest[i.filename]['sha256'],i.filename
            hashes.append({'path':i.filename,'sha256':h,'bytes':len(b),'match':True})
            split='train' if '/train/' in i.filename else 'val'
            f=Path(td)/Path(i.filename).name;f.write_bytes(b)
            video,frames=checker.kiem_file(f,counts[split],split=='val');f.unlink()
            assert video is not None
            if split=='val':val[video]=frames
            if k%200==0:print('checked',k,'files, fetched',round(http.da_tai/1e6,1),'MB',flush=True)
    coco_path=args.coco
    c=checker.doc_coco(coco_path)
    comparison,status=checker.doi_chieu_coco(dict(sorted(val.items())),c,coco_path,OUT/'real_target')
    errors={s:{k:d[k] for k in checker.KHOA_LOI if d[k]} for s,d in counts.items()}
    result={'native_files':len(hashes),'manifest_mismatches':0,'counts':counts,'schema_errors':errors,
            'comparison_status':status,'comparison':comparison,
            'coco_sha256':hashlib.sha256(coco_path.read_bytes()).hexdigest(),
            'scope':'Real native annotations through corrected kiem_file and doi_chieu_coco; no real image inventory available; not a full kiem_split/CLI data run.',
            'download_bytes':http.da_tai,'source':url}
    for s,d in counts.items():assert not d['json_hong'] and not d['khong_phai_list'] and not errors[s]
    assert status=='DIFFERENCES'
    expected={'nhan_chi_co_o_native':0,'annotation_chi_co_o_coco':0,'box_lech':0,
              'id_mot_native_nhieu_coco':0,'id_mot_coco_nhieu_native':0,'lop_khac_anh_xa':183,
              'ngoai_le_ignore':50,'iscrowd_ngoai_quy_tac':0,'coco_khoa_trung':0,
              'coco_image_id_trung':0,'coco_tham_chieu_anh_khong_ton_tai':0}
    assert comparison['khac_biet']==expected
    old=args.snapshot/'handoff/review_evidence/CODEX_c4d64a1/evidence'
    csv_checks={}
    for a,b in [('real_target_lop_khac.csv','class_mismatches.csv'),('real_target_ngoai_le_ignore.csv','ignore_exceptions.csv')]:
        csv_checks[a]=(OUT/a).read_bytes()==(old/b).read_bytes()
        assert csv_checks[a]
    result['csv_byte_equal_previous_review']=csv_checks
    (OUT/'real_target.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    (OUT/'native_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps({'native_files':len(hashes),'schema_errors':errors,'comparison_status':status,
                      'differences':comparison['khac_biet'],'csv_checks':csv_checks},indent=2),flush=True)


if __name__=='__main__':main()
