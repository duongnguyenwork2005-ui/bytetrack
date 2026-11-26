"""Verify pilot timestamps/outputs with the same frozen R105 scoring helpers."""
from pathlib import Path
import argparse,csv,hashlib,json,math
from statistics import mean
from score_full import verify_timing,frame_checks,HERE

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('bundle',type=Path);args=parser.parse_args()
    bundle=args.bundle.resolve();logs=bundle/'runs_absolute/pilot'
    collection=read(bundle/'pilot_collection.json');inputs=read(logs/'inputs.json')
    done=read(logs/'complete.json');after=read(logs/'after.json');selection=read(bundle/'selection.json')
    assert collection['status']==done['status']=='PASS' and collection['phase']=='negative_pilot'
    assert collection['input_sha256']==sha(logs/'inputs.json') and collection['after_sha256']==sha(logs/'after.json')
    assert collection['complete_sha256']==sha(logs/'complete.json') and collection['selection_sha256']==sha(bundle/'selection.json')
    assert inputs['boot']==after['boot']==collection['boot'] and after['errors']=='0'
    assert selection['full_skipped'] and selection['pilot_complete_sha256']==sha(logs/'complete.json')
    paths=[Path(__file__),bundle/'pilot_collection.json',bundle/'selection.json',logs/'inputs.json',logs/'complete.json',logs/'after.json',logs/'pairs.json']
    for path,h in collection['source_sha256'].items():
        relative=Path(path.replace('/home/jetson/nano_run/','',1));assert not relative.is_absolute() and '..' not in relative.parts
        p=bundle/'executed_sources'/relative;assert sha(p)==h;paths.append(p)
    config_relative=Path(inputs['config_path'].replace('/home/jetson/nano_run/','',1))
    config_path=bundle/'executed_sources'/config_relative
    assert sha(config_path)==done['models_sha256'] and read(config_path)['models']==inputs['models']
    known=read(HERE.parent/'review109_nano80/results/inputs.json')['videos'];expected=[]
    for ds in ('bdd','ua'):
        group=[r for r in known if r['dataset']==ds];assert len(group)==40
        expected.extend(group[ix] for ix in (0,10))
    assert len(inputs['videos'])==4
    for a,b in zip(expected,inputs['videos']):
        assert all(a[k]==b[k] for k in ('dataset','name','first','last','n_frames'))
    rows=read(logs/'pairs.json');modes=inputs['mode_order']
    assert len(rows)==done['runs']==4*len(modes)
    assert len({(r['dataset'],r['name'],r['mode']) for r in rows})==len(rows)
    for row in rows:
        ds,name,mode=row['dataset'],row['name'],row['mode'];model=inputs['models'][mode]
        assert row['frames']==row['n_inferred']==150 and not row['skipped_frames']
        assert row['engine_sha256']==model['engine_sha256'] and row['max_temp_c']<75
        image_manifest=logs/(ds+'_'+name+'_images.json')
        prior=HERE.parent/'review109_nano80/results'/image_manifest.name
        assert len(read(image_manifest))==150 and read(image_manifest)==read(prior)
        assert row['source_sha256']==sha(image_manifest);paths.append(image_manifest)
        folder=logs/mode/'runs/full/pipeline'/ds/name/'B';meta=read(folder/'run_meta.json')
        assert meta['hoan_tat'] and not meta['hien_thi']['bat'] and meta['ma_thoat']==0
        assert meta['provenance']['tracker_backend']=='np'
        assert meta['hello_con']['letterbox_hw']==model['hw']
        assert meta['cau_hinh']['conf_min']==meta['cau_hinh']['det_conf']==.25
        pipeline=folder/'pipeline'/(name+'.csv')
        with pipeline.open(encoding='utf-8',newline='') as stream:frames=list(csv.DictReader(stream))
        assert not frame_checks(frames,'full',ds)
        times=verify_timing(frames,meta['thoi_gian']['tom_tat_sau_warmup'])
        assert math.isclose(times['fps_recalculated'],row['fps_steady'],rel_tol=1e-10)
        assert math.isclose(times['p90_recalculated_ms'],row['p90_latency_ms'],rel_tol=1e-10)
        for p,key in ((folder/'run_meta.json','meta'),(pipeline,'pipeline'),
                      (folder/'data'/(name+'.txt'),'mot'),(folder/'data_cls'/(name+'.csv'),'cls')):
            assert sha(p)==row[key+'_sha256'];paths.append(p)
        if mode=='original':assert row['R109_MOT_CLS_byte_parity'] is True
    means={m:mean(r['fps_steady'] for r in rows if r['mode']==m) for m in modes}
    for mode,value in means.items():assert math.isclose(value,selection['pilot_mean_FPS'][mode],rel_tol=1e-12)
    result={'status':'PASS','phase':'negative_pilot','clips':4,'runs':len(rows),'mean_FPS':means,
        'full80_measured':False,'Nano_images_per_run':150,'warmup':25,
        'scope':'Only fixed pilot indices0/10 in BDD and UA; do not label as a full80 average.',
        'source_sha256':{str(p):sha(p) for p in paths}}
    target=bundle/'score_pilot_verified.json';assert not target.exists();target.write_text(json.dumps(result,indent=2)+'\n')
    print('NEGATIVE PILOT VERIFIED',json.dumps(means),flush=True)
if __name__=='__main__':main()
