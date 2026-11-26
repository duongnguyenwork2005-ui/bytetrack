"""Verify every Nano timestamp/output and reuse the locked TrackEval path."""
from argparse import ArgumentParser
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
from statistics import mean,median,harmonic_mean
import sys
from tempfile import TemporaryDirectory
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'review105_nano_fps'))
from eval_benchmark import quality,verify_timing,evaluation,SPLITS,frame_checks

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def speed(rows,modes):
    output={};rng=np.random.default_rng(155)
    for dataset in ('bdd','ua','all'):
        group=[r for r in rows if dataset=='all' or r['dataset']==dataset]
        per={};paired={};names=sorted({(r['dataset'],r['name']) for r in group})
        points={(r['dataset'],r['name'],r['mode']):r for r in group}
        for mode in modes:
            values=[points[ds,name,mode]['fps_steady'] for ds,name in names]
            per[mode]={'clips':len(values),'mean_FPS':mean(values),'median_FPS':median(values),
                'aggregate_FPS':harmonic_mean(values),'min_FPS':min(values),'max_FPS':max(values),
                'clips_at_least_15':sum(v>=15 for v in values),
                'median_clip_p90_latency_ms':median(points[ds,name,mode]['p90_latency_ms'] for ds,name in names)}
        if dataset=='all':
            strata=[np.array([ix for ix,(ds,_) in enumerate(names) if ds==label]) for label in ('bdd','ua')]
            sample=np.concatenate([rng.choice(index,size=(5000,len(index))) for index in strata],axis=1)
        else:sample=rng.integers(0,len(names),size=(5000,len(names)))
        for mode in modes[1:]:
            base=np.array([points[ds,name,modes[0]]['fps_steady'] for ds,name in names])
            candidate=np.array([points[ds,name,mode]['fps_steady'] for ds,name in names]);gain=100*(candidate/base-1)
            boot=np.median(gain[sample],axis=1)
            paired[mode]={'mean_FPS_gain_pct':100*(candidate.mean()/base.mean()-1),
                'median_paired_FPS_gain_pct':float(np.median(gain)),
                'CI95_median_paired_gain_pct':np.quantile(boot,[.025,.975]).tolist(),
                'videos_faster':int((candidate>base).sum()),'videos_total':len(names)}
        output[dataset]={'modes':per,'paired_vs_original':paired}
    return output

def main():
    parser=ArgumentParser();parser.add_argument('logs',type=Path);args=parser.parse_args();logs=args.logs.resolve()
    assert logs.is_relative_to(ROOT.resolve()),'Evidence must stay in the workspace'
    inputs=read(logs/'inputs.json');after=read(logs/'after.json');complete=read(logs/'complete.json')
    assert complete['status']=='PASS' and inputs['boot']==after['boot'] and after['errors']=='0'
    bundle=logs.parent.parent;collection=read(bundle/'collection.json')
    assert collection['status']=='PASS' and collection['boot']==inputs['boot']
    assert collection['input_sha256']==sha(logs/'inputs.json') and collection['after_sha256']==sha(logs/'after.json')
    assert collection['complete_sha256']==sha(logs/'complete.json')
    for path,digest in collection['source_sha256'].items():
        relative=Path(path.replace('/home/jetson/nano_run/','',1))
        assert not relative.is_absolute() and '..' not in relative.parts
        assert sha(bundle/'executed_sources'/relative)==digest,path
    videos=inputs['videos'];modes=inputs['mode_order'];assert len(videos)==80 and modes[0]=='original'
    models=inputs['models'];rows=read(logs/'pairs.json');assert len(rows)==complete['runs']==80*len(modes)
    points={(r['dataset'],r['name'],r['mode']):r for r in rows};assert len(points)==len(rows)
    prior=HERE.parent/'review109_nano80/results';historical=read(prior/'score_verified.json')
    known=read(prior/'inputs.json')['videos']
    for old,new in zip(known,videos):
        for key in ('dataset','name','first','last','n_frames'):assert old[key]==new[key]
    assert len(known)==80
    files=[Path(__file__),logs/'inputs.json',logs/'pairs.json',logs/'after.json',logs/'complete.json',bundle/'collection.json']
    detection={}
    for mode,model in models.items():
        path=bundle/'quality'/(mode+'.json');assert sha(path)==model['runtime_quality_sha256']
        detection[mode]=read(path);assert detection[mode]['engine']['engine_sha256']==model['engine_sha256'];files.append(path)
    verified=[];gtrows={};profiles=[]
    for video in videos:
        ds,name=video['dataset'],video['name'];split=SPLITS[ds][0]
        gt=ROOT/'data/processed'/split/name/'gt/gt.txt';files.append(gt)
        gtrows[ds,name]=evaluation.clipped_gt(gt,video['first'],video['last'])
        image_manifest=logs/(ds+'_'+name+'_images.json');images=read(image_manifest)
        assert len(images)==150 and images==read(prior/image_manifest.name);files.append(image_manifest)
        for mode in modes:
            row=points[ds,name,mode];model=models[mode]
            assert row['frames']==row['n_inferred']==150 and not row['skipped_frames']
            assert row['source_sha256']==sha(image_manifest) and row['engine_sha256']==model['engine_sha256']
            folder=logs/mode/'runs/full/pipeline'/ds/name/'B';meta=read(folder/'run_meta.json')
            assert meta['hoan_tat'] and meta['dem']['n_frame_da_xu_ly']==150
            assert meta['dem']['n_unreadable']==meta['dem']['n_truncated']==0 and meta['ma_thoat']==0
            hello=meta['hello_con'];cfg=meta['cau_hinh'];ua=ds=='ua'
            assert hello['engine_sha256']==model['engine_sha256'] and hello['onnx_sha256']==model['onnx_sha256']
            assert hello['letterbox_hw']==model['hw'] and hello['n_frames_khai_bao']==150
            assert cfg['conf_min']==cfg['det_conf']==.25 and cfg['det_iou']==.7
            assert cfg['max_det']==300 and cfg['det_max_det']==3000 and cfg['hang_doi']==4
            assert cfg['fps']==hello['fps']==(25 if ua else 5)
            assert cfg['track_buffer_frames']==(30 if ua else 6) and cfg['track_buffer_sec']==1.2
            assert cfg['elapsed_mode']==('skip' if ua else 'steps')
            assert cfg['classes']==([0,1,2,3] if ua else [0,1,2,3,4])
            assert cfg['class_map']=={'0':2,'1':7,'2':5,'3':3,'4':1}
            assert cfg['tracker_cfg']==dict(tracker_type='bytetrack',track_high_thresh=.25,track_low_thresh=.1,
                new_track_thresh=.25,track_buffer=30 if ua else 6,match_thresh=.8,fuse_score=True)
            assert meta['provenance']['tracker_backend']=='np' and not meta['hien_thi']['bat']
            assert (hello['regions_sha256'] is not None)==ua
            pipeline=folder/'pipeline'/(name+'.csv')
            with pipeline.open(encoding='utf-8',newline='') as f:frames=list(csv.DictReader(f))
            assert not frame_checks(frames,'full',ds)
            measured=verify_timing(frames,meta['thoi_gian']['tom_tat_sau_warmup'])
            assert math.isclose(measured['fps_recalculated'],row['fps_steady'],rel_tol=1e-10)
            assert math.isclose(measured['p90_recalculated_ms'],row['p90_latency_ms'],rel_tol=1e-10)
            stage_keys=('decode_ms','mask_ms','pre_ms','h2d_ms','exec_ms','d2h_ms','infer_ms','post_ms','loc_ms','xu_ly_ms')
            durations={key:median(float(r[key]) for r in frames[25:]) for key in stage_keys}
            assert all(math.isfinite(v) and v>=0 for v in durations.values())
            timing_file=folder/'timing'/(name+'.csv')
            with timing_file.open(encoding='utf-8',newline='') as f:tracker_times=list(csv.DictReader(f))
            assert len(tracker_times)==150
            steady={int(r['frame']) for r in frames[25:]}
            selected=[r for r in tracker_times if int(r['frame']) in steady];assert len(selected)==125
            for key in ('t_predict_ms','t_assoc_ms','t_update_ms'):
                durations[key]=median(float(r[key]) for r in selected)
                assert math.isfinite(durations[key]) and durations[key]>=0
            profiles.append({'dataset':ds,'name':name,'mode':mode,'median_ms':durations});files.append(timing_file)
            for path,key in ((folder/'run_meta.json','meta_sha256'),(pipeline,'pipeline_sha256'),
                  (folder/'data'/(name+'.txt'),'mot_sha256'),(folder/'data_cls'/(name+'.csv'),'cls_sha256')):
                assert sha(path)==row[key];files.append(path)
            mot=folder/'data'/(name+'.txt')
            if mot.stat().st_size:
                a=np.loadtxt(mot,delimiter=',',ndmin=2);assert a.shape[1]==10 and np.isfinite(a).all()
                assert len(np.unique(a[:,:2],axis=0))==len(a) and (a[:,2:6][:,2:]>0).all()
                assert (a[:,0]>=1).all() and (a[:,0]<=150).all()
            assert row['max_temp_c']<75
            if mode=='original':assert row['R109_MOT_CLS_byte_parity'] is True
            verified.append({'dataset':ds,'name':name,'mode':mode,**measured})
    metrics={ds:{} for ds in SPLITS}
    # Each candidate uses A=original and B=candidate with the same locked
    # scorer. No directory renames and no changes to the evaluation metric.
    for candidate in modes[1:]:
        with TemporaryDirectory(prefix='r155_score_',dir=str(logs)) as temp:
            stage=Path(temp).resolve();assert stage.is_relative_to(logs)
            for video in videos:
                ds,name=video['dataset'],video['name'];split=SPLITS[ds][0]
                gt=stage/'gt'/split/name/'gt/gt.txt';gt.parent.mkdir(parents=True)
                with gt.open('w',encoding='utf-8',newline='') as f:csv.writer(f).writerows(gtrows[ds,name])
                for tag,mode in (('A','original'),('B',candidate)):
                    source=logs/mode/'runs/full/pipeline'/ds/name/'B/data'/(name+'.txt')
                    destination=stage/'trackers'/split/tag/'data'/(name+'.txt');destination.parent.mkdir(parents=True,exist_ok=True)
                    destination.write_bytes(source.read_bytes())
            for ds,group in quality(stage,videos).items():
                metrics[ds]['original']=group['A'];metrics[ds][candidate]=group['B']
                for key,value in historical['quality'][ds]['overlap']['combined'].items():
                    assert math.isclose(group['A']['combined'][key],value,abs_tol=1e-9),('R109 control metric mismatch',ds,key)
    source_files=[Path(inspect.getfile(quality)),Path(inspect.getfile(verify_timing)),Path(inspect.getfile(evaluation.clipped_gt)),
                  Path(evaluation.rte.__file__)]
    source_files.extend(Path(inspect.getfile(evaluation.rte.dataset_class(e))) for _,e in SPLITS.values())
    files.extend(source_files)
    stage_summary={}
    for ds in ('bdd','ua','all'):
        stage_summary[ds]={}
        for mode in modes:
            selected=[r['median_ms'] for r in profiles if r['mode']==mode and (ds=='all' or r['dataset']==ds)]
            stage_summary[ds][mode]={key:median(r[key] for r in selected) for key in selected[0]}
    result={'status':'PASS','clips':80,'runs':len(rows),'verified_inference_frames':150*len(rows),'warmup':25,
        'quality':metrics,'speed':speed(rows,modes),'detection':detection,'models':models,'start':inputs['swap_activity'],'after':after,
        'stage_median_of_clip_medians_ms':stage_summary,'per_clip_stage_durations':profiles,
        'stage_note':'Decode overlaps processing; per-stage medians are not additive wall time.',
        'bootstrap_design':'Paired videos, 5000 replicates seed155; all80 preserves 40 BDD + 40 UA in every replicate.',
        'limitation':'150 JPEG/clip, display off, same-boot paired measurements; not full-video or camera/HDMI FPS.',
        'quality_FAIL_remains_FAIL':True,'sha256':{str(p):sha(p) for p in files}}
    (logs/'score_verified.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FULL80 SPEED AND QUALITY VERIFIED',len(rows),flush=True)
    print(json.dumps(result['speed'],indent=2),flush=True)

if __name__=='__main__':main()
