"""Transfer only if N100 preserves the predeclared dev quality gates."""
from collections import Counter
from pathlib import Path
import json
import sys
import numpy as np
import run_job as wrapper

job=wrapper.job
HERE=wrapper.HERE
PARENT=wrapper.PARENT
VIDEOS=[f"seq_{i}_{'morning' if i<26 else 'evening'}" for i in range(22,31)]


def preflight():
    assert json.loads((HERE/'dev_summary.json').read_text())['pass']
    assert not (HERE/'transfer_inputs.json').exists()
    prior=json.loads((HERE/'inputs.json').read_text())
    paths=[Path(p) for p in prior['sha256']]+[Path(__file__),HERE/'dev_summary.json']
    frames={}
    for video in VIDEOS:
        cache=job.base.det_cache.open_cache('r64_svmot_yolov8n_coco_c0.05',video,.25,[1,2,3,5,7],300)
        frames[video]=len(cache)
        paths += [job.ROOT/'data/interim/det_cache/r64_svmot_yolov8n_coco_c0.05'/f'{video}{suffix}' for suffix in ('.npz','.json')]
        paths += [job.ROOT/'data/processed/SVMOTSYS-test'/video/p for p in ('gt/gt.txt','seqinfo.ini')]
    paths += [PARENT/f'transfer/evaluation/candidate-s{s}/raw_fields.json' for s in range(3)]
    job.write(HERE/'transfer_inputs.json',{'videos':VIDEOS,'frames':frames,'sha256':{str(p):job.base.sha(p) for p in paths}})


def replay(seed):
    frozen=json.loads((HERE/'transfer_inputs.json').read_text())
    for video in VIDEOS:
        folder=HERE/f'transfer/tracking/SVMOTSYS-test/cartesian-s{seed}'
        marker=folder/f'meta/{video}.json'
        if marker.is_file():continue
        mot,timing=job.replay(video,seed,True,cache_id='r64_svmot_yolov8n_coco_c0.05',fps=30,classes=[1,2,3,5,7],buffer=36)
        assert len(timing)==frozen['frames'][video]
        output=folder/f'data/{video}.txt';output.parent.mkdir(parents=True,exist_ok=True)
        mot.to_csv(output,header=False,index=False,float_format='%.10f')
        target=folder/f'timing/{video}.csv';target.parent.mkdir(exist_ok=True)
        timing.to_csv(target,index=False)
        job.write(marker,{'frames':len(timing),'mot_sha256':job.base.sha(output)})
        print('TRANSFER',seed,video,len(timing),flush=True)
    job.score(seed,'SVMOTSYS-test',VIDEOS,stage='transfer')


def summary():
    frozen=json.loads((HERE/'transfer_inputs.json').read_text())
    for p,h in frozen['sha256'].items():assert job.base.sha(Path(p))==h,p
    rows={};seeds={};units=0
    for label,root,tag in [('N500',PARENT,'candidate'),('N100',HERE,'cartesian')]:
        values=[]
        for s in range(3):
            raw=job.base.boot_stats.load_raw_fields(root/f'transfer/evaluation/{tag}-s{s}/raw_fields.json')
            assert set(raw)==set(VIDEOS)
            combined=job.base.boot_stats.combine(raw,VIDEOS);values.append(combined)
            recorded=json.loads((root/f'transfer/evaluation/{tag}-s{s}/combined.json').read_text())
            assert all(abs(recorded[k]-combined[k])<1e-9 for k in job.base.METRICS)
            if label=='N100':
                for video in VIDEOS:
                    folder=HERE/f'transfer/tracking/SVMOTSYS-test/cartesian-s{s}'
                    marker=json.loads((folder/f'meta/{video}.json').read_text())
                    assert marker['frames']==frozen['frames'][video] and marker['mot_sha256']==job.base.sha(folder/f'data/{video}.txt')
                    units+=1
        seeds[label]=values
        rows[label]={k:float(np.mean([v[k] for v in values])) for k in job.base.METRICS}
    a,b=rows['N500'],rows['N100']
    conditions={'IDSW':b['IDSW']<=1.02*a['IDSW'],'HOTA':b['HOTA']>=a['HOTA']-.0025,'FN':b['FN']<=1.005*a['FN']}
    job.write(HERE/'transfer_summary.json',{'mean':rows,'seeds':seeds,'conditions':conditions,'pass':all(conditions.values()),'units':units})
    print('SUMMARY',rows,conditions,flush=True)


def occlusion():
    sys.path.insert(0,str(HERE.parent/'review99_roundabout_turns'))
    from audit_turns import trace
    totals={}
    for seed in range(3):
        tracker=f'cartesian-s{seed}'
        _,cfg,_=job.base.rte.build_configs('SVMOTSYS-test',tracker,None)
        cfg.update(PRINT_CONFIG=False,TRACKERS_FOLDER=str(HERE/'transfer/tracking'),SEQ_INFO={v:None for v in VIDEOS})
        dataset=job.trackeval.datasets.MotChallenge2DBox(cfg)
        metrics=job.base.boot_stats.load_raw_fields(HERE/f'transfer/evaluation/{tracker}/raw_fields.json')
        total=Counter()
        for video in VIDEOS:
            gt=np.loadtxt(job.ROOT/f'data/processed/SVMOTSYS-test/{video}/gt/gt.txt',delimiter=',')
            ids=np.unique(gt[:,1]).astype(int)
            raw=dataset.get_raw_seq_data(tracker,video);data=dataset.get_preprocessed_seq_data(raw,'pedestrian')
            shape=(data['num_timesteps'],len(ids));present=np.zeros(shape,bool);low=np.zeros_like(present)
            index=gt[:,0].astype(int)-1,np.searchsorted(ids,gt[:,1]);present[index]=True;low[index]=gt[:,8]<=.5
            assert present.sum()==data['num_gt_dets']
            assert all(np.array_equal(ids[g],h) for g,h in zip(data['gt_ids'],raw['gt_ids']))
            matched,changed=trace(data,shape);missed=present&~matched
            assert changed.sum()==metrics[video]['IDSW'] and missed.sum()==metrics[video]['CLR_FN']
            total.update({'boxes':int(low.sum()),'IDSW':int((low&changed).sum()),'FN':int((low&missed).sum())})
        totals[str(seed)]=dict(total)
    old=json.loads((PARENT/'occlusion_summary.json').read_text())['means']['candidate']
    means={k:float(np.mean([v[k] for v in totals.values()])) for k in ('boxes','IDSW','FN')}
    job.write(HERE/'occlusion_summary.json',{'N500':old,'N100':means,'seeds':totals,'CLEAR_parity':27})
    print('OCCLUSION',old,means,flush=True)


if __name__=='__main__':
    if sys.argv[1].isdigit():replay(int(sys.argv[1]))
    else:{'preflight':preflight,'summary':summary,'occlusion':occlusion}[sys.argv[1]]()
