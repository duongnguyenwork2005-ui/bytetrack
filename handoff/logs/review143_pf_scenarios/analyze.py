"""Scenario events from whole-sequence CLEAR; clustered HELMET uncertainty."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import run as job

HERE=Path(__file__).resolve().parent


def prepare():
    job.frozen_check()
    proof=job.read(HERE/'verification.json');assert proof['status']=='PASS' and proof['units']==765
    for p,h in proof['sha256'].items():assert job.sha(p)==h,p
    paths=[Path(__file__),HERE/'00_protocol.md',HERE/'inputs.json',HERE/'summary.json',HERE/'verification.json']
    locked={'sha256':{str(p):job.sha(p) for p in paths},'group_rule':'R99 turn25/35/straight8; native visibility<=0.5',
            'helmet_bootstrap':{'samples':5000,'seed':143,'unit':'12 scene groups; 3 clips per group'}}
    target=HERE/'analysis_inputs.json'
    if target.exists():assert job.read(target)==locked
    else:job.write(target,locked)
    return locked


def scenario_events():
    rows=[];all_means={};total_checks=0
    for name in ('roundabout','svmot'):
        c=job.cases()[name];sources={}
        for video in c['videos']:
            gtpath=job.ROOT/'data/processed'/c['split']/video/'gt/gt.txt'
            if name=='roundabout':
                ids,present,maps,episodes=job.labels_for_camera(gtpath,video)
                masks={k:v>=0 for k,v in maps.items()}
                counts={k:sum(e['level']==k for e in episodes) for k in masks}
            else:
                gt=np.loadtxt(gtpath,delimiter=',',ndmin=2);ids=np.unique(gt[:,1]).astype(int)
                n=job.read(HERE/'inputs.json')['cases'][name]['frames'][video]
                present=np.zeros((n,len(ids)),dtype=bool);low=np.zeros_like(present)
                ix=gt[:,0].astype(int)-1,np.searchsorted(ids,gt[:,1])
                present[ix]=True;low[ix]=gt[:,8]<=.5
                masks={'visibility_le_0_5':low,'visibility_gt_0_5':present & ~low}
                counts={k:None for k in masks}
            masks['all']=present
            sources[video]=(ids,present,masks,counts)
        means={}
        for mode in job.MODES:
            seed_rows=[]
            for seed in (range(3) if mode in job.MODES[3:] else (None,)):
                tracker=job.tag(mode,seed)
                _,cfg,_=job.rte.build_configs(c['split'],tracker,c['videos'])
                cfg.update(TRACKERS_FOLDER=str(HERE/name/'tracking'))
                dataset=job.trackeval.datasets.MotChallenge2DBox(cfg)
                metrics=job.boot_stats.load_raw_fields(HERE/name/'evaluation'/tracker/'raw_fields.json')
                totals={}
                for video,(ids,present,masks,counts) in sources.items():
                    raw=dataset.get_raw_seq_data(tracker,video)
                    data=dataset.get_preprocessed_seq_data(raw,'pedestrian')
                    assert present.sum()==data['num_gt_dets']
                    assert all(np.array_equal(ids[compact],original) for compact,original in zip(data['gt_ids'],raw['gt_ids']))
                    matched,changed=job.trace(data,present.shape);missed=present & ~matched
                    assert int(changed.sum())==metrics[video]['IDSW']
                    assert int(missed.sum())==metrics[video]['CLR_FN']
                    assert not (changed & ~present).any() and not (matched & ~present).any()
                    total_checks+=1
                    for group,mask in masks.items():
                        entry={'GT_boxes':int(mask.sum()),'IDSW':int((changed & mask).sum()),'FN':int((missed & mask).sum())}
                        totals.setdefault(group,Counter()).update(entry)
                        rows.append({'case':name,'mode':mode,'seed':seed,'video':video,'group':group,
                                     'episodes':counts.get(group),**entry})
                seed_rows.append({g:dict(values) for g,values in totals.items()})
            means[mode]={g:{f:float(np.mean([r[g][f] for r in seed_rows])) for f in seed_rows[0][g]}
                         for g in seed_rows[0]}
        all_means[name]=means
        if name=='roundabout':assert means['kf']['turn25']['GT_boxes']==6601
        if name=='svmot':assert means['kf']['visibility_le_0_5']['GT_boxes']==12051
        print('SCENARIO',name,{m:{g:r for g,r in v.items() if g not in ('all','straight8','visibility_gt_0_5')}
                               for m,v in means.items()},flush=True)
    assert total_checks==117
    job.write(HERE/'scenario_events.json',{'status':'PASS','whole_sequence_CLEAR_parity_checks':total_checks,
              'means':all_means,'by_video_seed':rows,'scope':'Event attribution after whole-sequence matching; no clip reset or subset HOTA.'})


def helmet_bootstrap():
    manifest=job.read(job.HELMET/'manifest_test36.json')
    groups={}
    for row in manifest['videos']:groups.setdefault(row['group'],[]).append(row['video_id'])
    assert len(groups)==12 and all(len(v)==3 for v in groups.values())
    summary=job.read(HERE/'summary.json');results={}
    for name in ('helmet_c25','helmet_c10'):
        c=job.cases()[name];names=c['videos'];indices={v:i for i,v in enumerate(names)}
        cluster=[np.asarray([indices[v] for v in videos]) for videos in groups.values()]
        raws={m:[job.boot_stats.load_raw_fields(HERE/name/'evaluation'/job.tag(m,s)/'raw_fields.json')
                   for s in (range(3) if m in job.MODES[3:] else (None,))] for m in job.MODES}
        stacks={m:[job.boot_stats._stack(raw,names) for raw in rows] for m,rows in raws.items()}
        fields=('HOTA','IDF1','IDSW','FN');refs=('kf','ekf','ukf','parent')
        samples={m:{f:[] for f in fields} for m in refs};rng=np.random.default_rng(143)
        for _ in range(5000):
            ix=np.concatenate([cluster[i] for i in rng.integers(0,12,size=12)])
            estimates={m:[job.boot_stats.combine_stacked({f:st[f][ix] for f in st}) for st in rows]
                       for m,rows in stacks.items()}
            values={m:{f:float(np.mean([r[f] for r in rows])) for f in fields} for m,rows in estimates.items()}
            for m in refs:
                for f in fields:samples[m][f].append(values['candidate'][f]-values[m][f])
        scores=summary[name]['scores']
        per_group={}
        for group,videos in groups.items():
            per_group[group]={}
            for mode,rows in raws.items():
                values=[job.combine(raw,videos) for raw in rows]
                per_group[group][mode]={f:float(np.mean([r[f] for r in values])) for f in job.FIELDS}
        results[name]={'unit':'scene_group','groups':12,'samples':5000,'seed':143,
                 'candidate_minus_reference':{m:{f:{'difference':scores['candidate']['mean'][f]-scores[m]['mean'][f],
                          'ci95':np.quantile(v,[.025,.975]).tolist()} for f,v in fieldsamples.items()}
                          for m,fieldsamples in samples.items()},'per_scene_group':per_group}
        print('BOOTSTRAP',name,flush=True)
    job.write(HERE/'helmet_bootstrap.json',results)


def finish(locked):
    job.frozen_check()
    for p,h in locked['sha256'].items():assert job.sha(p)==h,p
    paths=[Path(__file__),HERE/'analysis_inputs.json',HERE/'scenario_events.json',HERE/'helmet_bootstrap.json']
    job.write(HERE/'analysis_verification.json',{'status':'PASS','whole_sequence_CLEAR_checks':117,
                  'helmet_groups':12,'bootstrap_samples_per_condition':5000,
                  'sha256':{str(p):job.sha(p) for p in paths}})
    print('ANALYSIS PASS',flush=True)


if __name__=='__main__':
    locked=prepare();scenario_events();helmet_bootstrap();finish(locked)
