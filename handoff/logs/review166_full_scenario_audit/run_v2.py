"""Full-cohort post-hoc vehicle strata, reusing frozen tracking outputs only."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,importlib.util,inspect,json,os,sys
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
HERE=Path(__file__).resolve().parent;LOGS=HERE.parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(LOGS/'review144_shared_confidence'))
spec=importlib.util.spec_from_file_location('r166_existing_manifest',LOGS/'review144_shared_confidence/job.py')
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)
sys.path.insert(0,str(LOGS/'review99_roundabout_turns'))
from audit_turns import labels_for_camera,trace
import numpy as np,pandas as pd
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def cohorts(stage,c):
    rows=[]
    for mode in ('kf','ekf','ukf'):rows.append((mode+'_common',LOGS/'review148_common_references'/stage,mode))
    for mode in ('ekf','ukf'):rows.append((mode+'_floor',LOGS/'review154_velocity_noise'/stage,mode))
    for seed in range(3):rows.append(('pf_s%d'%seed,c['pf'],c['pf_tag']+'-s'+str(seed)))
    return rows
def bootstrap(rows):
    rng=np.random.default_rng(166);output={}
    for stage in ('bdd','test40'):
        group=[r for r in rows if r['stage']==stage];videos=sorted({r['video'] for r in group})
        draws=rng.integers(0,len(videos),size=(5000,len(videos)))
        for category,stratum in sorted({(r['category'],r['stratum']) for r in group}):
            points={(r['video'],r['method']):r for r in group if r['category']==category and r['stratum']==stratum}
            if not sum(points[v,'kf_common']['GT'] for v in videos):continue
            base=np.array([points[v,'kf_common']['IDSW'] for v in videos]);base_fn=np.array([points[v,'kf_common']['FN'] for v in videos])
            result={}
            for method in ('ekf_common','ukf_common','ekf_floor','ukf_floor','pf_mean'):
                def values(key):
                    return np.array([np.mean([points[v,'pf_s%d'%s][key] for s in range(3)]) if method=='pf_mean' else points[v,method][key] for v in videos])
                delta=values('IDSW')-base;fn=values('FN')-base_fn
                result[method]={'IDSW_delta_vs_KF':float(delta.sum()),
                    'paired_video_bootstrap_CI95':np.quantile(delta[draws].sum(axis=1),[.025,.975]).tolist(),
                    'FN_delta_vs_KF':float(fn.sum())}
            output[stage+'/'+category+'/'+stratum]=result
    return output
def main():
    target=HERE/'inputs_v2.json';assert not target.exists();files=[Path(__file__),HERE/'00_scope.md',HERE/'01_raw_schema.md',
        LOGS/'review99_roundabout_turns/audit_turns.py',LOGS/'review113_vehicle_scenarios/ua_occlusion.py',
        LOGS/'review113_vehicle_scenarios/bdd_turns.py',Path(j.rte.__file__),
        Path(inspect.getfile(j.trackeval.metrics.CLEAR)),ROOT/'data/processed/BDDCOCOVID-val/vehicle_types.csv']
    cases={s:j.case(s) for s in ('bdd','test40')}
    for stage,c in cases.items():
        assert len(c['videos'])==(200 if stage=='bdd' else 40)
        for _,folder,tag in cohorts(stage,c):
            files.append(folder/'evaluation'/tag/'raw_fields.json')
            files.extend(folder/'tracking'/c['split']/tag/'data'/(v+'.txt') for v in c['videos'])
        files.extend(ROOT/'data/processed'/c['split']/v/'gt/gt.txt' for v in c['videos'])
        for review in ('review148_common_references','review154_velocity_noise'):
            folder=LOGS/review/stage
            assert (folder/'summary.json').is_file() and (folder/'verification.json').is_file()
            verification=read(folder/'verification.json')
            for p,h in verification['sha256'].items():assert sha(p)==h,p
            files.extend((folder/'summary.json',folder/'verification.json',folder/'inputs.json'))
    lock={str(p):sha(p) for p in dict.fromkeys(files)}
    write(target,{'sha256':lock,'new_tracking':False,'new_metric':False,
        'cases':{s:{'videos':c['videos'],'frames':c['frames']} for s,c in cases.items()}})
    types=pd.read_csv(ROOT/'data/processed/BDDCOCOVID-val/vehicle_types.csv')
    results=[];totals=defaultdict(Counter);episodes=Counter();verified=0
    for stage,c in cases.items():
        definitions=cohorts(stage,c);datasets={};raws={}
        for name,folder,tag in definitions:
            _,config,_=j.rte.build_configs(c['split'],tag,c['videos'])
            config.update(PRINT_CONFIG=False,TRACKERS_FOLDER=str(folder/'tracking'),SEQ_INFO=c['frames'])
            datasets[name]=j.rte.dataset_class(c['evaluator'])(config);raws[name]=read(folder/'evaluation'/tag/'raw_fields.json')
        for video in c['videos']:
            gtpath=ROOT/'data/processed'/c['split']/video/'gt/gt.txt';gt=np.loadtxt(gtpath,delimiter=',',ndmin=2)
            eligible=gt[(gt[:,6]==1)&(gt[:,7]==1)]
            clean=HERE/'clean_gt_v2'/stage/(video+'.txt');clean.parent.mkdir(parents=True,exist_ok=True)
            np.savetxt(clean,eligible,delimiter=',',fmt='%.10g')
            ids,present,levels,events=labels_for_camera(clean,video,c['frames'][video],c['fps'])
            masks={'all':present,'turn25':levels['turn25']>=0,'straight8':levels['straight8']>=0}
            if stage=='test40':
                assert np.isfinite(eligible[:,8]).all() and ((eligible[:,8]>=0)&(eligible[:,8]<=1)).all()
                low=np.zeros_like(present);low[eligible[:,0].astype(int)-1,np.searchsorted(ids,eligible[:,1])]=eligible[:,8]<=.5
                recent=low.copy()
                for lag in range(1,c['fps']+1):recent[lag:]|=low[:-lag]
                masks.update(low=low,recent=recent&present,recovery=recent&present&~low,turn25_low=masks['turn25']&low)
            if stage=='bdd':
                mapping=types[types.video==video].set_index('track_id').category.to_dict();classes=np.array([mapping[k] for k in ids])
                categories={'vehicle':np.ones(len(ids),bool),'car':classes=='car','motorcycle':classes=='motorcycle'}
            else:categories={'vehicle':np.ones(len(ids),bool)}
            for event in events:episodes[stage+'/'+event['level']]+=1
            for name,folder,tag in definitions:
                raw=datasets[name].get_raw_seq_data(tag,video);data=datasets[name].get_preprocessed_seq_data(raw,'pedestrian')
                assert data['num_timesteps']==c['frames'][video] and data['num_gt_ids']==len(ids)
                for tick,idx in enumerate(data['gt_ids']):assert np.array_equal(ids[idx],eligible[eligible[:,0]==tick+1,1])
                matched,changed=trace(data,present.shape);miss=present&~matched;expected=raws[name][video]
                assert changed.sum()==expected['IDSW'] and miss.sum()==expected['CLR_FN'],(stage,video,name,int(changed.sum()),expected['IDSW'])
                verified+=1
                for category,selected in categories.items():
                    for stratum,mask in masks.items():
                        mask=mask&selected[None,:]
                        row={'stage':stage,'video':video,'method':name,'category':category,'stratum':stratum,
                            'GT':int(mask.sum()),'IDSW':int((mask&changed).sum()),'FN':int((mask&miss).sum())}
                        results.append(row);totals[stage+'/'+category+'/'+stratum+'/'+name].update({k:row[k] for k in ('GT','IDSW','FN')})
            write(HERE/'partial_v2.json',{'verified_outputs':verified,'aggregate':{k:dict(v) for k,v in totals.items()},'per_video':results})
            print('R166 EXISTING OUTPUTS TRACED',stage,video,verified,flush=True)
    for p,h in lock.items():assert sha(p)==h,p
    write(HERE/'summary_v2.json',{'status':'PASS','verified_outputs':verified,'new_tracking':False,'new_metric':False,
        'aggregate':{k:dict(v) for k,v in totals.items()},'paired_deltas':bootstrap(results),'episodes':dict(episodes),
        'source_contract_sha256':sha(target),'partial_sha256':sha(HERE/'partial_v2.json'),
        'limitations':['Post-hoc, multiple comparisons, reused tuning data; no blind generalization claim.',
            'Image-plane turns, not proven world-space path curvature.','Native occlusion flag only for UA.',
            'Global Gaussian prerequisite remains governed by original full results, not selected subgroups.']})
    print('R166 FULL SCENARIO DIAGNOSIS COMPLETE',verified,flush=True)
if __name__=='__main__':main()
