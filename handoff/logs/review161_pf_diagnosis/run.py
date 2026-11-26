"""Decompose exact CLEAR ID switches in already-completed dev outputs."""
from pathlib import Path
from collections import Counter
import ast
import hashlib
import importlib.util
import inspect
import json
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE=Path(__file__).resolve().parent;LOGS=HERE.parent
sys.path.insert(0,str(LOGS/'review144_shared_confidence'))
spec=importlib.util.spec_from_file_location('r161_shared_inputs',LOGS/'review144_shared_confidence/job.py')
j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    source=LOGS/'review96_pf_nano/idsw_events.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='trace')
    module=ast.Module(body=[function],type_ignores=[]);namespace={'np':np,'linear_sum_assignment':linear_sum_assignment}
    exec(compile(ast.fix_missing_locations(module),str(source),'exec'),namespace)
    trace=namespace['trace'];case=j.case('dev');videos=case['videos'];cohorts=[]
    for review,mode in [('review144_shared_confidence','kf'),('review144_shared_confidence','ekf'),
            ('review144_shared_confidence','ukf'),('review154_velocity_noise','ekf'),('review154_velocity_noise','ukf')]:
        cohorts.append((review+'_'+mode,LOGS/review/'dev',mode,review+'_'+mode,None))
    for review in ('review142_pf_confidence','review153_guided_turn','review156_guided_shape',
                   'review157_pf_init_transport','review159_guided_auxiliary'):
        for seed in range(3):
            tag=('cartesian' if review=='review142_pf_confidence' else 'pf')+'-s'+str(seed)
            cohorts.append((review+'_s'+str(seed),LOGS/review/'dev',tag,review,seed))
    files=[Path(__file__),HERE/'00_scope.md',source,Path(inspect.getfile(j.trackeval.metrics.CLEAR)),
           Path(j.rte.__file__),Path(inspect.getfile(j.rte.dataset_class('mot')))]
    for _,folder,tag,_,_ in cohorts:
        files.append(folder/'evaluation'/tag/'raw_fields.json')
        files.extend(folder/'tracking'/case['split']/tag/'data'/(v+'.txt') for v in videos)
    for v in videos:files.append(j.ROOT/'data/processed'/case['split']/v/'gt/gt.txt')
    lock={str(p):sha(p) for p in files};target=HERE/'inputs.json'
    assert not target.exists();j.write(target,{'sha256':lock,'videos':videos,'new_tracking':False})
    results={};runs=0
    for name,folder,tag,family,seed in cohorts:
        raw=j.read(folder/'evaluation'/tag/'raw_fields.json');assert set(raw)==set(videos)
        _,config,_=j.rte.build_configs(case['split'],tag,videos)
        config.update(TRACKERS_FOLDER=str(folder/'tracking'))
        dataset=j.rte.dataset_class('mot')(config);rows=[]
        for video in videos:
            data=dataset.get_preprocessed_seq_data(dataset.get_raw_seq_data(tag,video),'pedestrian')
            count,events=trace(data);assert count==raw[video]['IDSW'],(name,video,count,raw[video]['IDSW'])
            groups=Counter('after_gap' if e['at_reappearance'] else 'continuous' for e in events)
            gaps=Counter()
            for e in events:
                gap=e['unmatched_frames_since_last_match']
                if gap:gaps['gap1' if gap==1 else 'gap2_3' if gap<=3 else 'gap4_8' if gap<=8 else 'gap9_30' if gap<=30 else 'gap_over30']+=1
            row={'video':video,'IDSW':count,'after_gap':groups['after_gap'],'continuous':groups['continuous'],
                 'within_3_after_gap':sum(e['within_3_after_reappearance'] for e in events),
                 **{k:gaps[k] for k in ('gap1','gap2_3','gap4_8','gap9_30','gap_over30')},'events':events}
            assert row['IDSW']==row['after_gap']+row['continuous']
            assert row['after_gap']==sum(gaps.values()) and row['within_3_after_gap']<=row['continuous']
            rows.append(row);runs+=1
        keys=('IDSW','after_gap','continuous','within_3_after_gap','gap1','gap2_3','gap4_8','gap9_30','gap_over30')
        result={'family':family,'seed':seed,'summary':{k:sum(r[k] for r in rows) for k in keys},'videos':rows}
        results[name]=result;j.write(HERE/'partial.json',results)
        print('IDSW DECOMPOSED',name,result['summary'],flush=True)
    families={}
    for family in sorted({r['family'] for r in results.values()}):
        group=[r['summary'] for r in results.values() if r['family']==family]
        families[family]={k:float(np.mean([r[k] for r in group])) for k in group[0]}
    for p,h in lock.items():assert sha(p)==h,p
    j.write(HERE/'summary.json',{'status':'PASS','verified_video_outputs':runs,'families':families,
        'new_tracking':False,'new_metric':False,'limitation':'GT matching gap, not physical occlusion, internal Lost state or causal attribution.',
        'sha256':{str(target):sha(target),str(HERE/'partial.json'):sha(HERE/'partial.json')}})
    print('IDSW DIAGNOSIS COMPLETE',runs,flush=True)

if __name__=='__main__':main()
