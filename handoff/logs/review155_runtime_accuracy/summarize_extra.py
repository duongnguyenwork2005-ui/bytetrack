"""Summarize verified extra trials; retain pilot/full and quality distinctions."""
from pathlib import Path
import hashlib,json

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    target=HERE/'extra_final_summary.json';assert not target.exists()
    results={};sources=[Path(__file__)];qualified=[]
    for folder in ('review162_cuda_pre','review150_nano_pruning','review160_height352'):
        root=HERE.parent/folder;full=root/'full80_evidence/runs_absolute/full'
        verified=full/'score_verified.json'
        if not verified.exists():
            pilot=root/'negative_pilot_evidence/score_pilot_verified.json'
            record=read(pilot);assert record['status']=='PASS' and not record['full80_measured']
            for p,h in record['source_sha256'].items():assert sha(Path(p))==h,p
            results[folder]={'scope':'negative fixed pilot4','pilot':record,'full80_measured':False}
            sources.append(pilot);continue
        record=read(verified);assert record['status']=='PASS' and record['clips']==80
        for p,h in record['sha256'].items():assert sha(Path(p))==h,p
        rows=read(full/'pairs.json');points={(r['dataset'],r['name'],r['mode']):r for r in rows}
        original=record['models']['original'];candidates={}
        for mode,model in record['models'].items():
            if mode=='original':continue
            byte_equal={}
            effects={}
            for ds in ('bdd','ua'):
                names=sorted({r['name'] for r in rows if r['dataset']==ds});assert len(names)==40
                byte_equal[ds]={kind:sum(points[ds,name,mode][kind+'_sha256']==points[ds,name,'original'][kind+'_sha256']
                    for name in names) for kind in ('mot','cls')}
                baseline=record['quality'][ds]['original']['combined'];current=record['quality'][ds][mode]['combined']
                effects[ds]={key:{'original':baseline[key],'candidate':current[key],
                    'difference':current[key]-baseline[key]} for key in ('hota','idf1','idsw','fn','fp')}
            exact=all(count==40 for group in byte_equal.values() for count in group.values())
            eligible=bool(model['quality_recommended'] and model['historical_raw_parity']=='PASS' and exact)
            speed=record['speed']['all']['paired_vs_original'][mode]
            candidates[mode]={'speed':{ds:{'original':record['speed'][ds]['modes']['original'],
                    'candidate':record['speed'][ds]['modes'][mode],
                    'paired':record['speed'][ds]['paired_vs_original'][mode]} for ds in ('bdd','ua','all')},
                'detection':record['detection'][mode],'tracking_changes':effects,'byte_equal_clips':byte_equal,
                'all80_tracking_byte_exact':exact,'AP_recommended':model['quality_recommended'],
                'historical_raw_parity':model['historical_raw_parity'],
                'same_detection_and_tracking_quality':eligible,
                'stage_median_of_clip_medians_ms':{ds:{'original':record['stage_median_of_clip_medians_ms'][ds]['original'],
                    'candidate':record['stage_median_of_clip_medians_ms'][ds][mode]} for ds in ('bdd','ua','all')}}
            if eligible and all(record['speed'][ds]['paired_vs_original'][mode]['mean_FPS_gain_pct']>0 for ds in ('bdd','ua')):
                qualified.append({'trial':folder,'mode':mode,'paired_gain_pct':speed['mean_FPS_gain_pct']})
        results[folder]={'scope':'paired full80 windows,40BDD+40UA,150JPEG,25warmup,displayoff',
            'full80_measured':True,'original_detection':record['detection']['original'],
            'candidates':candidates,'boot':record['after']['boot'],'engine_sha256':original['engine_sha256']}
        sources.append(verified)
    output={'status':'PASS','trials':results,'quality_preserving_candidates':qualified,
        'default_or_launcher_changed':False,
        'comparison_note':'Use each trial fresh paired original. Do not compare absolute FPS across trials as a paired experiment. Decode overlaps processing; stage medians are not additive.',
        'selection_note':'Quality-preserving label requires actual AP gate, raw parity PASS and exact MOT/CLS on all80; this report does not change any preregistered test rule.',
        'source_sha256':{str(p):sha(p) for p in sources}}
    target.write_text(json.dumps(output,indent=2)+'\n')
    print('EXTRA VERIFIED SUMMARY COMPLETE',json.dumps(qualified),flush=True)
if __name__=='__main__':main()
