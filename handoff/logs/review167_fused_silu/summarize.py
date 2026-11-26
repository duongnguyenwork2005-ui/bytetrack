"""Recheck final evidence and compare speed, detection and tracking separately."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent;BUNDLE=HERE/'full80_evidence';FULL=BUNDLE/'runs_absolute/full'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def main():
    target=HERE/'final_summary.json';assert not target.exists()
    score=read(FULL/'score_verified.json');assert score['status']=='PASS' and score['clips']==80 and score['runs']==320
    for p,h in score['sha256'].items():assert sha(p)==h,p
    before=read(BUNDLE/'environment.json');after=read(BUNDLE/'environment_after.json');inputs=read(FULL/'inputs.json')
    assert before['boot']==after['boot']==inputs['boot'] and before['SD_errors']==after['SD_errors']=='0'
    for device,keys in (('CPU',('scaling_governor','scaling_min_freq','scaling_max_freq')),('GPU',('governor','min_freq','max_freq'))):
        for key in keys:assert before[device][key]==after[device][key],(device,key)
    compare={};det=score['detection']
    for mode in ('precise','fast'):
        detection={};gates={}
        for reference in ('original','control'):
            a,b=det[reference],det[mode]
            detection[reference]={'mAP50_delta_pp':100*(b['mAP50']-a['mAP50']),
                'mAP50_95_delta_pp':100*(b['mAP50_95']-a['mAP50_95']),
                'class_AP50_delta_pp':{c:100*(b['classes'][c]['AP50']-a['classes'][c]['AP50']) for c in a['classes']}}
            gates[reference]={'AP50':b['mAP50']>=a['mAP50']-.005,'AP50_95':b['mAP50_95']>=a['mAP50_95']-.005,
                'class_AP50':all(b['classes'][c]['AP50']>=a['classes'][c]['AP50']-.02 for c in a['classes'])}
        tracking={}
        for dataset in ('bdd','ua'):
            a=score['quality'][dataset]['original']['combined'];b=score['quality'][dataset][mode]['combined']
            tracking[dataset]={'original':a,'candidate':b,'IDSW_delta':b['idsw']-a['idsw'],
                'FN_delta':b['fn']-a['fn'],'FP_delta':b['fp']-a['fp'],'HOTA_delta_pp':100*(b['hota']-a['hota'])}
        raw=read(HERE/'screen_evidence'/(mode+'_raw_parity.json'))
        compare[mode]={'AP_gates':gates,'AP_deltas':detection,'tracking':tracking,
            'raw_parity_status':raw['status'],'raw_max_box_error':max(c['box_error'] for c in raw['checks']),
            'raw_max_score_error':max(c['score_error'] for c in raw['checks']),
            'speed':{dataset:score['speed'][dataset]['paired_vs_original'][mode] for dataset in ('bdd','ua','all')}}
    bound=[Path(__file__),FULL/'score_verified.json',BUNDLE/'environment.json',BUNDLE/'environment_after.json',
        HERE/'speed_preflight.json',HERE/'timing_bound_preflight.json',HERE/'screen_verification.json']
    target.write_text(json.dumps({'status':'PASS','meaning':'Final provenance/comparison verified; raw FAIL remains FAIL.',
        'clips':80,'runs':score['runs'],'inference_frames':score['verified_inference_frames'],'comparisons':compare,
        'speed':score['speed'],'detection':det,'quality':score['quality'],
        'environment':{'before':before,'after':after},'launcher_or_defaults_changed':False,
        'limitations':['150 JPEG per source video, display off, KF, warm25; not full-video/camera/HDMI FPS.',
            'AP uses repeatedly seen mixed5 dev7190; motorcycle226 and bicycle407 GT boxes.',
            'Same mathematical activation and weights, but different FP16 arithmetic/tactics; no bit-exact claim.'],
        'sha256':{str(p):sha(p) for p in bound}},indent=2)+'\n')
    print('R167 FINAL COMPARISON VERIFIED',flush=True)
    for mode,row in compare.items():print(mode,row['speed']['all'],row['tracking'],flush=True)
if __name__=='__main__':main()
