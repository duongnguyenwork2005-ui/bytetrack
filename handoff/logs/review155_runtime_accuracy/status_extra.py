"""Lightweight read-only progress snapshot; no imports of the measured code."""
from pathlib import Path
import json

BASE=Path('/home/jetson/nano_run')
def read(p):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return None
def snapshot():
    state=read(BASE/'r155_runtime_accuracy_20261120/extra_chain_state.json')
    output={'chain':state,'trials':{}}
    for folder in ('r162_cuda_pre_20261120','r150_pruning_20261120','r160_height352_20261120'):
        root=BASE/folder;entry={}
        for phase in ('pilot','full'):
            logs=root/'runs_absolute'/phase
            inputs=read(logs/'inputs.json');rows=read(logs/'pairs.json')
            if not inputs or not isinstance(rows,list):continue
            modes=inputs['mode_order'];points={(r['dataset'],r['name'],r['mode']):r for r in rows}
            names={(r['dataset'],r['name']) for r in rows}
            paired=[name for name in names if all((name[0],name[1],m) in points for m in modes)]
            equal={}
            for mode in modes[1:]:
                equal[mode]={kind:sum(points[ds,name,mode][kind+'_sha256']==points[ds,name,modes[0]][kind+'_sha256']
                    for ds,name in paired) for kind in ('mot','cls')}
            entry[phase]={'runs':len(rows),'total_runs':len(inputs['videos'])*len(modes),
                'paired_clips':len(paired),'total_clips':len(inputs['videos']),
                'complete':bool(read(logs/'complete.json')),'equal_to_original':equal}
        selection=read(root/'selection.json')
        if selection:entry['selection']=selection
        if entry:output['trials'][folder]=entry
    return output
if __name__=='__main__':print(json.dumps(snapshot(),ensure_ascii=False),flush=True)
