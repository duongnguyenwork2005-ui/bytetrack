"""Fit one physical-time OU turn-rate prior from the locked BDD train/dev139 labels."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import savgol_filter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SPLIT=ROOT/'configs/phase2/detector/splits/bdd5_train_v1.json'
LABELS=ROOT/'data/raw/bdd100k/labels/box_track_20/train'
CATEGORIES={'car','truck','bus','motorcycle','bicycle'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def pairs(rows):
    tracks={}
    for frame in rows:
        index=int(frame['frameIndex'])
        for label in frame.get('labels',[]):
            if label.get('category') not in CATEGORIES:continue
            attrs=label.get('attributes',{})
            if attrs.get('crowd') or attrs.get('ignore') or label.get('ignore'):continue
            box=label.get('box2d')
            if not box or box['x2']<=box['x1'] or box['y2']<=box['y1']:continue
            tracks.setdefault(str(label['id']),[]).append((index,(box['x1']+box['x2'])*.5,(box['y1']+box['y2'])*.5))
    output=[];segments=0
    for entries in tracks.values():
        a=np.asarray(sorted(entries),dtype=float)
        assert len(np.unique(a[:,0]))==len(a)
        for segment in np.split(a,np.where(np.diff(a[:,0])!=1)[0]+1):
            if len(segment)<7:continue
            velocity=savgol_filter(segment[:,1:],5,2,deriv=1,delta=.2,axis=0)[2:-2]
            acceleration=savgol_filter(segment[:,1:],5,2,deriv=2,delta=.2,axis=0)[2:-2]
            squared=np.sum(velocity**2,axis=1)
            valid=squared>=(.5/.2)**2
            omega=np.zeros(len(squared))
            np.divide(velocity[:,0]*acceleration[:,1]-velocity[:,1]*acceleration[:,0],squared,out=omega,where=valid)
            use=valid[:-1]&valid[1:]
            if use.any():
                output.extend(zip(omega[:-1][use],omega[1:][use]));segments+=1
    return output,segments


def main():
    assert not (HERE/'turn_prior.json').exists()
    split=json.loads(SPLIT.read_text());names=split['dev'];assert len(names)==139
    forbidden=json.loads((ROOT/'results/phase2/manifests/bdd100k_transfer_cocovid_val.json').read_text())['videos']
    assert not set(names)&set(forbidden)
    paths=[SPLIT,Path(__file__),HERE/'00_protocol.md']+[LABELS/f'{v}.json' for v in names]
    frozen={str(p):sha(p) for p in paths}
    (HERE/'calibration_inputs.json').write_text(json.dumps({'videos':names,'sha256':frozen},indent=2))
    rows=[];counts={}
    for name in names:
        result,segments=pairs(json.loads((LABELS/f'{name}.json').read_text()))
        rows.extend(result);counts[name]={'pairs':len(result),'segments':segments}
    a=np.asarray(rows,dtype=float);assert a.shape[1]==2 and len(a)>=1000 and np.isfinite(a).all()
    def profile(rate):
        rho=np.exp(-rate*.2);factor=-np.expm1(-2*rate*.2)
        s=float(np.mean((a[:,1]-rho*a[:,0])**2)/factor)
        return .5*len(a)*(np.log(s)+np.log(factor))
    result=minimize_scalar(profile,bounds=(.01,100.),method='bounded',options={'xatol':1e-8})
    assert result.success and .010001<result.x<99.999,('boundary or failure',result)
    rate=float(result.x);rho=np.exp(-rate*.2);s=float(np.mean((a[:,1]-rho*a[:,0])**2)/(-np.expm1(-2*rate*.2)))
    assert np.isfinite(s) and s>0
    for p,h in frozen.items():assert sha(Path(p))==h,p
    derived={}
    for fps in (5,25):
        dt=1/fps;r=float(np.exp(-rate*dt))
        derived[str(fps)]={'rho':r,'omega_initial_std_per_frame':float(np.sqrt(s)*dt),
                          'omega_process_std_per_frame':float(np.sqrt(s*(-np.expm1(-2*rate*dt)))*dt)}
    data={'lambda_per_second':rate,'stationary_variance_rad2_per_second2':s,
          'pairs':len(a),'videos':139,'counts':counts,'derived':derived,
          'objective':float(result.fun),'calibration_input_sha256':sha(HERE/'calibration_inputs.json')}
    (HERE/'turn_prior.json').write_text(json.dumps(data,indent=2))
    print('FITTED',rate,s,len(a),derived,flush=True)


if __name__=='__main__':main()
