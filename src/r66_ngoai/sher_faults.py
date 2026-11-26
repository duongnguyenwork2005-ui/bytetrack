"""Inject faults into isolated copies of a sealed smoke unit. Existing outputs stay intact."""
from pathlib import Path
import json
import shutil
import tempfile
from unittest.mock import patch

import numpy as np
import sher_matrix as m

def check_fails(fn, reason=None):
    try:
        fn()
    except (ValueError, KeyError, OSError, m.base.LoiHopDong) as exc:
        if reason and reason not in str(exc):
            raise
        return
    raise AssertionError("damaged unit passed")

def main():
    ds, d, c, l, seed = m.DS, "yolov8n_mixed5", "c25", "pf500", 0
    t=m.base.cau_hinh(ds,True)
    name=m.name_for(True,d,c,l,seed)
    tracker,result=m.paths(t,name)
    cid=m.base.ten_cache(ds,d,True)
    passed=[]
    with tempfile.TemporaryDirectory(dir=m.base.GOC/"tmp") as tmp:
        root=Path(tmp)
        tr,rs=root/"tracker",root/"result"
        shutil.copytree(tracker,tr)
        shutil.copytree(result,rs)
        verify=lambda: m.unit(t,True,0,d,c,l,seed)
        with patch.object(m,"paths",return_value=(tr,rs)):
            verify()
            for path in (tr/"data/sherbrooke.txt",tr/"run_meta.json",rs/"raw_fields.json",rs/"reviewed.json"):
                before=path.read_bytes()
                path.unlink()
                check_fails(verify, "orphan" if path.name=="reviewed.json" else path.name)
                path.write_bytes(before)
                passed.append("missing "+path.name)
            backup=tr.with_suffix(".retained")
            tr.rename(backup)
            check_fails(verify)
            backup.rename(tr)
            passed.append("orphan results")
            original=(tr/"run_meta.json").read_bytes()
            rm=m.read(tr/"run_meta.json")
            rm["seed"]=1
            m.write(tr/"run_meta.json",rm)
            check_fails(lambda:m.inspect(t,d,c,l,seed,name,cid),"run config/backend/fingerprint")
            (tr/"run_meta.json").write_bytes(original)
            passed.append("wrong seed even without seal")
            verify()
            # Simulate an interrupted sibling seed; the sealed seed still validates.
            seed_dir=root/"seed1_interrupted"
            seed_dir.mkdir()
            with patch.object(m,"paths",return_value=(seed_dir,root/"score_missing")):
                check_fails(lambda:m.unit(t,True,0,d,c,l,1),"incomplete/orphan")
            verify()
            passed.append("interrupted sibling seed")
        cache_root=root/"cache"
        shutil.copytree(m.base.det_cache.CACHE_ROOT/cid, cache_root/cid)
        cache=cache_root/cid
        meta=cache/"sherbrooke.xong.json"
        original=meta.read_bytes()
        with patch.object(m.base.det_cache,"CACHE_ROOT",cache_root):
            verify=lambda:m.base.kiem_don_vi_cache(ds,d,m.V,True)
            verify()
            marker=m.read(meta); marker["sha256"].pop("meta.json")
            m.write(meta,marker); check_fails(verify); meta.write_bytes(original)
            passed.append("incomplete artifact hash map")
            jp=cache/"sherbrooke.json"
            source=jp.read_bytes()
            for key in ("gt_luc_sinh_sha256","split","nguon","names","anh_xa_coco"):
                obj=m.read(jp); obj[key]={} if isinstance(obj[key],dict) else "bad"
                m.write(jp,obj); mk=m.read(meta); mk["sha256"][jp.name]=m.sha(jp); m.write(meta,mk)
                check_fails(verify); jp.write_bytes(source); meta.write_bytes(original)
                passed.append("wrong "+key)
            zp=cache/"sherbrooke.npz"
            before=zp.read_bytes()
            for kind in ("NaN","cap3000"):
                with np.load(zp) as f: ar={k:f[k].copy() for k in f}
                if kind=="NaN": ar["data"][0,0]=np.nan
                else: ar["n_per_frame"][0]=3000
                np.savez(zp,**ar); mk=m.read(meta); mk["sha256"][zp.name]=m.sha(zp); m.write(meta,mk)
                check_fails(verify); zp.write_bytes(before); meta.write_bytes(original)
                passed.append(kind)
            verify()
    mask=np.ones((2,2),np.uint8)
    assert m.roilines(b"",mask)==b""
    assert m.roilines(b"1,1,-10,0,2,2,1,-1,-1,-1\n",mask)==b""
    passed += ["empty output", "outside ROI"]
    bad=np.array([[1,1,5,6,10,10,1,1,1]]*2,dtype=float)
    check_fails(lambda:m.base.r64.kiem_gt(bad,128,"fixture"),"trung")
    passed.append("duplicate frame/id")
    lock=m.read(m.LOCK)
    # In-memory counterfactual at the same pre/post identity check as the worker.
    import copy
    drift=copy.deepcopy(lock)
    drift["splits"][t["split"]]["ini"]="0"*64
    with patch.object(m,"source_lock",return_value=drift):
        check_fails(lambda:m.run("smoke",0),"differs from locked manifest")
    passed.append("seqinfo changed")
    m.write(m.ROOT/"smoke/fault_checks.json",{"passed":passed,"n":len(passed)})
    print(passed)

if __name__=="__main__":
    main()

