"""Confirm already disclosed CSV limitations. Temporary fixtures only; no training.

Uses the real backup function and real het_epoch_train writer. Checkpoint marker
bytes are only copied, not loaded. This is not an Ultralytics end-to-end test.
Exit 0 means this observation script completed, NOT that CSV recovery is fixed.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

ap=argparse.ArgumentParser(__doc__)
ap.add_argument("--repo",type=Path,required=True)
ap.add_argument("--out",type=Path,required=True)
a=ap.parse_args()
src=a.repo/'src/tiep_tuc_huan_luyen.py'
s=importlib.util.spec_from_file_location('ttl_csv_review40',src)
t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
observations=[]
with tempfile.TemporaryDirectory(prefix='codex_csv40_') as td:
    root=Path(td)
    for name,tail in [('results_nul',b'\x00'*4),('results_incomplete_epoch10',b'1')]:
        r=root/name;r.mkdir()
        header=b'epoch,time,x\r\n'
        body=b''.join(f'{i},{i*10},0.5\r\n'.encode() for i in range(1,10))
        original=header+body+tail
        p=r/'results.csv';p.write_bytes(original)
        (r/'checkpoint.pt').write_bytes(b'marker-only')
        t.ghi_nhat_ky(r,'bat_dau',lan=0)
        err=None
        try:t.sao_luu_truoc_tiep_tuc(r,r/'checkpoint.pt',1,epoch_da_luu=9)
        except Exception as e:err={'type':type(e).__name__,'message':str(e),'is_LoiTiepTuc':isinstance(e,t.LoiTiepTuc)}
        rows=list(csv.DictReader(io.StringIO(p.read_text())))
        observations.append({'case':name,'error':err,'trailing_bytes_hex':p.read_bytes()[-20:].hex(),
                             'last_parsed_row':rows[-1],
                             'backup_exact':(r/'khoi_phuc/truoc_tiep_tuc_1/results.csv').read_bytes()==original,
                             'risk_observed':bool(err and err['type']=='ValueError' and not err['is_LoiTiepTuc']) if name=='results_nul'
                                 else err is None and p.read_bytes().endswith(b'\r\n1') and rows[-1].get('time') is None})
    r=root/'batch';r.mkdir()
    buf=io.StringIO(newline='');writer=csv.DictWriter(buf,fieldnames=t.COT_BATCH);writer.writeheader()
    writer.writerow({'lan':0,'epoch':1,'i':0})
    p=r/t.TEN_BATCH_CSV
    p.write_bytes(buf.getvalue().encode()+b'0,1,')
    recorder=t.GhiLuotDayDu({},r,{},'tiep_tuc',1,None,{})
    recorder._rows=[{'lan':1,'epoch':2,'i':0}]
    recorder.het_epoch_train(None)  # actual production CSV writer; trainer is unused by this method
    rows=list(csv.DictReader(io.StringIO(p.read_text())))
    observations.append({'case':'batch_partial_then_actual_writer','expected_fields':len(t.COT_BATCH),
                         'unexpected_extra_fields':rows[-1].get(None),'actual_lan_epoch_i':[rows[-1][k] for k in ('lan','epoch','i')],
                         'new_row_should_be':[1,2,0],'risk_observed':None in rows[-1] and rows[-1]['lan']=='0'})
out={'scope':__doc__,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'observations':observations,
     'known_risks_reproduced':sum(x['risk_observed'] for x in observations)}
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
