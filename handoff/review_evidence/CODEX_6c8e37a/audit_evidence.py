"""Cross-check changed-file base snapshots, embedded previous evidence and submitted log counts.
Counts from author logs are not reruns of those suites.
"""
import hashlib
import json
import re
from pathlib import Path

root=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=[]
for p in sorted((root/'base_preimages').rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts:continue
    rel=p.relative_to(root/'base_preimages')
    q=root.parent/'review_421dd53/tree'/rel
    base.append({'path':str(rel),'previous_available':q.is_file(),'same':q.is_file() and p.read_bytes()==q.read_bytes()})
old=root/'tree/handoff/review_evidence/CODEX_421dd53'
evidence=[]
for line in (old/'SHA256SUMS.txt').read_text().splitlines():
    if not line.strip():continue
    h,rel=line.split(None,1);p=old/rel.strip().lstrip('*')
    evidence.append({'path':rel.strip(),'match':p.is_file() and sha(p)==h})
logs=[]
for filename in ('03_test_tiep_tuc_huan_luyen_lan1.log','03_test_tiep_tuc_huan_luyen_lan2.log',
                 '04_nhom_c_tren_ma_cu_421dd53.log','05_test_detector_5lop.log','05_test_pilot_chan_doan.log'):
    p=root/'tree/handoff/logs/review41'/filename
    text=p.read_text(errors='replace')
    logs.append({'file':filename,'sha256':sha(p),'PASS_markers':text.count('[PASS]'),
                 'FAIL_markers':text.count('[FAIL]'),'SKIP_markers':text.count('[SKIP]'),
                 'summary':[x for x in text.splitlines() if re.search(r'KET QUA|exit=|EXIT_CODE|TONG KET',x)]})
scripts={name:json.loads((root/'evidence'/name).read_text()) for name in
         ('csv_recovery.json','journal_recovery.json','observer_base.json','observer_target.json','author_counter_rerun.json')}
result={'scope':__doc__,'base_files':base,'previous_evidence':{'checked':len(evidence),'mismatches':[x for x in evidence if not x['match']]},
        'author_logs':logs,'independent':{name:{k:val for k,val in obj.items() if k in ('pass','fail','known_risks_reproduced','source_sha256')}
                       for name,obj in scripts.items()}}
(root/'evidence/evidence_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
