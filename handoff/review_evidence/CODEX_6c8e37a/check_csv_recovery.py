"""Independent round-41 review: actual CSV planning, backup, recovery and batch writer.

No torch, Ultralytics, runner CLI or training. Checkpoints contain marker bytes and
are copied only. The results append below is an explicit plain CSV fixture, not
Ultralytics. Batch appends use the project's actual GhiLuotDayDu.het_epoch_train.
os.replace failure is injected, not a physical power failure.
"""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import tempfile
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--repo', type=Path, required=True)
ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args()
src = a.repo / 'src/tiep_tuc_huan_luyen.py'
s = importlib.util.spec_from_file_location('review_csv', src)
t = importlib.util.module_from_spec(s)
s.loader.exec_module(t)
checks = []

def check(name, condition):
    checks.append({'name': name, 'pass': bool(condition)})

def sha(b):
    return hashlib.sha256(b).hexdigest()

def snapshot(root):
    return {str(p.relative_to(root)): sha(p.read_bytes()) for p in root.rglob('*') if p.is_file()}

def row(lan, ep, i, optional_empty=False):
    d = {k: '' for k in t.COT_BATCH}
    d.update(lan=lan, epoch=ep, i=i, ni=10*(ep-1)+i, t=1790000000.5,
             loss_huu_han=True, accumulate=8, ema_updates=3, batch_size=8)
    if not optional_empty:
        for k in t.SO_TUY_CHON_BATCH: d[k] = 0.25
        d['thu_tu_batch'] = 'fedcba9876543210'
    return d

def batch_bytes(rows, nl):
    f = io.StringIO(newline='')
    w = csv.DictWriter(f, fieldnames=t.COT_BATCH, lineterminator=nl)
    w.writeheader(); w.writerows(rows)
    return f.getvalue().encode()

def parse(b):
    return list(csv.DictReader(io.StringIO(b.decode(), newline='')))

def fixture(root, name, results, batch):
    r = root / name
    r.mkdir()
    (r / 'checkpoint.pt').write_bytes(b'marker, not a model')
    t.ghi_nhat_ky(r, 'bat_dau', lan=0)
    (r / 'results.csv').write_bytes(results)
    (r / t.TEN_BATCH_CSV).write_bytes(batch)
    return r

with tempfile.TemporaryDirectory(prefix='codex_r41_') as td:
    root = Path(td)
    for nl_name, nl in [('LF', '\n'), ('CRLF', '\r\n')]:
        rb = nl.join(['epoch,time,x', '1,10,0.5', '2,20,0.4', '']).encode()
        bb = batch_bytes([row(0,1,0), row(1,2,0,True), row(1,3,0)], nl)
        # The batch epoch 3 must survive although the selected checkpoint is epoch 2.
        variants = [
            ('clean', rb, bb, 'khong', 'khong'),
            ('nul_partial', rb+b'\0'*4, bb+b'1,3,', 'cat_duoi', 'cat_duoi'),
            ('epoch_prefix_missing_nl', rb+b'1', bb[:-len(nl)], 'cat_duoi', 'them_newline'),
            ('missing_nl_bad_hex', rb[:-len(nl)], bb+bb.split(nl.encode())[1][:-5], 'them_newline', 'cat_duoi'),
        ]
        if nl_name == 'CRLF':
            variants.append(('half_crlf', rb[:-1], bb[:-1], 'them_newline', 'them_newline'))
        for name, rraw, braw, ra, ba in variants:
            name = nl_name+'_'+name
            r = fixture(root,name,rraw,braw)
            before = snapshot(r)
            plans = {f:{'co_file':True, **t.phan_tich_duoi_csv(raw,k,f,2 if k=='results' else None)}
                     for f,k,raw in [('results.csv','results',rraw),(t.TEN_BATCH_CSV,'batch',braw)]}
            check(name+': plan read-only',snapshot(r)==before)
            saved=t.sao_luu_truoc_tiep_tuc(r,r/'checkpoint.pt',1,epoch_da_luu=2,csv_duoi=plans)
            backup=Path(saved['thu_muc_luu_cu'])
            check(name+': exact recovered bytes and batch history',
                  (r/'results.csv').read_bytes()==rb and (r/t.TEN_BATCH_CSV).read_bytes()==bb)
            check(name+': byte backups and checkpoint copy',
                  (backup/'results.csv').read_bytes()==rraw and (backup/t.TEN_BATCH_CSV).read_bytes()==braw
                  and Path(saved['nguon']).read_bytes()==b'marker, not a model')
            check(name+': receipts',all(saved['csv'][f]['hanh_dong']==action
                  and saved['csv'][f]['sau_sha256']==sha(expected)
                  and saved['csv'][f]==json.loads((backup/f'xu_ly_{f}.json').read_text())
                  for f,action,expected in [('results.csv',ra,rb),(t.TEN_BATCH_CSV,ba,bb)]))
            # Actual project batch writer; explicit fixture append for results.
            writer=t.GhiLuotDayDu({},r,{},'tiep_tuc',2,None,{})
            writer._rows=[row(2,3,0)]
            writer.het_epoch_train(None)
            with (r/'results.csv').open('ab') as f:f.write(('3,30,0.3'+nl).encode())
            br=parse((r/t.TEN_BATCH_CSV).read_bytes()); rr=parse((r/'results.csv').read_bytes())
            check(name+': append preserves rows, lan and optional blanks',
                  br[:3]==parse(bb) and [(x['lan'],x['epoch']) for x in br]==[('0','1'),('1','2'),('1','3'),('2','3')]
                  and br[1]['loss_tong']=='' and br[1]['thu_tu_batch']==''
                  and [x['epoch'] for x in rr]==['1','2','3']
                  and all(None not in x and None not in x.values() for x in br+rr))
            current=snapshot(r)
            second=t.sao_luu_truoc_tiep_tuc(r,r/'checkpoint.pt',2,epoch_da_luu=3)
            check(name+': second resume preserves outputs and old backups',
                  all(sha((r/f).read_bytes())==current[f] and second['csv'][f]['hanh_dong']=='khong'
                      for f in ('results.csv',t.TEN_BATCH_CSV))
                  and (backup/'results.csv').read_bytes()==rraw)

    # Epoch filtering and tail repair must cooperate, while batch history is retained.
    rb=b'epoch,time,x\r\n1,10,0.5\r\n2,20,0.4\r\n3,30,0.3\r\n'
    bb=batch_bytes([row(0,1,0),row(0,3,0),row(2,2,0,True)],'\r\n')
    r=fixture(root,'filter',rb+b'4,',bb)
    saved=t.sao_luu_truoc_tiep_tuc(r,r/'checkpoint.pt',1,epoch_da_luu=2)
    check('results future epoch removed; all batch history and lan retained',
          [x['epoch'] for x in parse((r/'results.csv').read_bytes())]==['1','2']
          and (r/t.TEN_BATCH_CSV).read_bytes()==bb and saved['results_csv']=={'dong_giu':2,'dong_bo':1})

    bad = [('results_empty','results',b''),('results_header','results',rb.replace(b'epoch',b'epok',1)),
           ('results_middle','results',rb.replace(b'2,20,0.4',b'2,20')),
           ('batch_header','batch',bb.replace(b'lan,epoch',b'epoch,lan',1)),
           ('batch_middle','batch',bb.replace(b'0,3,0,20,',b'0,3,')),
           ('batch_empty','batch',b'')]
    for name,kind,raw in bad:
        r=fixture(root,name,rb,bb)
        p=r/('results.csv' if kind=='results' else t.TEN_BATCH_CSV)
        p.write_bytes(raw);before=snapshot(r)
        try:t.phan_tich_duoi_csv(raw,kind,p.name,2 if kind=='results' else None);rejected=False
        except t.LoiTiepTuc:rejected=True
        check(name+': controlled rejection without writes',rejected and snapshot(r)==before)

    # Guard inputs changed between planning and repair; no rewrite without exact backup.
    for mode in ('stale','missing_backup','wrong_backup','disappeared'):
        r=fixture(root,'guard_'+mode,rb+b'4,',bb);p=r/'results.csv';raw=p.read_bytes()
        plan={'co_file':True,**t.phan_tich_duoi_csv(raw,'results',p.name,2)}
        bdir=r/'save';bdir.mkdir()
        if mode!='missing_backup':(bdir/p.name).write_bytes(b'wrong' if mode=='wrong_backup' else raw)
        if mode=='stale':p.write_bytes(raw+b'9')
        if mode=='disappeared':p.unlink()
        before=snapshot(r)
        try:t.sua_duoi_csv(r,p.name,'results',bdir,plan,2);rejected=False
        except t.LoiTiepTuc:rejected=True
        check(mode+': controlled rejection and files unchanged',rejected and snapshot(r)==before)

    for target in ('results.csv',t.TEN_BATCH_CSV):
        r=fixture(root,'replace_'+target,rb+b'4,',bb+b'0,1,')
        p=r/target;raw=p.read_bytes();real=t.os.replace
        def fail_replace(source,dest):
            if Path(dest).resolve()==p.resolve():raise OSError('injected failure before CSV replacement')
            return real(source,dest)
        t.os.replace=fail_replace
        try:
            try:t.sao_luu_truoc_tiep_tuc(r,r/'checkpoint.pt',1,epoch_da_luu=2);rejected=False
            except t.LoiTiepTuc:rejected=True
        finally:t.os.replace=real
        backup=r/'khoi_phuc/truoc_tiep_tuc_1'/target
        check(target+': replace failure keeps original and backup',rejected and p.read_bytes()==raw and backup.read_bytes()==raw)
        next_lan=t.so_lan_tiep_tuc_da_co(r)+1
        t.sao_luu_truoc_tiep_tuc(r,r/'checkpoint.pt',next_lan,epoch_da_luu=2)
        check(target+': subsequent attempt recovers and preserves failed-attempt backup',
              next_lan==2 and backup.read_bytes()==raw
              and [x['epoch'] for x in parse((r/'results.csv').read_bytes())]==['1','2']
              and (r/t.TEN_BATCH_CSV).read_bytes()==bb)

result={'scope':__doc__,'source_sha256':sha(src.read_bytes()),'checks':checks,
        'pass':sum(x['pass'] for x in checks),'fail':sum(not x['pass'] for x in checks)}
a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'pass':result['pass'],'fail':result['fail'],'failed':[x for x in checks if not x['pass']]}))
raise SystemExit(bool(result['fail']))
