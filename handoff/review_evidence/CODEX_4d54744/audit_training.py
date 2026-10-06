#!/usr/bin/env python3
"""Read-only reconciliation of round-42 supplied evidence. Does NOT load weights/train.

Usage: python audit_training.py --tree TREE --base BASE_PREIMAGES --out OUT
Outputs checks on supplied bytes and cross-document claims; excluded-file hashes
are declarations, not independent hashes of unavailable checkpoint/data bytes.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import re
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def readj(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def linesj(p):
    return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def rowscsv(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def canonical_ast(p):
    t = ast.parse(p.read_bytes())
    for n in ast.walk(t):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str):
                n.body.pop(0)
    return ast.dump(t, include_attributes=False)


def relative_windows(s):
    s = s.replace('\\', '/')
    return s.split('/bytetrack/', 1)[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tree', type=Path, required=True)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    r = a.tree/'handoff/logs/review42'
    checks = []
    def ck(name, cond):
        checks.append({'check': name, 'pass': bool(cond)})
    result = {'scope': 'Reconcile supplied log/JSON/CSV bytes only; no training, inference, tensor reload or raw RAM CSV check.', 'models': {}, 'checks': checks}
    old = a.base/'src/tiep_tuc_huan_luyen.py'
    new = a.tree/'src/tiep_tuc_huan_luyen.py'
    ck('src change is docstring-only (AST)', canonical_ast(old) == canonical_ast(new))
    compile(new.read_bytes(), str(new), 'exec')
    result['source'] = {'base_sha256': sha(old), 'target_sha256': sha(new), 'syntax_compile': True}
    prior = a.tree/'handoff/review_evidence/CODEX_6c8e37a'
    previous_count = 0
    for line in (prior/'SHA256SUMS.txt').read_text().splitlines():
        h, name = line.split(None, 1)
        ck('prior evidence '+name, sha(prior/name.lstrip('*')) == h)
        previous_count += 1
    result['prior_evidence_files'] = previous_count
    ck('prior reviewed source equals base', sha(prior/'tree/src/tiep_tuc_huan_luyen.py') == sha(old))
    large = rowscsv(r/'ket_qua/tep_lon_khong_ban_giao.csv')
    index = {x['duong_dan']: x for x in large}
    ck('54 excluded files declared uniquely', len(large) == len(index) == 54)
    ck('excluded declarations syntactically valid', all(int(x['bytes']) > 0 and re.fullmatch('[0-9a-f]{64}', x['sha256']) for x in large))
    all_starts, all_monitors, epoch_output = [], [], []
    for model, sn, attempts in [('yolov8n', '16', 3), ('yolo11n', '20', 1)]:
        root = r/'ket_qua'/model
        journal = linesj(root/'luot/nhat_ky_luot.jsonl')
        starts = [x for x in journal if x['loai'] == 'bat_dau']
        ends = [x for x in journal if x['loai'] == 'ket_thuc']
        epochs = [x for x in journal if x['loai'] == 'epoch_da_luu']
        e_by = {x['epoch']: x for x in epochs}
        raw_csv = rowscsv(root/'luot/results.csv')
        by_csv = {int(x['epoch']): x for x in raw_csv}
        runners = sorted([readj(p) for p in (root/'runner').glob('*.json')], key=lambda x: (x.get('tiep_tuc') or {}).get('lan', 0))
        summary = readj(r/f'{sn}_tom_tat_{model}.json')
        final = runners[-1]
        ck(model+' attempt sequence', [x['lan'] for x in starts] == list(range(attempts)) == [x['lan'] for x in ends] and len(runners) == attempts)
        ck(model+' journal has exactly 15 unique ordered epochs', [x['epoch'] for x in epochs] == list(range(1, 16)))
        ck(model+' CSV has exactly 15 unique ordered epochs', [int(x['epoch']) for x in raw_csv] == list(range(1, 16)))
        ck(model+' CSV finite', all(math.isfinite(float(v)) for x in raw_csv for v in x.values()))
        ck(model+' journal timestamps monotonic', all(x['t'] < y['t'] for x, y in zip(journal, journal[1:])))
        ck(model+' attempt exits', [x['exit_code'] for x in ends] == ([130, 130, 0] if attempts == 3 else [0]))
        ck(model+' final completion', ends[-1]['trang_thai'] == 'hoan_tat' and ends[-1]['kiem_sau_train_dat'] and final['loi'] is None and not final.get('loi_callback') and final['loss_huu_han'])
        runner_epochs = [e for j in runners for e in j['epoch']]
        ck(model+' runner epochs exactly 1..15', [e['epoch'] for e in runner_epochs] == list(range(1, 16)))
        ck(model+' all completed batches and losses', all(e['so_batch'] == 6311 and e['loss_huu_han'] and all(math.isfinite(v) for v in e['loss'].values()) for e in runner_epochs))
        ck(model+' journal/runner/CSV metrics', all(e_by[e['epoch']]['metrics'] == e['metrics'] and all(float(by_csv[e['epoch']][k]) == v for k, v in e['metrics'].items()) for e in runner_epochs))
        ck(model+' journal backup declarations', all(e['last_sha256'] == e['ban_sao_sha256'] == index[f"models/finetune_phase2/train_bdd5_v1_{model}_day_du/{e['ban_sao']}"]['sha256'] and e['last_bytes'] == int(index[f"models/finetune_phase2/train_bdd5_v1_{model}_day_du/{e['ban_sao']}"]['bytes']) for e in epochs))
        ck(model+' mosaic only closed epochs 14,15', all(e['mosaic_thuc_te'] == (e['epoch'] < 14) and e['mosaic_p'] == (1.0 if e['epoch'] < 14 else 0.0) for e in epochs))
        fit_best = max(epochs, key=lambda e:e['fitness'])
        ck(model+' best selection', [e['epoch'] for e in epochs if e['best_ghi_epoch_nay']][-1] == fit_best['epoch'] and fit_best['epoch'] == (15 if model == 'yolov8n' else 13))
        ck(model+' best running maxima', all(e['best_fitness'] == max(v['fitness'] for v in epochs[:e['epoch']]) for e in epochs))
        for k in ['best', 'last']:
            c = final['checkpoint'][k]
            declared = index[relative_windows(c['duong_dan'])]
            ck(model+' '+k+' final checkpoint declarations', c['sha256'] == declared['sha256'] == summary['checkpoint'][k]['sha256'] == summary['checkpoint'][k]['sha256_ket_qua_runner'] and c['bytes'] == int(declared['bytes']) == summary['checkpoint'][k]['bytes'])
            ck(model+' '+k+' recorded reload checks', c['nap_lai'] and c['dung_5_lop'] and not c['train_args_lech_cau_hinh'] and list(c['names'].values()) == ['car','truck','bus','motorcycle','bicycle'])
        resumes = []
        for i, j in enumerate(runners):
            s = starts[i]
            ck(model+f' attempt {i} declared args', s['tham_so_khai_bao'] == j['tham_so_khai_bao'])
            ck(model+f' attempt {i} effective args', all(j['hieu_luc']['args'][k] == v for k, v in s['tham_so_khai_bao'].items() if k != 'model'))
            ck(model+f' attempt {i} dataset identity', all(j['dataset'][k] == v for k, v in s['dataset_van_tay'].items()))
            ck(model+f' attempt {i} loaded dataset counts', all(j['dau_vao_da_nap'][split]['so_anh_nap'] == count == j['dau_vao_da_nap'][split]['so_anh_da_xac_minh'] and all(j['dau_vao_da_nap'][split][k] == 0 for k in ['thieu','thua','nhan_khac','npy_canh_anh']) for split,count in [('train',50487),('dev',5559)]))
            ck(model+f' attempt {i} final event epoch subset', ends[i]['epoch_da_luu_lan_nay'] == [e['epoch'] for e in epochs if e['lan'] == i] == [e['epoch'] for e in j['epoch']])
            ck(model+f' attempt {i} source hashes', s['ma_nguon_sha256']['tiep_tuc_huan_luyen.py'] == sha(new) and s['ma_nguon_sha256']['train_detector.py'] == sha(prior/'tree/src/train_detector.py'))
            if i:
                t = j['tiep_tuc']; prev = e_by[t['epoch_da_luu']]
                ck(model+f' resume {i} correct boundary and hash', s['start_epoch'] == prev['epoch'] == (7 if i == 1 else 11) and t['epoch_bat_dau_tiep'] == prev['epoch'] + 1 and t['tong_epoch'] == 15 and t['chon']['sha256'] == prev['ban_sao_sha256'] == t['sao_luu']['nguon_sha256'] == index[relative_windows(t['sao_luu']['nguon'])]['sha256'])
                ck(model+f' resume {i} 13 restoration checks', len(j['kiem_khoi_phuc']) == 13 and all(x['dat'] for x in j['kiem_khoi_phuc'].values()))
                ck(model+f' resume {i} no CSV history removed', all(x['hanh_dong'] == 'khong' and x['bo_bytes'] == 0 for x in t['csv_duoi'].values()) and t['sao_luu']['results_csv']['dong_bo'] == 0)
                resumes.append({'attempt':i,'last_saved_epoch':prev['epoch'],'next_epoch':t['epoch_bat_dau_tiep'],'checkpoint_sha256':t['chon']['sha256'],'restoration_checks':len(j['kiem_khoi_phuc'])})
        mon = sorted([readj(p) for p in (root/'giam_sat').glob('*/tom_tat.json')], key=lambda x:x['bat_dau'])
        for i, x in enumerate(mon):
            ck(model+f' monitor {i} PID/exit match journal', x['pid'] == starts[i]['pid'] and x['exit_code'] == ends[i]['exit_code'])
            ck(model+f' monitor {i} encloses journal', datetime.fromisoformat(x['bat_dau']).timestamp() <= starts[i]['t'] < ends[i]['t'] <= datetime.fromisoformat(x['ket_thuc']).timestamp()+1)
        final_log = (root/'giam_sat'/f"{model}_{'tiep_tuc_2' if attempts==3 else 'lan0'}"/'log_loc.txt').read_text()
        ck(model+' final DAT marker', f'[DAT] day-du {model}' in final_log)
        ck(model+' summaries runner seconds', [j['thoi_gian_s'] for j in runners] == summary['thoi_gian_runner_s'])
        snapshots = linesj(r/f'theo_doi_{model}.log')
        gaps = []
        for x, y in zip(snapshots,snapshots[1:]):
            tx,ty=map(datetime.fromisoformat,[x['t'],y['t']])
            secs=(ty-tx).total_seconds()
            same_active_attempt=any(datetime.fromisoformat(m['bat_dau']) <= tx < ty <= datetime.fromisoformat(m['ket_thuc']) for m in mon)
            if secs > 3600 and same_active_attempt:
                gaps.append({'from':x['t'],'to':y['t'],'minutes':secs/60})
        for e in epochs:
            epoch_output.append({'model':model,'attempt':e['lan'],'epoch':e['epoch'],'timestamp':e['t_iso'],'mAP50':e['metrics']['metrics/mAP50(B)'],'mAP50_95':e['metrics']['metrics/mAP50-95(B)'],'best_epoch_so_far':e['dung_som']['best_epoch'],'saved_sha256':e['ban_sao_sha256']})
        result['models'][model] = {'epochs': [e['epoch'] for e in epochs], 'attempt_exits':[e['exit_code'] for e in ends], 'resumes':resumes, 'best_epoch':fit_best['epoch'], 'best_sha256':final['checkpoint']['best']['sha256'], 'final_dev_metrics':final['danh_gia_cuoi_ultralytics']['metrics'], 'runner_seconds':sum(j['thoi_gian_s'] for j in runners), 'monitor_seconds':sum(x['thoi_gian_s'] for x in mon), 'train_seconds_range':[min(e['train_s'] for e in runner_epochs),max(e['train_s'] for e in runner_epochs)], 'val_seconds_range':[min(e['val_s'] for e in runner_epochs),max(e['val_s'] for e in runner_epochs)], 'gpu_total_memory_max_MiB':max(j['vram_nvidia_smi']['max_mib'] for j in runners), 'RSS_sum_max_MiB':max(x['ram']['rss_tong_mib']['max'] for x in mon), 'monitor_available_RAM_min_MiB':min(x['ram']['ram_he_thong_con_trong_mib']['min'] for x in mon), 'runner_available_RAM_min_MiB':min(j['vram_nvidia_smi']['ram_con_trong_min_mib'] for j in runners), 'snapshots':len(snapshots),'active_snapshot_gaps_over_60_minutes':gaps}
        all_starts += starts; all_monitors += mon
    exemplar = all_starts[0]
    ck('all attempts common declared configuration', all({k:v for k,v in x['tham_so_khai_bao'].items() if k not in ['model','name']} == {k:v for k,v in exemplar['tham_so_khai_bao'].items() if k not in ['model','name']} for x in all_starts))
    ck('all attempts same dataset/config/source identities', all(x['dataset_van_tay'] == exemplar['dataset_van_tay'] and x['cau_hinh'] == exemplar['cau_hinh'] and x['ma_nguon_sha256'] == exemplar['ma_nguon_sha256'] and x['ultralytics_trainer_sha256'] == exemplar['ultralytics_trainer_sha256'] for x in all_starts))
    ck('training sequential', all(datetime.fromisoformat(x['ket_thuc']) < datetime.fromisoformat(y['bat_dau']) for x,y in zip(all_monitors,all_monitors[1:])))
    result['total_runner_hours'] = sum(x['runner_seconds'] for x in result['models'].values())/3600
    result['total_monitor_hours'] = sum(x['monitor_seconds'] for x in result['models'].values())/3600
    result['elapsed_calendar_hours'] = (datetime.fromisoformat(all_monitors[-1]['ket_thuc'])-datetime.fromisoformat(all_monitors[0]['bat_dau'])).total_seconds()/3600
    # Independent check of every displayed 3-decimal epoch AP, using source CSV decimals.
    doc=(a.tree/'handoff/REVIEW42_CHI_TIET.md').read_text()
    doc_rows=[line for line in doc.splitlines() if line.startswith('| mAP50')]
    rounding=[]
    for model, group in zip(['yolov8n','yolo11n'],[doc_rows[:2],doc_rows[2:]]):
        csv_rows=rowscsv(r/f'ket_qua/{model}/luot/results.csv')
        for line in group:
            cells=[x.strip().replace('**','').replace(',','.') for x in line.strip('|').split('|')]
            metric=cells[0]; key=f'metrics/{metric}(B)'
            for row,printed in zip(csv_rows,cells[1:]):
                rounded=str(Decimal(row[key]).quantize(Decimal('.001'),rounding=ROUND_HALF_UP))
                if Decimal(printed) != Decimal(rounded):
                    rounding.append({'model':model,'epoch':int(row['epoch']),'metric':metric,'source':row[key],'printed':printed,'correct_3dp':rounded})
    result['document_rounding_differences']=rounding
    result['nan_progress']={'onset_batch':4597,'onset_percent':4597/6311*100,'last_logged_batch_before_stop':4786,'last_logged_percent_before_stop':4786/6311*100,'source':'11_dung_luot_nan_yolov8n.log; this is the last logged progress, not proof of exact last executed batch'}
    result['checks_passed']=sum(x['pass'] for x in checks)
    result['checks_total']=len(checks)
    result['all_consistency_checks_pass']=all(x['pass'] for x in checks)
    (out/'training_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'epochs_reconciled.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(epoch_output[0]));w.writeheader();w.writerows(epoch_output)
    print(json.dumps({k:result[k] for k in ['checks_passed','checks_total','all_consistency_checks_pass','total_runner_hours','elapsed_calendar_hours','document_rounding_differences']},ensure_ascii=False,indent=2))
    for c in checks:
        if not c['pass']:print('FAIL',c['check'])
    return 0 if result['all_consistency_checks_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
