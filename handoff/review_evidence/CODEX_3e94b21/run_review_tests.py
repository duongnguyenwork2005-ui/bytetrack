"""Run round-44 fixture tests without models, GPU, training, or real datasets.
Usage: python run_review_tests.py --tree TREE --deps DEPS --old-entry OLD --repro REPRO --out OUT
Only the entrypoint is replaced in a temporary copy for the negative control.
No project module is mocked. The pre-existing metadata repro explicitly uses doubles.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

WRAPPER = r'''
import json, runpy, sys
from pathlib import Path
source=Path(sys.argv[1]).resolve()
report=Path(sys.argv[2]).resolve()
sys.argv=[str(source)]
try:
    runpy.run_path(str(source),run_name='__main__')
except SystemExit as exc:
    rc=exc.code or 0
else:
    rc=0
mods={n:str(Path(m.__file__).resolve()) for n,m in sys.modules.items()
      if getattr(m,'__file__',None) and (n in ('config','bdd_yolo_export','detector_classes','eval_detector_5lop',
      'danh_gia_detector_bdd_val') or n.startswith('mot_datasets'))}
report.write_text(json.dumps(mods,indent=2))
raise SystemExit(rc)
'''

SCORER = r'''
import json,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(sys.argv[1]).resolve()/'src'))
import test_detector_5lop as test
with tempfile.TemporaryDirectory(prefix='codex_scorer44_') as d:
    test.nhom_e(Path(d),None)
    test.nhom_e2()
result={'pass':sum(v for _,v in test._RESULTS),'fail':sum(not v for _,v in test._RESULTS),
        'skip':test._SKIP,'checks':test._RESULTS}
Path(sys.argv[2]).write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
raise SystemExit(int(result['fail']>0))
'''


def main():
    ap=argparse.ArgumentParser(__doc__)
    for key in ['tree','deps','old-entry','repro','out']:
        ap.add_argument('--'+key,type=Path,required=True)
    a=ap.parse_args()
    tree=a.tree.resolve();out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONPATH=str(a.deps.resolve()),PYTHONDONTWRITEBYTECODE='1')
    env.pop('DET5_MODELS_DIR',None)
    before={str(p.relative_to(tree)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tree.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}
    runs=[]
    def run(name,argv,expected=0,cwd=tree):
        p=subprocess.run([sys.executable,*argv],cwd=cwd,env=env,text=True,capture_output=True)
        (out/(name+'.log')).write_text(p.stdout+'\n'+p.stderr+'\nEXIT_CODE='+str(p.returncode)+'\n')
        r={'name':name,'exit':p.returncode,'expected':expected,'pass':p.returncode==expected}
        runs.append(r);print(json.dumps(r))
    run('cli_help',[str(tree/'src/danh_gia_detector_bdd_val.py'),'--help'])
    run('bdd_current',['-c',WRAPPER,str(tree/'src/test_danh_gia_bdd_val.py'),str(out/'module_origins.json')])
    run('evaluator_groups_E',['-c',SCORER,str(tree),str(out/'scorer_tests.json')])
    run('repro_current',[str(a.repro.resolve()),str(tree),'--out',str(out/'repro_current.json')])
    with tempfile.TemporaryDirectory(prefix='codex_r44_old_') as d:
        old=Path(d)/'tree'
        shutil.copytree(tree/'src',old/'src',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(a.old_entry,old/'src/danh_gia_detector_bdd_val.py')
        run('bdd_old_entry',['-c',WRAPPER,str(old/'src/test_danh_gia_bdd_val.py'),str(out/'old_module_origins.json')],1,old)
    mods=json.loads((out/'module_origins.json').read_text())
    origins_ok=all(Path(p).is_relative_to(tree) for p in mods.values())
    after={str(p.relative_to(tree)):hashlib.sha256(p.read_bytes()).hexdigest() for p in tree.rglob('*')
           if p.is_file() and '__pycache__' not in p.parts}
    result={'runs':runs,'all_expected':all(r['pass'] for r in runs), 'project_module_origins_in_tree':origins_ok,
            'project_module_count':len(mods),'existing_tree_bytes_unchanged':before==after,
            'scope':'CPU fixture and scoring only; no real model, dataset or training'}
    (out/'test_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return int(not(result['all_expected'] and origins_ok and before==after))


if __name__=='__main__':
    raise SystemExit(main())
