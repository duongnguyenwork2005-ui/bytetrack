"""Require both independent numerical suites before freezing any tracking input."""
from pathlib import Path
import importlib.util
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('r165_bound_preflight',HERE/'run.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
assert r.j.read(HERE/'numerical.json')['status']=='PASS'
rejection=r.j.read(HERE/'rejection_oracle.json');assert rejection['status']=='PASS'
for p,h in rejection['sha256'].items():assert r.j.sha(p)==h,p
r.preflight();path=HERE/'dev/inputs.json';contract=r.j.read(path)
contract['sha256'].update({str(p):r.j.sha(p) for p in (Path(__file__),HERE/'rejection_oracle.py',HERE/'rejection_oracle.json')})
r.j.write(path,contract)
print('R165 STRONG REFERENCE PREFLIGHT PASS',flush=True)
