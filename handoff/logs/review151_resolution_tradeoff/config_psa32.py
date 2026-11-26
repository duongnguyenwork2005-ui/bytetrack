"""Bind two prospective mixed precision engines without replacing failed FP16 records."""
from pathlib import Path
import hashlib
import json

ROOT=Path.home()/'nano_run/r151_resolution_20261120'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    old=json.loads((ROOT/'models.json').read_text())
    models={name:old['models'][name] for name in ('original','anchor_rebuilt')}
    for source in ('r352x608','r320x576'):
        name=source+'_psa32';proof=json.loads((ROOT/(name+'_build_complete.json')).read_text())
        assert proof['status']=='PASS' and sha(ROOT/(name+'_build_inputs.json'))==proof['inputs_sha256']
        engine=Path(proof['engine']);assert sha(engine)==proof['engine_sha256']
        item=old['models'][source].copy()
        item.update(engine=str(engine),engine_sha256=proof['engine_sha256'],
                    precision='FP32 C2PSA /model.10; FP16 allowed outside',build_inputs_sha256=proof['inputs_sha256'])
        models[name]=item
    target=ROOT/'models_psa32.json';assert not target.exists()
    target.write_text(json.dumps({'models':models,'package':old['package'],
          'original_config_sha256':sha(ROOT/'models.json'),'failed_pure_FP16_excluded':['r352x608','r320x576']},indent=2))
    print('PSA32 CONFIG LOCKED',flush=True)


if __name__=='__main__':main()
