"""Run the unchanged R126 harness with the native float64 implementation."""
from pathlib import Path
import json
import sys
import runner as base
import native_pf

TOP=Path(__file__).resolve().parent
HERE=TOP/'native_validation'
base.HERE=HERE
base.installed=native_pf.installed


def preflight():
    assert not (HERE/'inputs.json').exists()
    old=json.loads((TOP/'inputs.json').read_text())
    for p,h in old['sha256'].items():assert base.bdd.sha(p)==h,p
    build=json.loads((TOP/'native/build.json').read_text())
    assert base.bdd.sha(TOP/'native/kernel.c')==build['source_sha256']
    assert base.bdd.sha(native_pf.LIB)==build['library_sha256']
    paths=[Path(__file__),TOP/'native_pf.py',TOP/'native/kernel.c',TOP/'native/build.py',
           TOP/'native/build.json',native_pf.LIB,HERE/'00_protocol.md',HERE/'numerical.json']
    hashes={**old['sha256'],**{str(p):base.bdd.sha(p) for p in paths}}
    base.write(HERE/'inputs.json',{'sha256':hashes,'timing_videos':old['timing_videos'],
                                 'frames':old['frames'],'config':old['config']})


if __name__=='__main__':
    if sys.argv[1].isdigit():base.full(int(sys.argv[1]))
    elif sys.argv[1]=='preflight':preflight()
    elif sys.argv[1]=='smoke':base.smoke()
    elif sys.argv[1]=='verify':base.verify()
    elif sys.argv[1]=='measure':
        import measure
        measure.main()
