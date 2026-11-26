"""Run staged speed trials serially; stop before work when memory is unsuitable."""
from pathlib import Path
import hashlib,json,os,subprocess,sys

HERE=Path(__file__).resolve().parent;BASE=HERE.parent
CONDA='/home/jetson/miniforge3/envs/nano/bin/python'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):p.write_text(json.dumps(value,indent=2)+'\n')
def memory_guard(root,label):
    values={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
            if line.startswith(('SwapTotal:','SwapFree:'))}
    used=(values['SwapTotal']-values['SwapFree'])/1024.
    if used>=100.:
        state={'state':'WAITING_FOR_SWAP_RESET','before':label,'root':str(root),'swap_MiB':used,'required_start_MiB':'<100'}
        write(HERE/'extra_chain_state.json',state);print(json.dumps(state),flush=True);sys.exit(75)
    assert Path('/sys/fs/ext4/mmcblk1p1/errors_count').read_text().strip()=='0'
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name)==os.getpid():continue
        try:args=(folder/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError,PermissionError):continue
        if not args:continue
        program=Path(args[0]).name
        if program=='trtexec' or (program.startswith('python') and any(Path(a).name in
                ('capture.py','oracle.py','micro.py','timing_v2.py','timing_v3.py','nano_detect_stream.py') for a in args[1:3])):
            raise RuntimeError('Another GPU job is active: '+folder.name+' '+' '.join(args[:4]))
        if program=='bash' and any(Path(a).name=='prepare_timing.sh' for a in args[1:3]):
            raise RuntimeError('Root memory preparation is still running; retry after it finishes')
def run(root,label,command,log):
    memory_guard(root,label)
    state={'state':'RUNNING','root':str(root),'phase':label,'command':command,'driver_sha256':sha(Path(__file__))}
    write(HERE/'extra_chain_state.json',state)
    record=root/'extra_chain_commands.json'
    commands=read(record) if record.exists() else []
    commands.append(state);write(record,commands)
    print('EXTRA_CHAIN_START',label,str(root),flush=True)
    with (root/log).open('x') as stream:
        result=subprocess.run(command,cwd=str(root),stdout=stream,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
                         LD_PRELOAD='/usr/lib/aarch64-linux-gnu/libgomp.so.1'))
    commands[-1].update(returncode=result.returncode,log_sha256=sha(root/log));write(record,commands)
    assert result.returncode==0,(label,result.returncode,str(root/log))
    print('EXTRA_CHAIN_COMPLETE',label,flush=True)
def timing(root,version,phase):
    output=root/'runs_absolute'/phase;marker=output/'complete.json'
    config=root/('full_models.json' if phase=='full' else 'timing_models_'+version+'.json')
    if marker.exists():
        done=read(marker);assert done['status']=='PASS'
        if phase!='engine':assert done['models_sha256']==sha(config)
        return
    assert not output.exists(),'Existing partial trial requires review: '+str(output)
    run(root,phase,[CONDA,str(root/('timing_'+version+'.py')),phase,str(config),str(output)],phase+'.log')
def main():
    for folder,version in (('r162_cuda_pre_20261120','v3'),('r150_pruning_20261120','v2'),('r160_height352_20261120','v2')):
        root=BASE/folder
        if version=='v3':
            if (root/'micro.json').exists():assert read(root/'micro.json')['status']=='PASS'
            else:run(root,'cuda_micro',['python3',str(root/'micro.py')],'micro.log')
        else:timing(root,version,'engine')
        timing(root,version,'pilot')
        selection=root/'selection.json'
        if not selection.exists():
            # Selection is CPU-only; it only reads the frozen pilot outputs.
            with (root/'selection.log').open('x') as stream:
                subprocess.run([CONDA,str(root/'select_full.py'),str(root)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        selected=read(selection)
        assert selected['pilot_complete_sha256']==sha(root/'runs_absolute/pilot/complete.json')
        if selected['full_skipped']:
            print('EXTRA_CHAIN_NEGATIVE_PILOT',folder,flush=True);continue
        timing(root,version,'full')
        archive=root/'full80_evidence_20261120.tar.gz'
        if not archive.exists():
            with (root/'collection.log').open('x') as stream:
                subprocess.run([CONDA,str(HERE/'collect_extra.py'),str(root)],stdout=stream,stderr=subprocess.STDOUT,check=True)
        assert (root/'collection.json').exists()
    write(HERE/'extra_chain_state.json',{'state':'ALL_SPEED_TRIALS_COMPLETE','driver_sha256':sha(Path(__file__))})
    print('EXTRA_SPEED_CHAIN_COMPLETE',flush=True)
if __name__=='__main__':main()
