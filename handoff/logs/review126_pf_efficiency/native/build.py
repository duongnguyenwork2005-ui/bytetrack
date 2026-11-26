"""Build a workspace-local CPU library using an existing compiler."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
source=HERE/'kernel.c'
if os.name=='nt':
    vswhere=Path('C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe')
    vs=subprocess.check_output([str(vswhere),'-latest','-products','*','-requires',
                                 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64','-property','installationPath'],text=True).strip()
    bat=Path(vs)/'Common7/Tools/VsDevCmd.bat'
    env_command=f'cmd.exe /d /s /c "call "{bat}" -no_logo -arch=x64 >nul && set"'
    result=subprocess.run(env_command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    env=dict(os.environ)
    for line in result.stdout.splitlines():
        if '=' in line and not line.startswith('='):
            k,v=line.split('=',1);env[k]=v
    compiler=shutil.which('cl.exe',path=env.get('Path',env.get('PATH')))
    assert compiler
    output=HERE/'kernel.dll'
    command=[compiler,'/nologo','/LD','/O2','/fp:strict','/MD','/std:c11',str(source),f'/Fe:{output}',f'/Fo:{HERE/"kernel.obj"}']
else:
    compiler=shutil.which('cc') or shutil.which('gcc')
    assert compiler
    env=dict(os.environ);output=HERE/'kernel.so'
    command=[compiler,'-shared','-O3','-std=c11','-fPIC','-fno-fast-math','-ffp-contract=off',str(source),'-lm','-o',str(output)]
result=subprocess.run(command,cwd=HERE,env=env,capture_output=True,text=True)
(HERE/'build.log').write_text(result.stdout+'\n'+result.stderr,encoding='utf8')
assert result.returncode==0,(result.stdout,result.stderr)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
(HERE/'build.json').write_text(json.dumps({'source_sha256':sha(source),'library_sha256':sha(output),
       'compiler':compiler,'compiler_sha256':sha(compiler),'command':command,'platform':sys.platform},indent=2))
print('BUILD PASS',output,flush=True)
