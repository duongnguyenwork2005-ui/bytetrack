"""Download draft-release assets with Git credentials and verify GitHub SHA256.

Authorization is attached as an unredirected header, so it is not forwarded
to the signed object-storage URL returned by GitHub.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import os
import subprocess
import time
import urllib.request

API='https://api.github.com/repos/duongnguyenwork2005-ui/bytetrack'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--release-id',type=int,required=True)
    parser.add_argument('--dest',type=Path,required=True)
    parser.add_argument('--names',nargs='*')
    args=parser.parse_args()
    result=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
        text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,
        env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never'))
    fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    auth='Basic '+base64.b64encode((fields['username']+':'+fields['password']).encode()).decode()
    def request(url,accept='application/vnd.github+json'):
        req=urllib.request.Request(url,headers={'User-Agent':'Codex-repository-backup',
            'Accept':accept,'X-GitHub-Api-Version':'2022-11-28'})
        req.add_unredirected_header('Authorization',auth)
        return urllib.request.urlopen(req,timeout=120)
    with request(API+'/releases/'+str(args.release_id)) as response:
        release=json.load(response)
    assert release['id']==args.release_id
    assets=[];page=1
    while True:
        with request(API+'/releases/'+str(args.release_id)+'/assets?per_page=100&page='+str(page)) as response:
            group=json.load(response)
        assets.extend(group)
        if len(group)<100:break
        page+=1
    if args.names:
        wanted=set(args.names)
        assets=[a for a in assets if a['name'] in wanted]
        assert {a['name'] for a in assets}==wanted,'Requested asset is missing'
    dest=args.dest.resolve();dest.mkdir(parents=True,exist_ok=True)
    verified=[]
    for asset in assets:
        assert Path(asset['name']).name==asset['name']
        expected=asset.get('digest','')
        assert expected.startswith('sha256:') and asset['state']=='uploaded',asset['name']
        target=dest/asset['name']
        assert target.resolve().is_relative_to(dest)
        if target.exists():
            assert target.is_file() and target.stat().st_size==asset['size'] and 'sha256:'+sha(target)==expected,target
        else:
            temporary=dest/(asset['name']+'.download-'+str(time.time_ns()))
            actual=hashlib.sha256();count=0
            with request(asset['url'],'application/octet-stream') as response,temporary.open('xb') as stream:
                for chunk in iter(lambda:response.read(1024*1024),b''):
                    stream.write(chunk);actual.update(chunk);count+=len(chunk)
            assert count==asset['size'] and 'sha256:'+actual.hexdigest()==expected,asset['name']
            assert not target.exists()
            temporary.rename(target)
        verified.append({'name':asset['name'],'bytes':asset['size'],'sha256':expected.removeprefix('sha256:')})
        print('DOWNLOAD_SHA256_VERIFIED',asset['name'],asset['size'],flush=True)
    (dest/'download-verification.json').write_text(json.dumps({'status':'PASS','release_id':args.release_id,
        'draft':release['draft'],'assets':verified},indent=2)+'\n')


if __name__=='__main__':main()
