"""Verify every remote asset and every local transport assembly before closeout."""
from pathlib import Path
import argparse
import base64
import gzip
import hashlib
import json
import os
import subprocess
import urllib.request

API='https://api.github.com/repos/duongnguyenwork2005-ui/bytetrack'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--local',type=Path,required=True)
    args=parser.parse_args();base=args.local.resolve()
    package=read(base/'payload/package-complete.json')
    assert package['status']=='PASS'
    payload=read(base/'transfer/package-complete.json')['assets']
    native=read(base/'native/package-complete.json')['assets']
    assert len(payload)==201 and len(native)==3
    expected={a['name']:a for a in payload+native}
    assert len(expected)==204
    state=read(base/'release_state.json')
    assert state['release_id']==459160982
    result=subprocess.run(['git','credential','fill'],input='protocol=https\nhost=github.com\n\n',
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,check=True,
        env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never'))
    fields=dict(line.split('=',1) for line in result.stdout.splitlines() if '=' in line)
    auth='Basic '+base64.b64encode((fields['username']+':'+fields['password']).encode()).decode()
    def get(url):
        req=urllib.request.Request(url,headers={'User-Agent':'Codex-repository-backup',
            'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
        req.add_unredirected_header('Authorization',auth)
        with urllib.request.urlopen(req,timeout=60) as response:return json.load(response)
    release=get(API+'/releases/'+str(state['release_id']))
    assert release['draft'] is True and release['tag_name']=='backup-20261126-source'
    remote={};page=1
    while True:
        group=get(API+'/releases/'+str(release['id'])+'/assets?per_page=100&page='+str(page))
        remote.update({a['name']:a for a in group})
        if len(group)<100:break
        page+=1
    assert set(remote)==set(expected),{'missing':sorted(set(expected)-set(remote)),
                                      'unexpected':sorted(set(remote)-set(expected))}
    server=[]
    for name,asset in expected.items():
        actual=remote[name]
        assert actual['state']=='uploaded' and actual['size']==asset['bytes'],name
        assert actual.get('digest')=='sha256:'+asset['sha256'],name
        server.append({'name':name,'bytes':actual['size'],'sha256':asset['sha256'],'asset_id':actual['id']})
    assemblies=[]
    for folder,name in ((base/'transfer','transport-manifest.json.gz'),
                        (base/'native','nano-transport-manifest.json.gz')):
        with gzip.open(folder/name,'rt',encoding='utf-8') as stream:descriptor=json.load(stream)
        for archive in descriptor['archives']:
            full=hashlib.sha256();total=0
            for piece in archive['parts']:
                actual=hashlib.sha256();count=0
                with (folder/piece['name']).open('rb') as stream:
                    for data in iter(lambda:stream.read(1024*1024),b''):
                        actual.update(data);full.update(data);count+=len(data);total+=len(data)
                assert count==piece['bytes'] and actual.hexdigest()==piece['sha256'],piece['name']
                assert expected[piece['name']]['sha256']==piece['sha256']
            assert total==archive['bytes'] and full.hexdigest()==archive['sha256'],archive['name']
            assemblies.append({'name':archive['name'],'bytes':total,'sha256':full.hexdigest()})
            print('FULL_TRANSPORT_ASSEMBLY_VERIFIED',archive['name'],flush=True)
    assert len(assemblies)==31
    for name in ('payload-manifest.json.gz','transport-manifest.json.gz','nano-transport-manifest.json.gz'):
        folder=base/('native' if name.startswith('nano-') else 'transfer')
        with (folder/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        assert actual==expected[name]['sha256'],name
    assert read(base/'helper_validation.json')['status']=='PASS'
    assert read(base/'payload_validation.json')['status']=='PASS'
    assert read(base/'native_remote_restore_verified.json')['status']=='PASS'
    git_env=dict(os.environ,GIT_TERMINAL_PROMPT='0',GCM_INTERACTIVE='Never')
    refs=subprocess.run(['git','ls-remote','origin','refs/heads/phase2-mixed-traffic',
                         'refs/tags/backup-20261126-source','refs/tags/backup-20261126-restore-tools'],
                        check=True,stdout=subprocess.PIPE,text=True,env=git_env).stdout
    refs={line.split()[1]:line.split()[0] for line in refs.splitlines()}
    assert refs['refs/tags/backup-20261126-source']=='55916950112bca9ffd4327e554c4fa9a2faba346'
    out={'status':'PASS','release_id':release['id'],'release_url':release['html_url'],'draft':True,
         'remote_assets_verified':len(server),'remote_bytes_verified':sum(a['bytes'] for a in server),
         'payload_files':package['files'],'payload_logical_bytes':package['logical_bytes'],
         'unique_payload_objects':package['unique_objects'],'ignored_models_added':package['ignored_models_added'],
         'original_ZIP_archives':30,'native_files':15,'transport_assemblies_verified':31,
         'native_remote_download_and_file_roundtrip_verified':True,'Git_refs':refs,
         'assemblies':assemblies,'assets':server,
         'not_backed_up':['data/','tmp/','results/track_video/','handoff/archive/',
                         'other ignored files except selected local models','external drives','whole Nano filesystem']}
    (base/'final_verification.json').write_text(json.dumps(out,indent=2)+'\n')
    print('ALL_BACKUP_REMOTE_AND_TRANSPORT_VERIFIED',len(server),'assets',flush=True)


if __name__=='__main__':main()
