"""Resume this exact draft backup with four bounded, SHA256-verified uploads."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import base64
import hashlib
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

API = 'https://api.github.com/repos/duongnguyenwork2005-ui/bytetrack'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2, 4), default=4)
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    assert state['repository'] == 'duongnguyenwork2005-ui/bytetrack' and state['draft'] is True
    package = json.loads((args.assets/'package-complete.json').read_text())
    assert package['status'] == 'PASS'
    credentials_result = subprocess.run(['git', 'credential', 'fill'],
        input='protocol=https\nhost=github.com\n\n', text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        env=dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never'))
    credentials = dict(line.split('=', 1) for line in credentials_result.stdout.splitlines() if '=' in line)
    auth = 'Basic ' + base64.b64encode((credentials['username']+':'+credentials['password']).encode()).decode()
    lock = threading.Lock()

    def say(*values):
        with lock:
            print(*values, flush=True)

    class Reader:
        def __init__(self, stream, name, length):
            self.stream, self.name, self.length = stream, name, length
            self.count, self.last = 0, time.monotonic()
        def read(self, size=-1):
            data = self.stream.read(size); self.count += len(data)
            if time.monotonic()-self.last > 30:
                say('UPLOADING', self.name, self.count, '/', self.length)
                self.last = time.monotonic()
            return data

    def request(url, method='GET', path=None):
        headers = {'Authorization': auth, 'User-Agent': 'Codex-repository-backup',
                   'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'}
        if path is not None:
            headers.update({'Content-Type': 'application/octet-stream', 'Content-Length': str(path.stat().st_size)})
            with path.open('rb') as stream:
                req = urllib.request.Request(url, data=Reader(stream, path.name, path.stat().st_size), method=method, headers=headers)
                with urllib.request.urlopen(req, timeout=120) as response:
                    return json.load(response)
        req = urllib.request.Request(url, method=method, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response) if response.status != 204 else None

    release = request(API+'/releases/'+str(state['release_id']))
    assert release['draft'] is True and release['tag_name'] == state['tag']
    def list_assets():
        result, page = {}, 1
        while True:
            group = request(API+'/releases/'+str(release['id'])+'/assets?per_page=100&page='+str(page))
            result.update({a['name']: a for a in group})
            if len(group)<100:return result
            page += 1
    existing = list_assets()

    def save(asset, remote):
        assert remote['name'] == asset['name'] and remote['state'] == 'uploaded'
        assert remote['size'] == asset['bytes'] and remote.get('digest') == 'sha256:'+asset['sha256'], asset['name']
        with lock:
            state['uploaded'][asset['name']] = {
                'id': remote['id'], 'bytes': remote['size'], 'sha256': asset['sha256'],
                'server_digest': remote['digest'], 'browser_download_url': remote['browser_download_url']}
            state['status'] = 'RUNNING'
            temp = args.state.with_suffix('.writing.json')
            temp.write_text(json.dumps(state, indent=2)+'\n')
            os.replace(temp, args.state)
            print('VERIFIED_REMOTE_ASSET', asset['name'], remote['size'], 'total', len(state['uploaded']), flush=True)

    def upload(asset):
        path = args.assets/asset['name']
        assert path.stat().st_size == asset['bytes'] and sha(path) == asset['sha256'], path
        remote = existing.get(asset['name'])
        for attempt in range(4):
            try:
                if remote is not None and remote['state'] == 'starter':
                    request(remote['url'], 'DELETE'); remote = None
                if remote is None:
                    url = release['upload_url'].split('{',1)[0]+'?'+urllib.parse.urlencode({'name':path.name})
                    remote = request(url, 'POST', path)
                save(asset, remote)
                return
            except (OSError, urllib.error.HTTPError) as error:
                if attempt == 3:raise
                say('RETRY_ASSET', path.name, attempt+1, type(error).__name__)
                time.sleep(3)
                remote = list_assets().get(path.name)

    pending = []
    for asset in package['assets']:
        remote = existing.get(asset['name'])
        if remote is not None and remote['state']=='uploaded':
            assert remote['size']==asset['bytes'] and remote.get('digest')=='sha256:'+asset['sha256'], asset['name']
            if asset['name'] not in state['uploaded']:save(asset,remote)
        else:pending.append(asset)
    say('PARALLEL_UPLOAD_START', len(pending), 'assets', args.workers, 'workers')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(upload,asset) for asset in pending]
        for future in as_completed(futures):future.result()
    verified = list_assets()
    for asset in package['assets']:
        remote = verified[asset['name']]
        assert remote['state']=='uploaded' and remote['size']==asset['bytes']
        assert remote.get('digest')=='sha256:'+asset['sha256']
    assert request(API+'/releases/'+str(release['id']))['draft'] is True
    with lock:
        state['status']='PASS'
        args.state.write_text(json.dumps(state,indent=2)+'\n')
    say('PARALLEL_DRAFT_ALL_ASSETS_VERIFIED',len(package['assets']))


if __name__=='__main__':main()
