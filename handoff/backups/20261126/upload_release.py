"""Upload verified payload parts to a draft GitHub release using existing Git credentials.

Credentials remain in memory; this script never prints or saves them. Existing
uploaded assets are reused only when size and GitHub SHA256 digest match.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

REPO = 'duongnguyenwork2005-ui/bytetrack'
API = 'https://api.github.com/repos/' + REPO


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class ProgressReader:
    def __init__(self, stream, name, size):
        self.stream, self.name, self.size = stream, name, size
        self.sent, self.reported = 0, time.monotonic()

    def read(self, size=-1):
        chunk = self.stream.read(size)
        self.sent += len(chunk)
        if time.monotonic() - self.reported > 15:
            print('UPLOADING', self.name, self.sent, '/', self.size, flush=True)
            self.reported = time.monotonic()
        return chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--tag', default='backup-20261126-source')
    parser.add_argument('--state', type=Path, required=True)
    args = parser.parse_args()
    result = subprocess.run(['git', 'credential', 'fill'],
                            input='protocol=https\nhost=github.com\n\n', text=True,
                            env=dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='Never'),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    credentials = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    authorization = 'Basic ' + base64.b64encode((credentials['username'] + ':' + credentials['password']).encode()).decode()

    def request(url, method='GET', body=None, file=None):
        headers = {'Authorization': authorization, 'User-Agent': 'Codex-repository-backup',
                   'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'}
        if file is not None:
            headers.update({'Content-Type': 'application/octet-stream', 'Content-Length': str(file.stat().st_size)})
            with file.open('rb') as stream:
                req = urllib.request.Request(url, method=method, headers=headers,
                                             data=ProgressReader(stream, file.name, file.stat().st_size))
                with urllib.request.urlopen(req, timeout=120) as response:
                    return json.load(response)
        data = None if body is None else json.dumps(body).encode()
        if data is not None:
            headers['Content-Type'] = 'application/json'
        req = urllib.request.Request(url, method=method, headers=headers, data=data)
        with urllib.request.urlopen(req, timeout=60) as response:
            return json.load(response) if response.status != 204 else None

    package = json.loads((args.assets / 'package-complete.json').read_text())
    assert package['status'] == 'PASS'
    if args.state.exists():
        state = json.loads(args.state.read_text())
        assert state['commit'] == args.commit and state['tag'] == args.tag
        release = request(API + '/releases/' + str(state['release_id']))
    else:
        release = request(API + '/releases', 'POST', {
            'tag_name': args.tag, 'target_commitish': args.commit,
            'name': 'Private backup payload — 2026-11-26', 'draft': True,
            'body': 'Backup of generated research evidence and local models for commit ' + args.commit +
                    '. Keep this release as a draft. Raw datasets/ignored temporary trees/external drives/Nano filesystem are outside scope. '
                    'Clone the Git branch/tag for source and use restore_payload.py with all payload assets. '
                    'SHA256 hashes and duplicate-file mappings are in payload-manifest.json.gz.'})
        state = {'repository': REPO, 'release_id': release['id'], 'commit': args.commit,
                 'tag': args.tag, 'html_url': release['html_url'], 'draft': True, 'uploaded': {}}
        args.state.write_text(json.dumps(state, indent=2) + '\n')
    assert release['draft'] is True and release['tag_name'] == args.tag
    print('DRAFT_RELEASE', state['html_url'], flush=True)

    def remote_assets():
        all_assets, page = [], 1
        while True:
            group = request(API + '/releases/' + str(release['id']) + '/assets?per_page=100&page=' + str(page))
            all_assets.extend(group)
            if len(group) < 100:
                return {r['name']: r for r in all_assets}
            page += 1

    for asset in package['assets']:
        path = args.assets / asset['name']
        assert path.stat().st_size == asset['bytes'] and sha(path) == asset['sha256'], path
        expected = 'sha256:' + asset['sha256']
        remote = remote_assets().get(path.name)
        if remote is not None and remote['state'] == 'starter':
            # Only remove our incomplete asset in this exact still-draft backup.
            request(remote['url'], 'DELETE')
            remote = None
        if remote is None:
            upload_url = release['upload_url'].split('{', 1)[0] + '?' + urllib.parse.urlencode({'name': path.name})
            remote = request(upload_url, 'POST', file=path)
        assert remote['state'] == 'uploaded' and remote['size'] == asset['bytes'], remote['name']
        assert remote.get('digest') == expected, 'Server SHA256 digest unavailable or mismatched: ' + path.name
        state['uploaded'][path.name] = {'id': remote['id'], 'bytes': remote['size'], 'sha256': asset['sha256'],
                                       'server_digest': remote['digest'], 'browser_download_url': remote['browser_download_url']}
        args.state.write_text(json.dumps(state, indent=2) + '\n')
        print('VERIFIED_REMOTE_ASSET', path.name, remote['size'], flush=True)
    verified = remote_assets()
    for asset in package['assets']:
        remote = verified[asset['name']]
        assert remote['size'] == asset['bytes'] and remote.get('digest') == 'sha256:' + asset['sha256']
    state['status'] = 'PASS'
    state['draft'] = request(API + '/releases/' + str(release['id']))['draft']
    assert state['draft'] is True
    args.state.write_text(json.dumps(state, indent=2) + '\n')
    print('DRAFT_BACKUP_ALL_ASSETS_VERIFIED', len(package['assets']), flush=True)


if __name__ == '__main__':
    main()
