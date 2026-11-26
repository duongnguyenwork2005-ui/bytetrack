from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'review167_fused_silu'
REMOTE = '/home/jetson/nano_run/r174_silu_graph_20261126/'


def main():
    quality = json.loads((PARENT / 'final_summary.json').read_text())
    package = json.loads((PARENT / 'package_verification.json').read_text())
    assert quality['status'] == package['status'] == 'PASS'
    bind = {
        quality['detection'][name]['engine']['engine']: quality['detection'][name]['engine']['engine_sha256']
        for name in ('original', 'precise')
    }
    bind['/home/jetson/nano_run/r167_fused_silu_20261121/fused_silu_precise.so'] = package['candidate_entries']['src/fused_silu_precise.so']['sha256']
    bind.update({
        '/usr/src/tensorrt/bin/trtexec': '061419fab4d55d124c41868d06213b63316c84b9b165e06fdcc4e08604a19b90',
        '/usr/src/tensorrt/samples/common/sampleInference.cpp': '519ab04ce679fb866a5c574101e0d6fb4ec11a965cedab88267ca745506fd5a8',
        '/usr/src/tensorrt/samples/common/sampleDevice.h': 'b4609258199ec81d4791c669e450ddfa93c6a73cfe47e574bd1a9f976d53d1c6',
        REMOTE + 'sampleInference.cpp': '519ab04ce679fb866a5c574101e0d6fb4ec11a965cedab88267ca745506fd5a8',
        REMOTE + 'sampleDevice.h': 'b4609258199ec81d4791c669e450ddfa93c6a73cfe47e574bd1a9f976d53d1c6',
    })
    for name in ('00_protocol.md', 'screen.py'):
        bind[REMOTE + name] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    target = HERE / 'bindings.json'
    assert not target.exists()
    target.write_text(json.dumps(bind, indent=2) + '\n', encoding='utf-8')
    print('R174 INPUTS BOUND', len(bind))


if __name__ == '__main__':
    main()
