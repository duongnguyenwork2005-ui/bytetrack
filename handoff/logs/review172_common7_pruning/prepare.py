"""Bind the last common uninterrupted checkpoints; no new training or epoch search."""
from pathlib import Path
import csv
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent
PARENT = LOGS / 'review164_latency_pruning'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    target = HERE / 'training_inputs.json'
    assert not target.exists()
    parent = read(PARENT / 'training_inputs.json')
    bound = [Path(__file__), HERE / '00_protocol.md', HERE / 'evaluate.py', HERE / 'evaluate.ps1',
             HERE / 'export_recovered.py', PARENT / 'training_inputs.json',
             LOGS / 'review169_paired_cuda_resume/monitor.py']
    for mode in ('control', 'joint50'):
        for backend in ('cudnn', 'nativeBN'):
            path = LOGS / 'review171_cuda_batch_diagnosis' / (mode + '_' + backend + '.json')
            diagnostic = read(path)
            assert diagnostic['status'] == 'PASS' and diagnostic['steps_completed'] == 64
            assert diagnostic['optimizer_steps'] == diagnostic['checkpoints_written'] == 0
            bound.append(path)
    for path, digest in parent['sha256'].items():
        assert sha(path) == digest, path
    data = HERE / 'data'
    data.mkdir(exist_ok=False)
    shutil.copyfile(PARENT / 'data/dev300.txt', data / 'dev300.txt')
    proof = {}
    for mode in ('control', 'joint50'):
        source = PARENT / 'runs' / mode
        checkpoint = source / 'weights/epoch6.pt'
        shapes = read(source / 'batch_shapes.json')
        assert all(shapes[str(i)] == {'4x3x384x640': 1300} for i in range(7))
        with (source / 'results.csv').open(newline='') as stream:
            history = [row for row in csv.DictReader(stream) if int(row['epoch']) <= 7]
        assert [int(row['epoch']) for row in history] == list(range(1, 8))
        destination = HERE / 'checkpoints' / (mode + '_epoch6.pt')
        destination.parent.mkdir(exist_ok=True)
        assert not destination.exists()
        shutil.copyfile(checkpoint, destination)
        assert sha(destination) == sha(checkpoint)
        selection = {'status': 'PASS', 'rule': 'Last common completed uninterrupted R164 checkpoint; no AP search',
                     'selected': {'checkpoint': str(destination), 'checkpoint_sha256': sha(destination),
                                  'epoch_zero_based': 6},
                     'accepted_epochs': 7, 'parent_primary10_complete': False,
                     'opened_dev7190': False}
        selection_path = HERE / 'monitor' / mode / 'selection.json'
        write(selection_path, selection)
        proof[mode] = {'accepted_epochs': 7, 'parent_checkpoint': str(checkpoint),
                       'parent_checkpoint_sha256': sha(checkpoint), 'copied_checkpoint': str(destination),
                       'history_first7': history, 'batch_shapes_first7': {str(i): shapes[str(i)] for i in range(7)}}
        bound.extend((checkpoint, destination, selection_path, source / 'results.csv', source / 'batch_shapes.json'))
    proof_path = HERE / 'common_checkpoint_proof.json'
    write(proof_path, {'status': 'PASS', 'meaning': 'Seven-epoch completed-checkpoint integrity only',
                      'primary10_complete': False, 'new_training_performed': False, 'models': proof})
    bound.extend((proof_path, data / 'dev300.txt'))
    hashes = parent['sha256'].copy()
    hashes.update({str(path): sha(path) for path in bound})
    write(target, {'status': 'PASS', 'models': parent['models'], 'sha256': hashes,
                   'accepted_common_epochs': 7, 'planned_parent_epochs': 10,
                   'primary10_complete': False, 'secondary_analysis': True,
                   'checkpoint_rule': 'Fixed epoch6 of both uninterrupted R164 arms; no AP selection',
                   'seed': 164, 'rect': True, 'new_training_performed': False})
    print('R172 COMMON7 CHECKPOINTS VERIFIED; NOT A COMPLETE 10-EPOCH STUDY', flush=True)


if __name__ == '__main__':
    main()
