"""Captured-batch KD diagnostic; never steps an optimizer or saves a checkpoint."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['YOLO_CONFIG_DIR'] = str(HERE / 'settings')
sys.path.insert(0, str(LOGS / 'review104_yolo_compression'))
import torch
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils.torch_utils import init_seeds
from fit_channels import train_with_distillation

torch.set_num_threads(2)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    mode, backend = sys.argv[1:]
    assert mode in ('control', 'joint50') and backend in ('cudnn', 'nativeBN')
    target = HERE / (mode + '_' + backend + '.json')
    assert not target.exists()
    checkpoint = (LOGS / 'review169_paired_cuda_resume/runs/control/weights/epoch8.pt'
                  if mode == 'control' else LOGS / 'review164_latency_pruning/runs/joint50/weights/epoch6.pt')
    batch_path = LOGS / 'review169_paired_cuda_resume/control_failed_cpu_batch.pt'
    source = torch.load(batch_path, map_location='cpu', weights_only=False)
    assert source['img'].shape == (4, 3, 384, 640) and source['img'].dtype == torch.uint8
    assert all(torch.isfinite(value).all() for value in source.values() if isinstance(value, torch.Tensor))
    assert source['bboxes'].shape[1] == 4 and ((source['bboxes'] >= 0) & (source['bboxes'] <= 1)).all()
    assert ((source['cls'] >= 0) & (source['cls'] < 5)).all()
    init_seeds(164, deterministic=True)
    model = YOLO(str(checkpoint), task='detect').model.to('cuda:0').float().train()
    model.args = get_cfg()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_('.dfl.' not in name)
    batch = {key: value.to('cuda:0') if isinstance(value, torch.Tensor) else value for key, value in source.items()}
    batch['img'] = batch['img'].float() / 255
    original = torch.nn.modules.batchnorm._BatchNorm.forward
    if backend == 'nativeBN':
        def native_forward(module, value):
            with torch.backends.cudnn.flags(enabled=False):
                return original(module, value)
        torch.nn.modules.batchnorm._BatchNorm.forward = native_forward
    nodes = Counter()
    hooks = []
    def record(module, inputs, output):
        if output.grad_fn is not None:
            nodes[type(output.grad_fn).__name__] += 1
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            hooks.append(module.register_forward_hook(record))
    result = {'status': 'RUNNING', 'model': mode, 'BN_backend': backend, 'steps_completed': 0,
              'optimizer_steps': 0, 'checkpoints_written': 0, 'root_cause_identified': False,
              'sha256': {str(path): sha(path) for path in (Path(__file__), HERE / '00_protocol.md', checkpoint, batch_path)}}
    started = time.monotonic()
    try:
        class Probe:
            def train(self, **kwargs):
                for tick in range(64):
                    model.zero_grad(set_to_none=True)
                    loss, _ = model.loss(batch)
                    assert torch.isfinite(loss).all()
                    loss.sum().backward()
                    torch.cuda.synchronize()
                    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                               if parameter.grad is not None)
                    if tick == 0:
                        if backend == 'nativeBN':
                            assert nodes and all('NativeBatchNorm' in key for key in nodes), nodes
                        else:
                            assert any('CudnnBatchNorm' in key for key in nodes), nodes
                        for hook in hooks:
                            hook.remove()
                    result['steps_completed'] = tick + 1
                    if (tick + 1) % 16 == 0:
                        print(mode, backend, tick + 1, loss.detach().cpu().tolist(), flush=True)
                result['final_loss'] = loss.detach().cpu().tolist()
        train_with_distillation(Probe(), {})
        result['status'] = 'PASS'
    except BaseException as error:
        result['status'] = 'FAIL'
        result['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        torch.nn.modules.batchnorm._BatchNorm.forward = original
        result['BN_grad_fn_counts_first_step'] = dict(nodes)
        result['elapsed_seconds'] = time.monotonic() - started
        target.write_text(json.dumps(result, indent=2) + '\n')
    print('R171 CAPTURED BATCH DIAGNOSTIC', mode, backend, result['status'], flush=True)


if __name__ == '__main__':
    main()
