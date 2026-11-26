"""Read existing engine layer metadata only; no execution context or inference."""
from pathlib import Path
import ctypes
import json
import sys
import tensorrt as trt

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'r167_fused_silu_20261121'
sys.path.insert(0, str(HERE.parent / 'r164_latency_pruning_20261121'))
from build_screen import sha, state


def main():
    target = HERE / 'layer_inspection.json'
    assert not target.exists()
    assert not (HERE.parent / 'r172_common7_pruning_20261121/recovered/before.json').exists()
    config = json.loads((PARENT / 'runtime_models.json').read_text())
    handles = [ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL) for path in config['libraries']]
    records = {'original': config['original']}
    records.update(config['models'])
    before = state()
    results = {}
    for mode, model in records.items():
        assert sha(model['engine']) == model['engine_sha256']
        with trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
            engine = runtime.deserialize_cuda_engine(Path(model['engine']).read_bytes())
            assert engine is not None
            if not hasattr(engine, 'create_engine_inspector'):
                results[mode] = {'available': False}
            else:
                inspector = engine.create_engine_inspector()
                text = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
                path = HERE / (mode + '_layers.json')
                path.write_text(text)
                results[mode] = {'available': True, 'engine_sha256': model['engine_sha256'],
                                 'path': str(path), 'sha256': sha(path),
                                 'profiling_verbosity': str(engine.profiling_verbosity)}
                del inspector
            del engine
    after = state()
    assert before['boot'] == after['boot'] and before['SD_errors'] == after['SD_errors'] == '0'
    target.write_text(json.dumps({'status': 'PASS', 'models': results, 'before': before, 'after': after,
                      'inference_executed': False, 'new_engines_built': False,
                      'source_sha256': sha(Path(__file__))}, indent=2) + '\n')
    print('R173 STATIC ENGINE INSPECTION COMPLETE', results, flush=True)


if __name__ == '__main__':
    main()
