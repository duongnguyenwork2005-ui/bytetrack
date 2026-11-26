"""Nano TRT8.2 mixed precision: FP32 C2PSA only, unchanged ONNX and workspace."""
from pathlib import Path
import hashlib
import json
import argparse
import tensorrt as trt

ROOT=Path.home()/'nano_run/r151_resolution_20261120'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('r352x608','r320x576'));args=parser.parse_args()
    onnx=ROOT/'models'/(args.mode+'.onnx');output=ROOT/'models'/(args.mode+'_psa32.engine')
    assert not output.exists()
    expected=json.loads((ROOT/'models.json').read_text())['models'][args.mode]['onnx_sha256']
    assert sha(onnx)==expected
    logger=trt.Logger(trt.Logger.VERBOSE);builder=trt.Builder(logger)
    network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(onnx.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config();config.max_workspace_size=512*1024*1024
    config.set_flag(trt.BuilderFlag.FP16);config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    types={trt.LayerType.CONVOLUTION,trt.LayerType.ACTIVATION,trt.LayerType.ELEMENTWISE,
           trt.LayerType.POOLING,trt.LayerType.SOFTMAX,trt.LayerType.MATRIX_MULTIPLY,
           trt.LayerType.REDUCE,trt.LayerType.UNARY}
    changed=[]
    for index in range(network.num_layers):
        layer=network.get_layer(index)
        if not layer.name.startswith('/model.10/') or layer.type not in types:continue
        layer.precision=trt.float32;outputs=[]
        for k in range(layer.num_outputs):
            tensor=layer.get_output(k)
            if tensor.dtype in (trt.float32,trt.float16):
                layer.set_output_type(k,trt.float32);outputs.append(k)
        changed.append({'name':layer.name,'type':str(layer.type),'output_indices_FP32':outputs})
    assert changed and any('Softmax' in row['name'] for row in changed)
    record={'tensorrt':trt.__version__,'onnx_sha256':expected,'code_sha256':sha(Path(__file__)),
            'workspace_bytes':512*1024*1024,'FP16_enabled':True,'constraint':'OBEY','FP32_layers':changed}
    inputs=ROOT/(args.mode+'_psa32_build_inputs.json');inputs.write_text(json.dumps(record,indent=2))
    serialized=builder.build_serialized_network(network,config)
    assert serialized is not None
    output.write_bytes(bytes(serialized));assert sha(onnx)==expected
    (ROOT/(args.mode+'_psa32_build_complete.json')).write_text(json.dumps({'status':'PASS','engine':str(output),
          'engine_sha256':sha(output),'inputs_sha256':sha(inputs)},indent=2))
    print('PSA32 BUILD COMPLETE',args.mode,flush=True)


if __name__=='__main__':main()
