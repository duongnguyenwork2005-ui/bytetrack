"""Replace only singly-used x*sigmoid(x) pairs, preserving every initializer."""
from pathlib import Path
from collections import Counter
import hashlib,json,shutil
import onnx
from onnx import helper
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    target=HERE/'transform.json';assert not target.exists();source=HERE/'anchor.onnx'
    assert sha(source)=='a0ae9b42a0965876004eda1e56d4a4ccd801bb1a9cecd35051f47d5c8a794e48'
    graph=onnx.load(source);onnx.checker.check_model(graph)
    initial={p.name:p.SerializeToString() for p in graph.graph.initializer}
    producers={name:node for node in graph.graph.node for name in node.output}
    uses=Counter(name for node in graph.graph.node for name in node.input);replacements={};removed=set()
    pairs=[]
    for node in graph.graph.node:
        if node.op_type!='Mul' or len(node.input)!=2:continue
        for activation,x in ((node.input[0],node.input[1]),(node.input[1],node.input[0])):
            sigmoid=producers.get(activation)
            if sigmoid is not None and sigmoid.op_type=='Sigmoid' and list(sigmoid.input)==[x] and uses[activation]==1:
                replacement=helper.make_node('FusedSiluRounded',[x],list(node.output),name='r167_'+node.name,
                    domain='org.bytetrack.r167',plugin_version='1',plugin_namespace='')
                replacements[id(node)]=replacement;removed.add(activation)
                pairs.append({'sigmoid':sigmoid.name,'mul':node.name,'input':x,'output':list(node.output)})
                break
    assert len(pairs)==77,len(pairs)
    nodes=[]
    for node in graph.graph.node:
        if any(name in removed for name in node.output):continue
        nodes.append(replacements.get(id(node),node))
    # Repeated protobuf iteration may produce new Python wrappers: use names to
    # bind replacements, rather than relying on object identity across iterators.
    by_name={row['mul']:helper.make_node('FusedSiluRounded',[row['input']],row['output'],name='r167_'+row['mul'],
        domain='org.bytetrack.r167',plugin_version='1',plugin_namespace='') for row in pairs}
    nodes=[by_name.get(node.name,node) for node in graph.graph.node if not any(name in removed for name in node.output)]
    del graph.graph.node[:];graph.graph.node.extend(nodes)
    kept=[row for row in graph.graph.value_info if row.name not in removed]
    del graph.graph.value_info[:];graph.graph.value_info.extend(kept)
    graph.opset_import.extend([helper.make_opsetid('org.bytetrack.r167',1)])
    assert initial=={p.name:p.SerializeToString() for p in graph.graph.initializer}
    assert sum(node.op_type=='FusedSiluRounded' for node in graph.graph.node)==77
    assert sum(node.op_type=='Sigmoid' for node in graph.graph.node)==1
    onnx.checker.check_model(graph);folder=ROOT/'models/r167_fused_silu';folder.mkdir(exist_ok=False)
    fused=folder/'fused.onnx';onnx.save(graph,fused)
    parity=HERE/'parity';parity.mkdir(exist_ok=False)
    original=HERE.parent/'review151_resolution_tradeoff/exports/baseline/parity'
    for i in range(10):shutil.copyfile(original/('%02d.npz'%i),parity/('%02d.npz'%i))
    target.write_text(json.dumps({'status':'PASS','original_sha256':sha(source),'fused':str(fused),'fused_sha256':sha(fused),
        'initializers_byte_equal':True,'replaced_pairs':pairs,'remaining_class_sigmoid':1,
        'parity_sha256':{str(p):sha(p) for p in parity.glob('*.npz')},
        'source_sha256':{str(p):sha(p) for p in (Path(__file__),HERE/'00_protocol.md',HERE/'fused_silu.cu',HERE/'kernel_oracle.py')}},indent=2)+'\n')
    print('R167 GRAPH TRANSFORM PASS',len(pairs),'initializers byte-equal',flush=True)
if __name__=='__main__':main()
