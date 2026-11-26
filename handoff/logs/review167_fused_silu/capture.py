"""Load the bound SiLU plugins, reuse the immutable CUDA-preprocessing capture."""
from pathlib import Path
import ctypes,hashlib,importlib.util,json,sys
HERE=Path(__file__).resolve().parent;PARENT=HERE.parent/'r162_cuda_pre_20261120'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    config=json.loads(Path(sys.argv[1]).read_text())
    for p,h in config['source_sha256'].items():assert sha(p)==h,p
    handles=[ctypes.CDLL(path,mode=ctypes.RTLD_GLOBAL) for path in config['libraries']]
    sys.path.insert(0,str(PARENT))
    spec=importlib.util.spec_from_file_location('r167_immutable_actual_capture',PARENT/'capture.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base);base.main()
    for p,h in config['source_sha256'].items():assert sha(p)==h,p
if __name__=='__main__':main()
