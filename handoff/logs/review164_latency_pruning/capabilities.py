"""Native Nano capability evidence, not an INT8 FPS claim."""
from pathlib import Path
import ctypes as ct
import hashlib,json
import tensorrt as trt

HERE=Path(__file__).resolve().parent
def main():
    target=HERE/'capabilities.json';assert not target.exists()
    runtime=ct.CDLL('/usr/local/cuda/lib64/libcudart.so')
    major=ct.c_int();minor=ct.c_int();version=ct.c_int()
    assert runtime.cudaDeviceGetAttribute(ct.byref(major),75,0)==0
    assert runtime.cudaDeviceGetAttribute(ct.byref(minor),76,0)==0
    assert runtime.cudaRuntimeGetVersion(ct.byref(version))==0
    with trt.Builder(trt.Logger(trt.Logger.WARNING)) as builder:
        result={'TensorRT':trt.__version__,'compute_capability':[major.value,minor.value],
            'CUDA_runtime':version.value,'fast_fp16':bool(builder.platform_has_fast_fp16),
            'fast_int8':bool(builder.platform_has_fast_int8),
            'board':Path('/proc/device-tree/model').read_bytes().rstrip(b'\0').decode(),
            'boot':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'INT8_FPS_measured':False,'INT8_supported_by_NVIDIA_for_SM53':False,
            'precision_source':'https://forums.developer.nvidia.com/t/onnx-tensorrt-engines-fp16-32/346691/7',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    assert result['compute_capability']==[5,3]
    target.write_text(json.dumps(result,indent=2)+'\n')
    print('R164_NANO_CAPABILITY',json.dumps(result),flush=True)
if __name__=='__main__':main()
