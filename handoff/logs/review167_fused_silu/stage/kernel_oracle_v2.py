"""Native Python3.6: independent sigmoid rounding and odd-tail CUDA checks."""
from pathlib import Path
import ctypes as ct,hashlib,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    variant=sys.argv[1];assert variant in ('precise','fast');target=HERE/('kernel_oracle_'+variant+'.json');assert not target.exists()
    lib=ct.CDLL(str(HERE/('fused_silu_'+variant+'.so')),mode=ct.RTLD_GLOBAL)
    lib.r167_check_kernel.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int];lib.r167_check_kernel.restype=ct.c_int
    rng=np.random.RandomState(167);checks=[]
    for n in (1,3,17,1025,1048577):
        for half in (False,True):
            dtype=np.float16 if half else np.float32;x=rng.uniform(-20,20,n).astype(dtype)
            fixture=np.array([0.,-0.,1e-7,-1e-7,60.,-60.,.5,-.5,1.,-1.],dtype=dtype);x[:min(n,10)]=fixture[:min(n,10)]
            result=np.empty_like(x)
            status=lib.r167_check_kernel(x.ctypes.data,result.ctypes.data,n,int(half));assert status==0,status
            sigmoid=(1./(1.+np.exp(-x.astype(np.float64)))).astype(dtype);expected=(x*sigmoid).astype(dtype)
            assert np.isfinite(result).all()
            rtol,atol=(.002,1e-4) if half else (2e-6,1e-6)
            assert np.allclose(result,expected,rtol=rtol,atol=atol),(n,half,float(abs(result-expected).max()))
            checks.append({'size':n,'half':half,'max_error':float(abs(result.astype(float)-expected.astype(float)).max()),
                'rtol':rtol,'atol':atol,'bit_equal':bool(np.array_equal(result,expected))})
    target.write_text(json.dumps({'status':'PASS','checks':checks,'sha256':{str(p):sha(p) for p in (Path(__file__),HERE/('fused_silu_v2.cu' if variant=='precise' else 'fused_silu_fast.cu'),HERE/('fused_silu_'+variant+'.so'))}},indent=2)+'\n')
    print('R167 KERNEL ORACLE PASS',checks,flush=True)
if __name__=='__main__':main()
