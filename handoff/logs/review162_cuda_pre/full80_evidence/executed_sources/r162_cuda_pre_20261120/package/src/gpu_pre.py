"""Nano Python3.6: uint8 BGR upload and one CUDA resize/RGB/CHW/LUT kernel."""
from pathlib import Path
import ctypes as ct
import time
import numpy as np
from nano_detector import TRTEngine

HERE=Path(__file__).resolve().parent

def axis(source,target,horizontal):
    scale=1./(float(target)/source)
    position=((np.arange(target,dtype=np.float64)+.5)*scale-.5).astype(np.float32)
    index=np.floor(position).astype(np.int32)
    fraction=position-index.astype(np.float32)
    if horizontal:
        border=(index<0)|(index>=source-1)
        fraction[border]=0.;index=np.clip(index,0,source-1).astype(np.int32)
    coefficients=np.column_stack((np.rint((np.float32(1)-fraction)*np.float32(2048)),
                                   np.rint(fraction*np.float32(2048)))).astype(np.int16)
    return np.ascontiguousarray(index),np.ascontiguousarray(coefficients)

class CudaPreprocessor:
    def __init__(self,runtime,destination,hw):
        self.runtime=runtime;self.destination=destination;self.hw=tuple(hw);self.cache={}
        self.source=ct.c_void_p();self.capacity=0;self.allocated=[]
        runtime.cudaMalloc.argtypes=[ct.POINTER(ct.c_void_p),ct.c_size_t]
        runtime.cudaMemcpy.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_size_t,ct.c_int]
        runtime.cudaFree.argtypes=[ct.c_void_p]
        self.library=ct.CDLL(str(HERE/'fused_pre.so'))
        self.library.fused_preprocess.argtypes=[ct.c_void_p,ct.c_void_p]+[ct.c_int]*8+[ct.c_void_p]*5+[ct.c_int]
        self.library.fused_preprocess.restype=ct.c_int
        lut=np.arange(256,dtype=np.float32)/np.float32(255)
        self.lut=self.upload(lut)

    @staticmethod
    def check(result,label):
        if result:raise RuntimeError('%s CUDA error %s'%(label,result))

    def upload(self,array):
        array=np.ascontiguousarray(array);pointer=ct.c_void_p()
        self.check(self.runtime.cudaMalloc(ct.byref(pointer),array.nbytes),'allocate preprocessing table')
        self.allocated.append(pointer)
        self.check(self.runtime.cudaMemcpy(pointer,array.ctypes.data_as(ct.c_void_p),array.nbytes,1),'upload preprocessing table')
        return pointer

    def parameters(self,shape):
        h,w=shape
        if shape not in self.cache:
            oh,ow=self.hw;r=min(oh/float(h),ow/float(w));rw,rh=int(round(w*r)),int(round(h*r))
            left=int(round((ow-rw)/2.-.1));top=int(round((oh-rh)/2.-.1))
            xs,ax=axis(w,rw,True);ys,ay=axis(h,rh,False)
            self.cache[shape]=(rw,rh,left,top,self.upload(xs),self.upload(ys),self.upload(ax),self.upload(ay),int(w==2*rw and h==2*rh))
        return self.cache[shape]

    def run(self,frame):
        assert frame.dtype==np.uint8 and frame.ndim==3 and frame.shape[2]==3
        frame=np.ascontiguousarray(frame);h,w=frame.shape[:2];oh,ow=self.hw
        rw,rh,left,top,xs,ys,ax,ay,half=self.parameters((h,w))
        if frame.nbytes>self.capacity:
            if self.capacity:self.check(self.runtime.cudaFree(self.source),'free prior raw frame')
            self.check(self.runtime.cudaMalloc(ct.byref(self.source),frame.nbytes),'allocate raw frame')
            self.capacity=frame.nbytes
        start=time.perf_counter()
        self.check(self.runtime.cudaMemcpy(self.source,frame.ctypes.data_as(ct.c_void_p),frame.nbytes,1),'upload BGR')
        uploaded=time.perf_counter()
        self.check(self.library.fused_preprocess(self.source,self.destination,w,h,ow,oh,rw,rh,left,top,
                   xs,ys,ax,ay,self.lut,half),'launch preprocessing kernel')
        self.check(self.runtime.cudaDeviceSynchronize(),'synchronize preprocessing kernel')
        finished=time.perf_counter()
        return {'h2d_ms':1000*(uploaded-start),'gpu_pre_ms':1000*(finished-uploaded)}

class GpuTRTEngine(TRTEngine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        assert self.bo_dem=='cudart' and self.in_dtype==np.dtype('float32')
        self.preprocessor=CudaPreprocessor(self.cudart,self.d_in,self.hw)

    def tensor(self,frame):
        self.preprocessor.run(frame);output=np.empty(self.in_shape,dtype=np.float32)
        self._chk(self.cudart.cudaMemcpy(output.ctypes.data_as(ct.c_void_p),self.d_in,output.nbytes,2),'read oracle input')
        return output

    def __call__(self,x):
        if x.dtype!=np.uint8 or x.ndim!=3:return super().__call__(x)
        times=self.preprocessor.run(x);start=time.perf_counter()
        ok=self.ctx.execute_v2(bindings=self.bindings);executed=time.perf_counter()
        self._chk(self.cudart.cudaMemcpy(self.h_out.ctypes.data_as(ct.c_void_p),self.d_out,self.h_out.nbytes,2),'read detection output')
        finished=time.perf_counter()
        if not ok:raise RuntimeError('TensorRT execution failed after GPU preprocessing')
        times.update(exec_ms=1000*(executed-start),d2h_ms=1000*(finished-executed));self.t_last=times
        return self.h_out.reshape(self.out_shape).astype(np.float32,copy=False)

    def info(self):
        result=super().info();result['preprocessing']='CUDA integer bilinear + BGR/RGB + CHW + existing LUT'
        return result
