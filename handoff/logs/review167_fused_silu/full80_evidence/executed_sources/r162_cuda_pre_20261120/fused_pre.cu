// Original CUDA implementation of the existing uint8 bilinear/LUT pipeline.
// Integer interpolation conventions cross-checked against OpenCV 3.2.0
// modules/imgproc/src/imgwarp.cpp; no inference or geometry change.
#include <cuda_runtime.h>

__global__ void preprocess_kernel(const unsigned char* src, float* dst,
    int iw,int ih,int ow,int oh,int rw,int rh,int left,int top,
    const int* xs,const int* ys,const short* ax,const short* ay,
    const float* lut,int half_scale) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;
    if(p>=ow*oh)return;
    int x=p%ow-left,y=p/ow-top;
    if(x<0 || y<0 || x>=rw || y>=rh) {
        dst[p]=lut[114];dst[p+ow*oh]=lut[114];dst[p+2*ow*oh]=lut[114];return;
    }
    int sx=max(0,min(iw-1,xs[x])),sy=max(0,min(ih-1,ys[y]));
    int tx=min(sx+1,iw-1);
    // y coefficients retain their fractional border coordinate; both source
    // row indices are clamped separately, as in separable resize.
    int ty=max(0,min(ih-1,ys[y]+1));
    int a0=ax[2*x],a1=ax[2*x+1],b0=ay[2*y],b1=ay[2*y+1];
    #pragma unroll
    for(int c=0;c<3;++c) {
        int k=2-c;
        int v00=src[(sy*iw+sx)*3+k],v01=src[(sy*iw+tx)*3+k];
        int v10=src[(ty*iw+sx)*3+k],v11=src[(ty*iw+tx)*3+k];
        int value;
        if(half_scale)value=(v00+v01+v10+v11+2)>>2;
        else {
            int horizontal0=v00*a0+v01*a1;
            int horizontal1=v10*a0+v11*a1;
            int term0=((horizontal0>>4)*b0)>>16;
            int term1=((horizontal1>>4)*b1)>>16;
            value=(term0+term1+2)>>2;
        }
        dst[p+c*ow*oh]=lut[max(0,min(255,value))];
    }
}

extern "C" int fused_preprocess(const unsigned char* src,float* dst,
    int iw,int ih,int ow,int oh,int rw,int rh,int left,int top,
    const int* xs,const int* ys,const short* ax,const short* ay,const float* lut,int half_scale) {
    preprocess_kernel<<<(ow*oh+255)/256,256>>>(src,dst,iw,ih,ow,oh,rw,rh,left,top,xs,ys,ax,ay,lut,half_scale);
    return (int)cudaGetLastError();
}
