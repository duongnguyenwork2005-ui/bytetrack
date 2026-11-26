#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <string>
#include <algorithm>

using namespace nvinfer1;
namespace {
__device__ float sigmoid32(float x) { return 1.0f / (1.0f + expf(-x)); }
__global__ void silu32(float const* x, float* y, size_t n) {
    for (size_t i=blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=size_t(blockDim.x)*gridDim.x)
        y[i]=x[i]*sigmoid32(x[i]);
}
__global__ void silu16(__half const* x,__half* y,size_t n) {
    size_t tid=blockIdx.x*blockDim.x+threadIdx.x;
    auto input=reinterpret_cast<__half2 const*>(x);auto output=reinterpret_cast<__half2*>(y);
    for (size_t i=tid;i<n/2;i+=size_t(blockDim.x)*gridDim.x) {
        __half2 value=input[i];float2 a=__half22float2(value);
        __half2 rounded=__floats2half2_rn(sigmoid32(a.x),sigmoid32(a.y));
        output[i]=__hmul2(value,rounded);
    }
    if (tid==0 && n%2) y[n-1]=__hmul(x[n-1],__float2half_rn(sigmoid32(__half2float(x[n-1]))));
}
int launch(void const* input,void* output,size_t n,bool half,cudaStream_t stream) {
    if (!n) return 0;
    unsigned blocks=static_cast<unsigned>(std::min<size_t>(1024,(n+255)/256));
    if (half) silu16<<<blocks,256,0,stream>>>(static_cast<__half const*>(input),static_cast<__half*>(output),n);
    else silu32<<<blocks,256,0,stream>>>(static_cast<float const*>(input),static_cast<float*>(output),n);
    return static_cast<int>(cudaGetLastError());
}
int vectorWidth(TensorFormat format) {
    switch(format) {
        case TensorFormat::kLINEAR:return 1;
        case TensorFormat::kCHW2:return 2;
        case TensorFormat::kCHW4:return 4;
        case TensorFormat::kCHW16:return 16;
        case TensorFormat::kCHW32:return 32;
        case TensorFormat::kHWC8:return 8;
        default:return 0;
    }
}
class FusedSiluRounded final:public IPluginV2DynamicExt {
    std::string ns_;
public:
    char const* getPluginType() const noexcept override{return "FusedSiluRounded";}
    char const* getPluginVersion() const noexcept override{return "1";}
    int32_t getNbOutputs() const noexcept override{return 1;}
    int32_t initialize() noexcept override{return 0;}
    void terminate() noexcept override{}
    size_t getSerializationSize() const noexcept override{return 0;}
    void serialize(void*) const noexcept override{}
    void destroy() noexcept override{delete this;}
    void setPluginNamespace(char const* ns) noexcept override{ns_=ns?ns:"";}
    char const* getPluginNamespace() const noexcept override{return ns_.c_str();}
    FusedSiluRounded* clone() const noexcept override{
        auto p=new FusedSiluRounded();p->setPluginNamespace(ns_.c_str());return p;
    }
    DataType getOutputDataType(int32_t,DataType const* input,int32_t) const noexcept override{return input[0];}
    DimsExprs getOutputDimensions(int32_t,DimsExprs const* input,int32_t,IExprBuilder&) noexcept override{return input[0];}
    bool supportsFormatCombination(int32_t pos,PluginTensorDesc const* io,int32_t ni,int32_t no) noexcept override{
        if(ni!=1||no!=1||pos<0||pos>1||io[pos].dims.nbDims!=4)return false;
        if(pos==1)return io[1].type==io[0].type && io[1].format==io[0].format;
        return (io[0].type==DataType::kFLOAT && io[0].format==TensorFormat::kLINEAR)
            ||(io[0].type==DataType::kHALF && vectorWidth(io[0].format)>0);
    }
    void configurePlugin(DynamicPluginTensorDesc const*,int32_t,DynamicPluginTensorDesc const*,int32_t) noexcept override{}
    size_t getWorkspaceSize(PluginTensorDesc const*,int32_t,PluginTensorDesc const*,int32_t) const noexcept override{return 0;}
    int32_t enqueue(PluginTensorDesc const* in,PluginTensorDesc const*,void const* const* inputs,
        void* const* outputs,void*,cudaStream_t stream) noexcept override{
        int width=vectorWidth(in[0].format);if(!width||in[0].dims.nbDims!=4)return 1;
        size_t n=1;
        for(int i=0;i<4;++i){int d=in[0].dims.d[i];if(d<0)return 1;if(i==1)d=((d+width-1)/width)*width;n*=size_t(d);}
        if(in[0].type!=DataType::kFLOAT && in[0].type!=DataType::kHALF)return 1;
        return launch(inputs[0],outputs[0],n,in[0].type==DataType::kHALF,stream);
    }
};
class FusedSiluCreator final:public IPluginCreator {
    PluginFieldCollection fields_{0,nullptr};std::string ns_;
public:
    char const* getPluginName() const noexcept override{return "FusedSiluRounded";}
    char const* getPluginVersion() const noexcept override{return "1";}
    PluginFieldCollection const* getFieldNames() noexcept override{return &fields_;}
    IPluginV2* createPlugin(char const*,PluginFieldCollection const* fields) noexcept override{
        if(fields && fields->nbFields!=0)return nullptr;
        auto p=new FusedSiluRounded();p->setPluginNamespace(ns_.c_str());return p;
    }
    IPluginV2* deserializePlugin(char const*,void const*,size_t length) noexcept override{
        if(length!=0)return nullptr;auto p=new FusedSiluRounded();p->setPluginNamespace(ns_.c_str());return p;
    }
    void setPluginNamespace(char const* ns) noexcept override{ns_=ns?ns:"";}
    char const* getPluginNamespace() const noexcept override{return ns_.c_str();}
};
REGISTER_TENSORRT_PLUGIN(FusedSiluCreator);
}

// Numerical oracle entry point. Uses the same kernels as enqueue, releases all buffers.
extern "C" int r167_check_kernel(void const* input,void* output,size_t n,int half) {
    void* a=nullptr;void* b=nullptr;size_t bytes=n*(half?2:4);int result=0;
    auto run=[&]()->int {
        cudaError_t e=cudaMalloc(&a,bytes);if(e!=cudaSuccess)return int(e);
        e=cudaMalloc(&b,bytes);if(e!=cudaSuccess)return int(e);
        e=cudaMemcpy(a,input,bytes,cudaMemcpyHostToDevice);if(e!=cudaSuccess)return int(e);
        int status=launch(a,b,n,half!=0,nullptr);if(status)return status;
        e=cudaMemcpy(output,b,bytes,cudaMemcpyDeviceToHost);return int(e);
    };
    result=run();if(a)cudaFree(a);if(b)cudaFree(b);return result;
}
