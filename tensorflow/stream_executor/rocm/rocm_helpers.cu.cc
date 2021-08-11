#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <limits>
#include "fp8/hip_float8.h"
namespace stream_executor {
namespace gpu {

__global__ void broadcast_fp32_kernel(float* dst, int dst_stride, int batches,
                                      float* src, int size) {
  dst += blockIdx.y * 4 * dst_stride + blockIdx.z * dst_stride * batches;
  src += blockIdx.z * size;
  float* dst2 = dst + dst_stride;
  float* dst3 = dst + dst_stride*2;
  float* dst4 = dst + dst_stride*3;
  bool b2 = (blockIdx.y*4+1 < batches);
  bool b3 = (blockIdx.y*4+2 < batches);
  bool b4 = (blockIdx.y*4+3 < batches);
  for (int i = threadIdx.x + blockIdx.x * 256; i < size; i += blockDim.x*gridDim.x) 
  {
  	dst[i] = src[i];
  	if(b2)
  		dst2[i] = src[i];
  	if(b3)
  		dst3[i] = src[i];
  	if(b4)
  		dst4[i] = src[i];
  }
}

void broadcast_fp32(void* stream, float* dst, int dst_stride, int batches, int src_batches,
                    float* src, int size) {
  int x_blocks = (size+255)/256;
  hipLaunchKernelGGL(broadcast_fp32_kernel, dim3(x_blocks, (batches+3)/4, src_batches), min(256, (int)size), 0,
                     (hipStream_t)stream, dst, dst_stride, batches, src, size);
}

template <typename T>
__global__ void Quant8_52_inplace(T* _p, int32_t count, bool stoch, uint32_t seed) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= count) return;
  typedef typename std::conditional<sizeof(T)==2, uint16_t, uint32_t>::type IT;
  typedef typename std::conditional<sizeof(T)==2, __half, float>::type FT;
  IT* p = (IT*) _p;
  FT* fp = (FT*) _p;
  IT x = p[i];
  const int we=5, wm=2;

  uint8_t y;
  if(!stoch)
    y = hip_f8_impl::cast_to_f8<wm,we,FT,false,true>(fp[i]);
  else {
    uint32_t drop_bits = uint32_t(x) & 0xFFFFu;
    if(sizeof(x)==4)
      drop_bits ^= x>>16;
    drop_bits = ((drop_bits & 31)<<11) | (drop_bits>>5);
    drop_bits *= 0x7000149;
    uint32_t rng = (drop_bits ^ 0x13371337 ^ (i*229791) ^ seed);
    y = hip_f8_impl::cast_to_f8<wm,we,FT,false,true,true>(fp[i], rng);
  }
  fp[i] = hip_f8_impl::cast_from_f8<wm,we,FT,false>(y);
}

template <typename T>
__global__ void Quant8_43_inplace(T* _p, int32_t count, bool stoch, uint32_t seed) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= count) return;
  typedef typename std::conditional<sizeof(T)==2, uint16_t, uint32_t>::type IT;
  typedef typename std::conditional<sizeof(T)==2, __half, float>::type FT;
  IT* p = (IT*) _p;
  FT* fp = (FT*) _p;
  IT x = p[i];
  const int we=4, wm=3;

  uint8_t y;
  if(!stoch)
    y = hip_f8_impl::cast_to_f8<wm,we,FT,false,true>(fp[i]);
  else {
    uint32_t drop_bits = uint32_t(x) & 0xFFFFu;
    if(sizeof(x)==4)
      drop_bits ^= x>>16;
    drop_bits = ((drop_bits & 31)<<11) | (drop_bits>>5);
    drop_bits += i;
    drop_bits *= 0x7000149;
    uint32_t rng = (drop_bits ^ 0x13371337 ^ (i*229791) ^ seed);
    y = hip_f8_impl::cast_to_f8<wm,we,FT,false,true,true>(fp[i], rng);
  }
  fp[i] = hip_f8_impl::cast_from_f8<wm,we,FT,false>(y);
}

template __global__ void Quant8_52_inplace<__half>(__half* _p, int32_t count, bool stoch, uint32_t seed);
template __global__ void Quant8_43_inplace<__half>(__half* _p, int32_t count, bool stoch, uint32_t seed);

void Quant8_inplace(__half* _p, int32_t count, uint32_t seed, hipStream_t stream, bool f152) {
        auto fun = f152 ? Quant8_52_inplace<__half> : Quant8_43_inplace<__half>;
        uint32_t dim_a = count; 
        uint32_t grid_a = (dim_a+255)/256;
        hipLaunchKernelGGL(fun,
           dim3(grid_a,1,1), dim3(256,1,1), 0, stream, _p, dim_a, true, seed);
}

};  // namespace gpu
};  // namespace stream_executor

