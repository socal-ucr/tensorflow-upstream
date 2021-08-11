#pragma once
// FP8 header version 0.3, 2021/05/11

#define HIP_HOST_DEVICE __host__ __device__

enum class hip_f8_type {
  bf8 = 0, // 1:5:2
  fp8 = 1  // 1:4:3
};


enum class hip_f8_rounding_mode {
  standard,
  stochastic
};

template<hip_f8_type T>
struct hip_f8 {
  uint8_t data;

  // default constructor
  HIP_HOST_DEVICE hip_f8() = default;

  // constructor from bits
  HIP_HOST_DEVICE hip_f8(uint8_t v);

  // constructor from float
  HIP_HOST_DEVICE hip_f8(float v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // constructor from half
  HIP_HOST_DEVICE hip_f8(half v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // constructor from hip_bfloat16
  HIP_HOST_DEVICE hip_f8(hip_bfloat16 v, hip_f8_rounding_mode r=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // convert to float
  inline HIP_HOST_DEVICE operator float() const;

  // convert to half
  inline HIP_HOST_DEVICE operator half() const;

  // convert to hip_bfloat16
  inline HIP_HOST_DEVICE operator hip_bfloat16() const;
};

template<hip_f8_type T>
struct hip_f8x4 {
  // define some convenience types
  typedef float float32x2 __attribute__((ext_vector_type(2)));
  typedef float float32x4 __attribute__((ext_vector_type(4)));

  typedef _Float16 halfx2 __attribute__((ext_vector_type(2)));
  typedef _Float16 halfx4 __attribute__((ext_vector_type(4)));

  typedef uint16_t hip_bfloat16x2 __attribute__((ext_vector_type(2)));
  typedef uint16_t hip_bfloat16x4 __attribute__((ext_vector_type(4)));

  uint32_t data;

  // default constructor
  HIP_HOST_DEVICE hip_f8x4() = default;

  // constructor from bits
  HIP_HOST_DEVICE hip_f8x4(uint32_t v);

  // constructor from float
  HIP_HOST_DEVICE hip_f8x4(float v0, float v1=0, float v2=0, float v3=0, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(float32x2 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(float32x4 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // constructor from half
  HIP_HOST_DEVICE hip_f8x4(half v0, half v1=0, half v2=0, half v3=0, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(halfx2 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(halfx4 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // constructor from hip_bfloat16
  HIP_HOST_DEVICE hip_f8x4(hip_bfloat16 v0, hip_bfloat16 v1=hip_bfloat16(0.0f), hip_bfloat16 v2=hip_bfloat16(0.0f), hip_bfloat16 v3=hip_bfloat16(0.0f), hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(hip_bfloat16x2 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);
  HIP_HOST_DEVICE hip_f8x4(hip_bfloat16x4 v, hip_f8_rounding_mode rm=hip_f8_rounding_mode::standard, uint32_t rng=0);

  // convert to float32x4
  inline HIP_HOST_DEVICE operator float32x4() const;

  // convert to halfx4
  inline HIP_HOST_DEVICE operator halfx4() const;

  // convert to hip_bfloat16x4
  inline HIP_HOST_DEVICE operator hip_bfloat16x4() const;
};



template<hip_f8_type T>
struct hip_f8x8 {
  // define some convenience types
  typedef hip_f8x4<T>  f8x8 __attribute__((ext_vector_type(2)));

  f8x8 data;

  // default constructor
  HIP_HOST_DEVICE hip_f8x8() = default;

  // do we need to define other constructors or any conversion routines here?
};

// If we do not end up needing either any constructors or conversion routines for the above type, then
// we can simplify the above type to the following
#if USE_SIMPLER_HIP_F8x8
template <hip_f8_type T>
using hip_f8x8 = hip_f8x4<T> __attribute__((ext_vector_type(2)));
#endif

typedef float hip_float32x4  __attribute__((ext_vector_type(4)));
typedef float hip_float32x16 __attribute__((ext_vector_type(16)));

// these are device-specific and we don't expect them to exist unless we're compiling with hip-clang for MI300.
template<hip_f8_type T_A, hip_f8_type T_B>
__device__ hip_float32x4 mfma_f32_16x16x32(hip_f8x8<T_A> a, hip_f8x8<T_B> b, hip_float32x4 c);

template<hip_f8_type T_A, hip_f8_type T_B>
__device__ hip_float32x16 mfma_f32_32x32x16(hip_f8x8<T_A> a, hip_f8x8<T_B> b, hip_float32x16 c);


//
//
//   Implementations
//   TODO: bfloat16
//
namespace hip_f8_impl {

__host__ inline int clz(uint32_t x) { return __builtin_clz(x); }
__device__ inline int clz(uint32_t x) { return __clz(x); }

template <int wm, int we, bool stoch, typename T>
HIP_HOST_DEVICE 
uint8_t cast_to_f8_no_range_reduce(T _x, uint32_t rng = 0) {
  static_assert(we==5, "we==5");
  static_assert(sizeof(T)==2, "no_range_reduce only works for float16");

  uint32_t x = reinterpret_cast<uint16_t&>(_x);

  uint32_t y, head, mantissa, exponent;
  uint32_t sign;

  const int wmo = 10;
  head = x & 0xFC00;
  mantissa = x & 0x3FF;
  exponent = (head>>10) & 0x1F;
  sign = head >> 15;
  uint32_t signed_inf = (sign<<7) + (((1<<we)-1)<<wm);

  if((x & 0x7FFF)==0x7C00)
    return signed_inf;
  if((x & 0x7C00)==0x7C00)
    return signed_inf+1;
  if(x==0)
    return 0;
  if(x==0x8000)
    return 0x80;
  uint32_t drop_mask =  (1 << (wmo-wm)) - 1;
  mantissa += (stoch ? rng : mantissa) & drop_mask;
  if(exponent!=0)
    mantissa += 1<<wmo;
  if(mantissa >= (2<<wmo)) {
    mantissa >>= 1;
    exponent++;
  }
  else if(mantissa>=(1<<wmo) && exponent==0) {
    exponent++;
  }
  mantissa >>= (wmo-wm);
  mantissa &= (1<<wm) - 1;
  if(exponent == 31)
    return (sign<<7) | 0x7B;
  return (sign << 7) | (exponent << wm) | mantissa;
}

template <int wm, int we, typename T, bool negative_zero_nan, bool clip, bool stoch = false>
HIP_HOST_DEVICE
uint8_t cast_to_f8(T _x, uint32_t rng = 0) {
  constexpr bool is_half = std::is_same<T,__half>::value;
  constexpr bool is_float = std::is_same<T,float>::value;
  static_assert(wm+we==7, "wm+we==7");
  static_assert(is_half || is_float, "Only half and float can be cast to f8");

  if(sizeof(T)==2 && we==5 && !negative_zero_nan)
    return cast_to_f8_no_range_reduce<2, 5, stoch, __half>(_x, rng);

  const int wmo = (sizeof(T)==4) ? 23 : 10;
  uint32_t x;
  if(sizeof(T)==4)
    x = reinterpret_cast<uint32_t&>(_x);
  else
    x = reinterpret_cast<uint16_t&>(_x);

  uint32_t y, head, mantissa;
  int exponent;
  uint32_t sign;

  if(sizeof(T)==4) {
    head = x & 0xFF800000;
    mantissa = x & 0x7FFFFF;
    exponent = (head>>23) & 0xFF;
    sign = head >> 31;
  } else {
    head = x & 0xFC00;
    mantissa = x & 0x3FF;
    exponent = (head>>10) & 0x1F;
    sign = head >> 15;
  }

  uint32_t signed_inf = (sign<<7) + (((1<<we)-1)<<wm);

  if(negative_zero_nan) {
    if(sizeof(T)==4) {
      if((x & 0x7F800000) == 0x7F800000)
       return 0x80;
    } else {
      if((x & 0x7C00)==0x7C00)
       return 0x80;
    }
  }
  else {
    if(sizeof(T)==4) {
      if((x & 0x7F800000) == 0x7F800000)
        return signed_inf + (mantissa!=0 ? 1 : 0);
    } else {
      if((x & 0x7C00)==0x7C00)
        return signed_inf + (mantissa!=0 ? 1 : 0);
    }
  }
  if(x==0)
    return 0;

  if(is_half && we==5 && negative_zero_nan && exponent==0) {
     exponent+=1;
     int sh = 1 + clz(mantissa) - (32-wmo);
     mantissa <<= sh;
     exponent -= sh;
     mantissa &= ~(1<<wmo);
  }

  uint32_t drop_mask =  (1 << (wmo-wm)) - 1;
  constexpr int max_exp = (1<<we)-(negative_zero_nan ? 1 : 2);
  constexpr int exp_low_cutoff = (sizeof(T)==4 ? 128 : 16) - (1<<(we-1)) + 1 - (negative_zero_nan ? 1 : 0);

  exponent -= exp_low_cutoff-1;
  if(exponent<=0) 
    drop_mask = (1 << (wmo-wm+1-exponent)) - 1;
  mantissa += 1<<wmo;
  mantissa += (stoch ? rng : mantissa) & drop_mask;
  if(mantissa >= (2<<wmo)) {
    mantissa >>= 1;
    exponent++;
  }
  mantissa >>= (wmo-wm);

  if(exponent<=0) {
    // subnormal range; represented by a subnormal float8 (exponent 0)
    // and involves loss of accuracy
    mantissa >>= 1-exponent;
    exponent = 0;
  }
    // above range: quantize to maximum possible float of the same sign
  else if(exponent > max_exp) {
    if(clip) {
      mantissa = (1<<wm)-1;
      exponent = max_exp;
    } else {
      return signed_inf;
    }
  }
  if(exponent == 0 && mantissa == 0)
      return negative_zero_nan ? 0 : (sign<<7);
  mantissa &= (1<<wm)-1;
  return (sign << 7) | (exponent << wm) | mantissa;
}

template <int wm, int we, typename T, bool negative_zero_nan>
HIP_HOST_DEVICE
T cast_from_f8(uint8_t x) {
  constexpr bool is_half = std::is_same<T,__half>::value;
  constexpr bool is_float = std::is_same<T,float>::value;
  constexpr bool is_bf16 = std::is_same<T,hip_bfloat16>::value;
  static_assert(is_half || is_float, "only half and float are supported");

  constexpr int weo = is_half ? 5 : 8;
  constexpr int wmo = is_half ? 10 : (is_float ? 23 : 7);

  T fInf, fNegInf, fNaN, fNeg0;
  if(is_half) {
   const uint16_t ihInf = 0x7C00;
   const uint16_t ihNegInf = 0xFC00;
   const uint16_t ihNaN = 0x7C01;
   const uint16_t ihNeg0 = 0x8000;
   fInf = reinterpret_cast<const __half&>(ihInf);
   fNegInf = reinterpret_cast<const __half&>(ihNegInf);
   fNaN = reinterpret_cast<const __half&>(ihNaN);
   fNeg0 = reinterpret_cast<const __half&>(ihNeg0);
  } else if(is_float) {
    const uint32_t ifInf = 0x7F800000;
    const uint32_t ifNegInf = 0xFF800000;
    const uint32_t ifNaN = 0x7F800001;
    const uint32_t ifNeg0 = 0x80000000;
    fInf = reinterpret_cast<const float&>(ifInf);
    fNegInf = reinterpret_cast<const float&>(ifNegInf);
    fNaN = reinterpret_cast<const float&>(ifNaN);
    fNeg0 = reinterpret_cast<const float&>(ifNeg0);
  }

  if(x==0)
    return 0;

  uint32_t sign = x>>7;
  uint32_t mantissa = x & ((1<<wm)-1);
  int exponent = (x & 0x7F) >> wm;
  if(negative_zero_nan) {
    if(x==0x80)
      return fNaN;
  } else {
    if(x==0x80) 
      return fNeg0;
    if(exponent == ((1<<we)-1))
       return (mantissa == 0) ? (sign ? fNegInf : fInf) : fNaN;
  }
  typename std::conditional<sizeof(T)==2, uint16_t, uint32_t>::type retval;
  if(we==5 && is_half && !negative_zero_nan) {
     retval = x<<8;
     return reinterpret_cast<const T&>(retval);
  }

  const int exp_low_cutoff = (1<<(weo-1)) - (1<<(we-1)) + 1 - (negative_zero_nan ? 1 : 0);

  //subnormal input
  if(exponent == 0) {
    //guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
    int sh = 1 + clz(mantissa) - (32-wm);
    mantissa <<= sh;
    exponent += 1-sh;
    mantissa &= ((1<<wm)-1);
  }
  exponent += exp_low_cutoff-1;
  mantissa <<= wmo - wm;

  // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
  if(exponent<=0) {
    mantissa |= 1<<wmo;
    mantissa >>= 1-exponent;
    exponent = 0;
  }

  if(sizeof(T)==2)
    retval = (sign<<15) | (exponent<<10) | mantissa;
  else
    retval = (sign<<31) | (exponent<<23) | mantissa;
  return reinterpret_cast<const T&>(retval);
}

template <int wm, int we, bool stoch, typename T>
HIP_HOST_DEVICE 
uint16_t cast_to_f8x_no_range_reduce(T _x, uint32_t rng = 0) {
  static_assert(we==5, "we==5");
  static_assert(sizeof(T)==2, "no_range_reduce only works for float16");

  uint32_t x = reinterpret_cast<uint16_t&>(_x);

  uint32_t y, head, mantissa, exponent;
  uint32_t sign;

  const int wmo = 10;
  head = x & 0xFC00;
  mantissa = x & 0x3FF;
  exponent = (head>>10) & 0x1F;
  sign = head >> 15;
  uint32_t signed_inf = (sign<<15) + (((1<<we)-1)<<(15-we));

  if((x & 0x7FFF)==0x7C00)
    return signed_inf;
  if((x & 0x7C00)==0x7C00)
    return signed_inf+1;
  if(x==0)
    return 0;
  if(x==0x8000)
    return 0x8000;
  uint32_t drop_mask =  (1 << (wmo-wm)) - 1;
  mantissa += (stoch ? rng : mantissa) & drop_mask;
  if(exponent!=0)
    mantissa += 1<<wmo;
  if(mantissa >= (2<<wmo)) {
    mantissa >>= 1;
    exponent++;
  }
  else if(mantissa>=(1<<wmo) && exponent==0) {
    exponent++;
  }
  mantissa >>= (wmo-wm);
  mantissa &= (1<<wm) - 1;
  if(exponent == 31)
    return (sign<<15) | 0x7B00;
  //printf("%d %x %x\n", sign, exponent, mantissa);
  return (sign << 15) | (exponent << 10) | (mantissa << (10-wm));
}

template <int wm, int we, typename T, bool negative_zero_nan, bool clip, bool stoch = false>
HIP_HOST_DEVICE
uint16_t cast_to_f8x(T _x, uint32_t rng = 0) {
  constexpr bool is_half = std::is_same<T,__half>::value;
  constexpr bool is_float = std::is_same<T,float>::value;
  static_assert(we==4 || we==5,  "we 4 or 5");
  static_assert(is_half || is_float, "Only half and float can be cast to f8");

  if(sizeof(T)==2 && we==5 && wm==2 && !negative_zero_nan)
    return cast_to_f8x_no_range_reduce<2, 5, stoch, __half>(_x, rng);

  const int wmo = (sizeof(T)==4) ? 23 : 10;
  uint32_t x;
  if(sizeof(T)==4)
    x = reinterpret_cast<uint32_t&>(_x);
  else
    x = reinterpret_cast<uint16_t&>(_x);

  uint32_t y, head, mantissa;
  int exponent;
  uint32_t sign;

  if(sizeof(T)==4) {
    head = x & 0xFF800000;
    mantissa = x & 0x7FFFFF;
    exponent = (head>>23) & 0xFF;
    sign = head >> 31;
  } else {
    head = x & 0xFC00;
    mantissa = x & 0x3FF;
    exponent = (head>>10) & 0x1F;
    sign = head >> 15;
  }
  //printf("%d %x\n", exponent, mantissa);

  uint32_t signed_inf = (sign<<15) + (((1<<we)-1)<<(15-we));

  if(negative_zero_nan) {
    if(sizeof(T)==4) {
      if((x & 0x7F800000) == 0x7F800000)
       return 0x8000;
    } else {
      if((x & 0x7C00)==0x7C00)
       return 0x8000;
    }
  }
  else {
    if(sizeof(T)==4) {
      if((x & 0x7F800000) == 0x7F800000)
        return signed_inf + (mantissa!=0 ? 1 : 0);
    } else {
      if((x & 0x7C00)==0x7C00)
        return signed_inf + (mantissa!=0 ? 1 : 0);
    }
  }

  if(x==0)
    return 0;
  if(exponent==0 && mantissa==0)
    return negative_zero_nan ? 0 : 0x8000;

  if(is_half && we==5 && negative_zero_nan && exponent==0) {
     exponent += 1;
     int sh = 1 + clz(mantissa) - (32-wmo);
     mantissa <<= sh;
     exponent -= sh;
     mantissa &= ~(1<<wmo);
  }
  bool denorm = false;
  if(is_half && we==5 && !negative_zero_nan && exponent==0) {
     denorm = true;
     //exponent -= 1;
  }
//  printf("%d %x\n", exponent, mantissa);

  uint32_t drop_mask =  (1 << (wmo-wm)) - 1;
  constexpr int max_exp = (1<<we)-(negative_zero_nan ? 1 : 2);
  constexpr int exp_low_cutoff = (sizeof(T)==4 ? 128 : 16) - (1<<(we-1)) + 1 - (negative_zero_nan ? 1 : 0);

  if(denorm) {
    //int sh = 1 + clz(mantissa) - (32-wmo);
    //uint32_t drop_mask =  (1 << (wmo-wm - 1)) - 1;
    //printf("%x %d\n", mantissa, sh);
    //mantissa <<= sh;
    mantissa += (stoch ? rng : mantissa) & drop_mask;
    //mantissa >>= sh;
    if(mantissa >= (1<<wmo)) 
      exponent = 1;
    mantissa &= (1<<wmo) - 1;
  } else {
    mantissa += 1<<wmo;
    exponent -= exp_low_cutoff-1;
    if(exponent<=0) 
      drop_mask = (1 << (wmo-wm+1-exponent)) - 1;
    mantissa += (stoch ? rng : mantissa) & drop_mask;
    if(mantissa >= (2<<wmo)) {
      mantissa >>= 1;
      exponent++;
    }
  }
  mantissa >>= (wmo-wm);

  if(exponent<=0 && !denorm) {
    // subnormal range; represented by a subnormal float8 (exponent 0)
    // and involves loss of accuracy
    mantissa >>= 1-exponent;
    exponent = 0;
  }
    // above range: quantize to maximum possible float of the same sign
  else if(exponent > max_exp) {
    if(clip) {
      mantissa = (1<<wm)-1;
      exponent = max_exp;
    } else {
      return signed_inf;
    }
  }
  if(exponent == 0 && mantissa == 0)
      return negative_zero_nan ? 0 : (sign<<15);
  mantissa &= (1<<wm)-1;
  //printf("%d %x\n", exponent, mantissa);
  return (sign << 15) | (exponent << (15-we)) | (mantissa << (15-we-wm));
}

template <int wm, int we, typename T, bool negative_zero_nan>
HIP_HOST_DEVICE
T cast_from_f8x(uint16_t x) {
  constexpr bool is_half = std::is_same<T,__half>::value;
  constexpr bool is_float = std::is_same<T,float>::value;
  constexpr bool is_bf16 = std::is_same<T,hip_bfloat16>::value;
  static_assert(is_half || is_float, "only half and float are supported");

  constexpr int weo = is_half ? 5 : 8;
  constexpr int wmo = is_half ? 10 : (is_float ? 23 : 7);

  T fInf, fNegInf, fNaN, fNeg0;
  if(is_half) {
   const uint16_t ihInf = 0x7C00;
   const uint16_t ihNegInf = 0xFC00;
   const uint16_t ihNaN = 0x7C01;
   const uint16_t ihNeg0 = 0x8000;
   fInf = reinterpret_cast<const __half&>(ihInf);
   fNegInf = reinterpret_cast<const __half&>(ihNegInf);
   fNaN = reinterpret_cast<const __half&>(ihNaN);
   fNeg0 = reinterpret_cast<const __half&>(ihNeg0);
  } else if(is_float) {
    const uint32_t ifInf = 0x7F800000;
    const uint32_t ifNegInf = 0xFF800000;
    const uint32_t ifNaN = 0x7F800001;
    const uint32_t ifNeg0 = 0x80000000;
    fInf = reinterpret_cast<const float&>(ifInf);
    fNegInf = reinterpret_cast<const float&>(ifNegInf);
    fNaN = reinterpret_cast<const float&>(ifNaN);
    fNeg0 = reinterpret_cast<const float&>(ifNeg0);
  }

  if(x==0)
    return 0;

  uint32_t sign = x>>15;
  uint32_t mantissa = (x >> (15-we-wm)) & ((1<<wm)-1);
  int exponent = (x & 0x7FFF) >> (15-we);
  if(negative_zero_nan) {
    if(x==0x8000)
      return fNaN;
  } else {
    if(x==0x8000) 
      return fNeg0;
    if(exponent == ((1<<we)-1))
       return (mantissa == 0) ? (sign ? fNegInf : fInf) : fNaN;
  }
  typename std::conditional<sizeof(T)==2, uint16_t, uint32_t>::type retval;
  if(we==5 && is_half && !negative_zero_nan) {
     retval = x;
     return reinterpret_cast<const T&>(retval);
  }

  const int exp_low_cutoff = (1<<(weo-1)) - (1<<(we-1)) + 1 - (negative_zero_nan ? 1 : 0);

  //subnormal input
  if(exponent == 0) {
    //guaranteed mantissa!=0 since cases 0x0 and 0x80 are handled above
    int sh = 1 + clz(mantissa) - (32-wm);
    mantissa <<= sh;
    exponent += 1-sh;
    mantissa &= ((1<<wm)-1);
  }
  exponent += exp_low_cutoff-1;
  mantissa <<= wmo - wm;

  // subnormal output (occurs when T=half, we=5, negative_zero_nan=true)
  if(exponent<=0) {
    mantissa |= 1<<wmo;
    mantissa >>= 1-exponent;
    exponent = 0;
  }

  if(sizeof(T)==2)
    retval = (sign<<15) | (exponent<<10) | mantissa;
  else
    retval = (sign<<31) | (exponent<<23) | mantissa;
  return reinterpret_cast<const T&>(retval);
}


}