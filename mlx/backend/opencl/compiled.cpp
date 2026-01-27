// Copyright © 2025 MLX Contributors
// OpenCL implementation of Compiled (JIT kernel fusion) primitive

#include <cstring>
#include <fmt/format.h>
#include <sstream>

#include "mlx/backend/common/compiled.h"
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/graph_utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

// Get OpenCL type string for dtype
std::string opencl_type_string(Dtype d) {
  switch (d) {
    case float32: return "float";
    case float16: return "half";
    case bfloat16: return "bfloat16_t";  // bfloat16 type from preamble
    case float64: return "double";
    case complex64: return "float2";
    case bool_: return "uchar";
    case int8: return "char";
    case int16: return "short";
    case int32: return "int";
    case int64: return "long";
    case uint8: return "uchar";
    case uint16: return "ushort";
    case uint32: return "uint";
    case uint64: return "ulong";
    default:
      throw std::runtime_error("Unsupported dtype for OpenCL compiled kernel");
  }
}

// Get type suffix for type-specific operations (e.g., "_int", "_uint", "_half")
std::string get_type_suffix(Dtype d) {
  switch (d) {
    case float32: return "";  // Default, no suffix
    case float16: return "_half";
    case bfloat16: return "_bf16";
    case complex64: return "_complex";
    case bool_: return "_int";  // bool uses int operations
    case int8: return "_int";
    case int16: return "_int";
    case int32: return "_int";
    case int64: return "_long";
    case uint8: return "_uint";
    case uint16: return "_uint";
    case uint32: return "_uint";
    case uint64: return "_ulong";
    default:
      return "";  // Fall back to float
  }
}

// Generate OpenCL functor-style operation definitions
// These allow code like Add()(a, b) to work by defining Add() as a macro
// that expands to a function name
std::string generate_op_definitions() {
  return R"(
// ================== Bfloat16 operation wrappers ==================
// These use the bfloat16_to_float and float_to_bfloat16 functions from the preamble
// bfloat16_t is typedef'd to uint16_t (same as ushort)

inline bfloat16_t _Abs_op_bf16(bfloat16_t x) {
  return x & 0x7FFF;  // Clear sign bit
}

inline bfloat16_t _Negative_op_bf16(bfloat16_t x) {
  return x ^ 0x8000;  // Flip sign bit
}

inline bfloat16_t _Sigmoid_op_bf16(bfloat16_t x) {
  float f = bfloat16_to_float(x);
  float result = 1.0f / (1.0f + exp(-f));
  return float_to_bfloat16(result);
}

inline bfloat16_t _Exp_op_bf16(bfloat16_t x) {
  float f = bfloat16_to_float(x);
  return float_to_bfloat16(exp(f));
}

inline bfloat16_t _Log_op_bf16(bfloat16_t x) {
  float f = bfloat16_to_float(x);
  return float_to_bfloat16(log(f));
}

inline bfloat16_t _Sqrt_op_bf16(bfloat16_t x) {
  float f = bfloat16_to_float(x);
  return float_to_bfloat16(sqrt(f));
}

inline bfloat16_t _Rsqrt_op_bf16(bfloat16_t x) {
  float f = bfloat16_to_float(x);
  return float_to_bfloat16(rsqrt(f));
}

inline bfloat16_t _Tanh_op_bf16(bfloat16_t x) {
  float f = bfloat16_to_float(x);
  return float_to_bfloat16(tanh(f));
}

inline bfloat16_t _Sin_op_bf16(bfloat16_t x) {
  float f = bfloat16_to_float(x);
  return float_to_bfloat16(sin(f));
}

inline bfloat16_t _Cos_op_bf16(bfloat16_t x) {
  float f = bfloat16_to_float(x);
  return float_to_bfloat16(cos(f));
}

inline bfloat16_t _Add_op_bf16(bfloat16_t a, bfloat16_t b) {
  float fa = bfloat16_to_float(a);
  float fb = bfloat16_to_float(b);
  return float_to_bfloat16(fa + fb);
}

inline bfloat16_t _Subtract_op_bf16(bfloat16_t a, bfloat16_t b) {
  float fa = bfloat16_to_float(a);
  float fb = bfloat16_to_float(b);
  return float_to_bfloat16(fa - fb);
}

inline bfloat16_t _Multiply_op_bf16(bfloat16_t a, bfloat16_t b) {
  float fa = bfloat16_to_float(a);
  float fb = bfloat16_to_float(b);
  return float_to_bfloat16(fa * fb);
}

inline bfloat16_t _Divide_op_bf16(bfloat16_t a, bfloat16_t b) {
  float fa = bfloat16_to_float(a);
  float fb = bfloat16_to_float(b);
  return float_to_bfloat16(fa / fb);
}

inline bfloat16_t _Maximum_op_bf16(bfloat16_t a, bfloat16_t b) {
  float fa = bfloat16_to_float(a);
  float fb = bfloat16_to_float(b);
  return float_to_bfloat16(isnan(fa) ? fa : (fa > fb ? fa : fb));
}

inline bfloat16_t _Minimum_op_bf16(bfloat16_t a, bfloat16_t b) {
  float fa = bfloat16_to_float(a);
  float fb = bfloat16_to_float(b);
  return float_to_bfloat16(isnan(fa) ? fa : (fa < fb ? fa : fb));
}

inline bfloat16_t _Power_op_bf16(bfloat16_t a, bfloat16_t b) {
  float fa = bfloat16_to_float(a);
  float fb = bfloat16_to_float(b);
  return float_to_bfloat16(pow(fa, fb));
}

inline uchar _Less_op_bf16(bfloat16_t a, bfloat16_t b) {
  return bfloat16_to_float(a) < bfloat16_to_float(b);
}

inline uchar _LessEqual_op_bf16(bfloat16_t a, bfloat16_t b) {
  return bfloat16_to_float(a) <= bfloat16_to_float(b);
}

inline uchar _Greater_op_bf16(bfloat16_t a, bfloat16_t b) {
  return bfloat16_to_float(a) > bfloat16_to_float(b);
}

inline uchar _GreaterEqual_op_bf16(bfloat16_t a, bfloat16_t b) {
  return bfloat16_to_float(a) >= bfloat16_to_float(b);
}

inline uchar _Equal_op_bf16(bfloat16_t a, bfloat16_t b) {
  return a == b;  // Bitwise equal for bfloat16
}

inline uchar _NotEqual_op_bf16(bfloat16_t a, bfloat16_t b) {
  return a != b;
}

inline bfloat16_t _Select_op_bf16(uchar cond, bfloat16_t a, bfloat16_t b) {
  return cond ? a : b;
}

// Unary operation functors (macro style for OpenCL C)
// Usage: Abs()(x) becomes _Abs_op(x)
#define Abs() _Abs_op
#define ArcCos() _ArcCos_op
#define ArcCosh() _ArcCosh_op
#define ArcSin() _ArcSin_op
#define ArcSinh() _ArcSinh_op
#define ArcTan() _ArcTan_op
#define ArcTanh() _ArcTanh_op
#define Ceil() _Ceil_op
#define Conjugate() _Conjugate_op
#define Cos() _Cos_op
#define Cosh() _Cosh_op
#define Erf() _Erf_op
#define ErfInv() _ErfInv_op
#define Exp() _Exp_op
#define Expm1() _Expm1_op
#define Floor() _Floor_op
#define Imag() _Imag_op
#define Log() _Log_op
#define Log1p() _Log1p_op
#define Log2() _Log2_op
#define Log10() _Log10_op
#define LogicalNot() _LogicalNot_op
#define Negative() _Negative_op
#define Real() _Real_op
#define Round() _Round_op
#define Sigmoid() _Sigmoid_op
#define Sign() _Sign_op
#define Sin() _Sin_op
#define Sinh() _Sinh_op
#define Sqrt() _Sqrt_op
#define Rsqrt() _Rsqrt_op
#define Square() _Square_op
#define Tan() _Tan_op
#define Tanh() _Tanh_op
#define BitwiseInvert() _BitwiseInvert_op

// Binary operation functors
#define Add() _Add_op
#define Subtract() _Subtract_op
#define Multiply() _Multiply_op
#define Divide() _Divide_op
#define Remainder() _Remainder_op
#define Maximum() _Maximum_op
#define Minimum() _Minimum_op
#define Power() _Power_op
#define Equal() _Equal_op
#define NotEqual() _NotEqual_op
#define NaNEqual() _NaNEqual_op
#define Less() _Less_op
#define LessEqual() _LessEqual_op
#define Greater() _Greater_op
#define GreaterEqual() _GreaterEqual_op
#define LogicalAnd() _LogicalAnd_op
#define LogicalOr() _LogicalOr_op
#define LogAddExp() _LogAddExp_op
#define ArcTan2() _ArcTan2_op
#define BitwiseAnd() _BitwiseAnd_op
#define BitwiseOr() _BitwiseOr_op
#define BitwiseXor() _BitwiseXor_op
#define LeftShift() _LeftShift_op
#define RightShift() _RightShift_op

// Ternary operation functors
#define Select() _Select_op

// ================== Unary implementations ==================

// Scalar/vector compatible implementations
inline float _Abs_op(float x) { return fabs(x); }
inline half _Abs_op_half(half x) { return fabs(x); }
inline int _Abs_op_int(int x) { return abs(x); }
inline long _Abs_op_long(long x) { return x < 0 ? -x : x; }
inline float2 _Abs_op_complex(float2 x) { return (float2)(sqrt(x.x*x.x + x.y*x.y), 0.0f); }

inline float _ArcCos_op(float x) { return acos(x); }
inline float _ArcCosh_op(float x) { return acosh(x); }
inline float _ArcSin_op(float x) { return asin(x); }
inline float _ArcSinh_op(float x) { return asinh(x); }
inline float _ArcTan_op(float x) { return atan(x); }
inline float _ArcTanh_op(float x) { return atanh(x); }

inline float _Ceil_op(float x) { return ceil(x); }
inline half _Ceil_op_half(half x) { return ceil(x); }

inline float2 _Conjugate_op(float2 z) { return (float2)(z.x, -z.y); }

inline float _Cos_op(float x) { return cos(x); }
inline float _Cosh_op(float x) { return cosh(x); }

// Erf approximation (Abramowitz and Stegun)
inline float _Erf_op(float x) {
  float a1 =  0.254829592f;
  float a2 = -0.284496736f;
  float a3 =  1.421413741f;
  float a4 = -1.453152027f;
  float a5 =  1.061405429f;
  float p  =  0.3275911f;
  int s = x < 0 ? -1 : 1;
  x = fabs(x);
  float t = 1.0f / (1.0f + p * x);
  float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);
  return s * y;
}

// ErfInv approximation
inline float _ErfInv_op(float a) {
  float t = fma(a, -a, 1.0f);
  t = log(t);
  float p;
  if (fabs(t) > 6.125f) {
    p = 3.03697567e-10f;
    p = fma(p, t, 2.93243101e-8f);
    p = fma(p, t, 1.22150334e-6f);
    p = fma(p, t, 2.84108955e-5f);
    p = fma(p, t, 3.93552968e-4f);
    p = fma(p, t, 3.02698812e-3f);
    p = fma(p, t, 4.83185798e-3f);
    p = fma(p, t, -2.64646143e-1f);
    p = fma(p, t, 8.40016484e-1f);
  } else {
    p = 5.43877832e-9f;
    p = fma(p, t, 1.43285448e-7f);
    p = fma(p, t, 1.22774793e-6f);
    p = fma(p, t, 1.12963626e-7f);
    p = fma(p, t, -5.61530760e-5f);
    p = fma(p, t, -1.47697632e-4f);
    p = fma(p, t, 2.31468678e-3f);
    p = fma(p, t, 1.15392581e-2f);
    p = fma(p, t, -2.32015476e-1f);
    p = fma(p, t, 8.86226892e-1f);
  }
  return a * p;
}

inline float _Exp_op(float x) { return exp(x); }
inline half _Exp_op_half(half x) { return exp(x); }
inline float2 _Exp_op_complex(float2 z) {
  float er = exp(z.x);
  return (float2)(er * cos(z.y), er * sin(z.y));
}

inline float _Expm1_op(float x) { return expm1(x); }

inline float _Floor_op(float x) { return floor(x); }
inline half _Floor_op_half(half x) { return floor(x); }

inline float _Imag_op(float2 z) { return z.y; }
inline float _Real_op(float2 z) { return z.x; }

inline float _Log_op(float x) { return log(x); }
inline float _Log1p_op(float x) { return log1p(x); }
inline float _Log2_op(float x) { return log2(x); }
inline float _Log10_op(float x) { return log10(x); }

inline uchar _LogicalNot_op(uchar x) { return !x; }
inline int _LogicalNot_op_int(int x) { return !x; }

inline float _Negative_op(float x) { return -x; }
inline half _Negative_op_half(half x) { return -x; }
inline int _Negative_op_int(int x) { return -x; }
inline float2 _Negative_op_complex(float2 z) { return -z; }

inline float _Round_op(float x) { return round(x); }

inline float _Sigmoid_op(float x) { return 1.0f / (1.0f + exp(-x)); }
inline half _Sigmoid_op_half(half x) { return (half)(1.0f / (1.0f + exp(-(float)x))); }

inline float _Sign_op(float x) { return (x > 0.0f) - (x < 0.0f); }
inline int _Sign_op_int(int x) { return (x > 0) - (x < 0); }

inline float _Sin_op(float x) { return sin(x); }
inline float _Sinh_op(float x) { return sinh(x); }

inline float _Sqrt_op(float x) { return sqrt(x); }
inline half _Sqrt_op_half(half x) { return sqrt(x); }
inline float2 _Sqrt_op_complex(float2 z) {
  float r = sqrt(z.x*z.x + z.y*z.y);
  float angle = atan2(z.y, z.x) * 0.5f;
  float mag = sqrt(r);
  return (float2)(mag * cos(angle), mag * sin(angle));
}

inline float _Rsqrt_op(float x) { return rsqrt(x); }
inline half _Rsqrt_op_half(half x) { return rsqrt(x); }

inline float _Square_op(float x) { return x * x; }
inline half _Square_op_half(half x) { return x * x; }
inline int _Square_op_int(int x) { return x * x; }
inline float2 _Square_op_complex(float2 z) {
  return (float2)(z.x*z.x - z.y*z.y, 2.0f*z.x*z.y);
}

inline float _Tan_op(float x) { return tan(x); }
inline float _Tanh_op(float x) { return tanh(x); }

inline int _BitwiseInvert_op(int x) { return ~x; }
inline uint _BitwiseInvert_op_uint(uint x) { return ~x; }
inline uchar _BitwiseInvert_op_uchar(uchar x) { return ~x; }

// ================== Binary implementations ==================

inline float _Add_op(float a, float b) { return a + b; }
inline half _Add_op_half(half a, half b) { return a + b; }
inline int _Add_op_int(int a, int b) { return a + b; }
inline uint _Add_op_uint(uint a, uint b) { return a + b; }
inline long _Add_op_long(long a, long b) { return a + b; }
inline ulong _Add_op_ulong(ulong a, ulong b) { return a + b; }
inline float2 _Add_op_complex(float2 a, float2 b) { return a + b; }

inline float _Subtract_op(float a, float b) { return a - b; }
inline half _Subtract_op_half(half a, half b) { return a - b; }
inline int _Subtract_op_int(int a, int b) { return a - b; }
inline uint _Subtract_op_uint(uint a, uint b) { return a - b; }
inline float2 _Subtract_op_complex(float2 a, float2 b) { return a - b; }

inline float _Multiply_op(float a, float b) { return a * b; }
inline half _Multiply_op_half(half a, half b) { return a * b; }
inline int _Multiply_op_int(int a, int b) { return a * b; }
inline uint _Multiply_op_uint(uint a, uint b) { return a * b; }
inline float2 _Multiply_op_complex(float2 a, float2 b) {
  return (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

inline float _Divide_op(float a, float b) { return a / b; }
inline half _Divide_op_half(half a, half b) { return a / b; }
inline int _Divide_op_int(int a, int b) { return a / b; }
inline uint _Divide_op_uint(uint a, uint b) { return a / b; }
inline float2 _Divide_op_complex(float2 a, float2 b) {
  float denom = b.x*b.x + b.y*b.y;
  return (float2)((a.x*b.x + a.y*b.y)/denom, (a.y*b.x - a.x*b.y)/denom);
}

inline float _Remainder_op(float a, float b) { return fmod(a, b); }
inline int _Remainder_op_int(int a, int b) { return a % b; }

inline float _Maximum_op(float a, float b) { return isnan(a) ? a : (a > b ? a : b); }
inline half _Maximum_op_half(half a, half b) { return isnan(a) ? a : (a > b ? a : b); }
inline int _Maximum_op_int(int a, int b) { return max(a, b); }

inline float _Minimum_op(float a, float b) { return isnan(a) ? a : (a < b ? a : b); }
inline half _Minimum_op_half(half a, half b) { return isnan(a) ? a : (a < b ? a : b); }
inline int _Minimum_op_int(int a, int b) { return min(a, b); }

inline float _Power_op(float a, float b) { return pow(a, b); }
inline half _Power_op_half(half a, half b) { return (half)pow((float)a, (float)b); }
inline int _Power_op_int(int base, int exp) {
  if (exp < 0) return 0;
  int result = 1;
  while (exp > 0) {
    if (exp & 1) result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}

inline uchar _Equal_op(float a, float b) { return a == b; }
inline uchar _Equal_op_int(int a, int b) { return a == b; }
inline uchar _Equal_op_complex(float2 a, float2 b) { return a.x == b.x && a.y == b.y; }

inline uchar _NotEqual_op(float a, float b) { return a != b; }
inline uchar _NotEqual_op_int(int a, int b) { return a != b; }
inline uchar _NotEqual_op_complex(float2 a, float2 b) { return a.x != b.x || a.y != b.y; }

inline uchar _NaNEqual_op(float a, float b) { return (a == b) || (isnan(a) && isnan(b)); }

inline uchar _Less_op(float a, float b) { return a < b; }
inline uchar _Less_op_int(int a, int b) { return a < b; }

inline uchar _LessEqual_op(float a, float b) { return a <= b; }
inline uchar _LessEqual_op_int(int a, int b) { return a <= b; }

inline uchar _Greater_op(float a, float b) { return a > b; }
inline uchar _Greater_op_int(int a, int b) { return a > b; }

inline uchar _GreaterEqual_op(float a, float b) { return a >= b; }
inline uchar _GreaterEqual_op_int(int a, int b) { return a >= b; }

inline uchar _LogicalAnd_op(uchar a, uchar b) { return a && b; }
inline uchar _LogicalOr_op(uchar a, uchar b) { return a || b; }

inline float _LogAddExp_op(float x, float y) {
  if (isnan(x) || isnan(y)) return NAN;
  float maxval = max(x, y);
  float minval = min(x, y);
  return (isinf(minval) && minval < 0) || isinf(maxval)
      ? maxval : (maxval + log1p(exp(minval - maxval)));
}

inline float _ArcTan2_op(float a, float b) { return atan2(a, b); }

inline int _BitwiseAnd_op(int a, int b) { return a & b; }
inline int _BitwiseOr_op(int a, int b) { return a | b; }
inline int _BitwiseXor_op(int a, int b) { return a ^ b; }
inline int _LeftShift_op(int a, int b) { return a << b; }
inline int _RightShift_op(int a, int b) { return a >> b; }

// ================== Ternary implementations ==================

inline float _Select_op(uchar cond, float a, float b) { return cond ? a : b; }
inline half _Select_op_half(uchar cond, half a, half b) { return cond ? a : b; }
inline int _Select_op_int(uchar cond, int a, int b) { return cond ? a : b; }
inline float2 _Select_op_complex(uchar cond, float2 a, float2 b) { return cond ? a : b; }

)";
}

// Build contiguous kernel
void build_contiguous_kernel(
    std::ostringstream& os,
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::function<bool(size_t)>& is_constant,
    bool use_big_index) {

  NodeNamer namer;
  std::string idx_type = use_big_index ? "long" : "uint";

  // Track which inputs need offsets (non-constant, non-scalar)
  std::vector<std::pair<size_t, std::string>> inputs_with_offsets;

  // Function signature
  os << "__kernel void " << kernel_name << "(\n";

  int arg_idx = 0;
  // Input arguments
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant(i)) continue;
    const auto& x = inputs[i];
    const std::string& xname = namer.get_name(x);
    os << "    __global const " << opencl_type_string(x.dtype())
       << "* " << xname << ",\n";
    arg_idx++;
    // Track non-scalar inputs that need offsets
    if (!is_scalar(x)) {
      inputs_with_offsets.push_back({i, xname});
    }
  }

  // Input offset arguments (for sliced arrays)
  for (const auto& [idx, xname] : inputs_with_offsets) {
    os << "    const " << idx_type << " " << xname << "_offset,\n";
    arg_idx++;
  }

  // Output arguments
  for (const auto& x : outputs) {
    os << "    __global " << opencl_type_string(x.dtype())
       << "* " << namer.get_name(x) << ",\n";
    arg_idx++;
  }

  // Output offset arguments (for donated buffers with offsets)
  for (const auto& x : outputs) {
    os << "    const " << idx_type << " " << namer.get_name(x) << "_offset,\n";
    arg_idx++;
  }

  // Size argument
  os << "    const " << idx_type << " size) {\n";

  // Thread index
  os << "  " << idx_type << " index = get_global_id(0);\n";
  os << "  if (index >= size) return;\n\n";

  // Read inputs into temporaries
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& x = inputs[i];
    const std::string& xname = namer.get_name(x);
    std::string type = opencl_type_string(x.dtype());

    if (is_constant(i)) {
      // Print constant value
      std::ostringstream ss;
      print_constant(ss, x);
      // For bfloat16, we need to use float_to_bfloat16 since bfloat16_t is just ushort
      if (x.dtype() == bfloat16) {
        os << "  " << type << " tmp_" << xname << " = float_to_bfloat16((float)(" << ss.str() << "));\n";
      } else {
        os << "  " << type << " tmp_" << xname << " = (" << type << ")(" << ss.str() << ");\n";
      }
    } else if (is_scalar(x)) {
      os << "  " << type << " tmp_" << xname << " = " << xname << "[0];\n";
    } else {
      // Use index + offset to handle sliced arrays correctly
      os << "  " << type << " tmp_" << xname << " = " << xname << "[index + " << xname << "_offset];\n";
    }
  }

  os << "\n";

  // Execute tape operations
  for (const auto& x : tape) {
    const std::string& xname = namer.get_name(x);
    std::string type = opencl_type_string(x.dtype());

    os << "  " << type << " tmp_" << xname << " = ";

    if (is_static_cast(x.primitive())) {
      // Handle bfloat16 casts specially
      Dtype src_dtype = x.inputs()[0].dtype();
      Dtype dst_dtype = x.dtype();
      if (src_dtype == bfloat16 && dst_dtype == float32) {
        os << "bfloat16_to_float(tmp_" << namer.get_name(x.inputs()[0]) << ");\n";
      } else if (src_dtype == float32 && dst_dtype == bfloat16) {
        os << "float_to_bfloat16(tmp_" << namer.get_name(x.inputs()[0]) << ");\n";
      } else {
        os << "(" << type << ")(tmp_" << namer.get_name(x.inputs()[0]) << ");\n";
      }
    } else {
      // Get the type suffix for the output dtype to call the correct operation
      std::string op_name = x.primitive().name();
      std::string type_suffix = get_type_suffix(x.dtype());

      // Use the type-specific function directly
      os << "_" << op_name << "_op" << type_suffix << "(";
      for (size_t i = 0; i < x.inputs().size(); ++i) {
        if (i > 0) os << ", ";
        os << "tmp_" << namer.get_name(x.inputs()[i]);
      }
      os << ");\n";
    }
  }

  os << "\n";

  // Write outputs (using offset for donated buffers)
  for (const auto& x : outputs) {
    const std::string& xname = namer.get_name(x);
    os << "  " << xname << "[index + " << xname << "_offset] = tmp_" << xname << ";\n";
  }

  os << "}\n\n";
}

// Build strided kernel
void build_strided_kernel(
    std::ostringstream& os,
    const std::string& kernel_name,
    const std::vector<array>& inputs,
    const std::vector<array>& outputs,
    const std::vector<array>& tape,
    const std::function<bool(size_t)>& is_constant,
    int ndim,
    bool dynamic_dims,
    bool use_big_index) {

  NodeNamer namer;
  std::string idx_type = use_big_index ? "long" : "uint";

  // Count non-constant, non-scalar inputs and track for offsets
  std::vector<const array*> strided_inputs;
  std::vector<std::pair<size_t, std::string>> inputs_with_offsets;
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (!is_constant(i) && !is_scalar(inputs[i])) {
      strided_inputs.push_back(&inputs[i]);
    }
  }

  // Function signature
  os << "__kernel void " << kernel_name << "(\n";

  int arg_idx = 0;
  // Input arguments
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant(i)) continue;
    const auto& x = inputs[i];
    const std::string& xname = namer.get_name(x);
    os << "    __global const " << opencl_type_string(x.dtype())
       << "* " << xname << ",\n";
    arg_idx++;
    // Track non-scalar inputs that need offsets
    if (!is_scalar(x)) {
      inputs_with_offsets.push_back({i, xname});
    }
  }

  // Input offset arguments (for sliced arrays)
  for (const auto& [idx, xname] : inputs_with_offsets) {
    os << "    const " << idx_type << " " << xname << "_offset,\n";
    arg_idx++;
  }

  // Strides for non-scalar inputs (flattened array)
  if (!strided_inputs.empty()) {
    os << "    __global const long* in_strides,\n";
  }

  // Output arguments
  for (const auto& x : outputs) {
    os << "    __global " << opencl_type_string(x.dtype())
       << "* " << namer.get_name(x) << ",\n";
  }

  // Output offset arguments (for donated buffers with offsets)
  for (const auto& x : outputs) {
    os << "    const " << idx_type << " " << namer.get_name(x) << "_offset,\n";
  }

  // Shape and size
  os << "    __global const int* output_shape,\n";
  os << "    const " << idx_type << " size";
  if (dynamic_dims) {
    os << ",\n    const int ndim";
  }
  os << ") {\n";

  // Thread index
  os << "  " << idx_type << " index = get_global_id(0);\n";
  os << "  if (index >= size) return;\n\n";

  // Compute indices for strided inputs
  if (!strided_inputs.empty()) {
    // Initialize indices with offsets (for sliced arrays)
    for (size_t i = 0; i < strided_inputs.size(); ++i) {
      const std::string& xname = namer.get_name(*strided_inputs[i]);
      os << "  " << idx_type << " " << xname << "_idx = " << xname << "_offset;\n";
    }

    // Compute indices from linear index
    os << "  {\n";
    os << "    " << idx_type << " loc = index;\n";

    std::string ndim_str = dynamic_dims ? "ndim" : std::to_string(ndim);
    os << "    for (int d = " << ndim_str << " - 1; d >= 0; d--) {\n";
    os << "      int dim_idx = loc % output_shape[d];\n";
    os << "      loc /= output_shape[d];\n";

    for (size_t i = 0; i < strided_inputs.size(); ++i) {
      const std::string& xname = namer.get_name(*strided_inputs[i]);
      if (dynamic_dims) {
        os << "      " << xname << "_idx += dim_idx * in_strides[" << i << " * ndim + d];\n";
      } else {
        os << "      " << xname << "_idx += dim_idx * in_strides[" << i * ndim << " + d];\n";
      }
    }

    os << "    }\n";
    os << "  }\n\n";
  }

  // Read inputs into temporaries
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto& x = inputs[i];
    const std::string& xname = namer.get_name(x);
    std::string type = opencl_type_string(x.dtype());

    if (is_constant(i)) {
      std::ostringstream ss;
      print_constant(ss, x);
      // For bfloat16, we need to use float_to_bfloat16 since bfloat16_t is just ushort
      if (x.dtype() == bfloat16) {
        os << "  " << type << " tmp_" << xname << " = float_to_bfloat16((float)(" << ss.str() << "));\n";
      } else {
        os << "  " << type << " tmp_" << xname << " = (" << type << ")(" << ss.str() << ");\n";
      }
    } else if (is_scalar(x)) {
      os << "  " << type << " tmp_" << xname << " = " << xname << "[0];\n";
    } else {
      os << "  " << type << " tmp_" << xname << " = " << xname << "[" << xname << "_idx];\n";
    }
  }

  os << "\n";

  // Execute tape operations
  for (const auto& x : tape) {
    const std::string& xname = namer.get_name(x);
    std::string type = opencl_type_string(x.dtype());

    os << "  " << type << " tmp_" << xname << " = ";

    if (is_static_cast(x.primitive())) {
      // Handle bfloat16 casts specially
      Dtype src_dtype = x.inputs()[0].dtype();
      Dtype dst_dtype = x.dtype();
      if (src_dtype == bfloat16 && dst_dtype == float32) {
        os << "bfloat16_to_float(tmp_" << namer.get_name(x.inputs()[0]) << ");\n";
      } else if (src_dtype == float32 && dst_dtype == bfloat16) {
        os << "float_to_bfloat16(tmp_" << namer.get_name(x.inputs()[0]) << ");\n";
      } else {
        os << "(" << type << ")(tmp_" << namer.get_name(x.inputs()[0]) << ");\n";
      }
    } else {
      // Get the type suffix for the output dtype to call the correct operation
      std::string op_name = x.primitive().name();
      std::string type_suffix = get_type_suffix(x.dtype());

      // Use the type-specific function directly
      os << "_" << op_name << "_op" << type_suffix << "(";
      for (size_t i = 0; i < x.inputs().size(); ++i) {
        if (i > 0) os << ", ";
        os << "tmp_" << namer.get_name(x.inputs()[i]);
      }
      os << ");\n";
    }
  }

  os << "\n";

  // Write outputs (using offset for donated buffers)
  for (const auto& x : outputs) {
    const std::string& xname = namer.get_name(x);
    os << "  " << xname << "[index + " << xname << "_offset] = tmp_" << xname << ";\n";
  }

  os << "}\n\n";
}

} // anonymous namespace

void Compiled::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& d = opencl::device(s.device);

  OPENCL_DEBUG_LOG("[Compiled::eval_gpu] lib_name=" << lib_name()
            << " inputs=" << inputs.size() << " outputs=" << outputs.size()
            << " tape=" << tape_.size());

  // Debug: print tape contents
  for (size_t i = 0; i < tape_.size(); ++i) {
    OPENCL_DEBUG_LOG("[Compiled::eval_gpu] tape[" << i << "] = "
              << tape_[i].primitive().name()
              << " dtype=" << opencl_type_string(tape_[i].dtype())
              << " inputs=" << tape_[i].inputs().size());
  }

  // Debug: print inputs
  for (size_t i = 0; i < inputs_.size(); ++i) {
    OPENCL_DEBUG_LOG("[Compiled::eval_gpu] inputs_[" << i << "] dtype=" << opencl_type_string(inputs_[i].dtype())
              << " is_constant=" << is_constant_(i));
  }

  // Collapse contiguous dims to route to a faster kernel if possible
  auto [contiguous, shape, strides_vec] =
      compiled_collapse_contiguous_dims(inputs, outputs[0], is_constant_);

  // Whether to use large index
  bool large = compiled_use_large_index(inputs, outputs, contiguous);
  int ndim = shape.size();
  bool dynamic = ndim >= 8;

  OPENCL_DEBUG_LOG("[Compiled::eval_gpu] contiguous=" << contiguous
            << " ndim=" << ndim << " large=" << large << " dynamic=" << dynamic);

  // Generate kernel source
  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();
  kernel_source << generate_op_definitions();

  std::string kernel_name;
  if (contiguous) {
    kernel_name = lib_name() + "_contiguous";
    if (large) kernel_name += "_large";
    build_contiguous_kernel(
        kernel_source, kernel_name, inputs_, outputs_, tape_,
        is_constant_, large);
  } else {
    if (dynamic) {
      kernel_name = lib_name() + "_strided_dynamic";
    } else {
      kernel_name = lib_name() + "_strided_" + std::to_string(ndim);
    }
    if (large) kernel_name += "_large";
    build_strided_kernel(
        kernel_source, kernel_name, inputs_, outputs_, tape_,
        is_constant_, ndim, dynamic, large);
  }

  std::string source = kernel_source.str();

  OPENCL_DEBUG_LOG("[Compiled::eval_gpu] Generated kernel: " << kernel_name);

  // Get or compile kernel
  cl_kernel kernel = d.get_kernel(kernel_name, source);

  auto& encoder = d.get_command_encoder(s.index);
  encoder.set_kernel(kernel);

  // Set input arguments
  int arg = 0;
  std::vector<array> temp_arrays;

  // Track non-constant, non-scalar inputs for offset passing
  std::vector<size_t> inputs_needing_offsets;

  for (size_t i = 0; i < inputs.size(); ++i) {
    if (is_constant_(i)) continue;
    encoder.set_input_array(inputs[i], arg++);
    // Track non-scalar inputs that need offsets
    if (!is_scalar(inputs[i])) {
      inputs_needing_offsets.push_back(i);
    }
  }

  // Pass input offsets (for sliced arrays) - needed for both contiguous and strided kernels
  for (size_t idx : inputs_needing_offsets) {
    // Get the element offset (byte offset / element size)
    int64_t elem_offset = inputs[idx].offset() / size_of(inputs[idx].dtype());
    if (large) {
      encoder.set_bytes(elem_offset, arg++);
    } else {
      uint32_t offset32 = static_cast<uint32_t>(elem_offset);
      encoder.set_bytes(offset32, arg++);
    }
  }

  // Set strides for strided kernel
  std::vector<int64_t> in_strides;
  if (!contiguous) {
    int stride_idx = 1;  // idx 0 is output strides
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (is_constant_(i) || is_scalar(inputs[i])) continue;
      in_strides.insert(
          in_strides.end(),
          strides_vec[stride_idx].begin(),
          strides_vec[stride_idx].end());
      stride_idx++;
    }

    if (!in_strides.empty()) {
      // Allocate and copy strides to device via host-mapped memory
      size_t strides_size = in_strides.size() * sizeof(int64_t);
      auto strides_buffer = opencl::allocator().malloc(strides_size);
      cl_mem strides_buf = static_cast<cl_mem>(strides_buffer.ptr());

      // Get host pointer and copy data
      void* host_ptr = opencl::allocator().get_host_ptr(strides_buf);
      std::memcpy(host_ptr, in_strides.data(), strides_size);

      encoder.set_buffer(strides_buf, arg++);
    }
  }

  // Allocate outputs
  compiled_allocate_outputs(
      inputs, outputs, is_constant_, contiguous,
      [](size_t n) { return opencl::allocator().malloc(n); });

  // Set output arguments
  for (auto& x : outputs) {
    encoder.set_output_array(x, arg++);
  }

  // Set output offsets (for donated buffers with offsets)
  for (const auto& x : outputs) {
    int64_t elem_offset = x.offset() / size_of(x.dtype());
    if (large) {
      encoder.set_bytes(elem_offset, arg++);
    } else {
      uint32_t offset32 = static_cast<uint32_t>(elem_offset);
      encoder.set_bytes(offset32, arg++);
    }
  }

  // Set shape for strided kernel
  std::vector<int> shape_int;
  if (!contiguous) {
    for (auto s : shape) {
      shape_int.push_back(static_cast<int>(s));
    }

    size_t shape_size = shape_int.size() * sizeof(int);
    auto shape_buffer = opencl::allocator().malloc(shape_size);
    cl_mem shape_buf = static_cast<cl_mem>(shape_buffer.ptr());

    // Get host pointer and copy data
    void* shape_host_ptr = opencl::allocator().get_host_ptr(shape_buf);
    std::memcpy(shape_host_ptr, shape_int.data(), shape_size);

    encoder.set_buffer(shape_buf, arg++);
  }

  // Set size
  size_t total_size = outputs[0].data_size();
  if (large) {
    int64_t size64 = static_cast<int64_t>(total_size);
    encoder.set_bytes(size64, arg++);
  } else {
    uint32_t size32 = static_cast<uint32_t>(total_size);
    encoder.set_bytes(size32, arg++);
  }

  // Set ndim for dynamic kernel
  if (!contiguous && dynamic) {
    encoder.set_bytes(ndim, arg++);
  }

  // Dispatch kernel
  size_t global_size[3] = {total_size, 1, 1};
  size_t local_size[3] = {256, 1, 1};

  // Ensure global size is multiple of local size
  global_size[0] = ((total_size + local_size[0] - 1) / local_size[0]) * local_size[0];

  encoder.dispatch_threads(global_size, local_size, 1);

  // Mark inputs as temporaries
  for (const auto& x : inputs) {
    d.add_temporary(x, s.index);
  }

  d.end_encoding(s.index);

  OPENCL_DEBUG_LOG("[Compiled::eval_gpu] Dispatched kernel with " << total_size << " elements");
}

} // namespace mlx::core
