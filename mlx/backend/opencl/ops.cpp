// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/ops.h"

namespace mlx::core::opencl {

std::string get_unary_ops() {
  return R"(
// Unary operation definitions
// Type-specific abs functions - fabs for floats, abs for integers
// OpenCL's abs() returns unsigned, so we use (x < 0 ? -x : x) for signed ints
inline int8_t abs_int8(int8_t x) { return x < 0 ? -x : x; }
inline int16_t abs_int16(int16_t x) { return x < 0 ? -x : x; }
inline int32_t abs_int32(int32_t x) { return x < 0 ? -x : x; }
inline int64_t abs_int64(int64_t x) { return x < 0 ? -x : x; }
inline uint8_t abs_uint8(uint8_t x) { return x; }  // unsigned is always positive
inline uint16_t abs_uint16(uint16_t x) { return x; }
inline uint32_t abs_uint32(uint32_t x) { return x; }
inline uint64_t abs_uint64(uint64_t x) { return x; }
// bfloat16 abs - clear sign bit (bfloat16 is stored as uint16 with sign in MSB)
inline uint16_t abs_bfloat16(uint16_t x) { return x & 0x7FFF; }
// fabs is built-in for float/half

#define neg(x) (-(x))
// Use mlx_sign instead of sign to avoid conflict with OpenCL's built-in sign function
#define mlx_sign(x) (((x) > 0) - ((x) < 0))
#define sign(x) mlx_sign(x)
#define logical_not(x) (!(x))

// Identity function for integer types where rounding is a no-op
#define identity(x) (x)

// Math functions (most are built-in to OpenCL)
// exp, log, log2, log10, sin, cos, tan are built-in
// sqrt, rsqrt are built-in
// sinh, cosh, tanh are built-in
// asin, acos, atan are built-in
// asinh, acosh, atanh are built-in

// Custom implementations for operations without direct OpenCL equivalents
inline float sigmoid(float x) {
  return 1.0f / (1.0f + exp(-x));
}

inline half sigmoid_half(half x) {
  return (half)(1.0f / (1.0f + exp(-(float)x)));
}

inline float erf_approx(float x) {
  // Abramowitz and Stegun approximation
  float a1 =  0.254829592f;
  float a2 = -0.284496736f;
  float a3 =  1.421413741f;
  float a4 = -1.453152027f;
  float a5 =  1.061405429f;
  float p  =  0.3275911f;

  int sign = 1;
  if (x < 0) sign = -1;
  x = fabs(x);

  float t = 1.0f / (1.0f + p * x);
  float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

  return sign * y;
}

inline float erfc_approx(float x) {
  return 1.0f - erf_approx(x);
}

// Inverse error function approximation
// Based on polynomial approximation from:
// https://stackoverflow.com/questions/35148198/efficient-faithfully-rounded-implementation-of-error-function-erff
inline float erfinv_approx(float a) {
  float t = fma(a, -a, 1.0f);
  t = log(t);
  float p;
  if (fabs(t) > 6.125f) {
    // maximum ulp error = 2.35793
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
    // maximum ulp error = 2.35002
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

inline float gelu_approx(float x) {
  // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
  float x3 = x * x * x;
  float inner = 0.7978845608f * (x + 0.044715f * x3);
  return 0.5f * x * (1.0f + tanh(inner));
}

inline float gelu_exact(float x) {
  // GELU(x) = x * Φ(x) where Φ is the CDF of standard normal
  // Φ(x) ≈ 0.5 * (1 + erf(x / sqrt(2)))
  return x * 0.5f * (1.0f + erf_approx(x * 0.70710678118f));
}

inline float silu(float x) {
  // SiLU/Swish: x * sigmoid(x)
  return x * sigmoid(x);
}

inline float softplus(float x) {
  // softplus(x) = log(1 + exp(x))
  // For numerical stability
  if (x > 20.0f) return x;
  return log(1.0f + exp(x));
}

inline float relu(float x) {
  return max(x, 0.0f);
}

inline float leaky_relu(float x, float alpha) {
  return (x > 0.0f) ? x : alpha * x;
}

inline float elu(float x, float alpha) {
  return (x > 0.0f) ? x : alpha * (exp(x) - 1.0f);
}

inline float celu(float x, float alpha) {
  return (x > 0.0f) ? x : alpha * (exp(x / alpha) - 1.0f);
}

inline float relu6(float x) {
  return clamp(x, 0.0f, 6.0f);
}

inline float hard_swish(float x) {
  return x * clamp(x + 3.0f, 0.0f, 6.0f) / 6.0f;
}

inline float square(float x) {
  return x * x;
}

inline float reciprocal(float x) {
  return 1.0f / x;
}

// Bitwise NOT/invert for integer types
#define bitwise_invert(x) (~(x))

// Complex conjugate: (a + bi) -> (a - bi)
inline float2 conjugate(float2 z) {
  return (float2)(z.x, -z.y);
}

// Extract real part of complex number
inline float creal(float2 z) {
  return z.x;
}

// Extract imaginary part of complex number
inline float cimag(float2 z) {
  return z.y;
}

// Square function for complex numbers
inline float2 square_complex(float2 z) {
  // (a + bi)^2 = a^2 - b^2 + 2abi
  return (float2)(z.x * z.x - z.y * z.y, 2.0f * z.x * z.y);
}

// Complex abs (magnitude) - returns float, not float2
inline float abs_complex(float2 z) {
  return sqrt(z.x * z.x + z.y * z.y);
}

// Complex negative
inline float2 neg_complex(float2 z) {
  return (float2)(-z.x, -z.y);
}

// Complex exponential: exp(a+bi) = exp(a) * (cos(b) + i*sin(b))
// Special case: when real part is -inf, exp(-inf) = 0, and result is (0, 0)
// regardless of imaginary part (cos/sin of inf are undefined but 0*NaN should be 0)
inline float2 exp_complex(float2 z) {
  float ea = exp(z.x);
  // Handle the case where ea == 0 to avoid 0 * NaN when imag is infinite
  if (ea == 0.0f) {
    return (float2)(0.0f, 0.0f);
  }
  return (float2)(ea * cos(z.y), ea * sin(z.y));
}

// Complex sine: sin(a+bi) = sin(a)*cosh(b) + i*cos(a)*sinh(b)
inline float2 sin_complex(float2 z) {
  return (float2)(sin(z.x) * cosh(z.y), cos(z.x) * sinh(z.y));
}

// Complex cosine: cos(a+bi) = cos(a)*cosh(b) - i*sin(a)*sinh(b)
inline float2 cos_complex(float2 z) {
  return (float2)(cos(z.x) * cosh(z.y), -sin(z.x) * sinh(z.y));
}

// Complex tangent: tan(z) = sin(z) / cos(z)
inline float2 tan_complex(float2 z) {
  float2 s = sin_complex(z);
  float2 c = cos_complex(z);
  float denom = c.x * c.x + c.y * c.y;
  return (float2)((s.x * c.x + s.y * c.y) / denom, (s.y * c.x - s.x * c.y) / denom);
}

// Complex sinh: sinh(a+bi) = sinh(a)*cos(b) + i*cosh(a)*sin(b)
inline float2 sinh_complex(float2 z) {
  return (float2)(sinh(z.x) * cos(z.y), cosh(z.x) * sin(z.y));
}

// Complex cosh: cosh(a+bi) = cosh(a)*cos(b) + i*sinh(a)*sin(b)
inline float2 cosh_complex(float2 z) {
  return (float2)(cosh(z.x) * cos(z.y), sinh(z.x) * sin(z.y));
}

// Complex tanh: tanh(z) = sinh(z) / cosh(z)
inline float2 tanh_complex(float2 z) {
  float2 s = sinh_complex(z);
  float2 c = cosh_complex(z);
  float denom = c.x * c.x + c.y * c.y;
  return (float2)((s.x * c.x + s.y * c.y) / denom, (s.y * c.x - s.x * c.y) / denom);
}

// Complex log: log(z) = log(|z|) + i*arg(z)
inline float2 log_complex(float2 z) {
  float mag = sqrt(z.x * z.x + z.y * z.y);
  float arg = atan2(z.y, z.x);
  return (float2)(log(mag), arg);
}

// Complex log2: log2(z) = log(z) / log(2)
inline float2 log2_complex(float2 z) {
  float2 ln_z = log_complex(z);
  float inv_ln2 = 1.4426950408889634f;  // 1/ln(2)
  return (float2)(ln_z.x * inv_ln2, ln_z.y * inv_ln2);
}

// Complex log10: log10(z) = log(z) / log(10)
inline float2 log10_complex(float2 z) {
  float2 ln_z = log_complex(z);
  float inv_ln10 = 0.4342944819032518f;  // 1/ln(10)
  return (float2)(ln_z.x * inv_ln10, ln_z.y * inv_ln10);
}

// Complex log1p: log1p(z) = log(1 + z)
inline float2 log1p_complex(float2 z) {
  float2 one_plus_z = (float2)(1.0f + z.x, z.y);
  return log_complex(one_plus_z);
}

// Complex sqrt: sqrt(z) = sqrt(|z|) * (cos(arg/2) + i*sin(arg/2))
inline float2 sqrt_complex(float2 z) {
  float mag = sqrt(z.x * z.x + z.y * z.y);
  float arg = atan2(z.y, z.x);
  float sqrt_mag = sqrt(mag);
  float half_arg = arg * 0.5f;
  return (float2)(sqrt_mag * cos(half_arg), sqrt_mag * sin(half_arg));
}

// Complex rsqrt: rsqrt(z) = 1 / sqrt(z)
inline float2 rsqrt_complex(float2 z) {
  float2 sq = sqrt_complex(z);
  float mag_sq = sq.x * sq.x + sq.y * sq.y;
  if (mag_sq == 0.0f) {
    return (float2)(INFINITY, 0.0f);
  }
  return (float2)(sq.x / mag_sq, -sq.y / mag_sq);
}

// Complex multiply helper (z1 * z2)
inline float2 cmul(float2 a, float2 b) {
  return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Complex divide helper (z1 / z2)
inline float2 cdiv(float2 a, float2 b) {
  float denom = b.x * b.x + b.y * b.y;
  return (float2)((a.x * b.x + a.y * b.y) / denom, (a.y * b.x - a.x * b.y) / denom);
}

// Complex add helper
inline float2 cadd(float2 a, float2 b) {
  return (float2)(a.x + b.x, a.y + b.y);
}

// Complex subtract helper
inline float2 csub(float2 a, float2 b) {
  return (float2)(a.x - b.x, a.y - b.y);
}

// Complex arcsin: arcsin(z) = -i * log(i*z + sqrt(1 - z^2))
inline float2 arcsin_complex(float2 z) {
  // i*z = (-z.y, z.x)
  float2 iz = (float2)(-z.y, z.x);
  // z^2
  float2 z2 = cmul(z, z);
  // 1 - z^2
  float2 one_minus_z2 = (float2)(1.0f - z2.x, -z2.y);
  // sqrt(1 - z^2)
  float2 sq = sqrt_complex(one_minus_z2);
  // i*z + sqrt(1 - z^2)
  float2 arg = cadd(iz, sq);
  // log(...)
  float2 lg = log_complex(arg);
  // -i * log(...) = (lg.y, -lg.x)
  return (float2)(lg.y, -lg.x);
}

// Complex arccos: arccos(z) = pi/2 - arcsin(z)
inline float2 arccos_complex(float2 z) {
  float2 as = arcsin_complex(z);
  return (float2)(1.5707963267948966f - as.x, -as.y);
}

// Complex arctan: arctan(z) = 0.5i * (log(1 - iz) - log(1 + iz))
// This formulation handles branch cuts correctly
inline float2 arctan_complex(float2 z) {
  // 1 - i*z = (1 + z.y, -z.x)
  float2 one_minus_iz = (float2)(1.0f + z.y, -z.x);
  // 1 + i*z = (1 - z.y, z.x)
  float2 one_plus_iz = (float2)(1.0f - z.y, z.x);
  // log(1 - iz)
  float2 log1 = log_complex(one_minus_iz);
  // log(1 + iz)
  float2 log2 = log_complex(one_plus_iz);
  // diff = log(1 - iz) - log(1 + iz)
  float2 diff = csub(log1, log2);
  // 0.5i * diff = (-0.5 * diff.y, 0.5 * diff.x)
  return (float2)(-0.5f * diff.y, 0.5f * diff.x);
}

// Complex sign: z / |z| (normalizes the complex number)
inline float2 sign_complex(float2 z) {
  float mag = sqrt(z.x * z.x + z.y * z.y);
  if (mag == 0.0f) {
    return (float2)(0.0f, 0.0f);
  }
  return (float2)(z.x / mag, z.y / mag);
}

)";
}

std::string get_binary_ops() {
  return R"(
// Binary operation definitions (capitalized to match MLX primitive names)
#define Add(a, b) ((a) + (b))
#define Subtract(a, b) ((a) - (b))
#define Multiply(a, b) ((a) * (b))
#define Divide(a, b) ((a) / (b))

// Complex multiply: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
inline float2 Multiply_complex(float2 a, float2 b) {
  return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Complex divide: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2+d^2)
inline float2 Divide_complex(float2 a, float2 b) {
  float denom = b.x * b.x + b.y * b.y;
  return (float2)((a.x * b.x + a.y * b.y) / denom, (a.y * b.x - a.x * b.y) / denom);
}
// Python-style remainder (floored division semantics)
// If r != 0 && (r < 0 != y < 0), add y to get floored result
inline float Remainder_float(float a, float b) {
  float r = fmod(a, b);
  if (r != 0.0f && ((r < 0.0f) != (b < 0.0f))) {
    r += b;
  }
  return r;
}
inline half Remainder_half(half a, half b) {
  half r = fmod(a, b);
  if (r != (half)0.0f && ((r < (half)0.0f) != (b < (half)0.0f))) {
    r += b;
  }
  return r;
}
inline int Remainder_int(int a, int b) {
  int r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) {
    r += b;
  }
  return r;
}
inline long Remainder_long(long a, long b) {
  long r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) {
    r += b;
  }
  return r;
}
// Minimum/Maximum with NaN propagation for floating-point types
// If the first operand is NaN, it must be propagated (IEEE 754 behavior)
inline float Maximum_float(float a, float b) {
  if (isnan(a)) return a;
  return a > b ? a : b;
}
inline float Minimum_float(float a, float b) {
  if (isnan(a)) return a;
  return a < b ? a : b;
}
inline half Maximum_half(half a, half b) {
  if (isnan(a)) return a;
  return a > b ? a : b;
}
inline half Minimum_half(half a, half b) {
  if (isnan(a)) return a;
  return a < b ? a : b;
}
// Complex versions use lexicographic comparison (real first, then imaginary)
// NaN in either component propagates
inline float2 Maximum_float2(float2 a, float2 b) {
  if (isnan(a.x) || isnan(a.y)) return a;
  if (a.x > b.x) return a;
  if (a.x < b.x) return b;
  // Real parts equal, compare imaginary
  return a.y >= b.y ? a : b;
}
inline float2 Minimum_float2(float2 a, float2 b) {
  if (isnan(a.x) || isnan(a.y)) return a;
  if (a.x < b.x) return a;
  if (a.x > b.x) return b;
  // Real parts equal, compare imaginary
  return a.y <= b.y ? a : b;
}
// Integer versions use built-in min/max (no NaN concerns)
#define Minimum_int(a, b) (min((a), (b)))
#define Maximum_int(a, b) (max((a), (b)))

// Power for floating-point uses pow()
#define Power_float(a, b) (pow((a), (b)))
#define Power_half(a, b) ((half)pow((float)(a), (float)(b)))

// Complex log: log(z) = log(|z|) + i*arg(z)
inline float2 complex_log(float2 z) {
  float mag = sqrt(z.x * z.x + z.y * z.y);
  float arg = atan2(z.y, z.x);
  return (float2)(log(mag), arg);
}

// Complex multiply: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
inline float2 complex_mul(float2 a, float2 b) {
  return (float2)(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// Complex exp: exp(a+bi) = exp(a) * (cos(b) + i*sin(b))
inline float2 complex_exp_power(float2 z) {
  float er = exp(z.x);
  return (float2)(er * cos(z.y), er * sin(z.y));
}

// Complex power: z^w = exp(w * log(z))
inline float2 Power_float2(float2 z, float2 w) {
  // Handle special case: 0^0 = 1
  if (z.x == 0.0f && z.y == 0.0f && w.x == 0.0f && w.y == 0.0f) {
    return (float2)(1.0f, 0.0f);
  }
  // Handle 0^w where real(w) > 0 = 0
  if (z.x == 0.0f && z.y == 0.0f && w.x > 0.0f) {
    return (float2)(0.0f, 0.0f);
  }
  float2 log_z = complex_log(z);
  float2 w_log_z = complex_mul(w, log_z);
  return complex_exp_power(w_log_z);
}

// Integer power function - computes a^b for integers
inline int32_t ipow_int32(int32_t base, int32_t exp) {
  if (exp < 0) return 0;  // Integer division truncates
  int32_t result = 1;
  while (exp > 0) {
    if (exp & 1) result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}

inline int64_t ipow_int64(int64_t base, int64_t exp) {
  if (exp < 0) return 0;
  int64_t result = 1;
  while (exp > 0) {
    if (exp & 1) result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}

inline uint32_t ipow_uint32(uint32_t base, uint32_t exp) {
  uint32_t result = 1;
  while (exp > 0) {
    if (exp & 1) result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}

inline uint64_t ipow_uint64(uint64_t base, uint64_t exp) {
  uint64_t result = 1;
  while (exp > 0) {
    if (exp & 1) result *= base;
    exp >>= 1;
    base *= base;
  }
  return result;
}

// For bool, true^true=true, true^false=true, false^true=false, false^false=true
inline uint8_t ipow_bool(uint8_t base, uint8_t exp) {
  if (exp == 0) return 1;
  return base;
}

// Logical operations
#define Equal(a, b) ((a) == (b))
#define NaNEqual(a, b) (((a) == (b)) || (isnan(a) && isnan(b)))
#define NotEqual(a, b) ((a) != (b))
#define Less(a, b) ((a) < (b))
#define LessEqual(a, b) ((a) <= (b))
#define Greater(a, b) ((a) > (b))
#define GreaterEqual(a, b) ((a) >= (b))

// Complex (float2) comparison functions - return scalar boolean
// OpenCL's == on vectors returns a vector, so we need explicit functions
inline uint8_t Equal_complex(float2 a, float2 b) {
  return (a.x == b.x) && (a.y == b.y);
}
inline uint8_t NaNEqual_complex(float2 a, float2 b) {
  int eq_or_both_nan_x = (a.x == b.x) || (isnan(a.x) && isnan(b.x));
  int eq_or_both_nan_y = (a.y == b.y) || (isnan(a.y) && isnan(b.y));
  return eq_or_both_nan_x && eq_or_both_nan_y;
}
inline uint8_t NotEqual_complex(float2 a, float2 b) {
  return (a.x != b.x) || (a.y != b.y);
}
#define LogicalAnd(a, b) ((a) && (b))
#define LogicalOr(a, b) ((a) || (b))

// Bitwise operations
#define BitwiseAnd(a, b) ((a) & (b))
#define BitwiseOr(a, b) ((a) | (b))
#define BitwiseXor(a, b) ((a) ^ (b))
#define LeftShift(a, b) ((a) << (b))
#define RightShift(a, b) ((a) >> (b))

// Additional operations
// LogAddExp with numerical stability for float/half types
// Note: We use type-suffixed functions since OpenCL C doesn't support overloading
inline float LogAddExp_float(float x, float y) {
  if (isnan(x) || isnan(y)) return NAN;
  float maxval = max(x, y);
  float minval = min(x, y);
  return (isinf(minval) && minval < 0) || isinf(maxval)
      ? maxval
      : (maxval + log1p(exp(minval - maxval)));
}

inline half LogAddExp_half(half x, half y) {
  if (isnan(x) || isnan(y)) return (half)NAN;
  half maxval = max(x, y);
  half minval = min(x, y);
  return (isinf(minval) && minval < 0) || isinf(maxval)
      ? maxval
      : (maxval + log1p(exp(minval - maxval)));
}

// Complex (float2) version
// exp(a+bi) = exp(a) * (cos(b) + i*sin(b))
// Handle -inf real part specially since cos/sin(-inf) = NaN
inline float2 complex_exp(float2 z) {
  if (isinf(z.x) && z.x < 0) {
    return (float2)(0.0f, 0.0f);  // exp(-inf + anything*i) = 0
  }
  if (isinf(z.x) && z.x > 0) {
    // exp(+inf + bi) = inf * (cos(b) + i*sin(b)), which could be inf or nan
    // If b is also inf/nan, result is nan
    if (isnan(z.y) || isinf(z.y)) {
      return (float2)(INFINITY, NAN);
    }
    return (float2)(INFINITY * cos(z.y), INFINITY * sin(z.y));
  }
  float er = exp(z.x);
  return (float2)(er * cos(z.y), er * sin(z.y));
}

inline float2 LogAddExp_float2(float2 x, float2 y) {
  // Check for NaN
  if (isnan(x.x) || isnan(x.y) || isnan(y.x) || isnan(y.y)) {
    return (float2)(NAN, NAN);
  }
  float2 exp_x = complex_exp(x);
  float2 exp_y = complex_exp(y);
  float2 sum = exp_x + exp_y;
  float mag = sqrt(sum.x * sum.x + sum.y * sum.y);
  float arg = atan2(sum.y, sum.x);
  return (float2)(log(mag), arg);
}

// Default LogAddExp macro for float - complex uses different kernel
#define LogAddExp(a, b) LogAddExp_float((a), (b))
#define ArcTan2(a, b) (atan2((a), (b)))

)";
}

} // namespace mlx::core::opencl
