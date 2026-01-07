// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/copy/copy.cuh"
#include "mlx/backend/cuda/utils.h"
#include "mlx/dtype.h"

#include <cooperative_groups.h>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

template <typename In, typename Out, typename IdxT, int N_READS>
__global__ void copy_s(const In* in, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = cast_to<Out>(in[0]);
    }
  } else {
    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = cast_to<Out>(in[0]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

template <typename In, typename Out, typename IdxT, int N_READS>
__global__ void copy_v(const In* in, Out* out, IdxT size) {
  IdxT index = cg::this_grid().thread_rank();

  if ((index + 1) * N_READS > size) {
    for (IdxT i = index * N_READS; i < size; ++i) {
      out[i] = cast_to<Out>(in[i]);
    }
  } else {
    auto in_vec = load_vector<N_READS>(in, index);

    AlignedVector<Out, N_READS> out_vec;
#pragma unroll
    for (int i = 0; i < N_READS; ++i) {
      out_vec[i] = cast_to<Out>(in_vec[i]);
    }

    store_vector<N_READS>(out, index, out_vec);
  }
}

} // namespace cu

// Kernel pointer accessor functions - these are the SINGLE POINT where kernel
// addresses are taken. This ensures CUDA registration happens here and the same
// addresses are used everywhere.
//
// The key insight: NVCC registers kernels when their address is taken. If we
// take the address in multiple template instantiation contexts, we get multiple
// registrations of different instantiations that don't match. By centralizing
// address-taking here, we ensure consistent kernel pointers.

// Define these as global volatile pointers to force instantiation and prevent
// optimization Note: NOT in anonymous namespace so they can be accessed by
// register_copy_contiguous_kernels

// === copy_s kernel pointers ===
// sizeof=1 types (N_READS=16)
volatile void* g_copy_s_bool_bool_u32 =
    reinterpret_cast<void*>(&cu::copy_s<bool, bool, uint32_t, 16>);
volatile void* g_copy_s_i8_i8_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int8_t, int8_t, uint32_t, 16>);
volatile void* g_copy_s_u8_u8_u32 =
    reinterpret_cast<void*>(&cu::copy_s<uint8_t, uint8_t, uint32_t, 16>);
// sizeof=2 types (N_READS=8)
volatile void* g_copy_s_i16_i16_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int16_t, int16_t, uint32_t, 8>);
volatile void* g_copy_s_u16_u16_u32 =
    reinterpret_cast<void*>(&cu::copy_s<uint16_t, uint16_t, uint32_t, 8>);
volatile void* g_copy_s_half_half_u32 =
    reinterpret_cast<void*>(&cu::copy_s<__half, __half, uint32_t, 8>);
volatile void* g_copy_s_bf16_bf16_u32 = reinterpret_cast<void*>(
    &cu::copy_s<__nv_bfloat16, __nv_bfloat16, uint32_t, 8>);
// sizeof=4 types (N_READS=4)
volatile void* g_copy_s_i32_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int32_t, int32_t, uint32_t, 4>);
volatile void* g_copy_s_u32_u32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<uint32_t, uint32_t, uint32_t, 4>);
volatile void* g_copy_s_float_float_u32 =
    reinterpret_cast<void*>(&cu::copy_s<float, float, uint32_t, 4>);
// sizeof=8 types (N_READS=2)
volatile void* g_copy_s_i64_i64_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int64_t, int64_t, uint32_t, 2>);
volatile void* g_copy_s_u64_u64_u32 =
    reinterpret_cast<void*>(&cu::copy_s<uint64_t, uint64_t, uint32_t, 2>);
volatile void* g_copy_s_double_double_u32 =
    reinterpret_cast<void*>(&cu::copy_s<double, double, uint32_t, 2>);
volatile void* g_copy_s_c64_c64_u32 = reinterpret_cast<void*>(
    &cu::copy_s<cu::complex64_t, cu::complex64_t, uint32_t, 2>);

// === copy_v kernel pointers ===
// sizeof=1 types (N_READS=16)
volatile void* g_copy_v_bool_bool_u32 =
    reinterpret_cast<void*>(&cu::copy_v<bool, bool, uint32_t, 16>);
volatile void* g_copy_v_i8_i8_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int8_t, int8_t, uint32_t, 16>);
volatile void* g_copy_v_u8_u8_u32 =
    reinterpret_cast<void*>(&cu::copy_v<uint8_t, uint8_t, uint32_t, 16>);
// sizeof=2 types (N_READS=8)
volatile void* g_copy_v_i16_i16_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int16_t, int16_t, uint32_t, 8>);
volatile void* g_copy_v_u16_u16_u32 =
    reinterpret_cast<void*>(&cu::copy_v<uint16_t, uint16_t, uint32_t, 8>);
volatile void* g_copy_v_half_half_u32 =
    reinterpret_cast<void*>(&cu::copy_v<__half, __half, uint32_t, 8>);
volatile void* g_copy_v_bf16_bf16_u32 = reinterpret_cast<void*>(
    &cu::copy_v<__nv_bfloat16, __nv_bfloat16, uint32_t, 8>);
// sizeof=4 types (N_READS=4)
volatile void* g_copy_v_i32_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int32_t, int32_t, uint32_t, 4>);
volatile void* g_copy_v_u32_u32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<uint32_t, uint32_t, uint32_t, 4>);
volatile void* g_copy_v_float_float_u32 =
    reinterpret_cast<void*>(&cu::copy_v<float, float, uint32_t, 4>);
// sizeof=8 types (N_READS=2)
volatile void* g_copy_v_i64_i64_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int64_t, int64_t, uint32_t, 2>);
volatile void* g_copy_v_u64_u64_u32 =
    reinterpret_cast<void*>(&cu::copy_v<uint64_t, uint64_t, uint32_t, 2>);
volatile void* g_copy_v_double_double_u32 =
    reinterpret_cast<void*>(&cu::copy_v<double, double, uint32_t, 2>);
volatile void* g_copy_v_c64_c64_u32 = reinterpret_cast<void*>(
    &cu::copy_v<cu::complex64_t, cu::complex64_t, uint32_t, 2>);

// Type conversion kernels
volatile void* g_copy_v_i32_float_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int32_t, float, uint32_t, 4>);
volatile void* g_copy_v_float_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<float, int32_t, uint32_t, 4>);
volatile void* g_copy_s_i32_float_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int32_t, float, uint32_t, 4>);
volatile void* g_copy_s_float_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<float, int32_t, uint32_t, 4>);
volatile void* g_copy_v_float_half_u32 =
    reinterpret_cast<void*>(&cu::copy_v<float, __half, uint32_t, 4>);
volatile void* g_copy_v_half_float_u32 =
    reinterpret_cast<void*>(&cu::copy_v<__half, float, uint32_t, 8>);
volatile void* g_copy_v_float_bf16_u32 =
    reinterpret_cast<void*>(&cu::copy_v<float, __nv_bfloat16, uint32_t, 4>);
volatile void* g_copy_v_bf16_float_u32 =
    reinterpret_cast<void*>(&cu::copy_v<__nv_bfloat16, float, uint32_t, 8>);
// bool <-> float conversions
volatile void* g_copy_v_bool_float_u32 =
    reinterpret_cast<void*>(&cu::copy_v<bool, float, uint32_t, 16>);
volatile void* g_copy_v_float_bool_u32 =
    reinterpret_cast<void*>(&cu::copy_v<float, bool, uint32_t, 4>);
volatile void* g_copy_s_bool_float_u32 =
    reinterpret_cast<void*>(&cu::copy_s<bool, float, uint32_t, 16>);
volatile void* g_copy_s_float_bool_u32 =
    reinterpret_cast<void*>(&cu::copy_s<float, bool, uint32_t, 4>);
// int32 <-> bool conversions (for where/select operations)
volatile void* g_copy_v_i32_bool_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int32_t, bool, uint32_t, 4>);
volatile void* g_copy_v_bool_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<bool, int32_t, uint32_t, 16>);
volatile void* g_copy_s_i32_bool_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int32_t, bool, uint32_t, 4>);
volatile void* g_copy_s_bool_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<bool, int32_t, uint32_t, 16>);
// uint8/int8 <-> float conversions (for random number generation)
volatile void* g_copy_v_u8_float_u32 =
    reinterpret_cast<void*>(&cu::copy_v<uint8_t, float, uint32_t, 16>);
volatile void* g_copy_v_i8_float_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int8_t, float, uint32_t, 16>);
volatile void* g_copy_v_float_u8_u32 =
    reinterpret_cast<void*>(&cu::copy_v<float, uint8_t, uint32_t, 4>);
volatile void* g_copy_v_float_i8_u32 =
    reinterpret_cast<void*>(&cu::copy_v<float, int8_t, uint32_t, 4>);
// uint32 <-> float conversions
volatile void* g_copy_v_u32_float_u32 =
    reinterpret_cast<void*>(&cu::copy_v<uint32_t, float, uint32_t, 4>);
volatile void* g_copy_v_float_u32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<float, uint32_t, uint32_t, 4>);
volatile void* g_copy_s_u32_float_u32 =
    reinterpret_cast<void*>(&cu::copy_s<uint32_t, float, uint32_t, 4>);
volatile void* g_copy_s_float_u32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<float, uint32_t, uint32_t, 4>);
// float32 <-> complex64 conversions (for type promotion in binary ops)
// float32 has N_READS=4, complex64 has N_READS=2
volatile void* g_copy_s_float_c64_u32 =
    reinterpret_cast<void*>(&cu::copy_s<float, cu::complex64_t, uint32_t, 4>);
volatile void* g_copy_s_c64_float_u32 =
    reinterpret_cast<void*>(&cu::copy_s<cu::complex64_t, float, uint32_t, 2>);
volatile void* g_copy_v_float_c64_u32 =
    reinterpret_cast<void*>(&cu::copy_v<float, cu::complex64_t, uint32_t, 4>);
volatile void* g_copy_v_c64_float_u32 =
    reinterpret_cast<void*>(&cu::copy_v<cu::complex64_t, float, uint32_t, 2>);
// int32 <-> int64 conversions
volatile void* g_copy_s_i32_i64_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int32_t, int64_t, uint32_t, 4>);
volatile void* g_copy_s_i64_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int64_t, int32_t, uint32_t, 2>);
volatile void* g_copy_v_i32_i64_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int32_t, int64_t, uint32_t, 4>);
volatile void* g_copy_v_i64_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int64_t, int32_t, uint32_t, 2>);
// int32 <-> uint32 conversions
volatile void* g_copy_s_i32_u32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int32_t, uint32_t, uint32_t, 4>);
volatile void* g_copy_s_u32_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<uint32_t, int32_t, uint32_t, 4>);
volatile void* g_copy_v_i32_u32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int32_t, uint32_t, uint32_t, 4>);
volatile void* g_copy_v_u32_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<uint32_t, int32_t, uint32_t, 4>);
// float64 <-> integer conversions (for full() with float literals)
// float64 (double) has N_READS=2, int32/uint32 have N_READS=4
volatile void* g_copy_s_double_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<double, int32_t, uint32_t, 2>);
volatile void* g_copy_s_double_u32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<double, uint32_t, uint32_t, 2>);
volatile void* g_copy_s_double_i64_u32 =
    reinterpret_cast<void*>(&cu::copy_s<double, int64_t, uint32_t, 2>);
volatile void* g_copy_s_double_u64_u32 =
    reinterpret_cast<void*>(&cu::copy_s<double, uint64_t, uint32_t, 2>);
volatile void* g_copy_s_double_bool_u32 =
    reinterpret_cast<void*>(&cu::copy_s<double, bool, uint32_t, 2>);
volatile void* g_copy_s_double_float_u32 =
    reinterpret_cast<void*>(&cu::copy_s<double, float, uint32_t, 2>);
volatile void* g_copy_v_double_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<double, int32_t, uint32_t, 2>);
volatile void* g_copy_v_double_u32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<double, uint32_t, uint32_t, 2>);
volatile void* g_copy_v_double_i64_u32 =
    reinterpret_cast<void*>(&cu::copy_v<double, int64_t, uint32_t, 2>);
volatile void* g_copy_v_double_float_u32 =
    reinterpret_cast<void*>(&cu::copy_v<double, float, uint32_t, 2>);
// int32/uint32 -> float64
volatile void* g_copy_s_i32_double_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int32_t, double, uint32_t, 4>);
volatile void* g_copy_s_u32_double_u32 =
    reinterpret_cast<void*>(&cu::copy_s<uint32_t, double, uint32_t, 4>);
volatile void* g_copy_v_i32_double_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int32_t, double, uint32_t, 4>);
volatile void* g_copy_v_u32_double_u32 =
    reinterpret_cast<void*>(&cu::copy_v<uint32_t, double, uint32_t, 4>);
// uint32 <-> int64 conversions
volatile void* g_copy_s_u32_i64_u32 =
    reinterpret_cast<void*>(&cu::copy_s<uint32_t, int64_t, uint32_t, 4>);
volatile void* g_copy_v_u32_i64_u32 =
    reinterpret_cast<void*>(&cu::copy_v<uint32_t, int64_t, uint32_t, 4>);
volatile void* g_copy_s_i64_u32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int64_t, uint32_t, uint32_t, 2>);
volatile void* g_copy_v_i64_u32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int64_t, uint32_t, uint32_t, 2>);
// int64 <-> int32 scalar versions are already registered above
// (copy_s_i64_i32_u32, copy_s_i32_i64_u32) int64 <-> uint64
volatile void* g_copy_s_i64_u64_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int64_t, uint64_t, uint32_t, 2>);
volatile void* g_copy_v_i64_u64_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int64_t, uint64_t, uint32_t, 2>);
volatile void* g_copy_s_u64_i64_u32 =
    reinterpret_cast<void*>(&cu::copy_s<uint64_t, int64_t, uint32_t, 2>);
volatile void* g_copy_v_u64_i64_u32 =
    reinterpret_cast<void*>(&cu::copy_v<uint64_t, int64_t, uint32_t, 2>);
// int32 <-> uint8 conversions (for reduction output type widening)
// int32 has N_READS=4, uint8 has N_READS=16
volatile void* g_copy_s_i32_u8_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int32_t, uint8_t, uint32_t, 4>);
volatile void* g_copy_v_i32_u8_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int32_t, uint8_t, uint32_t, 4>);
volatile void* g_copy_s_u8_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<uint8_t, int32_t, uint32_t, 16>);
volatile void* g_copy_v_u8_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<uint8_t, int32_t, uint32_t, 16>);
// int32 <-> uint16 conversions
// int32 has N_READS=4, uint16 has N_READS=8
volatile void* g_copy_s_i32_u16_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int32_t, uint16_t, uint32_t, 4>);
volatile void* g_copy_v_i32_u16_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int32_t, uint16_t, uint32_t, 4>);
volatile void* g_copy_s_u16_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<uint16_t, int32_t, uint32_t, 8>);
volatile void* g_copy_v_u16_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<uint16_t, int32_t, uint32_t, 8>);
// int32 <-> int8 conversions
volatile void* g_copy_s_i32_i8_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int32_t, int8_t, uint32_t, 4>);
volatile void* g_copy_v_i32_i8_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int32_t, int8_t, uint32_t, 4>);
volatile void* g_copy_s_i8_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int8_t, int32_t, uint32_t, 16>);
volatile void* g_copy_v_i8_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int8_t, int32_t, uint32_t, 16>);
// int32 <-> int16 conversions
volatile void* g_copy_s_i32_i16_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int32_t, int16_t, uint32_t, 4>);
volatile void* g_copy_v_i32_i16_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int32_t, int16_t, uint32_t, 4>);
volatile void* g_copy_s_i16_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int16_t, int32_t, uint32_t, 8>);
volatile void* g_copy_v_i16_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int16_t, int32_t, uint32_t, 8>);
// bool <-> uint8 conversions (for bitwise shift operations)
volatile void* g_copy_s_bool_u8_u32 =
    reinterpret_cast<void*>(&cu::copy_s<bool, uint8_t, uint32_t, 16>);
volatile void* g_copy_v_bool_u8_u32 =
    reinterpret_cast<void*>(&cu::copy_v<bool, uint8_t, uint32_t, 16>);
volatile void* g_copy_s_u8_bool_u32 =
    reinterpret_cast<void*>(&cu::copy_s<uint8_t, bool, uint32_t, 16>);
volatile void* g_copy_v_u8_bool_u32 =
    reinterpret_cast<void*>(&cu::copy_v<uint8_t, bool, uint32_t, 16>);
// float32 <-> int16 conversions (for full_like)
volatile void* g_copy_s_float_i16_u32 =
    reinterpret_cast<void*>(&cu::copy_s<float, int16_t, uint32_t, 4>);
volatile void* g_copy_v_float_i16_u32 =
    reinterpret_cast<void*>(&cu::copy_v<float, int16_t, uint32_t, 4>);
volatile void* g_copy_s_i16_float_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int16_t, float, uint32_t, 8>);
volatile void* g_copy_v_i16_float_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int16_t, float, uint32_t, 8>);
// int32 <-> complex64 (for FFT)
// int32 has N_READS=4, complex64 has N_READS=2
volatile void* g_copy_s_i32_c64_u32 =
    reinterpret_cast<void*>(&cu::copy_s<int32_t, cu::complex64_t, uint32_t, 4>);
volatile void* g_copy_v_i32_c64_u32 =
    reinterpret_cast<void*>(&cu::copy_v<int32_t, cu::complex64_t, uint32_t, 4>);
volatile void* g_copy_s_c64_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_s<cu::complex64_t, int32_t, uint32_t, 2>);
volatile void* g_copy_v_c64_i32_u32 =
    reinterpret_cast<void*>(&cu::copy_v<cu::complex64_t, int32_t, uint32_t, 2>);

// Get kernel pointer for copy_s - uses the global volatile pointers
// NOTE: On Windows/MSVC, template type dispatch through nested lambdas fails.
// The runtime dtype-based selector in copy_contiguous() should be used instead.
// This function is kept as a fallback for non-Windows platforms.
template <typename InType, typename OutType, typename IdxT, int N_READS>
void* get_copy_s_kernel() {
  // Dispatch to the correct global pointer based on type
  // Use separate if statements with early return to satisfy MSVC's return value
  // checking
  if constexpr (
      std::is_same_v<InType, float> && std::is_same_v<OutType, float> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_s_float_float_u32);
  }
  if constexpr (
      std::is_same_v<InType, int32_t> && std::is_same_v<OutType, int32_t> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_s_i32_i32_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint32_t> && std::is_same_v<OutType, uint32_t> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_s_u32_u32_u32);
  }
  if constexpr (
      std::is_same_v<InType, int64_t> && std::is_same_v<OutType, int64_t> &&
      N_READS == 2) {
    return const_cast<void*>(g_copy_s_i64_i64_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint64_t> && std::is_same_v<OutType, uint64_t> &&
      N_READS == 2) {
    return const_cast<void*>(g_copy_s_u64_u64_u32);
  }
  if constexpr (
      std::is_same_v<InType, double> && std::is_same_v<OutType, double> &&
      N_READS == 2) {
    return const_cast<void*>(g_copy_s_double_double_u32);
  }
  if constexpr (
      std::is_same_v<InType, bool> && std::is_same_v<OutType, bool> &&
      N_READS == 16) {
    return const_cast<void*>(g_copy_s_bool_bool_u32);
  }
  if constexpr (
      std::is_same_v<InType, int8_t> && std::is_same_v<OutType, int8_t> &&
      N_READS == 16) {
    return const_cast<void*>(g_copy_s_i8_i8_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint8_t> && std::is_same_v<OutType, uint8_t> &&
      N_READS == 16) {
    return const_cast<void*>(g_copy_s_u8_u8_u32);
  }
  if constexpr (
      std::is_same_v<InType, int16_t> && std::is_same_v<OutType, int16_t> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_s_i16_i16_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint16_t> && std::is_same_v<OutType, uint16_t> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_s_u16_u16_u32);
  }
  if constexpr (
      std::is_same_v<InType, __half> && std::is_same_v<OutType, __half> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_s_half_half_u32);
  }
  if constexpr (
      std::is_same_v<InType, __nv_bfloat16> &&
      std::is_same_v<OutType, __nv_bfloat16> && N_READS == 8) {
    return const_cast<void*>(g_copy_s_bf16_bf16_u32);
  }
  if constexpr (
      std::is_same_v<InType, cu::complex64_t> &&
      std::is_same_v<OutType, cu::complex64_t> && N_READS == 2) {
    return const_cast<void*>(g_copy_s_c64_c64_u32);
  }
  if constexpr (
      std::is_same_v<InType, int32_t> && std::is_same_v<OutType, float> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_s_i32_float_u32);
  }
  if constexpr (
      std::is_same_v<InType, float> && std::is_same_v<OutType, int32_t> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_s_float_i32_u32);
  }
  // Fallback: use direct kernel address (may not be registered on Windows due
  // to MSVC template issues)
  return reinterpret_cast<void*>(&cu::copy_s<InType, OutType, IdxT, N_READS>);
}

// Get kernel pointer for copy_v - uses the global volatile pointers
template <typename InType, typename OutType, typename IdxT, int N_READS>
void* get_copy_v_kernel() {
  // Use separate if statements with early return to satisfy MSVC's return value
  // checking
  if constexpr (
      std::is_same_v<InType, float> && std::is_same_v<OutType, float> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_float_float_u32);
  }
  if constexpr (
      std::is_same_v<InType, int32_t> && std::is_same_v<OutType, int32_t> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_i32_i32_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint32_t> && std::is_same_v<OutType, uint32_t> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_u32_u32_u32);
  }
  if constexpr (
      std::is_same_v<InType, int64_t> && std::is_same_v<OutType, int64_t> &&
      N_READS == 2) {
    return const_cast<void*>(g_copy_v_i64_i64_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint64_t> && std::is_same_v<OutType, uint64_t> &&
      N_READS == 2) {
    return const_cast<void*>(g_copy_v_u64_u64_u32);
  }
  if constexpr (
      std::is_same_v<InType, double> && std::is_same_v<OutType, double> &&
      N_READS == 2) {
    return const_cast<void*>(g_copy_v_double_double_u32);
  }
  if constexpr (
      std::is_same_v<InType, bool> && std::is_same_v<OutType, bool> &&
      N_READS == 16) {
    return const_cast<void*>(g_copy_v_bool_bool_u32);
  }
  if constexpr (
      std::is_same_v<InType, int8_t> && std::is_same_v<OutType, int8_t> &&
      N_READS == 16) {
    return const_cast<void*>(g_copy_v_i8_i8_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint8_t> && std::is_same_v<OutType, uint8_t> &&
      N_READS == 16) {
    return const_cast<void*>(g_copy_v_u8_u8_u32);
  }
  if constexpr (
      std::is_same_v<InType, int16_t> && std::is_same_v<OutType, int16_t> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_v_i16_i16_u32);
  }
  if constexpr (
      std::is_same_v<InType, uint16_t> && std::is_same_v<OutType, uint16_t> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_v_u16_u16_u32);
  }
  if constexpr (
      std::is_same_v<InType, __half> && std::is_same_v<OutType, __half> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_v_half_half_u32);
  }
  if constexpr (
      std::is_same_v<InType, __nv_bfloat16> &&
      std::is_same_v<OutType, __nv_bfloat16> && N_READS == 8) {
    return const_cast<void*>(g_copy_v_bf16_bf16_u32);
  }
  if constexpr (
      std::is_same_v<InType, cu::complex64_t> &&
      std::is_same_v<OutType, cu::complex64_t> && N_READS == 2) {
    return const_cast<void*>(g_copy_v_c64_c64_u32);
  }
  if constexpr (
      std::is_same_v<InType, int32_t> && std::is_same_v<OutType, float> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_i32_float_u32);
  }
  if constexpr (
      std::is_same_v<InType, float> && std::is_same_v<OutType, int32_t> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_float_i32_u32);
  }
  if constexpr (
      std::is_same_v<InType, float> && std::is_same_v<OutType, __half> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_float_half_u32);
  }
  if constexpr (
      std::is_same_v<InType, __half> && std::is_same_v<OutType, float> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_v_half_float_u32);
  }
  if constexpr (
      std::is_same_v<InType, float> && std::is_same_v<OutType, __nv_bfloat16> &&
      N_READS == 4) {
    return const_cast<void*>(g_copy_v_float_bf16_u32);
  }
  if constexpr (
      std::is_same_v<InType, __nv_bfloat16> && std::is_same_v<OutType, float> &&
      N_READS == 8) {
    return const_cast<void*>(g_copy_v_bf16_float_u32);
  }
  // Fallback: use direct kernel address (may not be registered on Windows)
  return reinterpret_cast<void*>(&cu::copy_v<InType, OutType, IdxT, N_READS>);
}

// Helper function to launch copy kernel with explicit template parameters
// This avoids MSVC issues with constexpr in nested lambda contexts
// NOTE: On Windows, use the runtime dtype-based path in copy_contiguous()
// instead.
template <typename InType, typename OutType, typename IdxT, int N_READS>
void launch_copy_contiguous_kernel(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const InType* in_ptr,
    OutType* out_ptr,
    size_t data_size,
    const Shape& shape,
    const Strides& strides,
    bool large) {
  auto kernel = get_copy_s_kernel<InType, OutType, IdxT, N_READS>();
  if (ctype == CopyType::Vector) {
    kernel = get_copy_v_kernel<InType, OutType, IdxT, N_READS>();
  }

  auto [num_blocks, block_dims] =
      get_launch_args(data_size, shape, strides, large, N_READS);
  IdxT size_param = static_cast<IdxT>(data_size);
  void* params[] = {(void*)&in_ptr, (void*)&out_ptr, (void*)&size_param};
  encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
}

// Dispatch on N_READS at runtime to avoid MSVC constexpr issues
template <typename InType, typename OutType, typename IdxT>
void dispatch_copy_contiguous_n_reads(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const InType* in_ptr,
    OutType* out_ptr,
    size_t data_size,
    const Shape& shape,
    const Strides& strides,
    bool large,
    size_t type_size) {
  // Dispatch based on runtime type_size value
  // N_READS = 16 / sizeof(InType), so:
  // sizeof=1 -> N_READS=16, sizeof=2 -> N_READS=8, sizeof=4 -> N_READS=4,
  // sizeof=8 -> N_READS=2, sizeof=16+ -> N_READS=1
  if (type_size == 1) {
    launch_copy_contiguous_kernel<InType, OutType, IdxT, 16>(
        encoder, ctype, in_ptr, out_ptr, data_size, shape, strides, large);
  } else if (type_size == 2) {
    launch_copy_contiguous_kernel<InType, OutType, IdxT, 8>(
        encoder, ctype, in_ptr, out_ptr, data_size, shape, strides, large);
  } else if (type_size == 4) {
    launch_copy_contiguous_kernel<InType, OutType, IdxT, 4>(
        encoder, ctype, in_ptr, out_ptr, data_size, shape, strides, large);
  } else if (type_size == 8) {
    launch_copy_contiguous_kernel<InType, OutType, IdxT, 2>(
        encoder, ctype, in_ptr, out_ptr, data_size, shape, strides, large);
  } else {
    launch_copy_contiguous_kernel<InType, OutType, IdxT, 1>(
        encoder, ctype, in_ptr, out_ptr, data_size, shape, strides, large);
  }
}

// Runtime kernel selector that uses dtype enum values directly instead of
// template type matching This avoids MSVC issues with type deduction through
// nested lambdas
void* get_copy_kernel_by_dtype(
    Dtype in_dtype,
    Dtype out_dtype,
    CopyType ctype,
    bool large) {
  // For now, handle the common same-type case (in_dtype == out_dtype)
  // These are the most common cases used by Full primitive and other operations
  if (in_dtype == out_dtype && !large) {
    if (ctype == CopyType::Scalar) {
      switch (in_dtype.val()) {
        case bool_:
          return const_cast<void*>(g_copy_s_bool_bool_u32);
        case int8:
          return const_cast<void*>(g_copy_s_i8_i8_u32);
        case uint8:
          return const_cast<void*>(g_copy_s_u8_u8_u32);
        case int16:
          return const_cast<void*>(g_copy_s_i16_i16_u32);
        case uint16:
          return const_cast<void*>(g_copy_s_u16_u16_u32);
        case float16:
          return const_cast<void*>(g_copy_s_half_half_u32);
        case bfloat16:
          return const_cast<void*>(g_copy_s_bf16_bf16_u32);
        case int32:
          return const_cast<void*>(g_copy_s_i32_i32_u32);
        case uint32:
          return const_cast<void*>(g_copy_s_u32_u32_u32);
        case float32:
          return const_cast<void*>(g_copy_s_float_float_u32);
        case int64:
          return const_cast<void*>(g_copy_s_i64_i64_u32);
        case uint64:
          return const_cast<void*>(g_copy_s_u64_u64_u32);
        case float64:
          return const_cast<void*>(g_copy_s_double_double_u32);
        case complex64:
          return const_cast<void*>(g_copy_s_c64_c64_u32);
        default:
          break;
      }
    } else { // CopyType::Vector
      switch (in_dtype.val()) {
        case bool_:
          return const_cast<void*>(g_copy_v_bool_bool_u32);
        case int8:
          return const_cast<void*>(g_copy_v_i8_i8_u32);
        case uint8:
          return const_cast<void*>(g_copy_v_u8_u8_u32);
        case int16:
          return const_cast<void*>(g_copy_v_i16_i16_u32);
        case uint16:
          return const_cast<void*>(g_copy_v_u16_u16_u32);
        case float16:
          return const_cast<void*>(g_copy_v_half_half_u32);
        case bfloat16:
          return const_cast<void*>(g_copy_v_bf16_bf16_u32);
        case int32:
          return const_cast<void*>(g_copy_v_i32_i32_u32);
        case uint32:
          return const_cast<void*>(g_copy_v_u32_u32_u32);
        case float32:
          return const_cast<void*>(g_copy_v_float_float_u32);
        case int64:
          return const_cast<void*>(g_copy_v_i64_i64_u32);
        case uint64:
          return const_cast<void*>(g_copy_v_u64_u64_u32);
        case float64:
          return const_cast<void*>(g_copy_v_double_double_u32);
        case complex64:
          return const_cast<void*>(g_copy_v_c64_c64_u32);
        default:
          break;
      }
    }
  }
  // Handle some common cross-type conversions
  if (!large) {
    // int32 <-> float
    if (in_dtype.val() == int32 && out_dtype.val() == float32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i32_float_u32)
          : const_cast<void*>(g_copy_v_i32_float_u32);
    }
    if (in_dtype.val() == float32 && out_dtype.val() == int32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_float_i32_u32)
          : const_cast<void*>(g_copy_v_float_i32_u32);
    }
    // bool <-> float
    if (in_dtype.val() == bool_ && out_dtype.val() == float32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_bool_float_u32)
          : const_cast<void*>(g_copy_v_bool_float_u32);
    }
    if (in_dtype.val() == float32 && out_dtype.val() == bool_) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_float_bool_u32)
          : const_cast<void*>(g_copy_v_float_bool_u32);
    }
    // int32 <-> bool (for where/select operations)
    if (in_dtype.val() == int32 && out_dtype.val() == bool_) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i32_bool_u32)
          : const_cast<void*>(g_copy_v_i32_bool_u32);
    }
    if (in_dtype.val() == bool_ && out_dtype.val() == int32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_bool_i32_u32)
          : const_cast<void*>(g_copy_v_bool_i32_u32);
    }
    // float <-> half
    if (in_dtype.val() == float32 && out_dtype.val() == float16) {
      return ctype == CopyType::Vector
          ? const_cast<void*>(g_copy_v_float_half_u32)
          : nullptr;
    }
    if (in_dtype.val() == float16 && out_dtype.val() == float32) {
      return ctype == CopyType::Vector
          ? const_cast<void*>(g_copy_v_half_float_u32)
          : nullptr;
    }
    // float <-> bf16
    if (in_dtype.val() == float32 && out_dtype.val() == bfloat16) {
      return ctype == CopyType::Vector
          ? const_cast<void*>(g_copy_v_float_bf16_u32)
          : nullptr;
    }
    if (in_dtype.val() == bfloat16 && out_dtype.val() == float32) {
      return ctype == CopyType::Vector
          ? const_cast<void*>(g_copy_v_bf16_float_u32)
          : nullptr;
    }
    // uint8/int8 <-> float (for random number generation)
    if (in_dtype.val() == uint8 && out_dtype.val() == float32) {
      return ctype == CopyType::Vector
          ? const_cast<void*>(g_copy_v_u8_float_u32)
          : nullptr;
    }
    if (in_dtype.val() == int8 && out_dtype.val() == float32) {
      return ctype == CopyType::Vector
          ? const_cast<void*>(g_copy_v_i8_float_u32)
          : nullptr;
    }
    if (in_dtype.val() == float32 && out_dtype.val() == uint8) {
      return ctype == CopyType::Vector
          ? const_cast<void*>(g_copy_v_float_u8_u32)
          : nullptr;
    }
    if (in_dtype.val() == float32 && out_dtype.val() == int8) {
      return ctype == CopyType::Vector
          ? const_cast<void*>(g_copy_v_float_i8_u32)
          : nullptr;
    }
    // uint32 <-> float
    if (in_dtype.val() == uint32 && out_dtype.val() == float32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_u32_float_u32)
          : const_cast<void*>(g_copy_v_u32_float_u32);
    }
    if (in_dtype.val() == float32 && out_dtype.val() == uint32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_float_u32_u32)
          : const_cast<void*>(g_copy_v_float_u32_u32);
    }
    // float32 <-> complex64 (for type promotion in binary ops with complex)
    if (in_dtype.val() == float32 && out_dtype.val() == complex64) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_float_c64_u32)
          : const_cast<void*>(g_copy_v_float_c64_u32);
    }
    if (in_dtype.val() == complex64 && out_dtype.val() == float32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_c64_float_u32)
          : const_cast<void*>(g_copy_v_c64_float_u32);
    }
    // int32 <-> int64
    if (in_dtype.val() == int32 && out_dtype.val() == int64) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i32_i64_u32)
          : const_cast<void*>(g_copy_v_i32_i64_u32);
    }
    if (in_dtype.val() == int64 && out_dtype.val() == int32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i64_i32_u32)
          : const_cast<void*>(g_copy_v_i64_i32_u32);
    }
    // int32 <-> uint32
    if (in_dtype.val() == int32 && out_dtype.val() == uint32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i32_u32_u32)
          : const_cast<void*>(g_copy_v_i32_u32_u32);
    }
    if (in_dtype.val() == uint32 && out_dtype.val() == int32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_u32_i32_u32)
          : const_cast<void*>(g_copy_v_u32_i32_u32);
    }
    // float64 -> various types (for full() with float literals like 1.0)
    if (in_dtype.val() == float64 && out_dtype.val() == int32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_double_i32_u32)
          : const_cast<void*>(g_copy_v_double_i32_u32);
    }
    if (in_dtype.val() == float64 && out_dtype.val() == uint32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_double_u32_u32)
          : const_cast<void*>(g_copy_v_double_u32_u32);
    }
    if (in_dtype.val() == float64 && out_dtype.val() == int64) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_double_i64_u32)
          : const_cast<void*>(g_copy_v_double_i64_u32);
    }
    if (in_dtype.val() == float64 && out_dtype.val() == uint64) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_double_u64_u32)
          : nullptr; // vector version not registered
    }
    if (in_dtype.val() == float64 && out_dtype.val() == bool_) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_double_bool_u32)
          : nullptr; // vector version not registered
    }
    if (in_dtype.val() == float64 && out_dtype.val() == float32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_double_float_u32)
          : const_cast<void*>(g_copy_v_double_float_u32);
    }
    // int32/uint32 -> float64
    if (in_dtype.val() == int32 && out_dtype.val() == float64) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i32_double_u32)
          : const_cast<void*>(g_copy_v_i32_double_u32);
    }
    if (in_dtype.val() == uint32 && out_dtype.val() == float64) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_u32_double_u32)
          : const_cast<void*>(g_copy_v_u32_double_u32);
    }
    // uint32 <-> int64
    if (in_dtype.val() == uint32 && out_dtype.val() == int64) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_u32_i64_u32)
          : const_cast<void*>(g_copy_v_u32_i64_u32);
    }
    if (in_dtype.val() == int64 && out_dtype.val() == uint32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i64_u32_u32)
          : const_cast<void*>(g_copy_v_i64_u32_u32);
    }
    // int64 <-> int32
    if (in_dtype.val() == int64 && out_dtype.val() == int32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i64_i32_u32)
          : const_cast<void*>(g_copy_v_i64_i32_u32);
    }
    if (in_dtype.val() == int32 && out_dtype.val() == int64) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i32_i64_u32)
          : const_cast<void*>(g_copy_v_i32_i64_u32);
    }
    // int64 <-> uint64
    if (in_dtype.val() == int64 && out_dtype.val() == uint64) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i64_u64_u32)
          : const_cast<void*>(g_copy_v_i64_u64_u32);
    }
    if (in_dtype.val() == uint64 && out_dtype.val() == int64) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_u64_i64_u32)
          : const_cast<void*>(g_copy_v_u64_i64_u32);
    }
    // int32 <-> uint8 (for reduction output type widening)
    if (in_dtype.val() == int32 && out_dtype.val() == uint8) {
      return ctype == CopyType::Scalar ? const_cast<void*>(g_copy_s_i32_u8_u32)
                                       : const_cast<void*>(g_copy_v_i32_u8_u32);
    }
    if (in_dtype.val() == uint8 && out_dtype.val() == int32) {
      return ctype == CopyType::Scalar ? const_cast<void*>(g_copy_s_u8_i32_u32)
                                       : const_cast<void*>(g_copy_v_u8_i32_u32);
    }
    // int32 <-> uint16
    if (in_dtype.val() == int32 && out_dtype.val() == uint16) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i32_u16_u32)
          : const_cast<void*>(g_copy_v_i32_u16_u32);
    }
    if (in_dtype.val() == uint16 && out_dtype.val() == int32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_u16_i32_u32)
          : const_cast<void*>(g_copy_v_u16_i32_u32);
    }
    // int32 <-> int8
    if (in_dtype.val() == int32 && out_dtype.val() == int8) {
      return ctype == CopyType::Scalar ? const_cast<void*>(g_copy_s_i32_i8_u32)
                                       : const_cast<void*>(g_copy_v_i32_i8_u32);
    }
    if (in_dtype.val() == int8 && out_dtype.val() == int32) {
      return ctype == CopyType::Scalar ? const_cast<void*>(g_copy_s_i8_i32_u32)
                                       : const_cast<void*>(g_copy_v_i8_i32_u32);
    }
    // int32 <-> int16
    if (in_dtype.val() == int32 && out_dtype.val() == int16) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i32_i16_u32)
          : const_cast<void*>(g_copy_v_i32_i16_u32);
    }
    if (in_dtype.val() == int16 && out_dtype.val() == int32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i16_i32_u32)
          : const_cast<void*>(g_copy_v_i16_i32_u32);
    }
    // bool <-> uint8 (for bitwise shift operations)
    if (in_dtype.val() == bool_ && out_dtype.val() == uint8) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_bool_u8_u32)
          : const_cast<void*>(g_copy_v_bool_u8_u32);
    }
    if (in_dtype.val() == uint8 && out_dtype.val() == bool_) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_u8_bool_u32)
          : const_cast<void*>(g_copy_v_u8_bool_u32);
    }
    // float32 <-> int16 (for full_like)
    if (in_dtype.val() == float32 && out_dtype.val() == int16) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_float_i16_u32)
          : const_cast<void*>(g_copy_v_float_i16_u32);
    }
    if (in_dtype.val() == int16 && out_dtype.val() == float32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i16_float_u32)
          : const_cast<void*>(g_copy_v_i16_float_u32);
    }
    // int32 <-> complex64 (for FFT)
    if (in_dtype.val() == int32 && out_dtype.val() == complex64) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_i32_c64_u32)
          : const_cast<void*>(g_copy_v_i32_c64_u32);
    }
    if (in_dtype.val() == complex64 && out_dtype.val() == int32) {
      return ctype == CopyType::Scalar
          ? const_cast<void*>(g_copy_s_c64_i32_u32)
          : const_cast<void*>(g_copy_v_c64_i32_u32);
    }
  }
  // Return nullptr if no match - caller will use template fallback
  return nullptr;
}

void copy_contiguous(
    cu::CommandEncoder& encoder,
    CopyType ctype,
    const array& in,
    array& out,
    int64_t in_offset,
    int64_t out_offset) {
#ifdef _MSC_VER
  // On Windows, use runtime-selected registered kernel to work around MSVC
  // template issues. MSVC doesn't properly resolve template types through
  // nested lambdas, causing the template dispatch path to use unregistered
  // kernel addresses.
  bool large = out.data_size() > UINT32_MAX;
  void* kernel =
      get_copy_kernel_by_dtype(in.dtype(), out.dtype(), ctype, large);
  if (kernel != nullptr) {
    auto [num_blocks, block_dims] = get_launch_args(
        out.data_size(),
        out.shape(),
        out.strides(),
        large,
        16 / size_of(in.dtype()));

    // Get raw GPU pointers - must use gpu_ptr, not data<void>() which may
    // copy to managed/host memory
    const void* in_ptr_val = static_cast<const char*>(gpu_ptr<void>(in)) +
        in_offset * size_of(in.dtype());
    void* out_ptr_val = static_cast<char*>(gpu_ptr<void>(out)) +
        out_offset * size_of(out.dtype());
    uint32_t size_param = static_cast<uint32_t>(out.data_size());

    // Store pointers to the pointer values for kernel params
    const void** in_ptr_ptr = &in_ptr_val;
    void** out_ptr_ptr = &out_ptr_val;
    void* params[] = {(void*)in_ptr_ptr, (void*)out_ptr_ptr, &size_param};

    encoder.add_kernel_node(kernel, num_blocks, block_dims, 0, params);
    return;
  }
  // Fall through to template dispatch for unsupported type combinations
#endif

  // Compute type size outside the nested lambdas to avoid MSVC constexpr issues
  size_t in_type_size = size_of(in.dtype());
  dispatch_all_types(in.dtype(), [&](auto in_type_tag) {
    dispatch_all_types(out.dtype(), [&](auto out_type_tag) {
      dispatch_bool(out.data_size() > UINT32_MAX, [&](auto large) {
        using InType = cuda_type_t<MLX_GET_TYPE(in_type_tag)>;
        using OutType = cuda_type_t<MLX_GET_TYPE(out_type_tag)>;
        using IdxT = std::conditional_t<large(), int64_t, uint32_t>;
        const InType* in_ptr = gpu_ptr<InType>(in) + in_offset;
        OutType* out_ptr = gpu_ptr<OutType>(out) + out_offset;
        dispatch_copy_contiguous_n_reads<InType, OutType, IdxT>(
            encoder,
            ctype,
            in_ptr,
            out_ptr,
            out.data_size(),
            out.shape(),
            out.strides(),
            large(),
            in_type_size);
      });
    });
  });
}

} // namespace mlx::core
