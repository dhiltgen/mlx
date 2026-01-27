// Copyright © 2025 MLX Contributors

#pragma once

#include <type_traits>
#include "mlx/dtype.h"
#include "mlx/types/bf16.h"
#include "mlx/types/fp16.h"
#include "mlx/types/complex.h"

namespace mlx::core::opencl {

/**
 * Runtime dtype → compile-time type dispatch utilities.
 *
 * While OpenCL kernels are compiled at runtime (unlike CUDA/Metal templates),
 * these utilities are useful for:
 * 1. Type-safe configuration (sizes, alignment, work-per-thread)
 * 2. Consistent error handling for unsupported types
 * 3. Compile-time type traits in dispatch logic
 *
 * Pattern matches CUDA's dispatch approach for consistency across backends.
 */

// Helper to extract type from type_identity
#define MLX_GET_TYPE(x) typename decltype(x)::type

/**
 * Dispatch on all supported types.
 *
 * @param dt Runtime dtype value
 * @param f Functor accepting std::type_identity<T>
 *
 * Usage:
 * @code
 * dispatch_type(arr.dtype(), [&](auto type_tag) {
 *   using T = MLX_GET_TYPE(type_tag);
 *   size_t alignment = alignof(T);
 *   // ...
 * });
 * @endcode
 */
template <typename F>
void dispatch_type(Dtype dt, F&& f) {
  switch (dt.val()) {
    case Dtype::Val::bool_:    f(std::type_identity<bool>{}); break;
    case Dtype::Val::int8:     f(std::type_identity<int8_t>{}); break;
    case Dtype::Val::int16:    f(std::type_identity<int16_t>{}); break;
    case Dtype::Val::int32:    f(std::type_identity<int32_t>{}); break;
    case Dtype::Val::int64:    f(std::type_identity<int64_t>{}); break;
    case Dtype::Val::uint8:    f(std::type_identity<uint8_t>{}); break;
    case Dtype::Val::uint16:   f(std::type_identity<uint16_t>{}); break;
    case Dtype::Val::uint32:   f(std::type_identity<uint32_t>{}); break;
    case Dtype::Val::uint64:   f(std::type_identity<uint64_t>{}); break;
    case Dtype::Val::float16:  f(std::type_identity<float16_t>{}); break;
    case Dtype::Val::bfloat16: f(std::type_identity<bfloat16_t>{}); break;
    case Dtype::Val::float32:  f(std::type_identity<float>{}); break;
    case Dtype::Val::complex64:f(std::type_identity<complex64_t>{}); break;
    default:
      throw std::runtime_error("[dispatch_type] Unsupported dtype");
  }
}

/**
 * Dispatch on floating-point types only.
 * Throws for integer or complex types.
 *
 * @param dt Runtime dtype value (must be float16, bfloat16, or float32)
 * @param f Functor accepting std::type_identity<T>
 */
template <typename F>
void dispatch_float_type(Dtype dt, F&& f) {
  switch (dt.val()) {
    case Dtype::Val::float16:  f(std::type_identity<float16_t>{}); break;
    case Dtype::Val::bfloat16: f(std::type_identity<bfloat16_t>{}); break;
    case Dtype::Val::float32:  f(std::type_identity<float>{}); break;
    default:
      throw std::runtime_error("[dispatch_float_type] Expected floating-point type");
  }
}

/**
 * Dispatch on integer types only.
 * Includes bool (treated as uint8 in OpenCL).
 *
 * @param dt Runtime dtype value (must be integer type)
 * @param f Functor accepting std::type_identity<T>
 */
template <typename F>
void dispatch_int_type(Dtype dt, F&& f) {
  switch (dt.val()) {
    case Dtype::Val::bool_:    f(std::type_identity<bool>{}); break;
    case Dtype::Val::int8:     f(std::type_identity<int8_t>{}); break;
    case Dtype::Val::int16:    f(std::type_identity<int16_t>{}); break;
    case Dtype::Val::int32:    f(std::type_identity<int32_t>{}); break;
    case Dtype::Val::int64:    f(std::type_identity<int64_t>{}); break;
    case Dtype::Val::uint8:    f(std::type_identity<uint8_t>{}); break;
    case Dtype::Val::uint16:   f(std::type_identity<uint16_t>{}); break;
    case Dtype::Val::uint32:   f(std::type_identity<uint32_t>{}); break;
    case Dtype::Val::uint64:   f(std::type_identity<uint64_t>{}); break;
    default:
      throw std::runtime_error("[dispatch_int_type] Expected integer type");
  }
}

/**
 * Dispatch on real numeric types (float + integer, no complex).
 *
 * @param dt Runtime dtype value
 * @param f Functor accepting std::type_identity<T>
 */
template <typename F>
void dispatch_real_type(Dtype dt, F&& f) {
  switch (dt.val()) {
    case Dtype::Val::bool_:    f(std::type_identity<bool>{}); break;
    case Dtype::Val::int8:     f(std::type_identity<int8_t>{}); break;
    case Dtype::Val::int16:    f(std::type_identity<int16_t>{}); break;
    case Dtype::Val::int32:    f(std::type_identity<int32_t>{}); break;
    case Dtype::Val::int64:    f(std::type_identity<int64_t>{}); break;
    case Dtype::Val::uint8:    f(std::type_identity<uint8_t>{}); break;
    case Dtype::Val::uint16:   f(std::type_identity<uint16_t>{}); break;
    case Dtype::Val::uint32:   f(std::type_identity<uint32_t>{}); break;
    case Dtype::Val::uint64:   f(std::type_identity<uint64_t>{}); break;
    case Dtype::Val::float16:  f(std::type_identity<float16_t>{}); break;
    case Dtype::Val::bfloat16: f(std::type_identity<bfloat16_t>{}); break;
    case Dtype::Val::float32:  f(std::type_identity<float>{}); break;
    default:
      throw std::runtime_error("[dispatch_real_type] Expected real numeric type");
  }
}

/**
 * Type traits for OpenCL kernel configuration.
 *
 * These provide compile-time information useful for kernel dispatch:
 * - work_per_thread: Number of elements each work-item should process
 * - needs_conversion: Whether type needs float conversion (bf16/f16)
 */
template <typename T>
struct opencl_type_traits {
  static constexpr int work_per_thread = std::max(1, 8 / static_cast<int>(sizeof(T)));
  static constexpr bool needs_conversion = false;
  static constexpr bool is_floating = false;
};

template <>
struct opencl_type_traits<float> {
  static constexpr int work_per_thread = 2;  // 8 / 4 bytes
  static constexpr bool needs_conversion = false;
  static constexpr bool is_floating = true;
};

template <>
struct opencl_type_traits<float16_t> {
  static constexpr int work_per_thread = 4;  // 8 / 2 bytes
  static constexpr bool needs_conversion = true;
  static constexpr bool is_floating = true;
};

template <>
struct opencl_type_traits<bfloat16_t> {
  static constexpr int work_per_thread = 4;  // 8 / 2 bytes
  static constexpr bool needs_conversion = true;
  static constexpr bool is_floating = true;
};

template <>
struct opencl_type_traits<int8_t> {
  static constexpr int work_per_thread = 8;  // 8 / 1 byte
  static constexpr bool needs_conversion = false;
  static constexpr bool is_floating = false;
};

template <>
struct opencl_type_traits<uint8_t> {
  static constexpr int work_per_thread = 8;  // 8 / 1 byte
  static constexpr bool needs_conversion = false;
  static constexpr bool is_floating = false;
};

template <>
struct opencl_type_traits<bool> {
  static constexpr int work_per_thread = 8;  // Treated as uint8 in OpenCL
  static constexpr bool needs_conversion = false;
  static constexpr bool is_floating = false;
};

} // namespace mlx::core::opencl
