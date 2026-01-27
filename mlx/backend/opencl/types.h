// Copyright © 2025 MLX Contributors

#pragma once

#include <string>
#include "mlx/dtype.h"

namespace mlx::core::opencl {

/**
 * Type conversion utilities for OpenCL kernel code generation.
 *
 * These utilities centralize type handling to avoid code duplication
 * across kernel generators. The pattern matches Metal/CUDA backends
 * where type dispatch is handled through shared infrastructure.
 */

/**
 * Holds OpenCL code snippets for reading/writing a specific type.
 *
 * For types that need conversion (bf16, f16), this provides the
 * appropriate conversion functions. For native types, these are identity.
 */
struct TypeConversion {
  std::string read_fn;   // Function to read and convert to compute type, e.g., "bfloat16_to_float"
  std::string write_fn;  // Function to convert and write from compute type, e.g., "float_to_bfloat16"
  std::string acc_type;  // Accumulator/compute type, e.g., "float" for bf16
  bool needs_read_convert;   // True if read needs conversion function
  bool needs_write_convert;  // True if write needs conversion function
};

/**
 * Get type conversion info for a given dtype.
 *
 * @param dtype The MLX data type
 * @return TypeConversion with read/write expressions
 *
 * Usage in kernel generation:
 * @code
 * auto conv = get_type_conversion(dtype);
 * if (conv.needs_read_convert) {
 *   kernel << conv.read_fn << "(x[i])";  // e.g., "bfloat16_to_float(x[i])"
 * } else {
 *   kernel << "x[i]";
 * }
 * @endcode
 */
TypeConversion get_type_conversion(Dtype dtype);

/**
 * Generate a read expression for the given type.
 *
 * @param dtype The MLX data type
 * @param var_expr The variable expression to read, e.g., "x[i]" or "input[idx]"
 * @return The full read expression, e.g., "bfloat16_to_float(x[i])" or "x[i]"
 */
std::string make_read_expr(Dtype dtype, const std::string& var_expr);

/**
 * Generate a write expression for the given type.
 *
 * @param dtype The MLX data type
 * @param val_expr The value expression to write, e.g., "result" or "val * scale"
 * @return The full write expression, e.g., "float_to_bfloat16(result)" or "result"
 */
std::string make_write_expr(Dtype dtype, const std::string& val_expr);

/**
 * Get the accumulator type name for a given dtype.
 * This is the type used for intermediate computations.
 *
 * - bf16/f16 -> "float" (compute in float32 for precision)
 * - float32 -> "float"
 * - integers -> same type or promoted (e.g., int8 -> int32 for sum)
 */
std::string get_accumulator_type(Dtype dtype);

/**
 * Check if a type needs float conversion for computation.
 * True for bfloat16 and float16.
 */
bool needs_float_conversion(Dtype dtype);

/**
 * Check if a type is a floating-point type (including bf16/f16).
 */
bool is_floating_type(Dtype dtype);

/**
 * Check if a type is an integer type (including bool).
 */
bool is_integer_type(Dtype dtype);

/**
 * Get OpenCL type name from MLX dtype.
 *
 * This is the primary type name used in kernel declarations:
 * - float32 -> "float"
 * - float16 -> "half"
 * - bfloat16 -> "bfloat16_t" (emulated via uint16 + conversion helpers)
 * - int32 -> "int32_t"
 * etc.
 */
std::string type_to_name(const Dtype& dtype);

/**
 * Get a short suffix for kernel naming.
 *
 * - float32 -> "f32"
 * - float16 -> "f16"
 * - bfloat16 -> "bf16"
 * - int32 -> "i32"
 * etc.
 */
std::string type_to_suffix(Dtype dtype);

} // namespace mlx::core::opencl
