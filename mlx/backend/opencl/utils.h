// Copyright © 2025 MLX Contributors

#pragma once

#include <sstream>
#include <string>

#include "mlx/dtype.h"
#include "mlx/backend/opencl/types.h"  // Centralized type utilities

namespace mlx::core::opencl {

// Note: type_to_name() and other type utilities are now in types.h
// included above for backwards compatibility

/**
 * Get the OpenCL kernel preamble with helper functions.
 */
std::string get_kernel_preamble();

/**
 * Generate a specialized kernel definition from a template.
 *
 * @param kernel_name The name for the specialized kernel function
 * @param template_name The template type (e.g., "unary_v", "binary_vv")
 * @param input_type The input data type (e.g., "float", "int")
 * @param output_type The output data type
 * @param operation The operation name (e.g., "abs", "add")
 * @param work_per_thread Number of elements each thread processes (default 1)
 */
std::string get_template_definition(
    const std::string& kernel_name,
    const std::string& template_name,
    const std::string& input_type,
    const std::string& output_type,
    const std::string& operation,
    int work_per_thread = 1);

/**
 * Get the number of elements each thread should process.
 * Matches Metal's WorkPerThread pattern: 8 / sizeof(type)
 * - float (4 bytes): 2 elements per thread
 * - half (2 bytes): 4 elements per thread
 * - int8 (1 byte): 8 elements per thread
 */
inline int get_work_per_thread(const Dtype& dtype) {
  return std::max(1, 8 / static_cast<int>(dtype.size()));
}

/**
 * Get work per thread, but only for large arrays.
 * For small arrays (< 64K elements), returns 1 to avoid overhead.
 */
inline int get_work_per_thread(const Dtype& dtype, size_t size) {
  constexpr size_t wpt_threshold = 1 << 16;  // 65536 elements
  return size < wpt_threshold ? 1 : get_work_per_thread(dtype);
}

/**
 * Ceiling division helper.
 */
inline size_t ceildiv(size_t n, size_t m) {
  return (n + m - 1) / m;
}

} // namespace mlx::core::opencl
