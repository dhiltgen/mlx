// Copyright © 2025 MLX Contributors

#pragma once

#include <string>

namespace mlx::core::opencl {

/**
 * Get OpenCL source code for unary operation definitions.
 */
std::string get_unary_ops();

/**
 * Get OpenCL source code for binary operation definitions.
 */
std::string get_binary_ops();

} // namespace mlx::core::opencl
