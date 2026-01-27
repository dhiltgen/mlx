// Copyright © 2025 MLX Contributors

#pragma once

#include <cstdlib>
#include <iostream>

namespace mlx::core::opencl {

// Check if debug logging is enabled via MLX_OPENCL_DEBUG environment variable
inline bool is_debug_enabled() {
  static bool checked = false;
  static bool enabled = false;

  if (!checked) {
    const char* env = std::getenv("MLX_OPENCL_DEBUG");
    enabled = (env != nullptr && env[0] == '1');
    checked = true;
  }

  return enabled;
}

// Debug logging macro - only logs if MLX_OPENCL_DEBUG=1
#define OPENCL_DEBUG_LOG(msg) \
  do { \
    if (::mlx::core::opencl::is_debug_enabled()) { \
      std::cerr << msg << std::endl; \
    } \
  } while (0)

} // namespace mlx::core::opencl
