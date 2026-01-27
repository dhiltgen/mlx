// Copyright © 2025 MLX Contributors

#pragma once

#include "mlx/api.h"

namespace mlx::core::opencl {

/**
 * Check if OpenCL is available on this system.
 * Returns true if at least one OpenCL platform and device can be found.
 */
MLX_API bool is_available();

} // namespace mlx::core::opencl
