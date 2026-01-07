// Copyright © 2025 Apple Inc.

#include "mlx/backend/cuda/cuda.h"

#include <cuda_runtime.h>

namespace mlx::core::cu {

bool is_available() {
  static bool initialized = []() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      return false;
    }

    // Force context creation on device 0
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
      return false;
    }

    // Force lazy context initialization
    err = cudaFree(nullptr);
    return err == cudaSuccess;
  }();
  return initialized;
}

} // namespace mlx::core::cu
