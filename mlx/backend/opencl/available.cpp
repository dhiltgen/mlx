// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/available.h"

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

namespace mlx::core::opencl {

bool is_available() {
  cl_uint num_platforms = 0;
  cl_int status = clGetPlatformIDs(0, nullptr, &num_platforms);

  if (status != CL_SUCCESS || num_platforms == 0) {
    return false;
  }

  // Check if we have at least one device
  cl_platform_id* platforms = new cl_platform_id[num_platforms];
  status = clGetPlatformIDs(num_platforms, platforms, nullptr);

  if (status != CL_SUCCESS) {
    delete[] platforms;
    return false;
  }

  bool has_device = false;
  for (cl_uint i = 0; i < num_platforms; ++i) {
    cl_uint num_devices = 0;
    status = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (status == CL_SUCCESS && num_devices > 0) {
      has_device = true;
      break;
    }
  }

  delete[] platforms;
  return has_device;
}

} // namespace mlx::core::opencl
