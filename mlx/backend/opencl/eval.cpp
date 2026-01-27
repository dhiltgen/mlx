// Copyright © 2025 MLX Contributors
// OpenCL backend evaluation stubs

#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/primitives.h"
#include "mlx/scheduler.h"

namespace mlx::core::gpu {

bool is_available() {
  // TODO: Actually check if OpenCL devices are available
  // For now, just try to get the device and assume it exists
  try {
    opencl::device(mlx::core::Device::gpu);
    return true;
  } catch (...) {
    return false;
  }
}

void new_stream(Stream stream) {
  // Create OpenCL command queue for this stream
  if (stream.device == mlx::core::Device::gpu) {
    auto& dev = opencl::device(stream.device);
    dev.new_queue(stream.index);
    OPENCL_DEBUG_LOG("[NEW_STREAM_DEBUG] Created stream " << stream.index << " for GPU");
  }
}

void eval(array& arr) {
  // Execute the primitive's GPU evaluation method
  auto s = arr.primitive().stream();
  auto outputs = arr.outputs();

  // Call the OpenCL GPU kernel execution
  arr.primitive().eval_gpu(arr.inputs(), outputs);

  // TODO: Implement proper stream synchronization
  // For now, synchronization happens in buffer mapping via clFinish()
}

void finalize(Stream s) {
  // Flush command queue - ends current batch of work
  if (s.device == mlx::core::Device::gpu) {
    auto& dev = opencl::device(s.device);
    dev.end_encoding(s.index);
  }
}

void synchronize(Stream s) {
  // Wait for all operations on this stream to complete
  // Use synchronize() which just does clFinish without resetting encoder/buffers
  if (s.device == mlx::core::Device::gpu) {
    auto& dev = opencl::device(s.device);
    dev.synchronize(s.index);
  }
}

} // namespace mlx::core::gpu
