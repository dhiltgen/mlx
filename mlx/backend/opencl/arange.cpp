// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/primitives.h"

#include <iostream>
#include <sstream>

namespace mlx::core {

void Arange::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 0);

  OPENCL_DEBUG_LOG("[Arange::eval_gpu] out.shape()=" << out.size()
            << " out.strides().size()=" << out.strides().size());
  if (out.strides().size() > 0) {
    OPENCL_DEBUG_LOG("[Arange::eval_gpu] out.strides()[0]=" << out.strides()[0]);
  }

  // Allocate output
  out.set_data(opencl::allocator().malloc(out.nbytes()));

  OPENCL_DEBUG_LOG("[Arange::eval_gpu] After set_data: out.strides().size()=" << out.strides().size());

  auto& dev = opencl::device(Device::gpu);
  auto& s = stream();
  auto& encoder = dev.get_command_encoder(s.index);

  // Get type information
  std::string type_name = opencl::type_to_name(out.dtype());
  size_t n = out.size();

  // Build arange kernel
  // Kernel: out[index] = start + index * step
  // For float16/bfloat16, we pass float args and convert in kernel
  bool use_float_args = (out.dtype() == float16 || out.dtype() == bfloat16);
  bool is_bfloat16 = (out.dtype() == bfloat16);
  std::string arg_type = use_float_args ? "float" : type_name;

  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();
  // bfloat16 conversion helpers are now in the preamble

  kernel_source << "\n__kernel void arange_" << type_name << "(\n";
  kernel_source << "    __global " << type_name << "* out,\n";
  kernel_source << "    " << arg_type << " start,\n";
  kernel_source << "    " << arg_type << " step,\n";
  kernel_source << "    int n) {\n";
  kernel_source << "  int idx = get_global_id(0);\n";
  kernel_source << "  if (idx < n) {\n";
  if (is_bfloat16) {
    kernel_source << "    out[idx] = float_to_bfloat16(start + idx * step);\n";
  } else if (use_float_args) {
    kernel_source << "    out[idx] = (" << type_name << ")(start + idx * step);\n";
  } else {
    kernel_source << "    out[idx] = start + idx * step;\n";
  }
  kernel_source << "  }\n";
  kernel_source << "}\n";

  std::string source_str = kernel_source.str();
  std::string kernel_name = "arange_" + type_name;

  OPENCL_DEBUG_LOG("[Arange::eval_gpu] Compiling kernel: " << kernel_name
            << " for n=" << n << " start=" << start_ << " step=" << step_);

  // Get or compile kernel
  cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
  encoder.set_kernel(kernel);

  // Set arguments
  encoder.set_output_array(out, 0);

  // Set scalar arguments based on type
  switch (out.dtype()) {
    case uint8:
      encoder.set_bytes(static_cast<uint8_t>(start_), 1);
      encoder.set_bytes(static_cast<uint8_t>(step_), 2);
      break;
    case uint16:
      encoder.set_bytes(static_cast<uint16_t>(start_), 1);
      encoder.set_bytes(static_cast<uint16_t>(step_), 2);
      break;
    case uint32:
      encoder.set_bytes(static_cast<uint32_t>(start_), 1);
      encoder.set_bytes(static_cast<uint32_t>(step_), 2);
      break;
    case uint64:
      encoder.set_bytes(static_cast<uint64_t>(start_), 1);
      encoder.set_bytes(static_cast<uint64_t>(step_), 2);
      break;
    case int8:
      encoder.set_bytes(static_cast<int8_t>(start_), 1);
      encoder.set_bytes(static_cast<int8_t>(step_), 2);
      break;
    case int16:
      encoder.set_bytes(static_cast<int16_t>(start_), 1);
      encoder.set_bytes(static_cast<int16_t>(step_), 2);
      break;
    case int32:
      encoder.set_bytes(static_cast<int32_t>(start_), 1);
      encoder.set_bytes(static_cast<int32_t>(step_), 2);
      break;
    case int64:
      encoder.set_bytes(static_cast<int64_t>(start_), 1);
      encoder.set_bytes(static_cast<int64_t>(step_), 2);
      break;
    case float16:
      // OpenCL doesn't have native half type, use float
      encoder.set_bytes(static_cast<float>(start_), 1);
      encoder.set_bytes(static_cast<float>(step_), 2);
      break;
    case float32:
      encoder.set_bytes(static_cast<float>(start_), 1);
      encoder.set_bytes(static_cast<float>(step_), 2);
      break;
    case bfloat16:
      // bfloat16 not natively supported in OpenCL, use float
      encoder.set_bytes(static_cast<float>(start_), 1);
      encoder.set_bytes(static_cast<float>(step_), 2);
      break;
    case complex64:
      throw std::runtime_error("[Arange::eval_gpu] Complex types not yet supported");
    default:
      throw std::runtime_error("[Arange::eval_gpu] Unsupported dtype");
  }

  encoder.set_bytes(static_cast<int>(n), 3);

  // Dispatch
  size_t global_size[3] = {n, 0, 0};
  encoder.dispatch_threads(global_size, nullptr, 1);

  OPENCL_DEBUG_LOG("[Arange::eval_gpu] Dispatched kernel for " << n << " elements");
}

} // namespace mlx::core
