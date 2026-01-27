// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/primitives.h"

#include <sstream>

namespace mlx::core {

void LogSumExp::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const array& in = inputs[0];

  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error("[LogSumExp::eval_gpu] Does not support non-floating point types.");
  }

  // Check if input is truly contiguous (row-major with last dim stride == 1)
  bool is_contiguous = in.flags().contiguous && (in.ndim() == 0 || in.strides()[in.ndim() - 1] == 1);

  // Get dimensions - logsumexp operates on the last dimension
  int axis_size = in.ndim() > 0 ? in.shape().back() : 1;
  int n_rows = in.size() / axis_size;

  // Output has shape with last dimension reduced to 1
  // Compute proper row-major strides (out.strides()/data_size() may be uninitialized)
  Strides strides(out.ndim());
  if (!strides.empty()) {
    int64_t stride = 1;
    for (int i = out.ndim() - 1; i >= 0; i--) {
      strides[i] = stride;
      stride *= out.shape(i);
    }
  }
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  flags.col_contiguous = out.size() <= 1 || (out.ndim() > 0 && out.size() == *std::max_element(out.shape().begin(), out.shape().end()));
  out.set_data(
      opencl::allocator().malloc(out.nbytes()),
      out.size(),
      strides,
      flags);

  auto& dev = opencl::device(Device::gpu);
  auto& s = stream();
  auto& encoder = dev.get_command_encoder(s.index);

  // Get type information using centralized utilities
  std::string type_name = opencl::type_to_name(in.dtype());
  std::string type_suffix = opencl::type_to_suffix(in.dtype());
  bool needs_conv = opencl::needs_float_conversion(in.dtype());

  // Use float for accumulation with half types
  std::string acc_type = needs_conv ? "float" : type_name;

  if (is_contiguous) {
    // Contiguous path: simple linear access
    // Kernel naming convention: {op}_{variant}_{type}
    std::ostringstream kernel_source;
    kernel_source << opencl::get_kernel_preamble();
    kernel_source << "\n__kernel void logsumexp_" << type_suffix << "(\n";
    kernel_source << "    __global const " << type_name << "* input,\n";
    kernel_source << "    __global " << type_name << "* output,\n";
    kernel_source << "    int axis_size,\n";
    kernel_source << "    int n_rows) {\n";
    kernel_source << "  int row = get_global_id(0);\n";
    kernel_source << "  if (row >= n_rows) return;\n\n";
    kernel_source << "  __global const " << type_name << "* in_row = input + row * axis_size;\n\n";

    // Step 1: Find max for numerical stability
    kernel_source << "  // Find max for numerical stability\n";
    kernel_source << "  " << acc_type << " maxval = -INFINITY;\n";
    kernel_source << "  for (int i = 0; i < axis_size; i++) {\n";
    kernel_source << "    maxval = max(maxval, " << opencl::make_read_expr(in.dtype(), "in_row[i]") << ");\n";
    kernel_source << "  }\n\n";

    // Step 2: Compute sum of exp(x - max)
    kernel_source << "  // Compute sum(exp(x - max))\n";
    kernel_source << "  " << acc_type << " sum = 0.0f;\n";
    kernel_source << "  for (int i = 0; i < axis_size; i++) {\n";
    kernel_source << "    sum += exp(" << opencl::make_read_expr(in.dtype(), "in_row[i]") << " - maxval);\n";
    kernel_source << "  }\n\n";

    // Step 3: Output log(sum) + max
    kernel_source << "  // Output log(sum) + max\n";
    kernel_source << "  " << acc_type << " result = isinf(maxval) ? maxval : (log(sum) + maxval);\n";
    kernel_source << "  output[row] = " << opencl::make_write_expr(in.dtype(), "result") << ";\n";
    kernel_source << "}\n";

    std::string source_str = kernel_source.str();
    std::string kernel_name = "logsumexp_" + type_suffix;

    // Get or compile kernel
    cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
    encoder.set_kernel(kernel);

    // Set arguments
    encoder.set_input_array(in, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(axis_size, 2);
    encoder.set_bytes(n_rows, 3);

    // Dispatch one work-item per row
    size_t global_size[3] = {static_cast<size_t>(n_rows), 0, 0};
    encoder.dispatch_threads(global_size, nullptr, 1);
  } else {
    // Non-contiguous path: use strided access
    int ndim = in.ndim();
    std::vector<int> shape_vec(in.shape().begin(), in.shape().end());
    std::vector<int64_t> strides_vec(in.strides().begin(), in.strides().end());

    // Handle scalar case
    if (shape_vec.empty()) {
      shape_vec.push_back(1);
      strides_vec.push_back(0);
      ndim = 1;
    }

    // Kernel naming convention: {op}_{variant}_{type}
    std::ostringstream kernel_source;
    kernel_source << opencl::get_kernel_preamble();
    kernel_source << "\n__kernel void logsumexp_strided_" << type_suffix << "(\n";
    kernel_source << "    __global const " << type_name << "* input,\n";
    kernel_source << "    __global " << type_name << "* output,\n";
    kernel_source << "    __global const int* shape,\n";
    kernel_source << "    __global const long* strides,\n";
    kernel_source << "    int ndim,\n";
    kernel_source << "    int axis_size,\n";
    kernel_source << "    int n_rows) {\n";
    kernel_source << "  int row = get_global_id(0);\n";
    kernel_source << "  if (row >= n_rows) return;\n\n";

    // Compute the base offset for this row (all dimensions except last)
    kernel_source << "  // Compute base offset for this row\n";
    kernel_source << "  long base_offset = 0;\n";
    kernel_source << "  int remaining = row;\n";
    kernel_source << "  for (int d = ndim - 2; d >= 0; d--) {\n";
    kernel_source << "    int idx = remaining % shape[d];\n";
    kernel_source << "    remaining /= shape[d];\n";
    kernel_source << "    base_offset += idx * strides[d];\n";
    kernel_source << "  }\n\n";

    // Get stride for last dimension
    kernel_source << "  long last_stride = strides[ndim - 1];\n\n";

    // Step 1: Find max for numerical stability
    kernel_source << "  // Find max for numerical stability\n";
    kernel_source << "  " << acc_type << " maxval = -INFINITY;\n";
    kernel_source << "  for (int i = 0; i < axis_size; i++) {\n";
    kernel_source << "    long offset = base_offset + i * last_stride;\n";
    kernel_source << "    maxval = max(maxval, " << opencl::make_read_expr(in.dtype(), "input[offset]") << ");\n";
    kernel_source << "  }\n\n";

    // Step 2: Compute sum of exp(x - max)
    kernel_source << "  // Compute sum(exp(x - max))\n";
    kernel_source << "  " << acc_type << " sum = 0.0f;\n";
    kernel_source << "  for (int i = 0; i < axis_size; i++) {\n";
    kernel_source << "    long offset = base_offset + i * last_stride;\n";
    kernel_source << "    sum += exp(" << opencl::make_read_expr(in.dtype(), "input[offset]") << " - maxval);\n";
    kernel_source << "  }\n\n";

    // Step 3: Output log(sum) + max
    kernel_source << "  // Output log(sum) + max\n";
    kernel_source << "  " << acc_type << " result = isinf(maxval) ? maxval : (log(sum) + maxval);\n";
    kernel_source << "  output[row] = " << opencl::make_write_expr(in.dtype(), "result") << ";\n";
    kernel_source << "}\n";

    std::string source_str = kernel_source.str();
    std::string kernel_name = "logsumexp_strided_" + type_suffix;

    // Get or compile kernel
    cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
    encoder.set_kernel(kernel);

    // Create buffers for shape and strides
    cl_int err;
    cl_mem shape_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       shape_vec.size() * sizeof(int), shape_vec.data(), &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("[LogSumExp::eval_gpu] Failed to create shape buffer: " + std::to_string(err));
    }

    cl_mem strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         strides_vec.size() * sizeof(int64_t), strides_vec.data(), &err);
    if (err != CL_SUCCESS) {
      clReleaseMemObject(shape_buf);
      throw std::runtime_error("[LogSumExp::eval_gpu] Failed to create strides buffer: " + std::to_string(err));
    }

    // Set arguments
    encoder.set_input_array(in, 0);
    encoder.set_output_array(out, 1);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &shape_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &strides_buf);
    encoder.set_bytes(ndim, 4);
    encoder.set_bytes(axis_size, 5);
    encoder.set_bytes(n_rows, 6);

    // Dispatch one work-item per row
    size_t global_size[3] = {static_cast<size_t>(n_rows), 0, 0};
    encoder.dispatch_threads(global_size, nullptr, 1);

    // Track temp buffers for cleanup
    dev.add_temp_buffer(shape_buf, s.index);
    dev.add_temp_buffer(strides_buf, s.index);
    dev.end_encoding(s.index);
  }

  dev.add_temporary(in, s.index);
}

} // namespace mlx::core
