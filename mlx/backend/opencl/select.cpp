// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/primitives.h"

#include <iostream>
#include <sstream>

namespace mlx::core {

void Select::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);
  const auto& condition = inputs[0];
  const auto& a = inputs[1];
  const auto& b = inputs[2];

  if (::mlx::core::opencl::is_debug_enabled()) {
    std::cerr << "[Select::eval_gpu] condition shape: ";
    for (auto s : condition.shape()) std::cerr << s << " ";
    std::cerr << ", a shape: ";
    for (auto s : a.shape()) std::cerr << s << " ";
    std::cerr << ", b shape: ";
    for (auto s : b.shape()) std::cerr << s << " ";
    std::cerr << std::endl;
  }

  // Check if simple case (all contiguous, same size)
  bool simple_case = condition.flags().row_contiguous &&
                     a.flags().row_contiguous &&
                     b.flags().row_contiguous &&
                     condition.data_size() == out.size() &&
                     a.data_size() == out.size() &&
                     b.data_size() == out.size();

  // Allocate output - use out.size() as data_size since the output is contiguous
  // Also compute proper row-major strides
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

  // Get type information
  std::string cond_type = opencl::type_to_name(condition.dtype());
  std::string data_type = opencl::type_to_name(a.dtype());

  size_t n = out.size();

  if (simple_case) {
    // Simple case: all arrays contiguous and same size
    std::ostringstream kernel_source;
    kernel_source << opencl::get_kernel_preamble();
    kernel_source << "\n__kernel void select_" << cond_type << "_" << data_type << "(\n";
    kernel_source << "    __global const " << cond_type << "* condition,\n";
    kernel_source << "    __global const " << data_type << "* a,\n";
    kernel_source << "    __global const " << data_type << "* b,\n";
    kernel_source << "    __global " << data_type << "* out,\n";
    kernel_source << "    long cond_offset,\n";
    kernel_source << "    long a_offset,\n";
    kernel_source << "    long b_offset,\n";
    kernel_source << "    int n) {\n";
    kernel_source << "  int idx = get_global_id(0);\n";
    kernel_source << "  if (idx < n) {\n";
    kernel_source << "    out[idx] = condition[cond_offset + idx] ? a[a_offset + idx] : b[b_offset + idx];\n";
    kernel_source << "  }\n";
    kernel_source << "}\n";

    std::string source_str = kernel_source.str();
    std::string kernel_name = "select_" + cond_type + "_" + data_type;

    OPENCL_DEBUG_LOG("[Select::eval_gpu] Compiling simple kernel: " << kernel_name);

    cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
    encoder.set_kernel(kernel);

    encoder.set_input_array(condition, 0);
    encoder.set_input_array(a, 1);
    encoder.set_input_array(b, 2);
    encoder.set_output_array(out, 3);
    int64_t cond_offset = static_cast<int64_t>(condition.offset()) / condition.itemsize();
    int64_t a_offset = static_cast<int64_t>(a.offset()) / a.itemsize();
    int64_t b_offset = static_cast<int64_t>(b.offset()) / b.itemsize();
    encoder.set_bytes(cond_offset, 4);
    encoder.set_bytes(a_offset, 5);
    encoder.set_bytes(b_offset, 6);
    encoder.set_bytes(static_cast<int>(n), 7);

    size_t global_size[3] = {n, 0, 0};
    encoder.dispatch_threads(global_size, nullptr, 1);
  } else {
    // General case: broadcasting/strided arrays
    // Use output shape and compute indices for each input using their strides
    int ndim = out.ndim();
    std::vector<int> out_shape(out.shape().begin(), out.shape().end());
    std::vector<int64_t> out_strides(out.strides().begin(), out.strides().end());

    // For broadcasting: if input dim is 1, stride is effectively 0
    auto get_broadcast_strides = [ndim](const array& arr) {
      std::vector<int64_t> strides(ndim, 0);
      int arr_ndim = arr.ndim();
      int offset = ndim - arr_ndim;
      for (int i = 0; i < arr_ndim; i++) {
        // If dimension size is 1, broadcast (stride = 0)
        strides[offset + i] = (arr.shape(i) == 1) ? 0 : arr.strides(i);
      }
      return strides;
    };

    std::vector<int64_t> cond_strides = get_broadcast_strides(condition);
    std::vector<int64_t> a_strides = get_broadcast_strides(a);
    std::vector<int64_t> b_strides = get_broadcast_strides(b);

    // Handle scalar case
    if (out_shape.empty()) {
      out_shape.push_back(1);
      out_strides.push_back(0);
      cond_strides.push_back(0);
      a_strides.push_back(0);
      b_strides.push_back(0);
      ndim = 1;
    }

    std::ostringstream kernel_source;
    kernel_source << opencl::get_kernel_preamble();
    kernel_source << "\n__kernel void select_general_" << cond_type << "_" << data_type << "(\n";
    kernel_source << "    __global const " << cond_type << "* condition,\n";
    kernel_source << "    __global const " << data_type << "* a,\n";
    kernel_source << "    __global const " << data_type << "* b,\n";
    kernel_source << "    __global " << data_type << "* out,\n";
    kernel_source << "    __global const int* out_shape,\n";
    kernel_source << "    __global const long* cond_strides,\n";
    kernel_source << "    __global const long* a_strides,\n";
    kernel_source << "    __global const long* b_strides,\n";
    kernel_source << "    long cond_base_offset,\n";
    kernel_source << "    long a_base_offset,\n";
    kernel_source << "    long b_base_offset,\n";
    kernel_source << "    int ndim,\n";
    kernel_source << "    int n) {\n";
    kernel_source << "  int idx = get_global_id(0);\n";
    kernel_source << "  if (idx >= n) return;\n";
    kernel_source << "\n";
    kernel_source << "  // Compute multi-dimensional indices from linear output index\n";
    kernel_source << "  long cond_offset = cond_base_offset, a_offset = a_base_offset, b_offset = b_base_offset;\n";
    kernel_source << "  int remaining = idx;\n";
    kernel_source << "  for (int d = ndim - 1; d >= 0; d--) {\n";
    kernel_source << "    int dim_idx = remaining % out_shape[d];\n";
    kernel_source << "    remaining /= out_shape[d];\n";
    kernel_source << "    cond_offset += dim_idx * cond_strides[d];\n";
    kernel_source << "    a_offset += dim_idx * a_strides[d];\n";
    kernel_source << "    b_offset += dim_idx * b_strides[d];\n";
    kernel_source << "  }\n";
    kernel_source << "\n";
    kernel_source << "  out[idx] = condition[cond_offset] ? a[a_offset] : b[b_offset];\n";
    kernel_source << "}\n";

    std::string source_str = kernel_source.str();
    std::string kernel_name = "select_general_" + cond_type + "_" + data_type;

    OPENCL_DEBUG_LOG("[Select::eval_gpu] Compiling general kernel: " << kernel_name);

    cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
    encoder.set_kernel(kernel);

    encoder.set_input_array(condition, 0);
    encoder.set_input_array(a, 1);
    encoder.set_input_array(b, 2);
    encoder.set_output_array(out, 3);

    // Create buffers for shape and strides
    cl_int err;
    cl_mem shape_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       out_shape.size() * sizeof(int), out_shape.data(), &err);
    cl_mem cond_strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              cond_strides.size() * sizeof(int64_t), cond_strides.data(), &err);
    cl_mem a_strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           a_strides.size() * sizeof(int64_t), a_strides.data(), &err);
    cl_mem b_strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           b_strides.size() * sizeof(int64_t), b_strides.data(), &err);

    clSetKernelArg(kernel, 4, sizeof(cl_mem), &shape_buf);
    clSetKernelArg(kernel, 5, sizeof(cl_mem), &cond_strides_buf);
    clSetKernelArg(kernel, 6, sizeof(cl_mem), &a_strides_buf);
    clSetKernelArg(kernel, 7, sizeof(cl_mem), &b_strides_buf);
    int64_t cond_base_offset = static_cast<int64_t>(condition.offset()) / condition.itemsize();
    int64_t a_base_offset = static_cast<int64_t>(a.offset()) / a.itemsize();
    int64_t b_base_offset = static_cast<int64_t>(b.offset()) / b.itemsize();
    encoder.set_bytes(cond_base_offset, 8);
    encoder.set_bytes(a_base_offset, 9);
    encoder.set_bytes(b_base_offset, 10);
    encoder.set_bytes(ndim, 11);
    encoder.set_bytes(static_cast<int>(n), 12);

    size_t global_size[3] = {n, 0, 0};
    encoder.dispatch_threads(global_size, nullptr, 1);

    // Track buffers for cleanup after sync
    dev.add_temp_buffer(shape_buf, s.index);
    dev.add_temp_buffer(cond_strides_buf, s.index);
    dev.add_temp_buffer(a_strides_buf, s.index);
    dev.add_temp_buffer(b_strides_buf, s.index);
  }

  dev.add_temporary(condition, s.index);
  dev.add_temporary(a, s.index);
  dev.add_temporary(b, s.index);

  // Release temp buffers after kernel execution completes
  dev.end_encoding(s.index);

  OPENCL_DEBUG_LOG("[Select::eval_gpu] Dispatched kernel for " << n << " elements");
}

} // namespace mlx::core
