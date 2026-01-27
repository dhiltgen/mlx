// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/primitives.h"

#include <sstream>

namespace mlx::core {

void Softmax::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const array& in = inputs[0];

  if (!issubdtype(out.dtype(), floating)) {
    throw std::runtime_error("[Softmax::eval_gpu] Does not support non-floating point types.");
  }

  // Handle empty/scalar arrays - softmax of empty/scalar is itself
  if (in.size() == 0 || in.ndim() == 0) {
    // Compute proper row-major strides (out.strides()/data_size() may be uninitialized)
    Strides strides0(out.ndim());
    if (!strides0.empty()) {
      int64_t stride = 1;
      for (int i = out.ndim() - 1; i >= 0; i--) {
        strides0[i] = stride;
        stride *= out.shape(i);
      }
    }
    array::Flags flags0;
    flags0.contiguous = true;
    flags0.row_contiguous = true;
    flags0.col_contiguous = out.size() <= 1 || (out.ndim() > 0 && out.size() == *std::max_element(out.shape().begin(), out.shape().end()));
    out.set_data(
        opencl::allocator().malloc(out.nbytes() > 0 ? out.nbytes() : 1),
        out.size(),
        strides0,
        flags0);
    // Copy input to output (for scalar case, softmax is just the input)
    if (in.size() > 0) {
      auto& dev = opencl::device(Device::gpu);
      auto& s = stream();
      cl_mem in_buf = static_cast<cl_mem>(const_cast<void*>(in.buffer().ptr()));
      cl_mem out_buf = static_cast<cl_mem>(out.buffer().ptr());
      clEnqueueCopyBuffer(dev.get_queue(s), in_buf, out_buf, 0, 0, in.nbytes(), 0, nullptr, nullptr);
    }
    return;
  }

  // No contiguous copy needed - we handle strides directly in the kernel

  // Allocate output - compute proper row-major strides (out.strides()/data_size() may be uninitialized)
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

  // Get dimensions - softmax is always along the last axis
  int axis_size = in.shape().back();

  // Compute n_rows as product of all dimensions except the last
  int n_rows = 1;
  for (size_t i = 0; i < in.ndim() - 1; i++) {
    n_rows *= in.shape(i);
  }

  // Get type information
  std::string type_name = opencl::type_to_name(in.dtype());
  std::string type_suffix = opencl::type_to_suffix(in.dtype());
  bool needs_conv = opencl::needs_float_conversion(in.dtype());

  // Prepare shapes and strides for stride-aware kernel
  auto in_strides = in.strides();
  auto shape = in.shape();
  int64_t axis_stride = in_strides.back();

  // Remove the axis dimension from shape and strides
  Shape row_shape(shape.begin(), shape.end() - 1);
  Strides row_strides(in_strides.begin(), in_strides.end() - 1);
  size_t ndim = row_shape.size();

  // Check if input is row-contiguous for vectorized kernel
  bool is_row_contiguous = in.flags().row_contiguous;

  // For very large axis sizes (like 128K vocab), use parallel workgroup reduction
  // For medium sizes (1K-8K), use vectorized single-thread kernel
  // Only when contiguous and float32 (bf16/f16 use strided kernel for conversion)
  bool use_parallel = (axis_size >= 8192 && !needs_conv &&
                       is_row_contiguous && axis_stride == 1);
  bool use_vectorized = (!use_parallel && axis_size >= 1024 && !needs_conv &&
                         is_row_contiguous && axis_stride == 1);

  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();

  std::string kernel_name;

  if (use_parallel) {
    // Parallel workgroup reduction kernel for very large axis sizes
    // Each workgroup handles one row with 256 threads doing parallel reduction
    // Uses subgroup (wave) reduction for fast reductions (wave size = 64 on Adreno)
    // Kernel naming convention: {op}_{variant}_{type}
    kernel_name = "softmax_parallel_" + type_suffix;

    kernel_source << R"(
// Enable subgroup extension for fast wave-level reductions
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define WG_SIZE 256

__kernel void )" << kernel_name << R"((
    __global const float* input,
    __global float* output,
    int axis_size,
    int n_rows,
    long in_data_offset) {
  int row = get_group_id(0);
  int lid = get_local_id(0);

  // Local memory for wave-level results (only need space for wave sums)
  __local float wave_max[8];
  __local float wave_sum[8];

  // Initialize wave results to avoid uninitialized memory
  if (lid < 8) {
    wave_max[lid] = -INFINITY;
    wave_sum[lid] = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Early exit check - but all threads must participate in barriers
  bool valid = (row < n_rows);

  // Apply data offset for sliced arrays
  input = input + in_data_offset;

  __global const float* in_row = input + row * axis_size;
  __global float* out_row = output + row * axis_size;

  // Phase 1: Each thread finds local max for its portion
  float thread_max = -INFINITY;
  if (valid) {
    for (int i = lid; i < axis_size; i += WG_SIZE) {
      thread_max = fmax(thread_max, in_row[i]);
    }
  }

  // Subgroup reduction for max - instant within each wave
  float sg_max = sub_group_reduce_max(thread_max);

  // First thread of each wave stores to local memory
  uint wave_id = get_sub_group_id();
  uint num_waves = get_num_sub_groups();
  if (get_sub_group_local_id() == 0 && wave_id < 8) {
    wave_max[wave_id] = sg_max;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Thread 0 computes final max from all waves
  float global_max;
  if (lid == 0) {
    global_max = wave_max[0];
    for (uint i = 1; i < num_waves && i < 8; i++) {
      global_max = fmax(global_max, wave_max[i]);
    }
    wave_max[0] = global_max;  // Store result for all threads to read
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  global_max = wave_max[0];

  // Phase 2: Each thread computes local sum of exp(x - max)
  float thread_sum = 0.0f;
  if (valid) {
    for (int i = lid; i < axis_size; i += WG_SIZE) {
      thread_sum += exp(in_row[i] - global_max);
    }
  }

  // Subgroup reduction for sum - instant within each wave
  float sg_sum = sub_group_reduce_add(thread_sum);

  // First thread of each wave stores to local memory
  if (get_sub_group_local_id() == 0 && wave_id < 8) {
    wave_sum[wave_id] = sg_sum;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  // Thread 0 computes final sum from all waves
  float global_sum;
  if (lid == 0) {
    global_sum = 0.0f;
    for (uint i = 0; i < num_waves && i < 8; i++) {
      global_sum += wave_sum[i];
    }
    wave_sum[0] = global_sum;  // Store result for all threads to read
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  global_sum = wave_sum[0];
  float normalizer = 1.0f / global_sum;

  // Phase 3: Each thread normalizes its portion
  if (valid) {
    for (int i = lid; i < axis_size; i += WG_SIZE) {
      out_row[i] = exp(in_row[i] - global_max) * normalizer;
    }
  }
}
)";
  } else if (use_vectorized) {
    // Optimized kernel using float4 vectorized loads for better memory bandwidth
    // Only works for row-contiguous inputs with axis_stride == 1
    kernel_name = "softmax_vec4_" + type_suffix;

    kernel_source << R"(
__kernel void )" << kernel_name << R"((
    __global const float* input,
    __global float* output,
    int axis_size,
    int n_rows,
    long in_data_offset) {
  int row = get_global_id(0);
  if (row >= n_rows) return;

  // Apply data offset for sliced arrays
  __global const float* in_row = input + in_data_offset + row * axis_size;
  __global float* out_row = output + row * axis_size;

  // Pass 1: Find max using float4 vectorized loads
  float maxval = -INFINITY;
  int i = 0;

  // Process 16 elements per iteration (4 x float4) for better ILP
  for (; i + 15 < axis_size; i += 16) {
    float4 v0 = vload4(0, in_row + i);
    float4 v1 = vload4(0, in_row + i + 4);
    float4 v2 = vload4(0, in_row + i + 8);
    float4 v3 = vload4(0, in_row + i + 12);

    float4 m0 = fmax(v0, v1);
    float4 m1 = fmax(v2, v3);
    float4 m2 = fmax(m0, m1);
    float rowmax = fmax(fmax(m2.s0, m2.s1), fmax(m2.s2, m2.s3));
    maxval = fmax(maxval, rowmax);
  }
  // Handle remainder with float4
  for (; i + 3 < axis_size; i += 4) {
    float4 v = vload4(0, in_row + i);
    maxval = fmax(maxval, fmax(fmax(v.s0, v.s1), fmax(v.s2, v.s3)));
  }
  // Scalar remainder
  for (; i < axis_size; i++) {
    maxval = fmax(maxval, in_row[i]);
  }

  // Pass 2: Compute sum of exp using vectorized loads
  float sum = 0.0f;
  i = 0;
  for (; i + 15 < axis_size; i += 16) {
    float4 v0 = vload4(0, in_row + i);
    float4 v1 = vload4(0, in_row + i + 4);
    float4 v2 = vload4(0, in_row + i + 8);
    float4 v3 = vload4(0, in_row + i + 12);

    sum += exp(v0.s0 - maxval) + exp(v0.s1 - maxval) + exp(v0.s2 - maxval) + exp(v0.s3 - maxval);
    sum += exp(v1.s0 - maxval) + exp(v1.s1 - maxval) + exp(v1.s2 - maxval) + exp(v1.s3 - maxval);
    sum += exp(v2.s0 - maxval) + exp(v2.s1 - maxval) + exp(v2.s2 - maxval) + exp(v2.s3 - maxval);
    sum += exp(v3.s0 - maxval) + exp(v3.s1 - maxval) + exp(v3.s2 - maxval) + exp(v3.s3 - maxval);
  }
  for (; i + 3 < axis_size; i += 4) {
    float4 v = vload4(0, in_row + i);
    sum += exp(v.s0 - maxval) + exp(v.s1 - maxval) + exp(v.s2 - maxval) + exp(v.s3 - maxval);
  }
  for (; i < axis_size; i++) {
    sum += exp(in_row[i] - maxval);
  }

  // Pass 3: Normalize and write using vectorized stores
  float normalizer = 1.0f / sum;
  i = 0;
  for (; i + 3 < axis_size; i += 4) {
    float4 v = vload4(0, in_row + i);
    float4 o = (float4)(
      exp(v.s0 - maxval) * normalizer,
      exp(v.s1 - maxval) * normalizer,
      exp(v.s2 - maxval) * normalizer,
      exp(v.s3 - maxval) * normalizer
    );
    vstore4(o, 0, out_row + i);
  }
  for (; i < axis_size; i++) {
    out_row[i] = exp(in_row[i] - maxval) * normalizer;
  }
}
)";
  } else {
    // Stride-aware kernel for small axis, non-float types, or non-contiguous inputs
    kernel_name = "softmax_strided_" + type_suffix;

    // Use centralized type utilities for read/write expressions
    std::string read_val = opencl::make_read_expr(in.dtype(), "input[in_idx + i * axis_stride]");
    std::string write_expr = opencl::make_write_expr(in.dtype(), "result");

    kernel_source << "\n__kernel void " << kernel_name << "(\n";
    kernel_source << "    __global const " << type_name << "* input,\n";
    kernel_source << "    __global " << type_name << "* output,\n";
    kernel_source << "    __global const int* shape,\n";
    kernel_source << "    __global const long* in_strides,\n";
    kernel_source << "    int ndim,\n";
    kernel_source << "    long axis_stride,\n";
    kernel_source << "    int axis_size,\n";
    kernel_source << "    int n_rows,\n";
    kernel_source << "    long in_data_offset) {\n";
    kernel_source << "  int row = get_global_id(0);\n";
    kernel_source << "  if (row >= n_rows) return;\n\n";

    // Compute input offset using strides (elem_to_loc style)
    kernel_source << "  // Compute input offset using strides, starting from data offset for sliced arrays\n";
    kernel_source << "  long in_idx = in_data_offset;\n";
    kernel_source << "  int remaining = row;\n";
    kernel_source << "  for (int d = ndim - 1; d >= 0; d--) {\n";
    kernel_source << "    int coord = remaining % shape[d];\n";
    kernel_source << "    remaining /= shape[d];\n";
    kernel_source << "    in_idx += coord * in_strides[d];\n";
    kernel_source << "  }\n\n";

    // Output is always contiguous (row-major)
    kernel_source << "  // Output is contiguous\n";
    kernel_source << "  __global " << type_name << "* out_row = output + row * axis_size;\n\n";

    kernel_source << "  // Find max for numerical stability\n";
    kernel_source << "  float maxval = -INFINITY;\n";
    kernel_source << "  for (int i = 0; i < axis_size; i++) {\n";
    kernel_source << "    maxval = max(maxval, (float)" << read_val << ");\n";
    kernel_source << "  }\n\n";

    kernel_source << "  // Compute sum(exp(x - max))\n";
    kernel_source << "  float sum = 0.0f;\n";
    kernel_source << "  for (int i = 0; i < axis_size; i++) {\n";
    kernel_source << "    sum += exp((float)" << read_val << " - maxval);\n";
    kernel_source << "  }\n\n";

    kernel_source << "  // Compute exp(x - max) / sum and write output\n";
    kernel_source << "  float normalizer = 1.0f / sum;\n";
    kernel_source << "  for (int i = 0; i < axis_size; i++) {\n";
    kernel_source << "    float result = exp((float)" << read_val << " - maxval) * normalizer;\n";
    kernel_source << "    out_row[i] = " << write_expr << ";\n";
    kernel_source << "  }\n";
    kernel_source << "}\n";
  }

  std::string source_str = kernel_source.str();

  // Get or compile kernel - use OpenCL 2.0 for parallel kernel (subgroup operations)
  cl_kernel kernel;
  if (use_parallel) {
    kernel = dev.get_kernel(kernel_name, source_str, "-cl-std=CL2.0");
  } else {
    kernel = dev.get_kernel(kernel_name, source_str);
  }
  encoder.set_kernel(kernel);

  // Compute data offset for sliced arrays (offset is in bytes, convert to elements)
  int64_t in_data_offset = in.offset() / in.itemsize();

  if (use_parallel) {
    // Set arguments for parallel workgroup kernel
    encoder.set_input_array(in, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(axis_size, 2);
    encoder.set_bytes(n_rows, 3);
    encoder.set_bytes(in_data_offset, 4);

    // Dispatch one workgroup per row, 256 threads per workgroup
    size_t local_size[3] = {256, 0, 0};
    size_t global_size[3] = {static_cast<size_t>(n_rows) * 256, 0, 0};
    encoder.dispatch_threads(global_size, local_size, 1);

    dev.add_temporary(in, s.index);
    return;
  } else if (use_vectorized) {
    // Set arguments for vectorized kernel
    encoder.set_input_array(in, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(axis_size, 2);
    encoder.set_bytes(n_rows, 3);
    encoder.set_bytes(in_data_offset, 4);
  } else {
    // Set arguments for strided kernel
    encoder.set_input_array(in, 0);
    encoder.set_output_array(out, 1);

    // Upload shape and strides arrays
    std::vector<int> shape_vec(row_shape.begin(), row_shape.end());
    std::vector<int64_t> strides_vec(row_strides.begin(), row_strides.end());

    // Handle 0-dimensional case (1D input becomes 0D after removing axis)
    if (shape_vec.empty()) {
      shape_vec.push_back(1);
      strides_vec.push_back(1);
    }

    size_t shape_bytes = shape_vec.size() * sizeof(int);
    size_t strides_bytes = strides_vec.size() * sizeof(int64_t);

    cl_int err;
    cl_mem shape_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, shape_bytes, shape_vec.data(), &err);
    cl_mem strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, strides_bytes, strides_vec.data(), &err);

    clSetKernelArg(kernel, 2, sizeof(cl_mem), &shape_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &strides_buf);
    int ndim_int = static_cast<int>(ndim > 0 ? ndim : 1);
    encoder.set_bytes(ndim_int, 4);
    encoder.set_bytes(axis_stride, 5);
    encoder.set_bytes(axis_size, 6);
    encoder.set_bytes(n_rows, 7);
    encoder.set_bytes(in_data_offset, 8);
  }

  // Dispatch one work-item per row
  size_t global_size[3] = {static_cast<size_t>(n_rows), 0, 0};
  encoder.dispatch_threads(global_size, nullptr, 1);

  dev.add_temporary(in, s.index);
}

} // namespace mlx::core
