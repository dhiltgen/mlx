// Copyright © 2025 MLX Contributors
// OpenCL fast operations

#include "mlx/fast_primitives.h"
#include "mlx/primitives.h"
#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/gpu/copy.h"

#include <sstream>

namespace mlx::core::fast {

bool RMSNorm::use_fallback(Stream s) {
  return false;  // GPU kernel enabled with float16 support
}

void RMSNorm::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& s = stream();
  auto& dev = opencl::device(s.device);
  auto& out = outputs[0];

  // Get inputs
  array x = inputs[0];
  const array& w = inputs[1];

  // Ensure x is row contiguous (last dim stride = 1)
  bool needs_copy = !x.flags().contiguous || x.strides()[x.ndim() - 1] != 1;
  if (needs_copy && x.ndim() > 1) {
    auto st = x.strides()[x.ndim() - 2];
    needs_copy |= (st != 0 && st != x.shape().back() && x.shape(-2) != 1);
  }
  if (needs_copy) {
    x = contiguous_copy_gpu(x, s);
  }

  // Setup output
  out.set_data(
      opencl::allocator().malloc(x.data_size() * x.itemsize()),
      x.data_size(),
      x.strides(),
      x.flags());

  auto axis_size = static_cast<int>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  // Get type info using centralized utilities
  std::string type_name = opencl::type_to_name(x.dtype());
  std::string type_suffix = opencl::type_to_suffix(x.dtype());
  auto conv = opencl::get_type_conversion(x.dtype());

  // Build RMSNorm kernel - parallel version using subgroup reduction
  // Formula: out = x * rsqrt(mean(x^2) + eps) * weight
  // Uses 64 threads per row with sub_group_reduce_add for parallel reduction
  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();

  // Kernel naming convention: {op}_{variant}_{type}
  std::string kernel_name = "rmsnorm_parallel_" + type_suffix;

  // Calculate elements per thread
  constexpr int WG_SIZE = 64;
  int elems_per_thread = (axis_size + WG_SIZE - 1) / WG_SIZE;

  // Generate unified kernel using type conversion utilities
  kernel_source << R"(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

__kernel void )" << kernel_name << R"((
    __global const )" << type_name << R"(* x,
    __global const )" << type_name << R"(* w,
    __global )" << type_name << R"(* out,
    float eps,
    int axis_size,
    int w_stride,
    int elems_per_thread) {

    int row = get_group_id(0);
    int lid = get_local_id(0);

    __global const )" << type_name << R"(* x_row = x + row * axis_size;
    __global )" << type_name << R"(* out_row = out + row * axis_size;

    // Each thread computes partial sum of squares
    float partial_sum = 0.0f;
    for (int e = 0; e < elems_per_thread; e++) {
        int idx = lid + e * 64;
        if (idx < axis_size) {
            float val = )" << opencl::make_read_expr(x.dtype(), "x_row[idx]") << R"(;
            partial_sum += val * val;
        }
    }
)";

  // Only add barrier for non-float32 types (float32 doesn't need it in original)
  if (conv.needs_read_convert) {
    kernel_source << R"(
    // Ensure all threads have computed their partial sums before reduction
    barrier(CLK_LOCAL_MEM_FENCE);
)";
  }

  kernel_source << R"(
    // Parallel reduction using subgroup operations
    float sum_sq = sub_group_reduce_add(partial_sum);

    // Compute normalizer: rsqrt(mean(x^2) + eps)
    float mean_sq = sum_sq / (float)axis_size;
    float normalizer = rsqrt(mean_sq + eps);

    // Apply normalization and weight in parallel
    for (int e = 0; e < elems_per_thread; e++) {
        int idx = lid + e * 64;
        if (idx < axis_size) {
            float val = )" << opencl::make_read_expr(x.dtype(), "x_row[idx]") << R"(;
            float weight = )" << opencl::make_read_expr(x.dtype(), "w[idx * w_stride]") << R"(;
            out_row[idx] = )" << opencl::make_write_expr(x.dtype(), "val * normalizer * weight") << R"(;
        }
    }
}
)";


  std::string source_str = kernel_source.str();
  cl_kernel kernel = dev.get_kernel(kernel_name, source_str, "-cl-std=CL2.0");

  auto& encoder = dev.get_command_encoder(s.index);
  encoder.set_kernel(kernel);
  encoder.set_input_array(x, 0);
  encoder.set_input_array(w, 1);
  encoder.set_output_array(out, 2);
  encoder.set_bytes(eps_, 3);
  encoder.set_bytes(axis_size, 4);
  int w_stride = (w.ndim() == 1) ? static_cast<int>(w.strides()[0]) : 0;
  encoder.set_bytes(w_stride, 5);
  encoder.set_bytes(elems_per_thread, 6);

  // 64 threads per row (one workgroup per row)
  size_t local_size[3] = {WG_SIZE, 0, 0};
  size_t global_size[3] = {static_cast<size_t>(n_rows) * WG_SIZE, 0, 0};
  encoder.dispatch_threads(global_size, local_size, 1);

  dev.add_temporary(x, s.index);
  dev.end_encoding(s.index);
}

bool LayerNorm::use_fallback(Stream) {
  return false;  // GPU kernel enabled
}

void LayerNorm::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& s = stream();
  auto& dev = opencl::device(s.device);
  auto& out = outputs[0];

  // Get inputs: x, weight (optional), bias (optional)
  array x = inputs[0];
  bool has_weight = inputs.size() > 1 && inputs[1].size() > 0;
  bool has_bias = inputs.size() > 2 && inputs[2].size() > 0;

  // Ensure x is row contiguous (last dim stride = 1)
  bool needs_copy = !x.flags().contiguous || x.strides()[x.ndim() - 1] != 1;
  if (needs_copy && x.ndim() > 1) {
    auto st = x.strides()[x.ndim() - 2];
    needs_copy |= (st != 0 && st != x.shape().back() && x.shape(-2) != 1);
  }
  if (needs_copy) {
    x = contiguous_copy_gpu(x, s);
  }

  // Setup output
  out.set_data(
      opencl::allocator().malloc(x.data_size() * x.itemsize()),
      x.data_size(),
      x.strides(),
      x.flags());

  auto axis_size = static_cast<int>(x.shape().back());
  int n_rows = x.data_size() / axis_size;

  // Get type info using centralized utilities
  std::string type_name = opencl::type_to_name(x.dtype());
  std::string type_suffix = opencl::type_to_suffix(x.dtype());

  // Build LayerNorm kernel - parallel version using subgroup reduction
  // Formula: out = (x - mean) / sqrt(var + eps) * weight + bias
  // Where: mean = sum(x) / n, var = sum(x^2) / n - mean^2
  // Uses 64 threads per row with sub_group_reduce_add for parallel reduction
  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();

  // Kernel naming convention: {op}_{variant}_{type}
  std::string kernel_name = "layernorm_parallel_" + type_suffix;
  if (has_weight) kernel_name += "_w";
  if (has_bias) kernel_name += "_b";

  // Calculate elements per thread
  constexpr int WG_SIZE = 64;
  int elems_per_thread = (axis_size + WG_SIZE - 1) / WG_SIZE;

  kernel_source << R"(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

__kernel void )" << kernel_name << R"((
    __global const )" << type_name << R"(* x,)";

  if (has_weight) {
    kernel_source << R"(
    __global const )" << type_name << R"(* w,)";
  }
  if (has_bias) {
    kernel_source << R"(
    __global const )" << type_name << R"(* b,)";
  }

  kernel_source << R"(
    __global )" << type_name << R"(* out,
    float eps,
    int axis_size,)";

  if (has_weight) {
    kernel_source << R"(
    int w_stride,)";
  }
  if (has_bias) {
    kernel_source << R"(
    int b_stride,)";
  }

  kernel_source << R"(
    int elems_per_thread) {

    int row = get_group_id(0);
    int lid = get_local_id(0);

    __global const )" << type_name << R"(* x_row = x + row * axis_size;
    __global )" << type_name << R"(* out_row = out + row * axis_size;

    // Each thread computes partial sums of x and x^2
    float partial_sum = 0.0f;
    float partial_sum_sq = 0.0f;
    for (int e = 0; e < elems_per_thread; e++) {
        int idx = lid + e * 64;
        if (idx < axis_size) {
            float val = )" << opencl::make_read_expr(x.dtype(), "x_row[idx]") << R"(;
            partial_sum += val;
            partial_sum_sq += val * val;
        }
    }

    // Parallel reduction using subgroup operations
    float sum_x = sub_group_reduce_add(partial_sum);
    float sum_x2 = sub_group_reduce_add(partial_sum_sq);

    // Compute mean and variance
    // var = E[x^2] - E[x]^2 = sum(x^2)/n - (sum(x)/n)^2
    float mean = sum_x / (float)axis_size;
    float var = sum_x2 / (float)axis_size - mean * mean;
    float inv_std = rsqrt(var + eps);

    // Apply normalization, weight, and bias in parallel
    for (int e = 0; e < elems_per_thread; e++) {
        int idx = lid + e * 64;
        if (idx < axis_size) {
            float val = )" << opencl::make_read_expr(x.dtype(), "x_row[idx]") << R"(;
            float normalized = (val - mean) * inv_std;)";

  if (has_weight) {
    kernel_source << R"(
            float weight = )" << opencl::make_read_expr(x.dtype(), "w[idx * w_stride]") << R"(;
            normalized *= weight;)";
  }
  if (has_bias) {
    kernel_source << R"(
            float bias = )" << opencl::make_read_expr(x.dtype(), "b[idx * b_stride]") << R"(;
            normalized += bias;)";
  }

  kernel_source << R"(
            out_row[idx] = )" << opencl::make_write_expr(x.dtype(), "normalized") << R"(;
        }
    }
}
)";

  std::string source_str = kernel_source.str();
  cl_kernel kernel = dev.get_kernel(kernel_name, source_str, "-cl-std=CL2.0");

  auto& encoder = dev.get_command_encoder(s.index);
  encoder.set_kernel(kernel);

  int arg = 0;
  encoder.set_input_array(x, arg++);
  if (has_weight) {
    encoder.set_input_array(inputs[1], arg++);
  }
  if (has_bias) {
    encoder.set_input_array(inputs[2], arg++);
  }
  encoder.set_output_array(out, arg++);
  encoder.set_bytes(eps_, arg++);
  encoder.set_bytes(axis_size, arg++);
  if (has_weight) {
    int w_stride = (inputs[1].ndim() >= 1) ? static_cast<int>(inputs[1].strides()[0]) : 0;
    encoder.set_bytes(w_stride, arg++);
  }
  if (has_bias) {
    int b_stride = (inputs[2].ndim() >= 1) ? static_cast<int>(inputs[2].strides()[0]) : 0;
    encoder.set_bytes(b_stride, arg++);
  }
  encoder.set_bytes(elems_per_thread, arg++);

  // 64 threads per row (one workgroup per row)
  size_t local_size[3] = {WG_SIZE, 0, 0};
  size_t global_size[3] = {static_cast<size_t>(n_rows) * WG_SIZE, 0, 0};
  encoder.dispatch_threads(global_size, local_size, 1);

  dev.add_temporary(x, s.index);
  dev.end_encoding(s.index);
}

bool RoPE::use_fallback(Stream s) {
  return false;  // GPU kernel enabled with float16 support and freqs
}

// Helper functions to generate read/write expressions for different types
// These now delegate to the centralized utilities in types.h
static std::string gen_read_expr(Dtype dtype, const std::string& arr_access) {
  return opencl::make_read_expr(dtype, arr_access);
}

static std::string gen_write_expr(Dtype dtype, const std::string& value) {
  return opencl::make_write_expr(dtype, value);
}

// Helper to generate single-timestep RoPE kernel (with base)
static std::string gen_rope_single_base_kernel(const std::string& kernel_name, Dtype dtype) {
  std::ostringstream k;
  std::string in_type = opencl::type_to_name(dtype);

  k << R"(
__kernel void )" << kernel_name << R"((
    __global const )" << in_type << R"(* in,
    __global )" << in_type << R"(* out,
    __global const int* offset_ptr,
    float scale,
    long stride,
    float base_log2,
    int dims,
    int forward,
    int traditional,
    long in_data_offset,
    long out_data_offset) {

    int pos_x = get_global_id(0);
    int pos_y = get_global_id(1);
    int half_dims = dims / 2;
    if (pos_x >= half_dims) return;

    int offset_val = offset_ptr[0];
    float L = scale * (float)offset_val;

    float d = (float)pos_x / (float)half_dims;
    float inv_freq = exp2(-d * base_log2);
    float theta = L * inv_freq;
    float costheta = cos(theta);
    float sintheta = sin(theta);

    long in_index_1, in_index_2, out_index_1, out_index_2;
    if (traditional) {
        in_index_1 = 2 * pos_x + pos_y * stride + in_data_offset;
        in_index_2 = in_index_1 + 1;
        out_index_1 = 2 * pos_x + pos_y * stride + out_data_offset;
        out_index_2 = out_index_1 + 1;
    } else {
        in_index_1 = pos_x + pos_y * stride + in_data_offset;
        in_index_2 = in_index_1 + half_dims;
        out_index_1 = pos_x + pos_y * stride + out_data_offset;
        out_index_2 = out_index_1 + half_dims;
    }

    float x1 = )" << gen_read_expr(dtype, "in[in_index_1]") << R"(;
    float x2 = )" << gen_read_expr(dtype, "in[in_index_2]") << R"(;

    float rx1, rx2;
    if (forward) {
        rx1 = x1 * costheta - x2 * sintheta;
        rx2 = x1 * sintheta + x2 * costheta;
    } else {
        rx1 = x2 * sintheta + x1 * costheta;
        rx2 = x2 * costheta - x1 * sintheta;
    }

    out[out_index_1] = )" << gen_write_expr(dtype, "rx1") << R"(;
    out[out_index_2] = )" << gen_write_expr(dtype, "rx2") << R"(;
}
)";
  return k.str();
}

// Helper to generate single-timestep RoPE kernel (with freqs array)
static std::string gen_rope_single_freqs_kernel(const std::string& kernel_name, Dtype dtype) {
  std::ostringstream k;
  std::string in_type = opencl::type_to_name(dtype);

  k << R"(
__kernel void )" << kernel_name << R"((
    __global const )" << in_type << R"(* in,
    __global )" << in_type << R"(* out,
    __global const int* offset_ptr,
    __global const float* freqs,
    long freq_stride,
    float scale,
    long stride,
    int dims,
    int forward,
    int traditional,
    long in_data_offset,
    long out_data_offset) {

    int pos_x = get_global_id(0);
    int pos_y = get_global_id(1);
    int half_dims = dims / 2;
    if (pos_x >= half_dims) return;

    int offset_val = offset_ptr[0];
    float L = scale * (float)offset_val;

    // Read pre-computed frequency and compute inv_freq = 1/freq
    float inv_freq = 1.0f / freqs[freq_stride * pos_x];
    float theta = L * inv_freq;
    float costheta = cos(theta);
    float sintheta = sin(theta);

    long in_index_1, in_index_2, out_index_1, out_index_2;
    if (traditional) {
        in_index_1 = 2 * pos_x + pos_y * stride + in_data_offset;
        in_index_2 = in_index_1 + 1;
        out_index_1 = 2 * pos_x + pos_y * stride + out_data_offset;
        out_index_2 = out_index_1 + 1;
    } else {
        in_index_1 = pos_x + pos_y * stride + in_data_offset;
        in_index_2 = in_index_1 + half_dims;
        out_index_1 = pos_x + pos_y * stride + out_data_offset;
        out_index_2 = out_index_1 + half_dims;
    }

    float x1 = )" << gen_read_expr(dtype, "in[in_index_1]") << R"(;
    float x2 = )" << gen_read_expr(dtype, "in[in_index_2]") << R"(;

    float rx1, rx2;
    if (forward) {
        rx1 = x1 * costheta - x2 * sintheta;
        rx2 = x1 * sintheta + x2 * costheta;
    } else {
        rx1 = x2 * sintheta + x1 * costheta;
        rx2 = x2 * costheta - x1 * sintheta;
    }

    out[out_index_1] = )" << gen_write_expr(dtype, "rx1") << R"(;
    out[out_index_2] = )" << gen_write_expr(dtype, "rx2") << R"(;
}
)";
  return k.str();
}

// Helper to generate general RoPE kernel (with base)
static std::string gen_rope_general_base_kernel(const std::string& kernel_name, Dtype dtype) {
  std::ostringstream k;
  std::string in_type = opencl::type_to_name(dtype);

  k << R"(
__kernel void )" << kernel_name << R"((
    __global const )" << in_type << R"(* in,
    __global )" << in_type << R"(* out,
    __global const int* offset_ptr,
    float scale,
    long stride0, long stride1, long stride2,
    long out_stride0, long out_stride1, long out_stride2,
    long offset_stride,
    int n_head,
    float base_log2,
    int dims,
    int forward,
    int traditional,
    long in_data_offset,
    long out_data_offset) {

    int pos_x = get_global_id(0);
    int pos_y = get_global_id(1);
    int pos_z = get_global_id(2);
    int half_dims = dims / 2;
    if (pos_x >= half_dims) return;

    int head_idx = pos_z % n_head;
    int batch_idx = pos_z / n_head;
    int offset_val = offset_ptr[batch_idx * offset_stride];
    float L = scale * (float)(pos_y + offset_val);

    float d = (float)pos_x / (float)half_dims;
    float inv_freq = exp2(-d * base_log2);
    float theta = L * inv_freq;
    float costheta = cos(theta);
    float sintheta = sin(theta);

    long mat_idx = batch_idx * n_head + head_idx;
    long in_index_1 = pos_y * stride1 + mat_idx * stride0 + in_data_offset;
    long in_index_2;
    long out_index_1 = pos_y * out_stride1 + mat_idx * out_stride0 + out_data_offset;
    long out_index_2;

    if (traditional) {
        out_index_1 += 2 * pos_x * out_stride2;
        out_index_2 = out_index_1 + out_stride2;
        in_index_1 += 2 * pos_x * stride2;
        in_index_2 = in_index_1 + stride2;
    } else {
        out_index_1 += pos_x * out_stride2;
        out_index_2 = out_index_1 + half_dims * out_stride2;
        in_index_1 += pos_x * stride2;
        in_index_2 = in_index_1 + half_dims * stride2;
    }

    float x1 = )" << gen_read_expr(dtype, "in[in_index_1]") << R"(;
    float x2 = )" << gen_read_expr(dtype, "in[in_index_2]") << R"(;

    float rx1, rx2;
    if (forward) {
        rx1 = x1 * costheta - x2 * sintheta;
        rx2 = x1 * sintheta + x2 * costheta;
    } else {
        rx1 = x2 * sintheta + x1 * costheta;
        rx2 = x2 * costheta - x1 * sintheta;
    }

    out[out_index_1] = )" << gen_write_expr(dtype, "rx1") << R"(;
    out[out_index_2] = )" << gen_write_expr(dtype, "rx2") << R"(;
}
)";
  return k.str();
}

// Helper to generate general RoPE kernel (with freqs array)
static std::string gen_rope_general_freqs_kernel(const std::string& kernel_name, Dtype dtype) {
  std::ostringstream k;
  std::string in_type = opencl::type_to_name(dtype);

  k << R"(
__kernel void )" << kernel_name << R"((
    __global const )" << in_type << R"(* in,
    __global )" << in_type << R"(* out,
    __global const int* offset_ptr,
    __global const float* freqs,
    long freq_stride,
    float scale,
    long stride0, long stride1, long stride2,
    long out_stride0, long out_stride1, long out_stride2,
    long offset_stride,
    int n_head,
    int dims,
    int forward,
    int traditional,
    long in_data_offset,
    long out_data_offset) {

    int pos_x = get_global_id(0);
    int pos_y = get_global_id(1);
    int pos_z = get_global_id(2);
    int half_dims = dims / 2;
    if (pos_x >= half_dims) return;

    int head_idx = pos_z % n_head;
    int batch_idx = pos_z / n_head;
    int offset_val = offset_ptr[batch_idx * offset_stride];
    float L = scale * (float)(pos_y + offset_val);

    // Read pre-computed frequency and compute inv_freq = 1/freq
    float inv_freq = 1.0f / freqs[freq_stride * pos_x];
    float theta = L * inv_freq;
    float costheta = cos(theta);
    float sintheta = sin(theta);

    long mat_idx = batch_idx * n_head + head_idx;
    long in_index_1 = pos_y * stride1 + mat_idx * stride0 + in_data_offset;
    long in_index_2;
    long out_index_1 = pos_y * out_stride1 + mat_idx * out_stride0 + out_data_offset;
    long out_index_2;

    if (traditional) {
        out_index_1 += 2 * pos_x * out_stride2;
        out_index_2 = out_index_1 + out_stride2;
        in_index_1 += 2 * pos_x * stride2;
        in_index_2 = in_index_1 + stride2;
    } else {
        out_index_1 += pos_x * out_stride2;
        out_index_2 = out_index_1 + half_dims * out_stride2;
        in_index_1 += pos_x * stride2;
        in_index_2 = in_index_1 + half_dims * stride2;
    }

    float x1 = )" << gen_read_expr(dtype, "in[in_index_1]") << R"(;
    float x2 = )" << gen_read_expr(dtype, "in[in_index_2]") << R"(;

    float rx1, rx2;
    if (forward) {
        rx1 = x1 * costheta - x2 * sintheta;
        rx2 = x1 * sintheta + x2 * costheta;
    } else {
        rx1 = x2 * sintheta + x1 * costheta;
        rx2 = x2 * costheta - x1 * sintheta;
    }

    out[out_index_1] = )" << gen_write_expr(dtype, "rx1") << R"(;
    out[out_index_2] = )" << gen_write_expr(dtype, "rx2") << R"(;
}
)";
  return k.str();
}

void RoPE::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& s = stream();
  auto& dev = opencl::device(s.device);
  auto& out = outputs[0];

  array in = inputs[0];
  const array& offset = inputs[1];
  bool with_freqs = inputs.size() == 3;

  int ndim = in.ndim();
  int B = in.shape(0);
  int T = in.shape(-2);
  int D = in.shape(-1);
  size_t mat_size = T * D;

  int N = 1;
  for (int i = 1; i < (ndim - 2); ++i) {
    N *= in.shape(i);
  }

  // Handle contiguity
  int64_t strides[3];
  int64_t out_strides[3];
  bool donated = false;

  if (dims_ < D) {
    donated = true;
    auto ctype = (in.flags().row_contiguous) ? CopyType::Vector : CopyType::General;
    copy_gpu(in, out, ctype, s);
    strides[0] = mat_size;
    strides[1] = out.strides()[ndim - 2];
    strides[2] = out.strides()[ndim - 1];
  } else if (in.flags().row_contiguous) {
    if (in.is_donatable()) {
      donated = true;
      out.copy_shared_buffer(in);
    } else {
      out.set_data(opencl::allocator().malloc(out.nbytes()));
    }
    strides[0] = mat_size;
    strides[1] = in.strides()[ndim - 2];
    strides[2] = in.strides()[ndim - 1];
  } else {
    donated = true;
    copy_gpu(in, out, CopyType::General, s);
    strides[0] = mat_size;
    strides[1] = out.strides()[ndim - 2];
    strides[2] = out.strides()[ndim - 1];
  }
  out_strides[0] = mat_size;
  out_strides[1] = out.strides()[ndim - 2];
  out_strides[2] = out.strides()[ndim - 1];

  bool single = in.flags().row_contiguous && T == 1 && offset.size() == 1;

  std::string type_name = opencl::type_to_name(in.dtype());
  std::string type_suffix = opencl::type_to_suffix(in.dtype());
  Dtype dtype = in.dtype();

  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();

  float base_log2 = std::log2(base_);

  // Compute data offsets for sliced arrays
  // When reading: if donated, we read from out (which may share in's buffer); otherwise from in
  const array& read_arr = donated ? out : in;
  int64_t in_data_offset = static_cast<int64_t>(read_arr.offset() / read_arr.itemsize());
  int64_t out_data_offset = static_cast<int64_t>(out.offset() / out.itemsize());

  if (single && !with_freqs) {
    // Single timestep with base
    // Kernel naming convention: {op}_{variant}_{type}
    std::string kernel_name = "rope_single_" + type_suffix;
    kernel_source << gen_rope_single_base_kernel(kernel_name, dtype);

    std::string source_str = kernel_source.str();
    cl_kernel kernel = dev.get_kernel(kernel_name, source_str);

    auto& encoder = dev.get_command_encoder(s.index);
    encoder.set_kernel(kernel);
    encoder.set_input_array(donated ? out : in, 0);
    encoder.set_output_array(out, 1);
    encoder.set_input_array(offset, 2);
    encoder.set_bytes(scale_, 3);
    encoder.set_bytes(out_strides[0], 4);
    encoder.set_bytes(base_log2, 5);
    encoder.set_bytes(dims_, 6);
    encoder.set_bytes(forward_ ? 1 : 0, 7);
    encoder.set_bytes(traditional_ ? 1 : 0, 8);
    encoder.set_bytes(in_data_offset, 9);
    encoder.set_bytes(out_data_offset, 10);

    size_t global_size[3] = {static_cast<size_t>(dims_ / 2), static_cast<size_t>(N), 0};
    encoder.dispatch_threads(global_size, nullptr, 2);
    dev.end_encoding(s.index);

  } else if (single && with_freqs) {
    // Single timestep with freqs array
    std::string kernel_name = "rope_single_freqs_" + type_suffix;
    kernel_source << gen_rope_single_freqs_kernel(kernel_name, dtype);

    const array& freqs = inputs[2];
    int64_t freq_stride = (freqs.ndim() > 0) ? freqs.strides()[0] : 0;

    std::string source_str = kernel_source.str();
    cl_kernel kernel = dev.get_kernel(kernel_name, source_str);

    auto& encoder = dev.get_command_encoder(s.index);
    encoder.set_kernel(kernel);
    encoder.set_input_array(donated ? out : in, 0);
    encoder.set_output_array(out, 1);
    encoder.set_input_array(offset, 2);
    encoder.set_input_array(freqs, 3);
    encoder.set_bytes(freq_stride, 4);
    encoder.set_bytes(scale_, 5);
    encoder.set_bytes(out_strides[0], 6);
    encoder.set_bytes(dims_, 7);
    encoder.set_bytes(forward_ ? 1 : 0, 8);
    encoder.set_bytes(traditional_ ? 1 : 0, 9);
    encoder.set_bytes(in_data_offset, 10);
    encoder.set_bytes(out_data_offset, 11);

    size_t global_size[3] = {static_cast<size_t>(dims_ / 2), static_cast<size_t>(N), 0};
    encoder.dispatch_threads(global_size, nullptr, 2);
    dev.end_encoding(s.index);

  } else if (!single && !with_freqs) {
    // General (T > 1) with base
    std::string kernel_name = "rope_general_" + type_suffix;
    kernel_source << gen_rope_general_base_kernel(kernel_name, dtype);

    std::string source_str = kernel_source.str();
    cl_kernel kernel = dev.get_kernel(kernel_name, source_str);

    auto& encoder = dev.get_command_encoder(s.index);
    encoder.set_kernel(kernel);
    encoder.set_input_array(donated ? out : in, 0);
    encoder.set_output_array(out, 1);
    encoder.set_input_array(offset, 2);
    encoder.set_bytes(scale_, 3);
    encoder.set_bytes(strides[0], 4);
    encoder.set_bytes(strides[1], 5);
    encoder.set_bytes(strides[2], 6);
    encoder.set_bytes(out_strides[0], 7);
    encoder.set_bytes(out_strides[1], 8);
    encoder.set_bytes(out_strides[2], 9);
    int64_t offset_stride = (offset.ndim() > 0) ? offset.strides()[0] : 0;
    encoder.set_bytes(offset_stride, 10);
    encoder.set_bytes(N, 11);
    encoder.set_bytes(base_log2, 12);
    encoder.set_bytes(dims_, 13);
    encoder.set_bytes(forward_ ? 1 : 0, 14);
    encoder.set_bytes(traditional_ ? 1 : 0, 15);
    encoder.set_bytes(in_data_offset, 16);
    encoder.set_bytes(out_data_offset, 17);

    size_t global_size[3] = {
        static_cast<size_t>(dims_ / 2),
        static_cast<size_t>(T),
        static_cast<size_t>(B * N)
    };
    encoder.dispatch_threads(global_size, nullptr, 3);
    dev.end_encoding(s.index);

  } else {
    // General (T > 1) with freqs array
    std::string kernel_name = "rope_general_freqs_" + type_suffix;
    kernel_source << gen_rope_general_freqs_kernel(kernel_name, dtype);

    const array& freqs = inputs[2];
    int64_t freq_stride = (freqs.ndim() > 0) ? freqs.strides()[0] : 0;

    std::string source_str = kernel_source.str();
    cl_kernel kernel = dev.get_kernel(kernel_name, source_str);

    auto& encoder = dev.get_command_encoder(s.index);
    encoder.set_kernel(kernel);
    encoder.set_input_array(donated ? out : in, 0);
    encoder.set_output_array(out, 1);
    encoder.set_input_array(offset, 2);
    encoder.set_input_array(freqs, 3);
    encoder.set_bytes(freq_stride, 4);
    encoder.set_bytes(scale_, 5);
    encoder.set_bytes(strides[0], 6);
    encoder.set_bytes(strides[1], 7);
    encoder.set_bytes(strides[2], 8);
    encoder.set_bytes(out_strides[0], 9);
    encoder.set_bytes(out_strides[1], 10);
    encoder.set_bytes(out_strides[2], 11);
    int64_t offset_stride = (offset.ndim() > 0) ? offset.strides()[0] : 0;
    encoder.set_bytes(offset_stride, 12);
    encoder.set_bytes(N, 13);
    encoder.set_bytes(dims_, 14);
    encoder.set_bytes(forward_ ? 1 : 0, 15);
    encoder.set_bytes(traditional_ ? 1 : 0, 16);
    encoder.set_bytes(in_data_offset, 17);
    encoder.set_bytes(out_data_offset, 18);

    size_t global_size[3] = {
        static_cast<size_t>(dims_ / 2),
        static_cast<size_t>(T),
        static_cast<size_t>(B * N)
    };
    encoder.dispatch_threads(global_size, nullptr, 3);
    dev.end_encoding(s.index);
  }
}

bool ScaledDotProductAttention::use_fallback(
    const array& q,
    const array& k,
    const array& v,
    bool has_mask,
    bool has_arr_mask,
    bool do_causal,
    bool is_training,
    bool output_logsumexp,
    Stream s) {
  // Use CPU for training
  if (is_training || output_logsumexp) {
    return true;
  }

  // Only support generation case (single query, seq_len <= 8)
  const int query_sequence_length = q.shape(2);
  const int key_sequence_length = k.shape(2);
  const int query_head_dim = q.shape(-1);
  const int value_head_dim = v.shape(-1);

  // Support common head dimensions: 32, 64, 96, 128, 256
  // The kernel handles any head_dim via elems_per_thread = (head_dim + 63) / 64
  // with proper boundary checks (if idx < HEAD_DIM)
  const bool supported_head_dim = (query_head_dim == value_head_dim) &&
      (query_head_dim == 32 || query_head_dim == 64 || query_head_dim == 96 ||
       query_head_dim == 128 || query_head_dim == 256);

  // Support both generation (q_seq_len=1) and prefill (q_seq_len > 1)
  // The kernel launches one workgroup per (batch, head, query_position)
  // For prefill, causal masking is handled in the kernel via kv_limit
  const bool gpu_supported = (query_sequence_length <= key_sequence_length) &&
      supported_head_dim;

  // Causal mask is now supported on GPU
  // has_mask indicates "causal" string, which we handle
  // Float array masks are supported (added to score before softmax)
  // Bool masks are NOT supported yet

  return !gpu_supported;
}

bool ScaledDotProductAttention::supports_bool_mask() {
  return false;  // Bool masks not supported in OpenCL backend yet (float masks are)
}

// Helper to generate parallel SDPA kernel using subgroup operations
// Each workgroup (64 threads = 1 subgroup on Adreno) handles one query head
// Threads cooperate on dot product via sub_group_reduce_add
static std::string gen_sdpa_parallel_kernel(const std::string& kernel_name, Dtype dtype, int head_dim, bool has_sinks, bool do_causal, bool has_float_mask) {
  std::ostringstream k;
  std::string in_type = opencl::type_to_name(dtype);

  // For head_dim=64, each of 64 threads handles 1 element
  // For head_dim=128, each of 64 threads handles 2 elements
  int elems_per_thread = (head_dim + 63) / 64;

  k << R"(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define WG_SIZE 64
#define ELEMS_PER_THREAD )" << elems_per_thread << R"(
#define HEAD_DIM )" << head_dim << R"(

__kernel void )" << kernel_name << R"((
    __global const )" << in_type << R"(* queries,
    __global const )" << in_type << R"(* keys,
    __global const )" << in_type << R"(* values,
    __global )" << in_type << R"(* out,
    int gqa_factor,
    int N,
    long k_head_stride,
    long k_seq_stride,
    long v_head_stride,
    long v_seq_stride,
    float scale,
    int n_q_heads,
    long q_batch_stride,
    long q_head_stride,
    long o_batch_stride,
    long q_data_offset,
    long k_data_offset,
    long v_data_offset,
    int q_seq_len,
    int do_causal)";

  // Add sinks parameter if needed
  if (has_sinks) {
    k << R"(,
    __global const )" << in_type << R"(* sinks)";
  }

  // Add float mask parameters if needed
  if (has_float_mask) {
    k << R"(,
    __global const )" << in_type << R"(* mask,
    long mask_head_stride,
    long mask_q_seq_stride,
    long mask_kv_seq_stride,
    long mask_data_offset)";
  }

  k << R"() {

    // Workgroup handles one query position for one head
    // Total workgroups = B * n_q_heads * q_seq_len
    int wg_idx = get_group_id(0);
    int q_pos = wg_idx % q_seq_len;               // query position within sequence
    int temp = wg_idx / q_seq_len;
    int q_head_idx = temp % n_q_heads;            // query head index
    int batch_idx = temp / n_q_heads;             // batch index
    int kv_head_idx = q_head_idx / gqa_factor;

    int lid = get_local_id(0);

    // Base pointers - use data offsets for sliced arrays where buffer.ptr() != data()
    // q_ptr now accounts for the query position within the sequence
    __global const )" << in_type << R"(* q_ptr = queries + q_data_offset + batch_idx * q_batch_stride + q_head_idx * q_head_stride + q_pos * HEAD_DIM;
    __global const )" << in_type << R"(* k_base = keys + k_data_offset + batch_idx * (k_head_stride * (n_q_heads / gqa_factor)) + kv_head_idx * k_head_stride;
    __global const )" << in_type << R"(* v_base = values + v_data_offset + batch_idx * (v_head_stride * (n_q_heads / gqa_factor)) + kv_head_idx * v_head_stride;
    // Output pointer also accounts for query position
    __global )" << in_type << R"(* o_ptr = out + batch_idx * o_batch_stride + q_head_idx * HEAD_DIM * q_seq_len + q_pos * HEAD_DIM;
)";

  // Add mask base pointer if needed
  if (has_float_mask) {
    k << R"(
    // Mask pointer: mask[batch, head, q_pos, kv_pos]
    // Note: mask may be broadcast (strides = 0 for broadcast dims)
    __global const )" << in_type << R"(* mask_base = mask + mask_data_offset +
        batch_idx * (mask_head_stride * n_q_heads) +
        q_head_idx * mask_head_stride +
        q_pos * mask_q_seq_stride;
)";
  }

  k << R"(
    // For causal masking: how many key positions can this query attend to
    // In prefill: q_pos can attend to positions 0..q_pos (inclusive)
    // The key sequence (N) may be longer if there's a cache
    int max_kv_pos = do_causal ? (q_pos + 1) : N;

    // Each thread loads its portion of the query (scaled)
    float q[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int idx = lid + e * WG_SIZE;
        if (idx < HEAD_DIM) {
            q[e] = scale * )" << gen_read_expr(dtype, "q_ptr[idx]") << R"(;
        }
    }

    // Initialize output accumulator
    float o[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        o[e] = 0.0f;
    }

    // Online softmax state (same across all threads)
    float max_score = -1e20f;
    float sum_exp = 0.0f;
)";

  // Handle sinks initialization - all threads read the same sink value
  if (has_sinks) {
    k << R"(
    // Initialize with attention sink (virtual token with pre-computed attention weight)
    max_score = )" << gen_read_expr(dtype, "sinks[q_head_idx]") << R"(;
    sum_exp = 1.0f;
)";
  }

  k << R"(
    // Loop over all keys/values (limited by causal mask if applicable)
    // For causal: during prefill (q_seq_len > 1), limit to seq_idx <= q_pos
    int kv_limit = (do_causal && q_seq_len > 1) ? min(q_pos + 1, N) : N;
    for (int seq_idx = 0; seq_idx < kv_limit; seq_idx++) {
        __global const )" << in_type << R"(* k_ptr = k_base + seq_idx * k_seq_stride;
        __global const )" << in_type << R"(* v_ptr = v_base + seq_idx * v_seq_stride;

        // Each thread computes partial dot product for its elements
        float partial_score = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int idx = lid + e * WG_SIZE;
            if (idx < HEAD_DIM) {
                float k_val = )" << gen_read_expr(dtype, "k_ptr[idx]") << R"(;
                partial_score += q[e] * k_val;
            }
        }

        // Reduce across subgroup to get full dot product (all threads get same result)
        float score = sub_group_reduce_add(partial_score);
)";

  // Add mask value to score if needed
  if (has_float_mask) {
    k << R"(
        // Add mask value to score (for padding masks, use -inf to mask out)
        float mask_val = )" << gen_read_expr(dtype, "mask_base[seq_idx * mask_kv_seq_stride]") << R"(;
        score += mask_val;
)";
  }

  k << R"(
        // Online softmax update (all threads compute same values)
        float new_max = max(max_score, score);
        float factor = exp(max_score - new_max);
        float exp_score = exp(score - new_max);

        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;

        // Each thread updates its portion of output
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; e++) {
            int idx = lid + e * WG_SIZE;
            if (idx < HEAD_DIM) {
                float v_val = )" << gen_read_expr(dtype, "v_ptr[idx]") << R"(;
                o[e] = o[e] * factor + exp_score * v_val;
            }
        }
    }

    // Normalize and write output
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; e++) {
        int idx = lid + e * WG_SIZE;
        if (idx < HEAD_DIM) {
            o_ptr[idx] = )" << gen_write_expr(dtype, "o[e] * inv_sum") << R"(;
        }
    }
}
)";
  return k.str();
}

// Helper to generate SDPA vector kernel for single-query attention (generation)
// This is the naive fallback kernel (one work item per head)
static std::string gen_sdpa_vector_kernel(const std::string& kernel_name, Dtype dtype, int head_dim) {
  std::ostringstream k;
  std::string in_type = opencl::type_to_name(dtype);

  k << R"(
__kernel void )" << kernel_name << R"((
    __global const )" << in_type << R"(* queries,
    __global const )" << in_type << R"(* keys,
    __global const )" << in_type << R"(* values,
    __global )" << in_type << R"(* out,
    int gqa_factor,
    int N,
    long k_head_stride,
    long k_seq_stride,
    long v_head_stride,
    long v_seq_stride,
    float scale,
    int n_q_heads,
    long q_batch_stride,
    long o_batch_stride) {

    // Each work item handles one query head (batch_idx * n_q_heads + head_idx)
    int q_batch_head_idx = get_global_id(0);
    int batch_idx = q_batch_head_idx / n_q_heads;
    int q_head_idx = q_batch_head_idx % n_q_heads;
    int kv_head_idx = q_head_idx / gqa_factor;

    // Pointers
    __global const )" << in_type << R"(* q_ptr = queries + batch_idx * q_batch_stride + q_head_idx * )" << head_dim << R"(;
    __global const )" << in_type << R"(* k_base = keys + batch_idx * (k_head_stride * (n_q_heads / gqa_factor)) + kv_head_idx * k_head_stride;
    __global const )" << in_type << R"(* v_base = values + batch_idx * (v_head_stride * (n_q_heads / gqa_factor)) + kv_head_idx * v_head_stride;
    __global )" << in_type << R"(* o_ptr = out + batch_idx * o_batch_stride + q_head_idx * )" << head_dim << R"(;

    // Load query and scale
    float q[)" << head_dim << R"(];
    for (int i = 0; i < )" << head_dim << R"(; i++) {
        q[i] = scale * )" << gen_read_expr(dtype, "q_ptr[i]") << R"(;
    }

    // Initialize output accumulator and online softmax state
    float o[)" << head_dim << R"(];
    for (int i = 0; i < )" << head_dim << R"(; i++) {
        o[i] = 0.0f;
    }
    float max_score = -1e20f;
    float sum_exp = 0.0f;

    // Loop over all keys/values
    for (int seq_idx = 0; seq_idx < N; seq_idx++) {
        __global const )" << in_type << R"(* k_ptr = k_base + seq_idx * k_seq_stride;
        __global const )" << in_type << R"(* v_ptr = v_base + seq_idx * v_seq_stride;

        // Compute dot product: score = Q · K
        float score = 0.0f;
        for (int i = 0; i < )" << head_dim << R"(; i++) {
            float k_val = )" << gen_read_expr(dtype, "k_ptr[i]") << R"(;
            score += q[i] * k_val;
        }

        // Online softmax update
        float new_max = max(max_score, score);
        float factor = exp(max_score - new_max);
        float exp_score = exp(score - new_max);

        // Update accumulators
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;

        // Update output: o = o * factor + exp_score * V
        for (int i = 0; i < )" << head_dim << R"(; i++) {
            float v_val = )" << gen_read_expr(dtype, "v_ptr[i]") << R"(;
            o[i] = o[i] * factor + exp_score * v_val;
        }
    }

    // Normalize and write output
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
    for (int i = 0; i < )" << head_dim << R"(; i++) {
        o_ptr[i] = )" << gen_write_expr(dtype, "o[i] * inv_sum") << R"(;
    }
}
)";
  return k.str();
}

void ScaledDotProductAttention::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  auto& s = stream();
  auto& dev = opencl::device(s.device);

  auto& q = inputs[0];
  auto& k = inputs[1];
  auto& v = inputs[2];
  auto& out = outputs[0];

  // Check for mask array
  // inputs layout: [q, k, v, (sinks if has_sinks_), (mask if present)]
  bool has_arr_mask = inputs.size() > (3 + (has_sinks_ ? 1 : 0));
  bool has_float_mask = false;
  const array* mask_ptr = nullptr;

  if (has_arr_mask) {
    mask_ptr = &inputs[3 + (has_sinks_ ? 1 : 0)];
    // Bool masks fall back to CPU (checked in use_fallback)
    // Float masks are supported
    has_float_mask = (mask_ptr->dtype() != bool_);
    if (!has_float_mask) {
      throw std::runtime_error("[SDPA::eval_gpu] Bool masks not supported on OpenCL - should have fallen back to CPU");
    }
  }

  // Get dimensions
  int B = q.shape(0);           // batch size
  int n_q_heads = q.shape(1);   // number of query heads
  int q_seq_len = q.shape(2);   // query sequence length (1 for generation)
  int head_dim = q.shape(3);    // head dimension

  int n_kv_heads = k.shape(1);  // number of key/value heads
  int kv_seq_len = k.shape(2);  // key/value sequence length (cache length)

  int gqa_factor = n_q_heads / n_kv_heads;

  // Calculate strides
  size_t k_head_stride = k.shape(1) == 1 ? k.strides(0) : k.strides(1);
  size_t k_seq_stride = k.strides(2);
  size_t v_head_stride = v.shape(1) == 1 ? v.strides(0) : v.strides(1);
  size_t v_seq_stride = v.strides(2);

  // Query strides - properly handle non-contiguous (sliced) arrays
  size_t q_batch_stride = q.strides(0);
  size_t q_head_stride = q.strides(1);  // Actual stride between query heads
  // Output is always contiguous [B, n_q_heads, q_seq_len, head_dim]
  size_t o_batch_stride = n_q_heads * q_seq_len * head_dim;

  // For sliced arrays, the buffer pointer is the base of the allocation,
  // but the actual data starts at buffer + offset. We need to pass this
  // offset to the kernel so it reads from the correct location.
  // offset() returns bytes, so divide by itemsize to get element offset.
  int64_t q_data_offset = q.offset() / q.itemsize();
  int64_t k_data_offset = k.offset() / k.itemsize();
  int64_t v_data_offset = v.offset() / v.itemsize();

  // Mask strides if present
  // Mask shape is [batch, heads, q_seq, kv_seq] or broadcastable
  int64_t mask_head_stride = 0;
  int64_t mask_q_seq_stride = 0;
  int64_t mask_kv_seq_stride = 0;
  int64_t mask_data_offset = 0;

  if (has_float_mask) {
    const array& mask = *mask_ptr;
    // Handle broadcast: if dimension is 1, stride is effectively 0
    // Mask shape: [batch?, heads?, q_seq?, kv_seq]
    // The mask is broadcast to [B, n_q_heads, q_seq_len, kv_seq_len]
    int mask_ndim = mask.ndim();
    // kv_seq stride is always the last dim stride
    mask_kv_seq_stride = (mask_ndim >= 1 && mask.shape(mask_ndim - 1) > 1) ? mask.strides(mask_ndim - 1) : 0;
    // q_seq stride is second to last
    mask_q_seq_stride = (mask_ndim >= 2 && mask.shape(mask_ndim - 2) > 1) ? mask.strides(mask_ndim - 2) : 0;
    // head stride is third to last
    mask_head_stride = (mask_ndim >= 3 && mask.shape(mask_ndim - 3) > 1) ? mask.strides(mask_ndim - 3) : 0;
    // batch stride is computed from head stride * n_heads (if batch dim > 1)
    // For simplicity, we compute offset in the kernel using head_stride directly
    mask_data_offset = mask.offset() / mask.itemsize();
  }

  if (std::getenv("MLX_OPENCL_DEBUG")) {
    std::cerr << "[SDPA] q.shape: " << q.shape(0) << "," << q.shape(1) << "," << q.shape(2) << "," << q.shape(3)
              << " strides: " << q.strides(0) << "," << q.strides(1) << "," << q.strides(2) << "," << q.strides(3)
              << " head_stride=" << q_head_stride << " head_dim=" << head_dim
              << " q_data_offset=" << q_data_offset
              << " k_data_offset=" << k_data_offset
              << " v_data_offset=" << v_data_offset
              << " has_float_mask=" << has_float_mask << std::endl;
  }

  // Allocate output
  out.set_data(opencl::allocator().malloc(out.nbytes()));

  // Build kernel - use parallel kernel with subgroup operations
  std::string type_name = opencl::type_to_name(q.dtype());
  std::string type_suffix = opencl::type_to_suffix(q.dtype());

  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();

  // Use parallel kernel with 64 threads per head (subgroup-based reduction)
  // Kernel naming convention: {op}_{variant}_{type}_{params}
  std::string kernel_name = "sdpa_parallel_" + type_suffix + "_" + std::to_string(head_dim);
  kernel_name += has_sinks_ ? "_sinks" : "_nosinks";
  kernel_name += do_causal_ ? "_causal" : "_nocausal";
  kernel_name += has_float_mask ? "_fmask" : "_nomask";
  kernel_source << gen_sdpa_parallel_kernel(kernel_name, q.dtype(), head_dim, has_sinks_, do_causal_, has_float_mask);

  std::string source_str = kernel_source.str();
  cl_kernel kernel = dev.get_kernel(kernel_name, source_str, "-cl-std=CL2.0");

  auto& encoder = dev.get_command_encoder(s.index);
  encoder.set_kernel(kernel);
  encoder.set_input_array(q, 0);
  encoder.set_input_array(k, 1);
  encoder.set_input_array(v, 2);
  encoder.set_output_array(out, 3);
  encoder.set_bytes(gqa_factor, 4);
  encoder.set_bytes(kv_seq_len, 5);
  encoder.set_bytes(static_cast<int64_t>(k_head_stride), 6);
  encoder.set_bytes(static_cast<int64_t>(k_seq_stride), 7);
  encoder.set_bytes(static_cast<int64_t>(v_head_stride), 8);
  encoder.set_bytes(static_cast<int64_t>(v_seq_stride), 9);
  encoder.set_bytes(scale_, 10);
  encoder.set_bytes(n_q_heads, 11);
  encoder.set_bytes(static_cast<int64_t>(q_batch_stride), 12);
  encoder.set_bytes(static_cast<int64_t>(q_head_stride), 13);
  encoder.set_bytes(static_cast<int64_t>(o_batch_stride), 14);
  encoder.set_bytes(q_data_offset, 15);
  encoder.set_bytes(k_data_offset, 16);
  encoder.set_bytes(v_data_offset, 17);
  encoder.set_bytes(q_seq_len, 18);
  encoder.set_bytes(do_causal_ ? 1 : 0, 19);

  int next_arg = 20;

  // Pass sinks array if present
  if (has_sinks_) {
    // Sinks is input[3] when has_sinks_ is true
    encoder.set_input_array(inputs[3], next_arg++);
  }

  // Pass mask array and strides if present
  if (has_float_mask) {
    encoder.set_input_array(*mask_ptr, next_arg++);
    encoder.set_bytes(mask_head_stride, next_arg++);
    encoder.set_bytes(mask_q_seq_stride, next_arg++);
    encoder.set_bytes(mask_kv_seq_stride, next_arg++);
    encoder.set_bytes(mask_data_offset, next_arg++);
  }

  // Launch 64 threads (1 workgroup) per query head
  size_t total_heads = B * n_q_heads * q_seq_len;
  size_t local_size[3] = {64, 0, 0};
  size_t global_size[3] = {total_heads * 64, 0, 0};
  encoder.dispatch_threads(global_size, local_size, 1);

  dev.end_encoding(s.index);
}

bool ScaledDotProductAttentionVJP::use_fallback(const array&, Stream) {
  return true;  // Always use CPU until OpenCL flash attention is implemented
}

void ScaledDotProductAttentionVJP::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw std::runtime_error("[ScaledDotProductAttentionVJP::eval_gpu] OpenCL implementation not yet available");
}

// TODO: Implement RMSNormVJP for OpenCL
void RMSNormVJP::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw std::runtime_error("[RMSNormVJP::eval_gpu] OpenCL implementation not yet available");
}

// TODO: Implement LayerNormVJP for OpenCL
void LayerNormVJP::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw std::runtime_error("[LayerNormVJP::eval_gpu] OpenCL implementation not yet available");
}

// TODO: Implement CustomKernel for OpenCL
void CustomKernel::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw std::runtime_error("[CustomKernel::eval_gpu] OpenCL custom kernels not yet supported");
}

} // namespace mlx::core::fast
