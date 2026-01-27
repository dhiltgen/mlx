// Copyright © 2025 MLX Contributors
// OpenCL Convolution implementation

#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/copy.h"
#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/opencl/types.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/primitives.h"

#include <sstream>

namespace mlx::core {

namespace {

// Generate naive convolution kernel for 1D, 2D, or 3D
std::string generate_conv_kernel(
    Dtype dtype,
    int ndim,
    bool flip) {
  std::ostringstream kernel;
  kernel << opencl::get_kernel_preamble();

  // Use centralized type utilities for type handling
  std::string type_name = opencl::type_to_name(dtype);
  bool needs_conv = opencl::needs_float_conversion(dtype);
  std::string compute_type = needs_conv ? "float" : type_name;

  kernel << "\n__kernel void conv" << ndim << "d_" << type_name << "(\n";
  kernel << "    __global const " << type_name << "* input,\n";
  kernel << "    __global const " << type_name << "* weight,\n";
  kernel << "    __global " << type_name << "* output,\n";
  kernel << "    int N,\n";  // batch size
  kernel << "    int C,\n";  // input channels
  kernel << "    int O,\n";  // output channels
  kernel << "    int groups,\n";

  // Spatial dimensions
  if (ndim >= 1) {
    kernel << "    int iH, int oH, int wH, int strH, int padH, int kdilH, int idilH,\n";
  }
  if (ndim >= 2) {
    kernel << "    int iW, int oW, int wW, int strW, int padW, int kdilW, int idilW,\n";
  }
  if (ndim >= 3) {
    kernel << "    int iD, int oD, int wD, int strD, int padD, int kdilD, int idilD,\n";
  }

  // Strides for input, weight, output
  kernel << "    long in_stride_N, long in_stride_C,\n";
  if (ndim >= 1) kernel << "    long in_stride_H,\n";
  if (ndim >= 2) kernel << "    long in_stride_W,\n";
  if (ndim >= 3) kernel << "    long in_stride_D,\n";

  kernel << "    long wt_stride_O, long wt_stride_C,\n";
  if (ndim >= 1) kernel << "    long wt_stride_H,\n";
  if (ndim >= 2) kernel << "    long wt_stride_W,\n";
  if (ndim >= 3) kernel << "    long wt_stride_D,\n";

  kernel << "    long out_stride_N, long out_stride_O";
  if (ndim >= 1) kernel << ", long out_stride_H";
  if (ndim >= 2) kernel << ", long out_stride_W";
  if (ndim >= 3) kernel << ", long out_stride_D";
  kernel << ") {\n";

  // Each work item computes one output element
  kernel << "  int gid = get_global_id(0);\n";

  // Decode global id into output indices
  // Output layout: (N, spatial..., O)
  kernel << "  int remaining = gid;\n";
  kernel << "  int o = remaining % O; remaining /= O;\n";

  if (ndim >= 3) {
    kernel << "  int od = remaining % oD; remaining /= oD;\n";
  }
  if (ndim >= 2) {
    kernel << "  int ow = remaining % oW; remaining /= oW;\n";
  }
  if (ndim >= 1) {
    kernel << "  int oh = remaining % oH; remaining /= oH;\n";
  }
  kernel << "  int n = remaining;\n";

  // Bounds check
  kernel << "  if (n >= N) return;\n\n";

  // Compute group information
  kernel << "  int C_per_group = C / groups;\n";
  kernel << "  int O_per_group = O / groups;\n";
  kernel << "  int g = o / O_per_group;\n";
  kernel << "  int c_start = g * C_per_group;\n";
  kernel << "  int c_end = c_start + C_per_group;\n\n";

  // Accumulator
  kernel << "  " << compute_type << " acc = 0;\n\n";

  // Convolution loops over kernel spatial dimensions
  if (ndim >= 1) {
    kernel << "  for (int wh = 0; wh < wH; ++wh) {\n";
    if (flip) {
      kernel << "    int wh_idx = wH - wh - 1;\n";
    } else {
      kernel << "    int wh_idx = wh;\n";
    }
    kernel << "    int ih = oh * strH - padH + wh_idx * kdilH;\n";
    kernel << "    if (ih < 0 || ih >= iH) continue;\n";
    kernel << "    if (idilH > 1 && (ih % idilH) != 0) continue;\n";
    kernel << "    int ih_actual = ih / idilH;\n";
  }

  if (ndim >= 2) {
    kernel << "    for (int ww = 0; ww < wW; ++ww) {\n";
    if (flip) {
      kernel << "      int ww_idx = wW - ww - 1;\n";
    } else {
      kernel << "      int ww_idx = ww;\n";
    }
    kernel << "      int iw = ow * strW - padW + ww_idx * kdilW;\n";
    kernel << "      if (iw < 0 || iw >= iW) continue;\n";
    kernel << "      if (idilW > 1 && (iw % idilW) != 0) continue;\n";
    kernel << "      int iw_actual = iw / idilW;\n";
  }

  if (ndim >= 3) {
    kernel << "      for (int wd = 0; wd < wD; ++wd) {\n";
    if (flip) {
      kernel << "        int wd_idx = wD - wd - 1;\n";
    } else {
      kernel << "        int wd_idx = wd;\n";
    }
    kernel << "        int id = od * strD - padD + wd_idx * kdilD;\n";
    kernel << "        if (id < 0 || id >= iD) continue;\n";
    kernel << "        if (idilD > 1 && (id % idilD) != 0) continue;\n";
    kernel << "        int id_actual = id / idilD;\n";
  }

  // Loop over channels
  std::string indent = ndim == 1 ? "    " : (ndim == 2 ? "      " : "        ");
  kernel << indent << "for (int c = c_start; c < c_end; ++c) {\n";

  // Compute input and weight indices
  kernel << indent << "  long in_idx = n * in_stride_N + c * in_stride_C";
  if (ndim >= 1) kernel << " + ih_actual * in_stride_H";
  if (ndim >= 2) kernel << " + iw_actual * in_stride_W";
  if (ndim >= 3) kernel << " + id_actual * in_stride_D";
  kernel << ";\n";

  kernel << indent << "  long wt_idx = o * wt_stride_O + (c - c_start) * wt_stride_C";
  if (ndim >= 1) kernel << " + wh * wt_stride_H";
  if (ndim >= 2) kernel << " + ww * wt_stride_W";
  if (ndim >= 3) kernel << " + wd * wt_stride_D";
  kernel << ";\n";

  // Accumulate - use centralized type conversion utilities
  kernel << indent << "  acc += " << opencl::make_read_expr(dtype, "input[in_idx]")
         << " * " << opencl::make_read_expr(dtype, "weight[wt_idx]") << ";\n";

  kernel << indent << "}\n";  // end channel loop

  // Close spatial loops
  if (ndim >= 3) kernel << "      }\n";
  if (ndim >= 2) kernel << "    }\n";
  if (ndim >= 1) kernel << "  }\n";

  // Write output
  kernel << "\n  long out_idx = n * out_stride_N + o * out_stride_O";
  if (ndim >= 1) kernel << " + oh * out_stride_H";
  if (ndim >= 2) kernel << " + ow * out_stride_W";
  if (ndim >= 3) kernel << " + od * out_stride_D";
  kernel << ";\n";

  // Write output with type conversion if needed
  kernel << "  output[out_idx] = " << opencl::make_write_expr(dtype, "acc") << ";\n";

  kernel << "}\n";

  return kernel.str();
}

void conv_gpu_impl(
    const array& in,
    const array& wt,
    array& out,
    const std::vector<int>& kernel_strides,
    const std::vector<int>& padding_lo,
    const std::vector<int>& padding_hi,
    const std::vector<int>& kernel_dilation,
    const std::vector<int>& input_dilation,
    int groups,
    bool flip,
    const Stream& s) {

  int ndim = kernel_strides.size();  // 1, 2, or 3

  OPENCL_DEBUG_LOG("[conv_gpu] ndim=" << ndim << " groups=" << groups << " flip=" << flip);
  OPENCL_DEBUG_LOG("[conv_gpu] in.shape: N=" << in.shape(0));
  for (int i = 1; i <= ndim; i++) {
    OPENCL_DEBUG_LOG("  spatial[" << i-1 << "]=" << in.shape(i));
  }
  OPENCL_DEBUG_LOG("  C=" << in.shape(ndim + 1));

  auto& dev = opencl::device(s.device);

  std::string type_name = opencl::type_to_name(in.dtype());
  std::string kernel_name = "conv" + std::to_string(ndim) + "d_" + type_name;

  // Generate and compile kernel using centralized type utilities
  std::string source = generate_conv_kernel(in.dtype(), ndim, flip);
  cl_kernel kernel = dev.get_kernel(kernel_name, source);

  // Allocate output
  out.set_data(opencl::allocator().malloc(out.nbytes()));

  auto& encoder = dev.get_command_encoder(s.index);
  encoder.set_kernel(kernel);

  // Set kernel arguments
  int arg = 0;
  encoder.set_input_array(in, arg++);
  encoder.set_input_array(wt, arg++);
  encoder.set_output_array(out, arg++);

  // Batch and channel info
  int N = in.shape(0);
  int C = in.shape(ndim + 1);  // last dim
  int O = wt.shape(0);  // first dim of weights

  encoder.set_bytes(N, arg++);
  encoder.set_bytes(C, arg++);
  encoder.set_bytes(O, arg++);
  encoder.set_bytes(groups, arg++);

  // For each spatial dimension
  // Input spatial dims (with dilation applied)
  // Weight spatial dims
  // Output spatial dims

  for (int d = 0; d < ndim; d++) {
    int in_spatial = in.shape(d + 1);
    int wt_spatial = wt.shape(d + 1);
    int out_spatial = out.shape(d + 1);

    // Note: iH in kernel represents dilated input size
    int dilated_in = 1 + input_dilation[d] * (in_spatial - 1);

    encoder.set_bytes(dilated_in, arg++);     // iH/iW/iD (dilated)
    encoder.set_bytes(out_spatial, arg++);    // oH/oW/oD
    encoder.set_bytes(wt_spatial, arg++);     // wH/wW/wD
    encoder.set_bytes(kernel_strides[d], arg++);   // stride
    encoder.set_bytes(padding_lo[d], arg++);       // padding
    encoder.set_bytes(kernel_dilation[d], arg++);  // kernel dilation
    encoder.set_bytes(input_dilation[d], arg++);   // input dilation
  }

  // Input strides (layout: N, spatial..., C)
  encoder.set_bytes(static_cast<int64_t>(in.strides()[0]), arg++);  // N stride
  encoder.set_bytes(static_cast<int64_t>(in.strides()[ndim + 1]), arg++);  // C stride
  for (int d = 0; d < ndim; d++) {
    encoder.set_bytes(static_cast<int64_t>(in.strides()[d + 1]), arg++);
  }

  // Weight strides (layout: O, spatial..., C_per_group)
  encoder.set_bytes(static_cast<int64_t>(wt.strides()[0]), arg++);  // O stride
  encoder.set_bytes(static_cast<int64_t>(wt.strides()[ndim + 1]), arg++);  // C stride
  for (int d = 0; d < ndim; d++) {
    encoder.set_bytes(static_cast<int64_t>(wt.strides()[d + 1]), arg++);
  }

  // Output strides (layout: N, spatial..., O)
  encoder.set_bytes(static_cast<int64_t>(out.strides()[0]), arg++);  // N stride
  encoder.set_bytes(static_cast<int64_t>(out.strides()[ndim + 1]), arg++);  // O stride
  for (int d = 0; d < ndim; d++) {
    encoder.set_bytes(static_cast<int64_t>(out.strides()[d + 1]), arg++);
  }

  // Dispatch - one thread per output element
  size_t total_output = out.size();
  size_t global_size[3] = {total_output, 1, 1};
  encoder.dispatch_threads(global_size, nullptr, 1);

  dev.end_encoding(s.index);
}

}  // namespace

void Convolution::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];
  auto& wt = inputs[1];

  // Make inputs contiguous if needed
  auto check_input = [this](const array& x) {
    if (!x.flags().row_contiguous) {
      array x_copy(x.shape(), x.dtype(), nullptr, {});
      copy_gpu(x, x_copy, CopyType::General, stream());
      return x_copy;
    }
    return x;
  };

  auto in_contig = check_input(in);
  auto wt_contig = check_input(wt);

  conv_gpu_impl(
      in_contig,
      wt_contig,
      out,
      kernel_strides_,
      padding_lo_,
      padding_hi_,
      kernel_dilation_,
      input_dilation_,
      groups_,
      flip_,
      stream());
}

}  // namespace mlx::core
