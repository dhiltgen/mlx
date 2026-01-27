// Copyright © 2025 MLX Contributors
// OpenCL Scan (cumsum, cumprod, etc.) implementation

#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/primitives.h"

#include <cassert>
#include <sstream>

namespace mlx::core {

namespace {

std::string get_scan_op_code(Scan::ReduceType reduce_type, const std::string& type_name) {
  std::ostringstream src;

  bool is_float_type = (type_name == "float" || type_name == "half" || type_name == "bfloat16_t");

  switch (reduce_type) {
    case Scan::Sum:
      src << "#define SCAN_OP(a, b) ((a) + (b))\n";
      src << "#define SCAN_INIT ((" << type_name << ")0)\n";
      break;
    case Scan::Prod:
      src << "#define SCAN_OP(a, b) ((a) * (b))\n";
      src << "#define SCAN_INIT ((" << type_name << ")1)\n";
      break;
    case Scan::Max:
      if (is_float_type) {
        src << "#define SCAN_OP(a, b) fmax((a), (b))\n";
        src << "#define SCAN_INIT (-INFINITY)\n";
      } else {
        // For integer types, use conditional
        src << "#define SCAN_OP(a, b) ((a) > (b) ? (a) : (b))\n";
        if (type_name == "int32_t") {
          src << "#define SCAN_INIT INT_MIN\n";
        } else if (type_name == "int64_t") {
          src << "#define SCAN_INIT LONG_MIN\n";
        } else if (type_name == "uint32_t" || type_name == "uint64_t") {
          src << "#define SCAN_INIT ((" << type_name << ")0)\n";
        } else {
          src << "#define SCAN_INIT ((" << type_name << ")0)\n";
        }
      }
      break;
    case Scan::Min:
      if (is_float_type) {
        src << "#define SCAN_OP(a, b) fmin((a), (b))\n";
        src << "#define SCAN_INIT INFINITY\n";
      } else {
        src << "#define SCAN_OP(a, b) ((a) < (b) ? (a) : (b))\n";
        if (type_name == "int32_t") {
          src << "#define SCAN_INIT INT_MAX\n";
        } else if (type_name == "int64_t") {
          src << "#define SCAN_INIT LONG_MAX\n";
        } else if (type_name == "uint32_t") {
          src << "#define SCAN_INIT UINT_MAX\n";
        } else if (type_name == "uint64_t") {
          src << "#define SCAN_INIT ULONG_MAX\n";
        } else {
          src << "#define SCAN_INIT ((" << type_name << ")255)\n";
        }
      }
      break;
    case Scan::LogAddExp:
      // logaddexp(a, b) = max(a,b) + log1p(exp(-abs(a-b)))
      src << "#define SCAN_OP(a, b) (fmax((a), (b)) + log1p(exp(-fabs((a) - (b)))))\n";
      src << "#define SCAN_INIT (-INFINITY)\n";
      break;
  }

  return src.str();
}

std::string get_scan_kernel_source(
    const std::string& type_name,
    const std::string& acc_type,
    Scan::ReduceType reduce_type,
    bool reverse,
    bool inclusive) {
  std::ostringstream src;

  src << opencl::get_kernel_preamble();
  src << "\n";
  src << get_scan_op_code(reduce_type, acc_type);
  src << "\n";

  std::string kernel_name = "scan_" + type_name;
  if (reverse) kernel_name += "_reverse";
  if (!inclusive) kernel_name += "_exclusive";

  // Determine read/write expressions for bfloat16/half conversion
  bool is_bfloat16 = (type_name == "bfloat16_t");
  bool is_half = (type_name == "half");

  std::string read_val;
  std::string write_val;
  if (is_bfloat16) {
    read_val = "bfloat16_to_float(in[in_idx])";
    write_val = "float_to_bfloat16(acc)";
  } else if (is_half) {
    read_val = "convert_float(in[in_idx])";
    write_val = "convert_half(acc)";
  } else {
    read_val = "in[in_idx]";
    write_val = "acc";
  }

  // Simple scan kernel: each work item handles one row (sequence along scan axis)
  // The row index encodes positions in all dimensions except the scan axis.
  // We decompose row_idx into pre-axis and post-axis components to compute the base offset.
  src << "__kernel void " << kernel_name << "(\n";
  src << "    __global const " << type_name << "* in,\n";
  src << "    __global " << type_name << "* out,\n";
  src << "    int axis_size,\n";
  src << "    long axis_stride,\n";
  src << "    int post_axis_size,\n";
  src << "    long in_offset,\n";
  src << "    int num_rows) {\n";
  src << "  int row_idx = get_global_id(0);\n";
  src << "  if (row_idx >= num_rows) return;\n";
  src << "\n";
  src << "  // Decompose row_idx into pre_axis_idx and post_axis_idx\n";
  src << "  int pre_axis_idx = row_idx / post_axis_size;\n";
  src << "  int post_axis_idx = row_idx % post_axis_size;\n";
  src << "  // Base offset for output (no offset, freshly allocated)\n";
  src << "  // pre_axis elements contribute (axis_size * axis_stride) each\n";
  src << "  // post_axis elements contribute 1 each (for contiguous array)\n";
  src << "  long out_base = (long)pre_axis_idx * ((long)axis_size * axis_stride) + post_axis_idx;\n";
  src << "  // Base offset for input includes array's data offset for sliced arrays\n";
  src << "  long in_base = in_offset + out_base;\n";
  src << "\n";
  src << "  " << acc_type << " acc = SCAN_INIT;\n";
  src << "\n";

  if (reverse) {
    // Reverse scan: iterate from end to beginning
    if (inclusive) {
      src << "  for (int i = axis_size - 1; i >= 0; i--) {\n";
      src << "    long in_idx = in_base + (long)i * axis_stride;\n";
      src << "    long out_idx = out_base + (long)i * axis_stride;\n";
      src << "    " << acc_type << " val = " << read_val << ";\n";
      src << "    acc = SCAN_OP(acc, val);\n";
      src << "    out[out_idx] = " << write_val << ";\n";
      src << "  }\n";
    } else {
      // Exclusive reverse: output[i] = scan of elements after i
      src << "  for (int i = axis_size - 1; i >= 0; i--) {\n";
      src << "    long in_idx = in_base + (long)i * axis_stride;\n";
      src << "    long out_idx = out_base + (long)i * axis_stride;\n";
      src << "    " << acc_type << " val = " << read_val << ";\n";
      src << "    out[out_idx] = " << write_val << ";\n";
      src << "    acc = SCAN_OP(acc, val);\n";
      src << "  }\n";
    }
  } else {
    // Forward scan
    if (inclusive) {
      src << "  for (int i = 0; i < axis_size; i++) {\n";
      src << "    long in_idx = in_base + (long)i * axis_stride;\n";
      src << "    long out_idx = out_base + (long)i * axis_stride;\n";
      src << "    " << acc_type << " val = " << read_val << ";\n";
      src << "    acc = SCAN_OP(acc, val);\n";
      src << "    out[out_idx] = " << write_val << ";\n";
      src << "  }\n";
    } else {
      // Exclusive: output[i] = scan of elements before i
      src << "  for (int i = 0; i < axis_size; i++) {\n";
      src << "    long in_idx = in_base + (long)i * axis_stride;\n";
      src << "    long out_idx = out_base + (long)i * axis_stride;\n";
      src << "    " << acc_type << " val = " << read_val << ";\n";
      src << "    out[out_idx] = " << write_val << ";\n";
      src << "    acc = SCAN_OP(acc, val);\n";
      src << "  }\n";
    }
  }

  src << "}\n";
  src << "#undef SCAN_OP\n";
  src << "#undef SCAN_INIT\n";

  return src.str();
}

} // namespace

void Scan::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  auto in = inputs[0];
  auto& s = stream();

  // Make input contiguous if needed
  if (in.flags().contiguous && in.strides()[axis_] != 0) {
    if (in.is_donatable() && in.itemsize() == out.itemsize()) {
      out.copy_shared_buffer(in);
    } else {
      out.set_data(
          allocator::malloc(in.data_size() * out.itemsize()),
          in.data_size(),
          in.strides(),
          in.flags());
    }
  } else {
    in = contiguous_copy_gpu(in, s);
    out.copy_shared_buffer(in);
  }

  auto& dev = opencl::device(s.device);

  // Get type names
  std::string type_name = opencl::type_to_name(in.dtype());

  // Use float accumulator for half/bfloat16
  std::string acc_type = type_name;
  if (type_name == "half" || type_name == "bfloat16_t") {
    acc_type = "float";
  }

  // Build kernel
  std::string kernel_name = "scan_" + type_name;
  if (reverse_) kernel_name += "_reverse";
  if (!inclusive_) kernel_name += "_exclusive";

  std::string source = get_scan_kernel_source(
      type_name, acc_type, reduce_type_, reverse_, inclusive_);

  cl_kernel kernel = dev.get_kernel(kernel_name, source);
  auto& encoder = dev.get_command_encoder(s.index);
  encoder.set_kernel(kernel);

  // Compute grid dimensions
  int32_t axis_size = in.shape(axis_);
  int64_t axis_stride = in.strides()[axis_];

  // For contiguous data, compute number of independent scan rows
  // num_rows = product of all dimensions except the scan axis
  size_t num_rows = in.size() / axis_size;

  // Compute post_axis_size = product of dimensions after scan axis
  int post_axis_size = 1;
  for (int i = axis_ + 1; i < in.ndim(); i++) {
    post_axis_size *= in.shape(i);
  }

  // Set kernel arguments
  encoder.set_input_array(in, 0);
  encoder.set_output_array(out, 1);
  encoder.set_bytes(axis_size, 2);
  encoder.set_bytes(axis_stride, 3);
  encoder.set_bytes(post_axis_size, 4);
  int64_t in_offset = static_cast<int64_t>(in.offset()) / in.itemsize();
  encoder.set_bytes(in_offset, 5);
  encoder.set_bytes(static_cast<int>(num_rows), 6);

  // Dispatch - one work item per row
  size_t global_size[3] = {num_rows, 0, 0};
  encoder.dispatch_threads(global_size, nullptr, 1);

  dev.end_encoding(s.index);
}

} // namespace mlx::core
