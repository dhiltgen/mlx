// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/reduce.h"
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/primitives.h"

#include <sstream>

namespace mlx::core {

namespace {

// Compute row-major (C-contiguous) strides for a given shape
Strides compute_row_major_strides(const Shape& shape) {
  Strides strides(shape.size());
  if (shape.empty()) {
    return strides;
  }
  int64_t stride = 1;
  for (int i = shape.size() - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

// Get the reduction operation string for OpenCL
std::string get_reduce_op(Reduce::ReduceType reduce_type) {
  switch (reduce_type) {
    case Reduce::And:
      return "reduce_and";
    case Reduce::Or:
      return "reduce_or";
    case Reduce::Sum:
      return "reduce_sum";
    case Reduce::Prod:
      return "reduce_prod";
    case Reduce::Min:
      return "reduce_min";
    case Reduce::Max:
      return "reduce_max";
    default:
      throw std::runtime_error("[reduce] Unknown reduction type");
  }
}

// Get the initial value for the reduction
std::string get_reduce_init(Reduce::ReduceType reduce_type, const std::string& type_name) {
  bool is_float = (type_name.find("float") != std::string::npos || type_name == "half");
  bool is_unsigned = (type_name.find("uint") != std::string::npos);

  switch (reduce_type) {
    case Reduce::And:
      return "1";  // Use 1 for true in OpenCL
    case Reduce::Or:
      return "0";  // Use 0 for false in OpenCL
    case Reduce::Sum:
      return "0";
    case Reduce::Prod:
      return "1";
    case Reduce::Min:
      if (is_float) {
        return "INFINITY";
      }
      if (is_unsigned) {
        if (type_name == "uint8_t") return "255";
        if (type_name == "uint16_t") return "65535";
        if (type_name == "uint32_t") return "4294967295U";
        if (type_name == "uint64_t") return "18446744073709551615UL";
      }
      // Signed types need type-specific max values to avoid overflow
      if (type_name == "int8_t" || type_name == "char") return "127";
      if (type_name == "int16_t" || type_name == "short") return "32767";
      if (type_name == "int64_t" || type_name == "long") return "9223372036854775807L";
      return "INT_MAX";  // int32_t / int
    case Reduce::Max:
      if (is_float) {
        return "-INFINITY";
      }
      if (is_unsigned) {
        return "0";  // Min value for unsigned types
      }
      // Signed types need type-specific min values to avoid overflow
      if (type_name == "int8_t" || type_name == "char") return "-128";
      if (type_name == "int16_t" || type_name == "short") return "-32768";
      if (type_name == "int64_t" || type_name == "long") return "(-9223372036854775807L - 1)";
      return "INT_MIN";  // int32_t / int
    default:
      return "0";
  }
}

// Get the reduction operator for OpenCL
std::string get_reduce_operator(Reduce::ReduceType reduce_type, const std::string& type_name) {
  bool is_float = (type_name.find("float") != std::string::npos || type_name == "half");

  switch (reduce_type) {
    case Reduce::And:
      // For floats, use logical AND; for integers, use bitwise AND
      return is_float ? "&&" : "&";
    case Reduce::Or:
      // For floats, use logical OR; for integers, use bitwise OR
      return is_float ? "||" : "|";
    case Reduce::Sum:
      return "+";
    case Reduce::Prod:
      return "*";
    case Reduce::Min:
      return "min";
    case Reduce::Max:
      return "max";
    default:
      return "+";
  }
}

// Legacy overload for compatibility
std::string get_reduce_operator(Reduce::ReduceType reduce_type) {
  return get_reduce_operator(reduce_type, "int");
}

} // namespace

void Reduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const array& in = inputs[0];

  OPENCL_DEBUG_LOG("[Reduce::eval_gpu] reduce_type=" << reduce_type_
            << " axes.size=" << axes_.size() << " in.size=" << in.size());

  // Multi-axis reduction: reduce one axis at a time
  if (axes_.size() > 1) {
    OPENCL_DEBUG_LOG("[Reduce::eval_gpu] Multi-axis reduction: " << axes_.size() << " axes");

    // Sort axes in descending order to avoid shifting issues
    std::vector<int> sorted_axes = axes_;
    std::sort(sorted_axes.begin(), sorted_axes.end(), std::greater<int>());

    // Start with the input array
    array current = in;

    // Reduce one axis at a time
    for (size_t i = 0; i < sorted_axes.size(); ++i) {
      int axis = sorted_axes[i];
      OPENCL_DEBUG_LOG("[Reduce::eval_gpu] Reducing axis " << axis << " (iteration " << (i+1)
                << " of " << sorted_axes.size() << ")");

      // Calculate output shape for this reduction (keepdims=True semantics)
      // Keep the reduced dimension but set its size to 1
      Shape intermediate_shape;
      for (int d = 0; d < current.ndim(); ++d) {
        if (d == axis) {
          intermediate_shape.push_back(1);  // Keep dimension but size 1
        } else {
          intermediate_shape.push_back(current.shape(d));
        }
      }

      // Create intermediate output array
      // Use out.dtype() for all intermediates since it's the correct output type
      // (e.g., bool -> uint32 for Sum, bool -> bool for And/Or, etc.)
      array intermediate(intermediate_shape, out.dtype(), nullptr, {});

      // Create single-axis reduction primitive
      std::vector<int> single_axis = {axis};
      auto single_reduce = Reduce(stream(), reduce_type_, single_axis);

      // Evaluate the single-axis reduction
      std::vector<array> single_input = {current};
      single_reduce.eval_gpu(single_input, intermediate);

      // Use this intermediate result as input for next iteration
      current = intermediate;
    }

    // Copy final result to output
    OPENCL_DEBUG_LOG("[Reduce::eval_gpu] Multi-axis final: current.shape=["
              << (current.ndim() > 0 ? std::to_string(current.shape(0)) : "")
              << (current.ndim() > 1 ? "," + std::to_string(current.shape(1)) : "")
              << (current.ndim() > 2 ? "," + std::to_string(current.shape(2)) : "")
              << "] out.shape=["
              << (out.ndim() > 0 ? std::to_string(out.shape(0)) : "")
              << (out.ndim() > 1 ? "," + std::to_string(out.shape(1)) : "")
              << (out.ndim() > 2 ? "," + std::to_string(out.shape(2)) : "")
              << "] current.strides.size=" << current.strides().size()
              << " out.ndim=" << out.ndim());
    out.copy_shared_buffer(current);
    OPENCL_DEBUG_LOG("[Reduce::eval_gpu] Multi-axis reduction complete");
    return;
  }

  // Debug: log input array properties
  OPENCL_DEBUG_LOG("[Reduce::eval_gpu] in.ndim=" << in.ndim()
            << " in.flags().contiguous=" << in.flags().contiguous
            << " in.flags().row_contiguous=" << in.flags().row_contiguous
            << " in.data_size=" << in.data_size());
  if (in.ndim() > 0) {
    std::ostringstream strides_str;
    strides_str << "[";
    for (int i = 0; i < in.ndim(); i++) {
      if (i > 0) strides_str << ", ";
      strides_str << in.strides(i);
    }
    strides_str << "]";
    OPENCL_DEBUG_LOG("[Reduce::eval_gpu] in.strides=" << strides_str.str());
  }

  // Check if input is truly contiguous (row-major with no broadcast)
  bool is_truly_contiguous = in.flags().row_contiguous &&
                             (in.data_size() == in.size());

  // Allocate output and set up row-major strides
  auto strides = compute_row_major_strides(out.shape());
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
  std::string in_type_name = opencl::type_to_name(in.dtype());
  std::string out_type_name = opencl::type_to_name(out.dtype());
  std::string op_name = get_reduce_op(reduce_type_);

  // For And/Or reductions, input and output types may differ (e.g., float input -> bool output)
  bool is_logical = (reduce_type_ == Reduce::And || reduce_type_ == Reduce::Or);

  // Check if input is floating point (for NaN handling in min/max)
  bool is_float_type = issubdtype(in.dtype(), floating);

  // Check if type needs float conversion (bfloat16/float16)
  bool needs_conv = opencl::needs_float_conversion(in.dtype());

  // For full reduction (reduce all elements to a single value)
  if (axes_.size() == 0 || (axes_.size() == 1 && in.ndim() == 1)) {
    OPENCL_DEBUG_LOG("[Reduce::eval_gpu] Full reduction to scalar, in.size()=" << in.size()
              << " contiguous=" << is_truly_contiguous);

    // For logical ops on non-bool inputs, use int for accumulator to avoid float issues
    // For bfloat16/half, use float for accumulator (better precision and required for bf16)
    std::string accum_type = is_logical ? "int" : needs_conv ? "float" : in_type_name;
    std::string accum_init = is_logical ? (reduce_type_ == Reduce::And ? "1" : "0")
                                        : get_reduce_init(reduce_type_, needs_conv ? "float" : in_type_name);
    std::string op_str = get_reduce_operator(reduce_type_, in_type_name);
    bool is_minmax = (reduce_type_ == Reduce::Min || reduce_type_ == Reduce::Max);

    // Compute data offset for sliced arrays (offset is in bytes, convert to elements)
    int64_t in_data_offset = in.offset() / in.itemsize();

    if (is_truly_contiguous) {
      // Contiguous: use simple linear iteration
      std::string kernel_name = "reduce_full_" + op_name + "_" + in_type_name + "_" + out_type_name + (is_minmax && is_float_type ? "_nan" : "") + (needs_conv ? "_conv" : "");
      std::ostringstream kernel_source;
      kernel_source << opencl::get_kernel_preamble();
      kernel_source << "\n__kernel void " << kernel_name << "(\n";
      kernel_source << "    __global const " << in_type_name << "* input,\n";
      kernel_source << "    __global " << out_type_name << "* output,\n";
      kernel_source << "    int size,\n";
      kernel_source << "    long in_data_offset) {\n";
      kernel_source << "  if (get_global_id(0) != 0) return;\n";
      kernel_source << "  // Apply data offset for sliced arrays\n";
      kernel_source << "  input = input + in_data_offset;\n";
      kernel_source << "  " << accum_type << " result = " << accum_init << ";\n";
      kernel_source << "  for (int i = 0; i < size; i++) {\n";

      // Read input value with conversion if needed
      kernel_source << "    " << accum_type << " val = " << opencl::make_read_expr(in.dtype(), "input[i]") << ";\n";

      if (is_minmax) {
        // For floating point, propagate NaN; for integers, use standard min/max
        if (is_float_type) {
          kernel_source << "    result = isnan(val) ? val : (isnan(result) ? result : " << op_str << "(result, val));\n";
        } else {
          kernel_source << "    result = " << op_str << "(result, val);\n";
        }
      } else if (is_logical) {
        if (reduce_type_ == Reduce::And) {
          kernel_source << "    result = result && (val != 0);\n";
        } else {
          kernel_source << "    result = result || (val != 0);\n";
        }
      } else {
        kernel_source << "    result = result " << op_str << " val;\n";
      }

      kernel_source << "  }\n";

      // Write output with conversion if needed
      if (is_logical || !needs_conv) {
        kernel_source << "  output[0] = (" << out_type_name << ")result;\n";
      } else {
        kernel_source << "  output[0] = " << opencl::make_write_expr(in.dtype(), "result") << ";\n";
      }
      kernel_source << "}\n";

      std::string source_str = kernel_source.str();

      OPENCL_DEBUG_LOG("[Reduce::eval_gpu] Compiling kernel: " << kernel_name);

      cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
      encoder.set_kernel(kernel);
      encoder.set_input_array(in, 0);
      encoder.set_output_array(out, 1);
      encoder.set_bytes(static_cast<int>(in.size()), 2);
      encoder.set_bytes(in_data_offset, 3);

      size_t global_size[3] = {1, 0, 0};
      encoder.dispatch_threads(global_size, nullptr, 1);
    } else {
      // Non-contiguous: use strided iteration with shape/strides buffers
      int ndim = in.ndim();
      std::vector<int> shape_vec(in.shape().begin(), in.shape().end());
      std::vector<int64_t> strides_vec(in.strides().begin(), in.strides().end());

      // Handle scalar case
      if (shape_vec.empty()) {
        shape_vec.push_back(1);
        strides_vec.push_back(0);
        ndim = 1;
      }

      std::string kernel_name = "reduce_full_strided_" + op_name + "_" + in_type_name + "_" + out_type_name + (is_minmax && is_float_type ? "_nan" : "") + (needs_conv ? "_conv" : "");
      std::ostringstream kernel_source;
      kernel_source << opencl::get_kernel_preamble();
      kernel_source << "\n__kernel void " << kernel_name << "(\n";
      kernel_source << "    __global const " << in_type_name << "* input,\n";
      kernel_source << "    __global " << out_type_name << "* output,\n";
      kernel_source << "    __global const int* shape,\n";
      kernel_source << "    __global const long* strides,\n";
      kernel_source << "    int ndim,\n";
      kernel_source << "    int size,\n";
      kernel_source << "    long in_data_offset) {\n";
      kernel_source << "  if (get_global_id(0) != 0) return;\n";
      kernel_source << "  // Apply data offset for sliced arrays\n";
      kernel_source << "  input = input + in_data_offset;\n";
      kernel_source << "  " << accum_type << " result = " << accum_init << ";\n";
      kernel_source << "  // Iterate through all logical positions\n";
      kernel_source << "  for (int linear = 0; linear < size; linear++) {\n";
      kernel_source << "    // Convert linear index to multi-dimensional index and compute strided offset\n";
      kernel_source << "    long offset = 0;\n";
      kernel_source << "    int remaining = linear;\n";
      kernel_source << "    for (int d = ndim - 1; d >= 0; d--) {\n";
      kernel_source << "      int idx = remaining % shape[d];\n";
      kernel_source << "      remaining /= shape[d];\n";
      kernel_source << "      offset += idx * strides[d];\n";
      kernel_source << "    }\n";

      // Read input value with conversion if needed
      kernel_source << "    " << accum_type << " val = " << opencl::make_read_expr(in.dtype(), "input[offset]") << ";\n";

      if (is_minmax) {
        // For floating point, propagate NaN; for integers, use standard min/max
        if (is_float_type) {
          kernel_source << "    result = isnan(val) ? val : (isnan(result) ? result : " << op_str << "(result, val));\n";
        } else {
          kernel_source << "    result = " << op_str << "(result, val);\n";
        }
      } else if (is_logical) {
        if (reduce_type_ == Reduce::And) {
          kernel_source << "    result = result && (val != 0);\n";
        } else {
          kernel_source << "    result = result || (val != 0);\n";
        }
      } else {
        kernel_source << "    result = result " << op_str << " val;\n";
      }

      kernel_source << "  }\n";

      // Write output with conversion if needed
      if (is_logical || !needs_conv) {
        kernel_source << "  output[0] = (" << out_type_name << ")result;\n";
      } else {
        kernel_source << "  output[0] = " << opencl::make_write_expr(in.dtype(), "result") << ";\n";
      }
      kernel_source << "}\n";

      std::string source_str = kernel_source.str();

      OPENCL_DEBUG_LOG("[Reduce::eval_gpu] Compiling kernel: " << kernel_name);

      cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
      encoder.set_kernel(kernel);
      encoder.set_input_array(in, 0);
      encoder.set_output_array(out, 1);

      // Create buffers for shape and strides
      cl_int err;
      cl_mem shape_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         shape_vec.size() * sizeof(int), shape_vec.data(), &err);
      cl_mem strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           strides_vec.size() * sizeof(int64_t), strides_vec.data(), &err);

      clSetKernelArg(kernel, 2, sizeof(cl_mem), &shape_buf);
      clSetKernelArg(kernel, 3, sizeof(cl_mem), &strides_buf);
      encoder.set_bytes(ndim, 4);
      encoder.set_bytes(static_cast<int>(in.size()), 5);
      encoder.set_bytes(in_data_offset, 6);

      size_t global_size[3] = {1, 0, 0};
      encoder.dispatch_threads(global_size, nullptr, 1);

      // Track buffers for cleanup after sync
      dev.add_temp_buffer(shape_buf, s.index);
      dev.add_temp_buffer(strides_buf, s.index);
    }

    dev.add_temporary(in, s.index);
    dev.end_encoding(s.index);

    OPENCL_DEBUG_LOG("[Reduce::eval_gpu] Dispatched full reduction kernel");
  } else {
    // Single axis reduction
    assert(axes_.size() == 1);
    int axis = axes_[0];

    OPENCL_DEBUG_LOG("[Reduce::eval_gpu] Single-axis reduction: axis=" << axis
              << " in.ndim=" << in.ndim() << " out.ndim=" << out.ndim()
              << " contiguous=" << is_truly_contiguous);

    // Calculate dimensions
    int outer_size = 1;  // Product of dimensions before the reduction axis
    int inner_size = 1;  // Product of dimensions after the reduction axis
    int reduce_size = in.shape(axis);  // Size of the reduction axis

    for (int i = 0; i < axis; ++i) {
      outer_size *= in.shape(i);
    }
    for (int i = axis + 1; i < in.ndim(); ++i) {
      inner_size *= in.shape(i);
    }

    OPENCL_DEBUG_LOG("[Reduce::eval_gpu] outer=" << outer_size << " reduce=" << reduce_size
              << " inner=" << inner_size);

    // For logical ops on non-bool inputs, use int for accumulator
    // For bfloat16/half, use float for accumulator (better precision and required for bf16)
    std::string accum_type = is_logical ? "int" : needs_conv ? "float" : in_type_name;
    std::string accum_init = is_logical ? (reduce_type_ == Reduce::And ? "1" : "0")
                                        : get_reduce_init(reduce_type_, needs_conv ? "float" : in_type_name);
    std::string op_str = get_reduce_operator(reduce_type_, in_type_name);
    bool is_minmax = (reduce_type_ == Reduce::Min || reduce_type_ == Reduce::Max);

    // Compute data offset for sliced arrays (offset is in bytes, convert to elements)
    int64_t in_data_offset = in.offset() / in.itemsize();

    if (is_truly_contiguous) {
      // Contiguous: use simple index computation
      std::string kernel_name = "reduce_axis_" + op_name + "_" + in_type_name + "_" + out_type_name + (is_minmax && is_float_type ? "_nan" : "") + (needs_conv ? "_conv" : "");
      std::ostringstream kernel_source;
      kernel_source << opencl::get_kernel_preamble();
      kernel_source << "\n__kernel void " << kernel_name << "(\n";
      kernel_source << "    __global const " << in_type_name << "* input,\n";
      kernel_source << "    __global " << out_type_name << "* output,\n";
      kernel_source << "    int outer_size,\n";
      kernel_source << "    int reduce_size,\n";
      kernel_source << "    int inner_size,\n";
      kernel_source << "    long in_data_offset) {\n";
      kernel_source << "  int outer_idx = get_global_id(0);\n";
      kernel_source << "  int inner_idx = get_global_id(1);\n";
      kernel_source << "  if (outer_idx >= outer_size || inner_idx >= inner_size) return;\n";
      kernel_source << "  // Apply data offset for sliced arrays\n";
      kernel_source << "  input = input + in_data_offset;\n";
      kernel_source << "  " << accum_type << " result = " << accum_init << ";\n";
      kernel_source << "  for (int r = 0; r < reduce_size; r++) {\n";
      kernel_source << "    int input_idx = outer_idx * reduce_size * inner_size + r * inner_size + inner_idx;\n";

      // Read input value with conversion if needed
      kernel_source << "    " << accum_type << " val = " << opencl::make_read_expr(in.dtype(), "input[input_idx]") << ";\n";

      if (is_minmax) {
        // For floating point, propagate NaN; for integers, use standard min/max
        if (is_float_type) {
          kernel_source << "    result = isnan(val) ? val : (isnan(result) ? result : " << op_str << "(result, val));\n";
        } else {
          kernel_source << "    result = " << op_str << "(result, val);\n";
        }
      } else if (is_logical) {
        if (reduce_type_ == Reduce::And) {
          kernel_source << "    result = result && (val != 0);\n";
        } else {
          kernel_source << "    result = result || (val != 0);\n";
        }
      } else {
        kernel_source << "    result = result " << op_str << " val;\n";
      }

      kernel_source << "  }\n";
      kernel_source << "  int output_idx = outer_idx * inner_size + inner_idx;\n";

      // Write output with conversion if needed
      if (is_logical || !needs_conv) {
        kernel_source << "  output[output_idx] = (" << out_type_name << ")result;\n";
      } else {
        kernel_source << "  output[output_idx] = " << opencl::make_write_expr(in.dtype(), "result") << ";\n";
      }
      kernel_source << "}\n";

      std::string source_str = kernel_source.str();

      OPENCL_DEBUG_LOG("[Reduce::eval_gpu] Compiling kernel: " << kernel_name);

      cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
      encoder.set_kernel(kernel);
      encoder.set_input_array(in, 0);
      encoder.set_output_array(out, 1);
      encoder.set_bytes(static_cast<int>(outer_size), 2);
      encoder.set_bytes(static_cast<int>(reduce_size), 3);
      encoder.set_bytes(static_cast<int>(inner_size), 4);
      encoder.set_bytes(in_data_offset, 5);

      size_t global_size[3] = {static_cast<size_t>(outer_size), static_cast<size_t>(inner_size), 0};
      encoder.dispatch_threads(global_size, nullptr, 2);
    } else {
      // Non-contiguous: use strided access with full shape/strides
      int ndim = in.ndim();
      std::vector<int> shape_vec(in.shape().begin(), in.shape().end());
      std::vector<int64_t> strides_vec(in.strides().begin(), in.strides().end());

      std::string kernel_name = "reduce_axis_strided_" + op_name + "_" + in_type_name + "_" + out_type_name + (is_minmax && is_float_type ? "_nan" : "") + (needs_conv ? "_conv" : "");
      std::ostringstream kernel_source;
      kernel_source << opencl::get_kernel_preamble();
      kernel_source << "\n__kernel void " << kernel_name << "(\n";
      kernel_source << "    __global const " << in_type_name << "* input,\n";
      kernel_source << "    __global " << out_type_name << "* output,\n";
      kernel_source << "    __global const int* shape,\n";
      kernel_source << "    __global const long* strides,\n";
      kernel_source << "    int ndim,\n";
      kernel_source << "    int axis,\n";
      kernel_source << "    int outer_size,\n";
      kernel_source << "    int reduce_size,\n";
      kernel_source << "    int inner_size,\n";
      kernel_source << "    long in_data_offset) {\n";
      kernel_source << "  int outer_idx = get_global_id(0);\n";
      kernel_source << "  int inner_idx = get_global_id(1);\n";
      kernel_source << "  if (outer_idx >= outer_size || inner_idx >= inner_size) return;\n";
      kernel_source << "\n";
      kernel_source << "  // Compute multi-dimensional indices for non-axis dimensions\n";
      kernel_source << "  // outer_idx encodes dimensions [0, axis), inner_idx encodes dimensions (axis, ndim)\n";
      kernel_source << "  int indices[16];  // Max 16 dimensions\n";
      kernel_source << "  int temp_outer = outer_idx;\n";
      kernel_source << "  for (int d = axis - 1; d >= 0; d--) {\n";
      kernel_source << "    indices[d] = temp_outer % shape[d];\n";
      kernel_source << "    temp_outer /= shape[d];\n";
      kernel_source << "  }\n";
      kernel_source << "  int temp_inner = inner_idx;\n";
      kernel_source << "  for (int d = ndim - 1; d > axis; d--) {\n";
      kernel_source << "    indices[d] = temp_inner % shape[d];\n";
      kernel_source << "    temp_inner /= shape[d];\n";
      kernel_source << "  }\n";
      kernel_source << "\n";
      kernel_source << "  // Compute base offset (excluding axis dimension, starting from data offset for sliced arrays)\n";
      kernel_source << "  long base_offset = in_data_offset;\n";
      kernel_source << "  for (int d = 0; d < ndim; d++) {\n";
      kernel_source << "    if (d != axis) {\n";
      kernel_source << "      base_offset += indices[d] * strides[d];\n";
      kernel_source << "    }\n";
      kernel_source << "  }\n";
      kernel_source << "  long axis_stride = strides[axis];\n";
      kernel_source << "\n";
      kernel_source << "  " << accum_type << " result = " << accum_init << ";\n";
      kernel_source << "  for (int r = 0; r < reduce_size; r++) {\n";
      kernel_source << "    long input_idx = base_offset + r * axis_stride;\n";

      // Read input value with conversion if needed
      kernel_source << "    " << accum_type << " val = " << opencl::make_read_expr(in.dtype(), "input[input_idx]") << ";\n";

      if (is_minmax) {
        // For floating point, propagate NaN; for integers, use standard min/max
        if (is_float_type) {
          kernel_source << "    result = isnan(val) ? val : (isnan(result) ? result : " << op_str << "(result, val));\n";
        } else {
          kernel_source << "    result = " << op_str << "(result, val);\n";
        }
      } else if (is_logical) {
        if (reduce_type_ == Reduce::And) {
          kernel_source << "    result = result && (val != 0);\n";
        } else {
          kernel_source << "    result = result || (val != 0);\n";
        }
      } else {
        kernel_source << "    result = result " << op_str << " val;\n";
      }

      kernel_source << "  }\n";
      kernel_source << "  int output_idx = outer_idx * inner_size + inner_idx;\n";

      // Write output with conversion if needed
      if (is_logical || !needs_conv) {
        kernel_source << "  output[output_idx] = (" << out_type_name << ")result;\n";
      } else {
        kernel_source << "  output[output_idx] = " << opencl::make_write_expr(in.dtype(), "result") << ";\n";
      }
      kernel_source << "}\n";

      std::string source_str = kernel_source.str();

      OPENCL_DEBUG_LOG("[Reduce::eval_gpu] Compiling kernel: " << kernel_name);

      cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
      encoder.set_kernel(kernel);
      encoder.set_input_array(in, 0);
      encoder.set_output_array(out, 1);

      // Create buffers for shape and strides
      cl_int err;
      cl_mem shape_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         shape_vec.size() * sizeof(int), shape_vec.data(), &err);
      cl_mem strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           strides_vec.size() * sizeof(int64_t), strides_vec.data(), &err);

      clSetKernelArg(kernel, 2, sizeof(cl_mem), &shape_buf);
      clSetKernelArg(kernel, 3, sizeof(cl_mem), &strides_buf);
      encoder.set_bytes(ndim, 4);
      encoder.set_bytes(axis, 5);
      encoder.set_bytes(static_cast<int>(outer_size), 6);
      encoder.set_bytes(static_cast<int>(reduce_size), 7);
      encoder.set_bytes(static_cast<int>(inner_size), 8);
      encoder.set_bytes(in_data_offset, 9);

      size_t global_size[3] = {static_cast<size_t>(outer_size), static_cast<size_t>(inner_size), 0};
      encoder.dispatch_threads(global_size, nullptr, 2);

      // Track buffers for cleanup after sync
      dev.add_temp_buffer(shape_buf, s.index);
      dev.add_temp_buffer(strides_buf, s.index);
    }

    dev.add_temporary(in, s.index);
    dev.end_encoding(s.index);

    OPENCL_DEBUG_LOG("[Reduce::eval_gpu] Dispatched single-axis reduction kernel");
  }
}

void ArgReduce::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);
  const array& in = inputs[0];

  OPENCL_DEBUG_LOG("[ArgReduce::eval_gpu] reduce_type=" << reduce_type_
            << " axis=" << axis_ << " in.size=" << in.size());

  // Allocate output (uint32_t indices)
  auto out_strides_full = compute_row_major_strides(out.shape());
  array::Flags flags;
  flags.contiguous = true;
  flags.row_contiguous = true;
  flags.col_contiguous = out.size() <= 1;
  out.set_data(
      opencl::allocator().malloc(out.nbytes()),
      out.size(),
      out_strides_full,
      flags);

  auto& dev = opencl::device(Device::gpu);
  auto& s = stream();
  auto& encoder = dev.get_command_encoder(s.index);

  // Get type information
  std::string in_type_name = opencl::type_to_name(in.dtype());

  // Use float for comparison with half/bfloat16
  bool needs_conv = opencl::needs_float_conversion(in.dtype());
  std::string cmp_type = needs_conv ? "float" : in_type_name;

  // Determine comparison operator and initial value
  std::string cmp_op = (reduce_type_ == ArgReduce::ArgMin) ? "<" : ">";
  std::string init_val;
  bool is_float = opencl::is_floating_type(in.dtype());
  if (reduce_type_ == ArgReduce::ArgMin) {
    init_val = is_float ? "INFINITY" : "INT_MAX";
  } else {
    init_val = is_float ? "-INFINITY" : "INT_MIN";
  }

  // Prepare shapes and strides without the reduction axis (like Metal)
  auto in_strides = in.strides();
  auto shape = in.shape();
  auto out_strides = out.strides();
  int64_t axis_stride = in_strides[axis_];
  size_t axis_size = shape[axis_];

  // Remove the axis dimension from shape and strides
  if (static_cast<size_t>(out_strides.size()) == in_strides.size()) {
    out_strides.erase(out_strides.begin() + axis_);
  }
  in_strides.erase(in_strides.begin() + axis_);
  shape.erase(shape.begin() + axis_);
  size_t ndim = shape.size();

  // Compute output size (product of remaining dimensions)
  size_t out_size = 1;
  for (size_t i = 0; i < ndim; i++) {
    out_size *= shape[i];
  }

  std::string op_suffix = (reduce_type_ == ArgReduce::ArgMin) ? "argmin" : "argmax";

  // Check if input is row-contiguous for vectorized kernel
  bool is_row_contiguous = in.flags().row_contiguous;

  // Use parallel workgroup kernel for very large reductions (like vocab argmax)
  // Use vectorized single-thread kernel for medium sizes
  // axis must be the last dimension for these optimizations
  // Support float and half types for parallel/vectorized kernels
  bool is_fast_type = (in_type_name == "float" || in_type_name == "half");
  bool use_parallel = (axis_size >= 8192 && axis_ == in.ndim() - 1 &&
                       is_fast_type && is_row_contiguous);
  bool use_vectorized = (!use_parallel && axis_size >= 1024 && axis_ == in.ndim() - 1 &&
                         is_fast_type && is_row_contiguous);

  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();

  std::string kernel_name;

  if (use_parallel) {
    // Parallel workgroup reduction kernel for very large axis sizes
    // Each workgroup handles one row with 256 threads doing parallel reduction
    kernel_name = op_suffix + "_parallel_" + in_type_name;

    std::string par_cmp = (reduce_type_ == ArgReduce::ArgMin) ? "<" : ">";

    kernel_source << R"(
#define WG_SIZE 256

__kernel void )" << kernel_name << R"((
    __global const )" << in_type_name << R"(* input,
    __global uint* output,
    long in_data_offset,
    int axis_size,
    int n_rows) {
  int row = get_group_id(0);
  if (row >= n_rows) return;

  int lid = get_local_id(0);
  __local float local_val[WG_SIZE];
  __local uint local_idx[WG_SIZE];

  // Apply data offset for sliced arrays
  input = input + in_data_offset;
  __global const )" << in_type_name << R"(* in_row = input + row * axis_size;

  // Phase 1: Each thread finds local best for its portion
  float thread_best = )" << init_val << R"(;
  uint thread_idx = 0;
  for (int i = lid; i < axis_size; i += WG_SIZE) {
    float val = )" << opencl::make_read_expr(in.dtype(), "in_row[i]") << R"(;
    if (val )" << par_cmp << R"( thread_best) {
      thread_best = val;
      thread_idx = i;
    }
  }
  local_val[lid] = thread_best;
  local_idx[lid] = thread_idx;
  barrier(CLK_LOCAL_MEM_FENCE);

  // Parallel reduction for argmax/argmin
  for (int stride = WG_SIZE / 2; stride > 0; stride >>= 1) {
    if (lid < stride) {
      if (local_val[lid + stride] )" << par_cmp << R"( local_val[lid]) {
        local_val[lid] = local_val[lid + stride];
        local_idx[lid] = local_idx[lid + stride];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    output[row] = local_idx[0];
  }
}
)";
  } else if (use_vectorized) {
    // Optimized kernel using vectorized loads for large contiguous argmax/argmin
    // Only works when axis is the last dimension and input is contiguous
    kernel_name = op_suffix + "_vec4_" + in_type_name;

    std::string vec_cmp = (reduce_type_ == ArgReduce::ArgMin) ? "<" : ">";
    std::string vec_type = (in.dtype() == float16) ? "half4" : "float4";
    std::string val_type = "float";  // Always use float for comparisons

    kernel_source << R"(
__kernel void )" << kernel_name << R"((
    __global const )" << in_type_name << R"(* input,
    __global uint* output,
    __global const int* shape,
    __global const long* in_strides,
    __global const long* out_strides,
    int ndim,
    long axis_stride,
    int axis_size,
    long in_data_offset) {
  int row_idx = get_global_id(0);
  if (row_idx >= )" << out_size << R"() return;

  // Apply data offset for sliced arrays
  input = input + in_data_offset;

  // Compute input offset using strides (elem_to_loc)
  long in_idx = 0;
  int remaining = row_idx;
  for (int d = ndim - 1; d >= 0; d--) {
    int coord = remaining % shape[d];
    remaining /= shape[d];
    in_idx += coord * in_strides[d];
  }

  __global const )" << in_type_name << R"(* in_row = input + in_idx;

  float best_val = )" << init_val << R"(;
  uint best_idx = 0;

  int i = 0;
  // Process 16 elements per iteration using )" << vec_type << R"(
  for (; i + 15 < axis_size; i += 16) {
    )" << vec_type << R"( v0 = vload4(0, in_row + i);
    )" << vec_type << R"( v1 = vload4(0, in_row + i + 4);
    )" << vec_type << R"( v2 = vload4(0, in_row + i + 8);
    )" << vec_type << R"( v3 = vload4(0, in_row + i + 12);

    if ((float)v0.s0 )" << vec_cmp << R"( best_val) { best_val = (float)v0.s0; best_idx = i; }
    if ((float)v0.s1 )" << vec_cmp << R"( best_val) { best_val = (float)v0.s1; best_idx = i + 1; }
    if ((float)v0.s2 )" << vec_cmp << R"( best_val) { best_val = (float)v0.s2; best_idx = i + 2; }
    if ((float)v0.s3 )" << vec_cmp << R"( best_val) { best_val = (float)v0.s3; best_idx = i + 3; }
    if ((float)v1.s0 )" << vec_cmp << R"( best_val) { best_val = (float)v1.s0; best_idx = i + 4; }
    if ((float)v1.s1 )" << vec_cmp << R"( best_val) { best_val = (float)v1.s1; best_idx = i + 5; }
    if ((float)v1.s2 )" << vec_cmp << R"( best_val) { best_val = (float)v1.s2; best_idx = i + 6; }
    if ((float)v1.s3 )" << vec_cmp << R"( best_val) { best_val = (float)v1.s3; best_idx = i + 7; }
    if ((float)v2.s0 )" << vec_cmp << R"( best_val) { best_val = (float)v2.s0; best_idx = i + 8; }
    if ((float)v2.s1 )" << vec_cmp << R"( best_val) { best_val = (float)v2.s1; best_idx = i + 9; }
    if ((float)v2.s2 )" << vec_cmp << R"( best_val) { best_val = (float)v2.s2; best_idx = i + 10; }
    if ((float)v2.s3 )" << vec_cmp << R"( best_val) { best_val = (float)v2.s3; best_idx = i + 11; }
    if ((float)v3.s0 )" << vec_cmp << R"( best_val) { best_val = (float)v3.s0; best_idx = i + 12; }
    if ((float)v3.s1 )" << vec_cmp << R"( best_val) { best_val = (float)v3.s1; best_idx = i + 13; }
    if ((float)v3.s2 )" << vec_cmp << R"( best_val) { best_val = (float)v3.s2; best_idx = i + 14; }
    if ((float)v3.s3 )" << vec_cmp << R"( best_val) { best_val = (float)v3.s3; best_idx = i + 15; }
  }
  for (; i + 3 < axis_size; i += 4) {
    )" << vec_type << R"( v = vload4(0, in_row + i);
    if ((float)v.s0 )" << vec_cmp << R"( best_val) { best_val = (float)v.s0; best_idx = i; }
    if ((float)v.s1 )" << vec_cmp << R"( best_val) { best_val = (float)v.s1; best_idx = i + 1; }
    if ((float)v.s2 )" << vec_cmp << R"( best_val) { best_val = (float)v.s2; best_idx = i + 2; }
    if ((float)v.s3 )" << vec_cmp << R"( best_val) { best_val = (float)v.s3; best_idx = i + 3; }
  }
  for (; i < axis_size; i++) {
    float val = (float)in_row[i];
    if (val )" << vec_cmp << R"( best_val) {
      best_val = val;
      best_idx = i;
    }
  }

  output[row_idx] = best_idx;
}
)";
  } else {
    // Stride-aware kernel - handles non-contiguous inputs directly (like Metal)
    kernel_name = op_suffix + "_strided_" + in_type_name;

    kernel_source << "\n__kernel void " << kernel_name << "(\n";
    kernel_source << "    __global const " << in_type_name << "* input,\n";
    kernel_source << "    __global uint* output,\n";
    kernel_source << "    __global const int* shape,\n";
    kernel_source << "    __global const long* in_strides,\n";
    kernel_source << "    __global const long* out_strides,\n";
    kernel_source << "    int ndim,\n";
    kernel_source << "    long axis_stride,\n";
    kernel_source << "    int axis_size,\n";
    kernel_source << "    long in_data_offset) {\n";
    kernel_source << "  int row_idx = get_global_id(0);\n";
    kernel_source << "  if (row_idx >= " << out_size << ") return;\n\n";

    // Compute input and output offsets using strides (elem_to_loc style)
    kernel_source << "  // Compute input offset using strides (starting from data offset for sliced arrays)\n";
    kernel_source << "  long in_idx = in_data_offset;\n";
    kernel_source << "  long out_idx = 0;\n";
    kernel_source << "  int remaining = row_idx;\n";
    kernel_source << "  for (int d = ndim - 1; d >= 0; d--) {\n";
    kernel_source << "    int coord = remaining % shape[d];\n";
    kernel_source << "    remaining /= shape[d];\n";
    kernel_source << "    in_idx += coord * in_strides[d];\n";
    kernel_source << "    out_idx += coord * out_strides[d];\n";
    kernel_source << "  }\n\n";

    kernel_source << "  " << cmp_type << " best_val = " << init_val << ";\n";
    kernel_source << "  uint best_idx = 0;\n\n";

    kernel_source << "  // Reduce along axis using axis_stride\n";
    kernel_source << "  for (int i = 0; i < axis_size; i++) {\n";
    kernel_source << "    long idx = in_idx + (long)i * axis_stride;\n";
    kernel_source << "    " << cmp_type << " val = " << opencl::make_read_expr(in.dtype(), "input[idx]") << ";\n";
    kernel_source << "    if (val " << cmp_op << " best_val) {\n";
    kernel_source << "      best_val = val;\n";
    kernel_source << "      best_idx = (uint)i;\n";
    kernel_source << "    }\n";
    kernel_source << "  }\n\n";

    kernel_source << "  output[out_idx] = best_idx;\n";
    kernel_source << "}\n";
  }

  std::string source_str = kernel_source.str();

  // Compute data offset for sliced arrays (offset is in bytes, convert to elements)
  int64_t in_data_offset = in.offset() / in.itemsize();

  cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
  encoder.set_kernel(kernel);
  encoder.set_input_array(in, 0);
  encoder.set_output_array(out, 1);

  if (use_parallel) {
    // Set arguments for parallel workgroup kernel
    encoder.set_bytes(in_data_offset, 2);
    encoder.set_bytes(static_cast<int>(axis_size), 3);
    encoder.set_bytes(static_cast<int>(out_size), 4);

    // Dispatch one workgroup per row, 256 threads per workgroup
    size_t local_size[3] = {256, 0, 0};
    size_t global_size[3] = {out_size * 256, 0, 0};
    encoder.dispatch_threads(global_size, local_size, 1);

    dev.end_encoding(s.index);
    OPENCL_DEBUG_LOG("[ArgReduce::eval_gpu] Dispatched parallel argreduce kernel");
    return;
  }

  // Create buffers for shape and strides
  // Handle ndim == 0 case (scalar)
  std::vector<int> shape_vec;
  std::vector<int64_t> in_strides_vec;
  std::vector<int64_t> out_strides_vec;

  if (ndim == 0) {
    // Scalar case - pass placeholder
    shape_vec.push_back(1);
    in_strides_vec.push_back(0);
    out_strides_vec.push_back(0);
  } else {
    shape_vec.assign(shape.begin(), shape.end());
    in_strides_vec.assign(in_strides.begin(), in_strides.end());
    out_strides_vec.assign(out_strides.begin(), out_strides.end());
  }

  cl_int err;
  cl_mem shape_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     shape_vec.size() * sizeof(int), shape_vec.data(), &err);
  cl_mem in_strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                          in_strides_vec.size() * sizeof(int64_t), in_strides_vec.data(), &err);
  cl_mem out_strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           out_strides_vec.size() * sizeof(int64_t), out_strides_vec.data(), &err);

  clSetKernelArg(kernel, 2, sizeof(cl_mem), &shape_buf);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &in_strides_buf);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_strides_buf);
  encoder.set_bytes(static_cast<int>(ndim), 5);
  encoder.set_bytes(axis_stride, 6);
  encoder.set_bytes(static_cast<int>(axis_size), 7);
  encoder.set_bytes(in_data_offset, 8);

  // Dispatch: 1D, one thread per output element
  size_t global_size[3] = {out_size, 0, 0};
  encoder.dispatch_threads(global_size, nullptr, 1);

  dev.end_encoding(s.index);

  // Clean up temporary buffers
  clReleaseMemObject(shape_buf);
  clReleaseMemObject(in_strides_buf);
  clReleaseMemObject(out_strides_buf);

  OPENCL_DEBUG_LOG("[ArgReduce::eval_gpu] Dispatched stride-aware argreduce kernel");
}

} // namespace mlx::core
