// Copyright © 2025 MLX Contributors

#include "mlx/backend/common/binary.h"
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/ops.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/primitives.h"

#include <iostream>

#define BINARY_GPU(func)                                              \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    binary_op_gpu(inputs, out, name());                               \
  }

namespace mlx::core {

namespace {

// Get appropriate kernel for binary operation
cl_kernel get_binary_kernel(
    opencl::Device& d,
    const std::string& kernel_name,
    const Dtype& dtype,
    const std::string& op_name,
    int work_per_thread = 1) {

  OPENCL_DEBUG_LOG("[KERNEL_DEBUG] get_binary_kernel: kernel_name=" << kernel_name << ", op_name=" << op_name);

  // Build kernel source
  std::string source = opencl::get_kernel_preamble();
  source += opencl::get_binary_ops();

  // Determine template type from kernel name prefix
  std::string template_type;
  if (kernel_name.substr(0, 2) == "ss") {
    template_type = "binary_ss";
  } else if (kernel_name.substr(0, 2) == "sv") {
    template_type = "binary_sv";
  } else if (kernel_name.substr(0, 2) == "vs") {
    template_type = "binary_vs";
  } else if (kernel_name.substr(0, 2) == "vv") {
    template_type = "binary_vv";
  } else if (kernel_name[0] == 'g') {
    template_type = "binary_g";
  } else {
    throw std::runtime_error("Unknown binary kernel variant: " + kernel_name);
  }

  std::string type_name = opencl::type_to_name(dtype);

  // Comparison operations output bool/int instead of the input type
  // Note: OpenCL doesn't support bool* in global memory, so we use uint8_t (0 or 1)
  std::string output_type = type_name;
  if (op_name == "Equal" || op_name == "NotEqual" || op_name == "NaNEqual" ||
      op_name == "Less" || op_name == "LessEqual" ||
      op_name == "Greater" || op_name == "GreaterEqual") {
    output_type = "uint8_t";  // Use uint8_t for boolean results (0 or 1)
  }

  // Some operations need type-specific function names since OpenCL C doesn't support overloading
  std::string actual_op_name = op_name;
  if (op_name == "Multiply" && type_name == "float2") {
    // Complex multiply needs proper (a+bi)*(c+di) formula
    actual_op_name = "Multiply_complex";
  } else if (op_name == "Divide" && type_name == "float2") {
    // Complex divide needs proper formula
    actual_op_name = "Divide_complex";
  } else if (op_name == "LogAddExp" && type_name == "float2") {
    actual_op_name = "LogAddExp_float2";
  } else if (op_name == "LogAddExp" && type_name == "half") {
    actual_op_name = "LogAddExp_half";
  } else if (op_name == "Power") {
    // Power needs type-specific implementations
    if (type_name == "float") {
      actual_op_name = "Power_float";
    } else if (type_name == "float2") {
      actual_op_name = "Power_float2";
    } else if (type_name == "half" || type_name == "uint16_t") {  // half or bfloat16
      actual_op_name = "Power_half";
    } else if (type_name == "int32_t") {
      actual_op_name = "ipow_int32";
    } else if (type_name == "int64_t") {
      actual_op_name = "ipow_int64";
    } else if (type_name == "uint32_t") {
      actual_op_name = "ipow_uint32";
    } else if (type_name == "uint64_t") {
      actual_op_name = "ipow_uint64";
    } else if (type_name == "uchar" || type_name == "uint8_t") {
      actual_op_name = "ipow_bool";
    } else {
      actual_op_name = "Power_float";  // fallback
    }
  } else if (op_name == "Remainder") {
    // Remainder needs type-specific implementations for Python-style floored remainder
    if (type_name == "float" || type_name == "float2" || type_name == "bfloat16_t") {
      actual_op_name = "Remainder_float";
    } else if (type_name == "half") {
      actual_op_name = "Remainder_half";
    } else if (type_name == "long") {
      actual_op_name = "Remainder_long";
    } else {
      actual_op_name = "Remainder_int";
    }
  } else if (op_name == "Maximum") {
    // Maximum needs type-specific implementations for NaN handling
    if (type_name == "float") {
      actual_op_name = "Maximum_float";
    } else if (type_name == "float2") {
      actual_op_name = "Maximum_float2";
    } else if (type_name == "half" || type_name == "bfloat16_t") {
      actual_op_name = "Maximum_half";
    } else {
      actual_op_name = "Maximum_int";
    }
  } else if (op_name == "Minimum") {
    // Minimum needs type-specific implementations for NaN handling
    if (type_name == "float") {
      actual_op_name = "Minimum_float";
    } else if (type_name == "float2") {
      actual_op_name = "Minimum_float2";
    } else if (type_name == "half" || type_name == "bfloat16_t") {
      actual_op_name = "Minimum_half";
    } else {
      actual_op_name = "Minimum_int";
    }
  } else if (op_name == "Equal" && type_name == "float2") {
    // Complex comparison needs to return scalar bool, not vector
    actual_op_name = "Equal_complex";
  } else if (op_name == "NaNEqual" && type_name == "float2") {
    actual_op_name = "NaNEqual_complex";
  } else if (op_name == "NotEqual" && type_name == "float2") {
    actual_op_name = "NotEqual_complex";
  }

  source += opencl::get_template_definition(
      kernel_name, template_type, type_name, output_type, actual_op_name, work_per_thread);

  OPENCL_DEBUG_LOG("[KERNEL_DEBUG] Kernel source for " << kernel_name << ":");
  OPENCL_DEBUG_LOG("========== BEGIN KERNEL SOURCE ==========");
  OPENCL_DEBUG_LOG(source);
  OPENCL_DEBUG_LOG("========== END KERNEL SOURCE ==========");

  OPENCL_DEBUG_LOG("[KERNEL_DEBUG] Calling d.get_kernel to compile kernel...");

  cl_kernel result = d.get_kernel(kernel_name, source, "");

  OPENCL_DEBUG_LOG("[KERNEL_DEBUG] Kernel compiled successfully: " << (void*)result);
  OPENCL_DEBUG_LOG("[KERNEL_DEBUG] About to return kernel...");
  return result;
}

} // anonymous namespace

void binary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const std::string& op_name,
    const Stream& s) {

  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);

  if (out.size() == 0) {
    return;
  }

  auto& d = opencl::device(s.device);

  // Determine kernel variant and configure work sizes
  std::string kernel_name;
  size_t global_size[3] = {1, 1, 1};  // Initialize to 1s, not 0s
  size_t local_size[3] = {256, 1, 1};
  int work_dim = 1;  // Use 1D dispatch for simplicity (3D was causing crashes)

  // Calculate work-per-thread for contiguous operations
  int work_per_thread = 1;
  if (bopt != BinaryOpType::General && bopt != BinaryOpType::ScalarScalar) {
    work_per_thread = opencl::get_work_per_thread(a.dtype(), out.size());
  }

  std::string type_name_for_kernel = opencl::type_to_name(a.dtype());

  switch (bopt) {
    case BinaryOpType::ScalarScalar:
      kernel_name = "ss_" + op_name + "_" + type_name_for_kernel;
      global_size[0] = out.size();  // Exact size, no rounding
      break;

    case BinaryOpType::ScalarVector:
      kernel_name = "sv";
      if (work_per_thread > 1) kernel_name += "n";
      kernel_name += "_" + op_name + "_" + type_name_for_kernel;
      global_size[0] = opencl::ceildiv(b.size(), static_cast<size_t>(work_per_thread));
      break;

    case BinaryOpType::VectorScalar:
      kernel_name = "vs";
      if (work_per_thread > 1) kernel_name += "n";
      kernel_name += "_" + op_name + "_" + type_name_for_kernel;
      global_size[0] = opencl::ceildiv(a.size(), static_cast<size_t>(work_per_thread));
      break;

    case BinaryOpType::VectorVector:
      kernel_name = "vv";
      if (work_per_thread > 1) kernel_name += "n";
      kernel_name += "_" + op_name + "_" + type_name_for_kernel;
      global_size[0] = opencl::ceildiv(a.size(), static_cast<size_t>(work_per_thread));
      break;

    case BinaryOpType::General:
      kernel_name = "g_" + op_name + "_" + type_name_for_kernel;
      global_size[0] = out.size();  // Exact size, no rounding
      work_per_thread = 1;  // General case doesn't use work-per-thread yet
      break;
  }

  // Get or compile kernel
  cl_kernel kernel = get_binary_kernel(d, kernel_name, a.dtype(), op_name, work_per_thread);

  // Get command encoder
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_kernel(kernel);

  // Set kernel arguments based on operation type
  int arg_idx = 0;
  compute_encoder.set_input_array(a, arg_idx++);
  compute_encoder.set_input_array(b, arg_idx++);
  compute_encoder.set_output_array(out, arg_idx++);

  if (bopt == BinaryOpType::General) {
    // General broadcast kernel needs shape and strides as constant memory buffers
    std::vector<int> shape_vec(out.shape().begin(), out.shape().end());
    std::vector<int64_t> strides_a_vec(a.strides().begin(), a.strides().end());
    std::vector<int64_t> strides_b_vec(b.strides().begin(), b.strides().end());

    int ndim = out.ndim();
    int size = out.size();

    // Handle scalar case (ndim=0)
    if (shape_vec.empty()) shape_vec.push_back(1);
    if (strides_a_vec.empty()) strides_a_vec.push_back(0);
    if (strides_b_vec.empty()) strides_b_vec.push_back(0);

    // Create constant memory buffers for shape and strides
    cl_int err;
    cl_mem shape_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       shape_vec.size() * sizeof(int), shape_vec.data(), &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("[binary_op_gpu] Failed to create shape buffer: " + std::to_string(err));
    }

    cl_mem strides_a_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           strides_a_vec.size() * sizeof(int64_t), strides_a_vec.data(), &err);
    if (err != CL_SUCCESS) {
      clReleaseMemObject(shape_buf);
      throw std::runtime_error("[binary_op_gpu] Failed to create strides_a buffer: " + std::to_string(err));
    }

    cl_mem strides_b_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           strides_b_vec.size() * sizeof(int64_t), strides_b_vec.data(), &err);
    if (err != CL_SUCCESS) {
      clReleaseMemObject(shape_buf);
      clReleaseMemObject(strides_a_buf);
      throw std::runtime_error("[binary_op_gpu] Failed to create strides_b buffer: " + std::to_string(err));
    }

    // Set kernel arguments: shape, strides_a, strides_b, ndim, size, offset_a, offset_b
    clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &shape_buf);
    clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &strides_a_buf);
    clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &strides_b_buf);
    compute_encoder.set_bytes(ndim, arg_idx++);
    compute_encoder.set_bytes(size, arg_idx++);

    // Pass array offsets (convert from bytes to elements)
    int64_t offset_a = static_cast<int64_t>(a.offset()) / a.itemsize();
    int64_t offset_b = static_cast<int64_t>(b.offset()) / b.itemsize();
    int64_t offset_out = static_cast<int64_t>(out.offset()) / out.itemsize();
    compute_encoder.set_bytes(offset_a, arg_idx++);
    compute_encoder.set_bytes(offset_b, arg_idx++);
    compute_encoder.set_bytes(offset_out, arg_idx++);

    // Dispatch kernel
    compute_encoder.dispatch_threads(global_size, nullptr, work_dim);

    // Track buffers for cleanup after sync
    d.add_temp_buffer(shape_buf, s.index);
    d.add_temp_buffer(strides_a_buf, s.index);
    d.add_temp_buffer(strides_b_buf, s.index);

    // Release temp buffers after kernel execution completes
    d.end_encoding(s.index);

    return;  // Early return since we handled dispatch here
  } else {
    // Simple kernels need size and offsets
    int size = static_cast<int>(out.size());
    compute_encoder.set_bytes(size, arg_idx++);

    // Pass array offsets (convert from bytes to elements)
    int64_t offset_a = static_cast<int64_t>(a.offset()) / a.itemsize();
    int64_t offset_b = static_cast<int64_t>(b.offset()) / b.itemsize();
    int64_t offset_out = static_cast<int64_t>(out.offset()) / out.itemsize();
    compute_encoder.set_bytes(offset_a, arg_idx++);
    compute_encoder.set_bytes(offset_b, arg_idx++);
    compute_encoder.set_bytes(offset_out, arg_idx++);
  }

  // Dispatch kernel - pass nullptr for local_work_size to let OpenCL choose optimal size
  compute_encoder.dispatch_threads(global_size, nullptr, work_dim);
}

void binary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string& op_name,
    const Stream& s) {
  auto& a = inputs[0];
  auto& b = inputs[1];
  auto bopt = get_binary_op_type(a, b);

  // Use OpenCL allocator
  auto malloc_fn = [](size_t size) -> allocator::Buffer {
    return opencl::allocator().malloc(size);
  };
  set_binary_op_output_data(a, b, out, bopt, malloc_fn);
  binary_op_gpu_inplace(inputs, out, op_name, s);
}

void binary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string& op_name) {
  auto& s = out.primitive().stream();
  binary_op_gpu(inputs, out, op_name, s);
}

// Register all binary operations
BINARY_GPU(Add)
BINARY_GPU(ArcTan2)
BINARY_GPU(Divide)
BINARY_GPU(Equal)
BINARY_GPU(Greater)
BINARY_GPU(GreaterEqual)
BINARY_GPU(Less)
BINARY_GPU(LessEqual)
BINARY_GPU(LogAddExp)
BINARY_GPU(LogicalAnd)
BINARY_GPU(LogicalOr)
BINARY_GPU(Maximum)
BINARY_GPU(Minimum)
BINARY_GPU(Multiply)
BINARY_GPU(NotEqual)
BINARY_GPU(Power)
BINARY_GPU(Remainder)
BINARY_GPU(Subtract)

void BitwiseBinary::eval_gpu(const std::vector<array>& inputs, array& out) {
  // Map Op enum to operation name matching the macros in ops.cpp
  std::string op_name;
  switch (op_) {
    case BitwiseBinary::And:
      op_name = "BitwiseAnd";
      break;
    case BitwiseBinary::Or:
      op_name = "BitwiseOr";
      break;
    case BitwiseBinary::Xor:
      op_name = "BitwiseXor";
      break;
    case BitwiseBinary::LeftShift:
      op_name = "LeftShift";
      break;
    case BitwiseBinary::RightShift:
      op_name = "RightShift";
      break;
    default:
      throw std::runtime_error("[BitwiseBinary::eval_gpu] Unknown op");
  }
  binary_op_gpu(inputs, out, op_name);
}

void DivMod::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  assert(inputs.size() == 2);
  assert(outputs.size() == 2);

  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& out_quotient = outputs[0];
  auto& out_remainder = outputs[1];

  if (out_quotient.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = opencl::device(s.device);

  // Determine operation type (scalar/vector/general)
  auto bopt = get_binary_op_type(a, b);

  // Allocate outputs using set_binary_op_output_data pattern
  auto alloc_fn = [](size_t size) -> allocator::Buffer {
    return opencl::allocator().malloc(size);
  };
  set_binary_op_output_data(a, b, out_quotient, bopt, alloc_fn);
  // For remainder, copy the same data layout as quotient
  out_remainder.set_data(alloc_fn(out_remainder.nbytes()), out_quotient.data_size(),
                         out_quotient.strides(), out_quotient.flags());

  std::string type_name = opencl::type_to_name(a.dtype());
  bool is_float = (type_name == "float" || type_name == "half" || type_name == "bfloat16_t");
  bool is_bfloat16 = (type_name == "bfloat16_t");
  bool is_half = (type_name == "half");

  // Helper lambda to generate divmod computation code
  auto gen_divmod_compute = [&](std::ostringstream& ks, const std::string& va_expr, const std::string& vb_expr) {
    if (is_bfloat16) {
      ks << "  float va = bfloat16_to_float(" << va_expr << ");\n";
      ks << "  float vb = bfloat16_to_float(" << vb_expr << ");\n";
      ks << "  quotient[out_idx] = float_to_bfloat16(trunc(va / vb));\n";
      ks << "  remainder[out_idx] = float_to_bfloat16(fmod(va, vb));\n";
    } else if (is_half) {
      ks << "  float va = convert_float(" << va_expr << ");\n";
      ks << "  float vb = convert_float(" << vb_expr << ");\n";
      ks << "  quotient[out_idx] = convert_half(trunc(va / vb));\n";
      ks << "  remainder[out_idx] = convert_half(fmod(va, vb));\n";
    } else if (is_float) {
      ks << "  " << type_name << " va = " << va_expr << ";\n";
      ks << "  " << type_name << " vb = " << vb_expr << ";\n";
      ks << "  quotient[out_idx] = trunc(va / vb);\n";
      ks << "  remainder[out_idx] = fmod(va, vb);\n";
    } else {
      ks << "  " << type_name << " va = " << va_expr << ";\n";
      ks << "  " << type_name << " vb = " << vb_expr << ";\n";
      ks << "  quotient[out_idx] = va / vb;\n";
      ks << "  remainder[out_idx] = va % vb;\n";
    }
  };

  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();

  std::string kernel_name;
  size_t global_size[3] = {0, 0, 0};
  int arg_idx = 0;

  auto& encoder = d.get_command_encoder(s.index);

  switch (bopt) {
    case BinaryOpType::ScalarScalar:
    case BinaryOpType::VectorVector: {
      // Both arrays have same shape and are contiguous
      kernel_name = "divmod_vv_" + type_name;
      kernel_source << "\n__kernel void " << kernel_name << "(\n";
      kernel_source << "    __global const " << type_name << "* a,\n";
      kernel_source << "    __global const " << type_name << "* b,\n";
      kernel_source << "    __global " << type_name << "* quotient,\n";
      kernel_source << "    __global " << type_name << "* remainder,\n";
      kernel_source << "    int size) {\n";
      kernel_source << "  int out_idx = get_global_id(0);\n";
      kernel_source << "  if (out_idx >= size) return;\n";
      gen_divmod_compute(kernel_source, "a[out_idx]", "b[out_idx]");
      kernel_source << "}\n";

      cl_kernel kernel = d.get_kernel(kernel_name, kernel_source.str());
      encoder.set_kernel(kernel);
      encoder.set_input_array(a, arg_idx++);
      encoder.set_input_array(b, arg_idx++);
      encoder.set_output_array(out_quotient, arg_idx++);
      encoder.set_output_array(out_remainder, arg_idx++);
      encoder.set_bytes(static_cast<int>(out_quotient.size()), arg_idx++);

      global_size[0] = out_quotient.size();
      encoder.dispatch_threads(global_size, nullptr, 1);
      break;
    }

    case BinaryOpType::ScalarVector: {
      // a is scalar, b is vector
      kernel_name = "divmod_sv_" + type_name;
      kernel_source << "\n__kernel void " << kernel_name << "(\n";
      kernel_source << "    __global const " << type_name << "* a,\n";
      kernel_source << "    __global const " << type_name << "* b,\n";
      kernel_source << "    __global " << type_name << "* quotient,\n";
      kernel_source << "    __global " << type_name << "* remainder,\n";
      kernel_source << "    int size) {\n";
      kernel_source << "  int out_idx = get_global_id(0);\n";
      kernel_source << "  if (out_idx >= size) return;\n";
      gen_divmod_compute(kernel_source, "a[0]", "b[out_idx]");
      kernel_source << "}\n";

      cl_kernel kernel = d.get_kernel(kernel_name, kernel_source.str());
      encoder.set_kernel(kernel);
      encoder.set_input_array(a, arg_idx++);
      encoder.set_input_array(b, arg_idx++);
      encoder.set_output_array(out_quotient, arg_idx++);
      encoder.set_output_array(out_remainder, arg_idx++);
      encoder.set_bytes(static_cast<int>(out_quotient.size()), arg_idx++);

      global_size[0] = out_quotient.size();
      encoder.dispatch_threads(global_size, nullptr, 1);
      break;
    }

    case BinaryOpType::VectorScalar: {
      // a is vector, b is scalar
      kernel_name = "divmod_vs_" + type_name;
      kernel_source << "\n__kernel void " << kernel_name << "(\n";
      kernel_source << "    __global const " << type_name << "* a,\n";
      kernel_source << "    __global const " << type_name << "* b,\n";
      kernel_source << "    __global " << type_name << "* quotient,\n";
      kernel_source << "    __global " << type_name << "* remainder,\n";
      kernel_source << "    int size) {\n";
      kernel_source << "  int out_idx = get_global_id(0);\n";
      kernel_source << "  if (out_idx >= size) return;\n";
      gen_divmod_compute(kernel_source, "a[out_idx]", "b[0]");
      kernel_source << "}\n";

      cl_kernel kernel = d.get_kernel(kernel_name, kernel_source.str());
      encoder.set_kernel(kernel);
      encoder.set_input_array(a, arg_idx++);
      encoder.set_input_array(b, arg_idx++);
      encoder.set_output_array(out_quotient, arg_idx++);
      encoder.set_output_array(out_remainder, arg_idx++);
      encoder.set_bytes(static_cast<int>(out_quotient.size()), arg_idx++);

      global_size[0] = out_quotient.size();
      encoder.dispatch_threads(global_size, nullptr, 1);
      break;
    }

    case BinaryOpType::General: {
      // General broadcast case - need shape and strides
      kernel_name = "divmod_g_" + type_name;
      kernel_source << "\n__kernel void " << kernel_name << "(\n";
      kernel_source << "    __global const " << type_name << "* a,\n";
      kernel_source << "    __global const " << type_name << "* b,\n";
      kernel_source << "    __global " << type_name << "* quotient,\n";
      kernel_source << "    __global " << type_name << "* remainder,\n";
      kernel_source << "    __global const int* shape,\n";
      kernel_source << "    __global const long* strides_a,\n";
      kernel_source << "    __global const long* strides_b,\n";
      kernel_source << "    int ndim,\n";
      kernel_source << "    int size,\n";
      kernel_source << "    long offset_a,\n";
      kernel_source << "    long offset_b) {\n";
      kernel_source << "  int out_idx = get_global_id(0);\n";
      kernel_source << "  if (out_idx >= size) return;\n";
      kernel_source << "  // Convert linear index to multi-dimensional and compute offsets\n";
      kernel_source << "  long idx_a = offset_a;\n";
      kernel_source << "  long idx_b = offset_b;\n";
      kernel_source << "  int remaining = out_idx;\n";
      kernel_source << "  for (int d = ndim - 1; d >= 0; d--) {\n";
      kernel_source << "    int coord = remaining % shape[d];\n";
      kernel_source << "    remaining /= shape[d];\n";
      kernel_source << "    idx_a += coord * strides_a[d];\n";
      kernel_source << "    idx_b += coord * strides_b[d];\n";
      kernel_source << "  }\n";
      gen_divmod_compute(kernel_source, "a[idx_a]", "b[idx_b]");
      kernel_source << "}\n";

      cl_kernel kernel = d.get_kernel(kernel_name, kernel_source.str());
      encoder.set_kernel(kernel);
      encoder.set_input_array(a, arg_idx++);
      encoder.set_input_array(b, arg_idx++);
      encoder.set_output_array(out_quotient, arg_idx++);
      encoder.set_output_array(out_remainder, arg_idx++);

      // Create buffers for shape and strides
      std::vector<int> shape_vec(out_quotient.shape().begin(), out_quotient.shape().end());
      std::vector<int64_t> strides_a_vec(a.strides().begin(), a.strides().end());
      std::vector<int64_t> strides_b_vec(b.strides().begin(), b.strides().end());

      int ndim = out_quotient.ndim();
      int size = out_quotient.size();

      // Handle scalar case
      if (shape_vec.empty()) shape_vec.push_back(1);
      if (strides_a_vec.empty()) strides_a_vec.push_back(0);
      if (strides_b_vec.empty()) strides_b_vec.push_back(0);

      cl_int err;
      cl_mem shape_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         shape_vec.size() * sizeof(int), shape_vec.data(), &err);
      cl_mem strides_a_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             strides_a_vec.size() * sizeof(int64_t), strides_a_vec.data(), &err);
      cl_mem strides_b_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                             strides_b_vec.size() * sizeof(int64_t), strides_b_vec.data(), &err);

      clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &shape_buf);
      clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &strides_a_buf);
      clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &strides_b_buf);
      encoder.set_bytes(ndim, arg_idx++);
      encoder.set_bytes(size, arg_idx++);

      int64_t offset_a = static_cast<int64_t>(a.offset()) / a.itemsize();
      int64_t offset_b = static_cast<int64_t>(b.offset()) / b.itemsize();
      encoder.set_bytes(offset_a, arg_idx++);
      encoder.set_bytes(offset_b, arg_idx++);

      global_size[0] = out_quotient.size();
      encoder.dispatch_threads(global_size, nullptr, 1);

      // Track buffers for cleanup
      d.add_temp_buffer(shape_buf, s.index);
      d.add_temp_buffer(strides_a_buf, s.index);
      d.add_temp_buffer(strides_b_buf, s.index);
      break;
    }
  }

  d.add_temporary(a, s.index);
  d.add_temporary(b, s.index);
  d.end_encoding(s.index);
}

} // namespace mlx::core
