// Copyright © 2025 MLX Contributors

#include "mlx/backend/common/unary.h"
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/ops.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/primitives.h"

#include <algorithm>
#include <iostream>

#define UNARY_GPU(func)                                               \
  void func::eval_gpu(const std::vector<array>& inputs, array& out) { \
    unary_op_gpu(inputs, out, name());                                \
  }

namespace mlx::core {

namespace {

// Get appropriate kernel for unary operation
// Map MLX operation names to OpenCL function names
std::string map_op_to_opencl_func(const std::string& op_name) {
  // Map operation names to OpenCL function names
  if (op_name == "Negative") return "neg";
  if (op_name == "Abs") return "fabs";
  if (op_name == "Sign") return "sign";
  if (op_name == "LogicalNot") return "logical_not";
  if (op_name == "Square") return "square";

  // Math functions that map directly to OpenCL built-ins
  if (op_name == "Sqrt") return "sqrt";
  if (op_name == "Exp") return "exp";
  if (op_name == "Expm1") return "expm1";
  if (op_name == "Log") return "log";
  if (op_name == "Log1p") return "log1p";
  if (op_name == "Log2") return "log2";
  if (op_name == "Log10") return "log10";

  // Trig functions
  if (op_name == "Sin") return "sin";
  if (op_name == "Cos") return "cos";
  if (op_name == "Tan") return "tan";
  if (op_name == "ArcSin") return "asin";
  if (op_name == "ArcCos") return "acos";
  if (op_name == "ArcTan") return "atan";

  // Hyperbolic functions
  if (op_name == "Sinh") return "sinh";
  if (op_name == "Cosh") return "cosh";
  if (op_name == "Tanh") return "tanh";
  if (op_name == "ArcSinh") return "asinh";
  if (op_name == "ArcCosh") return "acosh";
  if (op_name == "ArcTanh") return "atanh";

  // Rounding
  if (op_name == "Ceil") return "ceil";
  if (op_name == "Floor") return "floor";
  // MLX round uses round-to-nearest-even (banker's rounding), which is rint() in OpenCL
  // OpenCL round() does round-half-away-from-zero which is different
  if (op_name == "Round") return "rint";

  // Special functions
  if (op_name == "Erf") return "erf_approx";
  if (op_name == "ErfInv") return "erfinv_approx";
  if (op_name == "Sigmoid") return "sigmoid";

  // Complex operations
  if (op_name == "Conjugate") return "conjugate";
  if (op_name == "Real") return "creal";
  if (op_name == "Imag") return "cimag";

  // Bitwise operations
  if (op_name == "BitwiseInvert") return "bitwise_invert";

  // Default: use lowercase version
  std::string result = op_name;
  std::transform(result.begin(), result.end(), result.begin(), ::tolower);
  return result;
}

cl_kernel get_unary_kernel(
    opencl::Device& d,
    const std::string& kernel_name,
    const Dtype& in_dtype,
    const Dtype& out_dtype,
    const std::string& op_name,
    int work_per_thread = 1) {

  // Build kernel source
  std::string source = opencl::get_kernel_preamble();
  source += opencl::get_unary_ops();

  // Determine template type from kernel name
  std::string template_type;
  if (kernel_name[0] == 'v') {
    template_type = "unary_v";
  } else if (kernel_name[0] == 'g') {
    template_type = "unary_g";
  } else {
    throw std::runtime_error("Unknown kernel variant: " + kernel_name);
  }

  std::string in_type = opencl::type_to_name(in_dtype);
  std::string out_type = opencl::type_to_name(out_dtype);

  // Map operation name to OpenCL function name
  std::string opencl_func = map_op_to_opencl_func(op_name);

  // Handle type-specific operations that need different functions per type
  if (op_name == "Abs") {
    // Use fabs for floats, type-specific abs for integers, abs_complex for complex
    switch (in_dtype.val()) {
      case Dtype::Val::int8:
        opencl_func = "abs_int8";
        break;
      case Dtype::Val::int16:
        opencl_func = "abs_int16";
        break;
      case Dtype::Val::int32:
        opencl_func = "abs_int32";
        break;
      case Dtype::Val::int64:
        opencl_func = "abs_int64";
        break;
      case Dtype::Val::uint8:
      case Dtype::Val::bool_:
        opencl_func = "abs_uint8";
        break;
      case Dtype::Val::uint16:
        opencl_func = "abs_uint16";
        break;
      case Dtype::Val::bfloat16:
        opencl_func = "abs_bfloat16";  // Clear sign bit for bfloat16
        break;
      case Dtype::Val::uint32:
        opencl_func = "abs_uint32";
        break;
      case Dtype::Val::uint64:
        opencl_func = "abs_uint64";
        break;
      case Dtype::Val::complex64:
        opencl_func = "abs_complex";  // Return magnitude sqrt(x^2 + y^2)
        break;
      default:
        // Keep fabs for float types
        break;
    }
  } else if (op_name == "Square" && in_dtype == complex64) {
    // Complex square: (a + bi)^2 = (a^2 - b^2) + 2abi
    opencl_func = "square_complex";
  } else if (in_dtype == complex64) {
    // Complex-specific functions for unary operations
    if (op_name == "Negative") {
      opencl_func = "neg_complex";
    } else if (op_name == "Exp") {
      opencl_func = "exp_complex";
    } else if (op_name == "Sin") {
      opencl_func = "sin_complex";
    } else if (op_name == "Cos") {
      opencl_func = "cos_complex";
    } else if (op_name == "Tan") {
      opencl_func = "tan_complex";
    } else if (op_name == "Sinh") {
      opencl_func = "sinh_complex";
    } else if (op_name == "Cosh") {
      opencl_func = "cosh_complex";
    } else if (op_name == "Tanh") {
      opencl_func = "tanh_complex";
    } else if (op_name == "Log") {
      opencl_func = "log_complex";
    } else if (op_name == "Log2") {
      opencl_func = "log2_complex";
    } else if (op_name == "Log10") {
      opencl_func = "log10_complex";
    } else if (op_name == "Log1p") {
      opencl_func = "log1p_complex";
    } else if (op_name == "Sqrt") {
      opencl_func = "sqrt_complex";
    } else if (op_name == "ArcSin") {
      opencl_func = "arcsin_complex";
    } else if (op_name == "ArcCos") {
      opencl_func = "arccos_complex";
    } else if (op_name == "ArcTan") {
      opencl_func = "arctan_complex";
    } else if (op_name == "Rsqrt") {
      opencl_func = "rsqrt_complex";
    }
  }

  // Handle Round for integer types (they're already rounded, use identity)
  if (op_name == "Round" && (issubdtype(in_dtype, integer) || in_dtype == bool_)) {
    opencl_func = "identity";
  }

  // Handle Sign for all types - use our own function to avoid OpenCL built-in conflict
  if (op_name == "Sign") {
    if (in_dtype == complex64) {
      opencl_func = "sign_complex";
    } else {
      opencl_func = "mlx_sign";
    }
  }

  source += opencl::get_template_definition(
      kernel_name, template_type, in_type, out_type, opencl_func, work_per_thread);

  // Debug: Print kernel source
  OPENCL_DEBUG_LOG("[KERNEL_SOURCE_DEBUG] Generated kernel source for " << kernel_name << ":\n" << source);

  return d.get_kernel(kernel_name, source, "");
}

} // anonymous namespace

void unary_op_gpu_inplace(
    const std::vector<array>& inputs,
    array& out,
    const std::string& op_name,
    const Stream& s) {

  OPENCL_DEBUG_LOG("[UNARY_DEBUG] Starting unary_op_gpu_inplace for " << op_name);

  auto& in = inputs[0];
  bool contig = in.flags().contiguous;

  if (in.size() == 0) {
    OPENCL_DEBUG_LOG("[UNARY_DEBUG] Empty array, returning");
    return;
  }

  OPENCL_DEBUG_LOG("[UNARY_DEBUG] Getting device...");
  auto& d = opencl::device(s.device);
  OPENCL_DEBUG_LOG("[UNARY_DEBUG] Got device");

  // Determine kernel variant
  std::string kernel_name;
  // Always use 3D work sizes
  size_t global_size[3] = {1, 1, 1};  // Initialize all dimensions to 1
  size_t local_size[3] = {64, 1, 1};  // Smaller work group size for compatibility
  int work_dim = 3;

  // Calculate work-per-thread for contiguous operations
  int work_per_thread = 1;
  std::string type_name_for_kernel = opencl::type_to_name(in.dtype());

  if (contig) {
    // Contiguous: use simple vector kernel with work-per-thread optimization
    work_per_thread = opencl::get_work_per_thread(in.dtype(), in.data_size());
    kernel_name = "v";
    if (work_per_thread > 1) kernel_name += "n";
    kernel_name += "_" + op_name + "_" + type_name_for_kernel;

    // Calculate grid size
    size_t num_elements = in.data_size();
    size_t num_threads = opencl::ceildiv(num_elements, static_cast<size_t>(work_per_thread));
    if (num_threads <= local_size[0]) {
      global_size[0] = num_threads;  // Don't round up for small sizes
      local_size[0] = 1;  // Use minimal work group size
    } else {
      global_size[0] = ((num_threads + local_size[0] - 1) / local_size[0]) * local_size[0];
    }
    // Keep dimensions 1 and 2 as 1
    OPENCL_DEBUG_LOG("[UNARY_DEBUG] Kernel config: num_elements=" << num_elements
              << ", work_per_thread=" << work_per_thread
              << ", num_threads=" << num_threads
              << ", local_size=" << local_size[0]
              << ", global_size=" << global_size[0]);
  } else {
    // Non-contiguous: use general strided kernel (no work-per-thread for now)
    kernel_name = "g_" + op_name + "_" + type_name_for_kernel;
    global_size[0] = (out.size() + local_size[0] - 1) / local_size[0] * local_size[0];
    // Keep dimensions 1 and 2 as 1
  }

  OPENCL_DEBUG_LOG("[UNARY_DEBUG] Kernel name: " << kernel_name);

  // Get or compile kernel
  OPENCL_DEBUG_LOG("[UNARY_DEBUG] Calling get_unary_kernel (work_per_thread=" << work_per_thread << ")...");
  cl_kernel kernel = get_unary_kernel(d, kernel_name, in.dtype(), out.dtype(), op_name, work_per_thread);
  OPENCL_DEBUG_LOG("[UNARY_DEBUG] Got kernel: " << kernel);

  // Get command encoder
  OPENCL_DEBUG_LOG("[UNARY_DEBUG] Getting command encoder for stream " << s.index);
  auto& compute_encoder = d.get_command_encoder(s.index);
  OPENCL_DEBUG_LOG("[UNARY_DEBUG] Got command encoder, setting kernel...");
  compute_encoder.set_kernel(kernel);
  OPENCL_DEBUG_LOG("[UNARY_DEBUG] Kernel set");

  // Set kernel arguments
  OPENCL_DEBUG_LOG("[UNARY_DEBUG] Setting input array at arg 0");
  compute_encoder.set_input_array(in, 0);
  OPENCL_DEBUG_LOG("[UNARY_DEBUG] Setting output array at arg 1");
  compute_encoder.set_output_array(out, 1);

  if (contig) {
    // Contiguous kernel: pass input offset, output offset, and size
    int64_t in_offset = static_cast<int64_t>(in.offset()) / in.itemsize();
    int64_t out_offset = static_cast<int64_t>(out.offset()) / out.itemsize();
    OPENCL_DEBUG_LOG("[UNARY_DEBUG] Setting in_offset: " << in_offset << " at arg 2");
    compute_encoder.set_bytes(in_offset, 2);
    OPENCL_DEBUG_LOG("[UNARY_DEBUG] Setting out_offset: " << out_offset << " at arg 3");
    compute_encoder.set_bytes(out_offset, 3);
    int data_size = static_cast<int>(in.data_size());
    OPENCL_DEBUG_LOG("[UNARY_DEBUG] Setting size param: " << data_size << " at arg 4");
    compute_encoder.set_bytes(data_size, 4);

    // Dispatch kernel
    OPENCL_DEBUG_LOG("[UNARY_DEBUG] Dispatching kernel...");
    compute_encoder.dispatch_threads(global_size, local_size, work_dim);
  } else {
    // General kernel: pass shape, strides, ndim, in_offset, out_offset, size as proper buffers
    std::vector<int> shape_vec(in.shape().begin(), in.shape().end());
    std::vector<int64_t> strides_vec(in.strides().begin(), in.strides().end());

    int ndim = in.ndim();
    int size = out.size();

    // Handle scalar case (ndim=0)
    if (shape_vec.empty()) shape_vec.push_back(1);
    if (strides_vec.empty()) strides_vec.push_back(0);

    OPENCL_DEBUG_LOG("[UNARY_DEBUG] General unary: ndim=" << ndim << " size=" << size);

    // Create constant memory buffers for shape and strides
    cl_int err;
    cl_mem shape_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       shape_vec.size() * sizeof(int), shape_vec.data(), &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("[unary_op_gpu] Failed to create shape buffer: " + std::to_string(err));
    }

    cl_mem strides_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         strides_vec.size() * sizeof(int64_t), strides_vec.data(), &err);
    if (err != CL_SUCCESS) {
      clReleaseMemObject(shape_buf);
      throw std::runtime_error("[unary_op_gpu] Failed to create strides buffer: " + std::to_string(err));
    }

    // Set kernel arguments: shape, strides, ndim, in_offset, out_offset, size
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &shape_buf);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &strides_buf);
    compute_encoder.set_bytes(ndim, 4);
    int64_t in_offset = static_cast<int64_t>(in.offset()) / in.itemsize();
    int64_t out_offset = static_cast<int64_t>(out.offset()) / out.itemsize();
    compute_encoder.set_bytes(in_offset, 5);
    compute_encoder.set_bytes(out_offset, 6);
    compute_encoder.set_bytes(size, 7);

    // Dispatch kernel
    OPENCL_DEBUG_LOG("[UNARY_DEBUG] Dispatching kernel...");
    compute_encoder.dispatch_threads(global_size, local_size, work_dim);

    // Track buffers for cleanup after sync
    d.add_temp_buffer(shape_buf, s.index);
    d.add_temp_buffer(strides_buf, s.index);

    // Release temp buffers after kernel execution completes
    d.end_encoding(s.index);
  }
}

void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string& op_name,
    const Stream& s) {
  // Use OpenCL allocator to allocate output buffer
  auto malloc_fn = [](size_t size) -> allocator::Buffer {
    return opencl::allocator().malloc(size);
  };
  set_unary_output_data(inputs[0], out, malloc_fn);
  unary_op_gpu_inplace(inputs, out, op_name, s);
}

void unary_op_gpu(
    const std::vector<array>& inputs,
    array& out,
    const std::string& op_name) {
  auto& s = out.primitive().stream();
  unary_op_gpu(inputs, out, op_name, s);
}

// Register all unary operations
UNARY_GPU(Abs)
UNARY_GPU(ArcCos)
UNARY_GPU(ArcCosh)
UNARY_GPU(ArcSin)
UNARY_GPU(ArcSinh)
UNARY_GPU(ArcTan)
UNARY_GPU(ArcTanh)
UNARY_GPU(BitwiseInvert)
UNARY_GPU(Ceil)
UNARY_GPU(Conjugate)
UNARY_GPU(Cos)
UNARY_GPU(Cosh)
UNARY_GPU(Erf)
UNARY_GPU(ErfInv)
UNARY_GPU(Exp)
UNARY_GPU(Expm1)
UNARY_GPU(Floor)
UNARY_GPU(Imag)
UNARY_GPU(Log1p)
UNARY_GPU(LogicalNot)
UNARY_GPU(Negative)
UNARY_GPU(Real)
UNARY_GPU(Sigmoid)
UNARY_GPU(Sign)
UNARY_GPU(Sin)
UNARY_GPU(Sinh)
UNARY_GPU(Square)
UNARY_GPU(Sqrt)
UNARY_GPU(Tan)
UNARY_GPU(Tanh)

// Log requires special handling due to base parameter
void Log::eval_gpu(const std::vector<array>& inputs, array& out) {
  std::string op_name;
  switch (base_) {
    case Base::two:
      op_name = "Log2";
      break;
    case Base::ten:
      op_name = "Log10";
      break;
    case Base::e:
      op_name = "Log";
      break;
  }
  unary_op_gpu(inputs, out, op_name);
}

// Round - simple rounding to nearest integer
UNARY_GPU(Round)

} // namespace mlx::core
