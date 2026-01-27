// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/utils.h"

namespace mlx::core::opencl {

std::string get_kernel_preamble() {
  return R"(
// OpenCL kernel preamble
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Type definitions
typedef char int8_t;
typedef uchar uint8_t;
typedef short int16_t;
typedef ushort uint16_t;
typedef int int32_t;
typedef uint uint32_t;
typedef long int64_t;
typedef ulong uint64_t;

// bfloat16 type definition and conversion helpers
typedef uint16_t bfloat16_t;

// Convert bfloat16 (stored as uint16) to float
inline float bfloat16_to_float(bfloat16_t val) {
  uint32_t bits = ((uint32_t)val) << 16;
  return as_float(bits);
}

// Convert float to bfloat16 (upper 16 bits of float32 with rounding)
inline bfloat16_t float_to_bfloat16(float val) {
  uint32_t bits = as_uint(val);
  bits += 0x7fff + ((bits >> 16) & 1);
  return (bfloat16_t)(bits >> 16);
}

// Helper functions for index calculations
inline int64_t elem_to_loc_1(int elem, int64_t stride_0) {
  return elem * stride_0;
}

inline int64_t elem_to_loc_2(
    int elem,
    __constant const int* shape,
    __constant const int64_t* strides) {
  int64_t loc = 0;
  for (int i = 1; i >= 0; i--) {
    int pos = elem % shape[i];
    elem /= shape[i];
    loc += pos * strides[i];
  }
  return loc;
}

inline int64_t elem_to_loc(
    int elem,
    __constant const int* shape,
    __constant const int64_t* strides,
    int ndim) {
  int64_t loc = 0;
  for (int i = ndim - 1; i >= 0; i--) {
    int pos = elem % shape[i];
    elem /= shape[i];
    loc += pos * strides[i];
  }
  return loc;
}

// Broadcasting helpers
inline int64_t elem_to_loc_broadcast(
    int elem,
    __constant const int* shape,
    __constant const int64_t* strides,
    int ndim) {
  int64_t loc = 0;
  for (int i = ndim - 1; i >= 0; i--) {
    int pos = elem % shape[i];
    elem /= shape[i];
    if (shape[i] != 1) {
      loc += pos * strides[i];
    }
  }
  return loc;
}

)";
}

std::string get_template_definition(
    const std::string& kernel_name,
    const std::string& template_name,
    const std::string& input_type,
    const std::string& output_type,
    const std::string& operation,
    int work_per_thread) {
  std::ostringstream kernel_source;

  kernel_source << "// Specialized kernel: " << kernel_name << " (work_per_thread=" << work_per_thread << ")\n";
  kernel_source << "#define INSTANTIATE_" << template_name
                << "(name, itype, otype, op) \\\n";
  kernel_source << "__kernel void name( \\\n";

  // Check if we need bfloat16 conversions
  bool is_bf16_input = (input_type == "bfloat16_t");
  bool is_bf16_output = (output_type == "bfloat16_t");

  // Shorthand for N
  int N = work_per_thread;

  if (template_name.find("unary") == 0) {
    kernel_source << "  __global const itype* input, \\\n";
    kernel_source << "  __global otype* output, \\\n";

    if (template_name == "unary_v") {
      kernel_source << "  long in_offset, \\\n";
      kernel_source << "  long out_offset, \\\n";
      kernel_source << "  int size) { \\\n";
      kernel_source << "  int tid = get_global_id(0); \\\n";
      if (N > 1) {
        // Work-per-thread pattern
        kernel_source << "  int id = tid * " << N << "; \\\n";
        kernel_source << "  if (" << N << " > 1 && id + " << N << " > size) { \\\n";
        kernel_source << "    for (int i = 0; id + i < size; ++i) { \\\n";
        if (is_bf16_input && is_bf16_output) {
          kernel_source << "      output[out_offset + id + i] = float_to_bfloat16(op(bfloat16_to_float(input[in_offset + id + i]))); \\\n";
        } else if (is_bf16_input) {
          kernel_source << "      output[out_offset + id + i] = (otype)op(bfloat16_to_float(input[in_offset + id + i])); \\\n";
        } else if (is_bf16_output) {
          kernel_source << "      output[out_offset + id + i] = float_to_bfloat16((float)op(input[in_offset + id + i])); \\\n";
        } else {
          kernel_source << "      output[out_offset + id + i] = op(input[in_offset + id + i]); \\\n";
        }
        kernel_source << "    } \\\n";
        kernel_source << "  } else { \\\n";
        kernel_source << "    for (int i = 0; i < " << N << "; ++i) { \\\n";
        if (is_bf16_input && is_bf16_output) {
          kernel_source << "      output[out_offset + id + i] = float_to_bfloat16(op(bfloat16_to_float(input[in_offset + id + i]))); \\\n";
        } else if (is_bf16_input) {
          kernel_source << "      output[out_offset + id + i] = (otype)op(bfloat16_to_float(input[in_offset + id + i])); \\\n";
        } else if (is_bf16_output) {
          kernel_source << "      output[out_offset + id + i] = float_to_bfloat16((float)op(input[in_offset + id + i])); \\\n";
        } else {
          kernel_source << "      output[out_offset + id + i] = op(input[in_offset + id + i]); \\\n";
        }
        kernel_source << "    } \\\n";
        kernel_source << "  } \\\n";
      } else {
        // Original single-element-per-thread pattern
        kernel_source << "  if (tid < size) { \\\n";
        if (is_bf16_input && is_bf16_output) {
          kernel_source << "    output[out_offset + tid] = float_to_bfloat16(op(bfloat16_to_float(input[in_offset + tid]))); \\\n";
        } else if (is_bf16_input) {
          kernel_source << "    output[out_offset + tid] = (otype)op(bfloat16_to_float(input[in_offset + tid])); \\\n";
        } else if (is_bf16_output) {
          kernel_source << "    output[out_offset + tid] = float_to_bfloat16((float)op(input[in_offset + tid])); \\\n";
        } else {
          kernel_source << "    output[out_offset + tid] = op(input[in_offset + tid]); \\\n";
        }
        kernel_source << "  } \\\n";
      }
      kernel_source << "}\n";
    } else if (template_name == "unary_g") {
      kernel_source << "  __constant const int* shape, \\\n";
      kernel_source << "  __constant const int64_t* strides, \\\n";
      kernel_source << "  int ndim, \\\n";
      kernel_source << "  long in_offset, \\\n";
      kernel_source << "  long out_offset, \\\n";
      kernel_source << "  int size) { \\\n";
      kernel_source << "  int id = get_global_id(0); \\\n";
      kernel_source << "  if (id < size) { \\\n";
      kernel_source << "    int64_t idx = in_offset + elem_to_loc(id, shape, strides, ndim); \\\n";
      if (is_bf16_input && is_bf16_output) {
        // bfloat16 -> float -> op -> bfloat16
        kernel_source << "    output[out_offset + id] = float_to_bfloat16(op(bfloat16_to_float(input[idx]))); \\\n";
      } else if (is_bf16_input) {
        // bfloat16 input, other output
        kernel_source << "    output[out_offset + id] = (otype)op(bfloat16_to_float(input[idx])); \\\n";
      } else if (is_bf16_output) {
        // other input, bfloat16 output
        kernel_source << "    output[out_offset + id] = float_to_bfloat16((float)op(input[idx])); \\\n";
      } else {
        kernel_source << "    output[out_offset + id] = op(input[idx]); \\\n";
      }
      kernel_source << "  } \\\n";
      kernel_source << "}\n";
    }
  } else if (template_name.find("binary") == 0) {
    kernel_source << "  __global const itype* input_a, \\\n";
    kernel_source << "  __global const itype* input_b, \\\n";
    kernel_source << "  __global otype* output, \\\n";

    // Helper macros for bfloat16 conversion in binary ops
    std::string get_a, get_b, set_out;
    if (is_bf16_input) {
      get_a = "bfloat16_to_float(input_a[%s])";
      get_b = "bfloat16_to_float(input_b[%s])";
    } else {
      get_a = "input_a[%s]";
      get_b = "input_b[%s]";
    }
    if (is_bf16_output) {
      set_out = "float_to_bfloat16(";
    }

    if (template_name == "binary_ss") {
      // Scalar-Scalar: both inputs are scalars
      kernel_source << "  int size, \\\n";
      kernel_source << "  long offset_a, \\\n";
      kernel_source << "  long offset_b, \\\n";
      kernel_source << "  long offset_out) { \\\n";
      kernel_source << "  int id = get_global_id(0); \\\n";
      kernel_source << "  if (id < size) { \\\n";
      if (is_bf16_input && is_bf16_output) {
        kernel_source << "    output[offset_out + id] = float_to_bfloat16(op(bfloat16_to_float(input_a[offset_a]), bfloat16_to_float(input_b[offset_b]))); \\\n";
      } else if (is_bf16_input) {
        kernel_source << "    output[offset_out + id] = (otype)op(bfloat16_to_float(input_a[offset_a]), bfloat16_to_float(input_b[offset_b])); \\\n";
      } else if (is_bf16_output) {
        kernel_source << "    output[offset_out + id] = float_to_bfloat16((float)op(input_a[offset_a], input_b[offset_b])); \\\n";
      } else {
        kernel_source << "    output[offset_out + id] = op(input_a[offset_a], input_b[offset_b]); \\\n";
      }
      kernel_source << "  } \\\n";
      kernel_source << "}\n";
    } else if (template_name == "binary_vv") {
      kernel_source << "  int size, \\\n";
      kernel_source << "  long offset_a, \\\n";
      kernel_source << "  long offset_b, \\\n";
      kernel_source << "  long offset_out) { \\\n";
      kernel_source << "  int tid = get_global_id(0); \\\n";
      if (N > 1) {
        // Work-per-thread pattern: each thread processes N elements
        kernel_source << "  int id = tid * " << N << "; \\\n";
        kernel_source << "  if (" << N << " > 1 && id + " << N << " > size) { \\\n";
        kernel_source << "    for (int i = 0; id + i < size; ++i) { \\\n";
        if (is_bf16_input && is_bf16_output) {
          kernel_source << "      output[offset_out + id + i] = float_to_bfloat16(op(bfloat16_to_float(input_a[offset_a + id + i]), bfloat16_to_float(input_b[offset_b + id + i]))); \\\n";
        } else if (is_bf16_input) {
          kernel_source << "      output[offset_out + id + i] = (otype)op(bfloat16_to_float(input_a[offset_a + id + i]), bfloat16_to_float(input_b[offset_b + id + i])); \\\n";
        } else if (is_bf16_output) {
          kernel_source << "      output[offset_out + id + i] = float_to_bfloat16((float)op(input_a[offset_a + id + i], input_b[offset_b + id + i])); \\\n";
        } else {
          kernel_source << "      output[offset_out + id + i] = op(input_a[offset_a + id + i], input_b[offset_b + id + i]); \\\n";
        }
        kernel_source << "    } \\\n";
        kernel_source << "  } else { \\\n";
        kernel_source << "    for (int i = 0; i < " << N << "; ++i) { \\\n";
        if (is_bf16_input && is_bf16_output) {
          kernel_source << "      output[offset_out + id + i] = float_to_bfloat16(op(bfloat16_to_float(input_a[offset_a + id + i]), bfloat16_to_float(input_b[offset_b + id + i]))); \\\n";
        } else if (is_bf16_input) {
          kernel_source << "      output[offset_out + id + i] = (otype)op(bfloat16_to_float(input_a[offset_a + id + i]), bfloat16_to_float(input_b[offset_b + id + i])); \\\n";
        } else if (is_bf16_output) {
          kernel_source << "      output[offset_out + id + i] = float_to_bfloat16((float)op(input_a[offset_a + id + i], input_b[offset_b + id + i])); \\\n";
        } else {
          kernel_source << "      output[offset_out + id + i] = op(input_a[offset_a + id + i], input_b[offset_b + id + i]); \\\n";
        }
        kernel_source << "    } \\\n";
        kernel_source << "  } \\\n";
      } else {
        // Original single-element-per-thread pattern
        kernel_source << "  if (tid < size) { \\\n";
        if (is_bf16_input && is_bf16_output) {
          kernel_source << "    output[offset_out + tid] = float_to_bfloat16(op(bfloat16_to_float(input_a[offset_a + tid]), bfloat16_to_float(input_b[offset_b + tid]))); \\\n";
        } else if (is_bf16_input) {
          kernel_source << "    output[offset_out + tid] = (otype)op(bfloat16_to_float(input_a[offset_a + tid]), bfloat16_to_float(input_b[offset_b + tid])); \\\n";
        } else if (is_bf16_output) {
          kernel_source << "    output[offset_out + tid] = float_to_bfloat16((float)op(input_a[offset_a + tid], input_b[offset_b + tid])); \\\n";
        } else {
          kernel_source << "    output[offset_out + tid] = op(input_a[offset_a + tid], input_b[offset_b + tid]); \\\n";
        }
        kernel_source << "  } \\\n";
      }
      kernel_source << "}\n";
    } else if (template_name == "binary_sv") {
      // Scalar-Vector: a is scalar, b is vector
      kernel_source << "  int size, \\\n";
      kernel_source << "  long offset_a, \\\n";
      kernel_source << "  long offset_b, \\\n";
      kernel_source << "  long offset_out) { \\\n";
      kernel_source << "  int tid = get_global_id(0); \\\n";
      if (N > 1) {
        // Work-per-thread pattern
        kernel_source << "  int id = tid * " << N << "; \\\n";
        // Load scalar value once
        if (is_bf16_input) {
          kernel_source << "  float scalar_a = bfloat16_to_float(input_a[offset_a]); \\\n";
        } else {
          kernel_source << "  itype scalar_a = input_a[offset_a]; \\\n";
        }
        kernel_source << "  if (" << N << " > 1 && id + " << N << " > size) { \\\n";
        kernel_source << "    for (int i = 0; id + i < size; ++i) { \\\n";
        if (is_bf16_input && is_bf16_output) {
          kernel_source << "      output[offset_out + id + i] = float_to_bfloat16(op(scalar_a, bfloat16_to_float(input_b[offset_b + id + i]))); \\\n";
        } else if (is_bf16_input) {
          kernel_source << "      output[offset_out + id + i] = (otype)op(scalar_a, bfloat16_to_float(input_b[offset_b + id + i])); \\\n";
        } else if (is_bf16_output) {
          kernel_source << "      output[offset_out + id + i] = float_to_bfloat16((float)op(scalar_a, input_b[offset_b + id + i])); \\\n";
        } else {
          kernel_source << "      output[offset_out + id + i] = op(scalar_a, input_b[offset_b + id + i]); \\\n";
        }
        kernel_source << "    } \\\n";
        kernel_source << "  } else { \\\n";
        kernel_source << "    for (int i = 0; i < " << N << "; ++i) { \\\n";
        if (is_bf16_input && is_bf16_output) {
          kernel_source << "      output[offset_out + id + i] = float_to_bfloat16(op(scalar_a, bfloat16_to_float(input_b[offset_b + id + i]))); \\\n";
        } else if (is_bf16_input) {
          kernel_source << "      output[offset_out + id + i] = (otype)op(scalar_a, bfloat16_to_float(input_b[offset_b + id + i])); \\\n";
        } else if (is_bf16_output) {
          kernel_source << "      output[offset_out + id + i] = float_to_bfloat16((float)op(scalar_a, input_b[offset_b + id + i])); \\\n";
        } else {
          kernel_source << "      output[offset_out + id + i] = op(scalar_a, input_b[offset_b + id + i]); \\\n";
        }
        kernel_source << "    } \\\n";
        kernel_source << "  } \\\n";
      } else {
        kernel_source << "  if (tid < size) { \\\n";
        if (is_bf16_input && is_bf16_output) {
          kernel_source << "    output[offset_out + tid] = float_to_bfloat16(op(bfloat16_to_float(input_a[offset_a]), bfloat16_to_float(input_b[offset_b + tid]))); \\\n";
        } else if (is_bf16_input) {
          kernel_source << "    output[offset_out + tid] = (otype)op(bfloat16_to_float(input_a[offset_a]), bfloat16_to_float(input_b[offset_b + tid])); \\\n";
        } else if (is_bf16_output) {
          kernel_source << "    output[offset_out + tid] = float_to_bfloat16((float)op(input_a[offset_a], input_b[offset_b + tid])); \\\n";
        } else {
          kernel_source << "    output[offset_out + tid] = op(input_a[offset_a], input_b[offset_b + tid]); \\\n";
        }
        kernel_source << "  } \\\n";
      }
      kernel_source << "}\n";
    } else if (template_name == "binary_vs") {
      // Vector-Scalar: a is vector, b is scalar
      kernel_source << "  int size, \\\n";
      kernel_source << "  long offset_a, \\\n";
      kernel_source << "  long offset_b, \\\n";
      kernel_source << "  long offset_out) { \\\n";
      kernel_source << "  int tid = get_global_id(0); \\\n";
      if (N > 1) {
        // Work-per-thread pattern
        kernel_source << "  int id = tid * " << N << "; \\\n";
        // Load scalar value once
        if (is_bf16_input) {
          kernel_source << "  float scalar_b = bfloat16_to_float(input_b[offset_b]); \\\n";
        } else {
          kernel_source << "  itype scalar_b = input_b[offset_b]; \\\n";
        }
        kernel_source << "  if (" << N << " > 1 && id + " << N << " > size) { \\\n";
        kernel_source << "    for (int i = 0; id + i < size; ++i) { \\\n";
        if (is_bf16_input && is_bf16_output) {
          kernel_source << "      output[offset_out + id + i] = float_to_bfloat16(op(bfloat16_to_float(input_a[offset_a + id + i]), scalar_b)); \\\n";
        } else if (is_bf16_input) {
          kernel_source << "      output[offset_out + id + i] = (otype)op(bfloat16_to_float(input_a[offset_a + id + i]), scalar_b); \\\n";
        } else if (is_bf16_output) {
          kernel_source << "      output[offset_out + id + i] = float_to_bfloat16((float)op(input_a[offset_a + id + i], scalar_b)); \\\n";
        } else {
          kernel_source << "      output[offset_out + id + i] = op(input_a[offset_a + id + i], scalar_b); \\\n";
        }
        kernel_source << "    } \\\n";
        kernel_source << "  } else { \\\n";
        kernel_source << "    for (int i = 0; i < " << N << "; ++i) { \\\n";
        if (is_bf16_input && is_bf16_output) {
          kernel_source << "      output[offset_out + id + i] = float_to_bfloat16(op(bfloat16_to_float(input_a[offset_a + id + i]), scalar_b)); \\\n";
        } else if (is_bf16_input) {
          kernel_source << "      output[offset_out + id + i] = (otype)op(bfloat16_to_float(input_a[offset_a + id + i]), scalar_b); \\\n";
        } else if (is_bf16_output) {
          kernel_source << "      output[offset_out + id + i] = float_to_bfloat16((float)op(input_a[offset_a + id + i], scalar_b)); \\\n";
        } else {
          kernel_source << "      output[offset_out + id + i] = op(input_a[offset_a + id + i], scalar_b); \\\n";
        }
        kernel_source << "    } \\\n";
        kernel_source << "  } \\\n";
      } else {
        kernel_source << "  if (tid < size) { \\\n";
        if (is_bf16_input && is_bf16_output) {
          kernel_source << "    output[offset_out + tid] = float_to_bfloat16(op(bfloat16_to_float(input_a[offset_a + tid]), bfloat16_to_float(input_b[offset_b]))); \\\n";
        } else if (is_bf16_input) {
          kernel_source << "    output[offset_out + tid] = (otype)op(bfloat16_to_float(input_a[offset_a + tid]), bfloat16_to_float(input_b[offset_b])); \\\n";
        } else if (is_bf16_output) {
          kernel_source << "    output[offset_out + tid] = float_to_bfloat16((float)op(input_a[offset_a + tid], input_b[offset_b])); \\\n";
        } else {
          kernel_source << "    output[offset_out + tid] = op(input_a[offset_a + tid], input_b[offset_b]); \\\n";
        }
        kernel_source << "  } \\\n";
      }
      kernel_source << "}\n";
    } else if (template_name == "binary_g") {
      kernel_source << "  __constant const int* shape, \\\n";
      kernel_source << "  __constant const int64_t* strides_a, \\\n";
      kernel_source << "  __constant const int64_t* strides_b, \\\n";
      kernel_source << "  int ndim, \\\n";
      kernel_source << "  int size, \\\n";
      kernel_source << "  long offset_a, \\\n";
      kernel_source << "  long offset_b, \\\n";
      kernel_source << "  long offset_out) { \\\n";
      kernel_source << "  int id = get_global_id(0); \\\n";
      kernel_source << "  if (id < size) { \\\n";
      kernel_source << "    int64_t idx_a = offset_a + elem_to_loc_broadcast(id, shape, strides_a, ndim); \\\n";
      kernel_source << "    int64_t idx_b = offset_b + elem_to_loc_broadcast(id, shape, strides_b, ndim); \\\n";
      if (is_bf16_input && is_bf16_output) {
        kernel_source << "    output[offset_out + id] = float_to_bfloat16(op(bfloat16_to_float(input_a[idx_a]), bfloat16_to_float(input_b[idx_b]))); \\\n";
      } else if (is_bf16_input) {
        kernel_source << "    output[offset_out + id] = (otype)op(bfloat16_to_float(input_a[idx_a]), bfloat16_to_float(input_b[idx_b])); \\\n";
      } else if (is_bf16_output) {
        kernel_source << "    output[offset_out + id] = float_to_bfloat16((float)op(input_a[idx_a], input_b[idx_b])); \\\n";
      } else {
        kernel_source << "    output[offset_out + id] = op(input_a[idx_a], input_b[idx_b]); \\\n";
      }
      kernel_source << "  } \\\n";
      kernel_source << "}\n";
    }
  }

  // Instantiate the template with specific types and operation
  kernel_source << "\nINSTANTIATE_" << template_name << "("
                << kernel_name << ", "
                << input_type << ", "
                << output_type << ", "
                << operation << ")\n";

  return kernel_source.str();
}

// Note: type_to_name() moved to types.cpp

} // namespace mlx::core::opencl
