// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/copy.h"
#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/primitives.h"
#include "mlx/backend/common/utils.h"

#include <sstream>

namespace mlx::core {

namespace {

constexpr int OPENCL_MAX_INDEX_ARRAYS = 10;

// Generate kernel for gather_axis operation
std::string get_gather_axis_kernel_source(
    const std::string& kernel_name,
    const std::string& src_type,
    const std::string& idx_type,
    bool src_contiguous,
    bool idx_contiguous) {

  std::ostringstream src;
  src << opencl::get_kernel_preamble();

  // Helper for negative index handling
  src << R"(
inline long offset_neg_idx(long idx, int size) {
  return (idx < 0) ? idx + size : idx;
}
)";

  src << "__kernel void " << kernel_name << "(\n";
  src << "    __global const " << src_type << "* src,\n";
  src << "    __global const " << idx_type << "* idx,\n";
  src << "    __global " << src_type << "* out,\n";
  src << "    __global const int* shape,\n";  // shape without axis (for non-axis dimensions)
  src << "    __global const long* src_strides,\n";  // src strides without axis
  src << "    __global const long* idx_strides,\n";  // idx strides without axis
  src << "    int ndim,\n";  // ndim - 1 (number of non-axis dimensions)
  src << "    int axis,\n";
  src << "    int src_axis_size,\n";  // size of source along axis
  src << "    long src_axis_stride,\n";  // src stride along axis
  src << "    long idx_axis_stride,\n";  // idx stride along axis
  src << "    int size_post,\n";  // product of dims after axis
  src << "    long src_base_offset,\n";  // base offset for source array
  src << "    long idx_base_offset) {\n";  // base offset for index array

  // Grid: [size_post, idx_axis_size, size_pre]
  src << "  int post = get_global_id(0);\n";
  src << "  int j = get_global_id(1);\n";  // position along axis in index
  src << "  int pre = get_global_id(2);\n";

  src << "  int idx_axis_size = get_global_size(1);\n";
  src << "  int size_pre = get_global_size(2);\n";

  // Compute the multi-dimensional coordinate in non-axis space
  // Linear index in non-axis space = pre * size_post + post
  src << "  long src_off = 0;\n";
  src << "  long idx_off = 0;\n";

  // For the non-axis dimensions, we need to compute the strided offset
  // The shape and strides arrays contain non-axis dimensions only
  // We always compute strided offsets since even contiguous arrays may have
  // different strides (e.g., when src and idx have different shapes)
  src << "  if (ndim > 0) {\n";
  src << "    int linear_nonaxis = pre * size_post + post;\n";
  src << "    int tmp = linear_nonaxis;\n";
  src << "    for (int i = ndim - 1; i >= 0; i--) {\n";
  src << "      int coord = tmp % shape[i];\n";
  src << "      tmp /= shape[i];\n";
  src << "      src_off += coord * src_strides[i];\n";
  src << "      idx_off += coord * idx_strides[i];\n";
  src << "    }\n";
  src << "  }\n";

  // Get index value along axis - add idx_base_offset for sliced arrays
  src << "  long idx_loc = idx_off + (long)j * idx_axis_stride + idx_base_offset;\n";
  src << "  long idx_val = offset_neg_idx(idx[idx_loc], src_axis_size);\n";

  // Read from source
  src << "  long src_loc = src_off + idx_val * src_axis_stride + src_base_offset;\n";

  // Calculate output linear index (output is row-contiguous)
  // Layout: [pre, j, post] -> index = pre * (idx_axis_size * size_post) + j * size_post + post
  src << "  long out_idx = (long)pre * (long)idx_axis_size * (long)size_post + (long)j * (long)size_post + (long)post;\n";
  src << "  out[out_idx] = src[src_loc];\n";
  src << "}\n";

  return src.str();
}

// Generate kernel for gather operation (multi-index)
std::string get_gather_kernel_source(
    const std::string& kernel_name,
    const std::string& src_type,
    const std::string& idx_type,
    int nidx,
    int idx_ndim) {

  std::ostringstream src;
  src << opencl::get_kernel_preamble();

  // Helper for negative index handling
  src << R"(
inline long offset_neg_idx(long idx, int size) {
  return (idx < 0) ? idx + size : idx;
}
)";

  src << "__kernel void " << kernel_name << "(\n";
  src << "    __global const " << src_type << "* src,\n";
  src << "    __global " << src_type << "* out,\n";
  src << "    __global const int* src_shape,\n";
  src << "    __global const long* src_strides,\n";
  src << "    int src_ndim,\n";
  src << "    __global const int* slice_sizes,\n";
  src << "    __global const int* axes,\n";

  // Index arrays and their info
  for (int i = 0; i < nidx; i++) {
    src << "    __global const " << idx_type << "* idx" << i << ",\n";
  }
  src << "    __global const int* idx_shapes,\n";  // flattened for all indices
  src << "    __global const long* idx_strides,\n";  // flattened for all indices
  src << "    __global const long* idx_offsets,\n";  // base offset for each index array
  src << "    int idx_ndim,\n";
  src << "    long slice_size,\n";
  src << "    long src_base_offset) {\n";  // base offset for source array

  // Grid: [dim0 from index, dim1 from index, slice_size]
  src << "  int idx_dim0 = get_global_id(0);\n";
  src << "  int idx_dim1 = get_global_id(1);\n";
  src << "  int slice_idx = get_global_id(2);\n";

  src << "  if (slice_idx >= slice_size) return;\n";

  // Compute linear index in index array
  src << "  long idx_linear = (long)idx_dim0 * (long)get_global_size(1) + idx_dim1;\n";

  // Compute source offset from index arrays
  src << "  long src_idx = 0;\n";
  for (int i = 0; i < nidx; i++) {
    src << "  {\n";
    // Compute offset into this index array (start with base offset for sliced arrays)
    src << "    long idx_off = idx_offsets[" << i << "];\n";
    if (idx_ndim > 0) {
      src << "    long tmp = idx_linear;\n";
      src << "    for (int d = idx_ndim - 1; d >= 0; d--) {\n";
      src << "      int coord = tmp % idx_shapes[" << i << " * idx_ndim + d];\n";
      src << "      tmp /= idx_shapes[" << i << " * idx_ndim + d];\n";
      src << "      idx_off += coord * idx_strides[" << i << " * idx_ndim + d];\n";
      src << "    }\n";
    }
    src << "    long idx_val = offset_neg_idx(idx" << i << "[idx_off], src_shape[axes[" << i << "]]);\n";
    src << "    src_idx += idx_val * src_strides[axes[" << i << "]];\n";
    src << "  }\n";
  }

  // Compute offset from slice index
  src << "  long tmp = slice_idx;\n";
  src << "  for (int d = src_ndim - 1; d >= 0; d--) {\n";
  src << "    int coord = tmp % slice_sizes[d];\n";
  src << "    tmp /= slice_sizes[d];\n";
  src << "    src_idx += coord * src_strides[d];\n";
  src << "  }\n";

  // Add source base offset (for sliced/strided source arrays)
  src << "  src_idx += src_base_offset;\n";

  // Compute output index
  src << "  long out_idx = idx_linear * slice_size + slice_idx;\n";
  src << "  out[out_idx] = src[src_idx];\n";
  src << "}\n";

  return src.str();
}

// Generate kernel for scatter_axis operation
// Generate atomic operation helper functions for scatter
std::string get_atomic_helpers(const std::string& val_type) {
  std::ostringstream src;

  // Atomic helpers for sum, prod, max, min operations
  // For floats, use CAS-based atomics since OpenCL doesn't have native float atomics
  if (val_type == "float") {
    src << R"(
// Atomic float add using compare-and-swap
inline void atomic_add_float(__global float* addr, float val) {
  union { uint u; float f; } expected, desired;
  expected.f = *addr;
  while (1) {
    desired.f = expected.f + val;
    uint old = atomic_cmpxchg((__global uint*)addr, expected.u, desired.u);
    if (old == expected.u) return;  // Success
    expected.u = old;  // Update expected and retry
  }
}

// Atomic float multiply using CAS
inline void atomic_mul_float(__global float* addr, float val) {
  union { uint u; float f; } expected, desired;
  expected.f = *addr;
  while (1) {
    desired.f = expected.f * val;
    uint old = atomic_cmpxchg((__global uint*)addr, expected.u, desired.u);
    if (old == expected.u) return;  // Success
    expected.u = old;  // Update expected and retry
  }
}

// Atomic float max using CAS
inline void atomic_max_float(__global float* addr, float val) {
  union { uint u; float f; } expected, desired;
  expected.f = *addr;
  while (val > expected.f) {
    desired.f = val;
    uint old = atomic_cmpxchg((__global uint*)addr, expected.u, desired.u);
    if (old == expected.u) return;
    expected.u = old;
  }
}

// Atomic float min using CAS
inline void atomic_min_float(__global float* addr, float val) {
  union { uint u; float f; } expected, desired;
  expected.f = *addr;
  while (val < expected.f) {
    desired.f = val;
    uint old = atomic_cmpxchg((__global uint*)addr, expected.u, desired.u);
    if (old == expected.u) return;
    expected.u = old;
  }
}
)";
  } else if (val_type == "half") {
    // For half, we need to load/store as ushort and convert
    src << R"(
// Atomic half add using CAS (pack into ushort)
inline void atomic_add_half(__global half* addr, half val) {
  __global ushort* addr_u = (__global ushort*)addr;
  ushort expected_u = *addr_u;
  while (1) {
    half expected_f = as_half(expected_u);
    half desired_f = expected_f + val;
    ushort desired_u = as_ushort(desired_f);
    ushort old = atom_cmpxchg(addr_u, expected_u, desired_u);
    if (old == expected_u) return;  // Success
    expected_u = old;  // Update expected and retry
  }
}

// Atomic half multiply using CAS
inline void atomic_mul_half(__global half* addr, half val) {
  __global ushort* addr_u = (__global ushort*)addr;
  ushort expected_u = *addr_u;
  while (1) {
    half expected_f = as_half(expected_u);
    half desired_f = expected_f * val;
    ushort desired_u = as_ushort(desired_f);
    ushort old = atom_cmpxchg(addr_u, expected_u, desired_u);
    if (old == expected_u) return;  // Success
    expected_u = old;  // Update expected and retry
  }
}

// Atomic half max using CAS
inline void atomic_max_half(__global half* addr, half val) {
  __global ushort* addr_u = (__global ushort*)addr;
  ushort expected_u = *addr_u;
  while (1) {
    half expected_f = as_half(expected_u);
    if (val <= expected_f) return;
    ushort desired_u = as_ushort(val);
    ushort old = atom_cmpxchg(addr_u, expected_u, desired_u);
    if (old == expected_u) return;
    expected_u = old;
  }
}

// Atomic half min using CAS
inline void atomic_min_half(__global half* addr, half val) {
  __global ushort* addr_u = (__global ushort*)addr;
  ushort expected_u = *addr_u;
  while (1) {
    half expected_f = as_half(expected_u);
    if (val >= expected_f) return;
    ushort desired_u = as_ushort(val);
    ushort old = atom_cmpxchg(addr_u, expected_u, desired_u);
    if (old == expected_u) return;
    expected_u = old;
  }
}
)";
  }
  // For int/uint types, OpenCL has native atomics (atomic_add, atomic_min, atomic_max)

  return src.str();
}

std::string get_scatter_axis_kernel_source(
    const std::string& kernel_name,
    const std::string& val_type,
    const std::string& idx_type,
    const std::string& op_type,  // "none" or "sum"
    bool upd_contiguous,
    bool idx_contiguous) {

  std::ostringstream src;
  src << opencl::get_kernel_preamble();

  // Add atomic helpers for reduce operations
  if (op_type != "none") {
    src << get_atomic_helpers(val_type);
  }

  // Helper for negative index handling
  src << R"(
inline long offset_neg_idx(long idx, int size) {
  return (idx < 0) ? idx + size : idx;
}
)";

  src << "__kernel void " << kernel_name << "(\n";
  src << "    __global const " << val_type << "* upd,\n";
  src << "    __global const " << idx_type << "* idx,\n";
  src << "    __global " << val_type << "* out,\n";
  src << "    __global const int* shape,\n";  // shape without axis
  src << "    __global const long* upd_strides,\n";  // strides without axis
  src << "    __global const long* idx_strides,\n";  // strides without axis
  src << "    int ndim,\n";  // ndim - 1
  src << "    int axis,\n";
  src << "    int out_axis_size,\n";
  src << "    long upd_axis_stride,\n";
  src << "    long idx_axis_stride,\n";
  src << "    int size_post) {\n";

  // Grid: [size_post, idx_axis_size, size_pre]
  src << "  int post = get_global_id(0);\n";
  src << "  int j = get_global_id(1);\n";  // position along axis in index
  src << "  int pre = get_global_id(2);\n";

  src << "  int idx_axis_size = get_global_size(1);\n";

  // Compute offsets for non-axis dimensions
  // Always compute strided offsets since upd and idx may have different strides
  src << "  long upd_off = 0;\n";
  src << "  long idx_off = 0;\n";
  src << "  long out_off = 0;\n";

  src << "  if (ndim > 0) {\n";
  src << "    int linear_nonaxis = pre * size_post + post;\n";
  src << "    int tmp = linear_nonaxis;\n";
  src << "    for (int i = ndim - 1; i >= 0; i--) {\n";
  src << "      int coord = tmp % shape[i];\n";
  src << "      tmp /= shape[i];\n";
  src << "      upd_off += coord * upd_strides[i];\n";
  src << "      idx_off += coord * idx_strides[i];\n";
  src << "    }\n";
  src << "  }\n";

  // Output offset for non-axis dimensions (assuming row-major output)
  src << "  out_off = (long)pre * (long)out_axis_size * (long)size_post + (long)post;\n";

  // Get index value
  src << "  long idx_val = offset_neg_idx(idx[idx_off + j * idx_axis_stride], out_axis_size);\n";

  // Get update value
  src << "  " << val_type << " upd_val = upd[upd_off + j * upd_axis_stride];\n";

  // Write to output
  src << "  long out_idx = out_off + idx_val * (long)size_post;\n";
  if (op_type == "none") {
    src << "  out[out_idx] = upd_val;\n";
  } else if (op_type == "sum") {
    // Use atomic operations to handle overlapping indices correctly
    if (val_type == "float") {
      src << "  atomic_add_float(&out[out_idx], upd_val);\n";
    } else if (val_type == "half") {
      src << "  atomic_add_half(&out[out_idx], upd_val);\n";
    } else if (val_type == "int" || val_type == "int32_t") {
      src << "  atomic_add(&out[out_idx], upd_val);\n";
    } else if (val_type == "uint" || val_type == "uint32_t") {
      src << "  atomic_add(&out[out_idx], upd_val);\n";
    } else {
      // Fallback for other types (may have race conditions)
      src << "  out[out_idx] += upd_val;\n";
    }
  }
  src << "}\n";

  return src.str();
}

// Generate kernel for scatter operation (multi-index)
std::string get_scatter_kernel_source(
    const std::string& kernel_name,
    const std::string& val_type,
    const std::string& idx_type,
    int nidx,
    const std::string& op_type,  // "none", "sum", "prod", "max", "min"
    bool upd_contiguous) {

  std::ostringstream src;
  src << opencl::get_kernel_preamble();

  // Add atomic helpers for reduce operations
  if (op_type != "none") {
    src << get_atomic_helpers(val_type);
  }

  // Helper for negative index handling
  src << R"(
inline long offset_neg_idx(long idx, int size) {
  return (idx < 0) ? idx + size : idx;
}
)";

  src << "__kernel void " << kernel_name << "(\n";
  src << "    __global const " << val_type << "* upd,\n";
  src << "    __global " << val_type << "* out,\n";

  // Update info
  src << "    __global const int* upd_shape,\n";
  src << "    __global const long* upd_strides,\n";
  src << "    int upd_ndim,\n";
  src << "    long upd_size,\n";  // size of update slice (after idx dims)

  // Output info
  src << "    __global const int* out_shape,\n";
  src << "    __global const long* out_strides,\n";
  src << "    int out_ndim,\n";
  src << "    __global const int* axes,\n";

  // Index info
  for (int i = 0; i < nidx; i++) {
    src << "    __global const " << idx_type << "* idx" << i << ",\n";
  }
  src << "    __global const int* idx_shapes,\n";
  src << "    __global const long* idx_strides,\n";
  src << "    __global const long* idx_offsets,\n";  // Base offsets for each index array
  src << "    int idx_ndim,\n";
  src << "    long idx_size) {\n";  // total number of index elements

  // Grid: [upd_size, idx_size / nwork]
  src << "  int upd_idx = get_global_id(0);\n";
  src << "  long idx_linear = get_global_id(1);\n";

  src << "  if (upd_idx >= upd_size || idx_linear >= idx_size) return;\n";

  // Compute output offset from index arrays
  src << "  long out_off = 0;\n";
  for (int i = 0; i < nidx; i++) {
    src << "  {\n";
    src << "    long idx_off = idx_offsets[" << i << "];\n";  // Start with base offset
    if (nidx > 0) {
      src << "    long tmp = idx_linear;\n";
      src << "    for (int d = idx_ndim - 1; d >= 0; d--) {\n";
      src << "      int coord = tmp % idx_shapes[" << i << " * idx_ndim + d];\n";
      src << "      tmp /= idx_shapes[" << i << " * idx_ndim + d];\n";
      src << "      idx_off += coord * idx_strides[" << i << " * idx_ndim + d];\n";
      src << "    }\n";
    }
    src << "    long idx_val = offset_neg_idx(idx" << i << "[idx_off], out_shape[axes[" << i << "]]);\n";
    src << "    out_off += idx_val * out_strides[axes[" << i << "]];\n";
    src << "  }\n";
  }

  // Compute output offset from update slice index
  // Use the slice shape (upd_shape after idx_ndim dimensions) to decompose upd_idx,
  // then apply coordinates with out_strides
  src << "  if (upd_size > 1) {\n";
  src << "    long tmp = upd_idx;\n";
  src << "    for (int d = out_ndim - 1; d >= 0; d--) {\n";
  src << "      int coord = tmp % upd_shape[idx_ndim + d];\n";
  src << "      tmp /= upd_shape[idx_ndim + d];\n";
  src << "      out_off += coord * out_strides[d];\n";
  src << "    }\n";
  src << "  }\n";

  // Compute update offset
  src << "  long upd_off = 0;\n";
  if (upd_contiguous) {
    src << "  upd_off = idx_linear * upd_size + upd_idx;\n";
  } else {
    src << "  {\n";
    src << "    long linear = idx_linear * upd_size + upd_idx;\n";
    src << "    for (int d = upd_ndim - 1; d >= 0; d--) {\n";
    src << "      int coord = linear % upd_shape[d];\n";
    src << "      linear /= upd_shape[d];\n";
    src << "      upd_off += coord * upd_strides[d];\n";
    src << "    }\n";
    src << "  }\n";
  }

  // Apply operation with atomic support for reduce operations
  src << "  " << val_type << " upd_val = upd[upd_off];\n";
  if (op_type == "none") {
    src << "  out[out_off] = upd_val;\n";
  } else if (op_type == "sum") {
    // Use atomic operations to handle overlapping indices correctly
    if (val_type == "float") {
      src << "  atomic_add_float(&out[out_off], upd_val);\n";
    } else if (val_type == "half") {
      src << "  atomic_add_half(&out[out_off], upd_val);\n";
    } else if (val_type == "int" || val_type == "int32_t") {
      src << "  atomic_add(&out[out_off], upd_val);\n";
    } else if (val_type == "uint" || val_type == "uint32_t") {
      src << "  atomic_add(&out[out_off], upd_val);\n";
    } else {
      // Fallback for other types (may have race conditions)
      src << "  out[out_off] += upd_val;\n";
    }
  } else if (op_type == "prod") {
    if (val_type == "float") {
      src << "  atomic_mul_float(&out[out_off], upd_val);\n";
    } else if (val_type == "half") {
      src << "  atomic_mul_half(&out[out_off], upd_val);\n";
    } else {
      // Fallback for other types (may have race conditions)
      src << "  out[out_off] *= upd_val;\n";
    }
  } else if (op_type == "max") {
    if (val_type == "float") {
      src << "  atomic_max_float(&out[out_off], upd_val);\n";
    } else if (val_type == "half") {
      src << "  atomic_max_half(&out[out_off], upd_val);\n";
    } else if (val_type == "int" || val_type == "int32_t") {
      src << "  atomic_max(&out[out_off], upd_val);\n";
    } else if (val_type == "uint" || val_type == "uint32_t") {
      src << "  atomic_max(&out[out_off], upd_val);\n";
    } else {
      // Fallback for other types (may have race conditions)
      src << "  out[out_off] = max(out[out_off], upd_val);\n";
    }
  } else if (op_type == "min") {
    if (val_type == "float") {
      src << "  atomic_min_float(&out[out_off], upd_val);\n";
    } else if (val_type == "half") {
      src << "  atomic_min_half(&out[out_off], upd_val);\n";
    } else if (val_type == "int" || val_type == "int32_t") {
      src << "  atomic_min(&out[out_off], upd_val);\n";
    } else if (val_type == "uint" || val_type == "uint32_t") {
      src << "  atomic_min(&out[out_off], upd_val);\n";
    } else {
      // Fallback for other types (may have race conditions)
      src << "  out[out_off] = min(out[out_off], upd_val);\n";
    }
  }
  src << "}\n";

  return src.str();
}

} // anonymous namespace

void GatherAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& src = inputs[0];
  auto& idx = inputs[1];

  out.set_data(opencl::allocator().malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = opencl::device(s.device);

  bool src_contig = src.flags().row_contiguous;
  bool idx_contig = idx.flags().row_contiguous;

  std::string src_type = opencl::type_to_name(src.dtype());
  std::string idx_type = opencl::type_to_name(idx.dtype());

  std::string kernel_name = "gather_axis_" + src_type + "_" + idx_type;
  kernel_name += src_contig ? "_sc" : "_snc";
  kernel_name += idx_contig ? "_ic" : "_inc";

  std::string kernel_source = get_gather_axis_kernel_source(
      kernel_name, src_type, idx_type, src_contig, idx_contig);

  OPENCL_DEBUG_LOG("[GatherAxis] kernel_name=" << kernel_name);

  cl_kernel kernel = d.get_kernel(kernel_name, kernel_source, "");

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_kernel(kernel);

  // Grid: [size_post, idx_axis_size, size_pre]
  size_t size_pre = 1;
  size_t size_post = 1;
  for (int i = 0; i < axis_; ++i) {
    size_pre *= idx.shape(i);
  }
  for (int i = axis_ + 1; i < idx.ndim(); ++i) {
    size_post *= idx.shape(i);
  }

  int idx_ax_size = idx.shape(axis_);

  // Prepare shape/strides without axis
  auto shape_no_axis = remove_index(idx.shape(), axis_);
  auto src_strides_no_axis = remove_index(src.strides(), axis_);
  auto idx_strides_no_axis = remove_index(idx.strides(), axis_);

  // Convert to int/long vectors
  std::vector<int> shape_vec(shape_no_axis.begin(), shape_no_axis.end());
  std::vector<int64_t> src_strides_vec(src_strides_no_axis.begin(), src_strides_no_axis.end());
  std::vector<int64_t> idx_strides_vec(idx_strides_no_axis.begin(), idx_strides_no_axis.end());

  // Handle empty case
  if (shape_vec.empty()) {
    shape_vec.push_back(1);
    src_strides_vec.push_back(0);
    idx_strides_vec.push_back(0);
  }

  int ndim = shape_vec.size();
  int src_axis_size = src.shape(axis_);
  int64_t src_axis_stride = src.strides(axis_);
  int64_t idx_axis_stride = idx.strides(axis_);

  // Create buffers
  cl_int err;
  cl_mem shape_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      shape_vec.size() * sizeof(int), shape_vec.data(), &err);
  cl_mem src_strides_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      src_strides_vec.size() * sizeof(int64_t), src_strides_vec.data(), &err);
  cl_mem idx_strides_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      idx_strides_vec.size() * sizeof(int64_t), idx_strides_vec.data(), &err);

  // Set kernel arguments
  int arg = 0;
  compute_encoder.set_input_array(src, arg++);
  compute_encoder.set_input_array(idx, arg++);
  compute_encoder.set_output_array(out, arg++);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &shape_buf);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &src_strides_buf);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx_strides_buf);
  compute_encoder.set_bytes(ndim, arg++);
  compute_encoder.set_bytes(axis_, arg++);
  compute_encoder.set_bytes(src_axis_size, arg++);
  compute_encoder.set_bytes(src_axis_stride, arg++);
  compute_encoder.set_bytes(idx_axis_stride, arg++);
  int size_post_int = static_cast<int>(size_post);
  compute_encoder.set_bytes(size_post_int, arg++);
  // Pass the source base offset (convert from bytes to elements)
  int64_t src_base_offset = src.offset() / size_of(src.dtype());
  compute_encoder.set_bytes(src_base_offset, arg++);
  // Pass the index base offset (convert from bytes to elements)
  int64_t idx_base_offset = idx.offset() / size_of(idx.dtype());
  compute_encoder.set_bytes(idx_base_offset, arg++);

  size_t global_size[3] = {size_post, (size_t)idx_ax_size, size_pre};
  compute_encoder.dispatch_threads(global_size, nullptr, 3);

  // Track buffers for cleanup after sync
  d.add_temp_buffer(shape_buf, s.index);
  d.add_temp_buffer(src_strides_buf, s.index);
  d.add_temp_buffer(idx_strides_buf, s.index);

  // Release temp buffers after kernel execution completes
  d.end_encoding(s.index);
}

void Gather::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& src = inputs[0];
  int nidx = inputs.size() - 1;

  if (nidx > OPENCL_MAX_INDEX_ARRAYS) {
    throw std::runtime_error(
        "[Gather::eval_gpu] Gathering with more than " +
        std::to_string(OPENCL_MAX_INDEX_ARRAYS) + " index arrays not yet supported.");
  }

  out.set_data(opencl::allocator().malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = opencl::device(s.device);

  size_t slice_size = 1;
  for (auto sz : slice_sizes_) {
    slice_size *= sz;
  }

  int idx_ndim = nidx ? inputs[1].ndim() : 0;

  std::string src_type = opencl::type_to_name(src.dtype());
  std::string idx_type = nidx ? opencl::type_to_name(inputs[1].dtype()) : "int";

  std::string kernel_name = "gather_" + src_type + "_" + idx_type +
      "_n" + std::to_string(nidx) + "_d" + std::to_string(idx_ndim);

  std::string kernel_source = get_gather_kernel_source(
      kernel_name, src_type, idx_type, nidx, idx_ndim);

  OPENCL_DEBUG_LOG("[Gather] kernel_name=" << kernel_name << " nidx=" << nidx);

  cl_kernel kernel = d.get_kernel(kernel_name, kernel_source, "");

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_kernel(kernel);

  // Grid dimensions based on index shape and slice size
  size_t dim0 = 1, dim1 = 1;
  if (nidx) {
    if (inputs[1].ndim() >= 1) {
      dim0 = inputs[1].shape(0);
    }
    if (inputs[1].ndim() >= 2) {
      dim1 = inputs[1].size() / dim0;
    }
  }

  // Prepare arrays for index info
  std::vector<int> idx_shapes;
  std::vector<int64_t> idx_strides;
  std::vector<int64_t> idx_offsets;
  for (int i = 0; i < nidx; ++i) {
    for (auto sh : inputs[i + 1].shape()) {
      idx_shapes.push_back(sh);
    }
    for (auto st : inputs[i + 1].strides()) {
      idx_strides.push_back(st);
    }
    // Collect the base offset for each index array (in elements, converted from bytes)
    idx_offsets.push_back(inputs[i + 1].offset() / size_of(inputs[i + 1].dtype()));
  }

  // Handle empty case
  if (idx_shapes.empty()) {
    idx_shapes.push_back(1);
    idx_strides.push_back(0);
  }
  if (idx_offsets.empty()) {
    idx_offsets.push_back(0);
  }

  // Prepare src shape/strides, slice_sizes, axes
  std::vector<int> src_shape(src.shape().begin(), src.shape().end());
  std::vector<int64_t> src_strides(src.strides().begin(), src.strides().end());
  std::vector<int> slice_sizes_vec(slice_sizes_.begin(), slice_sizes_.end());
  std::vector<int> axes_vec(axes_.begin(), axes_.end());

  if (src_shape.empty()) {
    src_shape.push_back(1);
    src_strides.push_back(0);
    slice_sizes_vec.push_back(1);
  }

  // Create buffers
  cl_int err;
  cl_mem src_shape_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      src_shape.size() * sizeof(int), src_shape.data(), &err);
  cl_mem src_strides_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      src_strides.size() * sizeof(int64_t), src_strides.data(), &err);
  cl_mem slice_sizes_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      slice_sizes_vec.size() * sizeof(int), slice_sizes_vec.data(), &err);
  cl_mem axes_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      axes_vec.empty() ? sizeof(int) : axes_vec.size() * sizeof(int),
      axes_vec.empty() ? src_shape.data() : axes_vec.data(), &err);
  cl_mem idx_shapes_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      idx_shapes.size() * sizeof(int), idx_shapes.data(), &err);
  cl_mem idx_strides_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      idx_strides.size() * sizeof(int64_t), idx_strides.data(), &err);
  cl_mem idx_offsets_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      idx_offsets.size() * sizeof(int64_t), idx_offsets.data(), &err);

  // Set kernel arguments
  int arg = 0;
  compute_encoder.set_input_array(src, arg++);
  compute_encoder.set_output_array(out, arg++);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &src_shape_buf);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &src_strides_buf);
  int src_ndim = src.ndim() > 0 ? src.ndim() : 1;
  compute_encoder.set_bytes(src_ndim, arg++);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &slice_sizes_buf);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &axes_buf);

  // Index arrays
  for (int i = 0; i < nidx; ++i) {
    compute_encoder.set_input_array(inputs[i + 1], arg++);
  }
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx_shapes_buf);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx_strides_buf);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx_offsets_buf);
  compute_encoder.set_bytes(idx_ndim, arg++);
  int64_t slice_size_64 = slice_size;
  compute_encoder.set_bytes(slice_size_64, arg++);
  // Pass the source base offset (convert from bytes to elements)
  int64_t src_base_offset = src.offset() / size_of(src.dtype());
  compute_encoder.set_bytes(src_base_offset, arg++);

  size_t global_size[3] = {dim0, dim1, slice_size};
  compute_encoder.dispatch_threads(global_size, nullptr, 3);

  // Track buffers for cleanup after sync
  d.add_temp_buffer(src_shape_buf, s.index);
  d.add_temp_buffer(src_strides_buf, s.index);
  d.add_temp_buffer(slice_sizes_buf, s.index);
  d.add_temp_buffer(axes_buf, s.index);
  d.add_temp_buffer(idx_shapes_buf, s.index);
  d.add_temp_buffer(idx_strides_buf, s.index);
  d.add_temp_buffer(idx_offsets_buf, s.index);

  // Release temp buffers after kernel execution completes
  d.end_encoding(s.index);
}

void ScatterAxis::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& src = inputs[0];
  auto& idx = inputs[1];
  auto& upd = inputs[2];

  // Copy src into out first
  CopyType copy_type;
  if (src.data_size() == 1) {
    copy_type = CopyType::Scalar;
  } else if (src.flags().row_contiguous) {
    copy_type = CopyType::Vector;
  } else {
    copy_type = CopyType::General;
  }
  copy_gpu(src, out, copy_type, stream());

  if (upd.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = opencl::device(s.device);

  bool upd_contig = upd.flags().row_contiguous;
  bool idx_contig = idx.flags().row_contiguous;

  std::string val_type = opencl::type_to_name(out.dtype());
  std::string idx_type = opencl::type_to_name(idx.dtype());
  std::string op_name = (reduce_type_ == ScatterAxis::None) ? "none" : "sum";

  std::string kernel_name = "scatter_axis_" + val_type + "_" + idx_type + "_" + op_name;
  kernel_name += upd_contig ? "_uc" : "_unc";
  kernel_name += idx_contig ? "_ic" : "_inc";

  std::string kernel_source = get_scatter_axis_kernel_source(
      kernel_name, val_type, idx_type, op_name, upd_contig, idx_contig);

  OPENCL_DEBUG_LOG("[ScatterAxis] kernel_name=" << kernel_name);

  cl_kernel kernel = d.get_kernel(kernel_name, kernel_source, "");

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_kernel(kernel);

  // Grid: [size_post, idx_axis_size, size_pre]
  size_t size_pre = 1;
  size_t size_post = 1;
  for (int i = 0; i < axis_; ++i) {
    size_pre *= idx.shape(i);
  }
  for (int i = axis_ + 1; i < idx.ndim(); ++i) {
    size_post *= idx.shape(i);
  }

  int idx_ax_size = idx.shape(axis_);

  // Prepare shape/strides without axis
  auto shape_no_axis = remove_index(idx.shape(), axis_);
  auto upd_strides_no_axis = remove_index(upd.strides(), axis_);
  auto idx_strides_no_axis = remove_index(idx.strides(), axis_);

  std::vector<int> shape_vec(shape_no_axis.begin(), shape_no_axis.end());
  std::vector<int64_t> upd_strides_vec(upd_strides_no_axis.begin(), upd_strides_no_axis.end());
  std::vector<int64_t> idx_strides_vec(idx_strides_no_axis.begin(), idx_strides_no_axis.end());

  if (shape_vec.empty()) {
    shape_vec.push_back(1);
    upd_strides_vec.push_back(0);
    idx_strides_vec.push_back(0);
  }

  int ndim = shape_vec.size();
  int out_axis_size = out.shape(axis_);
  int64_t upd_axis_stride = upd.strides(axis_);
  int64_t idx_axis_stride = idx.strides(axis_);

  // Create buffers
  cl_int err;
  cl_mem shape_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      shape_vec.size() * sizeof(int), shape_vec.data(), &err);
  cl_mem upd_strides_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      upd_strides_vec.size() * sizeof(int64_t), upd_strides_vec.data(), &err);
  cl_mem idx_strides_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      idx_strides_vec.size() * sizeof(int64_t), idx_strides_vec.data(), &err);

  // Set kernel arguments
  int arg = 0;
  compute_encoder.set_input_array(upd, arg++);
  compute_encoder.set_input_array(idx, arg++);
  compute_encoder.set_output_array(out, arg++);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &shape_buf);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &upd_strides_buf);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx_strides_buf);
  compute_encoder.set_bytes(ndim, arg++);
  compute_encoder.set_bytes(axis_, arg++);
  compute_encoder.set_bytes(out_axis_size, arg++);
  compute_encoder.set_bytes(upd_axis_stride, arg++);
  compute_encoder.set_bytes(idx_axis_stride, arg++);
  int size_post_int = static_cast<int>(size_post);
  compute_encoder.set_bytes(size_post_int, arg++);

  size_t global_size[3] = {size_post, (size_t)idx_ax_size, size_pre};
  compute_encoder.dispatch_threads(global_size, nullptr, 3);

  // Track buffers for cleanup after sync
  d.add_temp_buffer(shape_buf, s.index);
  d.add_temp_buffer(upd_strides_buf, s.index);
  d.add_temp_buffer(idx_strides_buf, s.index);

  // Release temp buffers after kernel execution completes
  d.end_encoding(s.index);
}

void Scatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  int nidx = axes_.size();

  if (nidx > OPENCL_MAX_INDEX_ARRAYS) {
    throw std::runtime_error(
        "[Scatter::eval_gpu] Scattering with more than " +
        std::to_string(OPENCL_MAX_INDEX_ARRAYS) + " index arrays not yet supported.");
  }

  // Copy src into out first
  auto& src = inputs[0];
  CopyType copy_type;
  if (src.data_size() == 1) {
    copy_type = CopyType::Scalar;
  } else if (src.flags().row_contiguous) {
    copy_type = CopyType::Vector;
  } else {
    copy_type = CopyType::General;
  }
  copy_gpu(src, out, copy_type, stream());

  auto& upd = inputs.back();
  if (upd.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = opencl::device(s.device);

  int idx_ndim = nidx ? inputs[1].ndim() : 0;
  size_t idx_size = nidx ? inputs[1].size() : 1;
  bool upd_contig = upd.flags().row_contiguous;

  std::string val_type = opencl::type_to_name(out.dtype());
  std::string idx_type = nidx ? opencl::type_to_name(inputs[1].dtype()) : "int";

  std::string op_name;
  switch (reduce_type_) {
    case Scatter::None: op_name = "none"; break;
    case Scatter::Sum: op_name = "sum"; break;
    case Scatter::Prod: op_name = "prod"; break;
    case Scatter::Max: op_name = "max"; break;
    case Scatter::Min: op_name = "min"; break;
  }

  std::string kernel_name = "scatter_" + val_type + "_" + idx_type + "_" + op_name +
      "_n" + std::to_string(nidx);
  kernel_name += upd_contig ? "_uc" : "_unc";

  std::string kernel_source = get_scatter_kernel_source(
      kernel_name, val_type, idx_type, nidx, op_name, upd_contig);

  OPENCL_DEBUG_LOG("[Scatter] kernel_name=" << kernel_name << " nidx=" << nidx);

  cl_kernel kernel = d.get_kernel(kernel_name, kernel_source, "");

  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder.set_kernel(kernel);

  // Compute update size (elements per index location)
  size_t upd_size = 1;
  for (int i = idx_ndim; i < upd.ndim(); ++i) {
    upd_size *= upd.shape(i);
  }

  // Prepare arrays
  std::vector<int> upd_shape(upd.shape().begin(), upd.shape().end());
  std::vector<int64_t> upd_strides(upd.strides().begin(), upd.strides().end());
  std::vector<int> out_shape(out.shape().begin(), out.shape().end());
  std::vector<int64_t> out_strides(out.strides().begin(), out.strides().end());
  std::vector<int> axes_vec(axes_.begin(), axes_.end());

  std::vector<int> idx_shapes;
  std::vector<int64_t> idx_strides;
  std::vector<int64_t> idx_offsets;
  for (int i = 0; i < nidx; ++i) {
    for (auto sh : inputs[i + 1].shape()) {
      idx_shapes.push_back(sh);
    }
    for (auto st : inputs[i + 1].strides()) {
      idx_strides.push_back(st);
    }
    // Collect the base offset for each index array (in elements, converted from bytes)
    idx_offsets.push_back(inputs[i + 1].offset() / size_of(inputs[i + 1].dtype()));
  }

  // Handle empty cases
  if (upd_shape.empty()) { upd_shape.push_back(1); upd_strides.push_back(0); }
  if (out_shape.empty()) { out_shape.push_back(1); out_strides.push_back(0); }
  if (axes_vec.empty()) { axes_vec.push_back(0); }
  if (idx_shapes.empty()) { idx_shapes.push_back(1); idx_strides.push_back(0); }
  if (idx_offsets.empty()) { idx_offsets.push_back(0); }

  // Create buffers
  cl_int err;
  cl_mem upd_shape_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      upd_shape.size() * sizeof(int), upd_shape.data(), &err);
  cl_mem upd_strides_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      upd_strides.size() * sizeof(int64_t), upd_strides.data(), &err);
  cl_mem out_shape_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      out_shape.size() * sizeof(int), out_shape.data(), &err);
  cl_mem out_strides_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      out_strides.size() * sizeof(int64_t), out_strides.data(), &err);
  cl_mem axes_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      axes_vec.size() * sizeof(int), axes_vec.data(), &err);
  cl_mem idx_shapes_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      idx_shapes.size() * sizeof(int), idx_shapes.data(), &err);
  cl_mem idx_strides_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      idx_strides.size() * sizeof(int64_t), idx_strides.data(), &err);
  cl_mem idx_offsets_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      idx_offsets.size() * sizeof(int64_t), idx_offsets.data(), &err);

  // Set kernel arguments
  int arg = 0;
  compute_encoder.set_input_array(upd, arg++);
  compute_encoder.set_output_array(out, arg++);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &upd_shape_buf);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &upd_strides_buf);
  int upd_ndim = upd.ndim() > 0 ? upd.ndim() : 1;
  compute_encoder.set_bytes(upd_ndim, arg++);
  int64_t upd_size_64 = upd_size;
  compute_encoder.set_bytes(upd_size_64, arg++);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &out_shape_buf);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &out_strides_buf);
  int out_ndim = out.ndim() > 0 ? out.ndim() : 1;
  compute_encoder.set_bytes(out_ndim, arg++);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &axes_buf);

  for (int i = 0; i < nidx; ++i) {
    compute_encoder.set_input_array(inputs[i + 1], arg++);
  }
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx_shapes_buf);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx_strides_buf);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx_offsets_buf);
  compute_encoder.set_bytes(idx_ndim, arg++);
  int64_t idx_size_64 = idx_size;
  compute_encoder.set_bytes(idx_size_64, arg++);

  size_t global_size[2] = {upd_size, idx_size};
  compute_encoder.dispatch_threads(global_size, nullptr, 2);

  // Defer buffer release until after kernel execution completes
  d.add_temp_buffer(upd_shape_buf, s.index);
  d.add_temp_buffer(upd_strides_buf, s.index);
  d.add_temp_buffer(out_shape_buf, s.index);
  d.add_temp_buffer(out_strides_buf, s.index);
  d.add_temp_buffer(axes_buf, s.index);
  d.add_temp_buffer(idx_shapes_buf, s.index);
  d.add_temp_buffer(idx_strides_buf, s.index);
  d.add_temp_buffer(idx_offsets_buf, s.index);

  // Release temp buffers after kernel execution completes
  d.end_encoding(s.index);
}

namespace {

// Generate kernel for computing exclusive prefix sum on boolean mask
std::string get_scan_bool_kernel_source(const std::string& kernel_name) {
  std::ostringstream src;
  src << opencl::get_kernel_preamble();

  // Simple sequential scan - each work item handles one row
  src << "__kernel void " << kernel_name << "(\n";
  src << "    __global const bool* in,\n";
  src << "    __global uint* out,\n";
  src << "    int size) {\n";
  src << "  int row_idx = get_global_id(0);\n";
  src << "  if (row_idx != 0) return;\n";  // Only one work item does the work
  src << "\n";
  src << "  uint acc = 0;\n";
  src << "  for (int i = 0; i < size; i++) {\n";
  src << "    out[i] = acc;\n";  // Exclusive: output before adding
  src << "    acc += (uint)(in[i] ? 1 : 0);\n";
  src << "  }\n";
  src << "}\n";

  return src.str();
}

// Generate kernel for masked assign operation
std::string get_masked_assign_kernel_source(
    const std::string& kernel_name,
    const std::string& val_type,
    bool src_contiguous) {
  std::ostringstream src;
  src << opencl::get_kernel_preamble();

  src << "__kernel void " << kernel_name << "(\n";
  src << "    __global const bool* mask,\n";
  src << "    __global const uint* scatter_offsets,\n";
  src << "    __global const " << val_type << "* src,\n";
  src << "    __global " << val_type << "* out,\n";
  src << "    __global const int* src_shapes,\n";
  src << "    __global const long* src_strides,\n";
  src << "    int src_ndim,\n";
  src << "    long src_batch_size,\n";
  src << "    long mask_batch_size,\n";
  src << "    int total) {\n";

  src << "  int idx = get_global_id(0);\n";
  src << "  if (idx >= total) return;\n";
  src << "\n";
  src << "  if (!mask[idx]) return;\n";
  src << "\n";
  src << "  uint src_index = scatter_offsets[idx];\n";
  src << "  if (src_index >= src_batch_size) return;\n";
  src << "\n";
  src << "  uint batch_idx = idx / mask_batch_size;\n";
  src << "\n";

  if (src_contiguous) {
    src << "  out[idx] = src[batch_idx * src_batch_size + src_index];\n";
  } else {
    // Non-contiguous: compute strided offset
    src << "  long src_off = 0;\n";
    src << "  long linear = batch_idx * src_batch_size + src_index;\n";
    src << "  for (int d = src_ndim - 1; d >= 0; d--) {\n";
    src << "    int coord = linear % src_shapes[d];\n";
    src << "    linear /= src_shapes[d];\n";
    src << "    src_off += coord * src_strides[d];\n";
    src << "  }\n";
    src << "  out[idx] = src[src_off];\n";
  }

  src << "}\n";

  return src.str();
}

} // anonymous namespace

void MaskedScatter::eval_gpu(const std::vector<array>& inputs, array& out) {
  const array& dst = inputs[0];
  const array& mask = inputs[1];
  const array& src = inputs[2];

  auto& s = stream();
  auto& d = opencl::device(s.device);

  const size_t total = mask.size();

  // Copy dst into out first
  CopyType ct = (total == 1)
      ? CopyType::Scalar
      : (dst.flags().row_contiguous ? CopyType::Vector : CopyType::General);
  copy_gpu(dst, out, ct, s);

  if (total == 0) {
    return;
  }

  cl_int err;

  // For the mask, we need a row-contiguous version for the scan to work properly
  array mask_work = mask;
  if (!mask.flags().row_contiguous) {
    // Make a contiguous copy - same logic as contiguous_copy_gpu
    mask_work = array(mask.shape(), mask.dtype(), nullptr, {});
    copy_gpu(mask, mask_work, CopyType::General, s);
  }

  // Create scatter_offsets buffer (exclusive prefix sum of mask)
  cl_mem scatter_offsets_buf = clCreateBuffer(d.context(), CL_MEM_READ_WRITE,
      total * sizeof(uint32_t), nullptr, &err);
  d.add_temp_buffer(scatter_offsets_buf, s.index);

  // Run exclusive prefix sum on mask
  {
    std::string scan_kernel_name = "scan_bool_exclusive";
    std::string scan_source = get_scan_bool_kernel_source(scan_kernel_name);
    cl_kernel scan_kernel = d.get_kernel(scan_kernel_name, scan_source, "");

    auto& encoder = d.get_command_encoder(s.index);
    encoder.set_kernel(scan_kernel);
    encoder.set_input_array(mask_work, 0);
    clSetKernelArg(scan_kernel, 1, sizeof(cl_mem), &scatter_offsets_buf);
    encoder.set_bytes(static_cast<int>(total), 2);

    // Single work item for simple sequential scan
    size_t global_size[3] = {1, 0, 0};
    encoder.dispatch_threads(global_size, nullptr, 1);
  }

  // Run masked assign kernel
  std::string val_type = opencl::type_to_name(out.dtype());
  bool src_contig = src.flags().row_contiguous;

  std::string kernel_name = "masked_assign_" + val_type;
  kernel_name += src_contig ? "_c" : "_nc";

  std::string kernel_source = get_masked_assign_kernel_source(kernel_name, val_type, src_contig);
  cl_kernel kernel = d.get_kernel(kernel_name, kernel_source, "");

  // Prepare src shape/strides
  std::vector<int> src_shape(src.shape().begin(), src.shape().end());
  std::vector<int64_t> src_strides(src.strides().begin(), src.strides().end());

  if (src_shape.empty()) {
    src_shape.push_back(1);
    src_strides.push_back(0);
  }

  cl_mem src_shape_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      src_shape.size() * sizeof(int), src_shape.data(), &err);
  cl_mem src_strides_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      src_strides.size() * sizeof(int64_t), src_strides.data(), &err);

  auto& encoder = d.get_command_encoder(s.index);
  encoder.set_kernel(kernel);

  int arg = 0;
  encoder.set_input_array(mask_work, arg++);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &scatter_offsets_buf);
  encoder.set_input_array(src, arg++);
  encoder.set_output_array(out, arg++);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &src_shape_buf);
  clSetKernelArg(kernel, arg++, sizeof(cl_mem), &src_strides_buf);
  encoder.set_bytes(static_cast<int>(src.ndim() > 0 ? src.ndim() : 1), arg++);
  encoder.set_bytes(static_cast<int64_t>(src.size() / (src.ndim() > 0 ? src.shape(0) : 1)), arg++);
  encoder.set_bytes(static_cast<int64_t>(mask.size() / (mask.ndim() > 0 ? mask.shape(0) : 1)), arg++);
  encoder.set_bytes(static_cast<int>(total), arg++);

  size_t global_size[3] = {total, 0, 0};
  encoder.dispatch_threads(global_size, nullptr, 1);

  d.add_temp_buffer(src_shape_buf, s.index);
  d.add_temp_buffer(src_strides_buf, s.index);

  d.end_encoding(s.index);
}

} // namespace mlx::core
