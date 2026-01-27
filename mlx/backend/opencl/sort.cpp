// Copyright © 2025 MLX Contributors
// OpenCL Sort, ArgSort, Partition, ArgPartition implementation
// Based on GPU merge sort algorithm from Metal implementation
// (which is based on NVIDIA CUB library)

#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/copy.h"
#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/primitives.h"

#include <sstream>
#include <cmath>

namespace mlx::core {

namespace {

// Configuration constants matching Metal implementation
constexpr int N_PER_THREAD = 4;  // Each thread handles 4 elements

// Get block size based on axis size (matching Metal logic)
int get_block_threads(int size_sorted_axis, size_t dtype_size) {
  int tn = N_PER_THREAD;
  int potential_bn = (size_sorted_axis + tn - 1) / tn;

  int bn;
  if (potential_bn > 256) {
    bn = 512;
  } else if (potential_bn > 128) {
    bn = 256;
  } else if (potential_bn > 64) {
    bn = 128;
  } else if (potential_bn > 32) {
    bn = 64;
  } else {
    bn = 32;
  }

  // Reduce block size for larger data types to avoid running out of local memory
  if (bn == 512 && dtype_size > 4) {
    bn = 256;
  }

  return bn;
}

// Generate the NaN-aware comparison helper for a given type
std::string generate_comparison_helper(const std::string& type_name) {
  std::ostringstream ss;
  bool is_bf16 = (type_name == "bfloat16_t");
  bool is_half = (type_name == "half");
  bool is_complex = (type_name == "float2");

  ss << "// NaN-aware comparison (NaN goes to end)\n";
  if (is_bf16) {
    ss << "inline bool comp_less(" << type_name << " a, " << type_name << " b) {\n";
    ss << "  float fa = bfloat16_to_float(a);\n";
    ss << "  float fb = bfloat16_to_float(b);\n";
    ss << "  if (isnan(fa)) return false;\n";
    ss << "  if (isnan(fb)) return true;\n";
    ss << "  return fa < fb;\n";
    ss << "}\n";
  } else if (is_half) {
    ss << "inline bool comp_less(half a, half b) {\n";
    ss << "  float fa = convert_float(a);\n";
    ss << "  float fb = convert_float(b);\n";
    ss << "  if (isnan(fa)) return false;\n";
    ss << "  if (isnan(fb)) return true;\n";
    ss << "  return fa < fb;\n";
    ss << "}\n";
  } else if (is_complex) {
    ss << "inline bool comp_less(float2 a, float2 b) {\n";
    ss << "  if (isnan(a.x)) return false;\n";
    ss << "  if (isnan(b.x)) return true;\n";
    ss << "  return a.x < b.x;\n";
    ss << "}\n";
  } else if (type_name == "float" || type_name == "double") {
    ss << "inline bool comp_less(" << type_name << " a, " << type_name << " b) {\n";
    ss << "  if (isnan(a)) return false;\n";
    ss << "  if (isnan(b)) return true;\n";
    ss << "  return a < b;\n";
    ss << "}\n";
  } else {
    // Integer types - no NaN handling needed
    ss << "inline bool comp_less(" << type_name << " a, " << type_name << " b) {\n";
    ss << "  return a < b;\n";
    ss << "}\n";
  }

  return ss.str();
}

// Get the initialization value for a type (used to fill padding)
// IMPORTANT: Padding values must compare as "greater than" all valid values
// so they sort to the end. For float types, we use NaN (which our comparison
// treats as greater than everything). For integer types, we use MAX value.
std::string get_init_value(const std::string& type_name) {
  bool is_bf16 = (type_name == "bfloat16_t");
  bool is_half = (type_name == "half");

  if (is_bf16) {
    // bfloat16 NaN: exponent=0xFF, mantissa!=0
    // 0x7FC0 is a quiet NaN in bfloat16 format
    return "((bfloat16_t)0x7FC0)";
  } else if (is_half) {
    // half NaN: exponent=0x1F, mantissa!=0
    // 0x7E00 is a quiet NaN in float16 format
    return "as_half((ushort)0x7E00)";
  } else if (type_name == "float") {
    return "NAN";
  } else if (type_name == "double") {
    return "NAN";
  } else if (type_name == "float2") {
    return "(float2)(NAN, 0)";
  } else {
    // For integer types, use max value
    if (type_name == "int" || type_name == "int32_t") return "INT_MAX";
    if (type_name == "uint" || type_name == "uint32_t") return "UINT_MAX";
    if (type_name == "long" || type_name == "int64_t") return "LONG_MAX";
    if (type_name == "ulong" || type_name == "uint64_t") return "ULONG_MAX";
    if (type_name == "short" || type_name == "int16_t") return "SHRT_MAX";
    if (type_name == "ushort" || type_name == "uint16_t") return "USHRT_MAX";
    if (type_name == "char" || type_name == "int8_t") return "CHAR_MAX";
    if (type_name == "uchar" || type_name == "uint8_t") return "UCHAR_MAX";
    return "0";  // Fallback
  }
}

// Generate the block sort kernel - sorts up to N_PER_BLOCK elements per workgroup
std::string generate_block_sort_kernel(
    const std::string& type_name,
    int block_threads,
    bool argsort) {

  int n_per_block = block_threads * N_PER_THREAD;
  std::string init_val = get_init_value(type_name);

  std::ostringstream kernel;
  kernel << opencl::get_kernel_preamble();
  kernel << generate_comparison_helper(type_name);

  // Thread-level swap helper
  kernel << "\ninline void val_swap(__private " << type_name << "* a, __private " << type_name << "* b) {\n";
  kernel << "  " << type_name << " t = *a; *a = *b; *b = t;\n";
  kernel << "}\n";
  kernel << "inline void idx_swap(__private uint* a, __private uint* b) {\n";
  kernel << "  uint t = *a; *a = *b; *b = t;\n";
  kernel << "}\n";

  // Merge partition helper - binary search to find split point
  kernel << "\n// Find partition point for merging two sorted sequences\n";
  kernel << "inline int merge_partition(\n";
  kernel << "    __local const " << type_name << "* As,\n";
  kernel << "    __local const " << type_name << "* Bs,\n";
  kernel << "    int A_sz, int B_sz, int sort_md) {\n";
  kernel << "  int A_st = max(0, sort_md - B_sz);\n";
  kernel << "  int A_ed = min(sort_md, A_sz);\n";
  kernel << "  while (A_st < A_ed) {\n";
  kernel << "    int md = A_st + (A_ed - A_st) / 2;\n";
  kernel << "    " << type_name << " a = As[md];\n";
  kernel << "    " << type_name << " b = Bs[sort_md - 1 - md];\n";
  kernel << "    if (comp_less(b, a)) {\n";
  kernel << "      A_ed = md;\n";
  kernel << "    } else {\n";
  kernel << "      A_st = md + 1;\n";
  kernel << "    }\n";
  kernel << "  }\n";
  kernel << "  return A_ed;\n";
  kernel << "}\n";

  // Main block sort kernel
  std::string suffix = argsort ? "_argsort" : "_sort";
  kernel << "\n__kernel void block_sort_" << type_name << suffix << "(\n";
  kernel << "    __global const " << type_name << "* input,\n";
  if (argsort) {
    kernel << "    __global uint* output,\n";
  } else {
    kernel << "    __global " << type_name << "* output,\n";
  }
  kernel << "    const int size_sorted_axis,\n";
  kernel << "    const int in_stride_sorted_axis,\n";
  kernel << "    const int out_stride_sorted_axis,\n";
  kernel << "    const int in_stride_segment,\n";
  kernel << "    const int out_stride_segment,\n";
  kernel << "    __local " << type_name << "* tgp_vals,\n";
  kernel << "    __local uint* tgp_idxs) {\n";

  kernel << "  int lid = get_local_id(0);\n";
  kernel << "  int segment_id = get_group_id(1);\n";
  kernel << "  \n";
  kernel << "  // Offset input/output by segment\n";
  kernel << "  input += segment_id * in_stride_segment;\n";
  kernel << "  output += segment_id * out_stride_segment;\n";
  kernel << "  \n";

  // Load data into local memory
  kernel << "  // Load data into local memory (strided pattern for coalescing)\n";
  kernel << "  for (int i = lid; i < " << n_per_block << "; i += " << block_threads << ") {\n";
  kernel << "    if (i < size_sorted_axis) {\n";
  kernel << "      tgp_vals[i] = input[i * in_stride_sorted_axis];\n";
  kernel << "    } else {\n";
  kernel << "      tgp_vals[i] = " << init_val << ";  // Padding value\n";
  kernel << "    }\n";
  if (argsort) {
    kernel << "    tgp_idxs[i] = i;\n";
  }
  kernel << "  }\n";
  kernel << "  barrier(CLK_LOCAL_MEM_FENCE);\n\n";

  // Load N_PER_THREAD elements into private registers
  kernel << "  // Each thread loads " << N_PER_THREAD << " elements into private registers\n";
  kernel << "  int base_idx = lid * " << N_PER_THREAD << ";\n";
  kernel << "  " << type_name << " thread_vals[" << N_PER_THREAD << "];\n";
  if (argsort) {
    kernel << "  uint thread_idxs[" << N_PER_THREAD << "];\n";
  }
  kernel << "  for (int i = 0; i < " << N_PER_THREAD << "; ++i) {\n";
  kernel << "    thread_vals[i] = tgp_vals[base_idx + i];\n";
  if (argsort) {
    kernel << "    thread_idxs[i] = tgp_idxs[base_idx + i];\n";
  }
  kernel << "  }\n\n";

  // Thread-level sort using odd-even transposition (simple and efficient for small N)
  kernel << "  // Thread-level odd-even transposition sort\n";
  kernel << "  if (base_idx < size_sorted_axis) {\n";
  kernel << "    for (int iter = 0; iter < " << N_PER_THREAD << "; ++iter) {\n";
  kernel << "      for (int j = (iter & 1); j < " << N_PER_THREAD - 1 << "; j += 2) {\n";
  kernel << "        if (comp_less(thread_vals[j + 1], thread_vals[j])) {\n";
  kernel << "          val_swap(&thread_vals[j], &thread_vals[j + 1]);\n";
  if (argsort) {
    kernel << "          idx_swap(&thread_idxs[j], &thread_idxs[j + 1]);\n";
  }
  kernel << "        }\n";
  kernel << "      }\n";
  kernel << "    }\n";
  kernel << "  }\n\n";

  // Iterative merge within workgroup
  kernel << "  // Iterative merge within workgroup\n";
  kernel << "  for (int merge_threads = 2; merge_threads <= " << block_threads << "; merge_threads *= 2) {\n";
  kernel << "    // Write thread registers back to local memory\n";
  kernel << "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  kernel << "    for (int i = 0; i < " << N_PER_THREAD << "; ++i) {\n";
  kernel << "      tgp_vals[base_idx + i] = thread_vals[i];\n";
  if (argsort) {
    kernel << "      tgp_idxs[base_idx + i] = thread_idxs[i];\n";
  }
  kernel << "    }\n";
  kernel << "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  kernel << "    \n";
  kernel << "    // Find merge group and lane\n";
  kernel << "    int merge_group = lid / merge_threads;\n";
  kernel << "    int merge_lane = lid % merge_threads;\n";
  kernel << "    \n";
  kernel << "    int sort_sz = " << N_PER_THREAD << " * merge_threads;\n";
  kernel << "    int sort_st = sort_sz * merge_group;\n";
  kernel << "    \n";
  kernel << "    // A is first half, B is second half\n";
  kernel << "    int A_st = sort_st;\n";
  kernel << "    int A_ed = sort_st + sort_sz / 2;\n";
  kernel << "    int B_st = A_ed;\n";
  kernel << "    int B_ed = sort_st + sort_sz;\n";
  kernel << "    \n";
  kernel << "    __local const " << type_name << "* As = tgp_vals + A_st;\n";
  kernel << "    __local const " << type_name << "* Bs = tgp_vals + B_st;\n";
  kernel << "    int A_sz = A_ed - A_st;\n";
  kernel << "    int B_sz = B_ed - B_st;\n";
  kernel << "    \n";
  kernel << "    // Find partition point\n";
  kernel << "    int sort_md = " << N_PER_THREAD << " * merge_lane;\n";
  kernel << "    int partition = merge_partition(As, Bs, A_sz, B_sz, sort_md);\n";
  kernel << "    \n";
  kernel << "    // Adjust pointers\n";
  kernel << "    As += partition;\n";
  kernel << "    Bs += sort_md - partition;\n";
  kernel << "    A_sz -= partition;\n";
  kernel << "    B_sz -= sort_md - partition;\n";
  kernel << "    \n";
  if (argsort) {
    kernel << "    __local const uint* As_idx = tgp_idxs + A_st + partition;\n";
    kernel << "    __local const uint* Bs_idx = tgp_idxs + B_st + sort_md - partition;\n";
  }
  kernel << "    \n";
  kernel << "    // Merge step: pick elements from A or B\n";
  kernel << "    int a_idx = 0;\n";
  kernel << "    int b_idx = 0;\n";
  kernel << "    for (int i = 0; i < " << N_PER_THREAD << "; ++i) {\n";
  kernel << "      " << type_name << " a_val = (a_idx < A_sz) ? As[a_idx] : " << init_val << ";\n";
  kernel << "      " << type_name << " b_val = (b_idx < B_sz) ? Bs[b_idx] : " << init_val << ";\n";
  kernel << "      bool pick_b = (b_idx < B_sz) && (a_idx >= A_sz || comp_less(b_val, a_val));\n";
  kernel << "      thread_vals[i] = pick_b ? b_val : a_val;\n";
  if (argsort) {
    kernel << "      thread_idxs[i] = pick_b ? Bs_idx[b_idx] : As_idx[a_idx];\n";
  }
  kernel << "      if (pick_b) b_idx++; else a_idx++;\n";
  kernel << "    }\n";
  kernel << "  }\n\n";

  // Write final output
  kernel << "  // Write to local memory then to global output\n";
  kernel << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  kernel << "  for (int i = 0; i < " << N_PER_THREAD << "; ++i) {\n";
  kernel << "    tgp_vals[base_idx + i] = thread_vals[i];\n";
  if (argsort) {
    kernel << "    tgp_idxs[base_idx + i] = thread_idxs[i];\n";
  }
  kernel << "  }\n";
  kernel << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  kernel << "  \n";
  kernel << "  // Write to global memory (strided for coalescing)\n";
  kernel << "  for (int i = lid; i < size_sorted_axis; i += " << block_threads << ") {\n";
  if (argsort) {
    kernel << "    output[i * out_stride_sorted_axis] = tgp_idxs[i];\n";
  } else {
    kernel << "    output[i * out_stride_sorted_axis] = tgp_vals[i];\n";
  }
  kernel << "  }\n";
  kernel << "}\n";

  return kernel.str();
}

// Generate multi-block sort kernel - each block sorts a portion
std::string generate_mb_block_sort_kernel(
    const std::string& type_name,
    int block_threads) {

  int n_per_block = block_threads * N_PER_THREAD;
  std::string init_val = get_init_value(type_name);

  std::ostringstream kernel;
  kernel << opencl::get_kernel_preamble();
  kernel << generate_comparison_helper(type_name);

  // Helpers
  kernel << "\ninline void val_swap(__private " << type_name << "* a, __private " << type_name << "* b) {\n";
  kernel << "  " << type_name << " t = *a; *a = *b; *b = t;\n";
  kernel << "}\n";
  kernel << "inline void idx_swap(__private uint* a, __private uint* b) {\n";
  kernel << "  uint t = *a; *a = *b; *b = t;\n";
  kernel << "}\n";

  kernel << "\ninline int merge_partition(\n";
  kernel << "    __local const " << type_name << "* As,\n";
  kernel << "    __local const " << type_name << "* Bs,\n";
  kernel << "    int A_sz, int B_sz, int sort_md) {\n";
  kernel << "  int A_st = max(0, sort_md - B_sz);\n";
  kernel << "  int A_ed = min(sort_md, A_sz);\n";
  kernel << "  while (A_st < A_ed) {\n";
  kernel << "    int md = A_st + (A_ed - A_st) / 2;\n";
  kernel << "    " << type_name << " a = As[md];\n";
  kernel << "    " << type_name << " b = Bs[sort_md - 1 - md];\n";
  kernel << "    if (comp_less(b, a)) A_ed = md;\n";
  kernel << "    else A_st = md + 1;\n";
  kernel << "  }\n";
  kernel << "  return A_ed;\n";
  kernel << "}\n";

  // Multi-block block sort kernel
  kernel << "\n__kernel void mb_block_sort_" << type_name << "(\n";
  kernel << "    __global const " << type_name << "* input,\n";
  kernel << "    __global " << type_name << "* out_vals,\n";
  kernel << "    __global uint* out_idxs,\n";
  kernel << "    const int size_sorted_axis,\n";
  kernel << "    const int stride_sorted_axis,\n";
  kernel << "    __local " << type_name << "* tgp_vals,\n";
  kernel << "    __local uint* tgp_idxs) {\n";

  kernel << "  int lid = get_local_id(0);\n";
  kernel << "  int block_id = get_group_id(0);\n";
  kernel << "  int segment_id = get_group_id(1);\n";
  kernel << "  int block_offset = block_id * " << n_per_block << ";\n";
  kernel << "  \n";
  kernel << "  // Offset by segment\n";
  kernel << "  input += segment_id * size_sorted_axis * stride_sorted_axis;\n";
  kernel << "  out_vals += segment_id * size_sorted_axis;\n";
  kernel << "  out_idxs += segment_id * size_sorted_axis;\n";
  kernel << "  \n";

  // Load into local memory
  kernel << "  for (int i = lid; i < " << n_per_block << "; i += " << block_threads << ") {\n";
  kernel << "    int idx = block_offset + i;\n";
  kernel << "    if (idx < size_sorted_axis) {\n";
  kernel << "      tgp_vals[i] = input[idx * stride_sorted_axis];\n";
  kernel << "    } else {\n";
  kernel << "      tgp_vals[i] = " << init_val << ";\n";
  kernel << "    }\n";
  kernel << "    tgp_idxs[i] = idx;\n";
  kernel << "  }\n";
  kernel << "  barrier(CLK_LOCAL_MEM_FENCE);\n\n";

  // Thread-level registers
  kernel << "  int base_idx = lid * " << N_PER_THREAD << ";\n";
  kernel << "  " << type_name << " thread_vals[" << N_PER_THREAD << "];\n";
  kernel << "  uint thread_idxs[" << N_PER_THREAD << "];\n";
  kernel << "  for (int i = 0; i < " << N_PER_THREAD << "; ++i) {\n";
  kernel << "    thread_vals[i] = tgp_vals[base_idx + i];\n";
  kernel << "    thread_idxs[i] = tgp_idxs[base_idx + i];\n";
  kernel << "  }\n\n";

  // Thread-level sort
  kernel << "  if ((block_offset + base_idx) < size_sorted_axis) {\n";
  kernel << "    for (int iter = 0; iter < " << N_PER_THREAD << "; ++iter) {\n";
  kernel << "      for (int j = (iter & 1); j < " << N_PER_THREAD - 1 << "; j += 2) {\n";
  kernel << "        if (comp_less(thread_vals[j + 1], thread_vals[j])) {\n";
  kernel << "          val_swap(&thread_vals[j], &thread_vals[j + 1]);\n";
  kernel << "          idx_swap(&thread_idxs[j], &thread_idxs[j + 1]);\n";
  kernel << "        }\n";
  kernel << "      }\n";
  kernel << "    }\n";
  kernel << "  }\n\n";

  // Merge iterations
  kernel << "  for (int merge_threads = 2; merge_threads <= " << block_threads << "; merge_threads *= 2) {\n";
  kernel << "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  kernel << "    for (int i = 0; i < " << N_PER_THREAD << "; ++i) {\n";
  kernel << "      tgp_vals[base_idx + i] = thread_vals[i];\n";
  kernel << "      tgp_idxs[base_idx + i] = thread_idxs[i];\n";
  kernel << "    }\n";
  kernel << "    barrier(CLK_LOCAL_MEM_FENCE);\n";
  kernel << "    \n";
  kernel << "    int merge_group = lid / merge_threads;\n";
  kernel << "    int merge_lane = lid % merge_threads;\n";
  kernel << "    int sort_sz = " << N_PER_THREAD << " * merge_threads;\n";
  kernel << "    int sort_st = sort_sz * merge_group;\n";
  kernel << "    \n";
  kernel << "    int A_st = sort_st, A_ed = sort_st + sort_sz / 2;\n";
  kernel << "    int B_st = A_ed, B_ed = sort_st + sort_sz;\n";
  kernel << "    __local const " << type_name << "* As = tgp_vals + A_st;\n";
  kernel << "    __local const " << type_name << "* Bs = tgp_vals + B_st;\n";
  kernel << "    int A_sz = A_ed - A_st, B_sz = B_ed - B_st;\n";
  kernel << "    \n";
  kernel << "    int sort_md = " << N_PER_THREAD << " * merge_lane;\n";
  kernel << "    int partition = merge_partition(As, Bs, A_sz, B_sz, sort_md);\n";
  kernel << "    \n";
  kernel << "    As += partition; Bs += sort_md - partition;\n";
  kernel << "    A_sz -= partition; B_sz -= sort_md - partition;\n";
  kernel << "    __local const uint* As_idx = tgp_idxs + A_st + partition;\n";
  kernel << "    __local const uint* Bs_idx = tgp_idxs + B_st + sort_md - partition;\n";
  kernel << "    \n";
  kernel << "    int a_idx = 0, b_idx = 0;\n";
  kernel << "    for (int i = 0; i < " << N_PER_THREAD << "; ++i) {\n";
  kernel << "      " << type_name << " a_val = (a_idx < A_sz) ? As[a_idx] : " << init_val << ";\n";
  kernel << "      " << type_name << " b_val = (b_idx < B_sz) ? Bs[b_idx] : " << init_val << ";\n";
  kernel << "      bool pick_b = (b_idx < B_sz) && (a_idx >= A_sz || comp_less(b_val, a_val));\n";
  kernel << "      thread_vals[i] = pick_b ? b_val : a_val;\n";
  kernel << "      thread_idxs[i] = pick_b ? Bs_idx[b_idx] : As_idx[a_idx];\n";
  kernel << "      if (pick_b) b_idx++; else a_idx++;\n";
  kernel << "    }\n";
  kernel << "  }\n\n";

  // Write output
  kernel << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  kernel << "  for (int i = 0; i < " << N_PER_THREAD << "; ++i) {\n";
  kernel << "    tgp_vals[base_idx + i] = thread_vals[i];\n";
  kernel << "    tgp_idxs[base_idx + i] = thread_idxs[i];\n";
  kernel << "  }\n";
  kernel << "  barrier(CLK_LOCAL_MEM_FENCE);\n\n";

  kernel << "  for (int i = lid; i < " << n_per_block << "; i += " << block_threads << ") {\n";
  kernel << "    int idx = block_offset + i;\n";
  kernel << "    if (idx < size_sorted_axis) {\n";
  kernel << "      out_vals[idx] = tgp_vals[i];\n";
  kernel << "      out_idxs[idx] = tgp_idxs[i];\n";
  kernel << "    }\n";
  kernel << "  }\n";
  kernel << "}\n";

  return kernel.str();
}

// Generate partition kernel for multi-block merge
std::string generate_mb_partition_kernel(
    const std::string& type_name,
    int block_threads) {

  int n_per_block = block_threads * N_PER_THREAD;

  std::ostringstream kernel;
  kernel << opencl::get_kernel_preamble();
  kernel << generate_comparison_helper(type_name);

  // Global memory merge partition
  kernel << "\ninline int global_merge_partition(\n";
  kernel << "    __global const " << type_name << "* As,\n";
  kernel << "    __global const " << type_name << "* Bs,\n";
  kernel << "    int A_sz, int B_sz, int sort_md) {\n";
  kernel << "  int A_st = max(0, sort_md - B_sz);\n";
  kernel << "  int A_ed = min(sort_md, A_sz);\n";
  kernel << "  while (A_st < A_ed) {\n";
  kernel << "    int md = A_st + (A_ed - A_st) / 2;\n";
  kernel << "    " << type_name << " a = As[md];\n";
  kernel << "    " << type_name << " b = Bs[sort_md - 1 - md];\n";
  kernel << "    if (comp_less(b, a)) A_ed = md;\n";
  kernel << "    else A_st = md + 1;\n";
  kernel << "  }\n";
  kernel << "  return A_ed;\n";
  kernel << "}\n";

  kernel << "\n__kernel void mb_partition_" << type_name << "(\n";
  kernel << "    __global uint* block_partitions,\n";
  kernel << "    __global const " << type_name << "* dev_vals,\n";
  kernel << "    const int size_sorted_axis,\n";
  kernel << "    const int merge_tiles,\n";
  kernel << "    const int n_blocks) {\n";

  kernel << "  int lid = get_local_id(0);\n";
  kernel << "  int segment_id = get_group_id(1);\n";
  kernel << "  int n_partitions = n_blocks + 1;\n";
  kernel << "  \n";
  kernel << "  block_partitions += segment_id * n_partitions;\n";
  kernel << "  dev_vals += segment_id * size_sorted_axis;\n";
  kernel << "  \n";
  kernel << "  for (int i = lid; i <= n_blocks; i += get_local_size(0)) {\n";
  kernel << "    int merge_group = i / merge_tiles;\n";
  kernel << "    int merge_lane = i % merge_tiles;\n";
  kernel << "    \n";
  kernel << "    int sort_sz = " << n_per_block << " * merge_tiles;\n";
  kernel << "    int sort_st = sort_sz * merge_group;\n";
  kernel << "    \n";
  kernel << "    int A_st = min(size_sorted_axis, sort_st);\n";
  kernel << "    int A_ed = min(size_sorted_axis, sort_st + sort_sz / 2);\n";
  kernel << "    int B_st = A_ed;\n";
  kernel << "    int B_ed = min(size_sorted_axis, B_st + sort_sz / 2);\n";
  kernel << "    \n";
  kernel << "    int partition_at = min(B_ed - A_st, " << n_per_block << " * merge_lane);\n";
  kernel << "    int partition = global_merge_partition(\n";
  kernel << "        dev_vals + A_st, dev_vals + B_st,\n";
  kernel << "        A_ed - A_st, B_ed - B_st, partition_at);\n";
  kernel << "    \n";
  kernel << "    block_partitions[i] = A_st + partition;\n";
  kernel << "  }\n";
  kernel << "}\n";

  return kernel.str();
}

// Generate merge kernel for multi-block sort
std::string generate_mb_merge_kernel(
    const std::string& type_name,
    int block_threads) {

  int n_per_block = block_threads * N_PER_THREAD;
  std::string init_val = get_init_value(type_name);

  std::ostringstream kernel;
  kernel << opencl::get_kernel_preamble();
  kernel << generate_comparison_helper(type_name);

  // Local memory merge partition
  kernel << "\ninline int local_merge_partition(\n";
  kernel << "    __local const " << type_name << "* As,\n";
  kernel << "    __local const " << type_name << "* Bs,\n";
  kernel << "    int A_sz, int B_sz, int sort_md) {\n";
  kernel << "  int A_st = max(0, sort_md - B_sz);\n";
  kernel << "  int A_ed = min(sort_md, A_sz);\n";
  kernel << "  while (A_st < A_ed) {\n";
  kernel << "    int md = A_st + (A_ed - A_st) / 2;\n";
  kernel << "    " << type_name << " a = As[md];\n";
  kernel << "    " << type_name << " b = Bs[sort_md - 1 - md];\n";
  kernel << "    if (comp_less(b, a)) A_ed = md;\n";
  kernel << "    else A_st = md + 1;\n";
  kernel << "  }\n";
  kernel << "  return A_ed;\n";
  kernel << "}\n";

  kernel << "\n__kernel void mb_merge_" << type_name << "(\n";
  kernel << "    __global const uint* block_partitions,\n";
  kernel << "    __global const " << type_name << "* dev_vals_in,\n";
  kernel << "    __global const uint* dev_idxs_in,\n";
  kernel << "    __global " << type_name << "* dev_vals_out,\n";
  kernel << "    __global uint* dev_idxs_out,\n";
  kernel << "    const int size_sorted_axis,\n";
  kernel << "    const int merge_tiles,\n";
  kernel << "    const int num_tiles,\n";
  kernel << "    __local " << type_name << "* tgp_vals,\n";
  kernel << "    __local uint* tgp_idxs) {\n";

  kernel << "  int lid = get_local_id(0);\n";
  kernel << "  int block_id = get_group_id(0);\n";
  kernel << "  int segment_id = get_group_id(1);\n";
  kernel << "  int n_partitions = num_tiles + 1;\n";
  kernel << "  \n";
  kernel << "  block_partitions += segment_id * n_partitions;\n";
  kernel << "  dev_vals_in += segment_id * size_sorted_axis;\n";
  kernel << "  dev_idxs_in += segment_id * size_sorted_axis;\n";
  kernel << "  dev_vals_out += segment_id * size_sorted_axis;\n";
  kernel << "  dev_idxs_out += segment_id * size_sorted_axis;\n";
  kernel << "  \n";
  kernel << "  int merge_group = block_id / merge_tiles;\n";
  kernel << "  int sort_st = " << n_per_block << " * merge_tiles * merge_group;\n";
  kernel << "  int sort_sz = " << n_per_block << " * merge_tiles;\n";
  kernel << "  int sort_md = " << n_per_block << " * block_id - sort_st;\n";
  kernel << "  \n";
  kernel << "  int A_st = block_partitions[block_id];\n";
  kernel << "  int A_ed = block_partitions[block_id + 1];\n";
  kernel << "  int B_st = min(size_sorted_axis, 2 * sort_st + sort_sz / 2 + sort_md - A_st);\n";
  kernel << "  int B_ed = min(size_sorted_axis, 2 * sort_st + sort_sz / 2 + sort_md + " << n_per_block << " - A_ed);\n";
  kernel << "  \n";
  kernel << "  if ((block_id % merge_tiles) == merge_tiles - 1) {\n";
  kernel << "    A_ed = min(size_sorted_axis, sort_st + sort_sz / 2);\n";
  kernel << "    B_ed = min(size_sorted_axis, sort_st + sort_sz);\n";
  kernel << "  }\n";
  kernel << "  \n";
  kernel << "  int A_sz = A_ed - A_st;\n";
  kernel << "  int B_sz = B_ed - B_st;\n";
  kernel << "  \n";

  // Load into local memory (strided for coalescing)
  kernel << "  // Load into local memory\n";
  kernel << "  for (int i = lid; i < " << n_per_block << "; i += " << block_threads << ") {\n";
  kernel << "    if (i < (A_sz + B_sz)) {\n";
  kernel << "      if (i < A_sz) {\n";
  kernel << "        tgp_vals[i] = dev_vals_in[A_st + i];\n";
  kernel << "        tgp_idxs[i] = dev_idxs_in[A_st + i];\n";
  kernel << "      } else {\n";
  kernel << "        tgp_vals[i] = dev_vals_in[B_st + i - A_sz];\n";
  kernel << "        tgp_idxs[i] = dev_idxs_in[B_st + i - A_sz];\n";
  kernel << "      }\n";
  kernel << "    } else {\n";
  kernel << "      tgp_vals[i] = " << init_val << ";\n";
  kernel << "      tgp_idxs[i] = 0;\n";
  kernel << "    }\n";
  kernel << "  }\n";
  kernel << "  barrier(CLK_LOCAL_MEM_FENCE);\n\n";

  // Find partition and merge into thread registers
  kernel << "  int sort_md_local = min(A_sz + B_sz, " << N_PER_THREAD << " * lid);\n";
  kernel << "  int A_st_local = local_merge_partition(tgp_vals, tgp_vals + A_sz, A_sz, B_sz, sort_md_local);\n";
  kernel << "  int A_ed_local = A_sz;\n";
  kernel << "  int B_st_local = sort_md_local - A_st_local;\n";
  kernel << "  int B_ed_local = B_sz;\n";
  kernel << "  int A_sz_local = A_ed_local - A_st_local;\n";
  kernel << "  int B_sz_local = B_ed_local - B_st_local;\n";
  kernel << "  \n";

  // Merge step
  kernel << "  " << type_name << " thread_vals[" << N_PER_THREAD << "];\n";
  kernel << "  uint thread_idxs[" << N_PER_THREAD << "];\n";
  kernel << "  int a_idx = 0, b_idx = 0;\n";
  kernel << "  for (int i = 0; i < " << N_PER_THREAD << "; ++i) {\n";
  kernel << "    " << type_name << " a_val = (a_idx < A_sz_local) ? tgp_vals[A_st_local + a_idx] : " << init_val << ";\n";
  kernel << "    " << type_name << " b_val = (b_idx < B_sz_local) ? tgp_vals[A_sz + B_st_local + b_idx] : " << init_val << ";\n";
  kernel << "    bool pick_b = (b_idx < B_sz_local) && (a_idx >= A_sz_local || comp_less(b_val, a_val));\n";
  kernel << "    thread_vals[i] = pick_b ? b_val : a_val;\n";
  kernel << "    thread_idxs[i] = pick_b ? tgp_idxs[A_sz + B_st_local + b_idx] : tgp_idxs[A_st_local + a_idx];\n";
  kernel << "    if (pick_b) b_idx++; else a_idx++;\n";
  kernel << "  }\n\n";

  // Write back
  kernel << "  barrier(CLK_LOCAL_MEM_FENCE);\n";
  kernel << "  int out_base = lid * " << N_PER_THREAD << ";\n";
  kernel << "  for (int i = 0; i < " << N_PER_THREAD << "; ++i) {\n";
  kernel << "    tgp_vals[out_base + i] = thread_vals[i];\n";
  kernel << "    tgp_idxs[out_base + i] = thread_idxs[i];\n";
  kernel << "  }\n";
  kernel << "  barrier(CLK_LOCAL_MEM_FENCE);\n\n";

  // Write output
  kernel << "  int base_idx = block_id * " << n_per_block << ";\n";
  kernel << "  for (int i = lid; i < " << n_per_block << "; i += " << block_threads << ") {\n";
  kernel << "    int idx = base_idx + i;\n";
  kernel << "    if (idx < size_sorted_axis) {\n";
  kernel << "      dev_vals_out[idx] = tgp_vals[i];\n";
  kernel << "      dev_idxs_out[idx] = tgp_idxs[i];\n";
  kernel << "    }\n";
  kernel << "  }\n";
  kernel << "}\n";

  return kernel.str();
}

// Single block sort - for small arrays that fit in one workgroup
void single_block_sort(
    const array& in,
    array& out,
    int axis,
    int block_threads,
    bool argsort,
    const Stream& s) {

  auto& dev = opencl::device(s.device);
  std::string type_name = opencl::type_to_name(in.dtype());
  int n_per_block = block_threads * N_PER_THREAD;

  // Calculate dimensions
  int n_rows = in.size() / in.shape(axis);
  int size_sorted_axis = in.shape(axis);
  int in_stride_sorted_axis = in.strides()[axis];
  int out_stride_sorted_axis = out.strides()[axis];

  // For contiguous case, find segment stride
  int in_stride_segment = 0;
  int out_stride_segment = 0;
  for (int i = 0; i < in.ndim(); ++i) {
    if (i != axis && in.shape(i) > 1) {
      in_stride_segment = in.strides()[i];
      out_stride_segment = out.strides()[i];
      break;
    }
  }
  if (in_stride_segment == 0) {
    // 1D case or all other dims are 1
    in_stride_segment = size_sorted_axis;
    out_stride_segment = size_sorted_axis;
  }

  // Generate and get kernel
  std::string suffix = argsort ? "_argsort" : "_sort";
  std::string kernel_name = "block_sort_" + type_name + suffix;
  std::string source = generate_block_sort_kernel(type_name, block_threads, argsort);

  cl_kernel kernel = dev.get_kernel(kernel_name, source);
  auto& encoder = dev.get_command_encoder(s.index);
  encoder.set_kernel(kernel);

  // Set arguments
  int arg = 0;
  encoder.set_input_array(in, arg++);
  encoder.set_output_array(out, arg++);
  encoder.set_bytes(size_sorted_axis, arg++);
  encoder.set_bytes(in_stride_sorted_axis, arg++);
  encoder.set_bytes(out_stride_sorted_axis, arg++);
  encoder.set_bytes(in_stride_segment, arg++);
  encoder.set_bytes(out_stride_segment, arg++);

  // Set local memory for values and indices
  size_t val_local_size = n_per_block * size_of(in.dtype());
  size_t idx_local_size = n_per_block * sizeof(uint32_t);
  clSetKernelArg(kernel, arg++, val_local_size, nullptr);
  clSetKernelArg(kernel, arg++, idx_local_size, nullptr);

  // Dispatch
  size_t global_size[3] = {(size_t)block_threads, (size_t)n_rows, 1};
  size_t local_size[3] = {(size_t)block_threads, 1, 1};
  encoder.dispatch_threads(global_size, local_size, 2);

  dev.end_encoding(s.index);
}

// Multi-block sort - for large arrays
void multi_block_sort(
    const array& in,
    array& out,
    int axis,
    int block_threads,
    int n_blocks,
    bool argsort,
    const Stream& s) {

  auto& dev = opencl::device(s.device);
  std::string type_name = opencl::type_to_name(in.dtype());
  int n_per_block = block_threads * N_PER_THREAD;

  int n_rows = in.size() / in.shape(axis);
  int size_sorted_axis = in.shape(axis);
  int stride_sorted_axis = in.strides()[axis];

  // Allocate temporary buffers
  size_t vals_size = n_rows * size_sorted_axis * size_of(in.dtype());
  size_t idxs_size = n_rows * size_sorted_axis * sizeof(uint32_t);
  size_t partitions_size = n_rows * (n_blocks + 1) * sizeof(uint32_t);

  cl_mem dev_vals_0 = clCreateBuffer(dev.context(), CL_MEM_READ_WRITE, vals_size, nullptr, nullptr);
  cl_mem dev_vals_1 = clCreateBuffer(dev.context(), CL_MEM_READ_WRITE, vals_size, nullptr, nullptr);
  cl_mem dev_idxs_0 = clCreateBuffer(dev.context(), CL_MEM_READ_WRITE, idxs_size, nullptr, nullptr);
  cl_mem dev_idxs_1 = clCreateBuffer(dev.context(), CL_MEM_READ_WRITE, idxs_size, nullptr, nullptr);
  cl_mem block_partitions = clCreateBuffer(dev.context(), CL_MEM_READ_WRITE, partitions_size, nullptr, nullptr);

  // Phase 1: Block sort
  {
    std::string kernel_name = "mb_block_sort_" + type_name;
    std::string source = generate_mb_block_sort_kernel(type_name, block_threads);
    cl_kernel kernel = dev.get_kernel(kernel_name, source);

    auto& encoder = dev.get_command_encoder(s.index);
    encoder.set_kernel(kernel);

    int arg = 0;
    encoder.set_input_array(in, arg++);
    clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dev_vals_0);
    clSetKernelArg(kernel, arg++, sizeof(cl_mem), &dev_idxs_0);
    encoder.set_bytes(size_sorted_axis, arg++);
    encoder.set_bytes(stride_sorted_axis, arg++);

    size_t val_local_size = n_per_block * size_of(in.dtype());
    size_t idx_local_size = n_per_block * sizeof(uint32_t);
    clSetKernelArg(kernel, arg++, val_local_size, nullptr);
    clSetKernelArg(kernel, arg++, idx_local_size, nullptr);

    size_t global_size[3] = {(size_t)(n_blocks * block_threads), (size_t)n_rows, 1};
    size_t local_size[3] = {(size_t)block_threads, 1, 1};
    encoder.dispatch_threads(global_size, local_size, 2);
    dev.end_encoding(s.index);
  }

  // Phase 2: Iterative merge
  bool ping = false;
  int partition_threads = std::min(n_blocks + 1, 256);

  for (int merge_tiles = 2; (merge_tiles / 2) < n_blocks; merge_tiles *= 2) {
    cl_mem vals_in = ping ? dev_vals_1 : dev_vals_0;
    cl_mem idxs_in = ping ? dev_idxs_1 : dev_idxs_0;
    cl_mem vals_out = ping ? dev_vals_0 : dev_vals_1;
    cl_mem idxs_out = ping ? dev_idxs_0 : dev_idxs_1;
    ping = !ping;

    // Partition kernel
    {
      std::string kernel_name = "mb_partition_" + type_name;
      std::string source = generate_mb_partition_kernel(type_name, block_threads);
      cl_kernel kernel = dev.get_kernel(kernel_name, source);

      auto& encoder = dev.get_command_encoder(s.index);
      encoder.set_kernel(kernel);

      int arg = 0;
      clSetKernelArg(kernel, arg++, sizeof(cl_mem), &block_partitions);
      clSetKernelArg(kernel, arg++, sizeof(cl_mem), &vals_in);
      encoder.set_bytes(size_sorted_axis, arg++);
      encoder.set_bytes(merge_tiles, arg++);
      encoder.set_bytes(n_blocks, arg++);

      size_t global_size[3] = {(size_t)partition_threads, (size_t)n_rows, 1};
      size_t local_size[3] = {(size_t)partition_threads, 1, 1};
      encoder.dispatch_threads(global_size, local_size, 2);
      dev.end_encoding(s.index);
    }

    // Merge kernel
    {
      std::string kernel_name = "mb_merge_" + type_name;
      std::string source = generate_mb_merge_kernel(type_name, block_threads);
      cl_kernel kernel = dev.get_kernel(kernel_name, source);

      auto& encoder = dev.get_command_encoder(s.index);
      encoder.set_kernel(kernel);

      int arg = 0;
      clSetKernelArg(kernel, arg++, sizeof(cl_mem), &block_partitions);
      clSetKernelArg(kernel, arg++, sizeof(cl_mem), &vals_in);
      clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idxs_in);
      clSetKernelArg(kernel, arg++, sizeof(cl_mem), &vals_out);
      clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idxs_out);
      encoder.set_bytes(size_sorted_axis, arg++);
      encoder.set_bytes(merge_tiles, arg++);
      encoder.set_bytes(n_blocks, arg++);

      size_t val_local_size = n_per_block * size_of(in.dtype());
      size_t idx_local_size = n_per_block * sizeof(uint32_t);
      clSetKernelArg(kernel, arg++, val_local_size, nullptr);
      clSetKernelArg(kernel, arg++, idx_local_size, nullptr);

      size_t global_size[3] = {(size_t)(n_blocks * block_threads), (size_t)n_rows, 1};
      size_t local_size[3] = {(size_t)block_threads, 1, 1};
      encoder.dispatch_threads(global_size, local_size, 2);
      dev.end_encoding(s.index);
    }
  }

  // Copy final result to output
  cl_mem final_vals = ping ? dev_vals_1 : dev_vals_0;
  cl_mem final_idxs = ping ? dev_idxs_1 : dev_idxs_0;

  {
    cl_command_queue queue = dev.get_queue(s);
    cl_mem out_buf = (cl_mem)const_cast<void*>(out.buffer().ptr());

    if (argsort) {
      // Copy indices directly if axis is last and contiguous
      if (axis == in.ndim() - 1 && out.flags().row_contiguous) {
        clEnqueueCopyBuffer(queue, final_idxs, out_buf,
            0, 0, n_rows * size_sorted_axis * sizeof(uint32_t), 0, nullptr, nullptr);
      } else {
        // Need strided copy - use the output strides
        // For simplicity, create a contiguous temp and do a general copy
        array temp({(int)n_rows, size_sorted_axis}, uint32, nullptr, {});
        temp.set_data(allocator::malloc(temp.nbytes()));
        cl_mem temp_buf = (cl_mem)const_cast<void*>(temp.buffer().ptr());
        clEnqueueCopyBuffer(queue, final_idxs, temp_buf,
            0, 0, n_rows * size_sorted_axis * sizeof(uint32_t), 0, nullptr, nullptr);
        dev.end_encoding(s.index);

        // Reshape and copy with proper strides
        auto out_shape = out.shape();
        Strides temp_strides(out.ndim());
        for (int i = axis + 1; i < out.ndim(); ++i) {
          temp_strides[i] = out.shape(axis);
        }
        temp_strides[axis] = 1;
        for (int i = 0; i < axis; ++i) {
          temp_strides[i] = 1;
          for (int j = i + 1; j < out.ndim(); ++j) {
            temp_strides[i] *= out.shape(j);
          }
        }

        copy_gpu_inplace(temp, out, out_shape, temp_strides, out.strides(), 0, 0,
            (axis == in.ndim() - 1) ? CopyType::Vector : CopyType::General, s);
      }
    } else {
      if (axis == in.ndim() - 1 && out.flags().row_contiguous) {
        clEnqueueCopyBuffer(queue, final_vals, out_buf,
            0, 0, n_rows * size_sorted_axis * size_of(in.dtype()), 0, nullptr, nullptr);
      } else {
        array temp({(int)n_rows, size_sorted_axis}, in.dtype(), nullptr, {});
        temp.set_data(allocator::malloc(temp.nbytes()));
        cl_mem temp_buf = (cl_mem)const_cast<void*>(temp.buffer().ptr());
        clEnqueueCopyBuffer(queue, final_vals, temp_buf,
            0, 0, n_rows * size_sorted_axis * size_of(in.dtype()), 0, nullptr, nullptr);
        dev.end_encoding(s.index);

        auto out_shape = out.shape();
        Strides temp_strides(out.ndim());
        for (int i = axis + 1; i < out.ndim(); ++i) {
          temp_strides[i] = out.shape(axis);
        }
        temp_strides[axis] = 1;
        for (int i = 0; i < axis; ++i) {
          temp_strides[i] = 1;
          for (int j = i + 1; j < out.ndim(); ++j) {
            temp_strides[i] *= out.shape(j);
          }
        }

        copy_gpu_inplace(temp, out, out_shape, temp_strides, out.strides(), 0, 0,
            (axis == in.ndim() - 1) ? CopyType::Vector : CopyType::General, s);
      }
    }
    dev.end_encoding(s.index);
  }

  // Clean up
  clReleaseMemObject(dev_vals_0);
  clReleaseMemObject(dev_vals_1);
  clReleaseMemObject(dev_idxs_0);
  clReleaseMemObject(dev_idxs_1);
  clReleaseMemObject(block_partitions);
}

void gpu_merge_sort(
    const array& in,
    array& out,
    int axis_,
    bool argsort,
    const Stream& s) {

  int axis = axis_ < 0 ? axis_ + in.ndim() : axis_;
  int size_sorted_axis = in.shape(axis);

  // Get block size
  int block_threads = get_block_threads(size_sorted_axis, size_of(in.dtype()));
  int n_per_block = block_threads * N_PER_THREAD;
  int n_blocks = (size_sorted_axis + n_per_block - 1) / n_per_block;

  OPENCL_DEBUG_LOG("[sort_gpu] axis=" << axis << " size=" << size_sorted_axis
      << " block_threads=" << block_threads << " n_blocks=" << n_blocks
      << " argsort=" << argsort);

  if (n_blocks > 1) {
    multi_block_sort(in, out, axis, block_threads, n_blocks, argsort, s);
  } else {
    single_block_sort(in, out, axis, block_threads, argsort, s);
  }
}

}  // namespace

void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];

  // Make input contiguous AND ensure it starts at buffer base.
  // Sliced arrays have non-zero offset which the kernel doesn't account for.
  auto in_contig = in;
  if (!in.flags().row_contiguous || in.offset() != 0) {
    in_contig = array(in.shape(), in.dtype(), nullptr, {});
    copy_gpu(in, in_contig, CopyType::General, stream());
  }

  out.set_data(opencl::allocator().malloc(out.nbytes()));
  gpu_merge_sort(in_contig, out, axis_, false, stream());
}

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];

  // Make input contiguous AND ensure it starts at buffer base.
  // Sliced arrays have non-zero offset which the kernel doesn't account for.
  auto in_contig = in;
  if (!in.flags().row_contiguous || in.offset() != 0) {
    in_contig = array(in.shape(), in.dtype(), nullptr, {});
    copy_gpu(in, in_contig, CopyType::General, stream());
  }

  out.set_data(opencl::allocator().malloc(out.nbytes()));
  gpu_merge_sort(in_contig, out, axis_, true, stream());
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  // For now, implement partition as a full sort
  // This is inefficient but correct
  auto& in = inputs[0];

  // Make input contiguous AND ensure it starts at buffer base.
  // Sliced arrays have non-zero offset which the kernel doesn't account for.
  auto in_contig = in;
  if (!in.flags().row_contiguous || in.offset() != 0) {
    in_contig = array(in.shape(), in.dtype(), nullptr, {});
    copy_gpu(in, in_contig, CopyType::General, stream());
  }

  out.set_data(opencl::allocator().malloc(out.nbytes()));
  gpu_merge_sort(in_contig, out, axis_, false, stream());
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  // For now, implement argpartition as a full argsort
  // This is inefficient but correct
  auto& in = inputs[0];

  // Make input contiguous AND ensure it starts at buffer base.
  // Sliced arrays have non-zero offset which the kernel doesn't account for -
  // it only receives buffer().ptr() but not the data offset.
  auto in_contig = in;
  if (!in.flags().row_contiguous || in.offset() != 0) {
    in_contig = array(in.shape(), in.dtype(), nullptr, {});
    copy_gpu(in, in_contig, CopyType::General, stream());
  }

  out.set_data(opencl::allocator().malloc(out.nbytes()));
  gpu_merge_sort(in_contig, out, axis_, true, stream());
}

}  // namespace mlx::core
