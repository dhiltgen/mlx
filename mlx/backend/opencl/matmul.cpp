// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/opencl/types.h"  // For centralized type utilities
#include "mlx/backend/opencl/debug.h"  // For OPENCL_DEBUG_LOG
#include "mlx/backend/gpu/copy.h"  // For contiguous_copy_gpu
#include "mlx/backend/common/matmul.h"  // For collapse_batches
#include "mlx/primitives.h"

#include <sstream>
#include <numeric>

namespace mlx::core {

namespace {

// ============================================================================
// Phase M3: MatmulParams and kernel selection
// ============================================================================

// Kernel types for matmul dispatch
enum class MatmulKernel {
  NAIVE,              // Simple kernel, one thread per output element
  TILED_LOCAL_MEM,    // Local memory tiled GEMM (64x64 tiles, 256 threads)
  TILED_2X2,          // Register-blocked 2x2 for transposed B
  LARGE_VOCAB_GEMV,   // Parallel reduction GEMV for M=1, large N
  GEMV,               // Standard GEMV (disabled on Adreno)
};

// Parameters for kernel selection - extracted from matrix dimensions and layout
struct MatmulParams {
  int M, N, K;
  bool b_transposed;
  size_t batch_size;
  bool simple_batch;  // Single collapsed batch dimension
  Dtype dtype;
};

// Select the best kernel based on matrix parameters
// Encapsulates all selection logic in one place for testability
MatmulKernel select_kernel(const MatmulParams& p) {
  bool use_batched = (p.batch_size > 1) && p.simple_batch;

  // Large vocabulary GEMV: M=1 with very large N (vocab projection)
  // Uses parallel reduction with subgroups - only for transposed B (coalesced access)
  if (p.M == 1 && p.N >= 10000 && p.b_transposed && !use_batched) {
    return MatmulKernel::LARGE_VOCAB_GEMV;
  }

  // Tiled kernels for large matrices
  bool large_matrix = (p.M >= 32 && p.N >= 256 && p.K >= 256);

  if (large_matrix) {
    if (p.b_transposed) {
      // Register-blocked 2x2 for transposed B - no local memory, safe for batched
      return MatmulKernel::TILED_2X2;
    } else if (!use_batched) {
      // Local memory tiled GEMM - only for non-batched (local memory issues)
      return MatmulKernel::TILED_LOCAL_MEM;
    }
  }

  // Default: naive kernel
  return MatmulKernel::NAIVE;
}

// Get kernel name suffix for kernel type
const char* kernel_type_name(MatmulKernel k) {
  switch (k) {
    case MatmulKernel::NAIVE: return "naive";
    case MatmulKernel::TILED_LOCAL_MEM: return "tiled";
    case MatmulKernel::TILED_2X2: return "tiled_2x2";
    case MatmulKernel::LARGE_VOCAB_GEMV: return "gemv_large_vocab";
    case MatmulKernel::GEMV: return "gemv";
    default: return "unknown";
  }
}

// ============================================================================
// Kernel generators using centralized type utilities
// ============================================================================

// Helper to generate unified matmul kernel with optional epilogue (TransformAxpby pattern)
// When use_epilogue is true, computes: D = alpha * (A @ B) + beta * C
// When use_epilogue is false, computes: D = A @ B
// When batched is true, uses get_global_id(2) for batch index and computes offsets from strides
std::string gen_matmul_kernel(
    const std::string& kernel_name,
    Dtype dtype,
    bool b_transposed,
    bool use_epilogue,
    bool batched = false) {

  std::ostringstream k;

  // Use centralized type utilities
  std::string type_name = opencl::type_to_name(dtype);
  bool needs_conv = opencl::needs_float_conversion(dtype);

  // Get conversion function names for use in generated kernel
  auto conv = opencl::get_type_conversion(dtype);
  std::string to_float = conv.needs_read_convert ? conv.read_fn : "";
  std::string from_float = conv.needs_write_convert ? conv.write_fn : "";

  // Kernel signature
  k << "__kernel void " << kernel_name << "(\n";
  k << "    __global const " << type_name << "* A,\n";
  k << "    __global const " << type_name << "* B,\n";
  if (use_epilogue) {
    k << "    __global const " << type_name << "* C,\n";
  }
  k << "    __global " << type_name << "* D,\n";
  k << "    int M, int N, int K, int ldb,\n";
  if (use_epilogue) {
    k << "    float alpha, float beta,\n";
  }
  if (batched) {
    // Batched kernel: batch strides and base offsets for sliced arrays
    k << "    int A_batch_stride, int B_batch_stride,\n";
    if (use_epilogue) {
      k << "    int C_batch_stride,\n";
    }
    k << "    int D_batch_stride,\n";
    k << "    int A_base_offset, int B_base_offset) {\n";
  } else {
    if (use_epilogue) {
      k << "    int A_offset, int B_offset, int C_offset, int D_offset) {\n";
    } else {
      k << "    int A_offset, int B_offset, int D_offset) {\n";
    }
  }

  k << "\n";
  k << "    int i = get_global_id(0);\n";
  k << "    int j = get_global_id(1);\n";

  if (batched) {
    k << "    int batch = get_global_id(2);\n";
    k << "\n";
    k << "    // Compute batch offsets from strides, adding base offsets for sliced arrays\n";
    k << "    int A_offset = A_base_offset + batch * A_batch_stride;\n";
    k << "    int B_offset = B_base_offset + batch * B_batch_stride;\n";
    k << "    int D_offset = batch * D_batch_stride;\n";
    if (use_epilogue) {
      k << "    int C_offset = batch * C_batch_stride;\n";
    }
  }

  k << "\n";
  k << "    if (i >= M || j >= N) return;\n";
  k << "\n";
  k << "    float sum = 0.0f;\n";
  k << "    int a_base = A_offset + i * K;\n";

  if (b_transposed) {
    k << "    int b_base = B_offset + j * ldb;\n";
  } else {
    k << "    int b_col = B_offset + j;\n";
  }

  k << "\n";
  k << "    int kk = 0;\n";
  k << "    for (; kk + 7 < K; kk += 8) {\n";

  // Unrolled inner loop
  for (int u = 0; u < 8; u++) {
    std::string a_access = "A[a_base + kk + " + std::to_string(u) + "]";
    std::string b_access;
    if (b_transposed) {
      b_access = "B[b_base + kk + " + std::to_string(u) + "]";
    } else {
      b_access = "B[b_col + (kk + " + std::to_string(u) + ") * ldb]";
    }

    if (!to_float.empty()) {
      k << "        sum += " << to_float << "(" << a_access << ") * " << to_float << "(" << b_access << ");\n";
    } else {
      k << "        sum += " << a_access << " * " << b_access << ";\n";
    }
  }

  k << "    }\n";
  k << "    for (; kk < K; kk++) {\n";

  std::string a_access = "A[a_base + kk]";
  std::string b_access = b_transposed ? "B[b_base + kk]" : "B[b_col + kk * ldb]";

  if (!to_float.empty()) {
    k << "        sum += " << to_float << "(" << a_access << ") * " << to_float << "(" << b_access << ");\n";
  } else {
    k << "        sum += " << a_access << " * " << b_access << ";\n";
  }

  k << "    }\n";
  k << "\n";

  // Epilogue
  std::string out_idx = "D_offset + i * N + j";

  if (use_epilogue) {
    k << "    float result = alpha * sum;\n";
    if (!to_float.empty()) {
      k << "    result += beta * " << to_float << "(C[C_offset + i * N + j]);\n";
      k << "    D[" << out_idx << "] = " << from_float << "(result);\n";
    } else {
      k << "    result += beta * C[C_offset + i * N + j];\n";
      k << "    D[" << out_idx << "] = result;\n";
    }
  } else {
    if (!from_float.empty()) {
      k << "    D[" << out_idx << "] = " << from_float << "(sum);\n";
    } else {
      k << "    D[" << out_idx << "] = sum;\n";
    }
  }

  k << "}\n";

  return k.str();
}

// Generate GEMV kernel - unified across bf16/f16/f32
// Handles both transposed and non-transposed B matrix
std::string gen_gemv_kernel(
    const std::string& kernel_name,
    Dtype dtype,
    bool b_transposed) {

  std::ostringstream k;

  // Use centralized type utilities
  std::string type_name = opencl::type_to_name(dtype);
  auto conv = opencl::get_type_conversion(dtype);
  bool needs_conv = conv.needs_read_convert;

  k << "__kernel void " << kernel_name << "(\n";
  k << "    __global const " << type_name << "* A,\n";
  k << "    __global const " << type_name << "* B,\n";
  k << "    __global " << type_name << "* C,\n";
  k << "    int M, int N, int K, int ldb,\n";
  k << "    int A_offset, int B_offset, int C_offset) {\n";
  k << "\n";
  k << "    int j = get_global_id(0);  // Output column\n";
  k << "    int m = get_global_id(1);  // Output row\n";
  k << "\n";
  k << "    if (j >= N || m >= M) return;\n";
  k << "\n";
  k << "    float sum = 0.0f;\n";
  k << "    int a_base = A_offset + m * K;\n";

  if (b_transposed) {
    k << "    int b_base = B_offset + j * ldb;\n";
  } else {
    k << "    int b_col = B_offset + j;\n";
  }

  k << "\n";
  k << "    int kk = 0;\n";
  k << "    for (; kk + 7 < K; kk += 8) {\n";

  // Generate unrolled read expressions for A
  for (int u = 0; u < 8; u++) {
    std::string a_idx = "a_base + kk + " + std::to_string(u);
    std::string a_read = needs_conv
        ? conv.read_fn + "(A[" + a_idx + "])"
        : "A[" + a_idx + "]";
    k << "        float a" << u << " = " << a_read << ";\n";
  }
  k << "\n";

  // Generate unrolled multiply-accumulate
  for (int u = 0; u < 8; u++) {
    std::string b_idx;
    if (b_transposed) {
      b_idx = "b_base + kk + " + std::to_string(u);
    } else {
      b_idx = "b_col + (kk + " + std::to_string(u) + ") * ldb";
    }
    std::string b_read = needs_conv
        ? conv.read_fn + "(B[" + b_idx + "])"
        : "B[" + b_idx + "]";
    k << "        sum += a" << u << " * " << b_read << ";\n";
  }

  k << "    }\n";
  k << "    for (; kk < K; kk++) {\n";

  // Scalar tail loop
  std::string a_tail_read = needs_conv
      ? conv.read_fn + "(A[a_base + kk])"
      : "A[a_base + kk]";
  std::string b_tail_idx = b_transposed ? "b_base + kk" : "b_col + kk * ldb";
  std::string b_tail_read = needs_conv
      ? conv.read_fn + "(B[" + b_tail_idx + "])"
      : "B[" + b_tail_idx + "]";
  k << "        sum += " << a_tail_read << " * " << b_tail_read << ";\n";
  k << "    }\n";
  k << "\n";

  // Write output
  std::string write_val = needs_conv
      ? conv.write_fn + "(sum)"
      : "sum";
  k << "    C[C_offset + m * N + j] = " << write_val << ";\n";
  k << "}\n";

  return k.str();
}

// Generate 2x2 register-blocked kernel for transposed B (A @ B.T)
// Each thread computes a 2x2 block of outputs using register blocking
// No local memory/barriers - avoids Adreno sync issues
// Works for both batched (3D dispatch) and non-batched (2D dispatch) modes
std::string gen_tiled_2x2_kernel(
    const std::string& kernel_name,
    Dtype dtype,
    bool batched) {

  std::ostringstream k;

  // Use centralized type utilities
  std::string type_name = opencl::type_to_name(dtype);
  auto conv = opencl::get_type_conversion(dtype);
  bool needs_conv = conv.needs_read_convert;
  bool is_f32 = (dtype == float32);

  k << "#define BM 2\n";
  k << "#define BN 2\n\n";

  k << "__kernel void " << kernel_name << "(\n";
  k << "    __global const " << type_name << "* A,\n";
  k << "    __global const " << type_name << "* B,\n";
  k << "    __global " << type_name << "* C,\n";
  k << "    int M, int N, int K, int ldb,\n";
  if (batched) {
    k << "    int A_batch_stride, int B_batch_stride, int C_batch_stride) {\n";
  } else {
    k << "    int A_offset, int B_offset, int C_offset) {\n";
  }
  k << "\n";
  k << "    int baseRow = get_global_id(0) * BM;\n";
  k << "    int baseCol = get_global_id(1) * BN;\n";

  if (batched) {
    k << "    int batch = get_global_id(2);\n";
    k << "\n";
    k << "    int A_offset = batch * A_batch_stride;\n";
    k << "    int B_offset = batch * B_batch_stride;\n";
    k << "    int C_offset = batch * C_batch_stride;\n";
  }

  k << "\n";
  k << "    float acc00 = 0.0f, acc01 = 0.0f;\n";
  k << "    float acc10 = 0.0f, acc11 = 0.0f;\n";
  k << "\n";
  k << "    int k = 0;\n";

  // For float32, use vectorized loads (vload4)
  if (is_f32) {
    k << "    for (; k + 3 < K; k += 4) {\n";
    k << "        float4 a0_vec = (baseRow < M) ? vload4(0, A + A_offset + baseRow * K + k) : (float4)(0.0f);\n";
    k << "        float4 a1_vec = (baseRow+1 < M) ? vload4(0, A + A_offset + (baseRow+1) * K + k) : (float4)(0.0f);\n";
    k << "\n";
    k << "        float4 b0_vec = (baseCol < N) ? vload4(0, B + B_offset + baseCol * ldb + k) : (float4)(0.0f);\n";
    k << "        float4 b1_vec = (baseCol+1 < N) ? vload4(0, B + B_offset + (baseCol+1) * ldb + k) : (float4)(0.0f);\n";
    k << "\n";
    k << "        acc00 += a0_vec.s0 * b0_vec.s0 + a0_vec.s1 * b0_vec.s1 + a0_vec.s2 * b0_vec.s2 + a0_vec.s3 * b0_vec.s3;\n";
    k << "        acc01 += a0_vec.s0 * b1_vec.s0 + a0_vec.s1 * b1_vec.s1 + a0_vec.s2 * b1_vec.s2 + a0_vec.s3 * b1_vec.s3;\n";
    k << "        acc10 += a1_vec.s0 * b0_vec.s0 + a1_vec.s1 * b0_vec.s1 + a1_vec.s2 * b0_vec.s2 + a1_vec.s3 * b0_vec.s3;\n";
    k << "        acc11 += a1_vec.s0 * b1_vec.s0 + a1_vec.s1 * b1_vec.s1 + a1_vec.s2 * b1_vec.s2 + a1_vec.s3 * b1_vec.s3;\n";
    k << "    }\n";
  } else {
    // For bf16/f16, use scalar loads with type conversion
    k << "    for (; k + 3 < K; k += 4) {\n";

    // Load A values for two rows
    for (int i = 0; i < 4; i++) {
      std::string a0_idx = "A_offset + baseRow * K + k + " + std::to_string(i);
      std::string a1_idx = "A_offset + (baseRow+1) * K + k + " + std::to_string(i);
      std::string a0_read = conv.read_fn + "(A[" + a0_idx + "])";
      std::string a1_read = conv.read_fn + "(A[" + a1_idx + "])";
      k << "        float a0" << i << " = (baseRow < M) ? " << a0_read << " : 0.0f;\n";
      k << "        float a1" << i << " = (baseRow+1 < M) ? " << a1_read << " : 0.0f;\n";
    }
    k << "\n";

    // Load B values for two columns
    for (int i = 0; i < 4; i++) {
      std::string b0_idx = "B_offset + baseCol * ldb + k + " + std::to_string(i);
      std::string b1_idx = "B_offset + (baseCol+1) * ldb + k + " + std::to_string(i);
      std::string b0_read = conv.read_fn + "(B[" + b0_idx + "])";
      std::string b1_read = conv.read_fn + "(B[" + b1_idx + "])";
      k << "        float b0" << i << " = (baseCol < N) ? " << b0_read << " : 0.0f;\n";
      k << "        float b1" << i << " = (baseCol+1 < N) ? " << b1_read << " : 0.0f;\n";
    }
    k << "\n";

    k << "        acc00 += a00 * b00 + a01 * b01 + a02 * b02 + a03 * b03;\n";
    k << "        acc01 += a00 * b10 + a01 * b11 + a02 * b12 + a03 * b13;\n";
    k << "        acc10 += a10 * b00 + a11 * b01 + a12 * b02 + a13 * b03;\n";
    k << "        acc11 += a10 * b10 + a11 * b11 + a12 * b12 + a13 * b13;\n";
    k << "    }\n";
  }

  k << "\n";
  k << "    // Handle remainder\n";
  k << "    for (; k < K; k++) {\n";

  // Scalar tail loop
  if (is_f32) {
    k << "        float a0 = (baseRow < M) ? A[A_offset + baseRow * K + k] : 0.0f;\n";
    k << "        float a1 = (baseRow+1 < M) ? A[A_offset + (baseRow+1) * K + k] : 0.0f;\n";
    k << "        float b0 = (baseCol < N) ? B[B_offset + baseCol * ldb + k] : 0.0f;\n";
    k << "        float b1 = (baseCol+1 < N) ? B[B_offset + (baseCol+1) * ldb + k] : 0.0f;\n";
  } else {
    k << "        float a0 = (baseRow < M) ? " << conv.read_fn << "(A[A_offset + baseRow * K + k]) : 0.0f;\n";
    k << "        float a1 = (baseRow+1 < M) ? " << conv.read_fn << "(A[A_offset + (baseRow+1) * K + k]) : 0.0f;\n";
    k << "        float b0 = (baseCol < N) ? " << conv.read_fn << "(B[B_offset + baseCol * ldb + k]) : 0.0f;\n";
    k << "        float b1 = (baseCol+1 < N) ? " << conv.read_fn << "(B[B_offset + (baseCol+1) * ldb + k]) : 0.0f;\n";
  }
  k << "        acc00 += a0 * b0;\n";
  k << "        acc01 += a0 * b1;\n";
  k << "        acc10 += a1 * b0;\n";
  k << "        acc11 += a1 * b1;\n";
  k << "    }\n";
  k << "\n";

  // Store results
  std::string write_00 = is_f32 ? "acc00" : conv.write_fn + "(acc00)";
  std::string write_01 = is_f32 ? "acc01" : conv.write_fn + "(acc01)";
  std::string write_10 = is_f32 ? "acc10" : conv.write_fn + "(acc10)";
  std::string write_11 = is_f32 ? "acc11" : conv.write_fn + "(acc11)";

  k << "    if (baseRow < M && baseCol < N)\n";
  k << "        C[C_offset + baseRow * N + baseCol] = " << write_00 << ";\n";
  k << "    if (baseRow < M && baseCol+1 < N)\n";
  k << "        C[C_offset + baseRow * N + baseCol + 1] = " << write_01 << ";\n";
  k << "    if (baseRow+1 < M && baseCol < N)\n";
  k << "        C[C_offset + (baseRow+1) * N + baseCol] = " << write_10 << ";\n";
  k << "    if (baseRow+1 < M && baseCol+1 < N)\n";
  k << "        C[C_offset + (baseRow+1) * N + baseCol + 1] = " << write_11 << ";\n";
  k << "}\n";

  return k.str();
}

// Generate tiled local-memory GEMM kernel
// High-performance tiled GEMM
// Uses BM=64, BN=64, BK=16 tiles, each thread computes TM=4 x TN=4 = 16 outputs
// Work group: 16 x 16 = 256 threads
std::string gen_tiled_local_mem_kernel(
    const std::string& kernel_name,
    Dtype dtype) {

  std::ostringstream k;

  // Use centralized type utilities
  std::string type_name = opencl::type_to_name(dtype);
  auto conv = opencl::get_type_conversion(dtype);
  bool needs_conv = conv.needs_read_convert;

  // Tile dimensions
  k << "// Tile dimensions\n";
  k << "#define BM 64\n";
  k << "#define BN 64\n";
  k << "#define BK 16\n";
  k << "#define TM 4\n";
  k << "#define TN 4\n\n";

  k << "__kernel void " << kernel_name << "(\n";
  k << "    __global const " << type_name << "* A,\n";
  k << "    __global const " << type_name << "* B,\n";
  k << "    __global " << type_name << "* C,\n";
  k << "    int M, int N, int K,\n";
  k << "    int A_offset, int B_offset, int C_offset) {\n";
  k << "\n";
  k << "    // Tile indices\n";
  k << "    const int ir = get_group_id(0);  // Row tile index\n";
  k << "    const int ic = get_group_id(1);  // Column tile index\n";
  k << "\n";
  k << "    // Thread position in workgroup (256 threads = 16x16)\n";
  k << "    const int tid = get_local_id(0) + get_local_id(1) * get_local_size(0);\n";
  k << "    const int th_r = tid % (BM / TM);  // Thread row in tile (0-15)\n";
  k << "    const int th_c = tid / (BM / TM);  // Thread col in tile (0-15)\n";
  k << "\n";
  k << "    // Load distribution\n";
  k << "    const int loadr_a = tid % (BK / 4);  // 4 elements per load\n";
  k << "    const int loadc_a = tid / (BK / 4);\n";
  k << "    const int loadr_b = tid % (BK / 4);\n";
  k << "    const int loadc_b = tid / (BK / 4);\n";
  k << "\n";
  k << "    // Local memory tiles\n";
  k << "    __local float buf_a[BM * BK];\n";
  k << "    __local float buf_b[BN * BK];\n";
  k << "\n";
  k << "    // Per-thread accumulators\n";
  k << "    float sums[TM * TN];\n";
  k << "    for (int i = 0; i < TM * TN; i++) sums[i] = 0.0f;\n";
  k << "\n";
  k << "    // Register cache\n";
  k << "    float cache_a[TM];\n";
  k << "    float cache_b[TN];\n";
  k << "\n";
  k << "    // Process K in BK-sized chunks\n";
  k << "    for (int block = 0; block < K; block += BK) {\n";
  k << "        // Cooperative load of A tile into local memory\n";
  k << "        for (int l = 0; l < BM; l += 64) {\n";
  k << "            int row = ir * BM + loadc_a + l;\n";
  k << "            int col = block + loadr_a * 4;\n";
  k << "            if (row < M && col < K) {\n";

  // Generate loads for A with type conversion
  for (int i = 0; i < 4; i++) {
    std::string idx_str = (i == 0) ? "col" : "col + " + std::to_string(i);
    std::string bound_check = (i == 0) ? "" : "if (" + idx_str + " < K) ";
    std::string load_expr = needs_conv
        ? conv.read_fn + "(A[A_offset + row * K + " + idx_str + "])"
        : "A[A_offset + row * K + " + idx_str + "]";
    k << "                " << bound_check << "buf_a[(loadr_a * 4 + " << i << ") * BM + loadc_a + l] = " << load_expr << ";\n";
  }

  k << "            } else {\n";
  for (int i = 0; i < 4; i++) {
    k << "                buf_a[(loadr_a * 4 + " << i << ") * BM + loadc_a + l] = 0.0f;\n";
  }
  k << "            }\n";
  k << "        }\n";
  k << "\n";
  k << "        // Cooperative load of B tile into local memory\n";
  k << "        for (int l = 0; l < BN; l += 64) {\n";
  k << "            int row = block + loadr_b * 4;\n";
  k << "            int col = ic * BN + loadc_b + l;\n";
  k << "            if (row < K && col < N) {\n";

  // Generate loads for B with type conversion
  for (int i = 0; i < 4; i++) {
    std::string row_offset = (i == 0) ? "row" : "(row + " + std::to_string(i) + ")";
    std::string bound_check = (i == 0) ? "" : "if (" + row_offset + " < K) ";
    std::string load_expr = needs_conv
        ? conv.read_fn + "(B[B_offset + " + row_offset + " * N + col])"
        : "B[B_offset + " + row_offset + " * N + col]";
    k << "                " << bound_check << "buf_b[(loadr_b * 4 + " << i << ") * BN + loadc_b + l] = " << load_expr << ";\n";
  }

  k << "            } else {\n";
  for (int i = 0; i < 4; i++) {
    k << "                buf_b[(loadr_b * 4 + " << i << ") * BN + loadc_b + l] = 0.0f;\n";
  }
  k << "            }\n";
  k << "        }\n";
  k << "\n";
  k << "        barrier(CLK_LOCAL_MEM_FENCE);\n";
  k << "\n";
  k << "        // Compute using register caching\n";
  k << "        for (int kk = 0; kk < BK; kk++) {\n";
  k << "            // Cache A column in registers\n";
  k << "            for (int j = 0; j < TM; j++) {\n";
  k << "                cache_a[j] = buf_a[kk * BM + th_r * TM + j];\n";
  k << "            }\n";
  k << "            // Cache B row in registers\n";
  k << "            for (int j = 0; j < TN; j++) {\n";
  k << "                cache_b[j] = buf_b[kk * BN + th_c * TN + j];\n";
  k << "            }\n";
  k << "\n";
  k << "            // Compute outer product\n";
  k << "            for (int cc = 0; cc < TN; cc++) {\n";
  k << "                for (int cr = 0; cr < TM; cr++) {\n";
  k << "                    sums[cc * TM + cr] = mad(cache_a[cr], cache_b[cc], sums[cc * TM + cr]);\n";
  k << "                }\n";
  k << "            }\n";
  k << "        }\n";
  k << "\n";
  k << "        barrier(CLK_LOCAL_MEM_FENCE);\n";
  k << "    }\n";
  k << "\n";
  k << "    // Store results (row-major: C[row, col] = C[row * N + col])\n";
  k << "    const int dr = ir * BM + th_r * TM;\n";
  k << "    const int dc = ic * BN + th_c * TN;\n";
  k << "\n";
  k << "    for (int cr = 0; cr < TM; cr++) {\n";
  k << "        for (int cc = 0; cc < TN; cc++) {\n";
  k << "            if (dr + cr < M && dc + cc < N) {\n";

  // Write with type conversion
  std::string write_expr = needs_conv
      ? conv.write_fn + "(sums[cc * TM + cr])"
      : "sums[cc * TM + cr]";
  k << "                C[C_offset + (dr + cr) * N + (dc + cc)] = " << write_expr << ";\n";

  k << "            }\n";
  k << "        }\n";
  k << "    }\n";
  k << "}\n";

  return k.str();
}

// Generate tiled local-memory AddMM kernel
// Same high-performance tiled GEMM pattern but with fused epilogue:
// out[i,j] = alpha * (A @ B)[i,j] + beta * C[i,j]
// Uses BM=64, BN=64, BK=16 tiles, each thread computes TM=4 x TN=4 = 16 outputs
// Work group: 16 x 16 = 256 threads
std::string gen_tiled_addmm_kernel(
    const std::string& kernel_name,
    Dtype dtype) {

  std::ostringstream k;

  // Use centralized type utilities
  std::string type_name = opencl::type_to_name(dtype);
  auto conv = opencl::get_type_conversion(dtype);
  bool needs_conv = conv.needs_read_convert;

  // Tile dimensions
  k << "// Tile dimensions\n";
  k << "#define BM 64\n";
  k << "#define BN 64\n";
  k << "#define BK 16\n";
  k << "#define TM 4\n";
  k << "#define TN 4\n\n";

  k << "__kernel void " << kernel_name << "(\n";
  k << "    __global const " << type_name << "* A,\n";
  k << "    __global const " << type_name << "* B,\n";
  k << "    __global const " << type_name << "* C,\n";  // AddMM bias input
  k << "    __global " << type_name << "* out,\n";
  k << "    int M, int N, int K,\n";
  k << "    float alpha, float beta,\n";  // Epilogue scalars
  k << "    int A_offset, int B_offset, int C_offset, int out_offset) {\n";
  k << "\n";
  k << "    // Tile indices\n";
  k << "    const int ir = get_group_id(0);  // Row tile index\n";
  k << "    const int ic = get_group_id(1);  // Column tile index\n";
  k << "\n";
  k << "    // Thread position in workgroup (256 threads = 16x16)\n";
  k << "    const int tid = get_local_id(0) + get_local_id(1) * get_local_size(0);\n";
  k << "    const int th_r = tid % (BM / TM);  // Thread row in tile (0-15)\n";
  k << "    const int th_c = tid / (BM / TM);  // Thread col in tile (0-15)\n";
  k << "\n";
  k << "    // Load distribution\n";
  k << "    const int loadr_a = tid % (BK / 4);  // 4 elements per load\n";
  k << "    const int loadc_a = tid / (BK / 4);\n";
  k << "    const int loadr_b = tid % (BK / 4);\n";
  k << "    const int loadc_b = tid / (BK / 4);\n";
  k << "\n";
  k << "    // Local memory tiles\n";
  k << "    __local float buf_a[BM * BK];\n";
  k << "    __local float buf_b[BN * BK];\n";
  k << "\n";
  k << "    // Per-thread accumulators\n";
  k << "    float sums[TM * TN];\n";
  k << "    for (int i = 0; i < TM * TN; i++) sums[i] = 0.0f;\n";
  k << "\n";
  k << "    // Register cache\n";
  k << "    float cache_a[TM];\n";
  k << "    float cache_b[TN];\n";
  k << "\n";
  k << "    // Process K in BK-sized chunks\n";
  k << "    for (int block = 0; block < K; block += BK) {\n";
  k << "        // Cooperative load of A tile into local memory\n";
  k << "        for (int l = 0; l < BM; l += 64) {\n";
  k << "            int row = ir * BM + loadc_a + l;\n";
  k << "            int col = block + loadr_a * 4;\n";
  k << "            if (row < M && col < K) {\n";

  // Generate loads for A with type conversion
  for (int i = 0; i < 4; i++) {
    std::string idx_str = (i == 0) ? "col" : "col + " + std::to_string(i);
    std::string bound_check = (i == 0) ? "" : "if (" + idx_str + " < K) ";
    std::string load_expr = needs_conv
        ? conv.read_fn + "(A[A_offset + row * K + " + idx_str + "])"
        : "A[A_offset + row * K + " + idx_str + "]";
    k << "                " << bound_check << "buf_a[(loadr_a * 4 + " << i << ") * BM + loadc_a + l] = " << load_expr << ";\n";
  }

  k << "            } else {\n";
  for (int i = 0; i < 4; i++) {
    k << "                buf_a[(loadr_a * 4 + " << i << ") * BM + loadc_a + l] = 0.0f;\n";
  }
  k << "            }\n";
  k << "        }\n";
  k << "\n";
  k << "        // Cooperative load of B tile into local memory\n";
  k << "        for (int l = 0; l < BN; l += 64) {\n";
  k << "            int row = block + loadr_b * 4;\n";
  k << "            int col = ic * BN + loadc_b + l;\n";
  k << "            if (row < K && col < N) {\n";

  // Generate loads for B with type conversion
  for (int i = 0; i < 4; i++) {
    std::string row_offset = (i == 0) ? "row" : "(row + " + std::to_string(i) + ")";
    std::string bound_check = (i == 0) ? "" : "if (" + row_offset + " < K) ";
    std::string load_expr = needs_conv
        ? conv.read_fn + "(B[B_offset + " + row_offset + " * N + col])"
        : "B[B_offset + " + row_offset + " * N + col]";
    k << "                " << bound_check << "buf_b[(loadr_b * 4 + " << i << ") * BN + loadc_b + l] = " << load_expr << ";\n";
  }

  k << "            } else {\n";
  for (int i = 0; i < 4; i++) {
    k << "                buf_b[(loadr_b * 4 + " << i << ") * BN + loadc_b + l] = 0.0f;\n";
  }
  k << "            }\n";
  k << "        }\n";
  k << "\n";
  k << "        barrier(CLK_LOCAL_MEM_FENCE);\n";
  k << "\n";
  k << "        // Compute using register caching\n";
  k << "        for (int kk = 0; kk < BK; kk++) {\n";
  k << "            // Cache A column in registers\n";
  k << "            for (int j = 0; j < TM; j++) {\n";
  k << "                cache_a[j] = buf_a[kk * BM + th_r * TM + j];\n";
  k << "            }\n";
  k << "            // Cache B row in registers\n";
  k << "            for (int j = 0; j < TN; j++) {\n";
  k << "                cache_b[j] = buf_b[kk * BN + th_c * TN + j];\n";
  k << "            }\n";
  k << "\n";
  k << "            // Compute outer product\n";
  k << "            for (int cc = 0; cc < TN; cc++) {\n";
  k << "                for (int cr = 0; cr < TM; cr++) {\n";
  k << "                    sums[cc * TM + cr] = mad(cache_a[cr], cache_b[cc], sums[cc * TM + cr]);\n";
  k << "                }\n";
  k << "            }\n";
  k << "        }\n";
  k << "\n";
  k << "        barrier(CLK_LOCAL_MEM_FENCE);\n";
  k << "    }\n";
  k << "\n";
  k << "    // Store results with fused epilogue: out = alpha * (A @ B) + beta * C\n";
  k << "    const int dr = ir * BM + th_r * TM;\n";
  k << "    const int dc = ic * BN + th_c * TN;\n";
  k << "\n";
  k << "    for (int cr = 0; cr < TM; cr++) {\n";
  k << "        for (int cc = 0; cc < TN; cc++) {\n";
  k << "            if (dr + cr < M && dc + cc < N) {\n";
  k << "                int idx = (dr + cr) * N + (dc + cc);\n";

  // Read C with type conversion and apply epilogue
  std::string c_read = needs_conv
      ? conv.read_fn + "(C[C_offset + idx])"
      : "C[C_offset + idx]";
  k << "                float c_val = " << c_read << ";\n";
  k << "                float result = alpha * sums[cc * TM + cr] + beta * c_val;\n";

  // Write with type conversion
  std::string write_expr = needs_conv
      ? conv.write_fn + "(result)"
      : "result";
  k << "                out[out_offset + idx] = " << write_expr << ";\n";

  k << "            }\n";
  k << "        }\n";
  k << "    }\n";
  k << "}\n";

  return k.str();
}

// Generate large vocab GEMV kernel with parallel reduction
// Uses subgroup operations for fast reduction across 64 threads
std::string gen_gemv_large_vocab_kernel(
    const std::string& kernel_name,
    Dtype dtype,
    bool b_transposed) {

  std::ostringstream k;

  // Use centralized type utilities
  std::string type_name = opencl::type_to_name(dtype);
  auto conv = opencl::get_type_conversion(dtype);
  bool needs_conv = conv.needs_read_convert;

  k << "#pragma OPENCL EXTENSION cl_khr_subgroups : enable\n\n";
  k << "#define WG_SIZE 64\n\n";

  k << "__kernel void " << kernel_name << "(\n";
  k << "    __global const " << type_name << "* A,\n";
  k << "    __global const " << type_name << "* B,\n";
  k << "    __global " << type_name << "* C,\n";
  k << "    int M, int N, int K, int ldb,\n";
  k << "    int A_offset, int B_offset, int C_offset) {\n";
  k << "\n";
  k << "    int j = get_group_id(0);  // Output column (one per workgroup)\n";
  k << "    if (j >= N) return;\n";
  k << "\n";
  k << "    int lid = get_local_id(0);\n";

  if (b_transposed) {
    k << "    int b_base = B_offset + j * ldb;\n";
  }

  k << "\n";
  k << "    // Each thread computes partial sum\n";
  k << "    float sum = 0.0f;\n";
  k << "    for (int kk = lid; kk < K; kk += WG_SIZE) {\n";

  // A access (always contiguous for M=1)
  std::string a_read = needs_conv
      ? conv.read_fn + "(A[A_offset + kk])"
      : "A[A_offset + kk]";

  // B access depends on transpose
  std::string b_read;
  if (b_transposed) {
    b_read = needs_conv
        ? conv.read_fn + "(B[b_base + kk])"
        : "B[b_base + kk]";
  } else {
    b_read = needs_conv
        ? conv.read_fn + "(B[B_offset + kk * ldb + j])"
        : "B[B_offset + kk * ldb + j]";
  }

  k << "        sum += " << a_read << " * " << b_read << ";\n";
  k << "    }\n";
  k << "\n";
  k << "    // Subgroup reduction (wave size = 64)\n";
  k << "    sum = sub_group_reduce_add(sum);\n";
  k << "\n";
  k << "    if (lid == 0) {\n";

  std::string write_val = needs_conv
      ? conv.write_fn + "(sum)"
      : "sum";
  k << "        C[C_offset + j] = " << write_val << ";\n";

  k << "    }\n";
  k << "}\n";

  return k.str();
}

// Generate scale kernel for AddMM K=0 case (out = beta * C)
std::string gen_scale_kernel(
    const std::string& kernel_name,
    Dtype dtype) {

  std::ostringstream k;

  std::string type_name = opencl::type_to_name(dtype);
  auto conv = opencl::get_type_conversion(dtype);
  bool needs_conv = conv.needs_read_convert;

  k << "__kernel void " << kernel_name << "(\n";
  k << "    __global const " << type_name << "* C,\n";
  k << "    __global " << type_name << "* out,\n";
  k << "    float beta, int size) {\n";
  k << "    int idx = get_global_id(0);\n";
  k << "    if (idx >= size) return;\n";

  if (needs_conv) {
    k << "    float val = " << conv.read_fn << "(C[idx]);\n";
    k << "    out[idx] = " << conv.write_fn << "(beta * val);\n";
  } else {
    k << "    out[idx] = beta * C[idx];\n";
  }

  k << "}\n";
  return k.str();
}

// Generate AddMM GEMV kernel (fused matmul + epilogue for M<=4)
// out[i,j] = alpha * sum_k(A[i,k] * B[k,j]) + beta * C[i,j]
std::string gen_addmm_gemv_kernel(
    const std::string& kernel_name,
    Dtype dtype) {

  std::ostringstream k;

  std::string type_name = opencl::type_to_name(dtype);
  auto conv = opencl::get_type_conversion(dtype);
  bool needs_conv = conv.needs_read_convert;

  k << "__kernel void " << kernel_name << "(\n";
  k << "    __global const " << type_name << "* A,\n";
  k << "    __global const " << type_name << "* B,\n";
  k << "    __global const " << type_name << "* C,\n";
  k << "    __global " << type_name << "* out,\n";
  k << "    int M, int N, int K,\n";
  k << "    float alpha, float beta,\n";
  k << "    int A_offset, int B_offset, int C_offset, int out_offset) {\n";
  k << "\n";
  k << "    int j = get_global_id(0);  // Output column\n";
  k << "    int m = get_global_id(1);  // Output row\n";
  k << "\n";
  k << "    if (j >= N || m >= M) return;\n";
  k << "\n";
  k << "    float sum = 0.0f;\n";
  k << "    int a_base = A_offset + m * K;\n";
  k << "    int b_col = B_offset + j;\n";
  k << "\n";
  k << "    // Unroll by 8 for better ILP\n";
  k << "    int kk = 0;\n";
  k << "    for (; kk + 7 < K; kk += 8) {\n";

  // Generate unrolled A and B reads
  for (int i = 0; i < 8; i++) {
    std::string a_idx = "a_base + kk + " + std::to_string(i);
    std::string a_read = needs_conv
        ? conv.read_fn + "(A[" + a_idx + "])"
        : "A[" + a_idx + "]";
    k << "        float a" << i << " = " << a_read << ";\n";
  }
  k << "\n";
  for (int i = 0; i < 8; i++) {
    std::string b_idx = "b_col + (kk + " + std::to_string(i) + ") * N";
    std::string b_read = needs_conv
        ? conv.read_fn + "(B[" + b_idx + "])"
        : "B[" + b_idx + "]";
    k << "        float b" << i << " = " << b_read << ";\n";
  }
  k << "\n";
  k << "        sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4 + a5 * b5 + a6 * b6 + a7 * b7;\n";
  k << "    }\n";
  k << "    // Handle remainder\n";
  k << "    for (; kk < K; kk++) {\n";

  std::string a_tail = needs_conv
      ? conv.read_fn + "(A[a_base + kk])"
      : "A[a_base + kk]";
  std::string b_tail = needs_conv
      ? conv.read_fn + "(B[b_col + kk * N])"
      : "B[b_col + kk * N]";
  k << "        sum += " << a_tail << " * " << b_tail << ";\n";
  k << "    }\n";
  k << "\n";

  // Read C and write output
  std::string c_read = needs_conv
      ? conv.read_fn + "(C[C_offset + m * N + j])"
      : "C[C_offset + m * N + j]";
  k << "    float c_val = " << c_read << ";\n";

  std::string out_write = needs_conv
      ? conv.write_fn + "(alpha * sum + beta * c_val)"
      : "alpha * sum + beta * c_val";
  k << "    out[out_offset + m * N + j] = " << out_write << ";\n";

  k << "}\n";
  return k.str();
}

} // anonymous namespace

void Matmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);

  const array& a = inputs[0];
  const array& b = inputs[1];

  OPENCL_DEBUG_LOG("[Matmul::eval_gpu] a.shape=" << a.shape() << " b.shape=" << b.shape()
            << " out.shape=" << out.shape());
  OPENCL_DEBUG_LOG("[Matmul::eval_gpu] a.status=" << (a.status() == array::Status::evaluated ? "evaluated" : "not_eval")
            << " a.row_contig=" << a.flags().row_contiguous);
  OPENCL_DEBUG_LOG("[Matmul::eval_gpu] b.status=" << (b.status() == array::Status::evaluated ? "evaluated" : "not_eval")
            << " b.row_contig=" << b.flags().row_contiguous);

  // Support float32, float16, and bfloat16
  bool is_bf16 = (a.dtype() == bfloat16 && b.dtype() == bfloat16);
  bool is_f16 = (a.dtype() == float16 && b.dtype() == float16);
  bool is_f32 = (a.dtype() == float32 && b.dtype() == float32);
  if (!is_bf16 && !is_f16 && !is_f32) {
    throw std::runtime_error("[Matmul::eval_gpu] Only float32, float16, and bfloat16 supported on GPU");
  }

  // Check contiguity and handle transposed matrices efficiently
  // Like Metal, we detect column-contiguous matrices (transposed row-contiguous)
  // and handle them with strided access instead of copying
  array a_contig = a;
  array b_work = b;
  bool b_transposed = false;
  int ldb = 0;  // Leading dimension of B

  if (!a.flags().row_contiguous) {
    OPENCL_DEBUG_LOG("[Matmul::eval_gpu] Matrix A not row-contiguous, making copy");
    a_contig = contiguous_copy_gpu(a, stream());
  }

  // Check B's memory layout
  // Detect column-contiguous (transposed) B: strides are [1, K] instead of [K, 1]
  // For column-contiguous B, we use a subgroup-based kernel with coalesced access
  if (b.flags().row_contiguous) {
    b_transposed = false;
    ldb = b.shape(-1);  // N
    OPENCL_DEBUG_LOG("[Matmul::eval_gpu] B is row-contiguous, ldb=" << ldb);
  } else {
    // Check if B is column-contiguous (transposed row-contiguous)
    // B has shape [K, N] and strides [stride_k, stride_n]
    // Column-contiguous means: stride_k == 1 and stride_n == K
    auto& b_strides = b.strides();
    size_t stx = b_strides[b_strides.size() - 2];  // stride for K dimension
    size_t sty = b_strides[b_strides.size() - 1];  // stride for N dimension
    size_t K_dim = b.shape(-2);

    if (stx == 1 && sty == K_dim) {
      // Column-contiguous (transposed): each column is contiguous
      // Use stride-aware kernel with subgroup reduction for coalesced access
      b_transposed = true;
      ldb = K_dim;  // Leading dimension is K (column stride)
      OPENCL_DEBUG_LOG("[Matmul::eval_gpu] B is column-contiguous (transposed), ldb=" << ldb);
    } else {
      // General non-contiguous case: make a contiguous copy
      OPENCL_DEBUG_LOG("[Matmul::eval_gpu] Matrix B not contiguous (strides=" << stx << "," << sty << "), making copy");
      b_work = contiguous_copy_gpu(b, stream());
      b_transposed = false;
      ldb = b_work.shape(-1);
    }
  }

  // Allocate output
  out.set_data(opencl::allocator().malloc(out.nbytes()));

  auto& dev = opencl::device(Device::gpu);
  auto& s = stream();
  auto& encoder = dev.get_command_encoder(s.index);

  // Matrix dimensions: A is M x K, B is K x N, C is M x N
  int M = a_contig.shape(-2);
  int K = a_contig.shape(-1);
  int N = b_work.shape(-1);

  OPENCL_DEBUG_LOG("[Matmul::eval_gpu] M=" << M << " K=" << K << " N=" << N);

  // Handle batched matmul
  auto [batch_shape, a_batch_strides, b_batch_strides] = collapse_batches(a_contig, b_work);
  size_t batch_size = std::accumulate(batch_shape.begin(), batch_shape.end(), 1, std::multiplies<int>());

  OPENCL_DEBUG_LOG("[Matmul::eval_gpu] batch_size=" << batch_size);

  // Use batched dispatch when batch_size > 1 with simple batch shape
  bool use_batched_dispatch = (batch_size > 1) && (batch_shape.size() == 1);

  // Select kernel using centralized selection logic (Phase M3)
  MatmulParams params{M, N, K, b_transposed, batch_size, batch_shape.size() == 1, a.dtype()};
  MatmulKernel kernel_type = select_kernel(params);

  OPENCL_DEBUG_LOG("[Matmul::eval_gpu] Selected kernel: " << kernel_type_name(kernel_type)
            << " batched=" << use_batched_dispatch);

  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();

  std::string kernel_name;
  std::string type_suffix = opencl::type_to_suffix(a.dtype());
  std::string bt_suffix = b_transposed ? "_bt" : "";

  // Tile parameters for tiled local-memory kernel
  const int BM_TILE = 64, BN_TILE = 64, TM_TILE = 4, TN_TILE = 4;
  const int WG_M = BM_TILE / TM_TILE, WG_N = BN_TILE / TN_TILE;

  // Generate kernel based on selected type
  if (kernel_type == MatmulKernel::TILED_2X2 && use_batched_dispatch) {
    // Batched register-blocked kernel for transposed B
    kernel_name = "matmul_tiled_batched_" + type_suffix + "_bt";
    kernel_source << gen_tiled_2x2_kernel(kernel_name, a.dtype(), true);
  } else if (use_batched_dispatch) {
    // Batched naive kernel
    kernel_name = "matmul_batched_" + type_suffix + bt_suffix;
    kernel_source << gen_matmul_kernel(kernel_name, a.dtype(), b_transposed, false, true);
  } else if (kernel_type == MatmulKernel::LARGE_VOCAB_GEMV) {
    // Large vocab GEMV with parallel reduction
    kernel_name = "gemv_large_vocab_" + type_suffix + bt_suffix;
    kernel_source << gen_gemv_large_vocab_kernel(kernel_name, a.dtype(), b_transposed);
  } else if (kernel_type == MatmulKernel::GEMV) {
    // Standard GEMV (currently disabled)
    kernel_name = "gemv_" + type_suffix + bt_suffix;
    kernel_source << gen_gemv_kernel(kernel_name, a.dtype(), b_transposed);
  } else if (kernel_type == MatmulKernel::TILED_LOCAL_MEM) {
    // High-performance tiled GEMM - unified generator
    kernel_name = "matmul_tiled_" + type_suffix;
    kernel_source << gen_tiled_local_mem_kernel(kernel_name, a.dtype());
  } else if (kernel_type == MatmulKernel::TILED_2X2) {
    // Register-blocked kernel for transposed B - unified generator
    kernel_name = "matmul_tiled_" + type_suffix + "_bt";
    kernel_source << gen_tiled_2x2_kernel(kernel_name, a.dtype(), false);
  } else {
    // Naive kernel for small matrices - unified generator
    kernel_name = "matmul_" + type_suffix + "_naive" + bt_suffix;
    kernel_source << gen_matmul_kernel(kernel_name, a.dtype(), b_transposed, false, false);
  }

  std::string source_str = kernel_source.str();

  OPENCL_DEBUG_LOG("[Matmul::eval_gpu] Compiling kernel: " << kernel_name);

  // Get or compile kernel
  // Use CL2.0 for subgroup operations in large vocab GEMV
  std::string compile_opts = (kernel_type == MatmulKernel::LARGE_VOCAB_GEMV) ? "-cl-std=CL2.0" : "";
  cl_kernel kernel = dev.get_kernel(kernel_name, source_str, compile_opts);

  // Calculate output batch stride
  size_t out_batch_stride = M * N;

  // Dispatch kernel
  size_t local_size[3];
  size_t global_size[3];

  if (kernel_type == MatmulKernel::TILED_2X2 && use_batched_dispatch) {
    // Batched tiled dispatch: 3D dispatch with batch_size in Z dimension
    // Each thread computes 2x2 block of outputs
    const int BM = 2, BN = 2;
    local_size[0] = 8;
    local_size[1] = 8;
    local_size[2] = 1;  // No local grouping in Z
    int numThreadsM = (M + BM - 1) / BM;
    int numThreadsN = (N + BN - 1) / BN;
    global_size[0] = static_cast<size_t>((numThreadsM + 7) / 8 * 8);
    global_size[1] = static_cast<size_t>((numThreadsN + 7) / 8 * 8);
    global_size[2] = batch_size;
  } else if (use_batched_dispatch) {
    // Batched dispatch: 3D dispatch with batch_size in Z dimension
    local_size[0] = 16;
    local_size[1] = 16;
    local_size[2] = 1;
    global_size[0] = static_cast<size_t>((M + 15) / 16 * 16);
    global_size[1] = static_cast<size_t>((N + 15) / 16 * 16);
    global_size[2] = batch_size;
  } else if (kernel_type == MatmulKernel::LARGE_VOCAB_GEMV) {
    // Large vocab GEMV: 1D dispatch, 64 threads per workgroup
    local_size[0] = 64;
    local_size[1] = 0;
    local_size[2] = 0;
    global_size[0] = static_cast<size_t>(N) * 64;
    global_size[1] = 0;
    global_size[2] = 0;
  } else if (kernel_type == MatmulKernel::GEMV) {
    // Standard GEMV
    local_size[0] = 256;
    local_size[1] = 1;
    local_size[2] = 0;
    global_size[0] = static_cast<size_t>((N + 255) / 256 * 256);
    global_size[1] = static_cast<size_t>(M);
    global_size[2] = 0;
  } else if (kernel_type == MatmulKernel::TILED_LOCAL_MEM) {
    // Tiled GEMM: 2D dispatch with 16x16 = 256 threads per workgroup
    local_size[0] = WG_M;
    local_size[1] = WG_N;
    local_size[2] = 0;
    int numTilesM = (M + BM_TILE - 1) / BM_TILE;
    int numTilesN = (N + BN_TILE - 1) / BN_TILE;
    global_size[0] = static_cast<size_t>(numTilesM * WG_M);
    global_size[1] = static_cast<size_t>(numTilesN * WG_N);
    global_size[2] = 0;
  } else if (kernel_type == MatmulKernel::TILED_2X2) {
    // Register-blocked 2x2: each thread computes 2x2 block
    const int BM = 2, BN = 2;
    local_size[0] = 8;
    local_size[1] = 8;
    local_size[2] = 0;
    int numThreadsM = (M + BM - 1) / BM;
    int numThreadsN = (N + BN - 1) / BN;
    global_size[0] = static_cast<size_t>((numThreadsM + 7) / 8 * 8);
    global_size[1] = static_cast<size_t>((numThreadsN + 7) / 8 * 8);
    global_size[2] = 0;
  } else {
    // Naive: one thread per output element
    local_size[0] = 16;
    local_size[1] = 16;
    local_size[2] = 0;
    global_size[0] = static_cast<size_t>((M + 15) / 16 * 16);
    global_size[1] = static_cast<size_t>((N + 15) / 16 * 16);
    global_size[2] = 0;
  }

  if (use_batched_dispatch) {
    // Single-dispatch batched GEMM: dispatch all batches in one kernel call
    // The kernel uses get_global_id(2) to get batch index and computes offsets from strides
    encoder.set_kernel(kernel);
    encoder.set_input_array(a_contig, 0);
    encoder.set_input_array(b_work, 1);
    encoder.set_output_array(out, 2);
    encoder.set_bytes(static_cast<int>(M), 3);
    encoder.set_bytes(static_cast<int>(N), 4);
    encoder.set_bytes(static_cast<int>(K), 5);
    encoder.set_bytes(static_cast<int>(ldb), 6);

    // For batched dispatch, we need to compute flat batch strides
    // The kernel assumes linear batch indexing with flat strides
    // For simple batching (single batch dim), this is straightforward
    // For multi-dim batching, we need the product of inner dims
    int a_batch_stride_flat = 0;
    int b_batch_stride_flat = 0;
    if (!a_batch_strides.empty()) {
      a_batch_stride_flat = static_cast<int>(a_batch_strides[0]);
    }
    if (!b_batch_strides.empty()) {
      b_batch_stride_flat = static_cast<int>(b_batch_strides[0]);
    }
    int out_batch_stride_int = static_cast<int>(out_batch_stride);

    encoder.set_bytes(a_batch_stride_flat, 7);
    encoder.set_bytes(b_batch_stride_flat, 8);
    encoder.set_bytes(out_batch_stride_int, 9);
    // Base offsets for sliced arrays
    int a_base_offset = static_cast<int>(a_contig.offset() / a_contig.itemsize());
    int b_base_offset = static_cast<int>(b_work.offset() / b_work.itemsize());
    encoder.set_bytes(a_base_offset, 10);
    encoder.set_bytes(b_base_offset, 11);

    OPENCL_DEBUG_LOG("[Matmul::eval_gpu] Batched dispatch: A_stride=" << a_batch_stride_flat
              << " B_stride=" << b_batch_stride_flat << " D_stride=" << out_batch_stride_int
              << " A_base=" << a_base_offset << " B_base=" << b_base_offset);

    encoder.dispatch_threads(global_size, local_size, 3);
  } else {
    // Non-batched dispatch: loop through batches (batch_size == 1 typically)
    for (size_t batch = 0; batch < batch_size; batch++) {
      // Calculate batch offsets using strides, starting from array base offsets
      size_t a_offset = a_contig.offset() / a_contig.itemsize();
      size_t b_offset = b_work.offset() / b_work.itemsize();
      size_t temp_batch = batch;

      for (int i = batch_shape.size() - 1; i >= 0; i--) {
        size_t batch_idx = temp_batch % batch_shape[i];
        temp_batch /= batch_shape[i];
        a_offset += batch_idx * a_batch_strides[i];
        b_offset += batch_idx * b_batch_strides[i];
      }

      size_t c_offset = batch * out_batch_stride;

      OPENCL_DEBUG_LOG("[Matmul::eval_gpu] Batch " << batch << "/" << batch_size
                << ": a_offset=" << a_offset << " b_offset=" << b_offset << " c_offset=" << c_offset);

      encoder.set_kernel(kernel);
      cl_mem a_buf = static_cast<cl_mem>(const_cast<void*>(a_contig.buffer().ptr()));
      cl_mem b_buf = static_cast<cl_mem>(const_cast<void*>(b_work.buffer().ptr()));
      cl_mem out_buf = static_cast<cl_mem>(out.buffer().ptr());
      OPENCL_DEBUG_LOG("[Matmul::eval_gpu] Reading from a_buf=" << a_buf << " b_buf=" << b_buf
                << " writing to out_buf=" << out_buf);
      encoder.set_input_array(a_contig, 0);
      encoder.set_input_array(b_work, 1);
      encoder.set_output_array(out, 2);
      encoder.set_bytes(static_cast<int>(M), 3);
      encoder.set_bytes(static_cast<int>(N), 4);
      encoder.set_bytes(static_cast<int>(K), 5);

      if (kernel_type == MatmulKernel::LARGE_VOCAB_GEMV) {
        // Large vocab GEMV: 1D dispatch
        encoder.set_bytes(static_cast<int>(ldb), 6);
        encoder.set_bytes(static_cast<int>(a_offset), 7);
        encoder.set_bytes(static_cast<int>(b_offset), 8);
        encoder.set_bytes(static_cast<int>(c_offset), 9);
        encoder.dispatch_threads(global_size, local_size, 1);
      } else if (kernel_type == MatmulKernel::GEMV) {
        // Standard GEMV: 2D dispatch
        encoder.set_bytes(static_cast<int>(ldb), 6);
        encoder.set_bytes(static_cast<int>(a_offset), 7);
        encoder.set_bytes(static_cast<int>(b_offset), 8);
        encoder.set_bytes(static_cast<int>(c_offset), 9);
        encoder.dispatch_threads(global_size, local_size, 2);
      } else if (kernel_type == MatmulKernel::TILED_LOCAL_MEM) {
        // Tiled GEMM: no ldb parameter
        encoder.set_bytes(static_cast<int>(a_offset), 6);
        encoder.set_bytes(static_cast<int>(b_offset), 7);
        encoder.set_bytes(static_cast<int>(c_offset), 8);
        encoder.dispatch_threads(global_size, local_size, 2);
      } else if (kernel_type == MatmulKernel::TILED_2X2) {
        // Register-blocked 2x2: with ldb
        encoder.set_bytes(static_cast<int>(ldb), 6);
        encoder.set_bytes(static_cast<int>(a_offset), 7);
        encoder.set_bytes(static_cast<int>(b_offset), 8);
        encoder.set_bytes(static_cast<int>(c_offset), 9);
        encoder.dispatch_threads(global_size, local_size, 2);
      } else {
        // Naive kernel: with ldb
        encoder.set_bytes(static_cast<int>(ldb), 6);
        encoder.set_bytes(static_cast<int>(a_offset), 7);
        encoder.set_bytes(static_cast<int>(b_offset), 8);
        encoder.set_bytes(static_cast<int>(c_offset), 9);
        encoder.dispatch_threads(global_size, local_size, 2);
      }
    }
  }

  // Add arrays as temporaries so they aren't freed until kernel completes
  dev.add_temporary(a_contig, s.index);
  dev.add_temporary(b_work, s.index);
  dev.add_temporary(out, s.index);

  // Ensure kernel completion for consistent results
  dev.end_encoding(s.index);

  OPENCL_DEBUG_LOG("[Matmul::eval_gpu] Dispatched " << batch_size << " batch(es): global=("
            << global_size[0] << "," << global_size[1] << ") local=("
            << local_size[0] << "," << local_size[1] << ")");
}

void AddMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 3);

  auto& a = inputs[0];
  auto& b = inputs[1];
  auto& c = inputs[2];

  auto [alpha, beta] = state();

  OPENCL_DEBUG_LOG("[AddMM::eval_gpu] a.shape=" << a.shape() << " b.shape=" << b.shape()
            << " c.shape=" << c.shape() << " out.shape=" << out.shape());
  OPENCL_DEBUG_LOG("[AddMM::eval_gpu] alpha=" << alpha << " beta=" << beta);

  // Support float32, float16, and bfloat16
  bool is_bf16 = (a.dtype() == bfloat16 && b.dtype() == bfloat16 && c.dtype() == bfloat16);
  bool is_f16 = (a.dtype() == float16 && b.dtype() == float16 && c.dtype() == float16);
  bool is_f32 = (a.dtype() == float32 && b.dtype() == float32 && c.dtype() == float32);
  if (!is_bf16 && !is_f16 && !is_f32) {
    throw std::runtime_error("[AddMM::eval_gpu] Only float32, float16, and bfloat16 supported on GPU");
  }

  // Return if output is empty
  if (out.size() == 0) {
    out.set_data(opencl::allocator().malloc(out.nbytes()));
    return;
  }

  auto& dev = opencl::device(Device::gpu);
  auto& s = stream();

  // Handle K=0 case: just return beta * C
  if (a.shape(-1) == 0) {
    OPENCL_DEBUG_LOG("[AddMM::eval_gpu] K=0 case, returning beta * C");
    out.set_data(opencl::allocator().malloc(out.nbytes()));

    // Make C contiguous if needed
    array c_contig = c;
    if (!c.flags().row_contiguous) {
      c_contig = contiguous_copy_gpu(c, s);
    }

    // Use unified scale kernel generator
    std::ostringstream kernel_source;
    kernel_source << opencl::get_kernel_preamble();

    std::string type_suffix = opencl::type_to_suffix(c.dtype());
    std::string kernel_name = "scale_" + type_suffix;
    kernel_source << gen_scale_kernel(kernel_name, c.dtype());

    std::string source_str = kernel_source.str();
    cl_kernel kernel = dev.get_kernel(kernel_name, source_str);

    auto& encoder = dev.get_command_encoder(s.index);
    encoder.set_kernel(kernel);
    encoder.set_input_array(c_contig, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(beta, 2);
    encoder.set_bytes(static_cast<int>(out.size()), 3);

    size_t local_size[3] = {256, 0, 0};
    size_t global_size[3] = {static_cast<size_t>((out.size() + 255) / 256 * 256), 0, 0};
    encoder.dispatch_threads(global_size, local_size, 1);

    dev.add_temporary(c_contig, s.index);
    dev.end_encoding(s.index);
    return;
  }

  // Handle input contiguity - support transposed B like Matmul does
  array a_contig = a;
  array b_work = b;
  array c_contig = c;
  bool b_transposed = false;
  int ldb = 0;

  if (!a.flags().row_contiguous) {
    OPENCL_DEBUG_LOG("[AddMM::eval_gpu] Matrix A not row-contiguous, making copy");
    a_contig = contiguous_copy_gpu(a, s);
  }

  // Check B's memory layout - detect column-contiguous (transposed) B
  if (b.flags().row_contiguous) {
    b_transposed = false;
    ldb = b.shape(-1);  // N
    OPENCL_DEBUG_LOG("[AddMM::eval_gpu] B is row-contiguous, ldb=" << ldb);
  } else {
    // Check if B is column-contiguous (transposed row-contiguous)
    auto& b_strides = b.strides();
    size_t stx = b_strides[b_strides.size() - 2];  // stride for K dimension
    size_t sty = b_strides[b_strides.size() - 1];  // stride for N dimension
    size_t K_dim = b.shape(-2);

    if (stx == 1 && sty == K_dim) {
      // Column-contiguous (transposed): use stride-aware kernel
      b_transposed = true;
      ldb = K_dim;
      OPENCL_DEBUG_LOG("[AddMM::eval_gpu] B is column-contiguous (transposed), ldb=" << ldb);
    } else {
      // General non-contiguous case: make a contiguous copy
      OPENCL_DEBUG_LOG("[AddMM::eval_gpu] Matrix B not contiguous (strides=" << stx << "," << sty << "), making copy");
      b_work = contiguous_copy_gpu(b, s);
      b_transposed = false;
      ldb = b_work.shape(-1);
    }
  }

  if (!c.flags().row_contiguous) {
    OPENCL_DEBUG_LOG("[AddMM::eval_gpu] Matrix C not row-contiguous, making copy");
    c_contig = contiguous_copy_gpu(c, s);
  }

  // Allocate output
  out.set_data(opencl::allocator().malloc(out.nbytes()));

  auto& encoder = dev.get_command_encoder(s.index);

  // Matrix dimensions: A is M x K, B is K x N, C is M x N, out is M x N
  int M = a_contig.shape(-2);
  int K = a_contig.shape(-1);
  int N = b_work.shape(-1);

  OPENCL_DEBUG_LOG("[AddMM::eval_gpu] M=" << M << " K=" << K << " N=" << N);

  // Handle batched addmm
  auto [batch_shape, a_batch_strides, b_batch_strides, c_batch_strides] =
      collapse_batches(a_contig, b_work, c_contig);
  size_t batch_size = std::accumulate(batch_shape.begin(), batch_shape.end(), 1, std::multiplies<int>());

  OPENCL_DEBUG_LOG("[AddMM::eval_gpu] batch_size=" << batch_size);

  // Single-dispatch batched AddMM: when batch_size > 1, use 3D dispatch instead of loop
  // Only use for simple batch shapes (single collapsed dimension) to keep kernel simple
  bool use_batched_dispatch = (batch_size > 1) && (batch_shape.size() == 1);

  // Build fused addmm kernel
  // out[i,j] = alpha * sum_k(A[i,k] * B[k,j]) + beta * C[i,j]
  // Choose kernel based on dimensions
  // GEMV path only supports non-transposed B for now
  // Disable special kernels for batched dispatch - use unified batched kernel
  bool use_gemv = (M <= 4 && N >= 64 && K >= 64 && !b_transposed) && !use_batched_dispatch;

  // Use tiled kernel for large matrices (non-batched, non-transposed B only)
  // The tiled kernel uses local memory and doesn't support transposed B
  bool large_matrix = (M >= 32 && N >= 256 && K >= 256);
  bool use_tiled = large_matrix && !b_transposed && !use_batched_dispatch;

  // Tile parameters for tiled AddMM kernel
  const int BM_TILE = 64, BN_TILE = 64, TM_TILE = 4, TN_TILE = 4;
  const int WG_M = BM_TILE / TM_TILE, WG_N = BN_TILE / TN_TILE;

  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();

  std::string kernel_name;

  if (use_gemv) {
    // Vector-matrix multiply kernel for AddMM - unified generator
    std::string suffix = opencl::type_to_suffix(a.dtype());
    kernel_name = "addmm_gemv_" + suffix;
    kernel_source << gen_addmm_gemv_kernel(kernel_name, a.dtype());
  } else if (use_tiled) {
    // High-performance tiled AddMM kernel for large matrices
    std::string suffix = opencl::type_to_suffix(a.dtype());
    kernel_name = "addmm_tiled_" + suffix;
    kernel_source << gen_tiled_addmm_kernel(kernel_name, a.dtype());
    OPENCL_DEBUG_LOG("[AddMM::eval_gpu] Using tiled kernel: " << kernel_name);
  } else if (use_batched_dispatch) {
    // Single-dispatch batched AddMM using gen_matmul_kernel with batched=true
    std::string suffix = opencl::type_to_suffix(a.dtype());
    std::string bt_suffix = b_transposed ? "_bt" : "";
    kernel_name = "addmm_batched_" + suffix + bt_suffix;
    kernel_source << gen_matmul_kernel(kernel_name, a.dtype(), b_transposed, true, true);
    OPENCL_DEBUG_LOG("[AddMM::eval_gpu] Using batched dispatch kernel: " << kernel_name);
  } else {
    // Use unified kernel generator with epilogue support
    // Handles both normal and transposed B, with fused alpha/beta epilogue
    std::string suffix = opencl::type_to_suffix(a.dtype());
    std::string bt_suffix = b_transposed ? "_bt" : "";
    kernel_name = "addmm_" + suffix + bt_suffix;
    kernel_source << gen_matmul_kernel(kernel_name, a.dtype(), b_transposed, true);
  }

  std::string source_str = kernel_source.str();

  OPENCL_DEBUG_LOG("[AddMM::eval_gpu] Compiling kernel: " << kernel_name << " use_gemv=" << use_gemv << " use_tiled=" << use_tiled << " b_transposed=" << b_transposed << " ldb=" << ldb);

  // Get or compile kernel
  cl_kernel kernel = dev.get_kernel(kernel_name, source_str);

  // Calculate output batch stride
  size_t out_batch_stride = M * N;

  // Dispatch kernel
  size_t local_size[3];
  size_t global_size[3];

  if (use_batched_dispatch) {
    // Batched dispatch: 3D dispatch with batch_size in Z dimension
    local_size[0] = 16;
    local_size[1] = 16;
    local_size[2] = 1;
    global_size[0] = static_cast<size_t>((M + 15) / 16 * 16);
    global_size[1] = static_cast<size_t>((N + 15) / 16 * 16);
    global_size[2] = batch_size;
  } else if (use_gemv) {
    // GEMV: one thread per output element
    local_size[0] = 256;
    local_size[1] = 1;
    local_size[2] = 0;
    global_size[0] = static_cast<size_t>((N + 255) / 256 * 256);
    global_size[1] = static_cast<size_t>(M);
    global_size[2] = 0;
  } else if (use_tiled) {
    // Tiled AddMM: 2D dispatch with 16x16 = 256 threads per workgroup
    // Each thread computes TM=4 x TN=4 outputs
    local_size[0] = WG_M;
    local_size[1] = WG_N;
    local_size[2] = 0;
    int numTilesM = (M + BM_TILE - 1) / BM_TILE;
    int numTilesN = (N + BN_TILE - 1) / BN_TILE;
    global_size[0] = static_cast<size_t>(numTilesM * WG_M);
    global_size[1] = static_cast<size_t>(numTilesN * WG_N);
    global_size[2] = 0;
  } else {
    local_size[0] = 16;
    local_size[1] = 16;
    local_size[2] = 0;
    global_size[0] = static_cast<size_t>((M + 15) / 16 * 16);
    global_size[1] = static_cast<size_t>((N + 15) / 16 * 16);
    global_size[2] = 0;
  }

  if (use_batched_dispatch) {
    // Single-dispatch batched AddMM: dispatch all batches in one kernel call
    encoder.set_kernel(kernel);
    encoder.set_input_array(a_contig, 0);
    encoder.set_input_array(b_work, 1);
    encoder.set_input_array(c_contig, 2);
    encoder.set_output_array(out, 3);
    encoder.set_bytes(static_cast<int>(M), 4);
    encoder.set_bytes(static_cast<int>(N), 5);
    encoder.set_bytes(static_cast<int>(K), 6);
    encoder.set_bytes(static_cast<int>(ldb), 7);
    encoder.set_bytes(alpha, 8);
    encoder.set_bytes(beta, 9);

    // Batch strides for single-dimension batch
    int a_batch_stride_flat = a_batch_strides.empty() ? 0 : static_cast<int>(a_batch_strides[0]);
    int b_batch_stride_flat = b_batch_strides.empty() ? 0 : static_cast<int>(b_batch_strides[0]);
    int c_batch_stride_flat = c_batch_strides.empty() ? 0 : static_cast<int>(c_batch_strides[0]);
    int out_batch_stride_int = static_cast<int>(out_batch_stride);

    encoder.set_bytes(a_batch_stride_flat, 10);
    encoder.set_bytes(b_batch_stride_flat, 11);
    encoder.set_bytes(c_batch_stride_flat, 12);
    encoder.set_bytes(out_batch_stride_int, 13);
    // Base offsets for sliced arrays
    int a_base_offset = static_cast<int>(a_contig.offset() / a_contig.itemsize());
    int b_base_offset = static_cast<int>(b_work.offset() / b_work.itemsize());
    encoder.set_bytes(a_base_offset, 14);
    encoder.set_bytes(b_base_offset, 15);

    OPENCL_DEBUG_LOG("[AddMM::eval_gpu] Batched dispatch: A_stride=" << a_batch_stride_flat
              << " B_stride=" << b_batch_stride_flat << " C_stride=" << c_batch_stride_flat
              << " D_stride=" << out_batch_stride_int << " A_base=" << a_base_offset << " B_base=" << b_base_offset);

    encoder.dispatch_threads(global_size, local_size, 3);
  } else {
    // Non-batched dispatch: loop through batches
    for (size_t batch = 0; batch < batch_size; batch++) {
      // Calculate batch offsets using strides, starting from array base offsets
      size_t a_offset = a_contig.offset() / a_contig.itemsize();
      size_t b_offset = b_work.offset() / b_work.itemsize();
      size_t c_offset = c_contig.offset() / c_contig.itemsize();
      size_t temp_batch = batch;

      for (int i = batch_shape.size() - 1; i >= 0; i--) {
        size_t batch_idx = temp_batch % batch_shape[i];
        temp_batch /= batch_shape[i];
        a_offset += batch_idx * a_batch_strides[i];
        b_offset += batch_idx * b_batch_strides[i];
        c_offset += batch_idx * c_batch_strides[i];
      }

      size_t out_offset = batch * out_batch_stride;

      OPENCL_DEBUG_LOG("[AddMM::eval_gpu] Batch " << batch << "/" << batch_size
                << ": a_offset=" << a_offset << " b_offset=" << b_offset
                << " c_offset=" << c_offset << " out_offset=" << out_offset);

      encoder.set_kernel(kernel);
      encoder.set_input_array(a_contig, 0);
      encoder.set_input_array(b_work, 1);
      encoder.set_input_array(c_contig, 2);
      encoder.set_output_array(out, 3);
      encoder.set_bytes(static_cast<int>(M), 4);
      encoder.set_bytes(static_cast<int>(N), 5);
      encoder.set_bytes(static_cast<int>(K), 6);

      // The unified kernel uses ldb, while GEMV and tiled use their own signatures
      if (use_gemv || use_tiled) {
        // GEMV/tiled kernel signature: M, N, K, alpha, beta, offsets (no ldb)
        encoder.set_bytes(alpha, 7);
        encoder.set_bytes(beta, 8);
        encoder.set_bytes(static_cast<int>(a_offset), 9);
        encoder.set_bytes(static_cast<int>(b_offset), 10);
        encoder.set_bytes(static_cast<int>(c_offset), 11);
        encoder.set_bytes(static_cast<int>(out_offset), 12);
      } else {
        // Unified kernel signature: M, N, K, ldb, alpha, beta, offsets
        encoder.set_bytes(static_cast<int>(ldb), 7);
        encoder.set_bytes(alpha, 8);
        encoder.set_bytes(beta, 9);
        encoder.set_bytes(static_cast<int>(a_offset), 10);
        encoder.set_bytes(static_cast<int>(b_offset), 11);
        encoder.set_bytes(static_cast<int>(c_offset), 12);
        encoder.set_bytes(static_cast<int>(out_offset), 13);
      }

      encoder.dispatch_threads(global_size, local_size, 2);
    }
  }

  // Add arrays as temporaries so they aren't freed until kernel completes
  dev.add_temporary(a_contig, s.index);
  dev.add_temporary(b_work, s.index);
  dev.add_temporary(c_contig, s.index);
  dev.add_temporary(out, s.index);

  // Ensure kernel completion for consistent results
  dev.end_encoding(s.index);

  OPENCL_DEBUG_LOG("[AddMM::eval_gpu] Dispatched " << batch_size << " batch(es): global=("
            << global_size[0] << "," << global_size[1] << ") local=("
            << local_size[0] << "," << local_size[1] << ")");
}

} // namespace mlx::core
