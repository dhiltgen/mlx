// Copyright © 2025 MLX Contributors
// OpenCL Quantization operations
//
// IMPLEMENTATION STATUS:
// Implemented:
//   - Dequantize (affine mode, 2/4/8-bit)
//   - Quantize (affine mode, 2/4/8-bit)
//   - QuantizedMatmul (affine mode)
//   - Dequantize (mxfp4 mode)
//   - QuantizedMatmul (mxfp4 mode)
//
// NOT Implemented (will throw exception):
//   - Mxfp8, Nvfp4 modes
//   - QQMatmul (both inputs quantized)
//   - GatherQMM (gather + quantized matmul)
//   - 3, 5, 6-bit quantization
//
// QUANTIZATION FORMAT (Affine):
//   packed_value: uint32 containing multiple quantized values
//   - 4-bit: 8 values per uint32
//   - 8-bit: 4 values per uint32
//   dequantized = scale * quantized_value + bias
//   Where scale and bias are per-group (group_size values share same scale/bias)
//
// QUANTIZATION FORMAT (Mxfp4):
//   packed_value: uint8 containing 2 fp4_e2m1 values (4 bits each)
//   scales: fp8_e8m0 (power of 2), one per 32-element group
//   dequantized = scale * fp4_to_float(quantized_value)
//   No biases used

#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/fast_primitives.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include "mlx/utils.h"

#include <sstream>

namespace mlx::core {

namespace {

std::string generate_dequantize_kernel_f32(int bits, int group_size) {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  int pack_factor = 8 / bits;  // Values per byte (4-bit: 2, 8-bit: 1)
  int bitmask = (1 << bits) - 1;

  oss << R"(
__kernel void affine_dequantize_f32(
    __global const uint* packed,      // Packed quantized values
    __global const float* scales,     // Scale per group
    __global const float* biases,     // Bias per group
    __global float* output,           // Dequantized output
    int N,                            // Output size (number of float values)
    int group_size                    // Values per group sharing same scale/bias
) {
    int gid = get_global_id(0);
    if (gid >= N) return;

    // Which group does this element belong to?
    int group_idx = gid / group_size;
    float scale = scales[group_idx];
    float bias = biases[group_idx];

    // Which packed uint32 and which position within it?
)";

  if (bits == 2) {
    oss << R"(
    int pack_idx = gid / 16;      // 16 values per uint32 for 2-bit
    int pos_in_pack = gid % 16;   // Position within the packed uint32
    uint packed_val = packed[pack_idx];

    // Extract 2-bit value: shift right by (pos * 2), mask with 0x3
    uint quantized = (packed_val >> (pos_in_pack * 2)) & 0x3;
)";
  } else if (bits == 4) {
    oss << R"(
    int pack_idx = gid / 8;       // 8 values per uint32 for 4-bit
    int pos_in_pack = gid % 8;    // Position within the packed uint32
    uint packed_val = packed[pack_idx];

    // Extract 4-bit value: shift right by (pos * 4), mask with 0xF
    uint quantized = (packed_val >> (pos_in_pack * 4)) & 0xF;
)";
  } else if (bits == 8) {
    oss << R"(
    int pack_idx = gid / 4;       // 4 values per uint32 for 8-bit
    int pos_in_pack = gid % 4;    // Position within the packed uint32
    uint packed_val = packed[pack_idx];

    // Extract 8-bit value: shift right by (pos * 8), mask with 0xFF
    uint quantized = (packed_val >> (pos_in_pack * 8)) & 0xFF;
)";
  } else {
    oss << "    uint quantized = 0; // Unsupported bit width\n";
  }

  oss << R"(
    // Dequantize: output = scale * quantized + bias
    output[gid] = scale * (float)quantized + bias;
}
)";

  return oss.str();
}

std::string generate_dequantize_kernel_f16(int bits, int group_size) {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  int pack_factor = 8 / bits;
  int bitmask = (1 << bits) - 1;

  oss << R"(
__kernel void affine_dequantize_f16(
    __global const uint* packed,      // Packed quantized values
    __global const half* scales,      // Scale per group
    __global const half* biases,      // Bias per group
    __global half* output,            // Dequantized output
    int N,                            // Output size (number of half values)
    int group_size                    // Values per group sharing same scale/bias
) {
    int gid = get_global_id(0);
    if (gid >= N) return;

    // Which group does this element belong to?
    int group_idx = gid / group_size;
    float scale = convert_float(scales[group_idx]);
    float bias = convert_float(biases[group_idx]);

    // Which packed uint32 and which position within it?
)";

  if (bits == 2) {
    oss << R"(
    int pack_idx = gid / 16;      // 16 values per uint32 for 2-bit
    int pos_in_pack = gid % 16;
    uint packed_val = packed[pack_idx];
    uint quantized = (packed_val >> (pos_in_pack * 2)) & 0x3;
)";
  } else if (bits == 4) {
    oss << R"(
    int pack_idx = gid / 8;       // 8 values per uint32 for 4-bit
    int pos_in_pack = gid % 8;
    uint packed_val = packed[pack_idx];
    uint quantized = (packed_val >> (pos_in_pack * 4)) & 0xF;
)";
  } else if (bits == 8) {
    oss << R"(
    int pack_idx = gid / 4;       // 4 values per uint32 for 8-bit
    int pos_in_pack = gid % 4;
    uint packed_val = packed[pack_idx];
    uint quantized = (packed_val >> (pos_in_pack * 8)) & 0xFF;
)";
  } else {
    oss << "    uint quantized = 0; // Unsupported bit width\n";
  }

  oss << R"(
    // Dequantize: output = scale * quantized + bias
    output[gid] = convert_half(scale * (float)quantized + bias);
}
)";

  return oss.str();
}

std::string generate_quantize_kernel(int bits, int group_size, const std::string& input_type = "float", const std::string& scales_type = "float") {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  int pack_factor = 8 / bits;
  int bitmask = (1 << bits) - 1;
  int max_val = (1 << bits) - 1;

  // Determine conversion functions based on input type
  std::string to_float;
  if (input_type == "bfloat16_t") {
    to_float = "bfloat16_to_float";
  } else if (input_type == "half") {
    to_float = "convert_float";
  } else {
    to_float = "";  // No conversion for float
  }

  std::string from_float;
  if (scales_type == "bfloat16_t") {
    from_float = "float_to_bfloat16";
  } else if (scales_type == "half") {
    from_float = "convert_half";
  } else {
    from_float = "";  // No conversion for float
  }

  // Generate conversion code based on type
  std::string input_to_float = to_float.empty() ? "" : (to_float + "(");
  std::string input_close = to_float.empty() ? "" : ")";
  std::string scales_from_float = from_float.empty() ? "" : (from_float + "(");
  std::string scales_close = from_float.empty() ? "" : ")";

  // Each thread handles one group, computes scale/bias and packs values
  oss << "// First pass: compute min/max per group\n"
      << "__kernel void affine_quantize_minmax(\n"
      << "    __global const " << input_type << "* input,\n"
      << "    __global float* group_mins,\n"
      << "    __global float* group_maxs,\n"
      << "    int N,\n"
      << "    int group_size\n"
      << ") {\n"
      << "    int group_idx = get_global_id(0);\n"
      << "    int start = group_idx * group_size;\n"
      << "    if (start >= N) return;\n"
      << "\n"
      << "    int end = min(start + group_size, N);\n"
      << "\n"
      << "    float min_val = " << input_to_float << "input[start]" << input_close << ";\n"
      << "    float max_val = " << input_to_float << "input[start]" << input_close << ";\n"
      << "\n"
      << "    for (int i = start + 1; i < end; i++) {\n"
      << "        float val = " << input_to_float << "input[i]" << input_close << ";\n"
      << "        min_val = fmin(min_val, val);\n"
      << "        max_val = fmax(max_val, val);\n"
      << "    }\n"
      << "\n"
      << "    group_mins[group_idx] = min_val;\n"
      << "    group_maxs[group_idx] = max_val;\n"
      << "}\n"
      << "\n";

  oss << "// Second pass: compute scales and biases from min/max\n"
      << "__kernel void affine_quantize_scales(\n"
      << "    __global const float* group_mins,\n"
      << "    __global const float* group_maxs,\n"
      << "    __global " << scales_type << "* scales,\n"
      << "    __global " << scales_type << "* biases,\n"
      << "    int num_groups,\n"
      << "    int max_quant_val\n"
      << ") {\n"
      << "    int group_idx = get_global_id(0);\n"
      << "    if (group_idx >= num_groups) return;\n"
      << "\n"
      << "    float min_val = group_mins[group_idx];\n"
      << "    float max_val = group_maxs[group_idx];\n"
      << "\n"
      << "    float range = max_val - min_val;\n"
      << "    float scale = range / (float)max_quant_val;\n"
      << "\n"
      << "    if (scale < 1e-10f) scale = 1e-10f;\n"
      << "\n"
      << "    scales[group_idx] = " << scales_from_float << "scale" << scales_close << ";\n"
      << "    biases[group_idx] = " << scales_from_float << "min_val" << scales_close << ";\n"
      << "}\n"
      << "\n";

  oss << "// Third pass: quantize and pack values\n"
      << "__kernel void affine_quantize_pack(\n"
      << "    __global const " << input_type << "* input,\n"
      << "    __global const " << scales_type << "* scales,\n"
      << "    __global const " << scales_type << "* biases,\n"
      << "    __global uint* output,\n"
      << "    int N,\n"
      << "    int group_size,\n"
      << "    int bits_per_val,\n"
      << "    int vals_per_uint\n"
      << ") {\n"
      << "    int pack_idx = get_global_id(0);\n"
      << "    int start = pack_idx * vals_per_uint;\n"
      << "    if (start >= N) return;\n"
      << "\n"
      << "    uint packed = 0;\n"
      << "\n"
      << "    for (int i = 0; i < vals_per_uint && (start + i) < N; i++) {\n"
      << "        int idx = start + i;\n"
      << "        int group_idx = idx / group_size;\n";

  // Handle scale/bias conversion
  if (scales_type == "bfloat16_t") {
    oss << "        float scale = bfloat16_to_float(scales[group_idx]);\n"
        << "        float bias = bfloat16_to_float(biases[group_idx]);\n";
  } else if (scales_type == "half") {
    oss << "        float scale = convert_float(scales[group_idx]);\n"
        << "        float bias = convert_float(biases[group_idx]);\n";
  } else {
    oss << "        float scale = scales[group_idx];\n"
        << "        float bias = biases[group_idx];\n";
  }

  oss << "\n"
      << "        float val = " << input_to_float << "input[idx]" << input_close << ";\n"
      << "\n"
      << "        float q_float = (val - bias) / scale;\n"
      << "        int q = (int)round(q_float);\n"
      << "        q = max(0, min(q, (1 << bits_per_val) - 1));\n"
      << "\n"
      << "        packed |= ((uint)q << (i * bits_per_val));\n"
      << "    }\n"
      << "\n"
      << "    output[pack_idx] = packed;\n"
      << "}\n";

  return oss.str();
}

std::string generate_quantized_matmul_kernel_f32(int bits, int group_size) {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  // Basic quantized matmul: x (float) @ w_q (quantized) = out (float)
  // w_q is stored transposed, so w_q[n, k/pack_factor] contains packed values
  oss << R"(
// Naive quantized matmul: out[m,n] = sum_k(x[m,k] * dequant(w_q[n,k]))
// w_q is stored as [N, K/pack_factor] where pack_factor = 8/bits
// scales and biases are [N, K/group_size]
__kernel void affine_qmatmul_f32(
    __global const float* x,          // Input [M, K]
    __global const uint* w_q,         // Quantized weights [N, K/pack_factor]
    __global const float* scales,     // Scales [N, K/group_size]
    __global const float* biases,     // Biases [N, K/group_size]
    __global float* out,              // Output [M, N]
    int M, int N, int K,
    int group_size,
    int bits_per_val,
    int pack_factor                   // Values per uint32
) {
    int m = get_global_id(0);
    int n = get_global_id(1);

    if (m >= M || n >= N) return;

    float sum = 0.0f;
    int bitmask = (1 << bits_per_val) - 1;
    int num_groups_per_row = K / group_size;

    for (int k = 0; k < K; k++) {
        // Get x value
        float x_val = x[m * K + k];

        // Get quantized w value and dequantize
        int pack_idx = k / pack_factor;
        int pos_in_pack = k % pack_factor;
        uint packed = w_q[n * (K / pack_factor) + pack_idx];
        uint q = (packed >> (pos_in_pack * bits_per_val)) & bitmask;

        // Get scale and bias for this group
        int group_idx = k / group_size;
        float scale = scales[n * num_groups_per_row + group_idx];
        float bias = biases[n * num_groups_per_row + group_idx];

        // Dequantize and accumulate
        float w_val = scale * (float)q + bias;
        sum += x_val * w_val;
    }

    out[m * N + n] = sum;
}
)";

  return oss.str();
}

std::string generate_quantized_matmul_kernel_f16(int bits, int group_size) {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  // Float16 quantized matmul: x (half) @ w_q (quantized) = out (half)
  // Compute in float32 for accuracy, load/store in float16
  oss << R"(
// Float16 quantized matmul: out[m,n] = sum_k(x[m,k] * dequant(w_q[n,k]))
// w_q is stored as [N, K/pack_factor] where pack_factor = 32/bits
// scales and biases are [N, K/group_size]
__kernel void affine_qmatmul_f16(
    __global const half* x,           // Input [M, K]
    __global const uint* w_q,         // Quantized weights [N, K/pack_factor]
    __global const half* scales,      // Scales [N, K/group_size]
    __global const half* biases,      // Biases [N, K/group_size]
    __global half* out,               // Output [M, N]
    int M, int N, int K,
    int group_size,
    int bits_per_val,
    int pack_factor                   // Values per uint32
) {
    int m = get_global_id(0);
    int n = get_global_id(1);

    if (m >= M || n >= N) return;

    float sum = 0.0f;
    int bitmask = (1 << bits_per_val) - 1;
    int num_groups_per_row = K / group_size;

    for (int k = 0; k < K; k++) {
        // Get x value (convert from half to float)
        float x_val = convert_float(x[m * K + k]);

        // Get quantized w value and dequantize
        int pack_idx = k / pack_factor;
        int pos_in_pack = k % pack_factor;
        uint packed = w_q[n * (K / pack_factor) + pack_idx];
        uint q = (packed >> (pos_in_pack * bits_per_val)) & bitmask;

        // Get scale and bias for this group (convert from half to float)
        int group_idx = k / group_size;
        float scale = convert_float(scales[n * num_groups_per_row + group_idx]);
        float bias = convert_float(biases[n * num_groups_per_row + group_idx]);

        // Dequantize and accumulate
        float w_val = scale * (float)q + bias;
        sum += x_val * w_val;
    }

    out[m * N + n] = convert_half(sum);
}
)";

  return oss.str();
}

std::string generate_quantized_matmul_kernel_bf16(int bits, int group_size) {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  // Bfloat16 quantized matmul: x (bfloat16) @ w_q (quantized) = out (bfloat16)
  // Compute in float32 for accuracy, load/store in bfloat16
  oss << R"(
// Bfloat16 quantized matmul: out[m,n] = sum_k(x[m,k] * dequant(w_q[n,k]))
// w_q is stored as [N, K/pack_factor] where pack_factor = 32/bits
// scales and biases are [N, K/group_size]
__kernel void affine_qmatmul_bf16(
    __global const bfloat16_t* x,      // Input [M, K]
    __global const uint* w_q,          // Quantized weights [N, K/pack_factor]
    __global const bfloat16_t* scales, // Scales [N, K/group_size]
    __global const bfloat16_t* biases, // Biases [N, K/group_size]
    __global bfloat16_t* out,          // Output [M, N]
    int M, int N, int K,
    int group_size,
    int bits_per_val,
    int pack_factor                    // Values per uint32
) {
    int m = get_global_id(0);
    int n = get_global_id(1);

    if (m >= M || n >= N) return;

    float sum = 0.0f;
    int bitmask = (1 << bits_per_val) - 1;
    int num_groups_per_row = K / group_size;

    for (int k = 0; k < K; k++) {
        // Get x value (convert from bfloat16 to float)
        float x_val = bfloat16_to_float(x[m * K + k]);

        // Get quantized w value and dequantize
        int pack_idx = k / pack_factor;
        int pos_in_pack = k % pack_factor;
        uint packed = w_q[n * (K / pack_factor) + pack_idx];
        uint q = (packed >> (pos_in_pack * bits_per_val)) & bitmask;

        // Get scale and bias for this group (convert from bfloat16 to float)
        int group_idx = k / group_size;
        float scale = bfloat16_to_float(scales[n * num_groups_per_row + group_idx]);
        float bias = bfloat16_to_float(biases[n * num_groups_per_row + group_idx]);

        // Dequantize and accumulate
        float w_val = scale * (float)q + bias;
        sum += x_val * w_val;
    }

    out[m * N + n] = float_to_bfloat16(sum);
}
)";

  return oss.str();
}

// Optimized quantized matrix-vector kernel for M=1 (token generation)
// Each workgroup computes one output element using subgroup (wave) reduction
// Uses vectorized x loading for better memory bandwidth
std::string generate_qmv_kernel_f16(int bits, int group_size) {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  // For 4-bit: pack_factor = 8, so each uint32 has 8 values
  int pack_factor = 32 / bits;
  int bitmask = (1 << bits) - 1;

  // Generate unique kernel name based on parameters
  std::string kernel_name = "qmv_f16_b" + std::to_string(bits) + "_g" + std::to_string(group_size);

  oss << R"(
// Enable subgroup extension for fast wave-level reductions
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

// Optimized quantized matrix-vector multiply for token generation (M=1)
// Each workgroup computes one output element
// Uses subgroup reduction for fast parallel sum (wave size = 64 on Adreno)
// With 256 work items = 4 waves, we do:
//   1. Each thread computes partial sum
//   2. sub_group_reduce_add() reduces within each wave (instant, no barrier)
//   3. 4 wave sums stored to local memory
//   4. Single barrier, then thread 0 sums 4 values
#define WORKGROUP_SIZE 256
#define VALS_PER_UINT )" << pack_factor << R"(
#define BITMASK )" << bitmask << R"(
#define BITS )" << bits << R"(

__kernel void )" << kernel_name << R"((
    __global const half* x,           // Input [1, K]
    __global const uint* w_q,         // Quantized weights [N, K/pack_factor]
    __global const half* scales,      // Scales [N, K/group_size]
    __global const half* biases,      // Biases [N, K/group_size]
    __global half* out,               // Output [1, N]
    int N, int K,
    int group_size
) {
    // Space for wave sums - use larger buffer for safety in case of different subgroup sizes
    __local float wave_sums[8];

    int n = get_group_id(0);  // Output column this workgroup computes
    int lid = get_local_id(0);

    // Initialize wave_sums to avoid uninitialized memory issues
    if (lid < 8) {
        wave_sums[lid] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Early exit check - but we only dispatch exactly N workgroups, so this should never trigger
    // Keeping for safety but all threads must participate in barriers below
    bool valid = (n < N);

    int K_packed = K / VALS_PER_UINT;
    int num_groups = K / group_size;

    // Each work item processes multiple packed values
    float sum = 0.0f;

    // Stride through K dimension
    for (int pack_idx = lid; pack_idx < K_packed; pack_idx += WORKGROUP_SIZE) {
        // Load packed quantized weights for this output row
        uint packed = w_q[n * K_packed + pack_idx];

        // Determine which scale/bias group this pack belongs to
        int k_start = pack_idx * VALS_PER_UINT;
        int group_idx = k_start / group_size;
        float scale = convert_float(scales[n * num_groups + group_idx]);
        float bias = convert_float(biases[n * num_groups + group_idx]);

        // Load x values using vectorized load (8 halves = 16 bytes)
        // vload8 loads 8 half values at once for better memory bandwidth
        half8 x_vec = vload8(0, x + k_start);

        // Process all values in this packed uint32
)";

  // Unroll the inner loop for extracting bits - use vector components
  for (int i = 0; i < pack_factor; i++) {
    std::string vec_component;
    if (i == 0) vec_component = "x_vec.s0";
    else if (i == 1) vec_component = "x_vec.s1";
    else if (i == 2) vec_component = "x_vec.s2";
    else if (i == 3) vec_component = "x_vec.s3";
    else if (i == 4) vec_component = "x_vec.s4";
    else if (i == 5) vec_component = "x_vec.s5";
    else if (i == 6) vec_component = "x_vec.s6";
    else if (i == 7) vec_component = "x_vec.s7";

    oss << "        {\n";
    oss << "            float x_val = convert_float(" << vec_component << ");\n";
    oss << "            uint q = (packed >> " << (i * bits) << ") & BITMASK;\n";
    oss << "            sum += x_val * (scale * (float)q + bias);\n";
    oss << "        }\n";
  }

  oss << R"(
    }

    // Subgroup (wave) reduction - instant within each wave
    float wave_sum = sub_group_reduce_add(sum);

    // First thread of each wave stores to local memory
    uint wave_id = get_sub_group_id();
    uint num_waves = get_num_sub_groups();
    if (get_sub_group_local_id() == 0 && wave_id < 8) {
        wave_sums[wave_id] = wave_sum;
    }

    // Single barrier to sync all wave sums
    barrier(CLK_LOCAL_MEM_FENCE);

    // Thread 0 does final reduction of all wave sums
    if (lid == 0 && valid) {
        float total = 0.0f;
        for (uint i = 0; i < num_waves && i < 8; i++) {
            total += wave_sums[i];
        }
        out[n] = convert_half(total);
    }
}
)";

  return oss.str();
}

// Multi-output kernel: 8 outputs per workgroup with subgroup reduction
// This is the optimized baseline - broadcast and multi-block K-loop tested but slower
std::string generate_qmv_multi_kernel_f16(int bits, int group_size) {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  int pack_factor = 32 / bits;
  int bitmask = (1 << bits) - 1;

  std::string kernel_name = "qmv_multi_f16_b" + std::to_string(bits) + "_g" + std::to_string(group_size);

  oss << R"(
// Multi-output qmv: each workgroup computes 8 output elements
// Uses parallel reduction in local memory for portability
// Note: Broadcast and multi-block K-loop optimizations were tested but found slower
//       due to non-coalesced memory access patterns on Adreno
#define WORKGROUP_SIZE 64
#define RESULTS_PER_WG 8
#define VALS_PER_UINT )" << pack_factor << R"(
#define BITMASK )" << bitmask << R"(
#define BITS )" << bits << R"(

__kernel void )" << kernel_name << R"((
    __global const half* x,           // Input [1, K]
    __global const uint* w_q,         // Quantized weights [N, K/pack_factor]
    __global const half* scales,      // Scales [N, K/group_size]
    __global const half* biases,      // Biases [N, K/group_size]
    __global half* out,               // Output [1, N]
    int N, int K,
    int group_size
) {
    // Local memory for parallel reduction
    __local float local_sums[RESULTS_PER_WG * WORKGROUP_SIZE];

    int wg_id = get_group_id(0);
    int lid = get_local_id(0);
    int n_base = wg_id * RESULTS_PER_WG;

    int K_packed = K / VALS_PER_UINT;
    int num_groups = K / group_size;

    float sums[RESULTS_PER_WG];
    #pragma unroll
    for (int r = 0; r < RESULTS_PER_WG; r++) {
        sums[r] = 0.0f;
    }

    // Stride through K dimension - adjacent threads access adjacent pack_idx for coalescing
    for (int pack_idx = lid; pack_idx < K_packed; pack_idx += WORKGROUP_SIZE) {
        int k_start = pack_idx * VALS_PER_UINT;
        int group_idx = k_start / group_size;

        half8 x_vec = vload8(0, x + k_start);

        #pragma unroll
        for (int r = 0; r < RESULTS_PER_WG; r++) {
            int n = n_base + r;
            if (n >= N) continue;

            uint packed = w_q[n * K_packed + pack_idx];
            float scale = convert_float(scales[n * num_groups + group_idx]);
            float bias = convert_float(biases[n * num_groups + group_idx]);

)";

  // Unroll inner loop
  for (int i = 0; i < pack_factor; i++) {
    std::string vec_component = "x_vec.s" + std::to_string(i);
    oss << "            sums[r] += convert_float(" << vec_component << ") * (scale * (float)((packed >> " << (i * bits) << ") & BITMASK) + bias);\n";
  }

  oss << R"(
        }
    }

    // Write partial sums to local memory
    #pragma unroll
    for (int r = 0; r < RESULTS_PER_WG; r++) {
        local_sums[r * WORKGROUP_SIZE + lid] = sums[r];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction in local memory
    for (int stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            #pragma unroll
            for (int r = 0; r < RESULTS_PER_WG; r++) {
                local_sums[r * WORKGROUP_SIZE + lid] += local_sums[r * WORKGROUP_SIZE + lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Thread 0 writes final results
    if (lid == 0) {
        #pragma unroll
        for (int r = 0; r < RESULTS_PER_WG; r++) {
            int n = n_base + r;
            if (n < N) {
                out[n] = convert_half(local_sums[r * WORKGROUP_SIZE]);
            }
        }
    }
}
)";

  return oss.str();
}

// ============================================================================
// MXFP4 (Microscaling FP4) Quantization Kernels
// ============================================================================
// MXFP4 uses:
//   - fp4_e2m1: 4-bit minifloat with 1 sign bit, 2 exponent bits, 1 mantissa bit
//   - fp8_e8m0: 8-bit scale with 8 exponent bits, 0 mantissa bits (power of 2)
//   - group_size = 32 (fixed)
//   - No biases (unlike affine)
//
// Dequantization formula: output = scale * fp4_to_float(quantized_value)

std::string generate_mxfp4_dequantize_kernel_f16() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  oss << R"(
// Convert fp4_e2m1 to float
// Format: 1 sign bit, 2 exponent bits, 1 mantissa bit
// Values map to: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (with sign for negative)
float fp4_to_float(uchar bits) {
    // Extract lower 3 bits (exp + mantissa), shift to half format position
    ushort h = (ushort)(bits & 7) << 9;
    // Reinterpret as half and scale
    half converted = as_half(h);
    converted *= (half)16384.0f;  // 2^14 scale factor
    // Apply sign (bit 3)
    return (bits & 8) ? -convert_float(converted) : convert_float(converted);
}

// Convert fp8_e8m0 scale to float
// Format: 8 exponent bits, 0 mantissa bits
// Represents power of 2: 2^(bits - 127)
float fp8_e8m0_to_float(uchar bits) {
    // fp8_e8m0 is pure exponent: value = 2^(bits - 127)
    // In IEEE float32: exponent field is bits 23-30
    // Special case: bits=0 represents 2^-127 (denormalized float 0x00400000)
    uint f_bits = (bits == 0) ? 0x00400000 : ((uint)bits << 23);
    return as_float(f_bits);
}

// MXFP4 dequantize kernel
// Weights are packed as uint8 (2 fp4 values per byte)
// Scales are fp8_e8m0 (one per 32-element group)
__kernel void mxfp4_dequantize_f16(
    __global const uchar* packed,      // Packed fp4 values [N, K/2]
    __global const uchar* scales,      // fp8_e8m0 scales [N, K/32]
    __global half* output,             // Dequantized output [N, K]
    int N,                             // Number of rows
    int K                              // Number of columns (unpacked)
) {
    int gid = get_global_id(0);
    int total_elements = N * K;
    if (gid >= total_elements) return;

    int n = gid / K;          // Row index
    int k = gid % K;          // Column index

    // Which byte and position within byte?
    int byte_idx = gid / 2;
    int pos_in_byte = gid % 2;  // 0 = low 4 bits, 1 = high 4 bits

    uchar packed_val = packed[byte_idx];
    uchar q = (pos_in_byte == 0) ? (packed_val & 0xF) : (packed_val >> 4);

    // Get scale for this group (group_size = 32)
    int group_idx = k / 32;
    int num_groups = K / 32;
    uchar scale_bits = scales[n * num_groups + group_idx];
    float scale = fp8_e8m0_to_float(scale_bits);

    // Dequantize: output = scale * fp4_to_float(q)
    float val = scale * fp4_to_float(q);
    output[gid] = convert_half(val);
}
)";

  return oss.str();
}

std::string generate_mxfp4_dequantize_kernel_f32() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  oss << R"(
// Convert fp4_e2m1 to float
// Lookup table for fp4_e2m1 to float conversion
// Avoids denormalized half-precision which some GPUs flush to zero
__constant float fp4_lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float fp4_to_float(uchar bits) {
    float val = fp4_lut_vals[bits & 7];
    return (bits & 8) ? -val : val;
}

// Convert fp8_e8m0 scale to float
float fp8_e8m0_to_float(uchar bits) {
    // Special case: bits=0 represents 2^-127 (denormalized float 0x00400000)
    uint f_bits = (bits == 0) ? 0x00400000 : ((uint)bits << 23);
    return as_float(f_bits);
}

// MXFP4 dequantize kernel (float32 output)
__kernel void mxfp4_dequantize_f32(
    __global const uchar* packed,
    __global const uchar* scales,
    __global float* output,
    int N,
    int K
) {
    int gid = get_global_id(0);
    int total_elements = N * K;
    if (gid >= total_elements) return;

    int n = gid / K;
    int k = gid % K;

    int byte_idx = gid / 2;
    int pos_in_byte = gid % 2;

    uchar packed_val = packed[byte_idx];
    uchar q = (pos_in_byte == 0) ? (packed_val & 0xF) : (packed_val >> 4);

    int group_idx = k / 32;
    int num_groups = K / 32;
    uchar scale_bits = scales[n * num_groups + group_idx];
    float scale = fp8_e8m0_to_float(scale_bits);

    output[gid] = scale * fp4_to_float(q);
}
)";

  return oss.str();
}

std::string generate_mxfp4_dequantize_kernel_bf16() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  oss << R"(
// Convert fp4_e2m1 to float
// Lookup table for fp4_e2m1 to float conversion
// Avoids denormalized half-precision which some GPUs flush to zero
__constant float fp4_lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float fp4_to_float(uchar bits) {
    float val = fp4_lut_vals[bits & 7];
    return (bits & 8) ? -val : val;
}

// Convert fp8_e8m0 scale to float
float fp8_e8m0_to_float(uchar bits) {
    // Special case: bits=0 represents 2^-127 (denormalized float 0x00400000)
    uint f_bits = (bits == 0) ? 0x00400000 : ((uint)bits << 23);
    return as_float(f_bits);
}

// MXFP4 dequantize kernel (bfloat16 output)
__kernel void mxfp4_dequantize_bf16(
    __global const uchar* packed,
    __global const uchar* scales,
    __global bfloat16_t* output,
    int N,
    int K
) {
    int gid = get_global_id(0);
    int total_elements = N * K;
    if (gid >= total_elements) return;

    int n = gid / K;
    int k = gid % K;

    int byte_idx = gid / 2;
    int pos_in_byte = gid % 2;

    uchar packed_val = packed[byte_idx];
    uchar q = (pos_in_byte == 0) ? (packed_val & 0xF) : (packed_val >> 4);

    int group_idx = k / 32;
    int num_groups = K / 32;
    uchar scale_bits = scales[n * num_groups + group_idx];
    float scale = fp8_e8m0_to_float(scale_bits);

    float val = scale * fp4_to_float(q);
    output[gid] = float_to_bfloat16(val);
}
)";

  return oss.str();
}

// Optimized MXFP4 matrix-vector kernel for M=1 (token generation)
std::string generate_mxfp4_qmv_kernel_f16() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  oss << R"(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

// Convert fp4_e2m1 to float
// Lookup table for fp4_e2m1 to float conversion
// Avoids denormalized half-precision which some GPUs flush to zero
__constant float fp4_lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float fp4_to_float(uchar bits) {
    float val = fp4_lut_vals[bits & 7];
    return (bits & 8) ? -val : val;
}

// Convert fp8_e8m0 scale to float
float fp8_e8m0_to_float(uchar bits) {
    // Special case: bits=0 represents 2^-127 (denormalized float 0x00400000)
    uint f_bits = (bits == 0) ? 0x00400000 : ((uint)bits << 23);
    return as_float(f_bits);
}

// MXFP4 quantized matrix-vector multiply for token generation (M=1)
// Each workgroup computes 8 output elements
// group_size = 32 (fixed for mxfp4)
// Simple parallel reduction using local memory
#define WORKGROUP_SIZE 64
#define RESULTS_PER_WG 8
#define GROUP_SIZE 32

__kernel void mxfp4_qmv_f16(
    __global const half* x,           // Input [1, K]
    __global const uchar* w_q,        // Packed fp4 weights [N, K/2]
    __global const uchar* scales,     // fp8_e8m0 scales [N, K/32]
    __global half* out,               // Output [1, N]
    int N, int K
) {
    // Local memory for parallel reduction
    __local float local_sums[RESULTS_PER_WG * WORKGROUP_SIZE];

    int wg_id = get_group_id(0);
    int lid = get_local_id(0);
    int n_base = wg_id * RESULTS_PER_WG;

    int K_packed = K / 2;  // 2 fp4 values per byte
    int num_groups = K / GROUP_SIZE;

    float sums[RESULTS_PER_WG];
    #pragma unroll
    for (int r = 0; r < RESULTS_PER_WG; r++) {
        sums[r] = 0.0f;
    }

    for (int byte_idx = lid; byte_idx < K_packed; byte_idx += WORKGROUP_SIZE) {
        int k_base = byte_idx * 2;
        int group_idx = k_base / GROUP_SIZE;

        float x0 = convert_float(x[k_base]);
        float x1 = convert_float(x[k_base + 1]);

        #pragma unroll
        for (int r = 0; r < RESULTS_PER_WG; r++) {
            int n = n_base + r;
            if (n >= N) continue;

            uchar packed = w_q[n * K_packed + byte_idx];
            float scale = fp8_e8m0_to_float(scales[n * num_groups + group_idx]);

            uchar q0 = packed & 0xF;
            uchar q1 = (packed >> 4) & 0xF;

            sums[r] += x0 * (scale * fp4_to_float(q0));
            sums[r] += x1 * (scale * fp4_to_float(q1));
        }
    }

    // Write partial sums to local memory
    #pragma unroll
    for (int r = 0; r < RESULTS_PER_WG; r++) {
        local_sums[r * WORKGROUP_SIZE + lid] = sums[r];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction in local memory
    for (int stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            #pragma unroll
            for (int r = 0; r < RESULTS_PER_WG; r++) {
                local_sums[r * WORKGROUP_SIZE + lid] += local_sums[r * WORKGROUP_SIZE + lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Thread 0 writes final results
    if (lid == 0) {
        #pragma unroll
        for (int r = 0; r < RESULTS_PER_WG; r++) {
            int n = n_base + r;
            if (n < N) {
                out[n] = convert_half(local_sums[r * WORKGROUP_SIZE]);
            }
        }
    }
}
)";

  return oss.str();
}

// General MXFP4 matmul kernel (for M > 1)
std::string generate_mxfp4_matmul_kernel_f16() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  oss << R"(
// Convert fp4_e2m1 to float
// Lookup table for fp4_e2m1 to float conversion
// Avoids denormalized half-precision which some GPUs flush to zero
__constant float fp4_lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float fp4_to_float(uchar bits) {
    float val = fp4_lut_vals[bits & 7];
    return (bits & 8) ? -val : val;
}

// Convert fp8_e8m0 scale to float
float fp8_e8m0_to_float(uchar bits) {
    // Special case: bits=0 represents 2^-127 (denormalized float 0x00400000)
    uint f_bits = (bits == 0) ? 0x00400000 : ((uint)bits << 23);
    return as_float(f_bits);
}

// MXFP4 quantized matmul: out[m,n] = sum_k(x[m,k] * dequant(w_q[n,k]))
// w_q is stored as [N, K/2] with 2 fp4 values per byte
// scales are [N, K/32]
#define GROUP_SIZE 32

__kernel void mxfp4_qmatmul_f16(
    __global const half* x,           // Input [M, K]
    __global const uchar* w_q,        // Packed fp4 weights [N, K/2]
    __global const uchar* scales,     // fp8_e8m0 scales [N, K/32]
    __global half* out,               // Output [M, N]
    int M, int N, int K
) {
    int m = get_global_id(0);
    int n = get_global_id(1);

    if (m >= M || n >= N) return;

    float sum = 0.0f;
    int K_packed = K / 2;
    int num_groups = K / GROUP_SIZE;

    for (int byte_idx = 0; byte_idx < K_packed; byte_idx++) {
        int k_base = byte_idx * 2;
        int group_idx = k_base / GROUP_SIZE;

        float x0 = convert_float(x[m * K + k_base]);
        float x1 = convert_float(x[m * K + k_base + 1]);

        uchar packed = w_q[n * K_packed + byte_idx];
        float scale = fp8_e8m0_to_float(scales[n * num_groups + group_idx]);

        uchar q0 = packed & 0xF;
        uchar q1 = (packed >> 4) & 0xF;

        sum += x0 * (scale * fp4_to_float(q0));
        sum += x1 * (scale * fp4_to_float(q1));
    }

    out[m * N + n] = convert_half(sum);
}
)";

  return oss.str();
}

// Bfloat16 version of MXFP4 qmv kernel
// Optimized using subgroup reduction and vectorized loads
std::string generate_mxfp4_qmv_kernel_bf16() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  oss << R"(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#endif

// Full lookup table for fp4_e2m1 (all 16 values including negative)
__constant float fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

float fp8_e8m0_to_float(uchar bits) {
    uint f_bits = (bits == 0) ? 0x00400000 : ((uint)bits << 23);
    return as_float(f_bits);
}

// Parameters for Adreno X1-85:
// - Subgroup size: 64 threads (half wavefront)
// - Each subgroup processes 2 output rows
// - 2 subgroups per workgroup = 128 threads total
#define N_ROWS_PER_SG 2
#define N_SUBGROUPS 2
#define SUBGROUP_SIZE 64
#define WORKGROUP_SIZE (N_SUBGROUPS * SUBGROUP_SIZE)
#define GROUP_SIZE 32

#ifdef ADRENO_GPU
__attribute__((qcom_reqd_sub_group_size("half")))
#endif
__kernel void mxfp4_qmv_bf16(
    __global const bfloat16_t* x,
    __global const uchar* w_q,
    __global const uchar* scales,
    __global bfloat16_t* out,
    int N, int K
) {
    // Cache fp4 LUT in local memory for faster access
    __local float shmem_lut[16];
    int lid = get_local_id(0);
    if (lid < 16) {
        shmem_lut[lid] = fp4_lut[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int wg_id = get_group_id(0);
    int sg_id = get_sub_group_id();
    int sg_lid = get_sub_group_local_id();

    // Each workgroup processes N_SUBGROUPS * N_ROWS_PER_SG outputs
    int n_base = wg_id * (N_SUBGROUPS * N_ROWS_PER_SG) + sg_id * N_ROWS_PER_SG;

    int K_packed = K / 2;  // 2 fp4 values per byte
    int num_groups = K / GROUP_SIZE;

    // Half of subgroup processes odd bytes, half processes even
    // Each thread pair processes 32 elements (1 group)
    int ix = sg_lid / 2;   // 0...31 (which group pair)
    int it = sg_lid % 2;   // 0 or 1 (which half of group)

    float sumf[N_ROWS_PER_SG] = {0.0f, 0.0f};

    // Process K in chunks of SUBGROUP_SIZE/2 groups (32 groups = 1024 elements)
    for (int group_base = ix; group_base < num_groups; group_base += SUBGROUP_SIZE/2) {
        int byte_start = group_base * (GROUP_SIZE / 2) + it * 8;  // 8 bytes per half-group

        // Load 8 input values (16 bytes = 8 bfloat16)
        int k_start = group_base * GROUP_SIZE + it * 16;
        if (k_start >= K) continue;

        // Load input values
        float4 x_vals0, x_vals1;
        x_vals0.s0 = bfloat16_to_float(x[k_start + 0]);
        x_vals0.s1 = bfloat16_to_float(x[k_start + 1]);
        x_vals0.s2 = bfloat16_to_float(x[k_start + 2]);
        x_vals0.s3 = bfloat16_to_float(x[k_start + 3]);
        x_vals1.s0 = bfloat16_to_float(x[k_start + 4]);
        x_vals1.s1 = bfloat16_to_float(x[k_start + 5]);
        x_vals1.s2 = bfloat16_to_float(x[k_start + 6]);
        x_vals1.s3 = bfloat16_to_float(x[k_start + 7]);

        // Second half of 16 elements
        float4 x_vals2, x_vals3;
        x_vals2.s0 = bfloat16_to_float(x[k_start + 8]);
        x_vals2.s1 = bfloat16_to_float(x[k_start + 9]);
        x_vals2.s2 = bfloat16_to_float(x[k_start + 10]);
        x_vals2.s3 = bfloat16_to_float(x[k_start + 11]);
        x_vals3.s0 = bfloat16_to_float(x[k_start + 12]);
        x_vals3.s1 = bfloat16_to_float(x[k_start + 13]);
        x_vals3.s2 = bfloat16_to_float(x[k_start + 14]);
        x_vals3.s3 = bfloat16_to_float(x[k_start + 15]);

        #pragma unroll
        for (int r = 0; r < N_ROWS_PER_SG; r++) {
            int n = n_base + r;
            if (n >= N) continue;

            // Get scale for this group
            float scale = fp8_e8m0_to_float(scales[n * num_groups + group_base]);

            // Load 8 packed bytes (16 fp4 values)
            int w_base = n * K_packed + byte_start;
            uchar4 packed0 = vload4(0, w_q + w_base);
            uchar4 packed1 = vload4(1, w_q + w_base);

            // Dequantize and accumulate
            float4 acc1 = x_vals0 * (float4)(shmem_lut[packed0.s0 & 0xF], shmem_lut[packed0.s1 & 0xF],
                                             shmem_lut[packed0.s2 & 0xF], shmem_lut[packed0.s3 & 0xF]);
            float4 acc2 = x_vals1 * (float4)(shmem_lut[packed0.s0 >> 4], shmem_lut[packed0.s1 >> 4],
                                             shmem_lut[packed0.s2 >> 4], shmem_lut[packed0.s3 >> 4]);
            float4 acc3 = x_vals2 * (float4)(shmem_lut[packed1.s0 & 0xF], shmem_lut[packed1.s1 & 0xF],
                                             shmem_lut[packed1.s2 & 0xF], shmem_lut[packed1.s3 & 0xF]);
            float4 acc4 = x_vals3 * (float4)(shmem_lut[packed1.s0 >> 4], shmem_lut[packed1.s1 >> 4],
                                             shmem_lut[packed1.s2 >> 4], shmem_lut[packed1.s3 >> 4]);

            acc1 = (acc1 + acc3) + (acc2 + acc4);
            sumf[r] += scale * ((acc1.s0 + acc1.s1) + (acc1.s2 + acc1.s3));
        }
    }

    // Subgroup reduction
    #pragma unroll
    for (int r = 0; r < N_ROWS_PER_SG; r++) {
        float sum_all = sub_group_reduce_add(sumf[r]);
        if (sg_lid == 0) {
            int n = n_base + r;
            if (n < N) {
                out[n] = float_to_bfloat16(sum_all);
            }
        }
    }
}
)";

  return oss.str();
}

// Float32 versions of MXFP4 kernels
std::string generate_mxfp4_qmv_kernel_f32() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  oss << R"(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

// Lookup table for fp4_e2m1 to float conversion
// Avoids denormalized half-precision which some GPUs flush to zero
__constant float fp4_lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float fp4_to_float(uchar bits) {
    float val = fp4_lut_vals[bits & 7];
    return (bits & 8) ? -val : val;
}

float fp8_e8m0_to_float(uchar bits) {
    // Special case: bits=0 represents 2^-127 (denormalized float 0x00400000)
    uint f_bits = (bits == 0) ? 0x00400000 : ((uint)bits << 23);
    return as_float(f_bits);
}

// Simple parallel reduction using local memory
#define WORKGROUP_SIZE 64
#define RESULTS_PER_WG 8
#define GROUP_SIZE 32

__kernel void mxfp4_qmv_f32(
    __global const float* x,
    __global const uchar* w_q,
    __global const uchar* scales,
    __global float* out,
    int N, int K
) {
    // Local memory for parallel reduction
    __local float local_sums[RESULTS_PER_WG * WORKGROUP_SIZE];

    int wg_id = get_group_id(0);
    int lid = get_local_id(0);
    int n_base = wg_id * RESULTS_PER_WG;

    int K_packed = K / 2;
    int num_groups = K / GROUP_SIZE;

    float sums[RESULTS_PER_WG];
    #pragma unroll
    for (int r = 0; r < RESULTS_PER_WG; r++) {
        sums[r] = 0.0f;
    }

    for (int byte_idx = lid; byte_idx < K_packed; byte_idx += WORKGROUP_SIZE) {
        int k_base = byte_idx * 2;
        int group_idx = k_base / GROUP_SIZE;

        float x0 = x[k_base];
        float x1 = x[k_base + 1];

        #pragma unroll
        for (int r = 0; r < RESULTS_PER_WG; r++) {
            int n = n_base + r;
            if (n >= N) continue;

            uchar packed = w_q[n * K_packed + byte_idx];
            float scale = fp8_e8m0_to_float(scales[n * num_groups + group_idx]);

            uchar q0 = packed & 0xF;
            uchar q1 = (packed >> 4) & 0xF;

            sums[r] += x0 * (scale * fp4_to_float(q0));
            sums[r] += x1 * (scale * fp4_to_float(q1));
        }
    }

    // Write partial sums to local memory
    #pragma unroll
    for (int r = 0; r < RESULTS_PER_WG; r++) {
        local_sums[r * WORKGROUP_SIZE + lid] = sums[r];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction in local memory
    for (int stride = WORKGROUP_SIZE / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            #pragma unroll
            for (int r = 0; r < RESULTS_PER_WG; r++) {
                local_sums[r * WORKGROUP_SIZE + lid] += local_sums[r * WORKGROUP_SIZE + lid + stride];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Thread 0 writes final results
    if (lid == 0) {
        #pragma unroll
        for (int r = 0; r < RESULTS_PER_WG; r++) {
            int n = n_base + r;
            if (n < N) {
                out[n] = local_sums[r * WORKGROUP_SIZE];
            }
        }
    }
}
)";

  return oss.str();
}

// Bfloat16 version of MXFP4 matmul kernel (for M > 1)
std::string generate_mxfp4_matmul_kernel_bf16() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  oss << R"(
// Lookup table for fp4_e2m1 to float conversion
// Avoids denormalized half-precision which some GPUs flush to zero
__constant float fp4_lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float fp4_to_float(uchar bits) {
    float val = fp4_lut_vals[bits & 7];
    return (bits & 8) ? -val : val;
}

float fp8_e8m0_to_float(uchar bits) {
    // Special case: bits=0 represents 2^-127 (denormalized float 0x00400000)
    uint f_bits = (bits == 0) ? 0x00400000 : ((uint)bits << 23);
    return as_float(f_bits);
}

#define GROUP_SIZE 32

__kernel void mxfp4_qmatmul_bf16(
    __global const bfloat16_t* x,
    __global const uchar* w_q,
    __global const uchar* scales,
    __global bfloat16_t* out,
    int M, int N, int K
) {
    int m = get_global_id(0);
    int n = get_global_id(1);

    if (m >= M || n >= N) return;

    float sum = 0.0f;
    int K_packed = K / 2;
    int num_groups = K / GROUP_SIZE;

    for (int byte_idx = 0; byte_idx < K_packed; byte_idx++) {
        int k_base = byte_idx * 2;
        int group_idx = k_base / GROUP_SIZE;

        // Convert bfloat16 to float
        float x0 = bfloat16_to_float(x[m * K + k_base]);
        float x1 = bfloat16_to_float(x[m * K + k_base + 1]);

        uchar packed = w_q[n * K_packed + byte_idx];
        float scale = fp8_e8m0_to_float(scales[n * num_groups + group_idx]);

        uchar q0 = packed & 0xF;
        uchar q1 = (packed >> 4) & 0xF;

        sum += x0 * (scale * fp4_to_float(q0));
        sum += x1 * (scale * fp4_to_float(q1));
    }

    out[m * N + n] = float_to_bfloat16(sum);
}
)";

  return oss.str();
}

std::string generate_mxfp4_matmul_kernel_f32() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  oss << R"(
// Lookup table for fp4_e2m1 to float conversion
// Avoids denormalized half-precision which some GPUs flush to zero
__constant float fp4_lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float fp4_to_float(uchar bits) {
    float val = fp4_lut_vals[bits & 7];
    return (bits & 8) ? -val : val;
}

float fp8_e8m0_to_float(uchar bits) {
    // Special case: bits=0 represents 2^-127 (denormalized float 0x00400000)
    uint f_bits = (bits == 0) ? 0x00400000 : ((uint)bits << 23);
    return as_float(f_bits);
}

#define GROUP_SIZE 32

__kernel void mxfp4_qmatmul_f32(
    __global const float* x,
    __global const uchar* w_q,
    __global const uchar* scales,
    __global float* out,
    int M, int N, int K
) {
    int m = get_global_id(0);
    int n = get_global_id(1);

    if (m >= M || n >= N) return;

    float sum = 0.0f;
    int K_packed = K / 2;
    int num_groups = K / GROUP_SIZE;

    for (int byte_idx = 0; byte_idx < K_packed; byte_idx++) {
        int k_base = byte_idx * 2;
        int group_idx = k_base / GROUP_SIZE;

        float x0 = x[m * K + k_base];
        float x1 = x[m * K + k_base + 1];

        uchar packed = w_q[n * K_packed + byte_idx];
        float scale = fp8_e8m0_to_float(scales[n * num_groups + group_idx]);

        uchar q0 = packed & 0xF;
        uchar q1 = (packed >> 4) & 0xF;

        sum += x0 * (scale * fp4_to_float(q0));
        sum += x1 * (scale * fp4_to_float(q1));
    }

    out[m * N + n] = sum;
}
)";

  return oss.str();
}

} // anonymous namespace

void fast::Quantize::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {

  auto& s = stream();
  auto& dev = opencl::device(s.device);

  // Support affine and mxfp4 modes
  if (mode_ != QuantizationMode::Affine && mode_ != QuantizationMode::Mxfp4) {
    throw std::runtime_error(
        "[Quantize::eval_gpu] Only Affine and Mxfp4 quantization modes supported on OpenCL");
  }

  // Validate bit width based on mode
  if (mode_ == QuantizationMode::Affine) {
    if (bits_ != 2 && bits_ != 4 && bits_ != 8) {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Only 2-bit, 4-bit and 8-bit quantization supported for Affine mode");
    }
  } else if (mode_ == QuantizationMode::Mxfp4) {
    if (bits_ != 4) {
      throw std::runtime_error(
          "[Quantize::eval_gpu] Mxfp4 mode requires 4-bit quantization");
    }
  }

  auto& encoder = dev.get_command_encoder(s.index);

  if (dequantize_) {
    // Dequantize operation
    auto& packed = inputs[0];
    auto& scales = inputs[1];
    auto& out = outputs[0];

    // Biases only used for affine mode (mxfp4 has no biases)
    bool has_biases = (mode_ == QuantizationMode::Affine) && inputs.size() > 2;

    OPENCL_DEBUG_LOG("[Quantize::eval_gpu] Dequantize ENTRY"
              << " mode=" << (mode_ == QuantizationMode::Mxfp4 ? "mxfp4" : "affine")
              << " bits=" << bits_ << " group_size=" << group_size_);
    OPENCL_DEBUG_LOG("[Quantize::eval_gpu] packed: shape=(" << packed.shape(0)
              << (packed.ndim() > 1 ? ", " + std::to_string(packed.shape(1)) : "")
              << (packed.ndim() > 2 ? ", " + std::to_string(packed.shape(2)) : "") << ")"
              << " dtype=" << packed.dtype());
    OPENCL_DEBUG_LOG("[Quantize::eval_gpu] scales: shape=(" << scales.shape(0)
              << (scales.ndim() > 1 ? ", " + std::to_string(scales.shape(1)) : "")
              << (scales.ndim() > 2 ? ", " + std::to_string(scales.shape(2)) : "") << ")"
              << " dtype=" << scales.dtype());
    OPENCL_DEBUG_LOG("[Quantize::eval_gpu] out: shape=(" << out.shape(0)
              << (out.ndim() > 1 ? ", " + std::to_string(out.shape(1)) : "")
              << (out.ndim() > 2 ? ", " + std::to_string(out.shape(2)) : "") << ")"
              << " dtype=" << out.dtype());

    // Ensure inputs are contiguous
    array packed_contig = packed.flags().row_contiguous ? packed : contiguous_copy_gpu(packed, s);
    array scales_contig = scales.flags().row_contiguous ? scales : contiguous_copy_gpu(scales, s);

    out.set_data(opencl::allocator().malloc(out.nbytes()));

    if (mode_ == QuantizationMode::Mxfp4) {
      // MXFP4 dequantization: no biases, uses fp4_e2m1 values and fp8_e8m0 scales
      bool is_f16 = (out.dtype() == float16);
      bool is_bf16 = (out.dtype() == bfloat16);

      std::string kernel_src;
      std::string kernel_name;
      if (is_f16) {
        kernel_src = generate_mxfp4_dequantize_kernel_f16();
        kernel_name = "mxfp4_dequantize_f16";
      } else if (is_bf16) {
        kernel_src = generate_mxfp4_dequantize_kernel_bf16();
        kernel_name = "mxfp4_dequantize_bf16";
      } else {
        kernel_src = generate_mxfp4_dequantize_kernel_f32();
        kernel_name = "mxfp4_dequantize_f32";
      }
      cl_kernel kernel = dev.get_kernel(kernel_name, kernel_src);

      // Get dimensions from output shape
      // For mxfp4, output is [N, K] where N = rows, K = columns (group_size=32)
      int total_elements = out.size();
      int N = out.shape().size() >= 2 ? out.shape(out.ndim() - 2) : 1;
      int K = out.shape(out.ndim() - 1);

      encoder.set_kernel(kernel);
      encoder.set_input_array(packed_contig, 0);
      encoder.set_input_array(scales_contig, 1);
      encoder.set_output_array(out, 2);
      encoder.set_bytes(N, 3);
      encoder.set_bytes(K, 4);

      size_t global_size[3] = {static_cast<size_t>((total_elements + 255) / 256 * 256), 0, 0};
      size_t local_size[3] = {256, 0, 0};
      encoder.dispatch_threads(global_size, local_size, 1);

      dev.add_temporary(packed_contig, s.index);
      dev.add_temporary(scales_contig, s.index);
      dev.end_encoding(s.index);
    } else {
      // Affine dequantization: uses biases
      auto& biases = inputs[2];
      array biases_contig = biases.flags().row_contiguous ? biases : contiguous_copy_gpu(biases, s);

      // Select kernel based on output dtype
      bool is_f16 = (out.dtype() == float16);
      std::string kernel_src;
      std::string kernel_name;
      if (is_f16) {
        kernel_src = generate_dequantize_kernel_f16(bits_, group_size_);
        kernel_name = "affine_dequantize_f16";
      } else {
        kernel_src = generate_dequantize_kernel_f32(bits_, group_size_);
        kernel_name = "affine_dequantize_f32";
      }
      cl_kernel kernel = dev.get_kernel(kernel_name, kernel_src);

      encoder.set_kernel(kernel);
      encoder.set_input_array(packed_contig, 0);
      encoder.set_input_array(scales_contig, 1);
      encoder.set_input_array(biases_contig, 2);
      encoder.set_output_array(out, 3);
      encoder.set_bytes(static_cast<int>(out.size()), 4);
      encoder.set_bytes(group_size_, 5);

      size_t global_size[3] = {static_cast<size_t>((out.size() + 255) / 256 * 256), 0, 0};
      size_t local_size[3] = {256, 0, 0};
      encoder.dispatch_threads(global_size, local_size, 1);

      dev.add_temporary(packed_contig, s.index);
      dev.add_temporary(scales_contig, s.index);
      dev.add_temporary(biases_contig, s.index);
      dev.end_encoding(s.index);
    }

  } else {
    // Quantize: [input] -> [packed, scales] for mxfp4, [packed, scales, biases] for affine
    auto& input = inputs[0];
    auto& packed_out = outputs[0];
    auto& scales_out = outputs[1];

    OPENCL_DEBUG_LOG("[Quantize::eval_gpu] Quantize: input.size=" << input.size()
              << " packed.size=" << packed_out.size()
              << " mode=" << (mode_ == QuantizationMode::Mxfp4 ? "mxfp4" : "affine"));

    // Ensure input is contiguous
    array input_contig = input.flags().row_contiguous ? input : contiguous_copy_gpu(input, s);

    packed_out.set_data(opencl::allocator().malloc(packed_out.nbytes()));
    scales_out.set_data(opencl::allocator().malloc(scales_out.nbytes()));

    if (mode_ == QuantizationMode::Mxfp4) {
      // MXFP4 quantization: convert float values to fp4_e2m1 with fp8_e8m0 scales
      // No biases for MXFP4 mode
      int N = input.size();
      int K = input.shape(input.ndim() - 1);  // Last dimension
      int num_rows = N / K;
      int num_groups_per_row = (K + 31) / 32;  // group_size=32 for mxfp4

      // Determine input type
      bool is_bf16 = (input.dtype() == bfloat16);
      bool is_f16 = (input.dtype() == float16);
      std::string input_type = is_bf16 ? "bfloat16_t" : (is_f16 ? "half" : "float");
      std::string kernel_name = std::string("mxfp4_quantize_") + (is_bf16 ? "bf16" : (is_f16 ? "f16" : "f32"));
      std::string convert_to_float = is_bf16 ? "bfloat16_to_float" : (is_f16 ? "convert_float" : "");

      // Generate MXFP4 quantize kernel
      std::ostringstream ksrc;
      ksrc << opencl::get_kernel_preamble();

      // FP4_E2M1 lookup table (same as dequantize)
      ksrc << R"(
// FP4_E2M1 values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
__constant float fp4_lut[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

// Find closest FP4 value (returns 4-bit code including sign)
inline uchar float_to_fp4(float val) {
  // Handle sign
  uchar sign = val < 0.0f ? 8 : 0;
  float abs_val = fabs(val);

  // Find closest FP4 magnitude
  uchar best_code = 0;
  float best_diff = abs_val;  // diff from 0

  for (uchar i = 1; i < 8; i++) {
    float diff = fabs(abs_val - fp4_lut[i]);
    if (diff < best_diff) {
      best_diff = diff;
      best_code = i;
    }
  }

  return sign | best_code;
}

// Compute FP8_E8M0 scale (power of 2) for a group
// Returns exponent byte where scale = 2^(byte - 127)
// Matches Metal implementation: scale = max_abs / 6.0, then round(log2(scale))
inline uchar compute_fp8_scale(float max_abs) {
  if (max_abs == 0.0f) return 127;  // 2^0 = 1.0

  // Match Metal: scale = max_abs / 6.0, then round(log2(scale))
  float scale = max_abs / 6.0f;
  int n = (int)round(log2(scale));

  // Clamp to valid range [-127, 127], then add 127 for final byte
  n = max(-127, min(127, n));
  return (uchar)(n + 127);
}

__kernel void )" << kernel_name << R"((
    __global const )" << input_type << R"(* input,
    __global uchar* packed_out,
    __global uchar* scales_out,
    int num_rows,
    int K,
    int num_groups_per_row) {

  int row = get_global_id(0);
  if (row >= num_rows) return;

  __global const )" << input_type << R"(* row_in = input + row * K;
  __global uchar* row_packed = packed_out + row * (K / 2);  // 2 fp4 values per byte
  __global uchar* row_scales = scales_out + row * num_groups_per_row;

  // Process each group of 32 elements
  for (int g = 0; g < num_groups_per_row; g++) {
    int group_start = g * 32;
    int group_end = min(group_start + 32, K);

    // Find max absolute value in group
    float max_abs = 0.0f;
    for (int i = group_start; i < group_end; i++) {
      float val = )" << (convert_to_float.empty() ? "row_in[i]" : convert_to_float + "(row_in[i])") << R"(;
      max_abs = fmax(max_abs, fabs(val));
    }

    // Compute scale
    uchar scale_byte = compute_fp8_scale(max_abs);
    row_scales[g] = scale_byte;

    // Compute actual scale value: 2^(scale_byte - 127)
    float scale = exp2((float)(scale_byte - 127));
    float inv_scale = 1.0f / scale;

    // Quantize and pack values
    for (int i = group_start; i < group_end; i += 2) {
      float raw0 = )" << (convert_to_float.empty() ? "row_in[i]" : convert_to_float + "(row_in[i])") << R"(;
      float raw1 = )" << (convert_to_float.empty() ? "row_in[i + 1]" : convert_to_float + "(row_in[i + 1])") << R"(;
      float v0 = (i < K) ? raw0 * inv_scale : 0.0f;
      float v1 = (i + 1 < K) ? raw1 * inv_scale : 0.0f;

      uchar code0 = float_to_fp4(v0);
      uchar code1 = float_to_fp4(v1);

      // Pack: low nibble = first value, high nibble = second value
      row_packed[(i - group_start) / 2 + (g * 16)] = code0 | (code1 << 4);
    }
  }
}
)";

      std::string kernel_src = ksrc.str();
      cl_kernel kernel = dev.get_kernel(kernel_name, kernel_src);

      encoder.set_kernel(kernel);
      encoder.set_input_array(input_contig, 0);
      encoder.set_output_array(packed_out, 1);
      encoder.set_output_array(scales_out, 2);
      encoder.set_bytes(num_rows, 3);
      encoder.set_bytes(K, 4);
      encoder.set_bytes(num_groups_per_row, 5);

      size_t global_size[3] = {static_cast<size_t>((num_rows + 255) / 256 * 256), 0, 0};
      size_t local_size[3] = {256, 0, 0};
      encoder.dispatch_threads(global_size, local_size, 1);

      dev.add_temporary(input_contig, s.index);
      dev.end_encoding(s.index);

    } else {
      // Affine quantization: uses biases
      auto& biases_out = outputs[2];
      biases_out.set_data(opencl::allocator().malloc(biases_out.nbytes()));

      int N = input.size();
      int num_groups = (N + group_size_ - 1) / group_size_;
      int pack_factor = 32 / bits_;  // Values per uint32

      // Allocate temporary buffers for min/max
      array group_mins = array({num_groups}, float32, nullptr, {});
      array group_maxs = array({num_groups}, float32, nullptr, {});
      group_mins.set_data(opencl::allocator().malloc(group_mins.nbytes()));
      group_maxs.set_data(opencl::allocator().malloc(group_maxs.nbytes()));

      // Determine input and scales types
      std::string input_type = "float";
      std::string scales_type = "float";
      if (input.dtype() == bfloat16) {
        input_type = "bfloat16_t";
        scales_type = "bfloat16_t";
      } else if (input.dtype() == float16) {
        input_type = "half";
        scales_type = "half";
      }

      std::string kernel_src = generate_quantize_kernel(bits_, group_size_, input_type, scales_type);

      // Pass 1: Compute min/max per group
      cl_kernel minmax_kernel = dev.get_kernel("affine_quantize_minmax", kernel_src);
      encoder.set_kernel(minmax_kernel);
      encoder.set_input_array(input_contig, 0);
      encoder.set_output_array(group_mins, 1);
      encoder.set_output_array(group_maxs, 2);
      encoder.set_bytes(N, 3);
      encoder.set_bytes(group_size_, 4);

      size_t minmax_global[3] = {static_cast<size_t>((num_groups + 255) / 256 * 256), 0, 0};
      size_t minmax_local[3] = {256, 0, 0};
      encoder.dispatch_threads(minmax_global, minmax_local, 1);
      dev.end_encoding(s.index);

      // Pass 2: Compute scales and biases
      auto& enc2 = dev.get_command_encoder(s.index);
      cl_kernel scales_kernel = dev.get_kernel("affine_quantize_scales", kernel_src);
      enc2.set_kernel(scales_kernel);
      enc2.set_input_array(group_mins, 0);
      enc2.set_input_array(group_maxs, 1);
      enc2.set_output_array(scales_out, 2);
      enc2.set_output_array(biases_out, 3);
      enc2.set_bytes(num_groups, 4);
      enc2.set_bytes((1 << bits_) - 1, 5);  // max_quant_val

      enc2.dispatch_threads(minmax_global, minmax_local, 1);
      dev.end_encoding(s.index);

      // Pass 3: Quantize and pack
      auto& enc3 = dev.get_command_encoder(s.index);
      cl_kernel pack_kernel = dev.get_kernel("affine_quantize_pack", kernel_src);
      enc3.set_kernel(pack_kernel);
      enc3.set_input_array(input_contig, 0);
      enc3.set_input_array(scales_out, 1);
      enc3.set_input_array(biases_out, 2);
      enc3.set_output_array(packed_out, 3);
      enc3.set_bytes(N, 4);
      enc3.set_bytes(group_size_, 5);
      enc3.set_bytes(bits_, 6);
      enc3.set_bytes(pack_factor, 7);

      int num_packs = (N + pack_factor - 1) / pack_factor;
      size_t pack_global[3] = {static_cast<size_t>((num_packs + 255) / 256 * 256), 0, 0};
      enc3.dispatch_threads(pack_global, minmax_local, 1);

      dev.add_temporary(input_contig, s.index);
      dev.add_temporary(group_mins, s.index);
      dev.add_temporary(group_maxs, s.index);
      dev.end_encoding(s.index);
    }
  }
}

void QuantizedMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& dev = opencl::device(s.device);

  auto [group_size, bits, mode, transpose] = state();

  // Support affine and mxfp4 modes
  if (mode != QuantizationMode::Affine && mode != QuantizationMode::Mxfp4) {
    throw std::runtime_error(
        "[QuantizedMatmul::eval_gpu] Only Affine and Mxfp4 quantization modes supported on OpenCL");
  }

  // Validate bit width based on mode
  if (mode == QuantizationMode::Affine) {
    if (bits != 2 && bits != 4 && bits != 8) {
      throw std::runtime_error(
          "[QuantizedMatmul::eval_gpu] Only 2-bit, 4-bit and 8-bit quantization supported for Affine mode");
    }
  } else if (mode == QuantizationMode::Mxfp4) {
    if (bits != 4) {
      throw std::runtime_error(
          "[QuantizedMatmul::eval_gpu] Mxfp4 mode requires 4-bit quantization");
    }
  }

  // inputs: [x, w_q, scales, biases (optional for affine)]
  auto& x = inputs[0];
  auto& w_q = inputs[1];
  auto& scales = inputs[2];
  // Biases only used for affine mode
  std::optional<array> biases = std::nullopt;
  if (mode == QuantizationMode::Affine && inputs.size() == 4) {
    biases = inputs[3];
  }

  OPENCL_DEBUG_LOG("[QuantizedMatmul::eval_gpu] ENTRY"
            << " mode=" << (mode == QuantizationMode::Mxfp4 ? "mxfp4" : "affine")
            << " bits=" << bits << " group_size=" << group_size << " transpose=" << transpose);
  OPENCL_DEBUG_LOG("[QuantizedMatmul::eval_gpu] x: shape=[" << x.shape(0)
            << (x.ndim() > 1 ? ", " + std::to_string(x.shape(1)) : "")
            << (x.ndim() > 2 ? ", " + std::to_string(x.shape(2)) : "") << "]"
            << " dtype=" << x.dtype());
  OPENCL_DEBUG_LOG("[QuantizedMatmul::eval_gpu] w_q: shape=[" << w_q.shape(0)
            << (w_q.ndim() > 1 ? ", " + std::to_string(w_q.shape(1)) : "")
            << (w_q.ndim() > 2 ? ", " + std::to_string(w_q.shape(2)) : "") << "]"
            << " dtype=" << w_q.dtype());
  OPENCL_DEBUG_LOG("[QuantizedMatmul::eval_gpu] scales: shape=[" << scales.shape(0)
            << (scales.ndim() > 1 ? ", " + std::to_string(scales.shape(1)) : "")
            << (scales.ndim() > 2 ? ", " + std::to_string(scales.shape(2)) : "") << "]"
            << " dtype=" << scales.dtype());
  OPENCL_DEBUG_LOG("[QuantizedMatmul::eval_gpu] out: shape=[" << out.shape(0)
            << (out.ndim() > 1 ? ", " + std::to_string(out.shape(1)) : "")
            << (out.ndim() > 2 ? ", " + std::to_string(out.shape(2)) : "") << "]"
            << " dtype=" << out.dtype());

  // Ensure inputs are contiguous
  array x_contig = x.flags().row_contiguous ? x : contiguous_copy_gpu(x, s);
  array w_contig = w_q.flags().row_contiguous ? w_q : contiguous_copy_gpu(w_q, s);
  array scales_contig = scales.flags().row_contiguous ? scales : contiguous_copy_gpu(scales, s);

  out.set_data(opencl::allocator().malloc(out.nbytes()));

  // Extract dimensions
  int M = x.shape(-2);
  int K = x.shape(-1);
  int N = out.shape(-1);

  auto& encoder = dev.get_command_encoder(s.index);

  // MXFP4 mode: uses fp4_e2m1 values and fp8_e8m0 scales, no biases
  if (mode == QuantizationMode::Mxfp4) {
    bool is_f16 = (x.dtype() == float16);
    bool is_bf16 = (x.dtype() == bfloat16);

    std::string kernel_src;
    std::string kernel_name;

    // Use optimized qmv kernel for M=1 (token generation)
    if (M == 1) {
      // bf16 kernel uses optimized parameters:
      // - 2 subgroups of 64 threads = 128 threads per workgroup
      // - 2 rows per subgroup = 4 outputs per workgroup
      // f16/f32 kernels use original parameters:
      // - 64 threads, 8 outputs per workgroup
      int results_per_wg;
      int workgroup_size;

      if (is_bf16) {
        results_per_wg = 4;    // N_SUBGROUPS * N_ROWS_PER_SG
        workgroup_size = 128;  // N_SUBGROUPS * SUBGROUP_SIZE
        kernel_src = generate_mxfp4_qmv_kernel_bf16();
        kernel_name = "mxfp4_qmv_bf16";
      } else if (is_f16) {
        results_per_wg = 8;
        workgroup_size = 64;
        kernel_src = generate_mxfp4_qmv_kernel_f16();
        kernel_name = "mxfp4_qmv_f16";
      } else {
        results_per_wg = 8;
        workgroup_size = 64;
        kernel_src = generate_mxfp4_qmv_kernel_f32();
        kernel_name = "mxfp4_qmv_f32";
      }
      cl_kernel kernel = dev.get_kernel(kernel_name, kernel_src, "-cl-std=CL2.0");

      encoder.set_kernel(kernel);
      encoder.set_input_array(x_contig, 0);
      encoder.set_input_array(w_contig, 1);
      encoder.set_input_array(scales_contig, 2);
      encoder.set_output_array(out, 3);
      encoder.set_bytes(N, 4);
      encoder.set_bytes(K, 5);

      int num_workgroups = (N + results_per_wg - 1) / results_per_wg;
      size_t local_size[3] = {static_cast<size_t>(workgroup_size), 0, 0};
      size_t global_size[3] = {static_cast<size_t>(num_workgroups * workgroup_size), 0, 0};
      encoder.dispatch_threads(global_size, local_size, 1);
    } else {
      // General matmul for M > 1
      if (is_f16) {
        kernel_src = generate_mxfp4_matmul_kernel_f16();
        kernel_name = "mxfp4_qmatmul_f16";
      } else if (is_bf16) {
        kernel_src = generate_mxfp4_matmul_kernel_bf16();
        kernel_name = "mxfp4_qmatmul_bf16";
      } else {
        kernel_src = generate_mxfp4_matmul_kernel_f32();
        kernel_name = "mxfp4_qmatmul_f32";
      }
      cl_kernel kernel = dev.get_kernel(kernel_name, kernel_src);

      encoder.set_kernel(kernel);
      encoder.set_input_array(x_contig, 0);
      encoder.set_input_array(w_contig, 1);
      encoder.set_input_array(scales_contig, 2);
      encoder.set_output_array(out, 3);
      encoder.set_bytes(M, 4);
      encoder.set_bytes(N, 5);
      encoder.set_bytes(K, 6);

      size_t local_size[3] = {16, 16, 0};
      size_t global_size[3] = {
        static_cast<size_t>((M + 15) / 16 * 16),
        static_cast<size_t>((N + 15) / 16 * 16),
        0
      };
      encoder.dispatch_threads(global_size, local_size, 2);
    }

    dev.add_temporary(x_contig, s.index);
    dev.add_temporary(w_contig, s.index);
    dev.add_temporary(scales_contig, s.index);
    dev.end_encoding(s.index);
    return;
  }

  // Affine mode: uses biases
  array biases_contig = biases.has_value() ?
      (biases->flags().row_contiguous ? *biases : contiguous_copy_gpu(*biases, s)) :
      array({}, float32, nullptr, {});

  int pack_factor = 32 / bits;

  // Select kernel based on input dtype and dimensions
  bool is_f16 = (x.dtype() == float16);
  std::string kernel_src;
  std::string kernel_name;

  // Use optimized qmv kernel for M=1 (token generation) with float16
  if (M == 1 && is_f16 && (K % pack_factor == 0)) {
    // Use multi-output kernel: 8 outputs per workgroup with subgroup reduction
    // Broadcast kernel was tested but creates non-coalesced weight access
    constexpr int RESULTS_PER_WG = 8;
    constexpr int WORKGROUP_SIZE = 64;

    kernel_src = generate_qmv_multi_kernel_f16(bits, group_size);
    kernel_name = "qmv_multi_f16_b" + std::to_string(bits) + "_g" + std::to_string(group_size);
    cl_kernel kernel = dev.get_kernel(kernel_name, kernel_src, "-cl-std=CL2.0");

    encoder.set_kernel(kernel);
    encoder.set_input_array(x_contig, 0);
    encoder.set_input_array(w_contig, 1);
    encoder.set_input_array(scales_contig, 2);
    encoder.set_input_array(biases_contig, 3);
    encoder.set_output_array(out, 4);
    encoder.set_bytes(N, 5);
    encoder.set_bytes(K, 6);
    encoder.set_bytes(group_size, 7);

    int num_workgroups = (N + RESULTS_PER_WG - 1) / RESULTS_PER_WG;
    size_t local_size[3] = {WORKGROUP_SIZE, 0, 0};
    size_t global_size[3] = {static_cast<size_t>(num_workgroups * WORKGROUP_SIZE), 0, 0};
    encoder.dispatch_threads(global_size, local_size, 1);
  } else {
    // Use standard matmul kernel
    bool is_bf16 = (x.dtype() == bfloat16);
    if (is_f16) {
      kernel_src = generate_quantized_matmul_kernel_f16(bits, group_size);
      kernel_name = "affine_qmatmul_f16";
    } else if (is_bf16) {
      kernel_src = generate_quantized_matmul_kernel_bf16(bits, group_size);
      kernel_name = "affine_qmatmul_bf16";
    } else {
      kernel_src = generate_quantized_matmul_kernel_f32(bits, group_size);
      kernel_name = "affine_qmatmul_f32";
    }
    cl_kernel kernel = dev.get_kernel(kernel_name, kernel_src);

    encoder.set_kernel(kernel);
    encoder.set_input_array(x_contig, 0);
    encoder.set_input_array(w_contig, 1);
    encoder.set_input_array(scales_contig, 2);
    encoder.set_input_array(biases_contig, 3);
    encoder.set_output_array(out, 4);
    encoder.set_bytes(M, 5);
    encoder.set_bytes(N, 6);
    encoder.set_bytes(K, 7);
    encoder.set_bytes(group_size, 8);
    encoder.set_bytes(bits, 9);
    encoder.set_bytes(pack_factor, 10);

    size_t local_size[3] = {16, 16, 0};
    size_t global_size[3] = {
      static_cast<size_t>((M + 15) / 16 * 16),
      static_cast<size_t>((N + 15) / 16 * 16),
      0
    };
    encoder.dispatch_threads(global_size, local_size, 2);
  }

  dev.add_temporary(x_contig, s.index);
  dev.add_temporary(w_contig, s.index);
  dev.add_temporary(scales_contig, s.index);
  if (biases.has_value()) {
    dev.add_temporary(biases_contig, s.index);
  }
  dev.end_encoding(s.index);
}

void QQMatmul::eval_gpu(const std::vector<array>& inputs, array& out) {
  throw std::runtime_error(
      "[QQMatmul::eval_gpu] OpenCL implementation not yet available");
}

// Generate GatherQMM kernel for MXFP4 mode (M=1 case - vector-matrix)
// Optimized version with parallel K reduction using subgroups
std::string generate_gather_qmv_mxfp4_kernel_bf16() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  oss << R"(
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#endif

// Full lookup table for fp4_e2m1 (all 16 values including negative)
__constant float fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

float fp8_e8m0_to_float(uchar bits) {
    uint f_bits = (bits == 0) ? 0x00400000 : ((uint)bits << 23);
    return as_float(f_bits);
}

#define GROUP_SIZE 32
#define PACK_FACTOR 8
#define SUBGROUP_SIZE 64
#define OUTPUTS_PER_WG 4

// GatherQMM kernel for MXFP4 with M=1 (vector-matrix)
// Optimized with parallel K reduction using subgroups
// Each workgroup of 64 threads computes OUTPUTS_PER_WG output elements
// The K dimension is parallelized across threads in the subgroup
#ifdef ADRENO_GPU
__attribute__((qcom_reqd_sub_group_size("half")))
#endif
__kernel void gather_qmv_mxfp4_bf16(
    __global const bfloat16_t* x,      // Input [lhs_B, K]
    __global const uint* w_q,          // Packed weights [E, N, K/8]
    __global const uchar* scales,      // Scales [E, N, K/32]
    __global const uint* lhs_indices,  // LHS gather indices [B]
    __global const uint* rhs_indices,  // RHS gather indices [B]
    __global bfloat16_t* out,          // Output [B, N]
    int B,                             // Batch size
    int N,                             // Output dimension
    int K,                             // Reduction dimension
    int x_batch_stride,                // Stride between batches in x
    int w_n_stride,                    // Stride between rows in w (K/8)
    int scales_n_stride                // Stride between rows in scales (K/32)
) {
    // Cache LUT in local memory
    __local float shmem_lut[16];
    int lid = get_local_id(0);
    if (lid < 16) {
        shmem_lut[lid] = fp4_lut[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Each workgroup handles OUTPUTS_PER_WG outputs
    int wg_id = get_group_id(0);
    int base_output = wg_id * OUTPUTS_PER_WG;

    int K_packed = K / PACK_FACTOR;
    int num_groups_k = K / GROUP_SIZE;

    // Initialize accumulators for each output this workgroup handles
    float sums[OUTPUTS_PER_WG] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Each thread processes K_packed / SUBGROUP_SIZE packed values
    // Thread lid processes pack_idx = lid, lid + SUBGROUP_SIZE, lid + 2*SUBGROUP_SIZE, ...
    for (int pack_offset = lid; pack_offset < K_packed; pack_offset += SUBGROUP_SIZE) {
        int k_base = pack_offset * PACK_FACTOR;
        int group_idx = k_base / GROUP_SIZE;

        // Load 8 input values for this packed position
        float x_vals[PACK_FACTOR];
        #pragma unroll
        for (int i = 0; i < PACK_FACTOR; i++) {
            int k = k_base + i;
            // We'll load x_ptr later per output
            x_vals[i] = (k < K) ? 1.0f : 0.0f;  // Placeholder
        }

        // Process each output this workgroup handles
        #pragma unroll
        for (int out_idx = 0; out_idx < OUTPUTS_PER_WG; out_idx++) {
            int output_id = base_output + out_idx;
            if (output_id >= B * N) continue;

            int b = output_id / N;
            int n = output_id % N;

            uint lhs_idx = lhs_indices[b];
            uint rhs_idx = rhs_indices[b];

            __global const bfloat16_t* x_ptr = x + lhs_idx * x_batch_stride;
            __global const uint* w_ptr = w_q + rhs_idx * N * w_n_stride + n * w_n_stride;
            __global const uchar* s_ptr = scales + rhs_idx * N * scales_n_stride + n * scales_n_stride;

            // Load packed weights
            uint packed = w_ptr[pack_offset];
            float scale = fp8_e8m0_to_float(s_ptr[group_idx]);

            // Accumulate
            #pragma unroll
            for (int i = 0; i < PACK_FACTOR; i++) {
                int k = k_base + i;
                if (k >= K) break;

                float x_val = bfloat16_to_float(x_ptr[k]);
                uint q = (packed >> (i * 4)) & 0xF;
                sums[out_idx] += x_val * (scale * shmem_lut[q]);
            }
        }
    }

    // Subgroup reduction for each output
    #pragma unroll
    for (int out_idx = 0; out_idx < OUTPUTS_PER_WG; out_idx++) {
        float sum_all = sub_group_reduce_add(sums[out_idx]);
        if (get_sub_group_local_id() == 0) {
            int output_id = base_output + out_idx;
            if (output_id < B * N) {
                int b = output_id / N;
                int n = output_id % N;
                out[b * N + n] = float_to_bfloat16(sum_all);
            }
        }
    }
}
)";

  return oss.str();
}

std::string generate_gather_qmv_mxfp4_kernel_f16() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  oss << R"(
// Convert fp4_e2m1 to float
// Lookup table for fp4_e2m1 to float conversion
// Avoids denormalized half-precision which some GPUs flush to zero
__constant float fp4_lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float fp4_to_float(uint bits) {
    float val = fp4_lut_vals[bits & 7];
    return (bits & 8) ? -val : val;
}

// Convert fp8_e8m0 scale to float
float fp8_e8m0_to_float(uchar bits) {
    // Special case: bits=0 represents 2^-127 (denormalized float 0x00400000)
    uint f_bits = (bits == 0) ? 0x00400000 : ((uint)bits << 23);
    return as_float(f_bits);
}

#define GROUP_SIZE 32
#define PACK_FACTOR 8

// GatherQMM kernel for MXFP4 with M=1 (vector-matrix)
// Weights are stored as uint32 with 8 fp4 values each
__kernel void gather_qmv_mxfp4_f16(
    __global const half* x,
    __global const uint* w_q,          // Packed weights [E, N, K/8] (8 fp4 per uint32)
    __global const uchar* scales,      // Scales [E, N, K/32]
    __global const uint* lhs_indices,
    __global const uint* rhs_indices,
    __global half* out,
    int B, int N, int K,
    int x_batch_stride, int w_n_stride, int scales_n_stride
) {
    int gid = get_global_id(0);
    int total_outputs = B * N;
    if (gid >= total_outputs) return;

    int b = gid / N;
    int n = gid % N;

    uint lhs_idx = lhs_indices[b];
    uint rhs_idx = rhs_indices[b];

    // Layout is [E, N, K_packed] where K_packed = K/8
    __global const half* x_ptr = x + lhs_idx * x_batch_stride;
    __global const uint* w_ptr = w_q + rhs_idx * N * w_n_stride + n * w_n_stride;
    __global const uchar* s_ptr = scales + rhs_idx * N * scales_n_stride + n * scales_n_stride;

    int K_packed = K / PACK_FACTOR;

    float sum = 0.0f;
    for (int pack_idx = 0; pack_idx < K_packed; pack_idx++) {
        // Load packed uint32 with 8 fp4 values
        uint packed = w_ptr[pack_idx];

        // Base k index for this packed value
        int k_base = pack_idx * PACK_FACTOR;

        // Each group of 32 values shares one scale
        int group_idx = k_base / GROUP_SIZE;
        float scale = fp8_e8m0_to_float(s_ptr[group_idx]);

        // Unroll 8 fp4 values from the uint32
        #pragma unroll
        for (int i = 0; i < PACK_FACTOR; i++) {
            int k = k_base + i;
            if (k >= K) break;

            // Check if we crossed a group boundary
            int current_group = k / GROUP_SIZE;
            if (current_group != group_idx) {
                group_idx = current_group;
                scale = fp8_e8m0_to_float(s_ptr[group_idx]);
            }

            float x_val = convert_float(x_ptr[k]);
            uint q = (packed >> (i * 4)) & 0xF;
            sum += x_val * (scale * fp4_to_float(q));
        }
    }

    out[b * N + n] = convert_half(sum);
}
)";

  return oss.str();
}

// Transpose version: weights stored as [E, K/8, N] instead of [E, N, K/8]
std::string generate_gather_qmv_t_mxfp4_kernel_bf16() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  oss << R"(
// Convert fp4_e2m1 to float
// Lookup table for fp4_e2m1 to float conversion
// Avoids denormalized half-precision which some GPUs flush to zero
__constant float fp4_lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float fp4_to_float(uint bits) {
    float val = fp4_lut_vals[bits & 7];
    return (bits & 8) ? -val : val;
}

// Convert fp8_e8m0 scale to float
float fp8_e8m0_to_float(uchar bits) {
    // Special case: bits=0 represents 2^-127 (denormalized float 0x00400000)
    uint f_bits = (bits == 0) ? 0x00400000 : ((uint)bits << 23);
    return as_float(f_bits);
}

#define GROUP_SIZE 32
#define PACK_FACTOR 8

// GatherQMM kernel for MXFP4 with transpose (weights are [E, K/8, N])
// Weights are stored as uint32 with 8 fp4 values each
__kernel void gather_qmv_t_mxfp4_bf16(
    __global const bfloat16_t* x,      // Input [B, K]
    __global const uint* w_q,          // Packed weights [E, K/8, N] (transposed)
    __global const uchar* scales,      // Scales [E, K/32, N]
    __global const uint* lhs_indices,
    __global const uint* rhs_indices,
    __global bfloat16_t* out,
    int B, int N, int K,
    int x_batch_stride,
    int w_expert_stride,               // K/8 * N
    int scales_expert_stride           // K/32 * N
) {
    int gid = get_global_id(0);
    int total_outputs = B * N;
    if (gid >= total_outputs) return;

    int b = gid / N;
    int n = gid % N;

    uint lhs_idx = lhs_indices[b];
    uint rhs_idx = rhs_indices[b];

    __global const bfloat16_t* x_ptr = x + lhs_idx * x_batch_stride;
    __global const uint* w_base = w_q + rhs_idx * w_expert_stride;
    __global const uchar* s_base = scales + rhs_idx * scales_expert_stride;

    int K_packed = K / PACK_FACTOR;

    float sum = 0.0f;
    for (int pack_idx = 0; pack_idx < K_packed; pack_idx++) {
        // For transposed weights: w[pack_idx, n] = w[pack_idx * N + n]
        uint packed = w_base[pack_idx * N + n];

        int k_base = pack_idx * PACK_FACTOR;
        int group_idx = k_base / GROUP_SIZE;
        // For transposed scales: s[group_idx, n] = s[group_idx * N + n]
        float scale = fp8_e8m0_to_float(s_base[group_idx * N + n]);

        // Unroll 8 fp4 values from the uint32
        #pragma unroll
        for (int i = 0; i < PACK_FACTOR; i++) {
            int k = k_base + i;
            if (k >= K) break;

            // Check if we crossed a group boundary
            int current_group = k / GROUP_SIZE;
            if (current_group != group_idx) {
                group_idx = current_group;
                scale = fp8_e8m0_to_float(s_base[group_idx * N + n]);
            }

            float x_val = bfloat16_to_float(x_ptr[k]);
            uint q = (packed >> (i * 4)) & 0xF;
            sum += x_val * (scale * fp4_to_float(q));
        }
    }

    out[b * N + n] = float_to_bfloat16(sum);
}
)";

  return oss.str();
}

std::string generate_gather_qmv_t_mxfp4_kernel_f16() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  oss << R"(
// Convert fp4_e2m1 to float
// Lookup table for fp4_e2m1 to float conversion
// Avoids denormalized half-precision which some GPUs flush to zero
__constant float fp4_lut_vals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

float fp4_to_float(uint bits) {
    float val = fp4_lut_vals[bits & 7];
    return (bits & 8) ? -val : val;
}

// Convert fp8_e8m0 scale to float
float fp8_e8m0_to_float(uchar bits) {
    // Special case: bits=0 represents 2^-127 (denormalized float 0x00400000)
    uint f_bits = (bits == 0) ? 0x00400000 : ((uint)bits << 23);
    return as_float(f_bits);
}

#define GROUP_SIZE 32
#define PACK_FACTOR 8

// GatherQMM kernel for MXFP4 with transpose (weights are [E, K/8, N])
// Weights are stored as uint32 with 8 fp4 values each
__kernel void gather_qmv_t_mxfp4_f16(
    __global const half* x,
    __global const uint* w_q,          // Packed weights [E, K/8, N] (transposed)
    __global const uchar* scales,      // Scales [E, K/32, N]
    __global const uint* lhs_indices,
    __global const uint* rhs_indices,
    __global half* out,
    int B, int N, int K,
    int x_batch_stride, int w_expert_stride, int scales_expert_stride
) {
    int gid = get_global_id(0);
    int total_outputs = B * N;
    if (gid >= total_outputs) return;

    int b = gid / N;
    int n = gid % N;

    uint lhs_idx = lhs_indices[b];
    uint rhs_idx = rhs_indices[b];

    __global const half* x_ptr = x + lhs_idx * x_batch_stride;
    __global const uint* w_base = w_q + rhs_idx * w_expert_stride;
    __global const uchar* s_base = scales + rhs_idx * scales_expert_stride;

    int K_packed = K / PACK_FACTOR;

    float sum = 0.0f;
    for (int pack_idx = 0; pack_idx < K_packed; pack_idx++) {
        // For transposed weights: w[pack_idx, n] = w[pack_idx * N + n]
        uint packed = w_base[pack_idx * N + n];

        int k_base = pack_idx * PACK_FACTOR;
        int group_idx = k_base / GROUP_SIZE;
        // For transposed scales: s[group_idx, n] = s[group_idx * N + n]
        float scale = fp8_e8m0_to_float(s_base[group_idx * N + n]);

        // Unroll 8 fp4 values from the uint32
        #pragma unroll
        for (int i = 0; i < PACK_FACTOR; i++) {
            int k = k_base + i;
            if (k >= K) break;

            // Check if we crossed a group boundary
            int current_group = k / GROUP_SIZE;
            if (current_group != group_idx) {
                group_idx = current_group;
                scale = fp8_e8m0_to_float(s_base[group_idx * N + n]);
            }

            float x_val = convert_float(x_ptr[k]);
            uint q = (packed >> (i * 4)) & 0xF;
            sum += x_val * (scale * fp4_to_float(q));
        }
    }

    out[b * N + n] = convert_half(sum);
}
)";

  return oss.str();
}

// Generate GatherQMM kernel for Affine mode with transpose (M=1 case)
// Weights are [E, K_packed, N] where K_packed = K * bits / 32
std::string generate_gather_qmv_affine_t_kernel_bf16(int bits, int group_size) {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  int pack_factor = 32 / bits;  // Values per uint32 (4-bit: 8)
  int bitmask = (1 << bits) - 1;

  oss << R"(
#define BITS )" << bits << R"(
#define GROUP_SIZE )" << group_size << R"(
#define PACK_FACTOR )" << pack_factor << R"(
#define BITMASK )" << bitmask << R"(

// GatherQMM kernel for Affine mode with transpose
// Weights are [E, K_packed, N] where K_packed = K / PACK_FACTOR
__kernel void gather_qmv_affine_t_bf16(
    __global const bfloat16_t* x,        // Input [B, K]
    __global const uint* w_q,            // Packed weights [E, K_packed, N]
    __global const bfloat16_t* scales,   // Scales [E, num_groups, N]
    __global const bfloat16_t* biases,   // Biases [E, num_groups, N]
    __global const uint* lhs_indices,
    __global const uint* rhs_indices,
    __global bfloat16_t* out,
    int B, int N, int K,
    int x_batch_stride,
    int w_expert_stride,                 // K_packed * N
    int scales_expert_stride             // num_groups * N
) {
    int gid = get_global_id(0);
    int total_outputs = B * N;
    if (gid >= total_outputs) return;

    int b = gid / N;
    int n = gid % N;

    uint lhs_idx = lhs_indices[b];
    uint rhs_idx = rhs_indices[b];

    __global const bfloat16_t* x_ptr = x + lhs_idx * x_batch_stride;
    __global const uint* w_base = w_q + rhs_idx * w_expert_stride;
    __global const bfloat16_t* s_base = scales + rhs_idx * scales_expert_stride;
    __global const bfloat16_t* bias_base = biases + rhs_idx * scales_expert_stride;

    int K_packed = K / PACK_FACTOR;
    int num_groups = (K + GROUP_SIZE - 1) / GROUP_SIZE;

    float sum = 0.0f;

    for (int pack_idx = 0; pack_idx < K_packed; pack_idx++) {
        // For transposed weights: w[pack_idx, n]
        uint packed = w_base[pack_idx * N + n];

        // Process PACK_FACTOR values from this uint32
        for (int i = 0; i < PACK_FACTOR; i++) {
            int k = pack_idx * PACK_FACTOR + i;
            if (k >= K) break;

            int group_idx = k / GROUP_SIZE;

            // Extract quantized value
            uint q = (packed >> (i * BITS)) & BITMASK;

            // Get scale and bias for this group (transposed: [group_idx, n])
            float scale = bfloat16_to_float(s_base[group_idx * N + n]);
            float bias = bfloat16_to_float(bias_base[group_idx * N + n]);

            // Dequantize and accumulate
            float x_val = bfloat16_to_float(x_ptr[k]);
            float w_val = scale * (float)q + bias;
            sum += x_val * w_val;
        }
    }

    out[b * N + n] = float_to_bfloat16(sum);
}
)";

  return oss.str();
}

void GatherQMM::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& dev = opencl::device(s.device);

  // Parse inputs
  auto& x = inputs[0];
  auto& w = inputs[1];
  auto& scales = inputs[2];
  const array& lhs_indices = inputs[inputs.size() - 2];
  const array& rhs_indices = inputs[inputs.size() - 1];

  // Get dimensions
  int K = x.shape(-1);
  int M = x.shape(-2);
  int N = out.shape(-1);
  int B = out.size() / M / N;

  OPENCL_DEBUG_LOG("[GatherQMM::eval_gpu] ENTRY"
            << " mode=" << (mode_ == QuantizationMode::Mxfp4 ? "mxfp4" : "affine")
            << " bits=" << bits_ << " group_size=" << group_size_ << " transpose=" << transpose_);
  OPENCL_DEBUG_LOG("[GatherQMM::eval_gpu] Dimensions: B=" << B << " M=" << M << " K=" << K << " N=" << N);
  OPENCL_DEBUG_LOG("[GatherQMM::eval_gpu] x: shape=(" << x.shape(0)
            << (x.ndim() > 1 ? ", " + std::to_string(x.shape(1)) : "")
            << (x.ndim() > 2 ? ", " + std::to_string(x.shape(2)) : "") << ")"
            << " dtype=" << x.dtype());
  OPENCL_DEBUG_LOG("[GatherQMM::eval_gpu] w: shape=(" << w.shape(0)
            << (w.ndim() > 1 ? ", " + std::to_string(w.shape(1)) : "")
            << (w.ndim() > 2 ? ", " + std::to_string(w.shape(2)) : "") << ")"
            << " dtype=" << w.dtype());
  OPENCL_DEBUG_LOG("[GatherQMM::eval_gpu] scales: shape=(" << scales.shape(0)
            << (scales.ndim() > 1 ? ", " + std::to_string(scales.shape(1)) : "")
            << (scales.ndim() > 2 ? ", " + std::to_string(scales.shape(2)) : "") << ")"
            << " dtype=" << scales.dtype());
  OPENCL_DEBUG_LOG("[GatherQMM::eval_gpu] lhs_indices: size=" << lhs_indices.size()
            << " rhs_indices: size=" << rhs_indices.size());
  OPENCL_DEBUG_LOG("[GatherQMM::eval_gpu] out: shape=(" << out.shape(0)
            << (out.ndim() > 1 ? ", " + std::to_string(out.shape(1)) : "")
            << (out.ndim() > 2 ? ", " + std::to_string(out.shape(2)) : "") << ")"
            << " dtype=" << out.dtype());

  out.set_data(opencl::allocator().malloc(out.nbytes()));

  if (M != 1) {
    throw std::runtime_error(
        "[GatherQMM::eval_gpu] Only M=1 (vector-matrix) currently supported");
  }

  // Select kernel based on dtype and transpose
  bool is_f16 = (x.dtype() == float16);
  bool is_bf16 = (x.dtype() == bfloat16);

  if (!is_bf16 && !is_f16) {
    throw std::runtime_error(
        "[GatherQMM::eval_gpu] Only float16 and bfloat16 currently supported");
  }

  // Helper to check if array needs copy (not contiguous OR has non-zero offset).
  // Sliced arrays have non-zero offset which the kernel doesn't account for -
  // it only receives buffer().ptr() but not the data offset.
  auto needs_copy = [](const array& a) {
    return !a.flags().row_contiguous || a.offset() != 0;
  };

  // Ensure contiguity AND zero offset
  array x_contig = needs_copy(x) ? contiguous_copy_gpu(x, s) : x;
  array w_contig = needs_copy(w) ? contiguous_copy_gpu(w, s) : w;
  array scales_contig = needs_copy(scales) ? contiguous_copy_gpu(scales, s) : scales;

  // Indices may be broadcast (e.g., lhs_indices broadcast from [1] to [B])
  // The kernel expects exactly B elements, so we must expand broadcast indices.
  // Note: Just making them contiguous is not enough - if shape is [1] with data [0],
  // contiguous copy still yields [1] element, but kernel reads indices[0..B-1].
  Shape indices_shape = {B};
  array lhs_indices_expanded = (lhs_indices.size() != B)
      ? broadcast_to(lhs_indices, indices_shape, s) : lhs_indices;
  array rhs_indices_expanded = (rhs_indices.size() != B)
      ? broadcast_to(rhs_indices, indices_shape, s) : rhs_indices;

  // After broadcast_to, the array has shape [B] but may have stride 0.
  // Make contiguous to materialize the broadcast into B actual elements.
  // Also check offset - sliced index arrays from mlx_topk have non-zero offset.
  array lhs_indices_contig = needs_copy(lhs_indices_expanded)
      ? contiguous_copy_gpu(lhs_indices_expanded, s) : lhs_indices_expanded;
  array rhs_indices_contig = needs_copy(rhs_indices_expanded)
      ? contiguous_copy_gpu(rhs_indices_expanded, s) : rhs_indices_expanded;

  OPENCL_DEBUG_LOG("[GatherQMM::eval_gpu] Indices: lhs_orig_size=" << lhs_indices.size()
            << " rhs_orig_size=" << rhs_indices.size() << " B=" << B
            << " lhs_contig_size=" << lhs_indices_contig.size()
            << " rhs_contig_size=" << rhs_indices_contig.size());

  auto& encoder = dev.get_command_encoder(s.index);

  std::string kernel_src;
  std::string kernel_name;

  if (mode_ == QuantizationMode::Affine) {
    // Affine mode: has biases, scales/biases are same dtype as x
    if (!transpose_) {
      throw std::runtime_error(
          "[GatherQMM::eval_gpu] Affine mode currently only supports transposed weights");
    }

    // Get biases (input[3] for affine mode)
    auto& biases = inputs[3];
    array biases_contig = needs_copy(biases) ? contiguous_copy_gpu(biases, s) : biases;

    int pack_factor = 32 / bits_;
    int K_packed = K / pack_factor;
    int num_groups = (K + group_size_ - 1) / group_size_;

    if (is_bf16) {
      kernel_src = generate_gather_qmv_affine_t_kernel_bf16(bits_, group_size_);
      kernel_name = "gather_qmv_affine_t_bf16";
    } else {
      throw std::runtime_error(
          "[GatherQMM::eval_gpu] Affine mode currently only supports bfloat16");
    }

    cl_kernel kernel = dev.get_kernel(kernel_name, kernel_src);

    int x_batch_stride = K;
    int w_expert_stride = K_packed * N;
    int scales_expert_stride = num_groups * N;

    encoder.set_kernel(kernel);
    encoder.set_input_array(x_contig, 0);
    encoder.set_input_array(w_contig, 1);
    encoder.set_input_array(scales_contig, 2);
    encoder.set_input_array(biases_contig, 3);
    encoder.set_input_array(lhs_indices_contig, 4);
    encoder.set_input_array(rhs_indices_contig, 5);
    encoder.set_output_array(out, 6);
    encoder.set_bytes(B, 7);
    encoder.set_bytes(N, 8);
    encoder.set_bytes(K, 9);
    encoder.set_bytes(x_batch_stride, 10);
    encoder.set_bytes(w_expert_stride, 11);
    encoder.set_bytes(scales_expert_stride, 12);

    dev.add_temporary(biases_contig, s.index);
  } else if (mode_ == QuantizationMode::Mxfp4) {
    // MXFP4 mode: no biases, scales are uint8 (fp8_e8m0)
    //
    // Weight layout notes (matching Metal's implementation):
    // - transpose=True (qmv): weights are [E, N, K/8], doing y = x @ W^T
    // - transpose=False (qvm): weights are [E, K/8, N], doing y = x @ W
    //
    if (transpose_) {
      // transpose=True (qmv): weights are [E, N, K/8] (8 fp4 values per uint32)
      // Each output element y[n] = sum_k(x[k] * w[n, k])
      if (is_bf16) {
        kernel_src = generate_gather_qmv_mxfp4_kernel_bf16();
        kernel_name = "gather_qmv_mxfp4_bf16";
      } else if (is_f16) {
        kernel_src = generate_gather_qmv_mxfp4_kernel_f16();
        kernel_name = "gather_qmv_mxfp4_f16";
      }

      cl_kernel kernel = dev.get_kernel(kernel_name, kernel_src);

      // For transpose=True: w is [E, N, K/8], scales is [E, N, K/32]
      int x_batch_stride = K;
      int w_n_stride = K / 8;  // 8 fp4 values per uint32
      int scales_n_stride = K / 32;

      encoder.set_kernel(kernel);
      encoder.set_input_array(x_contig, 0);
      encoder.set_input_array(w_contig, 1);
      encoder.set_input_array(scales_contig, 2);
      encoder.set_input_array(lhs_indices_contig, 3);
      encoder.set_input_array(rhs_indices_contig, 4);
      encoder.set_output_array(out, 5);
      encoder.set_bytes(B, 6);
      encoder.set_bytes(N, 7);
      encoder.set_bytes(K, 8);
      encoder.set_bytes(x_batch_stride, 9);
      encoder.set_bytes(w_n_stride, 10);
      encoder.set_bytes(scales_n_stride, 11);
    } else {
      // transpose=False (qvm): weights are [E, K/8, N] (8 fp4 values per uint32)
      // Each output element y[n] = sum_k(x[k] * w[k, n])
      if (is_bf16) {
        kernel_src = generate_gather_qmv_t_mxfp4_kernel_bf16();
        kernel_name = "gather_qmv_t_mxfp4_bf16";
      } else if (is_f16) {
        kernel_src = generate_gather_qmv_t_mxfp4_kernel_f16();
        kernel_name = "gather_qmv_t_mxfp4_f16";
      }

      cl_kernel kernel = dev.get_kernel(kernel_name, kernel_src);

      // For transpose=False: w is [E, K/8, N], scales is [E, K/32, N]
      int x_batch_stride = K;
      int w_expert_stride = (K / 8) * N;  // 8 fp4 values per uint32
      int scales_expert_stride = (K / 32) * N;

      encoder.set_kernel(kernel);
      encoder.set_input_array(x_contig, 0);
      encoder.set_input_array(w_contig, 1);
      encoder.set_input_array(scales_contig, 2);
      encoder.set_input_array(lhs_indices_contig, 3);
      encoder.set_input_array(rhs_indices_contig, 4);
      encoder.set_output_array(out, 5);
      encoder.set_bytes(B, 6);
      encoder.set_bytes(N, 7);
      encoder.set_bytes(K, 8);
      encoder.set_bytes(x_batch_stride, 9);
      encoder.set_bytes(w_expert_stride, 10);
      encoder.set_bytes(scales_expert_stride, 11);
    }
  } else {
    throw std::runtime_error(
        "[GatherQMM::eval_gpu] Only Affine and Mxfp4 modes currently supported on OpenCL");
  }

  int total_outputs = B * N;
  size_t global_size[3];
  size_t local_size[3];

  // Use optimized dispatch for bf16 MXFP4 transpose=True (most common case)
  // Optimized kernel uses 64 threads per workgroup, 4 outputs per workgroup
  bool use_optimized_dispatch = (mode_ == QuantizationMode::Mxfp4 && transpose_ && is_bf16);
  if (use_optimized_dispatch) {
    constexpr int OUTPUTS_PER_WG = 4;
    constexpr int SUBGROUP_SIZE = 64;
    int num_wgs = (total_outputs + OUTPUTS_PER_WG - 1) / OUTPUTS_PER_WG;
    local_size[0] = SUBGROUP_SIZE;
    local_size[1] = 0;
    local_size[2] = 0;
    global_size[0] = static_cast<size_t>(num_wgs * SUBGROUP_SIZE);
    global_size[1] = 0;
    global_size[2] = 0;
  } else {
    global_size[0] = static_cast<size_t>((total_outputs + 255) / 256 * 256);
    global_size[1] = 0;
    global_size[2] = 0;
    local_size[0] = 256;
    local_size[1] = 0;
    local_size[2] = 0;
  }
  encoder.dispatch_threads(global_size, local_size, 1);

  dev.add_temporary(x_contig, s.index);
  dev.add_temporary(w_contig, s.index);
  dev.add_temporary(scales_contig, s.index);
  dev.add_temporary(lhs_indices_contig, s.index);
  dev.add_temporary(rhs_indices_contig, s.index);
  dev.end_encoding(s.index);
}

namespace {

std::string generate_to_fp8_kernel(const std::string& in_type) {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  // FP8 E4M3 conversion algorithm from PyTorch
  // https://github.com/pytorch/pytorch/blob/e3643e1e0e923f0fc063dfab6f45c956d568919d/c10/util/Float8_e4m3fn.h
  oss << R"(
// Convert float to FP8 E4M3 format
uchar float_to_fp8(float f) {
    uint fp8_max = 543u << 21;     // Max FP8 value threshold
    uint denorm_mask = 141u << 23; // Denormal handling mask

    uint f_bits = as_uint(f);
    uint sign = f_bits & 0x80000000u;
    f_bits ^= sign;

    uchar bits;
    if (f_bits >= fp8_max) {
        // Saturate to max FP8 value
        bits = 0x7E;
    } else {
        if (f_bits < (121u << 23)) {
            // Denormal case
            f_bits = as_uint(as_float(f_bits) + as_float(denorm_mask));
            bits = (uchar)(f_bits - denorm_mask);
        } else {
            // Normal case
            uchar mant_odd = (uchar)((f_bits >> 20) & 1u);
            f_bits += ((uint)(7 - 127) << 23) + 0x7FFFFu;
            f_bits += mant_odd;
            bits = (uchar)(f_bits >> 20);
        }
    }
    bits |= (uchar)(sign >> 24);
    return bits;
}
)";

  std::string kernel_name = "to_fp8_" + in_type;
  std::string read_val;
  if (in_type == "bfloat16_t") {
    read_val = "bfloat16_to_float(in[gid])";
  } else if (in_type == "half") {
    read_val = "convert_float(in[gid])";
  } else {
    read_val = "in[gid]";
  }

  oss << "__kernel void " << kernel_name << "(\n";
  oss << "    __global const " << in_type << "* in,\n";
  oss << "    __global uchar* out,\n";
  oss << "    ulong N) {\n";
  oss << "    ulong gid = get_global_id(0);\n";
  oss << "    if (gid >= N) return;\n";
  oss << "    float val = " << read_val << ";\n";
  oss << "    out[gid] = float_to_fp8(val);\n";
  oss << "}\n";

  return oss.str();
}

std::string generate_from_fp8_kernel(const std::string& out_type) {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  // FP8 E4M3 to float conversion
  // Based on Metal's fp8.h implementation
  oss << R"(
// Convert FP8 E4M3 format to float
float fp8_to_float(uchar bits) {
    // Extract magnitude (lower 7 bits) and shift to half format
    ushort v = (ushort)(bits & 127) << 7;
    // Reinterpret as half and scale up by 256
    half converted = as_half(v);
    converted *= 256.0h;
    // Apply sign
    uchar sign = bits & 128;
    return sign ? -convert_float(converted) : convert_float(converted);
}
)";

  std::string kernel_name = "from_fp8_" + out_type;
  std::string write_val;
  if (out_type == "bfloat16_t") {
    write_val = "float_to_bfloat16(val)";
  } else if (out_type == "half") {
    write_val = "convert_half(val)";
  } else {
    write_val = "val";
  }

  oss << "__kernel void " << kernel_name << "(\n";
  oss << "    __global const uchar* in,\n";
  oss << "    __global " << out_type << "* out,\n";
  oss << "    ulong N) {\n";
  oss << "    ulong gid = get_global_id(0);\n";
  oss << "    if (gid >= N) return;\n";
  oss << "    float val = fp8_to_float(in[gid]);\n";
  oss << "    out[gid] = " << write_val << ";\n";
  oss << "}\n";

  return oss.str();
}

} // anonymous namespace

void fast::ConvertFP8::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  auto& s = stream();
  auto& dev = opencl::device(s.device);
  auto& encoder = dev.get_command_encoder(s.index);

  auto& in = inputs[0];
  auto& out = outputs[0];
  bool to_fp8 = state();

  // Ensure input is contiguous
  array in_contig = in.flags().row_contiguous ? in : contiguous_copy_gpu(in, s);

  out.set_data(opencl::allocator().malloc(out.nbytes()));

  size_t N = in.size();

  if (to_fp8) {
    // ToFP8: float/half/bfloat16 -> uint8
    std::string in_type = opencl::type_to_name(in.dtype());
    std::string kernel_name = "to_fp8_" + in_type;
    std::string kernel_src = generate_to_fp8_kernel(in_type);

    cl_kernel kernel = dev.get_kernel(kernel_name, kernel_src);
    encoder.set_kernel(kernel);
    encoder.set_input_array(in_contig, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(static_cast<uint64_t>(N), 2);
  } else {
    // FromFP8: uint8 -> float/half/bfloat16
    std::string out_type = opencl::type_to_name(out.dtype());
    std::string kernel_name = "from_fp8_" + out_type;
    std::string kernel_src = generate_from_fp8_kernel(out_type);

    cl_kernel kernel = dev.get_kernel(kernel_name, kernel_src);
    encoder.set_kernel(kernel);
    encoder.set_input_array(in_contig, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(static_cast<uint64_t>(N), 2);
  }

  size_t global_size[3] = {(N + 255) / 256 * 256, 0, 0};
  size_t local_size[3] = {256, 0, 0};
  encoder.dispatch_threads(global_size, local_size, 1);

  dev.add_temporary(in_contig, s.index);
  dev.end_encoding(s.index);
}

} // namespace mlx::core
