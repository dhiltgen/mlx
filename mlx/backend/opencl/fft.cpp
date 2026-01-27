// Copyright © 2025 MLX Contributors
// OpenCL FFT implementation
//
// IMPLEMENTATION STATUS:
// ✅ Implemented:
//   - 1D FFT/IFFT for any size (power-of-2 and non-power-of-2)
//   - Multi-dimensional FFT (fft2, fftn, ifft2, ifftn)
//   - Complex-to-complex (fft, ifft)
//   - Real-to-complex (rfft) and complex-to-real (irfft)
//   - FFT along any axis (via transpose to last axis)
//   - Exact twiddle factors for π/2 multiples (avoids floating-point error)
//
// ALGORITHMS:
//   - Power-of-2 sizes: Cooley-Tukey radix-2 decimation-in-time
//     1. Bit-reversal permutation
//     2. log2(n) butterfly stages, each with n/2 butterflies
//     3. Scale by 1/n for inverse FFT
//   - Non-power-of-2 sizes: Bluestein's algorithm (Chirp-Z transform)
//     1. Generate chirp sequence
//     2. Modulate input with chirp and zero-pad to next power-of-2
//     3. Convolve via power-of-2 FFT
//     4. Modulate result with chirp

#include <cmath>
#include <sstream>
#include <vector>

#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/copy.h"
#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/primitives.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mlx::core {

namespace {

// Check if n is a power of 2
bool is_power_of_2(int n) {
  return n > 0 && (n & (n - 1)) == 0;
}

// Find next power of 2 >= n
int next_power_of_2(int n) {
  if (n <= 1) return 1;
  int p = 1;
  while (p < n) p <<= 1;
  return p;
}

// Generate OpenCL kernel source for FFT operations
std::string generate_fft_kernels() {
  std::ostringstream oss;
  oss << opencl::get_kernel_preamble();

  // Butterfly kernel - one stage of Cooley-Tukey FFT
  // Each work-item handles one butterfly pair
  oss << R"(
// Perform one stage of Cooley-Tukey FFT
// global_work_size = batch_size * (n/2)
__kernel void fft_butterfly_stage(
    __global float2* data,
    const int n,
    const int stage,
    const int batch_size,
    const int inverse
) {
    const int gid = get_global_id(0);
    const int batch_idx = gid / (n / 2);
    const int butterfly_idx = gid % (n / 2);

    if (batch_idx >= batch_size) return;

    // Compute butterfly parameters for this stage
    int m = 1 << (stage + 1);      // butterfly size
    int half_m = m >> 1;           // half butterfly size

    int group = butterfly_idx / half_m;
    int k = butterfly_idx % half_m;

    int idx0 = batch_idx * n + group * m + k;
    int idx1 = idx0 + half_m;

    // Compute twiddle factor: W_n^(k*step) where step = n/m
    // angle = sign * 2π * twiddle_idx / n, where sign is -1 for forward, +1 for inverse
    // Use exact values for common angles (multiples of π/2) to avoid floating-point errors
    int twiddle_idx = k * (n / m);  // index into n-th roots of unity

    // Check if this is an exact multiple of π/2 (i.e., twiddle_idx is a multiple of n/4)
    float2 w;
    if (n >= 4 && (twiddle_idx * 4) % n == 0) {
        // Exact angle at π/2 boundaries
        // For forward FFT: twiddle_idx=0 -> (1,0), 1 -> (0,-1), 2 -> (-1,0), 3 -> (0,1)
        // For inverse FFT: twiddle_idx=0 -> (1,0), 1 -> (0,1), 2 -> (-1,0), 3 -> (0,-1)
        int quadrant = (twiddle_idx * 4 / n) % 4;
        switch (quadrant) {
            case 0: w = (float2)(1.0f, 0.0f); break;
            case 1: w = inverse ? (float2)(0.0f, 1.0f) : (float2)(0.0f, -1.0f); break;
            case 2: w = (float2)(-1.0f, 0.0f); break;
            case 3: w = inverse ? (float2)(0.0f, -1.0f) : (float2)(0.0f, 1.0f); break;
        }
    } else {
        // General case: compute with trig functions
        float angle = (inverse ? 1.0f : -1.0f) * 2.0f * M_PI_F * (float)twiddle_idx / (float)n;
        w = (float2)(cos(angle), sin(angle));
    }

    // Load values
    float2 a = data[idx0];
    float2 b = data[idx1];

    // Complex multiply: b * w
    float2 bw = (float2)(b.x * w.x - b.y * w.y, b.x * w.y + b.y * w.x);

    // Butterfly
    data[idx0] = a + bw;
    data[idx1] = a - bw;
}

// Bit-reversal permutation
__kernel void fft_bit_reverse(
    __global const float2* input,
    __global float2* output,
    const int n,
    const int log2n,
    const int batch_size
) {
    const int gid = get_global_id(0);
    const int batch_idx = gid / n;
    const int idx = gid % n;

    if (batch_idx >= batch_size) return;

    // Bit-reverse the index
    int rev = 0;
    int temp = idx;
    for (int j = 0; j < log2n; j++) {
        rev = (rev << 1) | (temp & 1);
        temp >>= 1;
    }

    output[batch_idx * n + rev] = input[batch_idx * n + idx];
}

// Scale kernel for inverse FFT
__kernel void fft_scale(
    __global float2* data,
    const int size,
    const float scale
) {
    const int gid = get_global_id(0);
    if (gid >= size) return;

    data[gid] = data[gid] * scale;
}

// Convert real input to complex
__kernel void real_to_complex(
    __global const float* input,
    __global float2* output,
    const int size
) {
    const int gid = get_global_id(0);
    if (gid >= size) return;

    output[gid] = (float2)(input[gid], 0.0f);
}

// Extract just the necessary output for RFFT (first n/2+1 values)
__kernel void rfft_extract(
    __global const float2* input,
    __global float2* output,
    const int n,
    const int out_n,
    const int batch_size
) {
    const int gid = get_global_id(0);
    const int batch_idx = gid / out_n;
    const int idx = gid % out_n;

    if (batch_idx >= batch_size || idx >= out_n) return;

    output[batch_idx * out_n + idx] = input[batch_idx * n + idx];
}

// Prepare input for IRFFT using Hermitian symmetry
__kernel void irfft_prepare(
    __global const float2* input,
    __global float2* output,
    const int n,
    const int in_n,
    const int batch_size
) {
    const int gid = get_global_id(0);
    const int batch_idx = gid / n;
    const int idx = gid % n;

    if (batch_idx >= batch_size) return;

    float2 val;
    if (idx < in_n) {
        val = input[batch_idx * in_n + idx];
    } else {
        // Use Hermitian symmetry: X[n-k] = conj(X[k])
        int conj_idx = n - idx;
        float2 orig = input[batch_idx * in_n + conj_idx];
        val = (float2)(orig.x, -orig.y);
    }
    output[batch_idx * n + idx] = val;
}

// Extract real part from complex output
__kernel void complex_to_real(
    __global const float2* input,
    __global float* output,
    const int size
) {
    const int gid = get_global_id(0);
    if (gid >= size) return;

    output[gid] = input[gid].x;
}

// ========== Bluestein's Algorithm Kernels ==========

// Generate chirp sequence: w[k] = exp(-i*π*k²/N) for forward, conjugate for inverse
__kernel void bluestein_chirp(
    __global float2* chirp,
    const int N,
    const int inverse
) {
    const int k = get_global_id(0);
    if (k >= N) return;

    // angle = -π * k² / N for forward FFT
    // angle = +π * k² / N for inverse FFT
    float angle = (inverse ? 1.0f : -1.0f) * M_PI_F * (float)(k * k) / (float)N;
    chirp[k] = (float2)(cos(angle), sin(angle));
}

// Modulate input with chirp: a[k] = x[k] * chirp[k], zero-pad to M
__kernel void bluestein_modulate(
    __global const float2* input,
    __global const float2* chirp,
    __global float2* output,
    const int N,
    const int M,
    const int batch_size
) {
    const int gid = get_global_id(0);
    const int batch_idx = gid / M;
    const int k = gid % M;

    if (batch_idx >= batch_size) return;

    if (k < N) {
        // Multiply input by chirp
        float2 x = input[batch_idx * N + k];
        float2 w = chirp[k];
        output[gid] = (float2)(x.x * w.x - x.y * w.y, x.x * w.y + x.y * w.x);
    } else {
        // Zero padding
        output[gid] = (float2)(0.0f, 0.0f);
    }
}

// Prepare chirp filter: c[k] = conj(chirp[k]) for k < N, c[M-k] = conj(chirp[k]) for k > 0
__kernel void bluestein_prepare_filter(
    __global const float2* chirp,
    __global float2* filter,
    const int N,
    const int M
) {
    const int k = get_global_id(0);
    if (k >= M) return;

    if (k < N) {
        // c[k] = conj(chirp[k])
        float2 w = chirp[k];
        filter[k] = (float2)(w.x, -w.y);
    } else if (k > M - N) {
        // c[M-k] = conj(chirp[k]) where k' = M - k, so chirp[M-k]
        int k_prime = M - k;
        float2 w = chirp[k_prime];
        filter[k] = (float2)(w.x, -w.y);
    } else {
        filter[k] = (float2)(0.0f, 0.0f);
    }
}

// Complex multiply: out = a * b
__kernel void bluestein_multiply(
    __global const float2* a,
    __global const float2* b,
    __global float2* out,
    const int size
) {
    const int gid = get_global_id(0);
    if (gid >= size) return;

    float2 va = a[gid];
    float2 vb = b[gid];
    out[gid] = (float2)(va.x * vb.x - va.y * vb.y, va.x * vb.y + va.y * vb.x);
}

// Extract and finalize Bluestein result: y[k] = chirp[k] * conv[k] / M (for inverse) or just chirp[k] * conv[k]
__kernel void bluestein_finalize(
    __global const float2* conv,
    __global const float2* chirp,
    __global float2* output,
    const int N,
    const int M,
    const int batch_size,
    const int inverse
) {
    const int gid = get_global_id(0);
    const int batch_idx = gid / N;
    const int k = gid % N;

    if (batch_idx >= batch_size || k >= N) return;

    float2 c = conv[batch_idx * M + k];
    float2 w = chirp[k];

    // Multiply by chirp (complex multiply)
    float2 result = (float2)(c.x * w.x - c.y * w.y, c.x * w.y + c.y * w.x);

    // Scale by 1/N for inverse FFT
    if (inverse) {
        result = result / (float)N;
    }

    output[batch_idx * N + k] = result;
}

)";

  return oss.str();
}

// Move axis to last position (for non-last-axis FFT)
std::vector<int> get_move_axis_to_last_perm(int ndim, int axis) {
  std::vector<int> perm;
  for (int i = 0; i < ndim; i++) {
    if (i != axis) perm.push_back(i);
  }
  perm.push_back(axis);
  return perm;
}

// Inverse permutation to move axis back from last position
std::vector<int> get_move_axis_from_last_perm(int ndim, int axis) {
  std::vector<int> perm(ndim);
  int j = 0;
  for (int i = 0; i < ndim; i++) {
    if (i < axis) {
      perm[i] = j++;
    } else if (i == axis) {
      perm[i] = ndim - 1;
    } else {
      perm[i] = j++;
    }
  }
  return perm;
}

// Forward declaration for power-of-2 FFT (used by Bluestein)
void fft_radix2_impl(
    const array& in,
    array& out,
    int n,
    int batch_size,
    bool inverse,
    Stream s);

void fft_c2c_impl(
    const array& in,
    array& out,
    int axis,
    bool inverse,
    Stream s) {

  auto& dev = opencl::device(s.device);

  // Get FFT size
  int n = in.shape(axis);
  int ndim = in.ndim();
  int last_axis = ndim - 1;

  // Allocate output if not already allocated
  if (out.data_shared_ptr() == nullptr) {
    out.set_data(opencl::allocator().malloc(out.nbytes()));
  }

  // Handle trivial case
  if (n == 1) {
    copy_gpu(in, out, CopyType::General, s);
    return;
  }

  // Calculate batch size
  int batch_size = in.size() / n;

  // Handle case where FFT axis is not the last axis
  // We need to move the axis to the end, do FFT on last axis, then move back
  array in_work = in;
  bool need_axis_swap = (axis != last_axis && ndim > 1);

  if (need_axis_swap) {
    // Move axis to last position using transpose
    auto perm = get_move_axis_to_last_perm(ndim, axis);
    Shape new_shape;
    Strides new_strides;
    for (int p : perm) {
      new_shape.push_back(in.shape(p));
      new_strides.push_back(in.strides()[p]);
    }

    // Create transposed view
    array transposed(new_shape, in.dtype(), nullptr, {});
    transposed.copy_shared_buffer(in, new_strides, in.flags(), in.data_size());

    // Make contiguous copy of transposed data
    in_work = array(new_shape, in.dtype(), nullptr, {});
    in_work.set_data(opencl::allocator().malloc(in_work.nbytes()));
    copy_gpu(transposed, in_work, CopyType::General, s);
  } else {
    // Make input contiguous if needed
    if (!in.flags().row_contiguous || in.strides()[axis] != 1) {
      in_work = array(in.shape(), in.dtype(), nullptr, {});
      in_work.set_data(opencl::allocator().malloc(in_work.nbytes()));
      copy_gpu(in, in_work, CopyType::General, s);
    }
  }

  // Create temporary output for FFT (or use final output if no axis swap needed)
  array fft_out = need_axis_swap ?
    array(in_work.shape(), in_work.dtype(), nullptr, {}) : out;

  if (need_axis_swap) {
    fft_out.set_data(opencl::allocator().malloc(fft_out.nbytes()));
  }

  // Choose algorithm based on size
  if (!is_power_of_2(n)) {
    // Use Bluestein's algorithm for non-power-of-2 sizes
    std::string source = generate_fft_kernels();

    // Compute M = next power of 2 >= 2*N - 1
    int M = next_power_of_2(2 * n - 1);
    int log2M = 0;
    for (int temp = M; temp > 1; temp >>= 1) log2M++;

    // Step 1: Generate chirp sequence
    array chirp({n}, complex64, nullptr, {});
    chirp.set_data(opencl::allocator().malloc(chirp.nbytes()));

    cl_kernel chirp_kernel = dev.get_kernel("bluestein_chirp", source);
    auto& enc1 = dev.get_command_encoder(s.index);
    enc1.set_kernel(chirp_kernel);
    enc1.set_output_array(chirp, 0);
    enc1.set_bytes(n, 1);
    enc1.set_bytes(inverse ? 1 : 0, 2);
    size_t chirp_global[3] = {static_cast<size_t>(n), 1, 1};
    enc1.dispatch_threads(chirp_global, nullptr, 1);
    dev.end_encoding(s.index);

    // Step 2: Modulate input with chirp and zero-pad to M
    array modulated({batch_size * M}, complex64, nullptr, {});
    modulated.set_data(opencl::allocator().malloc(modulated.nbytes()));

    cl_kernel mod_kernel = dev.get_kernel("bluestein_modulate", source);
    auto& enc2 = dev.get_command_encoder(s.index);
    enc2.set_kernel(mod_kernel);
    enc2.set_input_array(in_work, 0);
    enc2.set_input_array(chirp, 1);
    enc2.set_output_array(modulated, 2);
    enc2.set_bytes(n, 3);
    enc2.set_bytes(M, 4);
    enc2.set_bytes(batch_size, 5);
    size_t mod_global[3] = {static_cast<size_t>(batch_size * M), 1, 1};
    enc2.dispatch_threads(mod_global, nullptr, 1);
    dev.end_encoding(s.index);

    // Step 3: Prepare chirp filter (conj(chirp) with wrap-around)
    array filter({M}, complex64, nullptr, {});
    filter.set_data(opencl::allocator().malloc(filter.nbytes()));

    cl_kernel filter_kernel = dev.get_kernel("bluestein_prepare_filter", source);
    auto& enc3 = dev.get_command_encoder(s.index);
    enc3.set_kernel(filter_kernel);
    enc3.set_input_array(chirp, 0);
    enc3.set_output_array(filter, 1);
    enc3.set_bytes(n, 2);
    enc3.set_bytes(M, 3);
    size_t filter_global[3] = {static_cast<size_t>(M), 1, 1};
    enc3.dispatch_threads(filter_global, nullptr, 1);
    dev.end_encoding(s.index);

    // Step 4: FFT of modulated signal (power-of-2)
    array mod_fft({batch_size * M}, complex64, nullptr, {});
    mod_fft.set_data(opencl::allocator().malloc(mod_fft.nbytes()));
    fft_radix2_impl(modulated, mod_fft, M, batch_size, false, s);

    // Step 5: FFT of filter (power-of-2)
    array filter_fft({M}, complex64, nullptr, {});
    filter_fft.set_data(opencl::allocator().malloc(filter_fft.nbytes()));
    fft_radix2_impl(filter, filter_fft, M, 1, false, s);

    // Step 6: Multiply in frequency domain (for each batch)
    // We need to multiply each batch of mod_fft by filter_fft
    array product({batch_size * M}, complex64, nullptr, {});
    product.set_data(opencl::allocator().malloc(product.nbytes()));

    // Multiply each batch by the filter
    cl_kernel mul_kernel = dev.get_kernel("bluestein_multiply", source);
    for (int b = 0; b < batch_size; b++) {
      auto& enc4 = dev.get_command_encoder(s.index);
      enc4.set_kernel(mul_kernel);

      // Set buffer offsets for this batch
      cl_mem mod_buf = static_cast<cl_mem>(mod_fft.buffer().ptr());
      cl_mem filter_buf = static_cast<cl_mem>(filter_fft.buffer().ptr());
      cl_mem prod_buf = static_cast<cl_mem>(product.buffer().ptr());

      size_t mod_offset = b * M * sizeof(float) * 2;
      size_t prod_offset = b * M * sizeof(float) * 2;

      cl_buffer_region mod_region = {mod_offset, M * sizeof(float) * 2};
      cl_buffer_region prod_region = {prod_offset, M * sizeof(float) * 2};

      cl_int err;
      cl_mem mod_sub = clCreateSubBuffer(mod_buf, CL_MEM_READ_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &mod_region, &err);
      cl_mem prod_sub = clCreateSubBuffer(prod_buf, CL_MEM_WRITE_ONLY, CL_BUFFER_CREATE_TYPE_REGION, &prod_region, &err);

      enc4.set_buffer(mod_sub, 0);
      enc4.set_buffer(filter_buf, 1);
      enc4.set_buffer(prod_sub, 2);
      enc4.set_bytes(M, 3);

      size_t mul_global[3] = {static_cast<size_t>(M), 1, 1};
      enc4.dispatch_threads(mul_global, nullptr, 1);
      dev.end_encoding(s.index);

      clReleaseMemObject(mod_sub);
      clReleaseMemObject(prod_sub);
    }

    // Step 7: Inverse FFT of product
    array conv({batch_size * M}, complex64, nullptr, {});
    conv.set_data(opencl::allocator().malloc(conv.nbytes()));
    fft_radix2_impl(product, conv, M, batch_size, true, s);

    // Step 8: Extract and finalize result
    cl_kernel final_kernel = dev.get_kernel("bluestein_finalize", source);
    auto& enc5 = dev.get_command_encoder(s.index);
    enc5.set_kernel(final_kernel);
    enc5.set_input_array(conv, 0);
    enc5.set_input_array(chirp, 1);
    enc5.set_output_array(fft_out, 2);
    enc5.set_bytes(n, 3);
    enc5.set_bytes(M, 4);
    enc5.set_bytes(batch_size, 5);
    enc5.set_bytes(inverse ? 1 : 0, 6);
    size_t final_global[3] = {static_cast<size_t>(batch_size * n), 1, 1};
    enc5.dispatch_threads(final_global, nullptr, 1);
    dev.end_encoding(s.index);

  } else {
    // Power-of-2: use radix-2 FFT
    fft_radix2_impl(in_work, fft_out, n, batch_size, inverse, s);
  }

  // Step 4: Transpose back if needed
  if (need_axis_swap) {
    // Move axis from last position back to original position
    auto inv_perm = get_move_axis_from_last_perm(ndim, axis);
    Shape new_shape;
    Strides new_strides;
    for (int p : inv_perm) {
      new_shape.push_back(fft_out.shape(p));
      new_strides.push_back(fft_out.strides()[p]);
    }

    // Create transposed view of fft_out
    array transposed(new_shape, fft_out.dtype(), nullptr, {});
    transposed.copy_shared_buffer(fft_out, new_strides, fft_out.flags(), fft_out.data_size());

    // Copy to final output
    copy_gpu(transposed, out, CopyType::General, s);
  }
}

// Radix-2 FFT implementation (for power-of-2 sizes only)
// Assumes input is contiguous with FFT axis as last dimension
void fft_radix2_impl(
    const array& in,
    array& out,
    int n,
    int batch_size,
    bool inverse,
    Stream s) {

  auto& dev = opencl::device(s.device);

  // Compute log2(n)
  int log2n = 0;
  for (int temp = n; temp > 1; temp >>= 1) log2n++;

  // Generate kernel source
  std::string source = generate_fft_kernels();

  // Get kernels
  cl_kernel bit_reverse_kernel = dev.get_kernel("fft_bit_reverse", source);
  cl_kernel butterfly_kernel = dev.get_kernel("fft_butterfly_stage", source);
  cl_kernel scale_kernel = dev.get_kernel("fft_scale", source);

  auto& encoder = dev.get_command_encoder(s.index);

  // Step 1: Bit-reversal permutation (in -> out)
  encoder.set_kernel(bit_reverse_kernel);
  encoder.set_input_array(in, 0);
  encoder.set_output_array(out, 1);
  encoder.set_bytes(n, 2);
  encoder.set_bytes(log2n, 3);
  encoder.set_bytes(batch_size, 4);

  size_t total_elems = batch_size * n;
  size_t global_size[3] = {total_elems, 1, 1};
  encoder.dispatch_threads(global_size, nullptr, 1);
  dev.end_encoding(s.index);

  // Step 2: Butterfly stages (in-place on out)
  int inverse_int = inverse ? 1 : 0;
  for (int stage = 0; stage < log2n; stage++) {
    auto& enc = dev.get_command_encoder(s.index);
    enc.set_kernel(butterfly_kernel);

    // For in-place operation, we need to set both input and output to the same buffer
    cl_mem out_buf = static_cast<cl_mem>(out.buffer().ptr());
    enc.set_buffer(out_buf, 0);
    enc.set_bytes(n, 1);
    enc.set_bytes(stage, 2);
    enc.set_bytes(batch_size, 3);
    enc.set_bytes(inverse_int, 4);

    size_t butterfly_global[3] = {static_cast<size_t>(batch_size * n / 2), 1, 1};
    enc.dispatch_threads(butterfly_global, nullptr, 1);
    dev.end_encoding(s.index);
  }

  // Step 3: Scale for inverse FFT
  if (inverse) {
    auto& enc = dev.get_command_encoder(s.index);
    enc.set_kernel(scale_kernel);

    cl_mem out_buf = static_cast<cl_mem>(out.buffer().ptr());
    enc.set_buffer(out_buf, 0);
    int size_int = static_cast<int>(total_elems);
    enc.set_bytes(size_int, 1);
    float scale = 1.0f / n;
    enc.set_bytes(scale, 2);

    size_t scale_global[3] = {total_elems, 1, 1};
    enc.dispatch_threads(scale_global, nullptr, 1);
    dev.end_encoding(s.index);
  }
}

} // anonymous namespace

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];
  auto& s = stream();

  // Multi-dimensional FFT: iterate over axes
  // Each 1D FFT along an axis transforms the result of the previous FFT
  if (axes_.size() > 1) {
    // Multi-dimensional FFT
    array current = in;

    for (size_t i = 0; i < axes_.size(); i++) {
      size_t axis = axes_[i];
      bool is_first = (i == 0);
      bool is_last = (i == axes_.size() - 1);

      // Determine input/output types for this axis
      // First axis: uses actual input type
      // Subsequent axes: complex64 (result of previous FFT)
      // Last axis: uses actual output type
      Dtype in_dtype = is_first ? in.dtype() : complex64;
      Dtype out_dtype = is_last ? out.dtype() : complex64;

      // Create output shape for this axis
      Shape out_shape = current.shape();
      if (out_dtype == float32 && in_dtype == complex64) {
        // IRFFT: output is real, size comes from out array
        out_shape[axis] = out.shape(axis);
      } else if (out_dtype == complex64 && in_dtype == float32) {
        // RFFT: output is complex, n/2+1
        out_shape[axis] = current.shape(axis) / 2 + 1;
      }

      array axis_out = is_last ? out : array(out_shape, out_dtype, nullptr, {});

      // Perform 1D FFT along this axis
      if (in_dtype == complex64 && out_dtype == complex64) {
        fft_c2c_impl(current, axis_out, axis, inverse_, s);
      } else if (in_dtype == float32 && out_dtype == complex64) {
        // RFFT for first axis only
        int n = current.shape(axis);

        auto& dev = opencl::device(s.device);
        std::string source = generate_fft_kernels();
        int batch_size = current.size() / n;
        int out_n = n / 2 + 1;

        // Allocate temporary complex buffer for the full FFT
        array temp(current.shape(), complex64, nullptr, {});
        temp.set_data(opencl::allocator().malloc(batch_size * n * sizeof(float) * 2));

        // Convert real to complex
        cl_kernel r2c_kernel = dev.get_kernel("real_to_complex", source);
        auto& enc1 = dev.get_command_encoder(s.index);
        enc1.set_kernel(r2c_kernel);
        enc1.set_input_array(current, 0);
        enc1.set_output_array(temp, 1);
        int size_int = batch_size * n;
        enc1.set_bytes(size_int, 2);
        size_t r2c_global[3] = {static_cast<size_t>(size_int), 1, 1};
        enc1.dispatch_threads(r2c_global, nullptr, 1);
        dev.end_encoding(s.index);

        // Perform complex FFT
        array fft_out(temp.shape(), complex64, nullptr, {});
        fft_out.set_data(opencl::allocator().malloc(batch_size * n * sizeof(float) * 2));
        fft_c2c_impl(temp, fft_out, axis, false, s);

        // Extract first n/2+1 values
        axis_out.set_data(opencl::allocator().malloc(axis_out.nbytes()));
        cl_kernel extract_kernel = dev.get_kernel("rfft_extract", source);
        auto& enc2 = dev.get_command_encoder(s.index);
        enc2.set_kernel(extract_kernel);
        enc2.set_input_array(fft_out, 0);
        enc2.set_output_array(axis_out, 1);
        enc2.set_bytes(n, 2);
        enc2.set_bytes(out_n, 3);
        enc2.set_bytes(batch_size, 4);
        size_t extract_global[3] = {static_cast<size_t>(batch_size * out_n), 1, 1};
        enc2.dispatch_threads(extract_global, nullptr, 1);
        dev.end_encoding(s.index);

      } else if (in_dtype == complex64 && out_dtype == float32) {
        // IRFFT for last axis only
        int n = out.shape(axis);
        int in_n = current.shape(axis);

        auto& dev = opencl::device(s.device);
        std::string source = generate_fft_kernels();
        int batch_size = current.size() / in_n;

        // Reconstruct full spectrum using Hermitian symmetry
        array temp(axis_out.shape(), complex64, nullptr, {});
        temp.set_data(opencl::allocator().malloc(batch_size * n * sizeof(float) * 2));

        cl_kernel prepare_kernel = dev.get_kernel("irfft_prepare", source);
        auto& enc1 = dev.get_command_encoder(s.index);
        enc1.set_kernel(prepare_kernel);
        enc1.set_input_array(current, 0);
        enc1.set_output_array(temp, 1);
        enc1.set_bytes(n, 2);
        enc1.set_bytes(in_n, 3);
        enc1.set_bytes(batch_size, 4);
        size_t prep_global[3] = {static_cast<size_t>(batch_size * n), 1, 1};
        enc1.dispatch_threads(prep_global, nullptr, 1);
        dev.end_encoding(s.index);

        // Perform inverse complex FFT
        array ifft_out(temp.shape(), complex64, nullptr, {});
        ifft_out.set_data(opencl::allocator().malloc(batch_size * n * sizeof(float) * 2));
        fft_c2c_impl(temp, ifft_out, axis, true, s);

        // Extract real part
        axis_out.set_data(opencl::allocator().malloc(axis_out.nbytes()));
        cl_kernel c2r_kernel = dev.get_kernel("complex_to_real", source);
        auto& enc2 = dev.get_command_encoder(s.index);
        enc2.set_kernel(c2r_kernel);
        enc2.set_input_array(ifft_out, 0);
        enc2.set_output_array(axis_out, 1);
        int size_int = batch_size * n;
        enc2.set_bytes(size_int, 2);
        size_t c2r_global[3] = {static_cast<size_t>(size_int), 1, 1};
        enc2.dispatch_threads(c2r_global, nullptr, 1);
        dev.end_encoding(s.index);
      }

      current = axis_out;
    }
    return;
  }

  size_t axis = axes_[0];

  // Handle the different FFT types
  if (in.dtype() == complex64 && out.dtype() == complex64) {
    // Complex-to-complex FFT
    fft_c2c_impl(in, out, axis, inverse_, s);

  } else if (in.dtype() == float32 && out.dtype() == complex64) {
    // Real-to-complex FFT (RFFT)
    int n = in.shape(axis);

    auto& dev = opencl::device(s.device);
    std::string source = generate_fft_kernels();

    int batch_size = in.size() / n;
    int out_n = n / 2 + 1;

    // For n=1, just convert real to complex directly
    if (n == 1) {
      out.set_data(opencl::allocator().malloc(out.nbytes()));
      cl_kernel r2c_kernel = dev.get_kernel("real_to_complex", source);
      auto& enc = dev.get_command_encoder(s.index);
      enc.set_kernel(r2c_kernel);
      enc.set_input_array(in, 0);
      enc.set_output_array(out, 1);
      int size_int = batch_size;
      enc.set_bytes(size_int, 2);
      size_t global[3] = {static_cast<size_t>(batch_size), 1, 1};
      enc.dispatch_threads(global, nullptr, 1);
      dev.end_encoding(s.index);
      return;
    }

    // Allocate temporary complex buffer for the full FFT
    auto temp_shape = in.shape();
    array temp(temp_shape, complex64, nullptr, {});
    temp.set_data(opencl::allocator().malloc(batch_size * n * sizeof(float) * 2));

    // Step 1: Convert real to complex
    cl_kernel r2c_kernel = dev.get_kernel("real_to_complex", source);
    auto& enc1 = dev.get_command_encoder(s.index);
    enc1.set_kernel(r2c_kernel);
    enc1.set_input_array(in, 0);
    enc1.set_output_array(temp, 1);
    int size_int = batch_size * n;
    enc1.set_bytes(size_int, 2);
    size_t r2c_global[3] = {static_cast<size_t>(size_int), 1, 1};
    enc1.dispatch_threads(r2c_global, nullptr, 1);
    dev.end_encoding(s.index);

    // Step 2: Perform complex FFT
    array fft_out(temp.shape(), complex64, nullptr, {});
    fft_out.set_data(opencl::allocator().malloc(batch_size * n * sizeof(float) * 2));
    fft_c2c_impl(temp, fft_out, axis, false, s);

    // Step 3: Extract first n/2+1 values
    out.set_data(opencl::allocator().malloc(out.nbytes()));

    cl_kernel extract_kernel = dev.get_kernel("rfft_extract", source);
    auto& enc2 = dev.get_command_encoder(s.index);
    enc2.set_kernel(extract_kernel);
    enc2.set_input_array(fft_out, 0);
    enc2.set_output_array(out, 1);
    enc2.set_bytes(n, 2);
    enc2.set_bytes(out_n, 3);
    enc2.set_bytes(batch_size, 4);

    size_t extract_global[3] = {static_cast<size_t>(batch_size * out_n), 1, 1};
    enc2.dispatch_threads(extract_global, nullptr, 1);
    dev.end_encoding(s.index);

  } else if (in.dtype() == complex64 && out.dtype() == float32) {
    // Complex-to-real FFT (IRFFT)
    int n = out.shape(axis);  // Output size
    int in_n = in.shape(axis);  // n/2 + 1

    auto& dev = opencl::device(s.device);
    std::string source = generate_fft_kernels();

    int batch_size = in.size() / in_n;

    // For n=1, just extract real part directly
    if (n == 1) {
      out.set_data(opencl::allocator().malloc(out.nbytes()));
      cl_kernel c2r_kernel = dev.get_kernel("complex_to_real", source);
      auto& enc = dev.get_command_encoder(s.index);
      enc.set_kernel(c2r_kernel);
      enc.set_input_array(in, 0);
      enc.set_output_array(out, 1);
      int size_int = batch_size;
      enc.set_bytes(size_int, 2);
      size_t global[3] = {static_cast<size_t>(batch_size), 1, 1};
      enc.dispatch_threads(global, nullptr, 1);
      dev.end_encoding(s.index);
      return;
    }

    // Allocate temporary buffer for full spectrum
    array temp(out.shape(), complex64, nullptr, {});
    temp.set_data(opencl::allocator().malloc(batch_size * n * sizeof(float) * 2));

    // Step 1: Reconstruct full spectrum using Hermitian symmetry
    cl_kernel prepare_kernel = dev.get_kernel("irfft_prepare", source);
    auto& enc1 = dev.get_command_encoder(s.index);
    enc1.set_kernel(prepare_kernel);
    enc1.set_input_array(in, 0);
    enc1.set_output_array(temp, 1);
    enc1.set_bytes(n, 2);
    enc1.set_bytes(in_n, 3);
    enc1.set_bytes(batch_size, 4);

    size_t prep_global[3] = {static_cast<size_t>(batch_size * n), 1, 1};
    enc1.dispatch_threads(prep_global, nullptr, 1);
    dev.end_encoding(s.index);

    // Step 2: Perform inverse complex FFT
    array ifft_out(temp.shape(), complex64, nullptr, {});
    ifft_out.set_data(opencl::allocator().malloc(batch_size * n * sizeof(float) * 2));
    fft_c2c_impl(temp, ifft_out, axis, true, s);

    // Step 3: Extract real part
    out.set_data(opencl::allocator().malloc(out.nbytes()));

    cl_kernel c2r_kernel = dev.get_kernel("complex_to_real", source);
    auto& enc2 = dev.get_command_encoder(s.index);
    enc2.set_kernel(c2r_kernel);
    enc2.set_input_array(ifft_out, 0);
    enc2.set_output_array(out, 1);
    int size_int = batch_size * n;
    enc2.set_bytes(size_int, 2);

    size_t c2r_global[3] = {static_cast<size_t>(size_int), 1, 1};
    enc2.dispatch_threads(c2r_global, nullptr, 1);
    dev.end_encoding(s.index);

  } else {
    throw std::runtime_error(
        "[FFT::eval_gpu] Unsupported input/output type combination");
  }
}

} // namespace mlx::core
