// Copyright © 2026 Apple Inc.

#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/backend/cuda/quantized/qmm/qmm_utils.h"
#include "mlx/utils.h"

#include <cute/tensor.hpp>

namespace mlx::core {

namespace {

inline bool is_last_2_dims_row_contiguous(const array& x) {
  return x.flags().contiguous && (x.ndim() >= 2) && (x.strides(-1) == 1) &&
      (x.strides(-2) == x.shape(-1));
}

} // namespace

bool supports_qmm_sm90(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  if (device.compute_capability_major() != 9) {
    return false;
  }
  auto [m, n, k, l, broadcast_b] = make_problem_shape(x, w, out);
  if ((n * w.itemsize()) % 16 != 0) { // TMA alignment
    return false;
  }
  if (k % 64 != 0) {
    return false;
  }
  if (!biases) {
    return false;
  }
  if (!is_last_2_dims_row_contiguous(w) ||
      !is_last_2_dims_row_contiguous(scales) ||
      !is_last_2_dims_row_contiguous(*biases)) {
    return false;
  }
  if (!transpose) {
    return false;
  }
  if (bits != 4 && bits != 8) {
    return false;
  }
  if (group_size < k) {
    return false;
  }
  if (mode != QuantizationMode::Affine) {
    return false;
  }
  return true;
}

bool supports_qmm_sm80(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  // Opt-out probe: MLX_CUDA_QMM_DISABLE_SM80=1 forces the naive qmm path.
  // A/B knob for the N1x (615.x) dense-qqmm prefill regression — the sm80_ss
  // CuTe kernels run at ~40-55% of their pre-crash rate on the WDDM driver
  // while gathers/nvjet stay healthy (ollama notes/MLX-CUDA-UM-QMM-
  // EXPERIMENTS.md, 2026-08-07). Lets us compare naive vs sm80 vs dequant
  // on identical box state.
  static const bool disable_sm80 = env::get_var("MLX_CUDA_QMM_DISABLE_SM80", 0);
  if (disable_sm80) {
    return false;
  }
  if (device.compute_capability_major() < 8) {
    return false;
  }
  int n = out.shape(-1);
  int k = x.shape(-1);
  if ((n < 128) || ((n % 128 != 0) && mode == QuantizationMode::Affine) ||
      (k % std::max(64, group_size) != 0)) {
    return false;
  }
  if (!is_last_2_dims_row_contiguous(w) ||
      !is_last_2_dims_row_contiguous(scales)) {
    return false;
  }
  if (biases && !is_last_2_dims_row_contiguous(*biases)) {
    return false;
  }
  if (x.dtype() != float16 && x.dtype() != bfloat16) {
    return false;
  }
  if (!transpose) {
    return false;
  }
  if (bits != 4 && bits != 8) {
    return false;
  }
  return true;
}

bool supports_qmm_naive(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  int k = x.shape(-1);
  if (transpose && (k % std::max(64, group_size) != 0)) {
    return false;
  }
  if (!is_last_2_dims_row_contiguous(w) ||
      !is_last_2_dims_row_contiguous(scales)) {
    return false;
  }
  if (biases && !is_last_2_dims_row_contiguous(*biases)) {
    return false;
  }
  return true;
}

bool supports_fp_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  // The fp_qmv kernel uses less registers and is faster for sm120. For sm80/90
  // the qmv kernel is faster. We didn't test sm89/100.
  if (device.compute_capability_major() <= 9) {
    return false;
  }
  bool non_batched = w.ndim() == 2;
  int k = x.shape(-1);
  int n = out.shape(-1);
  int vec_batch = non_batched ? x.size() / k : x.shape(-2);
  if (vec_batch > 8) {
    return false;
  }
  if (!transpose) {
    return false;
  }
  if (mode == QuantizationMode::Affine) {
    return false;
  }
  return true;
}

bool supports_fp_gather_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  if (!supports_fp_qmv(
          x,
          w,
          scales,
          biases,
          out,
          transpose,
          bits,
          group_size,
          mode,
          device)) {
    return false;
  }
  int m = out.ndim() > 1 ? out.shape(-2) : 1;
  int k = x.shape(-1);
  int vec_batch = x.size() / k;
  if (m > 8 || vec_batch < m) {
    return false;
  }
  if (w.ndim() < 3) {
    return false;
  }
  int rows = out.shape(-1);
  int cols = x.shape(-1);
  int experts = w.size() / (w.shape(-1) * w.shape(-2));
  return experts > 0 && rows == w.shape(-2) && cols == w.shape(-1) * 32 / bits;
}

bool supports_qmv(
    const array& x,
    const array& w,
    const array& scales,
    const std::optional<array>& biases,
    const array& out,
    bool transpose,
    int bits,
    int group_size,
    QuantizationMode mode,
    cu::Device& device) {
  int k = x.shape(-1);
  if (k % 8 != 0) {
    return false;
  }
  if (!is_last_2_dims_row_contiguous(w) ||
      !is_last_2_dims_row_contiguous(scales)) {
    return false;
  }
  if (biases && !is_last_2_dims_row_contiguous(*biases)) {
    return false;
  }
  if (!transpose) {
    return false;
  }
  return true;
}

} // namespace mlx::core
