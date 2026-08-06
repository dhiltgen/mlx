// Copyright © 2025-2026 Apple Inc.

#include "mlx/backend/common/quantized.h"
#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/cuda/quantized/qmm/qmm.h"
#include "mlx/backend/cuda/quantized/quantized_utils.h"
#include "mlx/dtype_utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cutlass/float8.h>
#include <cutlass/numeric_conversion.h>

namespace mlx::core {

constexpr int rows_per_block = 8;
constexpr int fast_rows_per_warp = 2;
constexpr int fast_warps_per_block = 4;
constexpr int fast_rows_per_block = fast_rows_per_warp * fast_warps_per_block;

namespace cu {

namespace cg = cooperative_groups;

template <typename T>
__device__ void adjust_matrix_offsets(
    const T*& x,
    const uint32_t*& w,
    const uint8_t*& scales,
    T*& y,
    int output_stride,
    const int& x_batch_ndims,
    const Shape x_shape,
    const Strides x_strides,
    const int& w_batch_ndims,
    const Shape w_shape,
    const Strides w_strides,
    const Strides s_strides) {
  uint32_t idx = cg::this_grid().block_index().z;
  if (x_batch_ndims == 1) {
    x += idx * x_strides[0];
  } else {
    x += elem_to_loc(idx, x_shape.data(), x_strides.data(), x_batch_ndims);
  }
  if (w_batch_ndims == 1) {
    w += idx * w_strides[0];
    scales += idx * s_strides[0];
  } else {
    auto [w_idx, s_idx] = elem_to_loc(
        idx, w_shape.data(), w_strides.data(), s_strides.data(), w_batch_ndims);
    w += w_idx;
    scales += s_idx;
  }
  y += idx * output_stride;
}

template <
    typename T,
    int rows_per_block,
    int n_per_thread,
    int bits,
    int group_size,
    bool use_mx_scale,
    bool vectorize_scales = false>
__device__ void fp_qmv_impl(
    const uint32_t* mat,
    const uint8_t* scales_,
    const T* vec,
    T* out,
    int rows,
    int cols,
    int vector_index,
    int row_tile) {
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  constexpr int vals_per_item = bits == 8 ? 4 : 8;
  constexpr int nv_per_thread = vals_per_item * n_per_thread;
  auto t_idx = block.thread_index();
  int row = row_tile * rows_per_block + t_idx.y;

  vec += vector_index * cols;
  out += vector_index * rows;

  using ScaleType = std::conditional_t<
      use_mx_scale,
      cutlass::float_ue8m0_t,
      cutlass::float_e4m3_t>;
  auto scales = (ScaleType*)(scales_);
  auto packed_cols = cols / vals_per_item;

  if (row < rows) {
    constexpr int scales_per_step = std::max(nv_per_thread / group_size, 1);
    constexpr int scale_step = (WARP_SIZE * nv_per_thread) / group_size;
    constexpr int n_per_step = n_per_thread / scales_per_step;
    // Offset scales to correct row
    scales += row * (cols / group_size) +
        (warp.thread_rank() * nv_per_thread) / group_size;
    float sum = 0.0f;
    for (int col = n_per_thread * warp.thread_rank(); col < packed_cols;
         col += (WARP_SIZE * n_per_thread)) {
      auto local_vec =
          unsafe_load_vector<nv_per_thread>(vec + vals_per_item * col, 0);
      auto local_mat =
          unsafe_load_vector<n_per_thread>(mat + row * packed_cols + col, 0);
      AlignedVector<ScaleType, scales_per_step> local_scales;
      if constexpr (vectorize_scales) {
        local_scales = unsafe_load_vector<scales_per_step>(scales, 0);
      }
#pragma unroll
      for (int i = 0; i < scales_per_step; ++i) {
        float2 local_sum = {0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < n_per_step; ++j) {
          int k = n_per_step * i + j;
          if constexpr (bits == 8) {
            cutlass::NumericArrayConverter<float, cutlass::float_e4m3_t, 4>
                converter;
            auto v = converter(
                *reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 4>*>(
                    &local_mat[k]));
            local_sum.x +=
                v[0] * static_cast<float>(local_vec[vals_per_item * k]);
            local_sum.x +=
                v[1] * static_cast<float>(local_vec[vals_per_item * k + 1]);
            local_sum.y +=
                v[2] * static_cast<float>(local_vec[vals_per_item * k + 2]);
            local_sum.y +=
                v[3] * static_cast<float>(local_vec[vals_per_item * k + 3]);
          } else {
            cutlass::NumericArrayConverter<float, cutlass::float_e2m1_t, 8>
                converter;
            auto v = converter(
                *reinterpret_cast<cutlass::Array<cutlass::float_e2m1_t, 8>*>(
                    &local_mat[k]));
            local_sum.x +=
                v[0] * static_cast<float>(local_vec[vals_per_item * k]);
            local_sum.y +=
                v[1] * static_cast<float>(local_vec[vals_per_item * k + 1]);
            local_sum.x +=
                v[2] * static_cast<float>(local_vec[vals_per_item * k + 2]);
            local_sum.y +=
                v[3] * static_cast<float>(local_vec[vals_per_item * k + 3]);
            local_sum.x +=
                v[4] * static_cast<float>(local_vec[vals_per_item * k + 4]);
            local_sum.y +=
                v[5] * static_cast<float>(local_vec[vals_per_item * k + 5]);
            local_sum.x +=
                v[6] * static_cast<float>(local_vec[vals_per_item * k + 6]);
            local_sum.y +=
                v[7] * static_cast<float>(local_vec[vals_per_item * k + 7]);
          }
        }
        float scale;
        if constexpr (vectorize_scales) {
          scale = float(local_scales[i]);
        } else {
          scale = float(scales[i]);
        }
        sum += (local_sum.x + local_sum.y) * scale;
      }
      scales += scale_step;
    }

    sum = cg::reduce(warp, sum, cg::plus<float>{});
    if (warp.thread_rank() == 0) {
      out[row] = static_cast<T>(sum);
    }
  }
}

// cuBLASLt block-scaled GEMM cannot express MLX's gathered expert selection
// and scale layouts, and its setup does not amortize for these decode batches.
// Keep the multi-row loop specialized: factoring it with fp_qmv_impl changed
// CUDA code generation and regressed end-to-end decode by 8% on SM120.
template <
    typename T,
    int rows_per_warp,
    int n_per_thread,
    int bits,
    int group_size,
    bool use_mx_scale>
__device__ void fp_qmv_fast_impl(
    const uint32_t* mat,
    const uint8_t* scales_,
    const T* vec,
    T* out,
    int rows,
    int cols,
    int vector_index,
    int row_tile) {
  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<WARP_SIZE>(block);

  constexpr int vals_per_item = bits == 8 ? 4 : 8;
  constexpr int nv_per_thread = vals_per_item * n_per_thread;
  auto t_idx = block.thread_index();
  int row_base = row_tile * fast_rows_per_block + t_idx.y * rows_per_warp;

  if (row_base >= rows) {
    return;
  }

  vec += vector_index * cols;
  out += vector_index * rows;

  using ScaleType = std::conditional_t<
      use_mx_scale,
      cutlass::float_ue8m0_t,
      cutlass::float_e4m3_t>;
  auto scales = (ScaleType*)(scales_);
  auto packed_cols = cols / vals_per_item;

  constexpr int scales_per_step = std::max(nv_per_thread / group_size, 1);
  constexpr int scale_step = (WARP_SIZE * nv_per_thread) / group_size;
  constexpr int n_per_step = n_per_thread / scales_per_step;

  float sums[rows_per_warp] = {0.0f};
  int scale_col = (warp.thread_rank() * nv_per_thread) / group_size;
  for (int col = n_per_thread * warp.thread_rank(); col < packed_cols;
       col += (WARP_SIZE * n_per_thread)) {
    auto local_vec =
        unsafe_load_vector<nv_per_thread>(vec + vals_per_item * col, 0);

#pragma unroll
    for (int row_offset = 0; row_offset < rows_per_warp; ++row_offset) {
      int row = row_base + row_offset;
      if (row < rows) {
        auto local_mat =
            unsafe_load_vector<n_per_thread>(mat + row * packed_cols + col, 0);
        auto row_scales = scales + row * (cols / group_size) + scale_col;
#pragma unroll
        for (int i = 0; i < scales_per_step; ++i) {
          float2 local_sum = {0.0f, 0.0f};
#pragma unroll
          for (int j = 0; j < n_per_step; ++j) {
            int k = n_per_step * i + j;
            if constexpr (bits == 8) {
              cutlass::NumericArrayConverter<float, cutlass::float_e4m3_t, 4>
                  converter;
              auto v = converter(
                  *reinterpret_cast<cutlass::Array<cutlass::float_e4m3_t, 4>*>(
                      &local_mat[k]));
              local_sum.x +=
                  v[0] * static_cast<float>(local_vec[vals_per_item * k]);
              local_sum.x +=
                  v[1] * static_cast<float>(local_vec[vals_per_item * k + 1]);
              local_sum.y +=
                  v[2] * static_cast<float>(local_vec[vals_per_item * k + 2]);
              local_sum.y +=
                  v[3] * static_cast<float>(local_vec[vals_per_item * k + 3]);
            } else {
              cutlass::NumericArrayConverter<float, cutlass::float_e2m1_t, 8>
                  converter;
              auto v = converter(
                  *reinterpret_cast<cutlass::Array<cutlass::float_e2m1_t, 8>*>(
                      &local_mat[k]));
              local_sum.x +=
                  v[0] * static_cast<float>(local_vec[vals_per_item * k]);
              local_sum.y +=
                  v[1] * static_cast<float>(local_vec[vals_per_item * k + 1]);
              local_sum.x +=
                  v[2] * static_cast<float>(local_vec[vals_per_item * k + 2]);
              local_sum.y +=
                  v[3] * static_cast<float>(local_vec[vals_per_item * k + 3]);
              local_sum.x +=
                  v[4] * static_cast<float>(local_vec[vals_per_item * k + 4]);
              local_sum.y +=
                  v[5] * static_cast<float>(local_vec[vals_per_item * k + 5]);
              local_sum.x +=
                  v[6] * static_cast<float>(local_vec[vals_per_item * k + 6]);
              local_sum.y +=
                  v[7] * static_cast<float>(local_vec[vals_per_item * k + 7]);
            }
          }
          sums[row_offset] +=
              (local_sum.x + local_sum.y) * float(row_scales[i]);
        }
      }
    }
    scale_col += scale_step;
  }

#pragma unroll
  for (int row_offset = 0; row_offset < rows_per_warp; ++row_offset) {
    int row = row_base + row_offset;
    float sum = cg::reduce(warp, sums[row_offset], cg::plus<float>{});
    if (warp.thread_rank() == 0 && row < rows) {
      out[row] = static_cast<T>(sum);
    }
  }
}

template <
    typename T,
    int rows_per_block,
    int n_per_thread,
    int bits,
    int group_size,
    bool use_mx_scale,
    bool vectorize_scales = false>
__global__ void fp_qmv_single(
    const uint32_t* mat,
    const uint8_t* scales,
    const T* vec,
    T* out,
    int rows,
    int cols) {
  auto g_idx = cg::this_grid().block_index();
  fp_qmv_impl<
      T,
      rows_per_block,
      n_per_thread,
      bits,
      group_size,
      use_mx_scale,
      vectorize_scales>(mat, scales, vec, out, rows, cols, g_idx.x, g_idx.y);
}

template <
    typename T,
    int rows_per_block,
    int n_per_thread,
    int bits,
    int group_size,
    bool use_mx_scale>
__global__ void fp_qmv_batched(
    const uint32_t* mat,
    const uint8_t* scales,
    const T* vec,
    T* out,
    int rows,
    int cols,
    int vec_batch_ndims,
    const __grid_constant__ Shape vec_shape,
    const __grid_constant__ Strides vec_strides,
    int mat_batch_ndims,
    const __grid_constant__ Shape mat_shape,
    const __grid_constant__ Strides mat_strides,
    const __grid_constant__ Strides scales_strides) {
  adjust_matrix_offsets<T>(
      vec,
      mat,
      scales,
      out,
      rows * vec_shape[vec_batch_ndims],
      vec_batch_ndims,
      vec_shape,
      vec_strides,
      mat_batch_ndims,
      mat_shape,
      mat_strides,
      scales_strides);
  auto g_idx = cg::this_grid().block_index();
  fp_qmv_impl<T, rows_per_block, n_per_thread, bits, group_size, use_mx_scale>(
      mat, scales, vec, out, rows, cols, g_idx.x, g_idx.y);
}

template <
    typename T,
    int rows_per_block,
    int n_per_thread,
    int bits,
    int group_size,
    bool use_mx_scale>
__global__ void fp_gather_qmv_single(
    const uint32_t* mat,
    const uint8_t* scales,
    const T* vec,
    T* out,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    int rows,
    int cols,
    int vecs_per_index) {
  auto g_idx = cg::this_grid().block_index();
  uint32_t x_idx = lhs_indices[g_idx.z];
  uint32_t w_idx = rhs_indices[g_idx.z];

  constexpr int vals_per_item = bits == 8 ? 4 : 8;
  int packed_cols = cols / vals_per_item;
  mat += static_cast<int64_t>(w_idx) * rows * packed_cols;
  scales += static_cast<int64_t>(w_idx) * rows * (cols / group_size);
  vec += static_cast<int64_t>(x_idx) * vecs_per_index * cols;
  out += static_cast<int64_t>(g_idx.z) * vecs_per_index * rows;

  fp_qmv_impl<T, rows_per_block, n_per_thread, bits, group_size, use_mx_scale>(
      mat, scales, vec, out, rows, cols, g_idx.x, g_idx.y);
}

template <
    typename T,
    int rows_per_warp,
    int n_per_thread,
    int bits,
    int group_size,
    bool use_mx_scale>
__global__ void fp_gather_qmv_fast_single(
    const uint32_t* mat,
    const uint8_t* scales,
    const T* vec,
    T* out,
    const uint32_t* lhs_indices,
    const uint32_t* rhs_indices,
    int rows,
    int cols,
    int vecs_per_index) {
  auto g_idx = cg::this_grid().block_index();
  uint32_t x_idx = lhs_indices[g_idx.z];
  uint32_t w_idx = rhs_indices[g_idx.z];

  constexpr int vals_per_item = bits == 8 ? 4 : 8;
  int packed_cols = cols / vals_per_item;
  mat += static_cast<int64_t>(w_idx) * rows * packed_cols;
  scales += static_cast<int64_t>(w_idx) * rows * (cols / group_size);
  vec += static_cast<int64_t>(x_idx) * vecs_per_index * cols;
  out += static_cast<int64_t>(g_idx.z) * vecs_per_index * rows;

  fp_qmv_fast_impl<
      T,
      rows_per_warp,
      n_per_thread,
      bits,
      group_size,
      use_mx_scale>(mat, scales, vec, out, rows, cols, g_idx.x, g_idx.y);
}

} // namespace cu

template <typename F>
void dispatch_1_2_4(int n, F&& f) {
  switch (n) {
    case 1:
      f(std::integral_constant<int, 1>{});
      break;
    case 2:
      f(std::integral_constant<int, 2>{});
      break;
    case 4:
      f(std::integral_constant<int, 4>{});
      break;
  }
}

void fp_qmv(
    const array& x,
    const array& w,
    const array& scales_,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s) {
  uint32_t M = x.shape(-2);
  uint32_t N = out.shape(-1);
  uint32_t K = x.shape(-1);
  uint32_t B = out.size() / (M * N);

  // Make sure the last two dims of x and w, s, b are contiguous. This should
  // be relaxed for x.
  array vec = ensure_row_contiguous_matrix(x, encoder, s);
  array mat = ensure_row_contiguous_matrix(w, encoder, s);
  array scales = ensure_row_contiguous_matrix(scales_, encoder, s);

  encoder.set_input_array(mat);
  encoder.set_input_array(scales);
  encoder.set_input_array(vec);
  encoder.set_output_array(out);
  dispatch_float_types(out.dtype(), "qmv", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      dim3 block_dims{WARP_SIZE, rows_per_block};
      uint32_t blocks_y = (N + rows_per_block - 1) / rows_per_block;
      const uint32_t* mat_ptr = gpu_ptr<uint32_t>(mat);
      const T* vec_ptr = gpu_ptr<T>(vec);
      int n = 1;
      if (K % 32 == 0 && cu::is_aligned<4>(mat_ptr) &&
          ((bits == 4 && cu::is_aligned<8>(vec_ptr)) ||
           cu::is_aligned<4>(vec_ptr))) {
        n = 4;
      } else if (
          cu::is_aligned<2>(mat_ptr) &&
          ((bits == 4 && cu::is_aligned<4>(vec_ptr)) ||
           cu::is_aligned<2>(vec_ptr))) {
        n = 2;
      }
      auto cc = encoder.device().compute_capability_major() * 100 +
          encoder.device().compute_capability_minor() * 10;
      dispatch_1_2_4(n, [&](auto n) {
        if (B == 1) {
          auto kernel =
              cu::fp_qmv_single<T, rows_per_block, n.value, 4, 32, true>;
          if (bits == 8) {
            kernel = cu::fp_qmv_single<T, rows_per_block, n.value, 8, 32, true>;
          } else if (group_size == 16) {
            kernel =
                cu::fp_qmv_single<T, rows_per_block, n.value, 4, 16, false>;
            if (cc == 1210) {
              kernel = cu::
                  fp_qmv_single<T, rows_per_block, n.value, 4, 16, false, true>;
            }
          }
          encoder.add_kernel_node(
              kernel,
              {uint32_t(x.size() / K), blocks_y},
              block_dims,
              mat_ptr,
              gpu_ptr<uint8_t>(scales),
              vec_ptr,
              gpu_ptr<T>(out),
              N,
              K);
        } else {
          auto kernel =
              cu::fp_qmv_batched<T, rows_per_block, n.value, 4, 32, true>;
          if (bits == 8) {
            kernel =
                cu::fp_qmv_batched<T, rows_per_block, n.value, 8, 32, true>;
          } else if (group_size == 16) {
            kernel =
                cu::fp_qmv_batched<T, rows_per_block, n.value, 4, 16, false>;
          }
          encoder.add_kernel_node(
              kernel,
              {M, blocks_y, B},
              block_dims,
              mat_ptr,
              gpu_ptr<uint8_t>(scales),
              vec_ptr,
              gpu_ptr<T>(out),
              N,
              K,
              vec.ndim() - 2,
              const_param(vec.shape()),
              const_param(vec.strides()),
              mat.ndim() - 2,
              const_param(mat.shape()),
              const_param(mat.strides()),
              const_param(scales.strides()));
        }
      });
    }
  });
}

void fp_gather_qmv(
    const array& x,
    const array& w,
    const array& scales_,
    const array& lhs_indices,
    const array& rhs_indices,
    array& out,
    int bits,
    int group_size,
    cu::CommandEncoder& encoder,
    Stream s) {
  uint32_t M = out.shape(-2);
  uint32_t N = out.shape(-1);
  uint32_t K = x.shape(-1);
  uint32_t L = out.size() / (M * N);

  array vec = ensure_row_contiguous_matrix(x, encoder, s);
  array mat = ensure_row_contiguous_matrix(w, encoder, s);
  array scales = ensure_row_contiguous_matrix(scales_, encoder, s);

  encoder.set_input_array(mat);
  encoder.set_input_array(scales);
  encoder.set_input_array(vec);
  encoder.set_input_array(lhs_indices);
  encoder.set_input_array(rhs_indices);
  encoder.set_output_array(out);
  dispatch_float_types(out.dtype(), "gather_qmv", [&](auto type_tag) {
    using T = cuda_type_t<MLX_GET_TYPE(type_tag)>;
    if constexpr (!std::is_same_v<T, double>) {
      dim3 fast_block_dims{WARP_SIZE, fast_warps_per_block};
      dim3 block_dims{WARP_SIZE, rows_per_block};
      uint32_t blocks_y = (N + rows_per_block - 1) / rows_per_block;
      const uint32_t* mat_ptr = gpu_ptr<uint32_t>(mat);
      const T* vec_ptr = gpu_ptr<T>(vec);
      int n = 1;
      if (K % 32 == 0 && cu::is_aligned<4>(mat_ptr) &&
          ((bits == 4 && cu::is_aligned<8>(vec_ptr)) ||
           cu::is_aligned<4>(vec_ptr))) {
        n = 4;
      } else if (
          cu::is_aligned<2>(mat_ptr) &&
          ((bits == 4 && cu::is_aligned<4>(vec_ptr)) ||
           cu::is_aligned<2>(vec_ptr))) {
        n = 2;
      }
      dispatch_1_2_4(n, [&](auto n) {
        auto kernel =
            cu::fp_gather_qmv_single<T, rows_per_block, n.value, 4, 32, true>;
        if (bits == 8) {
          kernel =
              cu::fp_gather_qmv_single<T, rows_per_block, n.value, 8, 32, true>;
        } else if (group_size == 16) {
          kernel = cu::
              fp_gather_qmv_single<T, rows_per_block, n.value, 4, 16, false>;
        }
        if constexpr (n.value == 4) {
          if (N % fast_rows_per_block == 0) {
            auto fast_kernel = cu::fp_gather_qmv_fast_single<
                T,
                fast_rows_per_warp,
                n.value,
                4,
                32,
                true>;
            if (bits == 8) {
              fast_kernel = cu::fp_gather_qmv_fast_single<
                  T,
                  fast_rows_per_warp,
                  n.value,
                  8,
                  32,
                  true>;
            } else if (group_size == 16) {
              fast_kernel = cu::fp_gather_qmv_fast_single<
                  T,
                  fast_rows_per_warp,
                  n.value,
                  4,
                  16,
                  false>;
            }
            encoder.add_kernel_node(
                fast_kernel,
                {M, blocks_y, L},
                fast_block_dims,
                mat_ptr,
                gpu_ptr<uint8_t>(scales),
                vec_ptr,
                gpu_ptr<T>(out),
                gpu_ptr<uint32_t>(lhs_indices),
                gpu_ptr<uint32_t>(rhs_indices),
                N,
                K,
                M);
            return;
          }
        }
        encoder.add_kernel_node(
            kernel,
            {M, blocks_y, L},
            block_dims,
            mat_ptr,
            gpu_ptr<uint8_t>(scales),
            vec_ptr,
            gpu_ptr<T>(out),
            gpu_ptr<uint32_t>(lhs_indices),
            gpu_ptr<uint32_t>(rhs_indices),
            N,
            K,
            M);
      });
    }
  });
}

} // namespace mlx::core
