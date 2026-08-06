// Copyright © 2025-2026 Apple Inc.

// Required for using M_LOG2E in MSVC.
#define _USE_MATH_DEFINES

#include "mlx/backend/cuda/device.h"
#include "mlx/backend/cuda/device/config.h"
#include "mlx/backend/cuda/device/utils.cuh"
#include "mlx/backend/cuda/kernel_utils.cuh"
#include "mlx/backend/gpu/copy.h"
#include "mlx/dtype_utils.h"
#include "mlx/utils.h"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <mutex>

namespace mlx::core {

namespace cu {

namespace cg = cooperative_groups;

#define PRAGMA_LOOP_UNROLL #pragma unroll

struct AttnParams {
  int B;
  int H;
  int D;

  int qL;
  int kL;

  int gqa_factor;
  int maskH;
  float scale;

  int64_t Q_strides[3];
  int64_t K_strides[3];
  int64_t V_strides[3];
  int64_t O_strides[3];
  int64_t M_strides[4];
};

template <typename T, bool do_causal, int D>
__global__ void kernel_sdpav_1pass(
    const T* Q,
    const T* K,
    const T* V,
    T* O,
    const T* M,
    const T* sinks,
    __grid_constant__ const AttnParams params) {
  constexpr int BN = 32;
  constexpr int BD = 32;

  constexpr int v_per_thread = D / BD;

  const int inner_k_stride = BN * int(params.K_strides[2]);
  const int inner_v_stride = BN * int(params.V_strides[2]);

  typedef float U;

  U q[v_per_thread];
  U k[v_per_thread];
  U o[v_per_thread];

  __shared__ U outputs[BN][BD + 1];
  __shared__ U max_scores[BN];
  __shared__ U sum_exp_scores[BN];

  const U scale_log2 = params.scale * M_LOG2E;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const int lane_idx = warp.thread_rank();
  const int warp_idx = warp.meta_group_rank();

  // Adjust to thread block and thread
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.x;
  const int kv_head_idx = head_idx / params.gqa_factor;

  const int q_seq_idx = blockIdx.y;
  const int kv_seq_idx = warp_idx;

  Q += batch_idx * params.Q_strides[0] + // Batch
      head_idx * params.Q_strides[1] + // Head
      q_seq_idx * params.Q_strides[2]; // Sequence

  K += batch_idx * params.K_strides[0] + // Batch
      kv_head_idx * params.K_strides[1] + // Head
      kv_seq_idx * params.K_strides[2]; // Sequence

  V += batch_idx * params.V_strides[0] + // Batch
      kv_head_idx * params.V_strides[1] + // Head
      kv_seq_idx * params.V_strides[2]; // Sequence

  O += batch_idx * params.O_strides[0] + // Batch
      head_idx * params.O_strides[1] + // Head
      q_seq_idx * params.O_strides[2]; // Sequence

  if (M) {
    M += batch_idx * params.M_strides[0] + // Batch
        (params.maskH == 1 ? 0 : head_idx * params.M_strides[1]) + // Head
        q_seq_idx * params.M_strides[2]; // Sequence
  }

  // Read the query and 0 the output accumulator
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    q[i] = scale_log2 * static_cast<U>(Q[v_per_thread * lane_idx + i]);
  }

  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0.f;
  }

  U max_score = Limits<U>::finite_min();
  U sum_exp_score = 0.f;
  if (sinks && warp_idx == 0) {
    max_score = M_LOG2E * static_cast<U>(sinks[head_idx]);
    sum_exp_score = 1.f;
  }

  // For each key
  for (int i = kv_seq_idx; i < params.kL; i += BN) {
    bool use_key = true;
    if constexpr (do_causal) {
      use_key = i <= (params.kL - params.qL + q_seq_idx);
    }

    U bias = 0.f;
    if (M) {
      bias = static_cast<U>(M[i * params.M_strides[3]]) * M_LOG2E;
      use_key = use_key && (bias >= Limits<U>::finite_min() || isnan(bias));
    }
    if (use_key) {
      // Read the key
      PRAGMA_LOOP_UNROLL
      for (int j = 0; j < v_per_thread; j++) {
        k[j] = K[v_per_thread * lane_idx + j];
      }

      // Compute the i-th score
      U score = 0.f;
      PRAGMA_LOOP_UNROLL
      for (int j = 0; j < v_per_thread; j++) {
        score += q[j] * k[j];
      }

      // Warp sum
      score = cg::reduce(warp, score, cg::plus<U>());

      // Update the accumulators
      score += bias;
      U new_max = max(max_score, score);
      U factor = exp2f(max_score - new_max);
      U exp_score = exp2f(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      PRAGMA_LOOP_UNROLL
      for (int j = 0; j < v_per_thread; j++) {
        o[j] = o[j] * factor +
            exp_score * static_cast<U>(V[v_per_thread * lane_idx + j]);
      }
    }

    // Move the pointers to the next kv
    K += inner_k_stride;
    V += inner_v_stride;
  }

  if (lane_idx == 0) {
    max_scores[warp_idx] = max_score;
    sum_exp_scores[warp_idx] = sum_exp_score;
  }
  block.sync();

  max_score = max_scores[lane_idx];
  U new_max = cg::reduce(warp, max_score, cg::greater<U>());
  U factor = exp2f(max_score - new_max);
  sum_exp_score =
      cg::reduce(warp, sum_exp_scores[lane_idx] * factor, cg::plus<U>());
  sum_exp_score = sum_exp_score == 0 ? 0 : __frcp_rn(sum_exp_score);

  // Now we need to aggregate all the outputs
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    outputs[lane_idx][warp_idx] = o[i];
    block.sync();
    U ot = outputs[warp_idx][lane_idx] * factor;
    o[i] = cg::reduce(warp, ot, cg::plus<U>()) * sum_exp_score;
    block.sync();
  }

  // And write the output
  if (lane_idx == 0) {
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      O[v_per_thread * warp_idx + i] = static_cast<T>(o[i]);
    }
  }
}

// Query-tiled online-softmax attention for wide head dims at prefill.
//
// kernel_sdpav_1pass assigns one 1024-thread block per (head, query row) and
// every block streams the full K/V sequence, producing qL*kL*D global traffic.
// This kernel assigns QT query rows of one (batch, head) to each block and
// streams K/V through shared memory in KB-row tiles, so each K/V byte is
// loaded once per QT queries instead of once per query. Each warp owns one
// query row: lanes compute independent key dots in the score phase (no cross-
// lane reductions), then accumulate V weighted by the tile softmax.
template <typename T, bool do_causal, int D>
__global__ void kernel_sdpav_fvec(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    const T* __restrict__ M,
    const T* __restrict__ sinks,
    __grid_constant__ const AttnParams params) {
  constexpr int QT = 8;
  // Sized so both head dims fit >= 2 blocks per SM alongside the fp32 Q tile
  // (SM121 has ~102KB of shared memory per SM).
  constexpr int KB = D == 256 ? 32 : 16;
  constexpr int KPAD = 8;
  constexpr int KROW = D + KPAD;
  constexpr int VEC = 16 / sizeof(T);
  constexpr int THREADS = 256;
  constexpr int NCHUNK = D / (32 * VEC);

  typedef float U;

  extern __shared__ char smem_raw[];
  T* Ks = reinterpret_cast<T*>(smem_raw);
  T* Vs = Ks + KB * KROW;
  U* Qs = reinterpret_cast<U*>(Vs + KB * KROW);
  U* Ss = Qs + QT * D;

  const int tid = threadIdx.x;
  const int lane_idx = tid % 32;
  const int warp_idx = tid / 32;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.x;
  const int kv_head_idx = head_idx / params.gqa_factor;
  const int q0 = blockIdx.y * QT;
  const int j = warp_idx;
  const int q_row = min(q0 + j, params.qL - 1);

  const U scale_log2 = params.scale * M_LOG2E;

  const T* Qg =
      Q + batch_idx * params.Q_strides[0] + head_idx * params.Q_strides[1];
  const T* Kg =
      K + batch_idx * params.K_strides[0] + kv_head_idx * params.K_strides[1];
  const T* Vg =
      V + batch_idx * params.V_strides[0] + kv_head_idx * params.V_strides[1];
  T* Og = O + batch_idx * params.O_strides[0] + head_idx * params.O_strides[1];

  const T* Mg = nullptr;
  if (M) {
    Mg = M + batch_idx * params.M_strides[0] +
        (params.maskH == 1 ? 0 : head_idx * params.M_strides[1]);
  }

  // Load and pre-scale the Q tile into fp32 shared memory.
  for (int idx = tid; idx < QT * (D / VEC); idx += THREADS) {
    const int r = idx / (D / VEC);
    const int c = (idx % (D / VEC)) * VEC;
    const int qr = min(q0 + r, params.qL - 1);
    auto qv = load_vector<VEC>(Qg + qr * params.Q_strides[2] + c, 0);
    PRAGMA_LOOP_UNROLL
    for (int m = 0; m < VEC; ++m) {
      Qs[r * D + c + m] = scale_log2 * static_cast<U>(qv[m]);
    }
  }

  // Per-query online softmax state (uniform across the owning warp).
  U max_score = Limits<U>::finite_min();
  U sum_exp = 0.f;
  if (sinks) {
    max_score = M_LOG2E * static_cast<U>(sinks[head_idx]);
    sum_exp = 1.f;
  }

  U acc[NCHUNK][VEC];
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < NCHUNK; ++i) {
    PRAGMA_LOOP_UNROLL
    for (int m = 0; m < VEC; ++m) {
      acc[i][m] = 0.f;
    }
  }

  for (int kv0 = 0; kv0 < params.kL; kv0 += KB) {
    // Use-key tile: combine sequence bounds, causal bounds, and additive mask
    // into Ss (raw bias for live pairs, -inf for dead ones). Sliding-window
    // masks and causal bounds zero out entire tiles for most (query, kv0)
    // combinations; vote and skip those before paying the K/V tile loads.
    for (int idx = tid; idx < QT * KB; idx += THREADS) {
      const int rj = idx / KB;
      const int rkv = idx % KB;
      const int kv_glob = kv0 + rkv;
      const int qr = min(q0 + rj, params.qL - 1);
      U bias = -INFINITY;
      bool use_key = kv_glob < params.kL;
      if constexpr (do_causal) {
        use_key = use_key && (kv_glob <= (params.kL - params.qL + qr));
      }
      if (use_key && Mg) {
        bias =
            static_cast<U>(
                Mg[qr * params.M_strides[2] + kv_glob * params.M_strides[3]]) *
            M_LOG2E;
        use_key = bias >= Limits<U>::finite_min() || isnan(bias);
        if (!use_key) {
          bias = -INFINITY;
        }
      } else if (use_key) {
        bias = 0.f;
      }
      Ss[idx] = bias;
    }
    block.sync();
    int any_live = 0;
    for (int idx = tid; idx < QT * KB; idx += THREADS) {
      any_live |= (Ss[idx] != -INFINITY) ? 1 : 0;
    }
    if (__syncthreads_or(any_live) == 0) {
      continue;
    }

    // Cooperative load of the K and V tiles. Stage every vector through
    // registers first so the compiler issues all global reads back to back;
    // interleaved load/store chains expose full unified-memory latency per
    // load otherwise and dominate the kernel time.
    constexpr int LITERS = KB * (D / VEC) / THREADS;
    AlignedVector<T, VEC> k_reg[LITERS];
    AlignedVector<T, VEC> v_reg[LITERS];
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < LITERS; ++i) {
      const int idx = tid + i * THREADS;
      const int r = idx / (D / VEC);
      const int c = (idx % (D / VEC)) * VEC;
      const int kr = min(kv0 + r, params.kL - 1);
      k_reg[i] = load_vector<VEC>(Kg + kr * params.K_strides[2] + c, 0);
      v_reg[i] = load_vector<VEC>(Vg + kr * params.V_strides[2] + c, 0);
    }
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < LITERS; ++i) {
      const int idx = tid + i * THREADS;
      const int r = idx / (D / VEC);
      const int c = (idx % (D / VEC)) * VEC;
      *reinterpret_cast<AlignedVector<T, VEC>*>(Ks + r * KROW + c) = k_reg[i];
      *reinterpret_cast<AlignedVector<T, VEC>*>(Vs + r * KROW + c) = v_reg[i];
    }
    block.sync();

    // Scores: warp j computes dot(q[j], k[lane]); all lanes independent.
    // Lanes at or beyond KB have no tile column; keep them out of the shared
    // score/bias region (KB can be smaller than the warp width).
    U my_score = -INFINITY;
    const int kv_glob = kv0 + lane_idx;
    if (lane_idx < KB) {
      const U bias = Ss[j * KB + lane_idx];
      if (bias != -INFINITY) {
        const T* krow = &Ks[lane_idx * KROW];
        const U* qrow = &Qs[j * D];
        U score = 0.f;
        PRAGMA_LOOP_UNROLL
        for (int d = 0; d < D; d += VEC) {
          auto k_vec = unsafe_load_vector<VEC>(krow + d, 0);
          auto q_vec = unsafe_load_vector<VEC>(qrow + d, 0);
          PRAGMA_LOOP_UNROLL
          for (int m = 0; m < VEC; ++m) {
            score += q_vec[m] * static_cast<U>(k_vec[m]);
          }
        }
        my_score = score + bias;
      }
    }
    if (lane_idx < KB) {
      Ss[j * KB + lane_idx] = my_score;
    }

    // Online softmax update for this tile (uniform across the warp).
    U tile_max = cg::reduce(warp, my_score, cg::greater<U>());
    U new_max = max(max_score, tile_max);
    U factor = exp2f(max_score - new_max);
    // Masked lanes hold -inf; with a finite running max exp2f(-inf) = 0.
    U e = (my_score == -INFINITY) ? 0.f : exp2f(my_score - new_max);
    U tile_sum = cg::reduce(warp, e, cg::plus<U>());
    sum_exp = sum_exp * factor + tile_sum;
    max_score = new_max;
    if (lane_idx < KB) {
      Ss[j * KB + lane_idx] = e;
    }
    warp.sync();

    // PV accumulation: lanes cover D in VEC-wide slices per warp chunk.
    if (tile_sum > 0.f) {
      PRAGMA_LOOP_UNROLL
      for (int i = 0; i < NCHUNK; ++i) {
        PRAGMA_LOOP_UNROLL
        for (int m = 0; m < VEC; ++m) {
          acc[i][m] *= factor;
        }
      }

      for (int kv = 0; kv < KB && kv0 + kv < params.kL; ++kv) {
        const U w = Ss[j * KB + kv];
        if (w == 0.f) {
          continue;
        }
        const T* vrow = &Vs[kv * KROW];
        PRAGMA_LOOP_UNROLL
        for (int i = 0; i < NCHUNK; ++i) {
          auto v_vec =
              unsafe_load_vector<VEC>(vrow + (i * 32 + lane_idx) * VEC, 0);
          PRAGMA_LOOP_UNROLL
          for (int m = 0; m < VEC; ++m) {
            acc[i][m] += w * static_cast<U>(v_vec[m]);
          }
        }
      }
    }
    block.sync();
  }

  // Write the output row.
  if (q0 + j < params.qL) {
    const U inv = (sum_exp == 0.f) ? 0.f : __frcp_rn(sum_exp);
    T* orow = Og + (q0 + j) * params.O_strides[2];
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < NCHUNK; ++i) {
      AlignedVector<T, VEC> out;
      PRAGMA_LOOP_UNROLL
      for (int m = 0; m < VEC; ++m) {
        out[m] = static_cast<T>(acc[i][m] * inv);
      }
      *reinterpret_cast<AlignedVector<T, VEC>*>(
          orow + (i * 32 + lane_idx) * VEC) = out;
    }
  }
}

// Tensor-core (mma.sync m16n8k16 bf16/f16) query-tiled flash attention for
// wide head dims at prefill. Shares the tile architecture of
// kernel_sdpav_fvec (QT query rows per block, K/V through shared memory,
// whole-tile skip on fully masked tiles) but computes scores and the PV
// accumulation with tensor cores instead of CUDA-core dot products.
// Fragment idioms adapted from llama.cpp ggml-cuda/mma.cuh (MIT).
namespace fmma_detail {

struct FragA {
  uint32_t r[4];
};
struct FragB {
  uint32_t r[2];
};
struct FragC {
  float r[4];
};

template <typename T>
__device__ __forceinline__ void mma(FragC& c, const FragA& a, const FragB& b);

template <>
__device__ __forceinline__ void
mma<__nv_bfloat16>(FragC& c, const FragA& a, const FragB& b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c.r[0]), "+f"(c.r[1]), "+f"(c.r[2]), "+f"(c.r[3])
      : "r"(a.r[0]),
        "r"(a.r[1]),
        "r"(a.r[2]),
        "r"(a.r[3]),
        "r"(b.r[0]),
        "r"(b.r[1]));
}

template <>
__device__ __forceinline__ void
mma<__half>(FragC& c, const FragA& a, const FragB& b) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c.r[0]), "+f"(c.r[1]), "+f"(c.r[2]), "+f"(c.r[3])
      : "r"(a.r[0]),
        "r"(a.r[1]),
        "r"(a.r[2]),
        "r"(a.r[3]),
        "r"(b.r[0]),
        "r"(b.r[1]));
}

// Load an A fragment (16x16 tile of T, row-major, ldmatrix x4 non-trans).
// Lanes 0-15 supply rows 0-15 at column block 0, lanes 16-31 rows 0-15 at
// column block 1 (+8 halves).
template <typename T>
__device__ __forceinline__ void
ldsm_A(FragA& a, const T* base, int row_stride) {
  // ldmatrix lane addressing must use the warp-local lane, not the raw
  // thread index: ldmatrix x4 rows come from lanes 0-15 (rows 0-15) and
  // lanes 16-31 (rows 0-15 at the +8-halves column block).
  const int lane = threadIdx.x % 32;
  const char* p = reinterpret_cast<const char*>(
      base + (lane % 16) * row_stride + (lane / 16) * 8);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a.r[0]), "=r"(a.r[1]), "=r"(a.r[2]), "=r"(a.r[3])
               : "l"(p)
               : "memory");
}

// B fragment (16k x 8n) for the scores mma, loaded from the K tile.
// Empirically pinned on sm_121 (unit probe .tmp/mma_unit.cu): lane l holds
// B[k][n], B[k+1][n] and B[k+8][n], B[k+9][n] with n = kv0 + l/4 and
// k = d0 + (l%4)*2. Scores need B[k=d][n=kv] = K[kv][d].
template <typename T>
__device__ __forceinline__ void
load_B_rows(FragB& b, const T* rows, int kv0, int d0, int row_stride) {
  const int lane = threadIdx.x % 32;
  const int n = kv0 + lane / 4;
  const int k = d0 + (lane % 4) * 2;
  uint32_t lo0, hi0, lo1, hi1;
  memcpy(&lo0, rows + (n)*row_stride + k, 2);
  memcpy(&hi0, rows + (n)*row_stride + k + 1, 2);
  b.r[0] = lo0 | (hi0 << 16);
  memcpy(&lo1, rows + (n)*row_stride + k + 8, 2);
  memcpy(&hi1, rows + (n)*row_stride + k + 9, 2);
  b.r[1] = lo1 | (hi1 << 16);
}

// B fragment for PV: k = 16 consecutive kv rows, n = 8 columns of V.
// Lane l: n = (d0) + l/4 offset column, k = kv0 + (l%4)*2 rows.
template <typename T>
__device__ __forceinline__ void
load_V_trans(FragB& b, const T* rows, int kv0, int d0, int row_stride) {
  const int lane = threadIdx.x % 32;
  const int n = d0 + lane / 4;
  const int k = kv0 + (lane % 4) * 2;
  uint32_t lo0, hi0, lo1, hi1;
  memcpy(&lo0, rows + (k + 0) * row_stride + n, 2);
  memcpy(&hi0, rows + (k + 1) * row_stride + n, 2);
  b.r[0] = lo0 | (hi0 << 16);
  memcpy(&lo1, rows + (k + 8) * row_stride + n, 2);
  memcpy(&hi1, rows + (k + 9) * row_stride + n, 2);
  b.r[1] = lo1 | (hi1 << 16);
}

__device__ __forceinline__ int frag_row(int lane, int half) {
  return lane / 4 + half * 8;
}
__device__ __forceinline__ int frag_col(int lane, int off) {
  return (lane % 4) * 2 + off;
}

} // namespace fmma_detail

template <typename T, bool do_causal, int D>
__global__ void kernel_sdpav_fmma(
    const T* __restrict__ Q,
    const T* __restrict__ K,
    const T* __restrict__ V,
    T* __restrict__ O,
    const T* __restrict__ M,
    const T* __restrict__ sinks,
    __grid_constant__ const AttnParams params,
    float* __restrict__ dbg_out) {
  using namespace fmma_detail;
  constexpr int QT = 16;
  constexpr int KB = D == 256 ? 32 : 16;
  constexpr int KRP = D + 8;
  constexpr int PRP = KB + 8;
  constexpr int SRP = KB + 1;
  constexpr int VEC = 16 / sizeof(T);
  constexpr int THREADS = 256;
  constexpr int NB = KB / 8;
  constexpr int DSLICE = D / 8;
  constexpr int NFRAG = DSLICE / 8;

  typedef float U;

  extern __shared__ char smem_raw[];
  T* sQ = reinterpret_cast<T*>(smem_raw);
  T* sK = sQ + QT * KRP;
  T* sV = sK + KB * KRP;
  T* sP = sV + KB * KRP;
  U* sS = reinterpret_cast<U*>(sP + QT * PRP);
  U* sB = sS + QT * SRP;
  U* sF = sB + QT * SRP;
  U* sL = sF + QT;

  // 2-D block (32, 8): threadIdx.x is always the warp lane so the ldmatrix
  // fragment addressing (which is lane-based) works verbatim.
  const int lane_idx = threadIdx.x;
  const int warp_idx = threadIdx.y;
  const int tid = threadIdx.x + threadIdx.y * 32;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.x;
  const int kv_head_idx = head_idx / params.gqa_factor;
  const int q0 = blockIdx.y * QT;
  const int q_row = min(q0 + warp_idx, params.qL - 1);

  const U scale_log2 = params.scale * M_LOG2E;

  const T* Qg =
      Q + batch_idx * params.Q_strides[0] + head_idx * params.Q_strides[1];
  const T* Kg =
      K + batch_idx * params.K_strides[0] + kv_head_idx * params.K_strides[1];
  const T* Vg =
      V + batch_idx * params.V_strides[0] + kv_head_idx * params.V_strides[1];
  T* Og = O + batch_idx * params.O_strides[0] + head_idx * params.O_strides[1];

  const T* Mg = nullptr;
  if (M) {
    Mg = M + batch_idx * params.M_strides[0] +
        (params.maskH == 1 ? 0 : head_idx * params.M_strides[1]);
  }

  // Load the Q tile. Scores are scaled after the mma in fp32, so Q stays in
  // its native dtype here.
  constexpr int QLIT = QT * (D / VEC) / THREADS;
  AlignedVector<T, VEC> q_reg[QLIT];
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < QLIT; ++i) {
    const int idx = tid + i * THREADS;
    const int r = idx / (D / VEC);
    const int c = (idx % (D / VEC)) * VEC;
    const int qr = min(q0 + r, params.qL - 1);
    q_reg[i] = load_vector<VEC>(Qg + qr * params.Q_strides[2] + c, 0);
  }
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < QLIT; ++i) {
    const int idx = tid + i * THREADS;
    const int r = idx / (D / VEC);
    const int c = (idx % (D / VEC)) * VEC;
    *reinterpret_cast<AlignedVector<T, VEC>*>(sQ + r * KRP + c) = q_reg[i];
  }

  // Per-query-row online softmax state, owned by the softmax warps
  // (warp w owns rows w and w+8).
  U max_score_r0 = Limits<U>::finite_min();
  U max_score_r1 = Limits<U>::finite_min();
  U sum_exp_r0 = 0.f;
  U sum_exp_r1 = 0.f;
  if (sinks) {
    const U sink = M_LOG2E * static_cast<U>(sinks[head_idx]);
    max_score_r0 = sink;
    max_score_r1 = sink;
    sum_exp_r0 = 1.f;
    sum_exp_r1 = 1.f;
  }

  // PV accumulators: each warp owns a D-slice of the output.
  FragC o_acc[NFRAG];
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < NFRAG; ++i) {
    o_acc[i] = {0.f, 0.f, 0.f, 0.f};
  }

  for (int kv0 = 0; kv0 < params.kL; kv0 += KB) {
    // Use-key verdict tile (sequence bounds + causal + additive mask).
    for (int idx = tid; idx < QT * KB; idx += THREADS) {
      const int rj = idx / KB;
      const int rkv = idx % KB;
      const int kv_glob = kv0 + rkv;
      const int qr = min(q0 + rj, params.qL - 1);
      U bias = -INFINITY;
      bool use_key = kv_glob < params.kL;
      if constexpr (do_causal) {
        use_key = use_key && (kv_glob <= (params.kL - params.qL + qr));
      }
      if (use_key && Mg) {
        bias =
            static_cast<U>(
                Mg[qr * params.M_strides[2] + kv_glob * params.M_strides[3]]) *
            M_LOG2E;
        use_key = bias >= Limits<U>::finite_min() || isnan(bias);
        if (!use_key) {
          bias = -INFINITY;
        }
      } else if (use_key) {
        bias = 0.f;
      }
      sB[rj * SRP + rkv] = bias;
    }
    block.sync();

    // Skip tiles where every query/key pair is masked (sliding-window and
    // above-diagonal causal tiles). __syncthreads_or is the barrier too, so
    // no shared flag is needed across iterations.
    int any_live = 0;
    for (int idx = tid; idx < QT * KB; idx += THREADS) {
      const int rj = idx / KB;
      const int rkv = idx % KB;
      any_live |= (sB[rj * SRP + rkv] != -INFINITY) ? 1 : 0;
    }
    if (__syncthreads_or(any_live) == 0) {
      continue;
    }

    // Cooperative load of the K and V tiles (register staged).
    constexpr int LITERS = KB * (D / VEC) / THREADS;
    AlignedVector<T, VEC> k_reg[LITERS];
    AlignedVector<T, VEC> v_reg[LITERS];
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < LITERS; ++i) {
      const int idx = tid + i * THREADS;
      const int r = idx / (D / VEC);
      const int c = (idx % (D / VEC)) * VEC;
      const int kr = min(kv0 + r, params.kL - 1);
      k_reg[i] = load_vector<VEC>(Kg + kr * params.K_strides[2] + c, 0);
      v_reg[i] = load_vector<VEC>(Vg + kr * params.V_strides[2] + c, 0);
    }
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < LITERS; ++i) {
      const int idx = tid + i * THREADS;
      const int r = idx / (D / VEC);
      const int c = (idx % (D / VEC)) * VEC;
      *reinterpret_cast<AlignedVector<T, VEC>*>(sK + r * KRP + c) = k_reg[i];
      *reinterpret_cast<AlignedVector<T, VEC>*>(sV + r * KRP + c) = v_reg[i];
    }
    block.sync();

    // Scores via tensor cores: warp w (< NB) computes the 16 x 8 score slab
    // for kv columns [w*8, w*8+8).
    if (warp_idx < NB) {
      FragC cfrag = {0.f, 0.f, 0.f, 0.f};
      const int n0 = warp_idx * 8;
      PRAGMA_LOOP_UNROLL
      for (int d0 = 0; d0 < D; d0 += 16) {
        FragA a;
        ldsm_A(a, sQ + d0, KRP);
        FragB b;
        load_B_rows(b, sK, n0, d0, KRP);
        mma<T>(cfrag, a, b);
      }
      PRAGMA_LOOP_UNROLL
      for (int h = 0; h < 2; ++h) {
        const int r = frag_row(lane_idx, h);
        const int c0 = frag_col(lane_idx, 0);
        // Store pre-softmax scores scaled to the log2 domain. Masked entries
        // in sB are -inf and dominate the sum.
        sS[r * SRP + n0 + c0] =
            cfrag.r[h * 2 + 0] * scale_log2 + sB[r * SRP + n0 + c0];
        sS[r * SRP + n0 + c0 + 1] =
            cfrag.r[h * 2 + 1] * scale_log2 + sB[r * SRP + n0 + c0 + 1];
      }
    }
    block.sync();

    // Softmax update: warp w owns rows {w, w+8}; lanes span the KB columns.
    {
      const int row0 = warp_idx;
      const int row1 = warp_idx + 8;
      U x0 = sS[row0 * SRP + lane_idx];
      U x1 = sS[row1 * SRP + lane_idx];
      if (lane_idx >= KB) {
        x0 = -INFINITY;
        x1 = -INFINITY;
      }
      U tile_max0 = cg::reduce(warp, x0, cg::greater<U>());
      U tile_max1 = cg::reduce(warp, x1, cg::greater<U>());
      U new_max0 = max(max_score_r0, tile_max0);
      U new_max1 = max(max_score_r1, tile_max1);
      const U factor0 = exp2f(max_score_r0 - new_max0);
      const U factor1 = exp2f(max_score_r1 - new_max1);
      const U e0 = (x0 == -INFINITY || new_max0 == Limits<U>::finite_min())
          ? 0.f
          : exp2f(x0 - new_max0);
      const U e1 = (x1 == -INFINITY || new_max1 == Limits<U>::finite_min())
          ? 0.f
          : exp2f(x1 - new_max1);
      U tile_sum0 = cg::reduce(warp, e0, cg::plus<U>());
      U tile_sum1 = cg::reduce(warp, e1, cg::plus<U>());
      sum_exp_r0 = sum_exp_r0 * factor0 + tile_sum0;
      sum_exp_r1 = sum_exp_r1 * factor1 + tile_sum1;
      max_score_r0 = new_max0;
      max_score_r1 = new_max1;
      // Publish exp weights as bf16 A fragments and row rescale factors.
      if (lane_idx < KB) {
        sP[row0 * PRP + lane_idx] = static_cast<T>(e0);
        sP[row1 * PRP + lane_idx] = static_cast<T>(e1);
      }
      if (lane_idx == 0) {
        sF[row0] = factor0;
        sF[row1] = factor1;
        sL[row0] = sum_exp_r0;
        sL[row1] = sum_exp_r1;
      }
    }
    block.sync();

    // PV accumulation via tensor cores: each warp owns a D-slice.
    {
      const int dslice0 = warp_idx * DSLICE;
      // Rescale by each fragment row's factor.
      const U f0 = sF[frag_row(lane_idx, 0)];
      const U f1 = sF[frag_row(lane_idx, 1)];
      PRAGMA_LOOP_UNROLL
      for (int i = 0; i < NFRAG; ++i) {
        o_acc[i].r[0] *= f0;
        o_acc[i].r[1] *= f0;
        o_acc[i].r[2] *= f1;
        o_acc[i].r[3] *= f1;
      }
      for (int k0 = 0; k0 < KB; k0 += 16) {
        FragA pa;
        ldsm_A(pa, sP + k0, PRP);
        PRAGMA_LOOP_UNROLL
        for (int i = 0; i < NFRAG; ++i) {
          FragB vb;
          load_V_trans(vb, sV, k0, dslice0 + i * 8, KRP);
          // Forensics hook (always null in production): the d512
          // instantiation mis-compiles without this store's scheduling
          // effect; root cause open (see experiment journal 2026-08-06).
          if (dbg_out != nullptr && blockIdx.x == 0 && blockIdx.z == 0 &&
              tid < 8 && i == 0 && k0 == 0 && kv0 == 0) {
            for (int t = 0; t < 4; ++t) {
              dbg_out[tid * 8 + t * 2 + 0] = static_cast<float>(
                  __ushort_as_bfloat16((unsigned short)(pa.r[t] & 0xffff)));
              dbg_out[tid * 8 + t * 2 + 1] = static_cast<float>(
                  __ushort_as_bfloat16((unsigned short)(pa.r[t] >> 16)));
            }
          }
          mma<T>(o_acc[i], pa, vb);
        }
      }
    }
    block.sync();
  }

  // Write output rows from the PV D-slice fragments.
  {
    const int dslice0 = warp_idx * DSLICE;
    const U inv0 = sL[frag_row(lane_idx, 0)] == 0.f
        ? 0.f
        : __frcp_rn(sL[frag_row(lane_idx, 0)]);
    const U inv1 = sL[frag_row(lane_idx, 1)] == 0.f
        ? 0.f
        : __frcp_rn(sL[frag_row(lane_idx, 1)]);
    PRAGMA_LOOP_UNROLL
    for (int h = 0; h < 2; ++h) {
      const int r = frag_row(lane_idx, h);
      if (q0 + r >= params.qL) {
        continue;
      }
      const U inv = h == 0 ? inv0 : inv1;
      PRAGMA_LOOP_UNROLL
      for (int i = 0; i < NFRAG; ++i) {
        const int c0 = dslice0 + i * 8 + frag_col(lane_idx, 0);
        AlignedVector<T, 2> out;
        out[0] = static_cast<T>(o_acc[i].r[h * 2 + 0] * inv);
        out[1] = static_cast<T>(o_acc[i].r[h * 2 + 1] * inv);
        *reinterpret_cast<AlignedVector<T, 2>*>(
            Og + (q0 + r) * params.O_strides[2] + c0) = out;
      }
    }
  }
}

template <typename T, bool do_causal, int D>
__global__ void kernel_sdpav_2pass_1(
    const T* Q,
    const T* K,
    const T* V,
    const T* M,
    const T* sinks,
    float* partials,
    float* sums,
    float* maxs,
    __grid_constant__ const AttnParams params) {
  constexpr int BN = 8;
  constexpr int BD = 32;
  constexpr int blocks = 32;

  constexpr int v_per_thread = D / BD;

  const int inner_k_stride = blocks * BN * int(params.K_strides[2]);
  const int inner_v_stride = blocks * BN * int(params.V_strides[2]);

  typedef float U;

  U q[v_per_thread];
  U k[v_per_thread];
  U o[v_per_thread];

  __shared__ U outputs[BN][BD + 1];
  __shared__ U max_scores[BN];
  __shared__ U sum_exp_scores[BN];

  const U scale_log2 = params.scale * 1.44269504089f;

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const int lane_idx = warp.thread_rank();
  const int warp_idx = warp.meta_group_rank();

  // Adjust to thread block and thread
  const int batch_idx = blockIdx.z / blocks;
  const int block_idx = blockIdx.z % blocks;
  const int head_idx = blockIdx.x;
  const int kv_head_idx = head_idx / params.gqa_factor;

  const int q_seq_idx = blockIdx.y;
  const int kv_seq_idx = block_idx * BN + warp_idx;

  Q += batch_idx * params.Q_strides[0] + // Batch
      head_idx * params.Q_strides[1] + // Head
      q_seq_idx * params.Q_strides[2]; // Sequence

  K += batch_idx * params.K_strides[0] + // Batch
      kv_head_idx * params.K_strides[1] + // Head
      kv_seq_idx * params.K_strides[2]; // Sequence

  V += batch_idx * params.V_strides[0] + // Batch
      kv_head_idx * params.V_strides[1] + // Head
      kv_seq_idx * params.V_strides[2]; // Sequence

  if (M) {
    M += batch_idx * params.M_strides[0] + // Batch
        (params.maskH == 1 ? 0 : head_idx * params.M_strides[1]) + // Head
        q_seq_idx * params.M_strides[2]; // Sequence
  }

  const int p_stride_s = blocks;
  const int p_stride_h = params.qL * p_stride_s;
  const int p_stride_b = params.H * p_stride_h;
  const int p_offset = batch_idx * p_stride_b + // Batch
      head_idx * p_stride_h + // Head
      q_seq_idx * p_stride_s + // Sequence
      block_idx; // Block

  partials += p_offset * D;
  sums += p_offset;
  maxs += p_offset;

  // Read the query and 0 the output accumulator
  if constexpr (sizeof(T) == 2 && (D == 256 || D == 512)) {
    auto q_vec =
        unsafe_load_vector<v_per_thread>(Q + v_per_thread * lane_idx, 0);
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      q[i] = scale_log2 * static_cast<U>(q_vec[i]);
    }
  } else {
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      q[i] = scale_log2 * static_cast<U>(Q[v_per_thread * lane_idx + i]);
    }
  }

  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    o[i] = 0.f;
  }

  U max_score = Limits<U>::finite_min();
  U sum_exp_score = 0.f;
  if (sinks && warp_idx == 0 && block_idx == 0) {
    max_score = M_LOG2E * static_cast<U>(sinks[head_idx]);
    sum_exp_score = 1.f;
  }

  // For each key
  for (int i = kv_seq_idx; i < params.kL; i += blocks * BN) {
    bool use_key = true;
    if constexpr (do_causal) {
      use_key = i <= (params.kL - params.qL + q_seq_idx);
    }

    U bias = 0.f;
    if (M) {
      bias = static_cast<U>(M[i * params.M_strides[3]]) * M_LOG2E;
      use_key = use_key && (bias >= Limits<U>::finite_min() || isnan(bias));
    }
    if (use_key) {
      // Read the key
      if constexpr (sizeof(T) == 2 && (D == 256 || D == 512)) {
        auto k_vec =
            unsafe_load_vector<v_per_thread>(K + v_per_thread * lane_idx, 0);
        PRAGMA_LOOP_UNROLL
        for (int j = 0; j < v_per_thread; j++) {
          k[j] = k_vec[j];
        }
      } else {
        PRAGMA_LOOP_UNROLL
        for (int j = 0; j < v_per_thread; j++) {
          k[j] = K[v_per_thread * lane_idx + j];
        }
      }

      // Compute the i-th score
      U score = 0.f;
      PRAGMA_LOOP_UNROLL
      for (int j = 0; j < v_per_thread; j++) {
        score += q[j] * k[j];
      }

      // Warp sum
      score = cg::reduce(warp, score, cg::plus<U>());

      // Update the accumulators
      score += bias;
      U new_max = max(max_score, score);
      U factor = exp2f(max_score - new_max);
      U exp_score = exp2f(score - new_max);

      max_score = new_max;
      sum_exp_score = sum_exp_score * factor + exp_score;

      // Update the output accumulator
      if constexpr (sizeof(T) == 2 && (D == 256 || D == 512)) {
        auto v_vec =
            unsafe_load_vector<v_per_thread>(V + v_per_thread * lane_idx, 0);
        PRAGMA_LOOP_UNROLL
        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor + exp_score * static_cast<U>(v_vec[j]);
        }
      } else {
        PRAGMA_LOOP_UNROLL
        for (int j = 0; j < v_per_thread; j++) {
          o[j] = o[j] * factor +
              exp_score * static_cast<U>(V[v_per_thread * lane_idx + j]);
        }
      }
    }

    // Move the pointers to the next kv
    K += inner_k_stride;
    V += inner_v_stride;
  }

  if (lane_idx == 0) {
    max_scores[warp_idx] = max_score;
    sum_exp_scores[warp_idx] = sum_exp_score;
  }

  block.sync();

  max_score = (lane_idx < BN) ? max_scores[lane_idx] : -1e9;
  U new_max = cg::reduce(warp, max_score, cg::greater<U>());
  U factor = exp2f(max_score - new_max);
  sum_exp_score = (lane_idx < BN) ? sum_exp_scores[lane_idx] : 0.f;
  sum_exp_score = cg::reduce(warp, sum_exp_score * factor, cg::plus<U>());

  // Write the sum and new max
  if (warp_idx == 0) {
    sums[0] = sum_exp_score;
    maxs[0] = new_max;
  }

  // Now we need to aggregate all the outputs
  auto ff = exp2f(max_scores[warp_idx] - new_max);
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    outputs[warp_idx][lane_idx] = o[i] * ff;
    block.sync();

    if (warp_idx == 0) {
      U ot = outputs[0][lane_idx];
      PRAGMA_LOOP_UNROLL
      for (int j = 1; j < BN; j++) {
        ot += outputs[j][lane_idx];
        warp.sync();
      }
      o[i] = ot;
    }
    block.sync();
  }

  if (warp_idx == 0) {
    if constexpr (D == 256 || D == 512) {
      AlignedVector<U, v_per_thread> o_vec;
      PRAGMA_LOOP_UNROLL
      for (int i = 0; i < v_per_thread; i++) {
        o_vec[i] = o[i];
      }
      unsafe_store_vector<v_per_thread>(
          partials + v_per_thread * lane_idx, 0, o_vec);
    } else {
      PRAGMA_LOOP_UNROLL
      for (int i = 0; i < v_per_thread; i++) {
        partials[v_per_thread * lane_idx + i] = o[i];
      }
    }
  }
}

template <typename T, bool do_causal, int D>
__global__ void kernel_sdpav_2pass_2(
    const float* partials,
    const float* sums,
    const float* maxs,
    T* O,
    __grid_constant__ const AttnParams params) {
  constexpr int BN = 32;
  constexpr int BD = 32;
  constexpr int blocks = 32;

  constexpr int v_per_thread = D / BD;

  typedef float U;

  U o[v_per_thread];
  __shared__ U outputs[BN][BD + 1];

  auto block = cg::this_thread_block();
  auto warp = cg::tiled_partition<32>(block);

  const int lane_idx = warp.thread_rank();
  const int warp_idx = warp.meta_group_rank();

  // Adjust to thread block and thread
  const int batch_idx = blockIdx.z;
  const int head_idx = blockIdx.x;
  const int q_seq_idx = blockIdx.y;

  const int p_stride_s = blocks;
  const int p_stride_h = params.qL * p_stride_s;
  const int p_stride_b = params.H * p_stride_h;
  const int p_offset = batch_idx * p_stride_b + // Batch
      head_idx * p_stride_h + // Head
      q_seq_idx * p_stride_s; // Sequence

  partials += p_offset * D + warp_idx * D;
  sums += p_offset;
  maxs += p_offset;

  O += batch_idx * params.O_strides[0] + // Batch
      head_idx * params.O_strides[1] + // Head
      q_seq_idx * params.O_strides[2]; // Sequence

  U max_score = maxs[lane_idx];
  U new_max = cg::reduce(warp, max_score, cg::greater<U>());
  U factor = exp2f(max_score - new_max);
  U sum_exp_score = cg::reduce(warp, sums[lane_idx] * factor, cg::plus<U>());
  sum_exp_score = sum_exp_score == 0 ? 0 : __frcp_rn(sum_exp_score);

  if constexpr (D == 256 || D == 512) {
    auto o_vec =
        unsafe_load_vector<v_per_thread>(partials + v_per_thread * lane_idx, 0);
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      o[i] = o_vec[i];
    }
  } else {
    PRAGMA_LOOP_UNROLL
    for (int i = 0; i < v_per_thread; i++) {
      o[i] = partials[v_per_thread * lane_idx + i];
    }
  }

  // Now we need to aggregate all the outputs
  PRAGMA_LOOP_UNROLL
  for (int i = 0; i < v_per_thread; i++) {
    outputs[lane_idx][warp_idx] = o[i];
    block.sync();
    U ot = outputs[warp_idx][lane_idx] * factor;
    o[i] = cg::reduce(warp, ot, cg::plus<U>()) * sum_exp_score;
    block.sync();
  }

  // And write the output
  if (lane_idx == 0) {
    if constexpr (sizeof(T) == 2 && (D == 256 || D == 512)) {
      AlignedVector<T, v_per_thread> o_vec;
      PRAGMA_LOOP_UNROLL
      for (int i = 0; i < v_per_thread; i++) {
        o_vec[i] = static_cast<T>(o[i]);
      }
      unsafe_store_vector<v_per_thread>(O + v_per_thread * warp_idx, 0, o_vec);
    } else {
      PRAGMA_LOOP_UNROLL
      for (int i = 0; i < v_per_thread; i++) {
        O[v_per_thread * warp_idx + i] = static_cast<T>(o[i]);
      }
    }
  }
}

} // namespace cu

namespace {

template <typename F>
void dispatch_headdim(int n, F&& f) {
  switch (n) {
    case 64:
      f(std::integral_constant<int, 64>{});
      break;
    case 96:
      f(std::integral_constant<int, 96>{});
      break;
    case 128:
      f(std::integral_constant<int, 128>{});
      break;
    case 256:
      f(std::integral_constant<int, 256>{});
      break;
    case 512:
      f(std::integral_constant<int, 512>{});
      break;
  }
}

void sdpa_vector_1pass_fallback(
    const Stream& s,
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const std::optional<array>& sinks) {
  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  if (sinks) {
    encoder.set_input_array(*sinks);
  }
  if (mask_arr) {
    encoder.set_input_array(*mask_arr);
  }
  encoder.set_output_array(o);

  cu::AttnParams params{
      /* int B = */ q.shape(0),
      /* int H = */ q.shape(1),
      /* int D = */ q.shape(3),

      /* int qL = */ q.shape(2),
      /* int kL = */ k.shape(2),

      /* int gqa_factor = */ q.shape(1) / k.shape(1),
      /* int maskH = */ mask_arr ? mask_arr->shape(1) : 0,
      /* float scale = */ scale,

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)},
      /* int64_t M_strides[4] = */ {0, 0, 0, 0}};

  if (mask_arr) {
    params.M_strides[0] = mask_arr->strides(0);
    params.M_strides[1] = mask_arr->strides(1);
    params.M_strides[2] = mask_arr->strides(2);
    params.M_strides[3] = mask_arr->strides(3);
  }

  dim3 grid_dim(params.H, params.qL, params.B);
  dim3 block_dim(1024, 1, 1);

  dispatch_float_types(o.dtype(), "kernel_sdpav_1pass", [&](auto type_tag) {
    dispatch_bool(do_causal, [&](auto do_causal) {
      dispatch_headdim(params.D, [&](auto headdim) {
        using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

        auto kernel =
            cu::kernel_sdpav_1pass<DataType, do_causal.value, headdim.value>;
        encoder.add_kernel_node(
            kernel,
            grid_dim,
            block_dim,
            gpu_ptr<DataType>(q),
            gpu_ptr<DataType>(k),
            gpu_ptr<DataType>(v),
            gpu_ptr<DataType>(o),
            mask_arr ? gpu_ptr<DataType>(*mask_arr) : nullptr,
            sinks ? gpu_ptr<DataType>(*sinks) : nullptr,
            params);
      });
    });
  });
}

void sdpa_vector_2pass_fallback(
    const Stream& s,
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const std::optional<array>& sinks) {
  cu::AttnParams params{
      /* int B = */ q.shape(0),
      /* int H = */ q.shape(1),
      /* int D = */ q.shape(3),

      /* int qL = */ q.shape(2),
      /* int kL = */ k.shape(2),

      /* int gqa_factor = */ q.shape(1) / k.shape(1),
      /* int maskH = */ mask_arr ? mask_arr->shape(1) : 0,
      /* float scale = */ scale,

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)},
      /* int64_t M_strides[4] = */ {0, 0, 0, 0}};

  if (mask_arr) {
    params.M_strides[0] = mask_arr->strides(0);
    params.M_strides[1] = mask_arr->strides(1);
    params.M_strides[2] = mask_arr->strides(2);
    params.M_strides[3] = mask_arr->strides(3);
  }

  // Allocate the intermediates
  int blocks = 32;

  Shape intermediate_shape;
  intermediate_shape.reserve(o.ndim() + 1);
  intermediate_shape.insert(
      intermediate_shape.end(), o.shape().begin(), o.shape().end() - 1);
  intermediate_shape.push_back(blocks);
  intermediate_shape.push_back(o.shape().back());

  array intermediate(intermediate_shape, float32, nullptr, {});
  intermediate_shape.pop_back();
  array sums(intermediate_shape, float32, nullptr, {});
  array maxs(std::move(intermediate_shape), float32, nullptr, {});

  intermediate.set_data(cu::malloc_async(intermediate.nbytes(), encoder));
  sums.set_data(cu::malloc_async(sums.nbytes(), encoder));
  maxs.set_data(cu::malloc_async(maxs.nbytes(), encoder));

  encoder.add_temporary(intermediate);
  encoder.add_temporary(sums);
  encoder.add_temporary(maxs);

  dispatch_float_types(o.dtype(), "kernel_sdpav_2pass", [&](auto type_tag) {
    dispatch_bool(do_causal, [&](auto do_causal) {
      dispatch_headdim(params.D, [&](auto headdim) {
        using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

        {
          auto kernel = cu::
              kernel_sdpav_2pass_1<DataType, do_causal.value, headdim.value>;

          encoder.set_input_array(q);
          encoder.set_input_array(k);
          encoder.set_input_array(v);
          if (sinks) {
            encoder.set_input_array(*sinks);
          }
          if (mask_arr) {
            encoder.set_input_array(*mask_arr);
          }

          encoder.set_output_array(intermediate);
          encoder.set_output_array(sums);
          encoder.set_output_array(maxs);

          dim3 grid_dim(params.H, params.qL, params.B * 32);
          dim3 block_dim(8 * 32, 1, 1);

          encoder.add_kernel_node(
              kernel,
              grid_dim,
              block_dim,
              gpu_ptr<DataType>(q),
              gpu_ptr<DataType>(k),
              gpu_ptr<DataType>(v),
              mask_arr ? gpu_ptr<DataType>(*mask_arr) : nullptr,
              sinks ? gpu_ptr<DataType>(*sinks) : nullptr,
              gpu_ptr<float>(intermediate),
              gpu_ptr<float>(sums),
              gpu_ptr<float>(maxs),
              params);
        }

        {
          auto kernel = cu::
              kernel_sdpav_2pass_2<DataType, do_causal.value, headdim.value>;

          encoder.set_input_array(intermediate);
          encoder.set_input_array(sums);
          encoder.set_input_array(maxs);
          encoder.set_output_array(o);

          dim3 grid_dim(params.H, params.qL, params.B);
          dim3 block_dim(1024, 1, 1);

          encoder.add_kernel_node(
              kernel,
              grid_dim,
              block_dim,
              gpu_ptr<float>(intermediate),
              gpu_ptr<float>(sums),
              gpu_ptr<float>(maxs),
              gpu_ptr<DataType>(o),
              params);
        }
      });
    });
  });
}

template <typename DataType, bool do_causal, int D>
void sdpa_vector_fvec_launch(
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    array& o,
    const std::optional<array>& mask_arr,
    const std::optional<array>& sinks,
    cu::AttnParams params) {
  constexpr int QT = 8;
  constexpr int KB = D == 256 ? 32 : 16;
  constexpr int KROW = D + 8;
  constexpr int THREADS = 256;

  const uint32_t smem_bytes = 2 * KB * KROW * sizeof(DataType) +
      QT * D * sizeof(float) + QT * KB * sizeof(float);

  auto kernel = cu::kernel_sdpav_fvec<DataType, do_causal, D>;
  static std::once_flag smem_once;
  std::call_once(smem_once, [&]() {
    auto status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (status != cudaSuccess) {
      throw std::runtime_error(
          "kernel_sdpav_fvec: failed to set shared memory attribute");
    }
  });

  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  if (sinks) {
    encoder.set_input_array(*sinks);
  }
  if (mask_arr) {
    encoder.set_input_array(*mask_arr);
  }
  encoder.set_output_array(o);

  dim3 grid_dim(params.H, (params.qL + QT - 1) / QT, params.B);
  dim3 block_dim(THREADS, 1, 1);

  encoder.add_kernel_node_ex(
      kernel,
      grid_dim,
      block_dim,
      {},
      smem_bytes,
      gpu_ptr<DataType>(q),
      gpu_ptr<DataType>(k),
      gpu_ptr<DataType>(v),
      gpu_ptr<DataType>(o),
      mask_arr ? gpu_ptr<DataType>(*mask_arr) : nullptr,
      sinks ? gpu_ptr<DataType>(*sinks) : nullptr,
      params);
}

template <typename DataType, bool do_causal, int D>
void sdpa_vector_fmma_launch(
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    array& o,
    const std::optional<array>& mask_arr,
    const std::optional<array>& sinks,
    cu::AttnParams params) {
  constexpr int QT = 16;
  constexpr int KB = D == 256 ? 32 : 16;
  constexpr int KRP = D + 8;
  constexpr int PRP = KB + 8;
  constexpr int SRP = KB + 1;
  constexpr int THREADS = 256;

  const uint32_t smem_bytes = (QT + 2 * KB) * KRP * sizeof(DataType) +
      QT * PRP * sizeof(DataType) + (2 * QT * SRP + 2 * QT) * sizeof(float);

  auto kernel = cu::kernel_sdpav_fmma<DataType, do_causal, D>;
  static std::once_flag smem_once;
  std::call_once(smem_once, [&]() {
    auto status = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    if (status != cudaSuccess) {
      throw std::runtime_error(
          "kernel_sdpav_fmma: failed to set shared memory attribute");
    }
  });

  encoder.set_input_array(q);
  encoder.set_input_array(k);
  encoder.set_input_array(v);
  if (sinks) {
    encoder.set_input_array(*sinks);
  }
  if (mask_arr) {
    encoder.set_input_array(*mask_arr);
  }
  encoder.set_output_array(o);

  dim3 grid_dim(params.H, (params.qL + QT - 1) / QT, params.B);
  dim3 block_dim(32, THREADS / 32, 1);

  encoder.add_kernel_node_ex(
      kernel,
      grid_dim,
      block_dim,
      {},
      smem_bytes,
      gpu_ptr<DataType>(q),
      gpu_ptr<DataType>(k),
      gpu_ptr<DataType>(v),
      gpu_ptr<DataType>(o),
      mask_arr ? gpu_ptr<DataType>(*mask_arr) : nullptr,
      sinks ? gpu_ptr<DataType>(*sinks) : nullptr,
      params,
      static_cast<float*>(nullptr));
}

// Build the shared AttnParams used by the tiled prefill attention variants.
cu::AttnParams sdpa_tiled_params(
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    const std::optional<array>& mask_arr) {
  cu::AttnParams params{
      /* int B = */ q.shape(0),
      /* int H = */ q.shape(1),
      /* int D = */ q.shape(3),

      /* int qL = */ q.shape(2),
      /* int kL = */ k.shape(2),

      /* int gqa_factor = */ q.shape(1) / k.shape(1),
      /* int maskH = */ mask_arr ? mask_arr->shape(1) : 0,
      /* float scale = */ scale,

      /* int64_t Q_strides[3] = */ {q.strides(0), q.strides(1), q.strides(2)},
      /* int64_t K_strides[3] = */ {k.strides(0), k.strides(1), k.strides(2)},
      /* int64_t V_strides[3] = */ {v.strides(0), v.strides(1), v.strides(2)},
      /* int64_t O_strides[3] = */ {o.strides(0), o.strides(1), o.strides(2)},
      /* int64_t M_strides[4] = */ {0, 0, 0, 0}};

  if (mask_arr) {
    params.M_strides[0] = mask_arr->strides(0);
    params.M_strides[1] = mask_arr->strides(1);
    params.M_strides[2] = mask_arr->strides(2);
    params.M_strides[3] = mask_arr->strides(3);
  }
  return params;
}

// Query-tiled prefill attention for wide head dims (256/512). Each K/V tile
// is shared by 8 query rows instead of being re-streamed per query.
bool sdpa_vector_fvec_route(
    const char* tag,
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const std::optional<array>& sinks) {
  static bool enabled = env::get_var("MLX_CUDA_SDPA_FVEC_PREFILL", 0);
  if (!enabled) {
    return false;
  }

  const int head_dim = q.shape(-1);
  if ((head_dim != 256 && head_dim != 512) || v.shape(-1) != head_dim) {
    return false;
  }
  if (q.shape(2) < 32) {
    return false;
  }

  // Vectorized row loads require 16B-aligned rows and bases.
  const int vec = 16 / q.itemsize();
  auto rows_aligned = [vec](const array& a) {
    return a.strides(2) % vec == 0 &&
        (reinterpret_cast<uintptr_t>(gpu_ptr<void>(a)) % 16) == 0;
  };
  if (!rows_aligned(q) || !rows_aligned(k) || !rows_aligned(v) ||
      !rows_aligned(o)) {
    return false;
  }

  cu::AttnParams params = sdpa_tiled_params(q, k, v, scale, o, mask_arr);

  // Tensor-core variant (bf16/f16 only) takes precedence when enabled.
  static bool fmma_enabled = env::get_var("MLX_CUDA_SDPA_FMMA_PREFILL", 0);
  if (fmma_enabled && (o.dtype() == bfloat16 || o.dtype() == float16)) {
    dispatch_bool(do_causal, [&](auto causal_tag) {
      if (o.dtype() == bfloat16) {
        using DataType = __nv_bfloat16;
        if (head_dim == 256) {
          sdpa_vector_fmma_launch<DataType, causal_tag.value, 256>(
              encoder, q, k, v, o, mask_arr, sinks, params);
        } else {
          sdpa_vector_fmma_launch<DataType, causal_tag.value, 512>(
              encoder, q, k, v, o, mask_arr, sinks, params);
        }
      } else {
        using DataType = __half;
        if (head_dim == 256) {
          sdpa_vector_fmma_launch<DataType, causal_tag.value, 256>(
              encoder, q, k, v, o, mask_arr, sinks, params);
        } else {
          sdpa_vector_fmma_launch<DataType, causal_tag.value, 512>(
              encoder, q, k, v, o, mask_arr, sinks, params);
        }
      }
    });
    return true;
  }

  dispatch_float_types(o.dtype(), tag, [&](auto type_tag) {
    dispatch_bool(do_causal, [&](auto causal_tag) {
      using DataType = cuda_type_t<MLX_GET_TYPE(type_tag)>;

      if (head_dim == 256) {
        sdpa_vector_fvec_launch<DataType, causal_tag.value, 256>(
            encoder, q, k, v, o, mask_arr, sinks, params);
      } else {
        sdpa_vector_fvec_launch<DataType, causal_tag.value, 512>(
            encoder, q, k, v, o, mask_arr, sinks, params);
      }
    });
  });
  return true;
}

void sdpa_vector_fallback(
    const Stream& s,
    cu::CommandEncoder& encoder,
    const array& q,
    const array& k,
    const array& v,
    const float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& mask_arr,
    const std::optional<array>& sinks) {
  int kL = k.shape(2);

  if (sdpa_vector_fvec_route(
          "kernel_sdpav_fvec",
          encoder,
          q,
          k,
          v,
          scale,
          o,
          do_causal,
          mask_arr,
          sinks)) {
    return;
  }

  // The 2-pass kernel improves decode latency by parallelizing a single query
  // across KV blocks, but it materializes partials proportional to qL. For
  // prefill there is already enough parallelism across query positions, so use
  // the one-pass kernel to avoid enormous temporary buffers.
  if (kL >= 1024 && q.shape(2) < 4) {
    return sdpa_vector_2pass_fallback(
        s, encoder, q, k, v, scale, o, do_causal, mask_arr, sinks);
  } else {
    return sdpa_vector_1pass_fallback(
        s, encoder, q, k, v, scale, o, do_causal, mask_arr, sinks);
  }
}

} // namespace

bool supports_sdpa_vector(
    const array& q,
    const array& v,
    bool has_arr_mask,
    bool output_logsumexp) {
  if (output_logsumexp) {
    return false;
  }

  const int value_head_dim = v.shape(-1);
  const int query_head_dim = q.shape(-1);
  const int query_sequence_length = q.shape(2);
  const bool sdpa_supported_head_dim = query_head_dim == value_head_dim &&
      (query_head_dim == 64 || query_head_dim == 96 || query_head_dim == 128 ||
       query_head_dim == 256 || query_head_dim == 512);
  const bool wide_head_dim = query_head_dim == 256 || query_head_dim == 512;
  const bool supported_vector_config =
      wide_head_dim || (query_sequence_length < 4 && !has_arr_mask);

  return sdpa_supported_head_dim && supported_vector_config;
}

void sdpa_vector(
    const array& q_pre,
    const array& k_pre,
    const array& v_pre,
    float scale,
    array& o,
    bool do_causal,
    const std::optional<array>& mask_arr_pre,
    const std::optional<array>& sinks_pre,
    Stream s) {
  auto& encoder = cu::get_command_encoder(s);
  std::vector<array> copies;

  // Define some copy functions to ensure the layout of the inputs is as
  // expected.
  copies.reserve(5);
  auto copy_unless = [&copies, &s](
                         auto predicate, const array& arr) -> const array& {
    if (!predicate(arr)) {
      array arr_copy = contiguous_copy_gpu(arr, s);
      copies.push_back(std::move(arr_copy));
      return copies.back();
    } else {
      return arr;
    }
  };

  // Checks that the headdim dimension has stride 1.
  auto is_matrix_contiguous = [](const array& arr) {
    return arr.strides(-1) == 1;
  };

  std::optional<array> sinks = std::nullopt;
  if (sinks_pre) {
    sinks = copy_unless(is_matrix_contiguous, sinks_pre.value());
  }

  std::optional<array> mask_arr = std::nullopt;
  if (mask_arr_pre) {
    mask_arr = copy_unless(is_matrix_contiguous, mask_arr_pre.value());
  }

  // We are in vector mode ie single query
  if (q_pre.shape(2) < 4) {
    auto q_copy_unless = [](const array& arr) {
      if (arr.flags().row_contiguous) {
        return true;
      }
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (shape[0] == 1 || shape[1] == 1) {
        // If either the batch or head dimension is a singleton, the other can
        // be transposed with the sequence dimension
        auto bidx = shape[0] == 1 ? 1 : 0;
        return (strides[3] == 1) && (strides[2] == shape[3] * shape[bidx]) &&
            (strides[bidx] == shape[3]);
      }
      return false;
    };

    auto kv_copy_unless = [](const array& arr) {
      // keys and values should be copied if:
      // - the last dimension is not contiguous
      // - the batch and head dim are not contiguous
      auto& strides = arr.strides();
      auto& shape = arr.shape();
      if (strides.back() != 1) {
        return false;
      }
      if (shape[0] == 1 || shape[1] == 1) {
        return true;
      }
      return (strides[0] == strides[1] * shape[1]);
    };

    const auto& q = copy_unless(q_copy_unless, q_pre);
    const auto& k = copy_unless(kv_copy_unless, k_pre);
    const auto& v = copy_unless(kv_copy_unless, v_pre);

    // Donate the query if possible
    if (q.is_donatable() && q.flags().row_contiguous && q.size() == o.size()) {
      o.copy_shared_buffer(q);
    } else {
      int64_t str_oD = 1;
      int64_t str_oH = o.shape(3);
      int64_t str_oL = o.shape(1) * str_oH;
      int64_t str_oB = o.shape(2) * str_oL;

      array::Flags flags{
          /* bool contiguous = */ 1,
          /* bool row_contiguous = */ o.shape(2) == 1,
          /* bool col_contiguous = */ o.size() == o.shape(3),
      };

      o.set_data(
          cu::malloc_async(o.nbytes(), encoder),
          o.size(),
          {str_oB, str_oH, str_oL, str_oD},
          flags);
    }

    for (const auto& cp : copies) {
      encoder.add_temporary(cp);
    }

    sdpa_vector_fallback(
        s, encoder, q, k, v, scale, o, do_causal, mask_arr, sinks);
  }

  else {
    const auto& q = copy_unless(is_matrix_contiguous, q_pre);
    const auto& k = copy_unless(is_matrix_contiguous, k_pre);
    const auto& v = copy_unless(is_matrix_contiguous, v_pre);

    o.set_data(cu::malloc_async(o.nbytes(), encoder));

    for (const auto& cp : copies) {
      encoder.add_temporary(cp);
    }

    sdpa_vector_fallback(
        s, encoder, q, k, v, scale, o, do_causal, mask_arr, sinks);
  }
}

} // namespace mlx::core
