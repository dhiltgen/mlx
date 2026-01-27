// Copyright © 2025 MLX Contributors
// OpenCL primitive eval_gpu stubs

#include "mlx/primitives.h"
#include "mlx/fast_primitives.h"
#include "mlx/stream.h"  // For default_stream()

using mlx::core::array;

// TODO: This file contains stub implementations of eval_gpu() for all primitives
// These currently fall back to CPU evaluation
// Real implementations should execute OpenCL kernels

namespace mlx::core {

// Stub implementations - throw exceptions for unimplemented operations
// NOTE: We cannot safely call eval_cpu() from GPU context because primitives
// store a GPU stream, and eval_cpu() uses stream() which tries to access
// CPU StreamThread objects that don't exist for GPU streams.
// The proper fix requires creating new primitives with CPU streams, but many
// of these primitives have complex constructor signatures making that impractical.
// Instead, we throw exceptions to make it clear these operations aren't implemented.
#define STUB_EVAL_GPU(ClassName) \
  void ClassName::eval_gpu(const std::vector<array>& inputs, array& output) { \
    throw std::runtime_error("[" #ClassName "::eval_gpu] OpenCL implementation not yet available"); \
  }

// Primitives from arg_reduce.cpp - now implemented in reduce.cpp
// STUB_EVAL_GPU(ArgReduce)

// Primitives from primitives.cpp
// STUB_EVAL_GPU(Arange) // Now implemented in arange.cpp
// STUB_EVAL_GPU(RandomBits) // Now implemented in random.cpp

// Primitives from indexing.cpp - now implemented in indexing.cpp
// STUB_EVAL_GPU(Gather)
// STUB_EVAL_GPU(GatherAxis)
// STUB_EVAL_GPU(Scatter)
// STUB_EVAL_GPU(ScatterAxis)
// STUB_EVAL_GPU(MaskedScatter)  // Now implemented in indexing.cpp

// Primitives from binary.cpp
// STUB_EVAL_GPU(BitwiseBinary) // Now implemented in binary.cpp
// DivMod is now implemented in binary.cpp

// Primitives from unary.cpp
// STUB_EVAL_GPU(BitwiseInvert) // Now implemented in unary.cpp
// STUB_EVAL_GPU(Conjugate) // Now implemented in unary.cpp
// STUB_EVAL_GPU(ErfInv) // Now implemented in unary.cpp
// STUB_EVAL_GPU(Imag) // Now implemented in unary.cpp
// STUB_EVAL_GPU(Log) // Now implemented in unary.cpp
// STUB_EVAL_GPU(Real) // Now implemented in unary.cpp
// STUB_EVAL_GPU(Round) // Now implemented in unary.cpp

// Primitives from conv.cpp - now implemented in conv.cpp
// STUB_EVAL_GPU(Convolution)

// Primitives from masked_mm.cpp - stub implementations
STUB_EVAL_GPU(BlockMaskedMM)
STUB_EVAL_GPU(GatherMM)
STUB_EVAL_GPU(SegmentedMM)

// Primitives from sort.cpp - now implemented in sort.cpp
// STUB_EVAL_GPU(ArgPartition)
// STUB_EVAL_GPU(ArgSort)
// STUB_EVAL_GPU(Partition)
// STUB_EVAL_GPU(Sort)

// Primitives from quantized.cpp - now implemented in quantized.cpp
// STUB_EVAL_GPU(QuantizedMatmul)
// STUB_EVAL_GPU(QQMatmul)
// STUB_EVAL_GPU(GatherQMM)

// Additional primitives from matmul.cpp
// STUB_EVAL_GPU(AddMM)  // Now implemented in matmul.cpp
// STUB_EVAL_GPU(Scan) // Now implemented in scan.cpp
// STUB_EVAL_GPU(Matmul) // Now implemented in matmul.cpp
// STUB_EVAL_GPU(Reduce) // Now implemented in reduce.cpp
// STUB_EVAL_GPU(Select) // Now implemented in select.cpp
// STUB_EVAL_GPU(Softmax) // Now implemented in softmax.cpp
STUB_EVAL_GPU(Hadamard)
// STUB_EVAL_GPU(LogSumExp) // Now implemented in logsumexp.cpp
// STUB_EVAL_GPU(FFT) // Now implemented in fft.cpp
STUB_EVAL_GPU(Load)
STUB_EVAL_GPU(Inverse)
STUB_EVAL_GPU(Cholesky)

// Primitives with multiple outputs - throw exceptions for unimplemented ops
// Compiled is now implemented in compiled.cpp

void Eig::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw std::runtime_error("[Eig::eval_gpu] OpenCL implementation not yet available");
}

void QRF::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw std::runtime_error("[QRF::eval_gpu] OpenCL implementation not yet available");
}

void SVD::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw std::runtime_error("[SVD::eval_gpu] OpenCL implementation not yet available");
}

void Eigh::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw std::runtime_error("[Eigh::eval_gpu] OpenCL implementation not yet available");
}

void LUF::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {
  throw std::runtime_error("[LUF::eval_gpu] OpenCL implementation not yet available");
}

// Fast primitives with multiple outputs
namespace fast {
// ConvertFP8::eval_gpu is now implemented in quantized.cpp
// Quantize::eval_gpu is now implemented in quantized.cpp
} // namespace fast

// NOTE: Many metadata-only operations (Squeeze, Transpose, etc.) are already
// defined in backend/gpu/primitives.cpp which is compiled for all GPU backends.
// Only define OpenCL-specific implementations here for operations that need
// custom handling.

} // namespace mlx::core
