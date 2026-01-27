// Copyright © 2025 MLX Contributors

#include "mlx/distributed/primitives.h"

#include <stdexcept>

namespace mlx::core::distributed {

void AllReduce::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error(
      "[AllReduce::eval_gpu] Distributed operations not yet supported on OpenCL backend");
}

void AllGather::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error(
      "[AllGather::eval_gpu] Distributed operations not yet supported on OpenCL backend");
}

void Send::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error(
      "[Send::eval_gpu] Distributed operations not yet supported on OpenCL backend");
}

void Recv::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error(
      "[Recv::eval_gpu] Distributed operations not yet supported on OpenCL backend");
}

void ReduceScatter::eval_gpu(
    const std::vector<array>& inputs,
    std::vector<array>& outputs) {
  throw std::runtime_error(
      "[ReduceScatter::eval_gpu] Distributed operations not yet supported on OpenCL backend");
}

} // namespace mlx::core::distributed
