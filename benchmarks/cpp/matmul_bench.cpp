// Copyright © 2024 Apple Inc.
// Comprehensive matmul benchmark for performance analysis

#include "mlx/mlx.h"
#include "time_utils.h"

#include <iomanip>
#include <iostream>
#include <vector>

namespace mx = mlx::core;

struct BenchResult {
  int M, N, K;
  std::string transpose;
  double time_ms;
  double gflops;
};

double compute_gflops(int M, int N, int K, double time_ms, int iterations = 100) {
  // GFLOPS = 2 * M * N * K * iterations / time_seconds / 1e9
  double flops = 2.0 * M * N * K * iterations;
  double time_sec = time_ms * iterations / 1000.0;
  return flops / time_sec / 1e9;
}

double bench_matmul(int M, int N, int K, const std::string& transpose = "nn",
                    int warmup = 5, int iterations = 100) {
  // Create matrices based on transpose mode
  auto a = (transpose[0] == 'n')
    ? mx::random::normal({M, K})
    : mx::random::normal({K, M});
  auto b = (transpose[1] == 'n')
    ? mx::random::normal({K, N})
    : mx::random::normal({N, K});
  mx::eval(a, b);

  // Lambda to perform matmul based on transpose mode
  auto do_matmul = [&]() {
    if (transpose == "nn") {
      return mx::matmul(a, b);
    } else if (transpose == "nt") {
      return mx::matmul(a, mx::transpose(b));
    } else if (transpose == "tn") {
      return mx::matmul(mx::transpose(a), b);
    } else {
      return mx::matmul(mx::transpose(a), mx::transpose(b));
    }
  };

  // Warmup
  for (int i = 0; i < warmup; i++) {
    mx::eval(do_matmul());
  }

  // Benchmark
  auto start = time_now();
  for (int i = 0; i < iterations; i++) {
    mx::eval(do_matmul());
  }
  auto end = time_now();

  return milliseconds(end - start) / static_cast<double>(iterations);
}

void print_header(const std::string& title) {
  std::cout << "\n" << std::string(70, '=') << "\n";
  std::cout << title << "\n";
  std::cout << std::string(70, '=') << "\n";
  std::cout << std::setw(8) << "M"
            << std::setw(8) << "N"
            << std::setw(8) << "K"
            << std::setw(8) << "Trans"
            << std::setw(12) << "Time(ms)"
            << std::setw(12) << "GFLOPS"
            << "\n";
  std::cout << std::string(70, '-') << "\n";
}

void print_result(const BenchResult& r) {
  std::cout << std::setw(8) << r.M
            << std::setw(8) << r.N
            << std::setw(8) << r.K
            << std::setw(8) << r.transpose
            << std::setw(12) << std::fixed << std::setprecision(3) << r.time_ms
            << std::setw(12) << std::fixed << std::setprecision(1) << r.gflops
            << "\n";
}

int main(int argc, char* argv[]) {
  std::cout << "MLX MatMul Benchmark\n";
  std::cout << "Device: " << mx::default_device() << "\n";

  std::vector<BenchResult> results;

  // Square matrices (baseline)
  print_header("Square Matrices (A @ B)");
  std::vector<int> square_sizes = {256, 512, 1024, 2048, 4096};
  for (int size : square_sizes) {
    double time_ms = bench_matmul(size, size, size, "nn");
    double gflops = compute_gflops(size, size, size, time_ms);
    BenchResult r{size, size, size, "nn", time_ms, gflops};
    print_result(r);
    results.push_back(r);
  }

  // Transposed matrices (common LLM pattern: x @ W.T)
  print_header("Transposed Matrices (A @ B.T) - LLM Linear Layer Pattern");
  for (int size : square_sizes) {
    double time_ms = bench_matmul(size, size, size, "nt");
    double gflops = compute_gflops(size, size, size, time_ms);
    BenchResult r{size, size, size, "nt", time_ms, gflops};
    print_result(r);
    results.push_back(r);
  }

  // GEMV cases (M=1, single token generation)
  print_header("GEMV Cases (M=1) - Single Token Generation");
  std::vector<std::tuple<int, int, int>> gemv_shapes = {
    {1, 2048, 2048},   // Attention projection (1B)
    {1, 2048, 8192},   // FFN up-projection (1B)
    {1, 3072, 3072},   // Attention projection (3B)
    {1, 3072, 8192},   // FFN up-projection (3B)
    {1, 2048, 128256}, // LM head / vocab projection (1B)
  };
  for (auto [M, N, K] : gemv_shapes) {
    double time_ms = bench_matmul(M, N, K, "nt");
    double gflops = compute_gflops(M, N, K, time_ms);
    BenchResult r{M, N, K, "nt", time_ms, gflops};
    print_result(r);
    results.push_back(r);
  }

  // Small batch (typical prefill/batch inference)
  print_header("Small Batch (M=32) - Batched Token Processing");
  std::vector<std::tuple<int, int, int>> batch_shapes = {
    {32, 2048, 2048},
    {32, 2048, 8192},
    {32, 8192, 2048},
  };
  for (auto [M, N, K] : batch_shapes) {
    double time_ms = bench_matmul(M, N, K, "nt");
    double gflops = compute_gflops(M, N, K, time_ms);
    BenchResult r{M, N, K, "nt", time_ms, gflops};
    print_result(r);
    results.push_back(r);
  }

  // Prefill sizes (larger batches)
  print_header("Prefill Sizes (M=512) - Prompt Processing");
  std::vector<std::tuple<int, int, int>> prefill_shapes = {
    {512, 2048, 2048},
    {512, 2048, 8192},
    {512, 8192, 2048},
  };
  for (auto [M, N, K] : prefill_shapes) {
    double time_ms = bench_matmul(M, N, K, "nt");
    double gflops = compute_gflops(M, N, K, time_ms);
    BenchResult r{M, N, K, "nt", time_ms, gflops};
    print_result(r);
    results.push_back(r);
  }

  // Summary
  std::cout << "\n" << std::string(70, '=') << "\n";
  std::cout << "Summary\n";
  std::cout << std::string(70, '=') << "\n";

  double max_gflops = 0;
  double min_gflops = 1e9;
  for (const auto& r : results) {
    if (r.gflops > max_gflops) max_gflops = r.gflops;
    if (r.gflops < min_gflops) min_gflops = r.gflops;
  }

  std::cout << "Peak GFLOPS: " << std::fixed << std::setprecision(1) << max_gflops << "\n";
  std::cout << "Min GFLOPS:  " << std::fixed << std::setprecision(1) << min_gflops << "\n";

  // Arithmetic intensity analysis
  std::cout << "\nArithmetic Intensity (AI) for square matrices:\n";
  std::cout << std::setw(15) << "Size"
            << std::setw(15) << "AI (FLOP/B)"
            << "\n";

  for (int size : square_sizes) {
    // AI = 2*M*N*K / (4*(M*K + K*N + M*N)) for float32
    double flops = 2.0 * size * size * size;
    double bytes = 4.0 * (size * size + size * size + size * size);
    double ai = flops / bytes;
    std::cout << std::setw(15) << std::to_string(size) + "x" + std::to_string(size)
              << std::setw(15) << std::fixed << std::setprecision(1) << ai
              << "\n";
  }

  std::cout << "\nNote: Higher AI means more compute-bound, lower means memory-bound.\n";
  std::cout << "Ridge point depends on your GPU's compute/bandwidth ratio.\n";

  return 0;
}
