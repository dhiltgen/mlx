// Copyright © 2023 Apple Inc.

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include <cstdio>
#include <cstdlib>

#include "mlx/mlx.h"

// Include CUDA runtime if available (will be present when MLX_BUILD_CUDA=ON)
#if __has_include(<cuda_runtime.h>)
#include <cuda_runtime.h>
#define HAS_CUDA_RUNTIME 1
#endif

using namespace mlx::core;

// Global test listener to reset GPU state after each test
struct CudaCleanupListener : doctest::IReporter {
  std::ostream& stdout_stream;
  std::string current_test_name;

  explicit CudaCleanupListener(const doctest::ContextOptions& in)
      : stdout_stream(std::cout) {}

  void test_case_start(const doctest::TestCaseData& data) override {
    current_test_name = data.m_name;
  }

  void test_case_end(const doctest::CurrentTestCaseStats& stats) override {
#ifdef HAS_CUDA_RUNTIME
    // Clear any pending CUDA errors after each test to prevent cascading
    // failures This is especially important if a test hits an unregistered
    // kernel or other CUDA graph issue that could corrupt state for subsequent
    // tests
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(
          stderr,
          "\n[TEST CLEANUP] Cleared CUDA error after test '%s': %s\n",
          current_test_name.c_str(),
          cudaGetErrorString(err));
      fflush(stderr);
    }

    // Synchronize device to ensure all operations complete
    // This prevents incomplete operations from affecting the next test
    cudaDeviceSynchronize();
#endif
  }

  // Required virtual methods (minimal implementation)
  void report_query(const doctest::QueryData&) override {}
  void test_run_start() override {}
  void test_run_end(const doctest::TestRunStats&) override {}
  void test_case_reenter(const doctest::TestCaseData&) override {}
  void test_case_exception(const doctest::TestCaseException&) override {}
  void subcase_start(const doctest::SubcaseSignature&) override {}
  void subcase_end() override {}
  void log_assert(const doctest::AssertData&) override {}
  void log_message(const doctest::MessageData&) override {}
  void test_case_skipped(const doctest::TestCaseData&) override {}
};

REGISTER_LISTENER("cuda_cleanup", 1, CudaCleanupListener);

int main(int argc, char** argv) {
  doctest::Context context;

  const char* device = std::getenv("DEVICE");
  printf("[TEST DEBUG] DEVICE env: %s\n", device ? device : "(not set)");
  fflush(stdout);

  if (device != nullptr && std::string(device) == "cpu") {
    printf("[TEST DEBUG] Setting default device to CPU\n");
    set_default_device(Device::cpu);
  } else if (device != nullptr && std::string(device) == "gpu") {
    // Explicitly requested GPU - check if available
    printf("[TEST DEBUG] GPU explicitly requested, checking availability...\n");
    fflush(stdout);
    if (is_available(Device::gpu)) {
      printf("[TEST DEBUG] GPU is available, setting as default\n");
      set_default_device(Device::gpu);
    } else {
      printf("[TEST DEBUG] GPU requested but not available!\n");
    }
  } else {
    // Check CUDA availability on Windows/Linux
    if (is_available(Device::gpu)) {
      printf("[TEST DEBUG] GPU (CUDA) is available, setting as default\n");
      set_default_device(Device::gpu);
    } else if (metal::is_available()) {
      // macOS Metal backend
      printf("[TEST DEBUG] Metal is available, setting GPU as default\n");
      set_default_device(Device::gpu);
    }
  }

  printf(
      "[TEST DEBUG] Default device type: %d (0=cpu, 1=gpu)\n",
      static_cast<int>(default_device().type));
  fflush(stdout);

  context.applyCommandLine(argc, argv);
  return context.run();
}
