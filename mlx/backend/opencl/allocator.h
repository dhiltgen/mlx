// Copyright © 2025 MLX Contributors

#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <map>
#include <mutex>
#include <set>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/opencl/device.h"

namespace mlx::core::opencl {

using allocator::Buffer;

/**
 * OpenCL memory allocator.
 * Manages cl_mem buffers and provides caching to reduce allocation overhead.
 */
class OpenCLAllocator : public allocator::Allocator {
 public:
  virtual Buffer malloc(size_t size) override;
  virtual void free(Buffer buffer) override;
  virtual size_t size(Buffer buffer) const override;
  virtual Buffer make_buffer(void* ptr, size_t size) override;
  virtual void release(Buffer buffer) override;

  // Memory statistics
  size_t get_active_memory() { return active_memory_; }
  size_t get_peak_memory() { return peak_memory_; }
  void reset_peak_memory() {
    std::unique_lock lk(mutex_);
    peak_memory_ = 0;
  }

  size_t get_cache_memory();
  size_t set_cache_limit(size_t limit);
  size_t set_memory_limit(size_t limit);
  size_t get_memory_limit();
  void clear_cache();

  // Get host-accessible pointer (maps buffer if not already mapped)
  void* get_host_ptr(cl_mem buf);

  // Ensure buffer data is on device (unmaps buffer to flush CPU cache)
  // queue parameter specifies which command queue to use for the unmap
  void ensure_on_device(cl_mem buf, cl_command_queue queue = nullptr);

  // Check if buffer is currently mapped
  bool is_mapped(cl_mem buf);

  // Explicitly map buffer for host access (returns host pointer)
  void* map_for_host(cl_mem buf);

  // Explicitly unmap buffer (for GPU access)
  void unmap_for_device(cl_mem buf);

  // Check if a pointer is a known OpenCL buffer (vs CPU heap memory)
  bool is_opencl_buffer(void* ptr);

  // Create a temporary GPU buffer from CPU data
  cl_mem create_temp_buffer_from_cpu(const void* cpu_ptr, size_t size);

 private:
  OpenCLAllocator();
  ~OpenCLAllocator();
  friend OpenCLAllocator& allocator();


  // Simple buffer cache: size -> list of buffers
  std::map<size_t, std::vector<cl_mem>> buffer_cache_;

  // Track buffer sizes: buffer -> size
  std::map<cl_mem, size_t> buffer_sizes_;

  // Host allocations for zero-copy shared memory (CL_MEM_USE_HOST_PTR)
  // Maps cl_mem device buffer to the host memory we allocated
  std::map<cl_mem, void*> host_ptrs_;

  // Track which buffers are currently mapped (for proper map/unmap cycles)
  std::map<cl_mem, bool> is_mapped_;

  // Memory statistics
  size_t cache_limit_{1UL << 30};  // 1GB default cache limit
  size_t memory_limit_{0};
  size_t active_memory_{0};
  size_t peak_memory_{0};
  size_t cache_memory_{0};

  mutable std::mutex mutex_;
};

/**
 * Get the global OpenCL allocator instance.
 */
OpenCLAllocator& allocator();

} // namespace mlx::core::opencl
