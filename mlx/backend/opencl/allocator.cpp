// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/memory.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <set>
#include <stdexcept>

namespace mlx::core::opencl {

OpenCLAllocator::OpenCLAllocator() {
  // Set memory limit based on device
  auto& dev = device(mlx::core::Device::gpu);

  cl_ulong mem_size = 0;
  cl_int err = clGetDeviceInfo(
      dev.device_id(),
      CL_DEVICE_GLOBAL_MEM_SIZE,
      sizeof(cl_ulong),
      &mem_size,
      nullptr);

  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        std::string("Failed to get device memory size: ") + std::to_string(err));
  }

  // Use 80% of device memory as default limit
  memory_limit_ = static_cast<size_t>(mem_size * 0.8);
  OPENCL_DEBUG_LOG("[ALLOCATOR] Memory limit: " << (memory_limit_ / (1024*1024)) << " MB");
}

OpenCLAllocator::~OpenCLAllocator() {
  // Note: During static destruction, the Device might already be destroyed.
  // We can safely release cl_mem objects without unmapping first -
  // the OpenCL runtime handles this automatically.

  std::unique_lock lk(mutex_);

  // Collect all cached buffer pointers so we don't double-release
  std::set<cl_mem> cached_buffers;
  for (auto& [size, buffers] : buffer_cache_) {
    for (auto buf : buffers) {
      cached_buffers.insert(buf);
      clReleaseMemObject(buf);
    }
  }
  buffer_cache_.clear();

  // Release any remaining tracked buffers that weren't in the cache
  // (active buffers that weren't freed before shutdown)
  for (auto& [buf, size] : buffer_sizes_) {
    if (cached_buffers.find(buf) == cached_buffers.end()) {
      clReleaseMemObject(buf);
    }
  }
  buffer_sizes_.clear();
  host_ptrs_.clear();
  is_mapped_.clear();
}

Buffer OpenCLAllocator::malloc(size_t size) {
  if (size == 0) {
    return Buffer{nullptr};
  }

  auto& dev = device(mlx::core::Device::gpu);

  std::unique_lock lk(mutex_);

  // Check cache for existing buffer of this size
  auto it = buffer_cache_.find(size);
  if (it != buffer_cache_.end() && !it->second.empty()) {
    cl_mem buf = it->second.back();
    it->second.pop_back();
    cache_memory_ -= size;
    active_memory_ += size;
    peak_memory_ = std::max(active_memory_, peak_memory_);

    // Cached buffers may be mapped - track their state
    // They're unmapped when going to cache, so mark as unmapped
    is_mapped_[buf] = false;

    return Buffer{static_cast<void*>(buf)};
  }

  lk.unlock();

  // Create new OpenCL buffer with host-accessible memory
  // CL_MEM_ALLOC_HOST_PTR: allocates memory that can be mapped to host
  cl_int err;
  cl_mem buf = clCreateBuffer(
      dev.context(),
      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
      size,
      nullptr,
      &err);

  // If allocation failed, try clearing the cache and retrying
  if (err != CL_SUCCESS) {
    OPENCL_DEBUG_LOG("[ALLOCATOR] Allocation failed (err=" << err << "), clearing cache...");

    // Clear cache to free memory
    clear_cache();

    // Retry allocation
    buf = clCreateBuffer(
        dev.context(),
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        size,
        nullptr,
        &err);

    if (err != CL_SUCCESS) {
      OPENCL_DEBUG_LOG("[ALLOCATOR] FAILED: size=" << size << " active=" << active_memory_);
      throw std::runtime_error(
          std::string("OpenCL buffer allocation failed: ") + std::to_string(err));
    }
  }

  // DON'T persistently map here - map on-demand in get_host_ptr()
  // This ensures proper cache coherency via map/unmap cycles

  lk.lock();

  // Track buffer info
  buffer_sizes_[buf] = size;
  is_mapped_[buf] = false;  // Not mapped yet
  active_memory_ += size;
  peak_memory_ = std::max(active_memory_, peak_memory_);

  // Return buffer with cl_mem as ptr_ (cast to void*)
  return Buffer{static_cast<void*>(buf)};
}

void OpenCLAllocator::free(Buffer buffer) {
  if (buffer.ptr() == nullptr) {
    return;
  }

  cl_mem buf = static_cast<cl_mem>(buffer.ptr());

  std::unique_lock lk(mutex_);

  auto size_it = buffer_sizes_.find(buf);
  if (size_it == buffer_sizes_.end()) {
    OPENCL_DEBUG_LOG("[ALLOCATOR] WARNING: Freeing unknown buffer");
    return;
  }

  size_t sz = size_it->second;
  active_memory_ -= sz;

  // Unmap buffer if currently mapped (before caching or releasing)
  auto mapped_it = is_mapped_.find(buf);
  bool currently_mapped = (mapped_it != is_mapped_.end() && mapped_it->second);

  if (currently_mapped) {
    auto host_it = host_ptrs_.find(buf);
    if (host_it != host_ptrs_.end()) {
      void* host_ptr = host_it->second;
      lk.unlock();

      auto& dev = device(mlx::core::Device::gpu);
      clEnqueueUnmapMemObject(
          dev.get_queue(Stream(0, mlx::core::Device::gpu)),
          buf,
          host_ptr,
          0, nullptr, nullptr);
      clFinish(dev.get_queue(Stream(0, mlx::core::Device::gpu)));

      lk.lock();
      host_ptrs_.erase(host_it);
      is_mapped_[buf] = false;
    }
  }

  // Add to cache instead of releasing immediately
  if (cache_memory_ + sz <= cache_limit_) {
    buffer_cache_[sz].push_back(buf);
    cache_memory_ += sz;
  } else {
    // Cache full - release the buffer
    buffer_sizes_.erase(size_it);
    is_mapped_.erase(buf);
    lk.unlock();

    clReleaseMemObject(buf);
    return;
  }
}

size_t OpenCLAllocator::size(Buffer buffer) const {
  if (buffer.ptr() == nullptr) {
    return 0;
  }

  cl_mem buf = static_cast<cl_mem>(buffer.ptr());

  std::unique_lock lk(mutex_);
  auto it = buffer_sizes_.find(buf);
  if (it != buffer_sizes_.end()) {
    return it->second;
  }

  // Query OpenCL for buffer size if not in our map
  size_t buf_size = 0;
  cl_int err = clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
  if (err == CL_SUCCESS) {
    return buf_size;
  }

  return 0;
}

Buffer OpenCLAllocator::make_buffer(void* ptr, size_t size) {
  // OpenCL cannot wrap arbitrary host pointers as device buffers.
  // Return nullptr to indicate the caller should copy the data themselves.
  // This ensures the deleter is called with the correct original pointer.
  return Buffer{nullptr};
}

void OpenCLAllocator::release(Buffer buffer) {
  // Same as free
  free(buffer);
}

size_t OpenCLAllocator::get_cache_memory() {
  std::unique_lock lk(mutex_);
  return cache_memory_;
}

size_t OpenCLAllocator::set_cache_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(cache_limit_, limit);
  return limit;
}

size_t OpenCLAllocator::set_memory_limit(size_t limit) {
  std::unique_lock lk(mutex_);
  std::swap(memory_limit_, limit);
  return limit;
}

size_t OpenCLAllocator::get_memory_limit() {
  std::unique_lock lk(mutex_);
  return memory_limit_;
}

void OpenCLAllocator::clear_cache() {
  auto& dev = device(mlx::core::Device::gpu);

  std::unique_lock lk(mutex_);

  for (auto& [size, buffers] : buffer_cache_) {
    for (auto buf : buffers) {
      // Cached buffers should already be unmapped, but check anyway
      auto mapped_it = is_mapped_.find(buf);
      if (mapped_it != is_mapped_.end() && mapped_it->second) {
        auto host_it = host_ptrs_.find(buf);
        if (host_it != host_ptrs_.end()) {
          lk.unlock();
          clEnqueueUnmapMemObject(
              dev.get_queue(Stream(0, mlx::core::Device::gpu)),
              buf,
              host_it->second,
              0, nullptr, nullptr);
          clFinish(dev.get_queue(Stream(0, mlx::core::Device::gpu)));
          lk.lock();
          host_ptrs_.erase(host_it);
        }
      }
      buffer_sizes_.erase(buf);
      is_mapped_.erase(buf);
      lk.unlock();
      clReleaseMemObject(buf);
      lk.lock();
    }
  }

  buffer_cache_.clear();
  cache_memory_ = 0;
}

void* OpenCLAllocator::get_host_ptr(cl_mem buf) {
  if (buf == nullptr) {
    return nullptr;
  }

  // Map on-demand if not already mapped
  return map_for_host(buf);
}

bool OpenCLAllocator::is_mapped(cl_mem buf) {
  if (buf == nullptr) {
    return false;
  }

  std::unique_lock lk(mutex_);
  auto it = is_mapped_.find(buf);
  return (it != is_mapped_.end() && it->second);
}

void* OpenCLAllocator::map_for_host(cl_mem buf) {
  if (buf == nullptr) {
    return nullptr;
  }

  std::unique_lock lk(mutex_);

  // Check if already mapped
  auto mapped_it = is_mapped_.find(buf);
  if (mapped_it != is_mapped_.end() && mapped_it->second) {
    // Already mapped, return cached host pointer
    auto host_it = host_ptrs_.find(buf);
    if (host_it != host_ptrs_.end()) {
      return host_it->second;
    }
  }

  // Get buffer size
  auto size_it = buffer_sizes_.find(buf);
  if (size_it == buffer_sizes_.end()) {
    OPENCL_DEBUG_LOG("[ALLOCATOR] WARNING: map_for_host called for unknown buffer");
    return nullptr;
  }
  size_t size = size_it->second;

  lk.unlock();

  // Map the buffer for host access
  auto& dev = device(mlx::core::Device::gpu);
  cl_int err;
  void* host_ptr = clEnqueueMapBuffer(
      dev.get_queue(Stream(0, mlx::core::Device::gpu)),
      buf,
      CL_TRUE,  // blocking
      CL_MAP_READ | CL_MAP_WRITE,
      0,
      size,
      0, nullptr, nullptr,
      &err);

  if (err != CL_SUCCESS) {
    OPENCL_DEBUG_LOG("[ALLOCATOR] ERROR: map_for_host failed with error " << err);
    return nullptr;
  }

  lk.lock();
  host_ptrs_[buf] = host_ptr;
  is_mapped_[buf] = true;

  return host_ptr;
}

void OpenCLAllocator::unmap_for_device(cl_mem buf) {
  if (buf == nullptr) {
    return;
  }

  std::unique_lock lk(mutex_);

  // Check if currently mapped
  auto mapped_it = is_mapped_.find(buf);
  if (mapped_it == is_mapped_.end() || !mapped_it->second) {
    // Not mapped, nothing to do
    return;
  }

  auto host_it = host_ptrs_.find(buf);
  if (host_it == host_ptrs_.end()) {
    // No host pointer, can't unmap
    is_mapped_[buf] = false;
    return;
  }

  void* host_ptr = host_it->second;
  is_mapped_[buf] = false;
  host_ptrs_.erase(host_it);

  lk.unlock();

  // Unmap the buffer so GPU can access fresh data
  auto& dev = device(mlx::core::Device::gpu);
  clEnqueueUnmapMemObject(
      dev.get_queue(Stream(0, mlx::core::Device::gpu)),
      buf,
      host_ptr,
      0, nullptr, nullptr);

  // Wait for unmap to complete - this ensures CPU cache is flushed
  clFinish(dev.get_queue(Stream(0, mlx::core::Device::gpu)));
}

void OpenCLAllocator::ensure_on_device(cl_mem buf, cl_command_queue queue) {
  // Unmap the buffer if mapped, so GPU can see CPU's writes
  // The unmap operation flushes CPU caches and ensures data visibility
  if (buf == nullptr) {
    return;
  }

  std::unique_lock lk(mutex_);

  // Check if currently mapped
  auto mapped_it = is_mapped_.find(buf);
  if (mapped_it == is_mapped_.end() || !mapped_it->second) {
    // Not mapped, nothing to do
    return;
  }

  auto host_it = host_ptrs_.find(buf);
  if (host_it == host_ptrs_.end()) {
    // No host pointer, can't unmap
    is_mapped_[buf] = false;
    return;
  }

  void* host_ptr = host_it->second;
  is_mapped_[buf] = false;
  host_ptrs_.erase(host_it);

  lk.unlock();

  // IMPORTANT: Always use stream 0's queue for unmap operations.
  // This ensures consistency with map_for_host which also uses stream 0.
  // Using different queues for map/unmap can cause synchronization issues
  // on some OpenCL implementations (e.g., Qualcomm Adreno).
  auto& dev = device(mlx::core::Device::gpu);
  cl_command_queue q = dev.get_queue(Stream(0, mlx::core::Device::gpu));

  // Unmap the buffer so GPU can access fresh data
  clEnqueueUnmapMemObject(q, buf, host_ptr, 0, nullptr, nullptr);

  // Wait for unmap to complete - this ensures CPU cache is flushed
  clFinish(q);

  // If the provided queue is different from stream 0, we need to ensure
  // synchronization between the two queues. On some drivers, a kernel on
  // queue B might not see writes completed on queue A without explicit sync.
  if (queue != nullptr && queue != q) {
    // Add a barrier on the target queue to ensure it waits for stream 0
    clEnqueueBarrierWithWaitList(queue, 0, nullptr, nullptr);
    clFinish(queue);
  }
}

bool OpenCLAllocator::is_opencl_buffer(void* ptr) {
  if (ptr == nullptr) {
    return false;
  }

  std::unique_lock lk(mutex_);
  cl_mem buf = static_cast<cl_mem>(ptr);
  return buffer_sizes_.find(buf) != buffer_sizes_.end();
}

cl_mem OpenCLAllocator::create_temp_buffer_from_cpu(const void* cpu_ptr, size_t size) {
  if (cpu_ptr == nullptr || size == 0) {
    return nullptr;
  }

  // Create a new OpenCL buffer and copy the CPU data into it
  // IMPORTANT: We use explicit clEnqueueWriteBuffer with blocking mode (CL_TRUE)
  // instead of CL_MEM_COPY_HOST_PTR because some drivers (e.g., Qualcomm Adreno)
  // may defer the copy with CL_MEM_COPY_HOST_PTR, causing race conditions where
  // kernels read uninitialized data.
  auto& dev = device(mlx::core::Device::gpu);

  cl_int err;
  // First, create the buffer without data
  cl_mem buf = clCreateBuffer(
      dev.context(),
      CL_MEM_READ_WRITE,
      size,
      nullptr,
      &err);

  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        "[create_temp_buffer_from_cpu] Failed to create buffer: " + std::to_string(err));
  }

  // Then, use blocking write to copy data - this ensures data is available
  // before any kernel uses the buffer
  cl_command_queue queue = dev.get_queue(Stream(0, mlx::core::Device::gpu));
  err = clEnqueueWriteBuffer(
      queue,
      buf,
      CL_TRUE,  // blocking_write - wait for copy to complete
      0,        // offset
      size,
      cpu_ptr,
      0, nullptr, nullptr);

  if (err != CL_SUCCESS) {
    clReleaseMemObject(buf);
    throw std::runtime_error(
        "[create_temp_buffer_from_cpu] Failed to write buffer: " + std::to_string(err));
  }

  // Note: This buffer is NOT tracked in buffer_sizes_ because it's temporary
  // The caller is responsible for releasing it
  return buf;
}

OpenCLAllocator& allocator() {
  static OpenCLAllocator allocator_;
  return allocator_;
}

} // namespace mlx::core::opencl

// Global allocator interface for MLX
namespace mlx::core::allocator {

Allocator& allocator() {
  return opencl::allocator();
}

void* Buffer::raw_ptr() {
  if (!ptr_) {
    return nullptr;
  }
  // ptr_ is cl_mem, get the host pointer from the allocator
  return opencl::allocator().get_host_ptr(static_cast<cl_mem>(ptr_));
}

} // namespace mlx::core::allocator

// Memory management functions for mlx::core namespace
namespace mlx::core {

size_t get_active_memory() {
  return opencl::allocator().get_active_memory();
}

size_t get_peak_memory() {
  return opencl::allocator().get_peak_memory();
}

void reset_peak_memory() {
  opencl::allocator().reset_peak_memory();
}

size_t get_memory_limit() {
  return opencl::allocator().get_memory_limit();
}

size_t set_memory_limit(size_t limit) {
  return opencl::allocator().set_memory_limit(limit);
}

size_t get_cache_memory() {
  return opencl::allocator().get_cache_memory();
}

size_t set_cache_limit(size_t limit) {
  return opencl::allocator().set_cache_limit(limit);
}

size_t set_wired_limit(size_t) {
  // Wired memory is a Metal-specific concept, no-op for OpenCL
  return 0;
}

void clear_cache() {
  opencl::allocator().clear_cache();
}

} // namespace mlx::core
