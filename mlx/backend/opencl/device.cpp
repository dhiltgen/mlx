// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/available.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/backend/gpu/device_info.h"
#include "mlx/version.h"

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace mlx::core::opencl {

namespace {

// Helper to get OpenCL error string
const char* get_error_string(cl_int error) {
  switch (error) {
    case CL_SUCCESS: return "Success";
    case CL_DEVICE_NOT_FOUND: return "Device not found";
    case CL_DEVICE_NOT_AVAILABLE: return "Device not available";
    case CL_OUT_OF_HOST_MEMORY: return "Out of host memory";
    case CL_OUT_OF_RESOURCES: return "Out of resources";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "Memory object allocation failure";
    case CL_INVALID_VALUE: return "Invalid value";
    case CL_INVALID_DEVICE: return "Invalid device";
    case CL_INVALID_CONTEXT: return "Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES: return "Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE: return "Invalid command queue";
    case CL_INVALID_KERNEL: return "Invalid kernel";
    case CL_INVALID_PROGRAM: return "Invalid program";
    case CL_INVALID_BUILD_OPTIONS: return "Invalid build options";
    case CL_BUILD_PROGRAM_FAILURE: return "Build program failure";
    default: return "Unknown error";
  }
}

#define CL_CHECK(call) \
  do { \
    cl_int err = (call); \
    if (err != CL_SUCCESS) { \
      std::ostringstream oss; \
      oss << "OpenCL error at " << __FILE__ << ":" << __LINE__ \
          << " - " << get_error_string(err) << " (" << err << ")"; \
      throw std::runtime_error(oss.str()); \
    } \
  } while (0)

// ============================================================================
// Kernel Binary Disk Cache
// ============================================================================
// Caches compiled OpenCL programs to disk to avoid recompilation on startup.
// Pattern follows CUDA backend (jit_module.cpp).
//
// Cache structure:
//   <temp_dir>/mlx/<version>/opencl/<device_hash>/<kernel_hash>.bin
//
// Invalidation:
//   - MLX version change -> new directory
//   - Source/options hash in filename -> source changes invalidate
//   - Device hash in path -> different GPU uses different cache

// Simple hash function for cache keys (same as std::hash<std::string>)
size_t hash_string(const std::string& s) {
  return std::hash<std::string>{}(s);
}

// Create a sanitized device identifier for the cache directory
std::string get_device_hash(const std::string& device_name, const std::string& vendor, const std::string& version) {
  std::string id = device_name + "_" + vendor;
  // Replace spaces and special characters with underscores
  for (char& c : id) {
    if (c == ' ' || c == '/' || c == '\\' || c == ':' || c == '(' || c == ')') {
      c = '_';
    }
  }
  // Add a hash of the version string for driver version sensitivity
  id += "_" + std::to_string(hash_string(version) % 10000);
  return id;
}

// Get the cache directory, creating it if necessary
std::filesystem::path get_cache_directory(const std::string& device_hash) {
  static std::filesystem::path cache_dir;
  static std::string cached_device_hash;
  static bool initialized = false;

  if (initialized && cached_device_hash == device_hash) {
    return cache_dir;
  }

  // Check for environment variable override
  if (auto env = std::getenv("MLX_OPENCL_CACHE_DIR"); env) {
    cache_dir = env;
  } else {
    cache_dir = std::filesystem::temp_directory_path() / "mlx" / version() / "opencl" / device_hash;
  }

  // Create directory if it doesn't exist
  if (!cache_dir.empty() && !std::filesystem::exists(cache_dir)) {
    std::error_code error;
    if (!std::filesystem::create_directories(cache_dir, error)) {
      OPENCL_DEBUG_LOG("[CACHE] Failed to create cache directory: " << cache_dir);
      cache_dir.clear();  // Disable caching if we can't create directory
    } else {
      OPENCL_DEBUG_LOG("[CACHE] Created cache directory: " << cache_dir);
    }
  }

  cached_device_hash = device_hash;
  initialized = true;
  return cache_dir;
}

// Get the path for a cached binary
std::filesystem::path get_cache_path(const std::filesystem::path& cache_dir, const std::string& source, const std::string& options) {
  if (cache_dir.empty()) {
    return {};
  }
  // Use hash of source + options as filename
  size_t hash = hash_string(source + "|" + options);
  return cache_dir / (std::to_string(hash) + ".bin");
}

// Try to load a cached binary
bool read_cached_binary(
    const std::filesystem::path& cache_path,
    std::vector<unsigned char>& binary) {
  if (cache_path.empty()) {
    return false;
  }

  std::error_code error;
  auto file_size = std::filesystem::file_size(cache_path, error);
  if (error || file_size == 0) {
    return false;
  }

  std::ifstream file(cache_path, std::ios::binary);
  if (!file.good()) {
    return false;
  }

  binary.resize(file_size);
  file.read(reinterpret_cast<char*>(binary.data()), file_size);

  if (!file.good()) {
    binary.clear();
    return false;
  }

  OPENCL_DEBUG_LOG("[CACHE] Loaded cached binary from: " << cache_path << " (" << file_size << " bytes)");
  return true;
}

// Save a compiled binary to cache
void write_cached_binary(
    const std::filesystem::path& cache_path,
    const std::vector<unsigned char>& binary,
    const std::string& source) {
  if (cache_path.empty() || binary.empty()) {
    return;
  }

  // Write binary
  std::ofstream file(cache_path, std::ios::binary);
  if (!file.good()) {
    OPENCL_DEBUG_LOG("[CACHE] Failed to open cache file for writing: " << cache_path);
    return;
  }

  file.write(reinterpret_cast<const char*>(binary.data()), binary.size());

  // Also save source for debugging (optional)
  if (is_debug_enabled()) {
    auto source_path = cache_path;
    source_path.replace_extension(".cl");
    std::ofstream source_file(source_path);
    source_file << source;
  }

  OPENCL_DEBUG_LOG("[CACHE] Wrote cached binary to: " << cache_path << " (" << binary.size() << " bytes)");
}

// Get compiled binary from a program
bool get_program_binary(cl_program program, std::vector<unsigned char>& binary) {
  // Get binary size
  size_t binary_size = 0;
  cl_int err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, nullptr);
  if (err != CL_SUCCESS || binary_size == 0) {
    OPENCL_DEBUG_LOG("[CACHE] Failed to get program binary size: " << err);
    return false;
  }

  // Get binary
  binary.resize(binary_size);
  unsigned char* binary_ptr = binary.data();
  err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), &binary_ptr, nullptr);
  if (err != CL_SUCCESS) {
    OPENCL_DEBUG_LOG("[CACHE] Failed to get program binary: " << err);
    binary.clear();
    return false;
  }

  return true;
}

// Create program from cached binary
cl_program create_program_from_binary(
    cl_context context,
    cl_device_id device_id,
    const std::vector<unsigned char>& binary) {
  if (binary.empty()) {
    return nullptr;
  }

  cl_int binary_status;
  cl_int err;
  size_t binary_size = binary.size();
  const unsigned char* binary_ptr = binary.data();

  cl_program program = clCreateProgramWithBinary(
      context, 1, &device_id, &binary_size, &binary_ptr, &binary_status, &err);

  if (err != CL_SUCCESS || binary_status != CL_SUCCESS) {
    OPENCL_DEBUG_LOG("[CACHE] Failed to create program from binary: err=" << err << " status=" << binary_status);
    if (program) {
      clReleaseProgram(program);
    }
    return nullptr;
  }

  // Build the program (required even for binaries on some platforms)
  err = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) {
    OPENCL_DEBUG_LOG("[CACHE] Failed to build program from binary: " << err);
    clReleaseProgram(program);
    return nullptr;
  }

  return program;
}

// ============================================================================

// Get device info string
std::string get_device_info_string(cl_device_id device, cl_device_info param) {
  size_t size = 0;
  clGetDeviceInfo(device, param, 0, nullptr, &size);
  std::string result(size, '\0');
  clGetDeviceInfo(device, param, size, &result[0], nullptr);
  // Remove null terminator if present
  if (!result.empty() && result.back() == '\0') {
    result.pop_back();
  }
  return result;
}

} // anonymous namespace

//------------------------------------------------------------------------------
// CommandEncoder implementation
//------------------------------------------------------------------------------

CommandEncoder::CommandEncoder(DeviceStream& stream)
    : stream_(stream) {}

CommandEncoder::~CommandEncoder() {
  if (current_kernel_) {
    clReleaseKernel(current_kernel_);
  }
}

void CommandEncoder::set_input_array(const array& a, int idx) {
  all_inputs_.insert(a.id());

  void* raw_ptr = const_cast<void*>(a.buffer().ptr());
  cl_mem buf;

  // Check if this is an OpenCL buffer or CPU memory
  if (allocator().is_opencl_buffer(raw_ptr)) {
    // It's an OpenCL buffer - use directly
    buf = static_cast<cl_mem>(raw_ptr);

    // Ensure buffer data is on device (unmaps if mapped to flush CPU cache)
    allocator().ensure_on_device(buf, stream_.queue);

    // Debug: OpenCL buffer input
  } else {
    // It's CPU memory - need to create a temporary GPU buffer
    // Get the actual CPU data pointer (for CPU allocator, raw_ptr() returns data after size header)
    const void* cpu_data = a.data<void>();
    size_t data_size = a.nbytes();

    OPENCL_DEBUG_LOG("[SET_ARG] Input idx=" << idx << " from CPU, creating temp buffer (" << data_size << " bytes)");

    // Create temporary buffer from CPU data
    buf = allocator().create_temp_buffer_from_cpu(cpu_data, data_size);

    // Track this temporary buffer for cleanup after kernel execution
    // We need to get the Device to add this temp buffer
    auto& dev = device(mlx::core::Device::gpu);
    dev.add_temp_buffer(buf, stream_.stream_index);

    // Temp buffer created from CPU data
  }

  cl_int err = clSetKernelArg(current_kernel_, idx, sizeof(cl_mem), &buf);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("clSetKernelArg failed: " + std::to_string(err));
  }
}

void CommandEncoder::set_output_array(array& a, int idx) {
  all_outputs_.insert(a.id());
  // Get the actual cl_mem buffer object, not the mapped host pointer
  auto buf = static_cast<cl_mem>(const_cast<void*>(a.buffer().ptr()));
  // Ensure buffer data is on device (unmaps if mapped to flush CPU cache)
  // Use the same queue as the kernel for proper synchronization
  allocator().ensure_on_device(buf, stream_.queue);

  cl_int err = clSetKernelArg(current_kernel_, idx, sizeof(cl_mem), &buf);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("clSetKernelArg failed: " + std::to_string(err));
  }
}

void CommandEncoder::register_output_array(const array& a) {
  all_outputs_.insert(a.id());
}

void CommandEncoder::set_buffer(cl_mem buf, int idx) {
  CL_CHECK(clSetKernelArg(current_kernel_, idx, sizeof(cl_mem), &buf));
}

template <typename T>
void CommandEncoder::set_bytes(const T& value, int idx) {
  CL_CHECK(clSetKernelArg(current_kernel_, idx, sizeof(T), &value));
}

// Explicit instantiations for common types
template void CommandEncoder::set_bytes<uint8_t>(const uint8_t&, int);
template void CommandEncoder::set_bytes<uint16_t>(const uint16_t&, int);
template void CommandEncoder::set_bytes<uint32_t>(const uint32_t&, int);
template void CommandEncoder::set_bytes<int8_t>(const int8_t&, int);
template void CommandEncoder::set_bytes<int16_t>(const int16_t&, int);
template void CommandEncoder::set_bytes<int>(const int&, int);  // int32_t is same as int
template void CommandEncoder::set_bytes<float>(const float&, int);
template void CommandEncoder::set_bytes<size_t>(const size_t&, int);  // uint64_t is same as size_t on this platform
template void CommandEncoder::set_bytes<long long>(const long long&, int);  // int64_t

void CommandEncoder::dispatch_threads(
    size_t global_work_size[3],
    size_t local_work_size[3],
    int work_dim) {
  if (!current_kernel_) {
    throw std::runtime_error("No kernel set for dispatch");
  }

  if (is_debug_enabled()) {
    std::string local_str = local_work_size ?
      "[" + std::to_string(local_work_size[0]) + "," + std::to_string(local_work_size[1]) + "," + std::to_string(local_work_size[2]) + "]" : "auto";
    OPENCL_DEBUG_LOG("[DISPATCH] global=[" << global_work_size[0] << "," << global_work_size[1] << "," << global_work_size[2] << "]"
      << " local=" << local_str << " dim=" << work_dim);
  }

  // Verify queue and kernel are valid
  if (stream_.queue == nullptr) {
    throw std::runtime_error("Command queue is NULL");
  }
  if (current_kernel_ == nullptr) {
    throw std::runtime_error("Kernel is NULL");
  }

  // Use provided local_work_size if non-zero, otherwise let OpenCL choose
  size_t* actual_local_size = nullptr;
  if (local_work_size != nullptr && local_work_size[0] > 0) {
    actual_local_size = local_work_size;
  }

  // IMPORTANT: Ensure all pending operations on this queue are complete before
  // dispatching a new kernel. This helps avoid race conditions on some drivers
  // (e.g., Qualcomm Adreno) where kernel executions may not properly wait for
  // previous buffer operations to complete.
  clFinish(stream_.queue);

  cl_int err = clEnqueueNDRangeKernel(
      stream_.queue,
      current_kernel_,
      work_dim,
      nullptr,  // global_work_offset
      global_work_size,
      actual_local_size,  // local_work_size - use provided if non-zero
      0,  // num_events_in_wait_list
      nullptr,  // event_wait_list
      nullptr);  // event - don't track for now

  // Wait for completion
  if (err == CL_SUCCESS) {
    err = clFinish(stream_.queue);
  }

  if (err != CL_SUCCESS) {
    throw std::runtime_error(
        std::string("clEnqueueNDRangeKernel failed: ") + std::to_string(err));
  }
}

void CommandEncoder::set_kernel(cl_kernel kernel) {
  // Retain the new kernel
  CL_CHECK(clRetainKernel(kernel));

  // Release the old kernel if it exists
  if (current_kernel_) {
    clReleaseKernel(current_kernel_);
  }

  current_kernel_ = kernel;
}

void CommandEncoder::finish() {
  CL_CHECK(clFinish(stream_.queue));
}

//------------------------------------------------------------------------------
// DeviceStream implementation
//------------------------------------------------------------------------------

DeviceStream::DeviceStream(cl_command_queue queue)
    : queue(queue), mtx(std::make_unique<std::mutex>()) {
}

DeviceStream::~DeviceStream() {
  if (queue) {
    clReleaseCommandQueue(queue);
  }
}

DeviceStream::DeviceStream(DeviceStream&& other) noexcept
    : queue(other.queue), stream_index(other.stream_index),
      encoder(std::move(other.encoder)),
      temporaries(std::move(other.temporaries)),
      temp_buffers(std::move(other.temp_buffers)),
      mtx(std::move(other.mtx)) {
  other.queue = nullptr;  // CRITICAL: Prevent double-free!
}

DeviceStream& DeviceStream::operator=(DeviceStream&& other) noexcept {
  if (this != &other) {
    // Release current queue if we have one
    if (queue) {
      clReleaseCommandQueue(queue);
    }
    queue = other.queue;
    stream_index = other.stream_index;
    encoder = std::move(other.encoder);
    temporaries = std::move(other.temporaries);
    temp_buffers = std::move(other.temp_buffers);
    mtx = std::move(other.mtx);
    other.queue = nullptr;  // CRITICAL: Prevent double-free!
  }
  return *this;
}

//------------------------------------------------------------------------------
// Device implementation
//------------------------------------------------------------------------------

Device::Device() {
  // Get platform
  cl_uint num_platforms = 0;
  CL_CHECK(clGetPlatformIDs(0, nullptr, &num_platforms));

  if (num_platforms == 0) {
    throw std::runtime_error("No OpenCL platforms found");
  }

  std::vector<cl_platform_id> platforms(num_platforms);
  CL_CHECK(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));

  // Try to find Qualcomm/Adreno platform, otherwise use first available
  platform_id_ = platforms[0];
  for (auto platform : platforms) {
    size_t size = 0;
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &size);
    std::string vendor(size, '\0');
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, size, &vendor[0], nullptr);

    if (vendor.find("Qualcomm") != std::string::npos ||
        vendor.find("QUALCOMM") != std::string::npos) {
      platform_id_ = platform;
      break;
    }
  }

  // Get GPU device
  cl_uint num_devices = 0;
  CL_CHECK(clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices));

  if (num_devices == 0) {
    throw std::runtime_error("No GPU devices found");
  }

  std::vector<cl_device_id> devices(num_devices);
  CL_CHECK(clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr));

  device_id_ = devices[0];  // Use first GPU device

  // Get device info
  device_name_ = get_device_info_string(device_id_, CL_DEVICE_NAME);
  vendor_name_ = get_device_info_string(device_id_, CL_DEVICE_VENDOR);
  opencl_version_ = get_device_info_string(device_id_, CL_DEVICE_VERSION);

  OPENCL_DEBUG_LOG("[DEVICE] OpenCL Device: " << device_name_);
  OPENCL_DEBUG_LOG("[DEVICE] Vendor: " << vendor_name_);
  OPENCL_DEBUG_LOG("[DEVICE] OpenCL Version: " << opencl_version_);

  // Create context with platform property
  cl_int err;
  cl_context_properties properties[] = {
      CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id_,
      0
  };
  context_ = clCreateContext(properties, 1, &device_id_, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) {
    std::ostringstream oss;
    oss << "OpenCL error at " << __FILE__ << ":" << __LINE__ << " - " << err;
    throw std::runtime_error(oss.str());
  }
  OPENCL_DEBUG_LOG("[DEVICE_DEBUG] Created context with platform property: " << context_);

  // Create default stream 0 so it's available for allocator and early operations
  new_queue(0);
  OPENCL_DEBUG_LOG("[DEVICE_DEBUG] Created default stream 0");

  // Run a warmup kernel to fully initialize the device
  // This helps avoid race conditions on some drivers (e.g., Qualcomm Adreno)
  warmup_device();
  OPENCL_DEBUG_LOG("[DEVICE_DEBUG] Device warmup completed");
}

Device::~Device() {
  // CRITICAL: Clear all streams (and their encoders) BEFORE releasing kernels.
  // Encoders hold references to cached kernels, so they must release their
  // references before we release the cache's references.
  {
    std::lock_guard<std::mutex> lock(stream_mtx_);
    stream_map_.clear();  // Destroys all DeviceStreams and their CommandEncoders
  }

  {
    std::lock_guard<std::mutex> lock(kernel_mtx_);
    for (auto& [key, kernel] : kernel_cache_) {
      clReleaseKernel(kernel);
    }
    kernel_cache_.clear();

    for (auto& [key, program] : program_cache_) {
      clReleaseProgram(program);
    }
    program_cache_.clear();
  }

  if (context_) {
    clReleaseContext(context_);
  }
}

void Device::new_queue(int index) {
  std::lock_guard<std::mutex> lock(stream_mtx_);

  if (stream_map_.find(index) != stream_map_.end()) {
    return;  // Queue already exists
  }

  cl_int err;
  // Use deprecated clCreateCommandQueue for compatibility with Qualcomm driver
  // clCreateCommandQueueWithProperties seems to have a bug in Qualcomm's implementation
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
  cl_command_queue queue = clCreateCommandQueue(context_, device_id_, 0, &err);
  #pragma clang diagnostic pop

  if (err != CL_SUCCESS) {
    std::ostringstream oss;
    oss << "OpenCL error at " << __FILE__ << ":" << __LINE__ << " - " << err;
    throw std::runtime_error(oss.str());
  }

  auto [it, inserted] = stream_map_.emplace(index, DeviceStream(queue));
  it->second.stream_index = index;  // Set stream index for temp buffer tracking
}

void Device::warmup_device() {
  // Run warmup kernels to fully initialize the OpenCL driver
  // This helps avoid non-deterministic behavior on some drivers (e.g., Qualcomm Adreno)
  // where the first kernel execution may fail if the driver isn't fully ready.
  // We run multiple kernels with different compilation options to ensure all
  // code paths are initialized.

  cl_int err;
  auto& stream = get_stream_(0);

  // Warmup 1: Basic kernel (no special options)
  {
    const char* source = R"(
      __kernel void warmup_basic(__global float* out) {
        out[get_global_id(0)] = 1.0f;
      }
    )";

    cl_program program = clCreateProgramWithSource(context_, 1, &source, nullptr, &err);
    if (err == CL_SUCCESS) {
      err = clBuildProgram(program, 1, &device_id_, nullptr, nullptr, nullptr);
      if (err == CL_SUCCESS) {
        cl_kernel kernel = clCreateKernel(program, "warmup_basic", &err);
        if (err == CL_SUCCESS) {
          cl_mem buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, 256 * sizeof(float), nullptr, &err);
          if (err == CL_SUCCESS) {
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
            size_t global_size = 256;
            clEnqueueNDRangeKernel(stream.queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
            clFinish(stream.queue);
            clReleaseMemObject(buffer);
          }
          clReleaseKernel(kernel);
        }
      }
      clReleaseProgram(program);
    }
  }

  // Warmup 2: CL2.0 kernel with subgroups (matches mxfp4 kernels)
  {
    const char* source = R"(
      #pragma OPENCL EXTENSION cl_khr_subgroups : enable
      __kernel void warmup_cl20(__global float* out, __local float* scratch) {
        int lid = get_local_id(0);
        float val = (float)lid;
        float sum = sub_group_reduce_add(val);
        if (get_sub_group_local_id() == 0) {
          scratch[get_sub_group_id()] = sum;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid == 0) {
          out[get_group_id(0)] = scratch[0];
        }
      }
    )";

    cl_program program = clCreateProgramWithSource(context_, 1, &source, nullptr, &err);
    if (err == CL_SUCCESS) {
      err = clBuildProgram(program, 1, &device_id_, "-cl-std=CL2.0", nullptr, nullptr);
      if (err == CL_SUCCESS) {
        cl_kernel kernel = clCreateKernel(program, "warmup_cl20", &err);
        if (err == CL_SUCCESS) {
          cl_mem buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, 256 * sizeof(float), nullptr, &err);
          if (err == CL_SUCCESS) {
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
            size_t local_size = 64;
            clSetKernelArg(kernel, 1, local_size * sizeof(float), nullptr);  // Local memory
            size_t global_size = 64;
            clEnqueueNDRangeKernel(stream.queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
            clFinish(stream.queue);
            clReleaseMemObject(buffer);
          }
          clReleaseKernel(kernel);
        }
      }
      clReleaseProgram(program);
    }
  }

  // Warmup 3: Half precision kernel (matches model inference)
  {
    const char* source = R"(
      #pragma OPENCL EXTENSION cl_khr_fp16 : enable
      __kernel void warmup_fp16(__global half* out) {
        out[get_global_id(0)] = (half)1.0f;
      }
    )";

    cl_program program = clCreateProgramWithSource(context_, 1, &source, nullptr, &err);
    if (err == CL_SUCCESS) {
      err = clBuildProgram(program, 1, &device_id_, nullptr, nullptr, nullptr);
      if (err == CL_SUCCESS) {
        cl_kernel kernel = clCreateKernel(program, "warmup_fp16", &err);
        if (err == CL_SUCCESS) {
          cl_mem buffer = clCreateBuffer(context_, CL_MEM_READ_WRITE, 256 * sizeof(cl_half), nullptr, &err);
          if (err == CL_SUCCESS) {
            clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
            size_t global_size = 256;
            clEnqueueNDRangeKernel(stream.queue, kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
            clFinish(stream.queue);
            clReleaseMemObject(buffer);
          }
          clReleaseKernel(kernel);
        }
      }
      clReleaseProgram(program);
    }
  }

  // Final sync to ensure all warmup is complete
  clFinish(stream.queue);

  // Small delay to allow driver to fully stabilize
  // This helps with Qualcomm Adreno drivers which can have race conditions
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  OPENCL_DEBUG_LOG("[WARMUP] Device warmup completed successfully");
}

cl_command_queue Device::get_queue(Stream stream) {
  auto& device_stream = get_stream_(stream.index);
  return device_stream.queue;
}

CommandEncoder& Device::get_command_encoder(int index) {
  auto& stream = get_stream_(index);

  if (!stream.encoder) {
    stream.encoder = std::make_unique<CommandEncoder>(stream);
  }
  return *stream.encoder;
}

void Device::end_encoding(int index) {
  auto& stream = get_stream_(index);

  if (stream.encoder) {
    stream.encoder->finish();  // clFinish - sync point
    stream.encoder.reset();
  }

  // Release temporary cl_mem buffers (safe now after clFinish)
  size_t freed_memory = 0;
  size_t num_buffers = stream.temp_buffers.size();
  for (cl_mem buf : stream.temp_buffers) {
    // Track memory being freed
    size_t buf_size = 0;
    clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
    freed_memory += buf_size;
    temp_buffer_memory_ -= buf_size;

    clReleaseMemObject(buf);
  }
  stream.temp_buffers.clear();

  if (num_buffers > 0) {
    OPENCL_DEBUG_LOG("[TEMP_BUF] end_encoding: freed " << num_buffers << " buffers, "
              << freed_memory << " bytes, remaining_temp=" << temp_buffer_memory_.load()
              << " stream=" << index);
  }

  // Clear temporaries
  stream.temporaries.clear();
}

void Device::synchronize(int index) {
  auto& stream = get_stream_(index);
  // clFinish waits for all commands on the queue to complete
  clFinish(stream.queue);
  // Add a barrier to ensure memory visibility
  // On some platforms, clFinish alone may not ensure CPU cache coherency
  clEnqueueBarrierWithWaitList(stream.queue, 0, nullptr, nullptr);
  clFinish(stream.queue);
}

cl_program Device::compile_program(
    const std::string& source,
    const std::string& options) {
  // NOTE: Caller must hold kernel_mtx_!
  // This is called from get_kernel() which already holds the lock

  // Create a hash from source + options for in-memory cache
  std::string cache_key = source + "|" + options;

  // Check in-memory cache first
  auto it = program_cache_.find(cache_key);
  if (it != program_cache_.end()) {
    return it->second;
  }

  // Get disk cache path
  std::string device_hash = get_device_hash(device_name_, vendor_name_, opencl_version_);
  auto cache_dir = get_cache_directory(device_hash);
  auto cache_path = get_cache_path(cache_dir, source, options);

  // Try to load from disk cache
  cl_program program = nullptr;
  std::vector<unsigned char> binary;
  if (read_cached_binary(cache_path, binary)) {
    program = create_program_from_binary(context_, device_id_, binary);
    if (program) {
      OPENCL_DEBUG_LOG("[COMPILE_DEBUG] Loaded program from disk cache");
      program_cache_[cache_key] = program;
      return program;
    }
    // If loading failed, fall through to compile from source
    OPENCL_DEBUG_LOG("[COMPILE_DEBUG] Cached binary invalid, recompiling");
  }

  // Compile program from source
  OPENCL_DEBUG_LOG("[COMPILE_DEBUG] Creating program from source (" << source.length() << " chars)");

  // Dump kernel source for first compilation (helps debug)
  if (is_debug_enabled()) {
    static bool first_compile = true;
    if (first_compile) {
      OPENCL_DEBUG_LOG("[COMPILE_DEBUG] ===== KERNEL SOURCE (first compilation) =====");
      OPENCL_DEBUG_LOG(source);
      OPENCL_DEBUG_LOG("[COMPILE_DEBUG] ===== END KERNEL SOURCE =====");
      first_compile = false;
    }
  }

  cl_int err;
  const char* source_str = source.c_str();
  size_t source_len = source.length();

  program = clCreateProgramWithSource(
      context_, 1, &source_str, &source_len, &err);
  if (err != CL_SUCCESS) {
    std::ostringstream oss;
    oss << "OpenCL error at " << __FILE__ << ":" << __LINE__ << " - " << err;
    throw std::runtime_error(oss.str());
  }
  OPENCL_DEBUG_LOG("[COMPILE_DEBUG] Program created, building...");

  err = clBuildProgram(
      program, 1, &device_id_, options.c_str(), nullptr, nullptr);

  OPENCL_DEBUG_LOG("[COMPILE_DEBUG] clBuildProgram returned: " << err);

  if (err != CL_SUCCESS) {
    // Get build log
    size_t log_size = 0;
    cl_int log_err = clGetProgramBuildInfo(
        program, device_id_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    OPENCL_DEBUG_LOG("[COMPILE_DEBUG] Build failed, getting build log (size query returned: " << log_err << ", log_size=" << log_size << ")");

    std::string log(log_size, '\0');
    log_err = clGetProgramBuildInfo(
        program, device_id_, CL_PROGRAM_BUILD_LOG, log_size, &log[0], nullptr);
    OPENCL_DEBUG_LOG("[COMPILE_DEBUG] clGetProgramBuildInfo (log fetch) returned: " << log_err);

    clReleaseProgram(program);

    std::ostringstream oss;
    oss << "OpenCL program build failed:\n" << log;
    throw std::runtime_error(oss.str());
  }

  // Save to disk cache
  if (!cache_path.empty()) {
    binary.clear();
    if (get_program_binary(program, binary)) {
      write_cached_binary(cache_path, binary, source);
    }
  }

  program_cache_[cache_key] = program;
  return program;
}

cl_kernel Device::get_kernel(
    const std::string& name,
    const std::string& source,
    const std::string& options) {
  std::lock_guard<std::mutex> lock(kernel_mtx_);

  std::string cache_key = source + "|" + options + "|" + name;

  auto it = kernel_cache_.find(cache_key);
  if (it != kernel_cache_.end()) {
    return it->second;
  }

  OPENCL_DEBUG_LOG("[KERNEL] Compiling: " << name);

  // Compile program (will use cache if available)
  cl_program program = compile_program(source, options);

  // Create kernel
  cl_int err;
  cl_kernel kernel = clCreateKernel(program, name.c_str(), &err);
  if (err != CL_SUCCESS) {
    std::ostringstream oss;
    oss << "OpenCL error at " << __FILE__ << ":" << __LINE__ << " - " << err;
    throw std::runtime_error(oss.str());
  }

  kernel_cache_[cache_key] = kernel;

  return kernel;
}

void Device::add_temporary(array arr, int index) {
  auto& stream = get_stream_(index);
  stream.temporaries.push_back(std::move(arr));
}

void Device::add_temporaries(std::vector<array> arrays, int index) {
  auto& stream = get_stream_(index);
  stream.temporaries.insert(
      stream.temporaries.end(),
      std::make_move_iterator(arrays.begin()),
      std::make_move_iterator(arrays.end()));
}

void Device::add_temp_buffer(cl_mem buf, int index) {
  auto& stream = get_stream_(index);
  stream.temp_buffers.push_back(buf);

  // Track temp buffer memory for debugging
  size_t buf_size = 0;
  clGetMemObjectInfo(buf, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
  temp_buffer_memory_ += buf_size;

  OPENCL_DEBUG_LOG("[TEMP_BUF] add_temp_buffer: buf=" << buf << " size=" << buf_size
            << " total_temp=" << temp_buffer_memory_.load()
            << " stream=" << index << " pending=" << stream.temp_buffers.size());
}

DeviceStream& Device::get_stream_(int index) {
  auto it = stream_map_.find(index);
  if (it == stream_map_.end()) {
    throw std::runtime_error(
        "Stream " + std::to_string(index) + " not found");
  }
  return it->second;
}

Device& device(mlx::core::Device) {
  static Device device_;
  return device_;
}

} // namespace mlx::core::opencl

// Implementation of mlx::core::gpu interface for OpenCL backend
namespace mlx::core::gpu {

// Note: is_available() is defined in eval.cpp

int device_count() {
  // OpenCL backend currently supports single device
  return is_available() ? 1 : 0;
}

const std::unordered_map<std::string, std::variant<std::string, size_t>>&
device_info(int device_index) {
  static std::unordered_map<std::string, std::variant<std::string, size_t>> info;

  if (!is_available() || device_index != 0) {
    info.clear();
    return info;
  }

  auto& dev = opencl::device(Device::gpu);
  info["device_name"] = dev.get_name();
  info["vendor"] = dev.get_vendor();
  info["opencl_version"] = dev.get_version();

  // Query global memory size from OpenCL
  cl_ulong mem_size = 0;
  clGetDeviceInfo(dev.device_id(), CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem_size), &mem_size, nullptr);
  info["memory_size"] = static_cast<size_t>(mem_size);

  return info;
}

} // namespace mlx::core::gpu
