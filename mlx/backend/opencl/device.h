// Copyright © 2025 MLX Contributors

#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "mlx/array.h"
#include "mlx/device.h"

namespace mlx::core::opencl {

struct DeviceStream;

/**
 * CommandEncoder wraps an OpenCL command queue and manages kernel execution.
 * Similar to Metal's CommandEncoder but adapted for OpenCL.
 */
struct CommandEncoder {
  explicit CommandEncoder(DeviceStream& stream);
  CommandEncoder(const CommandEncoder&) = delete;
  CommandEncoder& operator=(const CommandEncoder&) = delete;
  ~CommandEncoder();

  // Set kernel arguments
  void set_input_array(const array& a, int idx);
  void set_output_array(array& a, int idx);
  void register_output_array(const array& a);

  // Set kernel argument from buffer
  void set_buffer(cl_mem buf, int idx);

  // Set scalar kernel arguments
  template <typename T>
  void set_bytes(const T& value, int idx);

  // Execute kernel
  void dispatch_threads(
      size_t global_work_size[3],
      size_t local_work_size[3],
      int work_dim = 1);

  // Set current kernel
  void set_kernel(cl_kernel kernel);

  // Finish all pending operations
  void finish();

  // Get inputs/outputs for dependency tracking
  std::unordered_set<std::uintptr_t>& inputs() { return all_inputs_; }
  std::unordered_set<std::uintptr_t>& outputs() { return all_outputs_; }

 private:
  DeviceStream& stream_;
  cl_kernel current_kernel_{nullptr};
  std::unordered_set<std::uintptr_t> all_inputs_;
  std::unordered_set<std::uintptr_t> all_outputs_;
};

/**
 * DeviceStream manages an OpenCL command queue and associated state.
 */
struct DeviceStream {
  DeviceStream(cl_command_queue queue);
  ~DeviceStream();

  // Make movable for unordered_map
  DeviceStream(DeviceStream&& other) noexcept;
  DeviceStream& operator=(DeviceStream&& other) noexcept;

  // Delete copy operations
  DeviceStream(const DeviceStream&) = delete;
  DeviceStream& operator=(const DeviceStream&) = delete;

  cl_command_queue queue;
  int stream_index{0};  // Index of this stream for temp buffer tracking
  std::unique_ptr<CommandEncoder> encoder{nullptr};
  std::vector<array> temporaries;
  std::vector<cl_mem> temp_buffers;  // Raw cl_mem buffers to release after sync
  std::unique_ptr<std::mutex> mtx;
};

/**
 * Device represents an OpenCL device (GPU) and manages:
 * - Platform and device initialization
 * - Context and command queues
 * - Kernel compilation and caching
 * - Memory management
 */
class Device {
 public:
  Device();
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  ~Device();

  // OpenCL device accessors
  cl_device_id device_id() const { return device_id_; }
  cl_context context() const { return context_; }
  cl_platform_id platform_id() const { return platform_id_; }

  // Device information
  const std::string& get_name() const { return device_name_; }
  const std::string& get_vendor() const { return vendor_name_; }
  const std::string& get_version() const { return opencl_version_; }

  // Queue management
  void new_queue(int index);
  cl_command_queue get_queue(Stream stream);

  // Encoder management
  CommandEncoder& get_command_encoder(int index);
  void end_encoding(int index);
  void synchronize(int index);  // Just clFinish without resetting encoder
  void warmup_device();  // Initialize driver with a warmup kernel

  // Kernel compilation and caching
  cl_program compile_program(
      const std::string& source,
      const std::string& options = "");

  cl_kernel get_kernel(
      const std::string& name,
      const std::string& source,
      const std::string& options = "");

  // Temporary array tracking
  void add_temporary(array arr, int index);
  void add_temporaries(std::vector<array> arrays, int index);

  // Temporary cl_mem buffer tracking (released after sync)
  void add_temp_buffer(cl_mem buf, int index);

  // Memory tracking for debugging
  size_t get_temp_buffer_memory() const { return temp_buffer_memory_.load(); }

 private:
  DeviceStream& get_stream_(int index);

  // OpenCL objects
  cl_platform_id platform_id_;
  cl_device_id device_id_;
  cl_context context_;

  // Device info
  std::string device_name_;
  std::string vendor_name_;
  std::string opencl_version_;

  // Stream management
  std::unordered_map<int32_t, DeviceStream> stream_map_;
  std::mutex stream_mtx_;

  // Kernel cache: source_hash -> program
  std::unordered_map<std::string, cl_program> program_cache_;
  // Kernel cache: program + name -> kernel
  std::unordered_map<std::string, cl_kernel> kernel_cache_;
  std::mutex kernel_mtx_;

  // Memory tracking for debugging (atomic for thread safety)
  std::atomic<size_t> temp_buffer_memory_{0};
};

/**
 * Get the global OpenCL device instance for the given MLX device.
 */
Device& device(mlx::core::Device d);

} // namespace mlx::core::opencl
