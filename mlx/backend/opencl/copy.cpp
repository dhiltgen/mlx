// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/copy.h"
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/backend/common/copy.h"
#include "mlx/backend/cpu/copy.h"
#include "mlx/primitives.h"

#include <complex>
#include <iostream>
#include <numeric>
#include <sstream>

namespace mlx::core {

void copy_gpu(
    const array& src,
    array& dst,
    CopyType copy_type,
    const Stream& s) {

  OPENCL_DEBUG_LOG("[copy_gpu] src.size=" << src.size()
            << " dst.size=" << dst.size());

  // Allocate output buffer if needed, or donate if possible
  bool donated = set_copy_output_data(src, dst, copy_type);
  if (donated && src.dtype() == dst.dtype()) {
    // If the output has the same type as the input then there is nothing to
    // copy, just use the buffer.
    OPENCL_DEBUG_LOG("[copy_gpu] Buffer donated, no copy needed");
    return;
  }

  if (!donated) {
    OPENCL_DEBUG_LOG("[copy_gpu] Allocated new buffer for dst");
  }

  // When donated but types differ, still need to do a type conversion copy
  // GeneralGeneral can be treated as General since dst is always contiguous
  if (copy_type == CopyType::GeneralGeneral) {
    copy_type = CopyType::General;
  }

  auto& dev = opencl::device(Device::gpu);
  cl_command_queue queue = dev.get_queue(s);
  
  switch (copy_type) {
    case CopyType::Scalar: {
      // Fill array with a scalar value using OpenCL kernel
      OPENCL_DEBUG_LOG("[copy_gpu] CopyType::Scalar - filling " << dst.size() << " elements");

      std::string type_name = opencl::type_to_name(dst.dtype());
      size_t n = dst.size();

      // Build fill kernel: out[i] = value
      // For complex types, pass real/imag separately since float2 can't be passed directly
      std::ostringstream kernel_source;
      kernel_source << opencl::get_kernel_preamble();
      if (dst.dtype() == complex64) {
        kernel_source << "\n__kernel void fill_" << type_name << "(\n";
        kernel_source << "    __global " << type_name << "* out,\n";
        kernel_source << "    float value_re,\n";
        kernel_source << "    float value_im,\n";
        kernel_source << "    int n) {\n";
        kernel_source << "  int idx = get_global_id(0);\n";
        kernel_source << "  if (idx < n) {\n";
        kernel_source << "    out[idx] = (float2)(value_re, value_im);\n";
        kernel_source << "  }\n";
        kernel_source << "}\n";
      } else if (dst.dtype() == float16) {
        // For half, pass as uint to avoid scalar half argument issues
        kernel_source << "\n__kernel void fill_" << type_name << "(\n";
        kernel_source << "    __global " << type_name << "* out,\n";
        kernel_source << "    uint value_bits,\n";
        kernel_source << "    int n) {\n";
        kernel_source << "  int idx = get_global_id(0);\n";
        kernel_source << "  if (idx < n) {\n";
        kernel_source << "    out[idx] = as_half((ushort)value_bits);\n";
        kernel_source << "  }\n";
        kernel_source << "}\n";
      } else if (dst.dtype() == bfloat16) {
        // For bfloat16, pass as uint
        kernel_source << "\n__kernel void fill_" << type_name << "(\n";
        kernel_source << "    __global " << type_name << "* out,\n";
        kernel_source << "    uint value_bits,\n";
        kernel_source << "    int n) {\n";
        kernel_source << "  int idx = get_global_id(0);\n";
        kernel_source << "  if (idx < n) {\n";
        kernel_source << "    out[idx] = (bfloat16_t)value_bits;\n";
        kernel_source << "  }\n";
        kernel_source << "}\n";
      } else {
        kernel_source << "\n__kernel void fill_" << type_name << "(\n";
        kernel_source << "    __global " << type_name << "* out,\n";
        kernel_source << "    " << type_name << " value,\n";
        kernel_source << "    int n) {\n";
        kernel_source << "  int idx = get_global_id(0);\n";
        kernel_source << "  if (idx < n) {\n";
        kernel_source << "    out[idx] = value;\n";
        kernel_source << "  }\n";
        kernel_source << "}\n";
      }

      std::string source_str = kernel_source.str();
      std::string kernel_name = "fill_" + type_name;

      // Get or compile kernel
      cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
      auto& encoder = dev.get_command_encoder(s.index);
      encoder.set_kernel(kernel);

      // Set arguments
      cl_mem dst_buf = static_cast<cl_mem>(dst.buffer().ptr());
      OPENCL_DEBUG_LOG("[copy_gpu] Fill kernel writing to buffer: " << dst_buf);
      encoder.set_output_array(dst, 0);

      // Set the scalar value from source array (which has size 1)
      const void* src_data = src.data<void>();
      OPENCL_DEBUG_LOG("[copy_gpu] src_data pointer: " << src_data
                << " src.data_size=" << src.data_size());
      if (dst.dtype() == float32) {
        OPENCL_DEBUG_LOG("[copy_gpu] Scalar float value to fill: " << *static_cast<const float*>(src_data));
      }
      switch (dst.dtype()) {
        case bool_:
          encoder.set_bytes(*static_cast<const uint8_t*>(src_data), 1);
          break;
        case uint8:
          encoder.set_bytes(*static_cast<const uint8_t*>(src_data), 1);
          break;
        case uint16:
          encoder.set_bytes(*static_cast<const uint16_t*>(src_data), 1);
          break;
        case uint32:
          encoder.set_bytes(*static_cast<const uint32_t*>(src_data), 1);
          break;
        case uint64:
          encoder.set_bytes(*static_cast<const uint64_t*>(src_data), 1);
          break;
        case int8:
          encoder.set_bytes(*static_cast<const int8_t*>(src_data), 1);
          break;
        case int16:
          encoder.set_bytes(*static_cast<const int16_t*>(src_data), 1);
          break;
        case int32:
          encoder.set_bytes(*static_cast<const int32_t*>(src_data), 1);
          break;
        case int64:
          encoder.set_bytes(*static_cast<const int64_t*>(src_data), 1);
          break;
        case float16: {
          // Pass as uint to avoid scalar half argument issues
          uint32_t bits = *static_cast<const uint16_t*>(src_data);
          encoder.set_bytes(bits, 1);
          break;
        }
        case float32:
          encoder.set_bytes(*static_cast<const float*>(src_data), 1);
          break;
        case bfloat16: {
          // Pass as uint
          uint32_t bits = *static_cast<const uint16_t*>(src_data);
          encoder.set_bytes(bits, 1);
          break;
        }
        case complex64: {
          // Pass real and imaginary parts separately
          const auto* cval = static_cast<const complex64_t*>(src_data);
          encoder.set_bytes(cval->real(), 1);
          encoder.set_bytes(cval->imag(), 2);
          break;
        }
        default:
          throw std::runtime_error("[copy_gpu] Unsupported dtype for scalar fill");
      }

      // Set size argument (at index 2 for most types, 3 for complex which has 2 value args)
      int n_arg_idx = (dst.dtype() == complex64) ? 3 : 2;
      encoder.set_bytes(static_cast<int>(n), n_arg_idx);

      // Dispatch
      size_t global_size[3] = {n, 0, 0};
      encoder.dispatch_threads(global_size, nullptr, 1);

      OPENCL_DEBUG_LOG("[copy_gpu] Scalar fill dispatched for " << n << " elements");
      break;
    }
    
    case CopyType::Vector: {
      // Contiguous copy - may need type conversion
      OPENCL_DEBUG_LOG("[copy_gpu] CopyType::Vector - contiguous copy");

      // If types match, use simple buffer copy
      if (src.dtype() == dst.dtype()) {
        // Use data_size() which is the actual stored element count, not size()
        // which is the logical element count (may differ for views/broadcasts)
        size_t bytes = src.data_size() * src.itemsize();
        cl_mem src_buf = static_cast<cl_mem>(const_cast<void*>(src.buffer().ptr()));
        cl_mem dst_buf = static_cast<cl_mem>(dst.buffer().ptr());

        // Query actual buffer sizes for debugging
        size_t src_buf_size = 0, dst_buf_size = 0;
        clGetMemObjectInfo(src_buf, CL_MEM_SIZE, sizeof(size_t), &src_buf_size, nullptr);
        clGetMemObjectInfo(dst_buf, CL_MEM_SIZE, sizeof(size_t), &dst_buf_size, nullptr);

        OPENCL_DEBUG_LOG("[copy_gpu] Vector copy: bytes=" << bytes
                  << " src_buf=" << src_buf << " (size=" << src_buf_size << ")"
                  << " dst_buf=" << dst_buf << " (size=" << dst_buf_size << ")"
                  << " src==dst? " << (src_buf == dst_buf));

        // Check if copy is valid
        if (bytes > src_buf_size || bytes > dst_buf_size) {
          OPENCL_DEBUG_LOG("[copy_gpu] ERROR: Copy size exceeds buffer size!");
          throw std::runtime_error("[copy_gpu] Copy size exceeds buffer size");
        }

        if (src_buf == dst_buf) {
          // Same buffer - this is a no-op
          OPENCL_DEBUG_LOG("[copy_gpu] Same buffer, skipping copy");
          break;
        }

        // Account for source and destination array offsets (for sliced arrays)
        size_t src_byte_offset = src.offset();
        size_t dst_byte_offset = dst.offset();

        // Validate that offsets + bytes don't exceed buffer sizes
        if (src_byte_offset + bytes > src_buf_size) {
          OPENCL_DEBUG_LOG("[copy_gpu] ERROR: src_offset + bytes exceeds buffer! offset="
                    << src_byte_offset << " bytes=" << bytes << " buf_size=" << src_buf_size);
          throw std::runtime_error("[copy_gpu] Source offset + bytes exceeds buffer size");
        }
        if (dst_byte_offset + bytes > dst_buf_size) {
          OPENCL_DEBUG_LOG("[copy_gpu] ERROR: dst_offset + bytes exceeds buffer! offset="
                    << dst_byte_offset << " bytes=" << bytes << " buf_size=" << dst_buf_size);
          throw std::runtime_error("[copy_gpu] Destination offset + bytes exceeds buffer size");
        }

        OPENCL_DEBUG_LOG("[copy_gpu] Vector copy with offsets: src_offset=" << src_byte_offset
                  << " dst_offset=" << dst_byte_offset);

        cl_int err = clEnqueueCopyBuffer(
            queue,
            src_buf,
            dst_buf,
            src_byte_offset,  // src_offset - use actual array offset
            dst_byte_offset,  // dst_offset - use actual array offset
            bytes,
            0,  // num_events_in_wait_list
            nullptr,  // event_wait_list
            nullptr   // event
        );

        if (err != CL_SUCCESS) {
          throw std::runtime_error(
              std::string("[copy_gpu] clEnqueueCopyBuffer failed: ") + std::to_string(err));
        }

        OPENCL_DEBUG_LOG("[copy_gpu] Copied " << bytes << " bytes (same type)");
      } else {
        // Type conversion needed - use kernel
        OPENCL_DEBUG_LOG("[copy_gpu] Type conversion needed");

        std::string src_type = opencl::type_to_name(src.dtype());
        std::string dst_type = opencl::type_to_name(dst.dtype());
        size_t n = src.size();

        // Get the source and destination offsets in elements (arrays may be slices with non-zero offset)
        int64_t src_offset_bytes = src.offset();
        int64_t src_offset_elements = src_offset_bytes / src.itemsize();
        int64_t dst_offset_bytes = dst.offset();
        int64_t dst_offset_elements = dst_offset_bytes / dst.itemsize();

        OPENCL_DEBUG_LOG("[copy_gpu] Type conversion: src_offset_bytes=" << src_offset_bytes
                  << " src_offset_elements=" << src_offset_elements
                  << " dst_offset_bytes=" << dst_offset_bytes
                  << " dst_offset_elements=" << dst_offset_elements);

        // Build type conversion kernel with offset support for both src and dst
        std::ostringstream kernel_source;
        kernel_source << opencl::get_kernel_preamble();
        kernel_source << "\n__kernel void copy_" << src_type << "_to_" << dst_type << "_offset(\n";
        kernel_source << "    __global const " << src_type << "* src,\n";
        kernel_source << "    __global " << dst_type << "* dst,\n";
        kernel_source << "    int n,\n";
        kernel_source << "    int src_offset,\n";
        kernel_source << "    int dst_offset) {\n";
        kernel_source << "  int idx = get_global_id(0);\n";
        kernel_source << "  if (idx < n) {\n";
        kernel_source << "    int src_idx = idx + src_offset;\n";
        kernel_source << "    int dst_idx = idx + dst_offset;\n";

        // Handle type conversions properly for bfloat16 and half
        bool src_is_bf16 = (src.dtype() == bfloat16);
        bool dst_is_bf16 = (dst.dtype() == bfloat16);
        bool src_is_half = (src.dtype() == float16);
        bool dst_is_half = (dst.dtype() == float16);

        if (dst.dtype() == bool_) {
          // For boolean destination, convert non-zero to 1
          if (src_is_bf16) {
            kernel_source << "    dst[dst_idx] = (bfloat16_to_float(src[src_idx]) != 0.0f) ? 1 : 0;\n";
          } else if (src_is_half) {
            kernel_source << "    dst[dst_idx] = (convert_float(src[src_idx]) != 0.0f) ? 1 : 0;\n";
          } else {
            kernel_source << "    dst[dst_idx] = (src[src_idx] != 0) ? 1 : 0;\n";
          }
        } else if (src_is_bf16 && !dst_is_bf16) {
          // bfloat16 source to other type
          kernel_source << "    dst[dst_idx] = (" << dst_type << ")bfloat16_to_float(src[src_idx]);\n";
        } else if (!src_is_bf16 && dst_is_bf16) {
          // Other type to bfloat16 destination
          kernel_source << "    dst[dst_idx] = float_to_bfloat16((float)src[src_idx]);\n";
        } else if (src_is_half && !dst_is_half) {
          // half source to other type
          kernel_source << "    dst[dst_idx] = (" << dst_type << ")convert_float(src[src_idx]);\n";
        } else if (!src_is_half && dst_is_half) {
          // Other type to half destination
          kernel_source << "    dst[dst_idx] = convert_half((float)src[src_idx]);\n";
        } else if (dst.dtype() == complex64 && src.dtype() != complex64) {
          // Scalar to complex64 - set imaginary part to 0
          kernel_source << "    dst[dst_idx] = (float2)((float)src[src_idx], 0.0f);\n";
        } else if (dst.dtype() != complex64 && src.dtype() == complex64) {
          // Complex64 to scalar - extract real part
          kernel_source << "    dst[dst_idx] = (" << dst_type << ")src[src_idx].x;\n";
        } else {
          // Standard conversion
          kernel_source << "    dst[dst_idx] = (" << dst_type << ")src[src_idx];\n";
        }
        kernel_source << "  }\n";
        kernel_source << "}\n";

        std::string source_str = kernel_source.str();
        std::string kernel_name = "copy_" + src_type + "_to_" + dst_type + "_offset";

        // Get or compile kernel
        cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
        auto& encoder = dev.get_command_encoder(s.index);
        encoder.set_kernel(kernel);

        // Set arguments
        encoder.set_input_array(src, 0);
        encoder.set_output_array(dst, 1);
        encoder.set_bytes(static_cast<int>(n), 2);
        encoder.set_bytes(static_cast<int>(src_offset_elements), 3);
        encoder.set_bytes(static_cast<int>(dst_offset_elements), 4);

        // Dispatch
        size_t global_size[3] = {n, 0, 0};
        encoder.dispatch_threads(global_size, nullptr, 1);
        dev.end_encoding(s.index);

        OPENCL_DEBUG_LOG("[copy_gpu] Type conversion dispatched for " << n << " elements with src_offset=" << src_offset_elements << " dst_offset=" << dst_offset_elements);
      }
      break;
    }
    
    case CopyType::General:
    case CopyType::GeneralGeneral: {
      // Strided copy - use kernel that handles strides
      OPENCL_DEBUG_LOG("[copy_gpu] General/strided copy");

      std::string src_type = opencl::type_to_name(src.dtype());
      std::string dst_type = opencl::type_to_name(dst.dtype());
      size_t n = dst.size();
      int ndim = dst.ndim();

      if (ndim == 0) {
        // Scalar case - just copy the value
        ndim = 1;
      }

      // Build strided copy kernel that flattens the index computation
      // Each thread handles one output element
      std::ostringstream kernel_source;
      kernel_source << opencl::get_kernel_preamble();
      kernel_source << "\n__kernel void copy_strided_" << src_type << "_to_" << dst_type << "(\n";
      kernel_source << "    __global const " << src_type << "* src,\n";
      kernel_source << "    __global " << dst_type << "* dst,\n";
      kernel_source << "    __global const int* shape,\n";
      kernel_source << "    __global const long* src_strides,\n";
      kernel_source << "    __global const long* dst_strides,\n";
      kernel_source << "    int ndim,\n";
      kernel_source << "    long src_base_offset,\n";  // Base offset in elements
      kernel_source << "    int n) {\n";
      kernel_source << "  int idx = get_global_id(0);\n";
      kernel_source << "  if (idx >= n) return;\n\n";
      kernel_source << "  // Convert flat index to multi-dimensional indices and compute offsets\n";
      kernel_source << "  long src_offset = src_base_offset;\n";  // Start with base offset
      kernel_source << "  int dst_offset = 0;\n";
      kernel_source << "  int remaining = idx;\n";
      kernel_source << "  for (int d = ndim - 1; d >= 0; d--) {\n";
      kernel_source << "    int coord = remaining % shape[d];\n";
      kernel_source << "    remaining /= shape[d];\n";
      kernel_source << "    src_offset += coord * src_strides[d];\n";
      kernel_source << "    dst_offset += coord * dst_strides[d];\n";
      kernel_source << "  }\n\n";

      // Handle type conversions properly for bfloat16 and half
      bool src_is_bf16 = (src.dtype() == bfloat16);
      bool dst_is_bf16 = (dst.dtype() == bfloat16);
      bool src_is_half = (src.dtype() == float16);
      bool dst_is_half = (dst.dtype() == float16);

      if (dst.dtype() == bool_) {
        // For boolean destination, convert non-zero to 1
        if (src_is_bf16) {
          kernel_source << "  dst[dst_offset] = (bfloat16_to_float(src[src_offset]) != 0.0f) ? 1 : 0;\n";
        } else if (src_is_half) {
          kernel_source << "  dst[dst_offset] = (convert_float(src[src_offset]) != 0.0f) ? 1 : 0;\n";
        } else {
          kernel_source << "  dst[dst_offset] = (src[src_offset] != 0) ? 1 : 0;\n";
        }
      } else if (src_is_bf16 && !dst_is_bf16) {
        // bfloat16 source to other type
        kernel_source << "  dst[dst_offset] = (" << dst_type << ")bfloat16_to_float(src[src_offset]);\n";
      } else if (!src_is_bf16 && dst_is_bf16) {
        // Other type to bfloat16 destination
        kernel_source << "  dst[dst_offset] = float_to_bfloat16((float)src[src_offset]);\n";
      } else if (src_is_half && !dst_is_half) {
        // half source to other type
        kernel_source << "  dst[dst_offset] = (" << dst_type << ")convert_float(src[src_offset]);\n";
      } else if (!src_is_half && dst_is_half) {
        // Other type to half destination
        kernel_source << "  dst[dst_offset] = convert_half((float)src[src_offset]);\n";
      } else {
        // Standard conversion
        kernel_source << "  dst[dst_offset] = (" << dst_type << ")src[src_offset];\n";
      }
      kernel_source << "}\n";

      std::string source_str = kernel_source.str();
      std::string kernel_name = "copy_strided_" + src_type + "_to_" + dst_type;

      // Get or compile kernel
      cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
      auto& encoder = dev.get_command_encoder(s.index);
      encoder.set_kernel(kernel);

      // Create buffers for shape and strides
      std::vector<int> shape_vec(dst.shape().begin(), dst.shape().end());
      if (shape_vec.empty()) shape_vec.push_back(1);

      std::vector<int64_t> src_strides_vec(src.strides().begin(), src.strides().end());
      std::vector<int64_t> dst_strides_vec(dst.strides().begin(), dst.strides().end());
      if (src_strides_vec.empty()) src_strides_vec.push_back(1);
      if (dst_strides_vec.empty()) dst_strides_vec.push_back(1);

      // Allocate temporary buffers for shape/strides
      cl_int err;
      cl_mem shape_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         shape_vec.size() * sizeof(int), shape_vec.data(), &err);
      if (err != CL_SUCCESS) throw std::runtime_error("[copy_gpu] Failed to create shape buffer");

      cl_mem src_strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                               src_strides_vec.size() * sizeof(int64_t), src_strides_vec.data(), &err);
      if (err != CL_SUCCESS) {
        clReleaseMemObject(shape_buf);
        throw std::runtime_error("[copy_gpu] Failed to create src_strides buffer");
      }

      cl_mem dst_strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                               dst_strides_vec.size() * sizeof(int64_t), dst_strides_vec.data(), &err);
      if (err != CL_SUCCESS) {
        clReleaseMemObject(shape_buf);
        clReleaseMemObject(src_strides_buf);
        throw std::runtime_error("[copy_gpu] Failed to create dst_strides buffer");
      }

      // Set arguments
      encoder.set_input_array(src, 0);
      encoder.set_output_array(dst, 1);
      clSetKernelArg(kernel, 2, sizeof(cl_mem), &shape_buf);
      clSetKernelArg(kernel, 3, sizeof(cl_mem), &src_strides_buf);
      clSetKernelArg(kernel, 4, sizeof(cl_mem), &dst_strides_buf);
      encoder.set_bytes(ndim, 5);
      // Pass the source array's base offset (convert from bytes to elements)
      int64_t src_base_offset = src.offset() / size_of(src.dtype());
      encoder.set_bytes(src_base_offset, 6);
      encoder.set_bytes(static_cast<int>(n), 7);

      // Dispatch
      size_t global_size[3] = {n, 0, 0};
      encoder.dispatch_threads(global_size, nullptr, 1);
      dev.end_encoding(s.index);

      // Clean up temp buffers
      clReleaseMemObject(shape_buf);
      clReleaseMemObject(src_strides_buf);
      clReleaseMemObject(dst_strides_buf);

      OPENCL_DEBUG_LOG("[copy_gpu] Strided copy dispatched for " << n << " elements, ndim=" << ndim);
      break;
    }
  }
  
  // Ensure copy completes
  clFinish(queue);
}

void copy_gpu_inplace(
    const array& src,
    array& dst,
    const SmallVector<int, 10>& data_shape,
    const SmallVector<int64_t, 10>& strides,
    const SmallVector<int64_t, 10>& out_strides,
    int64_t src_offset,
    int64_t dst_offset,
    CopyType copy_type,
    const Stream& s,
    std::optional<array> indices_src,
    std::optional<array> indices_dst) {

  auto& dev = opencl::device(Device::gpu);
  cl_command_queue queue = dev.get_queue(s);

  std::string src_type = opencl::type_to_name(src.dtype());
  std::string dst_type = opencl::type_to_name(dst.dtype());

  // Calculate total number of elements
  size_t n = 1;
  for (auto s : data_shape) n *= s;
  if (n == 0) return;

  int ndim = data_shape.size();
  if (ndim == 0) ndim = 1;

  bool has_dynamic_src = indices_src.has_value();
  bool has_dynamic_dst = indices_dst.has_value();

  // Build strided copy kernel with offset support
  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();

  std::string kernel_suffix = (has_dynamic_src || has_dynamic_dst) ? "_dynamic" : "";
  kernel_source << "\n__kernel void copy_inplace_" << src_type << "_to_" << dst_type << kernel_suffix << "(\n";
  kernel_source << "    __global const " << src_type << "* src,\n";
  kernel_source << "    __global " << dst_type << "* dst,\n";
  kernel_source << "    __global const int* shape,\n";
  kernel_source << "    __global const long* src_strides,\n";
  kernel_source << "    __global const long* dst_strides,\n";
  kernel_source << "    long src_base_offset,\n";
  kernel_source << "    long dst_base_offset,\n";
  kernel_source << "    int ndim,\n";
  kernel_source << "    int n";
  if (has_dynamic_src || has_dynamic_dst) {
    kernel_source << ",\n    __global const long* dynamic_src_offset,\n";
    kernel_source << "    __global const long* dynamic_dst_offset";
  }
  kernel_source << ") {\n";
  kernel_source << "  int idx = get_global_id(0);\n";
  kernel_source << "  if (idx >= n) return;\n\n";
  kernel_source << "  // Convert flat index to multi-dimensional indices and compute offsets\n";
  kernel_source << "  long src_off = src_base_offset";
  if (has_dynamic_src) {
    kernel_source << " + (dynamic_src_offset ? *dynamic_src_offset : 0)";
  }
  kernel_source << ";\n";
  kernel_source << "  long dst_off = dst_base_offset";
  if (has_dynamic_dst) {
    kernel_source << " + (dynamic_dst_offset ? *dynamic_dst_offset : 0)";
  }
  kernel_source << ";\n";
  kernel_source << "  int remaining = idx;\n";
  kernel_source << "  for (int d = ndim - 1; d >= 0; d--) {\n";
  kernel_source << "    int coord = remaining % shape[d];\n";
  kernel_source << "    remaining /= shape[d];\n";
  kernel_source << "    src_off += coord * src_strides[d];\n";
  kernel_source << "    dst_off += coord * dst_strides[d];\n";
  kernel_source << "  }\n\n";
  kernel_source << "  dst[dst_off] = (" << dst_type << ")src[src_off];\n";
  kernel_source << "}\n";

  std::string source_str = kernel_source.str();
  std::string kernel_name = "copy_inplace_" + src_type + "_to_" + dst_type + kernel_suffix;

  // Get or compile kernel
  cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
  auto& encoder = dev.get_command_encoder(s.index);
  encoder.set_kernel(kernel);

  // Create buffers for shape and strides
  std::vector<int> shape_vec(data_shape.begin(), data_shape.end());
  if (shape_vec.empty()) shape_vec.push_back(1);

  std::vector<int64_t> src_strides_vec(strides.begin(), strides.end());
  if (src_strides_vec.empty()) src_strides_vec.push_back(0);

  std::vector<int64_t> dst_strides_vec(out_strides.begin(), out_strides.end());
  if (dst_strides_vec.empty()) dst_strides_vec.push_back(0);

  cl_int err;
  cl_mem shape_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     shape_vec.size() * sizeof(int), shape_vec.data(), &err);
  cl_mem src_strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           src_strides_vec.size() * sizeof(int64_t), src_strides_vec.data(), &err);
  cl_mem dst_strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                           dst_strides_vec.size() * sizeof(int64_t), dst_strides_vec.data(), &err);

  // Set arguments
  encoder.set_input_array(src, 0);
  encoder.set_output_array(dst, 1);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &shape_buf);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &src_strides_buf);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &dst_strides_buf);
  encoder.set_bytes(src_offset, 5);
  encoder.set_bytes(dst_offset, 6);
  encoder.set_bytes(ndim, 7);
  encoder.set_bytes(static_cast<int>(n), 8);

  if (has_dynamic_src || has_dynamic_dst) {
    // Pass dynamic offset arrays
    if (has_dynamic_src) {
      encoder.set_input_array(*indices_src, 9);
    } else {
      // Pass a null buffer - kernel will check for null
      cl_mem null_buf = nullptr;
      clSetKernelArg(kernel, 9, sizeof(cl_mem), &null_buf);
    }
    if (has_dynamic_dst) {
      encoder.set_input_array(*indices_dst, 10);
    } else {
      cl_mem null_buf = nullptr;
      clSetKernelArg(kernel, 10, sizeof(cl_mem), &null_buf);
    }
  }

  // Dispatch
  size_t global_size[3] = {n, 0, 0};
  encoder.dispatch_threads(global_size, nullptr, 1);

  // Track buffers for cleanup after sync
  dev.add_temp_buffer(shape_buf, s.index);
  dev.add_temp_buffer(src_strides_buf, s.index);
  dev.add_temp_buffer(dst_strides_buf, s.index);

  // Release temp buffers after kernel execution completes
  dev.end_encoding(s.index);
}

void reshape_gpu(const array& in, array& out, Stream s) {
  // Reshape is typically just a metadata operation (changing shape/strides)
  // Only need actual copy if data layout changes

  if (::mlx::core::opencl::is_debug_enabled()) {
    std::cerr << "[reshape_gpu] in.shape=";
    for (auto d : in.shape()) std::cerr << d << " ";
    std::cerr << "out.shape=";
    for (auto d : out.shape()) std::cerr << d << " ";
    std::cerr << std::endl;
  }

  auto [copy_necessary, out_strides] = prepare_reshape(in, out);

  if (copy_necessary) {
    // Need actual data movement
    // IMPORTANT: Use input shape for iteration (like Metal backend)
    // The output shape may be different but contains the same number of elements
    OPENCL_DEBUG_LOG("[reshape_gpu] Non-contiguous reshape - copying data");
    out.set_data(opencl::allocator().malloc(out.nbytes()));

    // Create contiguous strides for the output based on input shape
    // (output will be written contiguously, then reinterpreted with new shape)
    SmallVector<int64_t, 10> contiguous_strides;
    int64_t stride = 1;
    for (int i = in.ndim() - 1; i >= 0; i--) {
      contiguous_strides.insert(contiguous_strides.begin(), stride);
      stride *= in.shape()[i];
    }

    SmallVector<int, 10> data_shape(in.shape().begin(), in.shape().end());
    SmallVector<int64_t, 10> in_strides(in.strides().begin(), in.strides().end());

    // Pass the input array's base offset (convert from bytes to elements)
    int64_t src_element_offset = in.offset() / size_of(in.dtype());
    copy_gpu_inplace(
        in,
        out,
        data_shape,
        in_strides,
        contiguous_strides,
        src_element_offset,  // src_offset
        0,  // dst_offset
        CopyType::General,
        s);
  } else {
    // Just share the buffer with new shape/strides
    shared_buffer_reshape(in, out_strides, out);
    OPENCL_DEBUG_LOG("[reshape_gpu] Shared buffer (no copy needed)");
  }
}

void fill_gpu(const array& src, array& dst, const Stream& s) {
  if (dst.size() == 0) {
    return;
  }

  // Allocate output buffer
  dst.set_data(opencl::allocator().malloc(dst.nbytes()));

  auto& dev = opencl::device(Device::gpu);

  std::string type_name = opencl::type_to_name(dst.dtype());
  size_t n = dst.data_size();

  OPENCL_DEBUG_LOG("[fill_gpu] Filling " << n << " elements of type " << type_name);

  // Build fill kernel: out[i] = value
  // For half/bfloat16, pass value as uint and convert in kernel to avoid
  // platform-specific issues with scalar half arguments
  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();
  kernel_source << "\n__kernel void fill_" << type_name << "(\n";
  kernel_source << "    __global " << type_name << "* out,\n";
  if (dst.dtype() == float16) {
    // Pass half as uint and convert via as_half
    kernel_source << "    uint value_bits,\n";
  } else if (dst.dtype() == bfloat16) {
    // Pass bfloat16 as uint
    kernel_source << "    uint value_bits,\n";
  } else {
    kernel_source << "    " << type_name << " value,\n";
  }
  kernel_source << "    long n) {\n";
  kernel_source << "  long idx = get_global_id(0);\n";
  kernel_source << "  if (idx < n) {\n";
  if (dst.dtype() == float16) {
    kernel_source << "    out[idx] = as_half((ushort)value_bits);\n";
  } else if (dst.dtype() == bfloat16) {
    kernel_source << "    out[idx] = (bfloat16_t)value_bits;\n";
  } else {
    kernel_source << "    out[idx] = value;\n";
  }
  kernel_source << "  }\n";
  kernel_source << "}\n";

  std::string source_str = kernel_source.str();
  std::string kernel_name = "fill_" + type_name;

  // Get or compile kernel
  cl_kernel kernel = dev.get_kernel(kernel_name, source_str, "");
  auto& encoder = dev.get_command_encoder(s.index);
  encoder.set_kernel(kernel);

  // Set arguments
  encoder.set_output_array(dst, 0);

  // Set the scalar value from source array (which has size 1)
  // Need to read the CPU data from src
  switch (dst.dtype()) {
    case bool_:
      encoder.set_bytes(*src.data<uint8_t>(), 1);
      break;
    case uint8:
      encoder.set_bytes(*src.data<uint8_t>(), 1);
      break;
    case uint16:
      encoder.set_bytes(*src.data<uint16_t>(), 1);
      break;
    case uint32:
      encoder.set_bytes(*src.data<uint32_t>(), 1);
      break;
    case uint64:
      encoder.set_bytes(*src.data<uint64_t>(), 1);
      break;
    case int8:
      encoder.set_bytes(*src.data<int8_t>(), 1);
      break;
    case int16:
      encoder.set_bytes(*src.data<int16_t>(), 1);
      break;
    case int32:
      encoder.set_bytes(*src.data<int32_t>(), 1);
      break;
    case int64:
      encoder.set_bytes(*src.data<int64_t>(), 1);
      break;
    case float16: {
      // Pass as uint to avoid scalar half argument issues on some platforms
      uint32_t bits = *src.data<uint16_t>();
      encoder.set_bytes(bits, 1);
      break;
    }
    case float32:
      encoder.set_bytes(*src.data<float>(), 1);
      break;
    case bfloat16: {
      // Pass as uint
      uint32_t bits = *src.data<uint16_t>();
      encoder.set_bytes(bits, 1);
      break;
    }
    case complex64: {
      // complex64 is float2 in OpenCL (two floats)
      // Pass as two separate floats via a struct
      auto c = *src.data<std::complex<float>>();
      float real_part = c.real();
      float imag_part = c.imag();
      // Set as a float2 (8 bytes)
      struct { float x, y; } float2_val = {real_part, imag_part};
      clSetKernelArg(kernel, 1, sizeof(float2_val), &float2_val);
      break;
    }
    default:
      throw std::runtime_error("[fill_gpu] Unsupported dtype");
  }

  int64_t n_val = static_cast<int64_t>(n);
  encoder.set_bytes(n_val, 2);

  // Dispatch kernel
  size_t global_size[3] = {n, 1, 1};
  size_t local_size[3] = {1, 1, 1};
  encoder.dispatch_threads(global_size, local_size, 1);

  dev.end_encoding(s.index);
}

array compute_dynamic_offset(
    const array& indices,
    const SmallVector<int64_t, 10>& strides,
    const std::vector<int>& axes,
    const Stream& s) {
  auto& dev = opencl::device(Device::gpu);

  // Create output array for the offset (single int64 value)
  array offset({1}, int64, nullptr, {});
  bool donate = indices.is_donatable() &&
      (indices.data_size() * indices.itemsize()) >= offset.itemsize();
  if (donate) {
    offset.copy_shared_buffer(indices);
  } else {
    offset.set_data(opencl::allocator().malloc(offset.itemsize()));
  }

  std::string idx_type = opencl::type_to_name(indices.dtype());
  int n_axes = axes.size();

  // Build kernel to compute offset = sum(indices[i] * strides[axes[i]])
  std::ostringstream kernel_source;
  kernel_source << opencl::get_kernel_preamble();
  kernel_source << "\n__kernel void compute_dynamic_offset_" << idx_type << "(\n";
  kernel_source << "    __global const " << idx_type << "* indices,\n";
  kernel_source << "    __global long* offset,\n";
  kernel_source << "    __global const long* strides,\n";
  kernel_source << "    __global const int* axes,\n";
  kernel_source << "    int n_axes) {\n";
  kernel_source << "  if (get_global_id(0) != 0) return;\n";
  kernel_source << "  long acc = 0;\n";
  kernel_source << "  for (int i = 0; i < n_axes; ++i) {\n";
  kernel_source << "    acc += (long)indices[i] * strides[axes[i]];\n";
  kernel_source << "  }\n";
  kernel_source << "  *offset = acc;\n";
  kernel_source << "}\n";

  std::string source_str = kernel_source.str();
  std::string kernel_name = "compute_dynamic_offset_" + idx_type;

  cl_kernel kernel = dev.get_kernel(kernel_name, source_str);
  auto& encoder = dev.get_command_encoder(s.index);
  encoder.set_kernel(kernel);

  // Create buffers for strides and axes
  std::vector<int64_t> strides_vec(strides.begin(), strides.end());
  std::vector<int> axes_vec(axes.begin(), axes.end());

  cl_int err;
  cl_mem strides_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       strides_vec.size() * sizeof(int64_t), strides_vec.data(), &err);
  if (err != CL_SUCCESS) {
    throw std::runtime_error("[compute_dynamic_offset] Failed to create strides buffer");
  }

  cl_mem axes_buf = clCreateBuffer(dev.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    axes_vec.size() * sizeof(int), axes_vec.data(), &err);
  if (err != CL_SUCCESS) {
    clReleaseMemObject(strides_buf);
    throw std::runtime_error("[compute_dynamic_offset] Failed to create axes buffer");
  }

  // Set kernel arguments
  encoder.set_input_array(indices, 0);
  encoder.set_output_array(offset, 1);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &strides_buf);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &axes_buf);
  encoder.set_bytes(n_axes, 4);

  // Dispatch single work item
  size_t global_size[3] = {1, 0, 0};
  encoder.dispatch_threads(global_size, nullptr, 1);

  // Track buffers for cleanup after sync
  dev.add_temp_buffer(strides_buf, s.index);
  dev.add_temp_buffer(axes_buf, s.index);

  dev.add_temporary(offset, s.index);

  // Release temp buffers after kernel execution completes
  dev.end_encoding(s.index);

  return offset;
}

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& output,
    int axis,
    const Stream& s) {
  // Compute cumulative sizes along the concatenation axis
  std::vector<int> sizes;
  sizes.push_back(0);
  for (auto& p : inputs) {
    sizes.push_back(p.shape(axis));
  }
  std::partial_sum(sizes.cbegin(), sizes.cend(), sizes.begin());

  // Allocate output
  output.set_data(opencl::allocator().malloc(output.nbytes()));

  auto strides = output.strides();
  auto flags = output.flags();
  flags.row_contiguous = false;
  flags.col_contiguous = false;
  flags.contiguous = false;

  // Copy each input to its slice in the output
  for (size_t i = 0; i < inputs.size(); i++) {
    array out_slice(inputs[i].shape(), output.dtype(), nullptr, {});
    size_t data_offset = strides[axis] * sizes[i];
    out_slice.copy_shared_buffer(
        output, strides, flags, out_slice.size(), data_offset);

    // Use general copy to handle strided output
    SmallVector<int, 10> data_shape(inputs[i].shape().begin(), inputs[i].shape().end());
    SmallVector<int64_t, 10> in_strides(inputs[i].strides().begin(), inputs[i].strides().end());
    SmallVector<int64_t, 10> out_strides(strides.begin(), strides.end());

    // Get the element offset from the slice's byte offset
    int64_t dst_element_offset = out_slice.offset() / out_slice.itemsize();
    int64_t src_element_offset = inputs[i].offset() / inputs[i].itemsize();

    copy_gpu_inplace(
        inputs[i],
        out_slice,
        data_shape,
        in_strides,
        out_strides,
        src_element_offset,
        dst_element_offset,
        CopyType::GeneralGeneral,
        s);
  }
}
} // namespace mlx::core
