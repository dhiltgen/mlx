// Copyright © 2025 MLX Contributors

#include <sstream>
#include "mlx/backend/opencl/allocator.h"
#include "mlx/backend/opencl/device.h"
#include "mlx/backend/opencl/utils.h"
#include "mlx/backend/opencl/debug.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

std::string get_threefry_kernel_source() {
  std::ostringstream src;
  src << opencl::get_kernel_preamble();
  src << R"(
// Threefry 2x32 hash function - returns the two uint32 values
void threefry2x32_hash(
    uint key_x, uint key_y,
    uint count_x, uint count_y,
    uint* out_x, uint* out_y) {

  uint ks0 = key_x;
  uint ks1 = key_y;
  uint ks2 = ks0 ^ ks1 ^ 0x1BD11BDA;

  uint vx = count_x + ks0;
  uint vy = count_y + ks1;

  // Round 0: rotations {13, 15, 26, 6}
  vx += vy; vy = (vy << 13) | (vy >> 19); vy ^= vx;
  vx += vy; vy = (vy << 15) | (vy >> 17); vy ^= vx;
  vx += vy; vy = (vy << 26) | (vy >> 6); vy ^= vx;
  vx += vy; vy = (vy << 6) | (vy >> 26); vy ^= vx;
  vx += ks1;
  vy += ks2 + 1;

  // Round 1: rotations {17, 29, 16, 24}
  vx += vy; vy = (vy << 17) | (vy >> 15); vy ^= vx;
  vx += vy; vy = (vy << 29) | (vy >> 3); vy ^= vx;
  vx += vy; vy = (vy << 16) | (vy >> 16); vy ^= vx;
  vx += vy; vy = (vy << 24) | (vy >> 8); vy ^= vx;
  vx += ks2;
  vy += ks0 + 2;

  // Round 2: rotations {13, 15, 26, 6}
  vx += vy; vy = (vy << 13) | (vy >> 19); vy ^= vx;
  vx += vy; vy = (vy << 15) | (vy >> 17); vy ^= vx;
  vx += vy; vy = (vy << 26) | (vy >> 6); vy ^= vx;
  vx += vy; vy = (vy << 6) | (vy >> 26); vy ^= vx;
  vx += ks0;
  vy += ks1 + 3;

  // Round 3: rotations {17, 29, 16, 24}
  vx += vy; vy = (vy << 17) | (vy >> 15); vy ^= vx;
  vx += vy; vy = (vy << 29) | (vy >> 3); vy ^= vx;
  vx += vy; vy = (vy << 16) | (vy >> 16); vy ^= vx;
  vx += vy; vy = (vy << 24) | (vy >> 8); vy ^= vx;
  vx += ks1;
  vy += ks2 + 4;

  // Round 4: rotations {13, 15, 26, 6}
  vx += vy; vy = (vy << 13) | (vy >> 19); vy ^= vx;
  vx += vy; vy = (vy << 15) | (vy >> 17); vy ^= vx;
  vx += vy; vy = (vy << 26) | (vy >> 6); vy ^= vx;
  vx += vy; vy = (vy << 6) | (vy >> 26); vy ^= vx;
  vx += ks2;
  vy += ks0 + 5;

  *out_x = vx;
  *out_y = vy;
}

// Write bytes from a uint32 value
void write_bytes(__global uchar* out, uint val, int num_bytes) {
  if (num_bytes > 0) out[0] = (uchar)(val);
  if (num_bytes > 1) out[1] = (uchar)(val >> 8);
  if (num_bytes > 2) out[2] = (uchar)(val >> 16);
  if (num_bytes > 3) out[3] = (uchar)(val >> 24);
}

// Contiguous keys kernel
// Grid: (num_keys, half_size + odd)
// Each thread generates up to 8 bytes of random data
__kernel void rbitsc(
    __global const uint* keys,
    __global uchar* out,
    int odd,
    uint bytes_per_key,
    uint num_keys,
    uint half_size_param,
    long key_offset) {
  uint key_idx = get_global_id(0);
  uint y_idx = get_global_id(1);

  if (key_idx >= num_keys) return;
  if (y_idx >= half_size_param) return;

  uint half_size = half_size_param - odd;

  // Get key for this work item (accounting for array offset)
  long kidx = key_offset + 2 * key_idx;
  uint key_x = keys[kidx];
  uint key_y = keys[kidx + 1];

  // Output pointer for this key
  __global uchar* out_ptr = out + (ulong)key_idx * bytes_per_key;

  // Check if this is the last thread that handles odd case
  int drop_last = odd && (y_idx == half_size);

  // Calculate counter values
  uint count_x = y_idx;
  uint count_y = drop_last ? 0 : (y_idx + half_size_param);

  // Generate random bits
  uint vx, vy;
  threefry2x32_hash(key_x, key_y, count_x, count_y, &vx, &vy);

  // Write first 4 bytes
  ulong idx1 = (ulong)y_idx << 2;
  write_bytes(out_ptr + idx1, vx, 4);

  // Write second 4 bytes (handling edge case for partial writes)
  if (!drop_last) {
    ulong idx2 = (ulong)(y_idx + half_size_param) << 2;
    // Check if this is the last write and needs partial bytes
    if ((y_idx + 1) == half_size && (bytes_per_key % 4) > 0) {
      int edge_bytes = (bytes_per_key % 4);
      write_bytes(out_ptr + idx2, vy, edge_bytes);
    } else {
      write_bytes(out_ptr + idx2, vy, 4);
    }
  }
}

// Non-contiguous keys kernel (handles arbitrary strides)
__kernel void rbits(
    __global const uint* keys,
    __global uchar* out,
    int odd,
    uint bytes_per_key,
    uint num_keys,
    uint half_size_param,
    long key_offset,
    int ndim,
    __constant const int* key_shape,
    __constant const long* key_strides) {
  uint key_idx = get_global_id(0);
  uint y_idx = get_global_id(1);

  if (key_idx >= num_keys) return;
  if (y_idx >= half_size_param) return;

  uint half_size = half_size_param - odd;

  // Get key with strided access (accounting for array offset)
  uint kidx = 2 * key_idx;
  long k1_elem = key_offset + elem_to_loc(kidx, key_shape, key_strides, ndim);
  long k2_elem = key_offset + elem_to_loc(kidx + 1, key_shape, key_strides, ndim);
  uint key_x = keys[k1_elem];
  uint key_y = keys[k2_elem];

  // Output pointer for this key
  __global uchar* out_ptr = out + (ulong)key_idx * bytes_per_key;

  // Check if this is the last thread that handles odd case
  int drop_last = odd && (y_idx == half_size);

  // Calculate counter values
  uint count_x = y_idx;
  uint count_y = drop_last ? 0 : (y_idx + half_size_param);

  // Generate random bits
  uint vx, vy;
  threefry2x32_hash(key_x, key_y, count_x, count_y, &vx, &vy);

  // Write first 4 bytes
  ulong idx1 = (ulong)y_idx << 2;
  write_bytes(out_ptr + idx1, vx, 4);

  // Write second 4 bytes (handling edge case for partial writes)
  if (!drop_last) {
    ulong idx2 = (ulong)(y_idx + half_size_param) << 2;
    // Check if this is the last write and needs partial bytes
    if ((y_idx + 1) == half_size && (bytes_per_key % 4) > 0) {
      int edge_bytes = (bytes_per_key % 4);
      write_bytes(out_ptr + idx2, vy, edge_bytes);
    } else {
      write_bytes(out_ptr + idx2, vy, 4);
    }
  }
}
)";
  return src.str();
}

} // anonymous namespace

void RandomBits::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  // keys has shape (N1, ..., NK, 2)
  // out has shape (N1, ..., NK, M1, M2, ...)
  auto& keys = inputs[0];
  size_t num_keys = keys.size() / 2;

  if (num_keys == 0 || out.size() == 0) {
    out.set_data(opencl::allocator().malloc(out.nbytes()));
    return;
  }

  size_t elems_per_key = out.size() / num_keys;
  size_t bytes_per_key = out.itemsize() * elems_per_key;

  // Allocate output
  out.set_data(opencl::allocator().malloc(out.nbytes()));

  // Calculate grid dimensions
  // out_skip = number of 32-bit words needed per key
  size_t out_skip = (bytes_per_key + 3) / 4;  // Round up to nearest 4 bytes
  size_t half_size = (out_skip + 1) / 2;  // Half size, rounding up
  bool odd = (out_skip % 2 != 0);

  auto& s = stream();
  auto& d = opencl::device(s.device);

  // Build kernel source
  std::string kernel_source = get_threefry_kernel_source();

  // Choose kernel based on key contiguity
  // Use row_contiguous (not just contiguous) because our index calculation
  // assumes row-major layout
  bool keys_row_contiguous = keys.flags().row_contiguous;
  std::string kernel_name = keys_row_contiguous ? "rbitsc" : "rbits";

  // Pass the keys array element offset (convert from bytes to elements)
  int64_t key_offset = static_cast<int64_t>(keys.offset()) / keys.itemsize();

  OPENCL_DEBUG_LOG("[RandomBits] num_keys=" << num_keys
            << " bytes_per_key=" << bytes_per_key
            << " half_size=" << half_size
            << " odd=" << odd
            << " keys_row_contiguous=" << keys_row_contiguous
            << " key_offset=" << key_offset
            << " keys.id=" << keys.id()
            << " width=" << width_);

  // Get or compile kernel
  cl_kernel kernel = d.get_kernel(kernel_name, kernel_source, "");

  // Get command encoder
  auto& encoder = d.get_command_encoder(s.index);
  encoder.set_kernel(kernel);

  // Set kernel arguments
  encoder.set_input_array(keys, 0);
  encoder.set_output_array(out, 1);

  int odd_int = odd ? 1 : 0;
  encoder.set_bytes(odd_int, 2);

  cl_uint bytes_per_key_u = static_cast<cl_uint>(bytes_per_key);
  encoder.set_bytes(bytes_per_key_u, 3);

  cl_uint num_keys_u = static_cast<cl_uint>(num_keys);
  encoder.set_bytes(num_keys_u, 4);

  cl_uint half_size_u = static_cast<cl_uint>(half_size);
  encoder.set_bytes(half_size_u, 5);

  encoder.set_bytes(key_offset, 6);

  if (!keys_row_contiguous) {
    int ndim = static_cast<int>(keys.ndim());
    encoder.set_bytes(ndim, 7);

    // Create constant memory buffers for shape and strides
    std::vector<int> shape_vec(keys.shape().begin(), keys.shape().end());
    std::vector<int64_t> strides_vec(keys.strides().begin(), keys.strides().end());

    // Handle scalar case
    if (shape_vec.empty()) shape_vec.push_back(1);
    if (strides_vec.empty()) strides_vec.push_back(0);

    cl_int err;
    cl_mem shape_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       shape_vec.size() * sizeof(int), shape_vec.data(), &err);
    if (err != CL_SUCCESS) {
      throw std::runtime_error("[RandomBits] Failed to create shape buffer: " + std::to_string(err));
    }

    cl_mem strides_buf = clCreateBuffer(d.context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         strides_vec.size() * sizeof(int64_t), strides_vec.data(), &err);
    if (err != CL_SUCCESS) {
      clReleaseMemObject(shape_buf);
      throw std::runtime_error("[RandomBits] Failed to create strides buffer: " + std::to_string(err));
    }

    clSetKernelArg(kernel, 8, sizeof(cl_mem), &shape_buf);
    clSetKernelArg(kernel, 9, sizeof(cl_mem), &strides_buf);

    // Track buffers for cleanup after sync
    d.add_temp_buffer(shape_buf, s.index);
    d.add_temp_buffer(strides_buf, s.index);
  }

  // Dispatch kernel with 2D grid
  size_t global_size[3] = {num_keys, half_size, 1};
  size_t local_size[3] = {1, 1, 1};  // Let OpenCL choose work group size
  encoder.dispatch_threads(global_size, local_size, 2);

  d.end_encoding(s.index);
}

} // namespace mlx::core
