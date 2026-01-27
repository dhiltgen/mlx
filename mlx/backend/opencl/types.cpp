// Copyright © 2025 MLX Contributors

#include "mlx/backend/opencl/types.h"
#include <stdexcept>

namespace mlx::core::opencl {

TypeConversion get_type_conversion(Dtype dtype) {
  TypeConversion conv;

  switch (dtype.val()) {
    case Dtype::Val::bfloat16:
      conv.read_fn = "bfloat16_to_float";
      conv.write_fn = "float_to_bfloat16";
      conv.acc_type = "float";
      conv.needs_read_convert = true;
      conv.needs_write_convert = true;
      break;

    case Dtype::Val::float16:
      conv.read_fn = "convert_float";
      conv.write_fn = "convert_half";
      conv.acc_type = "float";
      conv.needs_read_convert = true;
      conv.needs_write_convert = true;
      break;

    case Dtype::Val::float32:
      conv.read_fn = "";
      conv.write_fn = "";
      conv.acc_type = "float";
      conv.needs_read_convert = false;
      conv.needs_write_convert = false;
      break;

    case Dtype::Val::complex64:
      conv.read_fn = "";
      conv.write_fn = "";
      conv.acc_type = "float2";
      conv.needs_read_convert = false;
      conv.needs_write_convert = false;
      break;

    default:
      // Integer types - no conversion needed
      conv.read_fn = "";
      conv.write_fn = "";
      conv.acc_type = type_to_name(dtype);
      conv.needs_read_convert = false;
      conv.needs_write_convert = false;
      break;
  }

  return conv;
}

std::string make_read_expr(Dtype dtype, const std::string& var_expr) {
  auto conv = get_type_conversion(dtype);
  if (!conv.needs_read_convert) {
    return var_expr;
  }

  // All conversions use function-call style: func(var_expr)
  // bf16: bfloat16_to_float(var_expr)
  // f16: convert_float(var_expr)
  return conv.read_fn + "(" + var_expr + ")";
}

std::string make_write_expr(Dtype dtype, const std::string& val_expr) {
  auto conv = get_type_conversion(dtype);
  if (!conv.needs_write_convert) {
    return val_expr;
  }

  // All conversions use function-call style: func(val_expr)
  // bf16: float_to_bfloat16(val_expr)
  // f16: convert_half(val_expr)
  return conv.write_fn + "(" + val_expr + ")";
}

std::string get_accumulator_type(Dtype dtype) {
  return get_type_conversion(dtype).acc_type;
}

bool needs_float_conversion(Dtype dtype) {
  return dtype == bfloat16 || dtype == float16;
}

bool is_floating_type(Dtype dtype) {
  switch (dtype.val()) {
    case Dtype::Val::float16:
    case Dtype::Val::bfloat16:
    case Dtype::Val::float32:
      return true;
    default:
      return false;
  }
}

bool is_integer_type(Dtype dtype) {
  switch (dtype.val()) {
    case Dtype::Val::bool_:
    case Dtype::Val::uint8:
    case Dtype::Val::uint16:
    case Dtype::Val::uint32:
    case Dtype::Val::uint64:
    case Dtype::Val::int8:
    case Dtype::Val::int16:
    case Dtype::Val::int32:
    case Dtype::Val::int64:
      return true;
    default:
      return false;
  }
}

std::string type_to_name(const Dtype& dtype) {
  switch (dtype.val()) {
    case Dtype::Val::bool_:
      return "uchar";  // MLX bool is 1 byte, OpenCL bool is often 4 bytes
    case Dtype::Val::uint8:
      return "uint8_t";
    case Dtype::Val::uint16:
      return "uint16_t";
    case Dtype::Val::uint32:
      return "uint32_t";
    case Dtype::Val::uint64:
      return "uint64_t";
    case Dtype::Val::int8:
      return "int8_t";
    case Dtype::Val::int16:
      return "int16_t";
    case Dtype::Val::int32:
      return "int32_t";
    case Dtype::Val::int64:
      return "int64_t";
    case Dtype::Val::float16:
      return "half";
    case Dtype::Val::float32:
      return "float";
    case Dtype::Val::bfloat16:
      return "bfloat16_t"; // Emulated as uint16 with conversion helpers
    case Dtype::Val::complex64:
      return "float2"; // Emulated as float2
    default:
      throw std::runtime_error("Unsupported dtype for OpenCL");
  }
}

std::string type_to_suffix(Dtype dtype) {
  switch (dtype.val()) {
    case Dtype::Val::bool_:
      return "bool";
    case Dtype::Val::uint8:
      return "u8";
    case Dtype::Val::uint16:
      return "u16";
    case Dtype::Val::uint32:
      return "u32";
    case Dtype::Val::uint64:
      return "u64";
    case Dtype::Val::int8:
      return "i8";
    case Dtype::Val::int16:
      return "i16";
    case Dtype::Val::int32:
      return "i32";
    case Dtype::Val::int64:
      return "i64";
    case Dtype::Val::float16:
      return "f16";
    case Dtype::Val::float32:
      return "f32";
    case Dtype::Val::bfloat16:
      return "bf16";
    case Dtype::Val::complex64:
      return "c64";
    default:
      return "unknown";
  }
}

} // namespace mlx::core::opencl
