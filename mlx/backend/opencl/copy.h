// Copyright © 2025 MLX Contributors

#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include "mlx/array.h"
#include "mlx/backend/common/copy.h"

namespace mlx::core {

void copy_gpu(
    const array& src,
    array& dst,
    CopyType copy_type,
    const Stream& s);

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
    std::optional<array> indices_src = std::nullopt,
    std::optional<array> indices_dst = std::nullopt);

void reshape_gpu(const array& in, array& out, Stream s);

void fill_gpu(const array& src, array& dst, const Stream& s);

array compute_dynamic_offset(
    const array& indices,
    const SmallVector<int64_t, 10>& strides,
    const std::vector<int>& axes,
    const Stream& s);

void concatenate_gpu(
    const std::vector<array>& inputs,
    array& output,
    int axis,
    const Stream& s);

} // namespace mlx::core
