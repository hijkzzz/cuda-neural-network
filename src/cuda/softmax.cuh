#pragma once

#include <blas.cuh>

Storage *operator_log_softmax(const Storage *input1, std::size_t dim);

__global__ void operator_log_softmax_h(const float *input1, float *output,
                                       std::size_t *input1_shape,
                                       std::size_t input1_dims,
                                       std::size_t temp_shape_ptr,
                                       std::size_t dim, std::size_t dim_stride,
                                       std::size_t size);