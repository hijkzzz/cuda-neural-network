#pragma once

#include <storage.cuh>
#include <utils.cuh>

Storage *operator_add(const Storage *input1, const Storage *input2);

Storage *operator_add(const Storage *input1, float value);

Storage *operator_sub(const Storage *input1, const Storage *input2);

Storage *operator_mul(const Storage *input1, const Storage *input2);

Storage *operator_mul(const Storage *input1, float value);

Storage *operator_div(const Storage *input1, const Storage *input2);

Storage *operator_pow(const Storage *input1, float e);

Storage *operator_log(const Storage *input1);

Storage *operator_exp(const Storage *input1);

Storage *operator_sigmoid(const Storage *input1);

Storage *operator_tanh(const Storage *input1);

Storage *operator_matmul(const Storage *input1, const Storage *input2);

__global__ void operator_matmul_h(const float *input1, const float *input2,
                                  float *output, std::size_t height,
                                  std::size_t k, std::size_t width);

Storage *operator_transpose(const Storage *input1, std::size_t dim0,
                            std::size_t dim1);
__global__ void operator_transpose_h(const float *input1, float *output,
                                     std::size_t *input1_shape,
                                     std::size_t input1_dims,
                                     std::size_t output_shape, std::size_t dim0,
                                     std::size_t dim1, std::size_t size);

Storage *operator_log_softmax(const Storage *input1, std::size_t dim);

__global__ void operator_log_softmax_h(const float *input1, float *output,
                                       std::size_t *input1_shape,
                                       std::size_t input1_dims,
                                       std::size_t temp_shape_ptr,
                                       std::size_t dim, std::size_t dim_stride,
                                       std::size_t size);

Storage *operator_mean(const Storage *input1, std::size_t dim);

__global__ void operator_mean_h(const float *input1, float *output,
                                std::size_t *input1_shape,
                                std::size_t input1_dims,
                                std::size_t *output_shape, std::size_t dim,
                                std::size_t dim_stride, std::size_t size);