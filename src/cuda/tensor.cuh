#pragma once

#include <storage.cuh>
#include <utils.cuh>

Storage *tensor_add(const Storage *a, const Storage *b);

Storage *tensor_add(const Storage *a, float value);

Storage *tensor_sub(const Storage *a, const Storage *b);

Storage *tensor_mul(const Storage *a, const Storage *b);

Storage *tensor_mul(const Storage *a, float value);

Storage *tensor_div(const Storage *a, const Storage *b);

Storage *tensor_pow(const Storage *a, float e);

Storage *tensor_log(const Storage *a);

Storage *tensor_exp(const Storage *a);

Storage *tensor_sigmoid(const Storage *a);

Storage *tensor_tanh(const Storage *a);

Storage *tensor_matmul(const Storage *a, const Storage *b);
__global__ void tensor_matmul_h(const float *a, const float *b, float *c,
                                std::size_t height, std::size_t k,
                                std::size_t width);

Storage *tensor_transpose(const Storage *a);
__global__ void tensor_transpose_h(const float *a, float *c, std::size_t height,
                                   std::size_t width);

Storage *tensor_log_softmax(const Storage *a);
__global__ void tensor_log_softmax_h(const float *a, float *c, std::size_t size,
                                     std::size_t stride);

Storage *tensor_mean(const Storage *a);
__global__ void tensor_mean_h(const float *a, float *c, std::size_t size,
                              std::size_t stride);