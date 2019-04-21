#pragma once

#include <utils.cuh>
#include <storage.cuh>

Storage* tensor_add(const Storage *a, const Storage *b);

Storage* tensor_sub(const Storage *a, const Storage *b);

Storage* tensor_mul(const Storage *a, const Storage *b);

Storage* tensor_div(const Storage *a, const Storage *b);

Storage* tensor_pow(const Storage *a, float e);
__global__ void tensor_pow_h(const float *a, float *c, float e, std::size_t size);

Storage* tensor_log(const Storage *a);
__global__ void tensor_log_h(const float *a, float *c, std::size_t size);

Storage* tensor_exp(const Storage *a);
__global__ void tensor_exp_h(const float *a, float *c, std::size_t size);

Storage* tensor_sigmoid(const Storage *a);
__global__ void tensor_sigmoid_h(const float *a, float *c, std::size_t size);

Storage* tensor_tanh(const Storage *a);
__global__ void tensor_tanh_h(const float *a, float *c, std::size_t size);

Storage* tensor_matmul(const Storage *a, const Storage *b);
__global__ void tensor_matmul_h(const float *a, const float *b, float *c, std::size_t width, std::size_t k, std::size_t height);

Storage* tensor_transpose(const Storage *a);
__global__ void tensor_transpose_h(const float *a, float *c, std::size_t width, std::size_t height);

Storage* tensor_log_softmax(const Storage *a);
__global__ void tensor_log_softmax_h(const float *a, float *c, std::size_t stride);

Storage* tensor_mean(const Storage *a);
__global__ void tensor_mean_h(const float *a, float *c, std::size_t stride);