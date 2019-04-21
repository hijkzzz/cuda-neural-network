#pragma once

#include <utils.cuh>
#include <storage.cuh>

void tensor_add(const Storage *a, const Storage *b, Storage *c);
__global__ void tensor_add_h(const float *a, const float *b, float *c,
                             std::size_t size);

void tensor_sub(const Storage *a, const Storage *b, Storage *c);
__global__ void tensor_sub_h(const float *a, const float *b, float *c,
                             std::size_t size);

void tensor_mul(const Storage *a, const Storage *b, Storage *c);
__global__ void tensor_mul_h(const float *a, const float *b, float *c,
                             std::size_t size);

void tensor_div(const Storage *a, const Storage *b, Storage *c);
__global__ void tensor_div_h(const float *a, const float *b, float *c,
                             std::size_t size);

void tensor_matmul(const Storage *a, const Storage *b, Storage *c);
__global__ void tensor_matmul_h(const float *a, const float *b, float *c,
                                std::size_t *shape_a, std::size_t *shape_b,
                                std::size_t dims);

void tensor_transpose(const Storage *a, unsigned int dim0, unsigned int dim1,
                      Storage *c);
__global__ void tensor_transpose_h(const float *a, unsigned int dim0,
                                   unsigned int dim1, float *c,
                                   std::size_t *shape_a, std::size_t dims);

void tensor_log_softmax(const Storage *a, unsigned int dim, Storage *c);
__global__ void tensor_log_softmax_h(const float *a, unsigned int dim, float *c,
                                     std::size_t *shape_a, std::size_t dims);

void tensor_mean(const Storage *a, unsigned int dim, Storage *c);
__global__ void tensor_mean_h(const float *a, unsigned int dim, float *c,
                              std::size_t *shape_a, std::size_t dims);

void tensor_pow(const Storage *a, float e, Storage *c);
__global__ void tensor_pow_h(const float *a, float *c, float e, std::size_t size);

void tensor_log(const Storage *a, Storage *c);
__global__ void tensor_log_h(const float *a, float *c, std::size_t size);

void tensor_exp(const Storage *a, Storage *c);
__global__ void tensor_exp_h(const float *a, float *c, std::size_t size);

void tensor_sigmoid(const Storage *a, Storage *c);
__global__ void tensor_sigmoid_h(const float *a, float *c, std::size_t size);

void tensor_tanh(const Storage *a, Storage *c);
__global__ void tensor_tanh_h(const float *a, float *c, std::size_t size);