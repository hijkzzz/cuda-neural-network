#pragma once

__global__ void tensor_add(float *a, float *b, float *c, unsigned int width, unsigned int height);
__global__ void tensor_sub(float *a, float *b, float *c, unsigned int width, unsigned int height);
__global__ void tensor_mul(float *a, float *b, float *c, unsigned int width, unsigned int height);
__global__ void tensor_div(float *a, float *b, float *c, unsigned int width, unsigned int height);

__global__ void tensor_matmul(float *a, float *b, float *c, unsigned int width, unsigned int k, unsigned int height);
__global__ void tensor_transpose(float *a, float *c, unsigned int width, unsigned int height);

__global__ void tensor_pow(float *a, float *c, unsigned int width, unsigned int height, unsigned int e);
__global__ void tensor_log(float *a, float *c, unsigned int width, unsigned int height);
__global__ void tensor_exp(float *a, float *c, unsigned int width, unsigned int height);
__global__ void tensor_log_softmax(float *a, float *c, unsigned int width, unsigned int height, unsigned int dim);
__global__ void tensor_mean(float *a, float *c, unsigned int width, unsigned int height, unsigned int dim);
__global__ void tensor_sigmoid(float *a, float *c, unsigned int width, unsigned int height);
__global__ void tensor_tanh(float *a, float *c, unsigned int width, unsigned int height);