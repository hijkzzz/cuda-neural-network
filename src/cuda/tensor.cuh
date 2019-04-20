#pragma once

__global__  Tensor::data_type add(float *a, float *b, unsigned int width, unsigned int height);
__global__  Tensor::data_type sub(float *a, float *b, unsigned int width, unsigned int height);
__global__  Tensor::data_type mul(float *a, float *b, unsigned int width, unsigned int height);
__global__  Tensor::data_type div(float *a, float *b, unsigned int width, unsigned int height);

__global__  Tensor::data_type matmul(float *a, float *b, unsigned int width, unsigned int k, unsigned int height);
__global__  Tensor::data_type transpose(float *a, unsigned int width, unsigned int height, unsigned int dim0, unsigned int dim1);

__global__  Tensor::data_type pow(float *a, unsigned int width, unsigned int height, unsigned int e);
__global__  Tensor::data_type log_softmax(float *a, unsigned int width, unsigned int height, unsigned int dim);
__global__  Tensor::data_type mean(float *a, unsigned int width, unsigned int height, unsigned int dim);
__global__  Tensor::data_type sigmoid(float *a, unsigned int width, unsigned int height);
__global__  Tensor::data_type tanh(float *a, unsigned int width, unsigned int height);

