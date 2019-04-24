#pragma once

#include <cuda_runtime.h>
#include <storage.cuh>

Storage *operator_bias(const Storage *inputs, const Storage *bias);
__global__ void operator_bias_h(const float *inputs, const float *bias,
                                float *outputs);

Storage *operator_d_bias(const Storage *outputs_grad, const Storage *bias);
__global__ void operator_d_bias_h(const float *outputs_grad,
                                  const Storage *bias, float *inputs_grad);