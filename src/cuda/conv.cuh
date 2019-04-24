#pragma once

#include <cuda_runtime.h>
#include <storage.cuh>

Storage *operator_conv(const Storage *inputs, const Storage *filters);
__global__ void operator_conv_h(const float *inputs, const float *filters,
                                float *outputs);

Storage *operator_d_conv(const Storage *outputs_grad, const Storage *filters);
__global__ void operator_d_conv_h(const float *outputs_grad,
                                 const Storage *filters, float *inputs_grad);