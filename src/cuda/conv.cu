#include <conv.cuh>

#include <cuda_runtime.h>

Storage *tensor_conv(const Storage *inputs, const Storage *filters) {}
void tensor_conv_h(const float *inputs, const float *filters, float *outputs) {}

Storage *tensor_d_conv(const Storage *outputs_grad, const Storage *filters) {}
void tensor_d_conv_h(const float *outputs_grad, const Storage *filters,
                     float *inputs_grad) {}