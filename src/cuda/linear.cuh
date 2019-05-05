#pragma once

#include <blas.cuh>

Storage *operator_linear(const Storage *inputs, const Storage *weights);

Storage *operator_d_linear(const Storage *outputs_grad, const Storage *inputs,
                           const Storage *weights, Storage *weights_grad);

Storage *operator_bias(const Storage *inputs, const Storage *bias);

Storage *operator_d_bias(const Storage *outputs_grad, Storage *bias_grad);