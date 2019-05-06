#pragma once

#include <blas.cuh>

void operator_linear(const Storage *inputs, const Storage *weights,
                     Storage *output);

void operator_d_linear(const Storage *outputs_grad, const Storage *inputs,
                       const Storage *weights, Storage *weights_grad,
                       Storage *inputs_grad);

void operator_bias(const Storage *inputs, const Storage *bias, Storage *output);

void operator_d_bias(const Storage *outputs_grad, Storage *bias_grad,
                     Storage *inputs_grad);