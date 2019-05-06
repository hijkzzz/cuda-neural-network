#pragma once

#include <blas.cuh>

void operator_relu(const Storage *input1, Storage *outputs);

void operator_d_relu(const Storage *outputs_grad, const Storage *input1,
                     Storage *intputs_grad);