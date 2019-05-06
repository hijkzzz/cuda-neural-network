#pragma once

#include <blas.cuh>

void operator_sigmoid(const Storage *input1, Storage *output);

void operator_d_sigmoid(const Storage *outputs_grad,
                            const Storage *input1, Storage *inputs_grad);