#pragma once

#include <blas.cuh>

void operator_log_softmax(const Storage *input1, int dim, Storage *outputs);

void operator_d_log_softmax(const Storage *output_grads, const Storage *input1,
                            int dim, Storage *inputs_grad);