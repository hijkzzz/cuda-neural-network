#pragma once

#include <blas.cuh>

Storage *operator_log_softmax(const Storage *input1, int dim);

Storage *operator_d_log_softmax(const Storage *output_grads,
                                const Storage *input1, int dim);