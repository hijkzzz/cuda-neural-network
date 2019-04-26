#pragma once

#include <blas.cuh>

Storage *operator_sigmoid(const Storage *input1);

Storage *operator_d_sigmoid(const Storage *input1, const Storage *outputs_grad);