#pragma once

#include <blas.cuh>

void operator_nll_loss(const Storage *log_p, const Storage *y, Storage *output);

void operator_d_nll_loss(const Storage *y, Storage *inputs_grad);