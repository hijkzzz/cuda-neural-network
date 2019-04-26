#pragma once

#include <blas.cuh>

Storage *operator_nll_loss(const Storage *log_p, const Storage *y);

Storage *operator_d_nll_loss(const Storage *log_p, const Storage *y,
                             const Storage *outputs_grads);