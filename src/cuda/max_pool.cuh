#pragma once

#include <blas.cuh>

Storage* operator_max_pool(const Storage* inputs, Storage* mask);

Storage* operator_d_max_pool(const Storage* output_grads, const Storage* inputs,
                             const Storage* mask);