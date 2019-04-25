#pragma once

#include <storage.cuh>
#include <utils.cuh>

#include <cuda_runtime.h>

Storage *operator_add(const Storage *input1, const Storage *input2);

Storage *operator_add(const Storage *input1, float value);

Storage *operator_sub(const Storage *input1, const Storage *input2);

Storage *operator_mul(const Storage *input1, const Storage *input2);

Storage *operator_mul(const Storage *input1, float value);

Storage *operator_div(const Storage *input1, const Storage *input2);

Storage *operator_log(const Storage *input1);

Storage *operator_exp(const Storage *input1);

Storage *operator_pow(const Storage *input1, float e);

Storage *operator_matmul(const Storage *input1, const Storage *input2);

Storage *operator_transpose(const Storage *input1, std::size_t dim0,
                            std::size_t dim1);

Storage *operator_mean(const Storage *input1, std::size_t dim);

Storage *operator_sum(const Storage *input1, std::size_t dim);