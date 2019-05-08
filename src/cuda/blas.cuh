#pragma once

#include <storage.cuh>
#include <utils.cuh>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void operator_add(const Storage *input1, const Storage *input2,
                  Storage *outputs);
void operator_add(const Storage *input1, float value, Storage *outputs);

void operator_sub(const Storage *input1, const Storage *input2,
                  Storage *outputs);

void operator_mul(const Storage *input1, const Storage *input2,
                  Storage *outputs);
void operator_mul(const Storage *input1, float value, Storage *outputs);

void operator_div(const Storage *input1, const Storage *input2,
                  Storage *outputs);

void operator_log(const Storage *input1, Storage *outputs);

void operator_exp(const Storage *input1, Storage *outputs);

void operator_pow(const Storage *input1, float e, Storage *outputs);

void operator_matmul(const Storage *input1, const Storage *input2,
                     Storage *outputs, int broadcast = 0);

void operator_transpose(const Storage *input1, Storage *outputs);

void operator_mean(const Storage *input1, int dim, Storage *outputs);

void operator_sum(const Storage *input1, int dim, Storage *outputs);