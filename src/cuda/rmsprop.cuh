#pragma once
#include <blas.cuh>

void rmsprop_update(Storage *square_grads, Storage *weights,
                    const Storage *grads, float learning_rate = 1e-2,
                    float l2 = 0, float beta = 0.99);