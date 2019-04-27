#include <rmsprop.cuh>

void rmsprop_update(Storage *square_grads, Storage *weights,
                    const Storage *grads, float beta = 0.999,
                    float learning_rate = 1e-3, float l2 = 1e-4);
