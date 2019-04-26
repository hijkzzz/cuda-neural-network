#include <rmsprop.cuh>

void rmsprop_update(Storage *square_grads, Storage *weights,
                    const Storage *grads, float beta = 0.999);
