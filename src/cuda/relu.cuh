#pragma once

#include <Layer.cuh>
#include <blas.cuh>

#ifdef DEBUG

void operator_relu(const Storage *input1, Storage *outputs);
void operator_d_relu(const Storage *outputs_grad, const Storage *input1,
                     Storage *intputs_grad);

#endif  // DEBUG

class ReLU : public Layer {
  void forward();
  void backward();
};
