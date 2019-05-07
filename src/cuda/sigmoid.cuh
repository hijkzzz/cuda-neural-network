#pragma once

#include <blas.cuh>
#include <layer.cuh>

#ifdef DEBUG

void operator_sigmoid(const Storage *input1, Storage *output);

void operator_d_sigmoid(const Storage *outputs_grad,
                            const Storage *input1, Storage *inputs_grad);

#endif  // DEBUG

class Sigmoid : public Layer {
  void forward();
  void backward();
};