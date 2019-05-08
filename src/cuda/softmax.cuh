#pragma once

#include <blas.cuh>
#include <layer.cuh>

#ifdef DEBUG

void operator_log_softmax(const Storage *input1, int dim, Storage *outputs);

void operator_d_log_softmax(const Storage *output_grads, const Storage *input1,
                            int dim, Storage *inputs_grad);

#endif  // DEBUG

class LogSoftmax : public Layer{
 public:
  explicit LogSoftmax(int dim = 1) : dim(dim) {}
  void forward();
  void backward();

 private:
  int dim;
};