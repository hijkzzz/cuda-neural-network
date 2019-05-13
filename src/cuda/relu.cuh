#pragma once

#include <layer.cuh>
#include <blas.cuh>

#ifdef DEBUG

void operator_relu(const Storage *input1, Storage *outputs);
void operator_d_relu(const Storage *outputs_grad, const Storage *input1,
                     Storage *intputs_grad);

#endif  // DEBUG

class ReLU : public Layer {
 public:
  ReLU(bool inplace) : inplace(inplace) {}

  void forward();
  void backward();

  Storage *get_grad() {
    return this->inplace ? this->next->get_grad() : this->grad.get();
  }
  Storage *get_output() {
    return this->inplace ? this->pre->get_output() : this->output.get();
  }

 private:
  bool inplace;
};
