#pragma once

#include <blas.cuh>
#include <layer.cuh>

#ifdef DEBUG

void operator_nll_loss(const Storage *log_p, const Storage *y, Storage *output);

void operator_d_nll_loss(const Storage *y, Storage *inputs_grad);

#endif  // DEBUG

class NLLLoss : public Layer {
 public:
  NLLLoss() { this->output.reset(new Storage({1, 1})); }
  void forward(const Storage *y);
  void backward();

 private:
  const Storage *y;  // backup
};