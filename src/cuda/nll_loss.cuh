#pragma once

#include <blas.cuh>
#include <layer.cuh>
#include <unordered_map>

#ifdef DEBUG

void operator_nll_loss(
    const Storage *log_p, const Storage *y, Storage *output,
    std::unordered_map<std::string, std::unique_ptr<Storage>> &temp);

void operator_d_nll_loss(const Storage *y, Storage *inputs_grad);

#endif  // DEBUG

class NLLLoss : public Layer {
 public:
  NLLLoss() { this->output.reset(new Storage({1, 1})); }
  void forward(const Storage *y);
  void backward();

 private:
  const Storage *y;  // backup

  std::unordered_map<std::string, std::unique_ptr<Storage>> temp;
};