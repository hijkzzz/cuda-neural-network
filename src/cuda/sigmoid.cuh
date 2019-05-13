#pragma once

#include <blas.cuh>
#include <layer.cuh>

#include <unordered_map>

#ifdef DEBUG

void operator_sigmoid(const Storage *input1, Storage *output);

void operator_d_sigmoid(
    const Storage *outputs_grad, const Storage *input1, Storage *inputs_grad,
    std::unordered_map<std::string, std::unique_ptr<Storage>> &temp);

#endif  // DEBUG

class Sigmoid : public Layer {
 public:
  void forward();
  void backward();

 private:
  std::unordered_map<std::string, std::unique_ptr<Storage>> temp;
};