#pragma once

#include <blas.cuh>
#include <layer.cuh>

#ifdef DEBUG

void operator_linear(const Storage *inputs, const Storage *weights,
                     Storage *output);

void operator_d_linear(const Storage *outputs_grad, const Storage *inputs,
                       const Storage *weights, Storage *weights_grad,
                       Storage *inputs_grad);

void operator_linear_bias(const Storage *inputs, const Storage *bias,
                          Storage *output);

void operator_d_linear_bias(const Storage *outputs_grad, Storage *bias_grad);

#endif  // DEBUG

class Linear : public Layer {
 public:
  Linear(int in_size, int out_size, bool is_bias);

  std::vector<std::pair<Storage *, Storage *>> parameters();
  void forward();
  void backward();

 private:
  std::unique_ptr<Storage> weights;
  std::unique_ptr<Storage> weights_grad;
  std::unique_ptr<Storage> bias;
  std::unique_ptr<Storage> bias_grad;

  int in_size;
  int out_size;
  bool is_bias;
};
