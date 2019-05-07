#pragma once

#include <layer.cuh>
#include <storage.cuh>

class ReShape : public Layer {
 public:
  void forward(std::vector<int> out_shape) {
    const Storage *input = this->pre->get_output();
    this->in_shape = input->get_shape();

    if (this->output.get() == nullptr ||
        this->output->get_shape() != out_shape) {
      this->output.reset(new Storage(out_shape));
    }

    *this->output = *input;
  }

  void backward() {
    const Storage *output_grad = this->next->get_grad();

    if (this->grad.get() == nullptr ||
        this->grad->get_shape() != this->in_shape) {
      this->grad.reset(new Storage(in_shape));
    }

    *this->grad = *output_grad;
  }

 private:
  std::vector<int> in_shape;
};