#pragma once

#include <layer.cuh>
#include <storage.cuh>

class ReShape : public Layer {
 public:
  void forward(std::vector<int> out_shape) {
    const Storage *input = this->pre->get_output();
    this->in_shape = input->get_shape();

    this->output.reset(new Storage(*input));
    this->output->reshape(out_shape);
  }

  void backward() {
    const Storage *output_grad = this->next->get_grad();
    this->grad.reset(new Storage(*output_grad));
    this->grad->reshape(in_shape);
  }

 private:
  std::vector<int> in_shape;
};