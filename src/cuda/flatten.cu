#pragma once

#include <layer.cuh>
#include <storage.cuh>
#include <utils.cuh>
#include <flatten.cuh>

void Flatten::forward() {
  Storage *input = this->pre->get_output();
  this->in_shape = input->get_shape();

  int num = 1;
  for (int i = 1; i < this->in_shape.size(); i++) num *= this->in_shape[i];
  std::vector<int> out_shape{this->in_shape[0], num};

  if (this->inplace) {
    input->reshape(out_shape);
  } else {
    INIT_STORAGE(this->output, out_shape);
    this->output->get_data() = input->get_data();
  }
}

void Flatten::backward() {
  Storage *output_grad = this->next->get_grad();

  if (this->inplace) {
    output_grad->reshape(this->in_shape);
  } else {
    INIT_STORAGE(this->grad, this->in_shape);
    this->grad->get_data() = output_grad->get_data();
  }
}
