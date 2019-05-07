#include <sigmoid.cuh>

struct sigmoid_functor {
  __host__ __device__ float operator()(const float &x) const {
    return 1 / (1 + expf(-x));
  }
};

struct sigmoid_d_functor {
  __host__ __device__ float operator()(const float &x) const {
    float s = 1 / (1 + expf(-x));
    return s * (1 - s);
  }
};

void operator_sigmoid(const Storage *input1, Storage *output) {
  sigmoid_functor f;
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    output->get_data().begin(), f);
}

// Y = sigmoid(X)
// dL/dX = sigmoid'(X) element_mul dL/dY
void operator_d_sigmoid(const Storage *outputs_grad, const Storage *input1,
                        Storage *inputs_grad) {
  Storage d_sigmoid(input1->get_shape());
  sigmoid_d_functor f;
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    d_sigmoid.get_data().begin(), f);

  operator_mul(&d_sigmoid, outputs_grad, inputs_grad);
}

void Sigmoid::forward() {
  const Storage *input = this->pre->get_output();

  if (this->output.get() == nullptr ||
      this->output->get_shape() != input->get_shape()) {
    this->output.reset(new Storage(input->get_shape()));
  }
  operator_sigmoid(input, this->output.get());
}

void Sigmoid::backward() {
  const Storage *output_grad = this->next->get_output();
  const Storage *input = this->pre->get_output();

  if (this->grad.get() == nullptr ||
      this->grad->get_shape() != output_grad->get_shape()) {
    this->grad.reset(new Storage(output_grad->get_shape()));
  }
  operator_d_sigmoid(output_grad, input, this->grad.get());
}
