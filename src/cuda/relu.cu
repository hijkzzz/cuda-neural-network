#include <relu.cuh>

struct relu_functor {
  __host__ __device__ float operator()(const float &x) const {
    return fmaxf(0, x);
  }
};

struct relu_d_functor {
  __host__ __device__ float operator()(const float &x) const {
    return x > FLT_EPSILON ? 1 : 0;
  }
};

void operator_relu(const Storage *input1, Storage *outputs) {
  relu_functor f;
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    outputs->get_data().begin(), f);
}

// Y = relu(X)
// dL/dX = relu'(X) element_mul dL/dY
void operator_d_relu(const Storage *outputs_grad, const Storage *input1,
                     Storage *intputs_grad) {
  Storage d_relu(input1->get_shape());
  relu_d_functor f;
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    d_relu.get_data().begin(), f);

  operator_mul(&d_relu, outputs_grad, intputs_grad);
}

void ReLU::forward() { 
  const Storage *input = this->pre->get_output();

  if (this->output.get() == nullptr ||
      this->output->get_shape() != input->get_shape()) {
    this->output.reset(new Storage(input->get_shape()));
  }
  operator_relu(input, this->output.get());
}

void ReLU::backward() {
  const Storage *input = this->pre->get_output();
  const Storage *output_grad = this->next->get_grad();

  if (this->grad.get() == nullptr ||
      this->grad->get_shape() != input->get_shape()) {
    this->grad.reset(new Storage(input->get_shape()));
  }
  operator_d_relu(output_grad, input, this->grad.get());
}