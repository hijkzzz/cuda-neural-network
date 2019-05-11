#include <relu.cuh>

struct relu_functor {
  __host__ __device__ float operator()(const float &x) const {
    return fmaxf(0, x);
  }
};

struct relu_d_functor {
  __host__ __device__ float operator()(const float &x, const float &y) const {
    return x > FLT_EPSILON ? y : 0;
  }
};

void operator_relu(const Storage *input1, Storage *outputs) {
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    outputs->get_data().begin(), relu_functor());
}

// Y = relu(X)
// dL/dX = relu'(X) element_mul dL/dY
void operator_d_relu(const Storage *outputs_grad, const Storage *input1,
                     Storage *intputs_grad) {
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    outputs_grad->get_data().begin(),
                    intputs_grad->get_data().begin(), relu_d_functor());
}

void ReLU::forward() {
  Storage *input = this->pre->get_output();

  if (this->inplace) {
    operator_relu(input, input);
  } else {
    INIT_STORAGE(this->output, input->get_shape());
    operator_relu(input, this->output.get());
  }
}

void ReLU::backward() {
  Storage *input = this->pre->get_output();
  Storage *output_grad = this->next->get_grad();

  if (this->inplace) {
    operator_d_relu(output_grad, input, output_grad);
  } else {
    INIT_STORAGE(this->grad, input->get_shape());
    operator_d_relu(output_grad, input, this->grad.get());
  }
}