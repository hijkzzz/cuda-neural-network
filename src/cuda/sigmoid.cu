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
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    output->get_data().begin(), sigmoid_functor());
}

// Y = sigmoid(X)
// dL/dX = sigmoid'(X) element_mul dL/dY
void operator_d_sigmoid(
    const Storage *outputs_grad, const Storage *input1, Storage *inputs_grad,
    std::unordered_map<std::string, std::unique_ptr<Storage>> &temp) {
  INIT_TEMP(temp, "d_sigmoid", input1->get_shape());
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    temp["d_sigmoid"]->get_data().begin(), sigmoid_d_functor());

  operator_mul(temp["d_sigmoid"].get(), outputs_grad, inputs_grad);
}

void Sigmoid::forward() {
  const Storage *input = this->pre->get_output();

  INIT_STORAGE(this->output, input->get_shape());
  operator_sigmoid(input, this->output.get());
}

void Sigmoid::backward() {
  const Storage *input = this->pre->get_output();
  const Storage *output_grad = this->next->get_grad();

  INIT_STORAGE(this->grad, input->get_shape());
  operator_d_sigmoid(output_grad, input, this->grad.get(), this->temp);
}
