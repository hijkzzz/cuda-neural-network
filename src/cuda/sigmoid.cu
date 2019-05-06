#include <sigmoid.cuh>

struct sigmoid_functor {
  __host__ __device__ float operator()(const float &x) const {
    return 1 / (1 + expf(-x));
  }
};

void operator_sigmoid(const Storage *input1, Storage *output) {
  output->data.resize(input1->data.size());
  output->reshape(input1->shape);

  sigmoid_functor f;
  thrust::transform(input1->data.begin(), input1->data.end(),
                    output->data.begin(), f);
}

struct sigmoid_d_functor {
  __host__ __device__ float operator()(const float &x) const {
    float s = 1 / (1 + expf(-x));
    return s * (1 - s);
  }
};

// Y = sigmoid(X)
// dL/dX = sigmoid'(X) element_mul dL/dY
void operator_d_sigmoid(const Storage *outputs_grad, const Storage *input1,
                        Storage *inputs_grad) {
  Storage d_sigmoid(input1->shape);
  sigmoid_d_functor f;
  thrust::transform(input1->data.begin(), input1->data.end(),
                    d_sigmoid.data.begin(), f);

  operator_mul(&d_sigmoid, outputs_grad, inputs_grad);
}
