#include <relu.cuh>

struct relu_functor {
  __host__ __device__ float operator()(const float &x) const {
    return fmaxf(0, x);
  }
};

void operator_relu(const Storage *input1, Storage *outputs) {
  outputs->data.resize(input1->data.size());
  outputs->reshape(input1->shape);

  relu_functor f;
  thrust::transform(input1->data.begin(), input1->data.end(),
                    outputs->data.begin(), f);
}

struct relu_d_functor {
  __host__ __device__ float operator()(const float &x) const {
    return x > FLT_EPSILON ? 1 : 0;
  }
};

// Y = relu(X)
// dL/dX = relu'(X) element_mul dL/dY
void operator_d_relu(const Storage *outputs_grad, const Storage *input1,
                     Storage *intputs_grad) {
  Storage d_relu(input1->shape);
  relu_d_functor f;
  thrust::transform(input1->data.begin(), input1->data.end(),
                    d_relu.data.begin(), f);

  operator_mul(&d_relu, outputs_grad, intputs_grad);
}