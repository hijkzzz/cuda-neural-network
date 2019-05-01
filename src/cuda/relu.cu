#include <relu.cuh>

struct relu_functor {
  __host__ __device__ float operator()(const float &x) const {
    return fmaxf(0, x);
  }
};

Storage *operator_relu(const Storage *input1) {
  Storage *output = new Storage(input1->shape);
  relu_functor f;
  thrust::transform(input1->data.begin(), input1->data.end(),
                    output->data.begin(), f);

  return output;
}

struct relu_d_functor {
  __host__ __device__ float operator()(const float &x) const {
    return x > 0 ? 1 : 0;
  }
};

// Y = relu(X)
// dL/dX = relu'(X) element_mul dL/dY
Storage *operator_d_relu(const Storage *input1, const Storage *outputs_grad) {
  Storage *d_relu = new Storage(input1->shape);
  relu_d_functor f;
  thrust::transform(input1->data.begin(), input1->data.end(),
                    d_relu->data.begin(), f);

  return operator_mul(d_relu, outputs_grad);
}