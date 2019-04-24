#include <sigmoid.cuh>

struct sigmoid_functor {
  __host__ __device__ float operator()(const float &x) const {
    return 1 / (1 + expf(-x));
  }
};

Storage *operator_sigmoid(const Storage *input1) {
  Storage *output = new Storage(input1->shape);
  sigmoid_functor f;
  thrust::transform(input1->data.begin(), input1->data.end(),
                    output->data.begin(), f);

  return output;
}