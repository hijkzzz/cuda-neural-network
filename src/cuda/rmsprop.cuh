#pragma once

#include <blas.cuh>
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <memory>

struct rms_suqare_grads_functor {
  const float b;

  rms_suqare_grads_functor(float _b) : b(_b) {}

  __host__ __device__ float operator()(const float &x, const float &y) const {
    float sq = powf(y, 2);
    return b * x + (1 - b) * sq;
  }
};

struct rms_grads_functor {
  const float a;

  rms_grads_functor(float _a) : a(_a) {}

  __host__ __device__ float operator()(const float &x, const float &y) const {
    return a * (x / sqrtf(y));
  }
};

void rmsprop_update(Storage *square_grads, Storage *weights,
                    const Storage *grads, float beta = 0.999,
                    float learning_rate) {
  rms_suqare_grads_functor sgf(beta);
  thrust::device_vector<float> new_square_grads(square_grads->data.size());
  thrust::transform(square_grads->data.begin(), square_grads->data.end(),
                    grads->data.begin(), grads->data.end(),
                    new_square_grads.begin(), sgf);
  square_grads->data = std::move(new_square_grads);

  rms_grads_functor gf(learning_rate);
  thrust::device_vector<float> new_grads(grads->data.size());
  thrust::transform(grads->data.begin(), grads->data.end(),
                    square_grads->data.begin(), square_grads->data.end(),
                    new_grads.begin(), gf);

  thrust::device_vector<float> new_weights(weights->data.size());
  thrust::transform(weights->data.begin(), weights->data.end(),
                    new_grads->begin(), new_grads->end(), new_weights.begin(),
                    thrust::minus<float>());
  weights->data = std::move(new_weights);
}
