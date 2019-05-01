#include <rmsprop.cuh>

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
    return a * x / (sqrtf(y) + 1e-10);
  }
};

struct l2_grads_functor {
  const float l2;

  l2_grads_functor(float _l2) : l2(_l2) {}

  __host__ __device__ float operator()(const float &x, const float &y) const {
    return x + l2 * 2 * y;
  }
};

void rmsprop_update(Storage *square_grads, Storage *weights,
                    const Storage *grads, float beta, float learning_rate,
                    float l2) {
  // add L2 weights grads
  l2_grads_functor l2f(l2);
  thrust::device_vector<float> l2_grads(grads->data.size());
  thrust::transform(grads->data.begin(), grads->data.end(),
                    weights->data.begin(), l2_grads.begin(), l2f);

  // rms
  rms_suqare_grads_functor sgf(beta);
  thrust::device_vector<float> new_square_grads(square_grads->data.size());
  thrust::transform(square_grads->data.begin(), square_grads->data.end(),
                    l2_grads.begin(), new_square_grads.begin(), sgf);
  square_grads->data = std::move(new_square_grads);

  rms_grads_functor gf(learning_rate);
  thrust::device_vector<float> rms_grads(grads->data.size());
  thrust::transform(grads->data.begin(), grads->data.end(),
                    square_grads->data.begin(), rms_grads.begin(), gf);

  // update grads
  thrust::device_vector<float> new_weights(weights->data.size());
  thrust::transform(weights->data.begin(), weights->data.end(),
                    rms_grads.begin(), new_weights.begin(),
                    thrust::minus<float>());
  weights->data = std::move(new_weights);
}
