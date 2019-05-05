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
                    const Storage *grads, float learning_rate, float l2,
                    float beta) {
  // need reduce
  const Storage *reduce_grads = grads;
  if (grads->data.size() > weights->data.size()) {
    reduce_grads = operator_sum(grads, 0);
  }

  CHECK_EQ(square_grads->data.size(), reduce_grads->data.size(),
           "RMSProp: grads size error 1");
  CHECK_EQ(weights->data.size(), reduce_grads->data.size(),
           "RMSProp: grads size error 2");

  // add L2 weights grads
  l2_grads_functor l2f(l2);
  thrust::device_vector<float> l2_grads(reduce_grads->data.size());
  thrust::transform(reduce_grads->data.begin(), reduce_grads->data.end(),
                    weights->data.begin(), l2_grads.begin(), l2f);

  // rms grads
  rms_suqare_grads_functor sgf(beta);
  thrust::device_vector<float> new_square_grads(reduce_grads->data.size());
  thrust::transform(square_grads->data.begin(), square_grads->data.end(),
                    l2_grads.begin(), new_square_grads.begin(), sgf);
  square_grads->data = std::move(new_square_grads);

  rms_grads_functor gf(learning_rate);
  thrust::device_vector<float> rms_grads(reduce_grads->data.size());
  thrust::transform(reduce_grads->data.begin(), reduce_grads->data.end(),
                    square_grads->data.begin(), rms_grads.begin(), gf);

  // update weights
  thrust::device_vector<float> new_weights(weights->data.size());
  thrust::transform(weights->data.begin(), weights->data.end(),
                    rms_grads.begin(), new_weights.begin(),
                    thrust::minus<float>());
  weights->data = std::move(new_weights);

  // clean
  if (grads->data.size() > weights->data.size()) {
    delete reduce_grads;
  }
}
