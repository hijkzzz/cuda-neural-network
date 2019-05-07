#include <rmsprop.cuh>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <memory>

#define SQUARE_GRDAD_DEFALUT 0.01

struct l2_grads_functor {
  const float l2;

  l2_grads_functor(float _l2) : l2(_l2) {}

  __host__ __device__ float operator()(const float &x, const float &y) const {
    return x + l2 * y;
  }
};

struct suqare_grads_functor {
  const float b;

  suqare_grads_functor(float _b) : b(_b) {}

  __host__ __device__ float operator()(const float &x, const float &y) const {
    float sq = powf(y, 2);
    return b * x + (1 - b) * sq;
  }
};

struct update_functor {
  const float a;

  update_functor(float _a) : a(_a) {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    // weights -= learning_rate * grads / (sqrt(suaqre_grads) + 1e-10)
    thrust::get<0>(t) -=
        a * thrust::get<1>(t) / (sqrtf(thrust::get<2>(t)) + 1e-10);
  }
};

void rmsprop_update(Storage *square_grads, Storage *weights,
                    const Storage *grads, float learning_rate, float l2,
                    float beta) {
  CHECK_EQ(square_grads->get_data().size(), grads->get_data().size(),
           "RMSProp: grads size error 1");
  CHECK_EQ(weights->get_data().size(), grads->get_data().size(),
           "RMSProp: grads size error 2");

  // add L2 weights grads
  thrust::device_vector<float> l2_grads(grads->get_data().size());
  thrust::transform(grads->get_data().begin(), grads->get_data().end(),
                    weights->get_data().begin(), l2_grads.begin(),
                    l2_grads_functor(l2));

  // square grads
  thrust::transform(square_grads->get_data().begin(),
                    square_grads->get_data().end(), l2_grads.begin(),
                    square_grads->get_data().begin(),
                    suqare_grads_functor(beta));

  // update weights
  thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                       weights->get_data().begin(), l2_grads.begin(),
                       square_grads->get_data().begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(
                       weights->get_data().end(), l2_grads.end(),
                       square_grads->get_data().end())),
                   update_functor(learning_rate));
}

void RMSProp::regist(std::vector<std::pair<Storage *, Storage *>> params) {
  for (auto iter = params.begin(); iter != params.end(); iter++) {
    this->parameter_list.push_back(iter->first);
    this->grad_list.push_back(iter->second);
    this->square_grad.emplace_back(std::make_unique<Storage>(
        iter->second->get_shape(), SQUARE_GRDAD_DEFALUT));
  }
}

void RMSProp::step() {
  for (int i = 0; i < this->parameter_list.size(); i++) {
    rmsprop_update(this->square_grad[i].get(), this->parameter_list[i],
                   this->grad_list[i], this->learning_rate, this->l2,
                   this->beta);
  }
}