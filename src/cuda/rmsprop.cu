#include <rmsprop.cuh>

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <memory>

#define SQUARE_GRDAD_DEFALUT 0.001

struct update_functor {
  const float lr;
  const float l2;
  const float beta;

  update_functor(float lr, float l2, float beta) : lr(lr), l2(l2), beta(beta) {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    float grad = thrust::get<1>(t) + l2 * thrust::get<0>(t);
    float square_grad = grad * grad;
    float mean_square_grad =
        beta * thrust::get<2>(t) + (1 - beta) * square_grad;
    thrust::get<2>(t) = mean_square_grad;

    // weights -= learning_rate * grad / (sqrt(mean_square_grad) + 1e-10)
    thrust::get<0>(t) -= (lr * grad / (sqrtf(mean_square_grad) + 1e-10));
  }
};

void rmsprop_update(Storage *square_grads, Storage *weights,
                    const Storage *grads, float learning_rate, float l2,
                    float beta) {
  CHECK_EQ(square_grads->get_data().size(), grads->get_data().size(),
           "RMSProp: grads size error 1");
  CHECK_EQ(weights->get_data().size(), grads->get_data().size(),
           "RMSProp: grads size error 2");

  // update weights
  thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(
                       weights->get_data().begin(), grads->get_data().begin(),
                       square_grads->get_data().begin())),
                   thrust::make_zip_iterator(thrust::make_tuple(
                       weights->get_data().end(), grads->get_data().end(),
                       square_grads->get_data().end())),
                   update_functor(learning_rate, l2, beta));
}

void RMSProp::regist(std::vector<std::pair<Storage *, Storage *>> params) {
  for (auto iter = params.begin(); iter != params.end(); iter++) {
    this->parameter_list.push_back(iter->first);
    this->grad_list.push_back(iter->second);
    this->square_grad.emplace_back(std::make_unique<Storage>(
        iter->first->get_shape(), SQUARE_GRDAD_DEFALUT));
  }
}

void RMSProp::step() {
  for (int i = 0; i < this->parameter_list.size(); i++) {
    rmsprop_update(this->square_grad[i].get(), this->parameter_list[i],
                   this->grad_list[i], this->learning_rate, this->l2,
                   this->beta);
  }
}