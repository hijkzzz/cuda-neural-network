#pragma once
#include <blas.cuh>
#include <optimizer.cuh>
#include <unordered_map>
#include <vector>

#ifdef DEBUG

void rmsprop_update(Storage *square_grads, Storage *weights,
                    const Storage *grads, float learning_rate = 1e-2,
                    float l2 = 0, float beta = 0.99);

#endif  // DEBUG

class RMSProp : public Optimizer {
 public:
  explicit RMSProp(float learning_rate = 0.01, float l2 = 0.001,
                   float beta = 0.99)
      : learning_rate(learning_rate), l2(l2), beta(beta) {
    std::cout << "learning rate: " << learning_rate << ", l2: " << l2
              << ", beta: " << beta << std::endl;
  }

  void regist(std::vector<std::pair<Storage *, Storage *>> params);
  void step();

 private:
  std::vector<std::unique_ptr<Storage>> square_grad;

  float learning_rate;
  float l2;
  float beta;
};