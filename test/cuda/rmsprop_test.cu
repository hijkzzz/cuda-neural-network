#include <test_tools.h>
#include <rmsprop.cuh>

#include <gtest/gtest.h>
#include <thrust/copy.h>

TEST(RMSProp, Update) {
  Storage grads({3, 3}, 0.1);
  Storage weights(std::vector<int>({3, 3}), 0.3);
  Storage square_grads(std::vector<int>({3, 3}), 1);

  for (int i = 1; i <= 100; i++) {
    rmsprop_update(&square_grads, &weights, &grads, 0.01);
    if (i % 20 == 0) {
      std::cout << "update grads:" << i << std::endl;
      device_vector_cout(square_grads.get_data());
      device_vector_cout(weights.get_data());
    }
  }
}

TEST(RMSProp, UpdateWithL2) {
  Storage grads({3, 3}, 0.1);
  Storage weights(std::vector<int>({3, 3}), 0.3);
  Storage square_grads(std::vector<int>({3, 3}), 1);


  for (int i = 1; i <= 100; i++) {
    rmsprop_update(&square_grads, &weights, &grads, 0.01, 0.001);
    if (i % 20 == 0) {
      std::cout << "update grads with L2:" << i << std::endl;
      device_vector_cout(square_grads.get_data());
      device_vector_cout(weights.get_data());
    }
  }
}