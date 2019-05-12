#include <test_tools.h>
#include <linear.cuh>

#include <gtest/gtest.h>
#include <thrust/copy.h>

#include <iostream>

TEST(LinearTest, WeightForward) {
  // duplicate
}

TEST(LinearTest, WeightBackward) {
  Storage outputs_grad({2, 3}, {0, 1, 2, 3, 4, 5});
  Storage inputs({2, 3}, {0, 1, 2, 3, 4, 5});
  Storage weights({3, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8});

  Storage weights_grad({3, 3});
  Storage inputs_grad({2, 3});
  std::unordered_map<std::string, std::unique_ptr<Storage>> temp;
  operator_d_linear(&outputs_grad, &inputs, &weights, &weights_grad,
                    &inputs_grad, temp);

  ASSERT_TRUE(device_vector_equals_vector(weights_grad.get_data(),
                                          {9, 12, 15, 12, 17, 22, 15, 22, 29}));
  ASSERT_TRUE(device_vector_equals_vector(inputs_grad.get_data(),
                                          {5, 14, 23, 14, 50, 86}));
}

TEST(LinearTest, BiasForward) {
  Storage bias({1, 3}, {0, 1, 2});
  Storage inputs({2, 3}, {0, 1, 2, 3, 4, 5});

  Storage output({2, 3});
  operator_linear_bias(&inputs, &bias, &output);

  ASSERT_TRUE(
      device_vector_equals_vector(output.get_data(), {0, 2, 4, 3, 5, 7}));
}

TEST(LinearTest, BiasBackward) {
  Storage outputs_grad({2, 3}, {0, 1, 2, 3, 4, 5});
  Storage biad_grad({1, 3});

  operator_d_linear_bias(&outputs_grad, &biad_grad);
  ASSERT_TRUE(
      device_vector_equals_vector(biad_grad.get_data(), {3, 5, 7}));
}