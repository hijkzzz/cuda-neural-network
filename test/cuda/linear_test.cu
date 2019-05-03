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
  Storage weights_grad({1, 1, 1}, 1);

  std::unique_ptr<Storage> inputs_grad(
      operator_d_linear(&outputs_grad, &inputs, &weights, &weights_grad));

  ASSERT_TRUE(device_vector_equals_vector(weights_grad.shape, {3, 3}));
  ASSERT_TRUE(device_vector_equals_vector(weights_grad.data,
                                          {9, 12, 15, 12, 17, 22, 15, 22, 29}));

  ASSERT_TRUE(device_vector_equals_vector(inputs_grad->shape, {2, 3}));
  ASSERT_TRUE(
      device_vector_equals_vector(inputs_grad->data, {5, 14, 23, 14, 50, 86}));
}

TEST(LinearTest, BiasForward) {
  Storage bias({1, 3}, {0, 1, 2});
  Storage inputs({2, 3}, {0, 1, 2, 3, 4, 5});

  std::unique_ptr<Storage> output(operator_bias(&inputs, &bias));

  ASSERT_TRUE(device_vector_equals_vector(output->shape, {2, 3}));
  ASSERT_TRUE(
      device_vector_equals_vector(output->data, {0, 2, 4, 3, 5, 7}));
}

TEST(LinearTest, BiasBackward) {
  // duplicate
}