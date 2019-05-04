#include <test_tools.h>
#include <nll_loss.cuh>

#include <gtest/gtest.h>
#include <thrust/copy.h>

TEST(NLLLoss, Forward) {
  Storage Y(std::vector<int>{3, 3}, {0, 0, 1, 0, 1, 0, 1, 0, 0});
  Storage logP(std::vector<int>{3, 3},
               {0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33});

  std::unique_ptr<Storage> loss(operator_nll_loss(&logP, &Y));
  ASSERT_TRUE(device_vector_equals_vector(loss->shape, {1, 1}));
  device_vector_cout(loss->data);

  Storage logP2(std::vector<int>{3, 3}, {0, 0, 1, 0, 1, 0, 1, 0, 0});
  std::unique_ptr<Storage> loss2(operator_nll_loss(&logP2, &Y));
  ASSERT_TRUE(device_vector_equals_vector(loss->shape, {1, 1}));
  device_vector_cout(loss2->data);
}

TEST(NLLLoss, Backward) {
  Storage Y(std::vector<int>{3, 3}, {0, 0, 1, 0, 1, 0, 1, 0, 0});
  std::unique_ptr<Storage> loss_grad(operator_d_nll_loss(&Y));
  ASSERT_TRUE(device_vector_equals_vector(loss_grad->shape, {3, 3}));
  device_vector_cout(loss_grad->data);
}