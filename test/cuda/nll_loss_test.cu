#include <test_tools.h>
#include <nll_loss.cuh>

#include <gtest/gtest.h>
#include <thrust/copy.h>

TEST(NLLLoss, Forward) {
  Storage Y({3, 3}, {0, 0, 1, 0, 1, 0, 1, 0, 0});
  Storage logP({3, 3}, {0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33});

  Storage loss({1, 1});
  std::unordered_map<std::string, std::unique_ptr<Storage>> temp;
  operator_nll_loss(&logP, &Y, &loss, temp);
  device_vector_cout(loss.get_data());

  operator_nll_loss(&Y, &Y, &loss, temp);
  device_vector_cout(loss.get_data());
}

TEST(NLLLoss, Backward) {
  Storage Y(std::vector<int>{3, 3}, {0, 0, 1, 0, 1, 0, 1, 0, 0});
  Storage loss_grad({3, 3});

  operator_d_nll_loss(&Y, &loss_grad);
  device_vector_cout(loss_grad.get_data());
}