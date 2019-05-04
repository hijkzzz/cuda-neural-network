#include <test_tools.h>
#include <nll_loss.cuh>
#include <softmax.cuh>

#include <gtest/gtest.h>
#include <thrust/copy.h>

TEST(SoftMax, Forward) {
  Storage X(std::vector<int>{3, 3}, {1, 2, 3, 1, 2, 3, 1, 2, 3});
  std::unique_ptr<Storage> output(operator_log_softmax(&X, 1));

  ASSERT_TRUE(device_vector_equals_vector(output->shape, {3, 3}));
  device_vector_cout(output->data);
  // {-2.4076, -1.4076, -0.4076, -2.4076, -1.4076, -0.4076,
  //                   -2.4076, -1.4076, -0.4076}
}

TEST(SoftMax, Backward) {
  Storage output_grad(std::vector<int>{3, 3},
                      {0, 0, -1 / 3, -1 / 3, 0, 0, 0, -1 / 3, 0});
  Storage input(std::vector<int>{3, 3},
                {0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33});

  std::unique_ptr<Storage> input_grad(
      operator_d_log_softmax(&output_grad, 1, &input));
  ASSERT_TRUE(device_vector_equals_vector(input_grad->shape, {3, 3}));
  device_vector_cout(input_grad->data);
}

TEST(SoftMax, SoftMaxLossForwardBackward1) {
  // case 1
  Storage Y(std::vector<int>{3, 3}, {0, 0, 1, 0, 1, 0, 1, 0, 0});
  Storage X(std::vector<int>{3, 3},
            {0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33});

  std::unique_ptr<Storage> logp(operator_log_softmax(&X, 1));
  std::unique_ptr<Storage> nll_loss(operator_nll_loss(logp.get(), &Y));
  device_vector_cout(nll_loss->data);

  std::unique_ptr<Storage> nll_grad(operator_d_nll_loss(&Y));
  device_vector_cout(nll_grad->data);
  std::unique_ptr<Storage> softmax_grad(
      operator_d_log_softmax(nll_grad.get(), 1, &X));

  ASSERT_TRUE(device_vector_equals_vector(softmax_grad->shape, {3, 3}));
  device_vector_cout(softmax_grad->data);

  // test with dL/dX = softmax(X) - dL/dY
  // tensor([[ 0.3333,  0.3333, -0.6667],
  //       [-0.6667,  0.3333, -0.6667],
  //       [-0.6667,  0.3333,  0.3333]])
}

TEST(SoftMax, SoftMaxLossForwardBackward2) {
  // case 2
  Storage Y(std::vector<int>{3, 3}, {0, 0, 1, 0, 1, 0, 1, 0, 0});
  Storage X(std::vector<int>{3, 3}, {0, 0, 999999, 0, 999999, 0, 999999, 0, 0});

  std::unique_ptr<Storage> logp(operator_log_softmax(&X, 1));
  std::unique_ptr<Storage> nll_loss(operator_nll_loss(logp.get(), &Y));
  device_vector_cout(nll_loss->data);

  std::unique_ptr<Storage> nll_grad(operator_d_nll_loss(&Y));
  std::unique_ptr<Storage> softmax_grad(
      operator_d_log_softmax(nll_grad.get(), 1, &X));

  ASSERT_TRUE(device_vector_equals_vector(softmax_grad->shape, {3, 3}));
  device_vector_cout(softmax_grad->data);

  // test with dL/dX = softmax(X) - Y
}