#include <test_tools.h>
#include <nll_loss.cuh>
#include <softmax.cuh>

#include <gtest/gtest.h>
#include <thrust/copy.h>

TEST(SoftMax, Forward) {
  Storage X({3, 3}, {1, 2, 3, 1, 2, 3, 1, 2, 3});
  Storage output({3, 3});
  operator_log_softmax(&X, 1, &output);

  device_vector_cout(output.get_data());
  // {-2.4076, -1.4076, -0.4076, -2.4076, -1.4076, -0.4076,
  //                   -2.4076, -1.4076, -0.4076}
}

TEST(SoftMax, Backward) {
  Storage output_grad({3, 3}, {0, 0, -1 / 3, -1 / 3, 0, 0, 0, -1 / 3, 0});
  Storage input({3, 3}, {0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33});

  Storage input_grad({3, 3});
  operator_d_log_softmax(&output_grad, &input, 1, &input_grad);
  device_vector_cout(input_grad.get_data());
}

TEST(SoftMax, SoftMaxLossForwardBackward1) {
  // case 1
  Storage Y({3, 3}, {0, 0, 1, 0, 1, 0, 1, 0, 0});
  Storage X({3, 3}, {0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 0.33});

  // forward
  Storage logp({3, 3});
  operator_log_softmax(&X, 1, &logp);
  Storage nll_loss({1, 1});
  std::unordered_map<std::string, std::unique_ptr<Storage>> temp;
  operator_nll_loss(&logp, &Y, &nll_loss, temp);
  device_vector_cout(nll_loss.get_data());

  // backward
  Storage nll_grad({3, 3});
  operator_d_nll_loss(&Y, &nll_grad);
  device_vector_cout(nll_grad.get_data());

  Storage softmax_grad({3, 3});
  operator_d_log_softmax(&nll_grad, &X, 1, &softmax_grad);
  device_vector_cout(softmax_grad.get_data());

  // test with dL/dX = softmax(X) - dL/dY
  // tensor([[ 0.3333,  0.3333, -0.6667],
  //       [-0.6667,  0.3333, -0.6667],
  //       [-0.6667,  0.3333,  0.3333]])
}