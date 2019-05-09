#include <test_tools.h>
#include <blas.cuh>

#include <gtest/gtest.h>
#include <thrust/copy.h>

#include <iostream>

TEST(BlasTest, Add) {
  Storage a({3, 3, 3}, -1);
  Storage b({3, 3, 3}, 1);

  Storage result({3, 3, 3}, 0.5);
  operator_add(&a, &b, &result);
  ASSERT_TRUE(device_vector_equals_vector(result.get_data(),
                                          std::vector<float>(27, 0)));
}

TEST(BlasTest, Sub) {
  Storage a({3, 3, 3}, 1);
  Storage b({3, 3, 3}, 1);

  Storage result({3, 3, 3}, 0.5);
  operator_sub(&a, &b, &result);
  ASSERT_TRUE(device_vector_equals_vector(result.get_data(),
                                          std::vector<float>(27, 0)));
}

TEST(BlasTest, Mul) {
  Storage a({3, 3, 3}, -1.5);
  Storage b({3, 3, 3}, 2);

  Storage result({3, 3, 3}, 0.5);
  operator_mul(&a, &b, &result);
  ASSERT_TRUE(device_vector_equals_vector(result.get_data(),
                                          std::vector<float>(27, -3)));
}

TEST(BlasTest, Div) {
  Storage a({3, 3, 3}, 1);
  Storage b({3, 3, 3}, 2);

  Storage result({3, 3, 3});
  operator_div(&a, &b, &result);
  ASSERT_TRUE(device_vector_equals_vector(result.get_data(),
                                          std::vector<float>(27, 0.5)));
}

TEST(BlasTest, Log) {
  Storage a({3, 3, 3}, 3);

  std::vector<float> temp(27, log(3));
  Storage result({3, 3, 3});
  operator_log(&a, &result);
  ASSERT_TRUE(device_vector_equals_vector(result.get_data(), temp));
}

TEST(BlasTest, Pow) {
  Storage a({3, 3, 3}, 3);

  std::vector<float> temp(27, 9);
  Storage result({3, 3, 3});
  operator_pow(&a, 2, &result);
  ASSERT_TRUE(device_vector_equals_vector(result.get_data(), temp));
}

TEST(BlasTest, Exp) {
  Storage a({3, 3, 3}, 2);

  std::vector<float> temp(27, exp(2));
  Storage result({3, 3, 3});
  operator_exp(&a, &result);
  ASSERT_TRUE(device_vector_equals_vector(result.get_data(), temp));
}

TEST(BlasTest, Matmul) {
  // matrix
  Storage wt({3, 3}, {0, 3, 6, 1, 4, 7, 2, 5, 8});
  Storage o_grad({2, 3}, {0, 1, 2, 3, 4, 5});
  Storage x_grad({2, 3});
  operator_matmul(&o_grad, &wt, &x_grad);

  ASSERT_TRUE(
      device_vector_equals_vector(x_grad.get_data(), {5, 14, 23, 14, 50, 86}));

  // batch matrix
  Storage a({2, 2, 3}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
  Storage b({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

  Storage result({2, 2, 2});
  operator_matmul(&a, &b, &result);

  ASSERT_TRUE(device_vector_equals_vector(
      result.get_data(), {10, 13, 28, 40, 172, 193, 244, 274}));

  // death
  Storage k({2, 2, 2}, 1.5);
  ASSERT_EXIT(operator_matmul(&a, &k, &result), ::testing::ExitedWithCode(1),
              "error");

  // broadcast
  Storage c({3, 2}, {0, 1, 2, 3, 4, 5});
  operator_matmul(&a, &c, &result, 2);
  ASSERT_TRUE(device_vector_equals_vector(
      result.get_data(), {10, 13, 28, 40, 46, 67, 64, 94}));

  Storage d({2, 3}, {0, 1, 2, 3, 4, 5});
  operator_matmul(&d, &b, &result, 1);
  ASSERT_TRUE(device_vector_equals_vector(result.get_data(),
                                          {10, 13, 28, 40, 28, 31, 100, 112}));
}

TEST(BlasTest, Transpose) {
  Storage a({3, 2, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  Storage result({3, 3, 2});
  operator_transpose(&a, &result);

  std::vector<float> temp{0,  3, 1,  4,  2,  5,  6,  9,  7,
                          10, 8, 11, 12, 15, 13, 16, 14, 17};
  ASSERT_TRUE(device_vector_equals_vector(result.get_data(), temp));
}

TEST(BlasTest, Mean) {
  std::vector<float> tempp(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  Storage a({2, 3, 3}, tempp);
  Storage result({3, 3});
  operator_mean(&a, 0, &result);
  device_vector_cout(result.get_data());

  std::vector<float> temp({4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5});
  ASSERT_TRUE(device_vector_equals_vector(result.get_data(), temp));

  // corner case
  Storage b(std::vector<int>({1, 5}), {1, 2, 3, 4, 5});
  Storage result2({1, 1});
  operator_mean(&b, 1, &result2);
  ASSERT_TRUE(device_vector_equals_vector(result2.get_data(), {3}));
}

TEST(BlasTest, Sum) {
  std::vector<float> tempp(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  Storage a({2, 3, 3}, tempp);
  Storage result({3, 3});
  operator_sum(&a, 0, &result);
  device_vector_cout(result.get_data());

  std::vector<float> temp({9, 11, 13, 15, 17, 19, 21, 23, 25});
  ASSERT_TRUE(device_vector_equals_vector(result.get_data(), temp));

  // corner case
  Storage b(std::vector<int>({1, 5}), {1, 2, 3, 4, 5});
  Storage result2({1, 1});
  operator_sum(&b, 1, &result2);
  ASSERT_TRUE(device_vector_equals_vector(result2.get_data(), {15}));
}
