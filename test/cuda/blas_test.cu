#include <test_tools.h>
#include <blas.cuh>

#include <gmock/gmock.h>
#include <thrust/copy.h>

#include <iostream>

TEST(BlasTest, Add) {
  Storage a({3, 3, 3}, -1);
  Storage b({3, 3, 3}, 1);

  std::vector<float> temp(27, 0);
  std::unique_ptr<Storage> result(operator_add(&a, &b));
  ASSERT_TRUE(device_vector_equals_vector(result->data, temp));
}

TEST(BlasTest, Sub) {
  Storage a({3, 3, 3}, 1);
  Storage b({3, 3, 3}, 1);

  std::vector<float> temp(27, 0);
  std::unique_ptr<Storage> result(operator_sub(&a, &b));
  ASSERT_TRUE(device_vector_equals_vector(result->data, temp));
}

TEST(BlasTest, Mul) {
  Storage a({3, 3, 3}, -1.5);
  Storage b({3, 3, 3}, 2);

  std::vector<float> temp(27, -3);
  std::unique_ptr<Storage> result(operator_mul(&a, &b));
  ASSERT_TRUE(device_vector_equals_vector(result->data, temp));
}

TEST(BlasTest, Div) {
  Storage a({3, 3, 3}, 1);
  Storage b({3, 3, 3}, 2);

  std::vector<float> temp(27, 0.5);
  std::unique_ptr<Storage> result(operator_div(&a, &b));
  ASSERT_TRUE(device_vector_equals_vector(result->data, temp));
}

TEST(BlasTest, Log) {
  Storage a({3, 3, 3}, 3);

  std::vector<float> temp(27, log(3));
  std::unique_ptr<Storage> result(operator_log(&a));
  ASSERT_TRUE(device_vector_equals_vector(result->data, temp));
}

TEST(BlasTest, Pow) {
  Storage a({3, 3, 3}, 3);

  std::vector<float> temp(27, 9);
  std::unique_ptr<Storage> result(operator_pow(&a, 2));
  ASSERT_TRUE(device_vector_equals_vector(result->data, temp));
}

TEST(BlasTest, Exp) {
  Storage a({3, 3, 3}, 2);

  std::vector<float> temp(27, exp(2));
  std::unique_ptr<Storage> result(operator_exp(&a));
  ASSERT_TRUE(device_vector_equals_vector(result->data, temp));
}

TEST(BlasTest, Matmul) {
  Storage a({2, 2, 3}, 1);
  Storage b({2, 3, 2}, 1.5);
  Storage c({2, 2, 2}, 1.5);
  std::unique_ptr<Storage> result(operator_matmul(&a, &b));

  std::vector<float> temp(8, 4.5);
  std::vector<int> temp2({2, 2, 2});
  ASSERT_TRUE(device_vector_equals_vector(result->shape, temp2));
  ASSERT_TRUE(device_vector_equals_vector(result->data, temp));
  ASSERT_ANY_THROW(operator_matmul(&a, &c));

  Storage d({2, 3, 3},{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  std::vector<float> temp3{15,  18,  21,  42,  54,  66,  69,  90,  111,
                           366, 396, 426, 474, 513, 552, 582, 630, 678};

  std::unique_ptr<Storage> result2(operator_matmul(&d, &d));
  ASSERT_TRUE(device_vector_equals_vector(result2->data, temp3));
}

TEST(BlasTest, Transpose) {
  Storage a({2, 3, 3},
            {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  std::unique_ptr<Storage> result(operator_transpose(&a, 0, 1));

  std::vector<float> temp{0,  1,  2,  9, 10, 11, 3,  4,  5,
                          12, 13, 14, 6, 7,  8,  15, 16, 17};
  std::vector<int> temp2({3, 2, 3});
  ASSERT_TRUE(device_vector_equals_vector(result->shape, temp2));
  ASSERT_TRUE(device_vector_equals_vector(result->data, temp));
}

TEST(BlasTest, Mean) {
  std::vector<float> tempp(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  Storage a({2, 3, 3}, tempp);
  std::unique_ptr<Storage> result(operator_mean(&a, 0));

  std::vector<float> temp({4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5});
  std::vector<int> temp2({3, 3});
  ASSERT_TRUE(device_vector_equals_vector(result->shape, temp2));
  ASSERT_TRUE(device_vector_equals_vector(result->data, temp));
}

TEST(BlasTest, Sum) {
  std::vector<float> tempp(
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  Storage a({2, 3, 3}, tempp);
  std::unique_ptr<Storage> result(operator_mean(&a, 0));

  std::vector<float> temp{9, 11, 13, 15, 17, 19, 21, 23, 25};
  std::vector<int> temp2({3, 3});
  ASSERT_TRUE(device_vector_equals_vector(result->shape, temp2));
  ASSERT_TRUE(device_vector_equals_vector(result->data, temp));
}
