#include <test_tools.h>
#include <max_pool.cuh>

#include <gtest/gtest.h>
#include <thrust/copy.h>

TEST(MaxPool, Forward) {
  Storage a({2, 2, 4, 4},
            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
             48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
  Storage m({1, 1, 1});
  std::unique_ptr<Storage> pooled(operator_max_pool(&a, &m, 2, 2, 0, 0, 2, 2));

  ASSERT_TRUE(device_vector_equals_vector(pooled->shape, {2, 2, 2, 2}));
  ASSERT_TRUE(device_vector_equals_vector(m.shape, {2, 2, 2, 2}));

  std::cout << "pooled" << std::endl;
  device_vector_cout(pooled->data);
  std::cout << "mask" << std::endl;
  device_vector_cout(m.data);
}

TEST(MaxPool, Backward) {
  Storage a({2, 2, 4, 4},
            {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
             16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
             32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
             48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63});
  Storage m({1, 1, 1});
  std::unique_ptr<Storage> pooled(operator_max_pool(&a, &m, 2, 2, 0, 0, 2, 2));

  ASSERT_TRUE(device_vector_equals_vector(pooled->shape, {2, 2, 2, 2}));
  ASSERT_TRUE(device_vector_equals_vector(m.shape, {2, 2, 2, 2}));

  std::unique_ptr<Storage> unpooled(
      operator_d_max_pool(pooled.get(), &a, &m, 2, 2, 0, 0, 2, 2));

  ASSERT_TRUE(device_vector_equals_vector(unpooled->shape, {2, 2, 4, 4}));
  std::cout << "unpooled" << std::endl;
  device_vector_cout(unpooled->data);
}