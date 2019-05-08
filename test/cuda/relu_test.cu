#include <test_tools.h>
#include <relu.cuh>

#include <gtest/gtest.h>
#include <thrust/copy.h>

TEST(ReLU, Forward) {
  Storage a({3, 3, 3}, {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0,  1,  2, 3,
                        4,   5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16});
  Storage result({3, 3, 3});
  operator_relu(&a, &result);

  ASSERT_TRUE(device_vector_equals_vector(
      result.get_data(), {0, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  1,  2, 3,
                          4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}));
}

TEST(ReLU, Backward) {
  Storage a({3, 3, 3}, {-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0,  1,  2, 3,
                        4,   5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16});
  Storage o({3, 3, 3}, 3);
  Storage result({3, 3, 3});
  operator_d_relu(&o, &a, &result);

  ASSERT_TRUE(device_vector_equals_vector(
      result.get_data(), {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3,
                          3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3}));
}