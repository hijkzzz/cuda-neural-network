#include <test_tools.h>
#include <sigmoid.cuh>

#include <gtest/gtest.h>
#include <thrust/copy.h>

TEST(Sigmoid, Forward) {
  Storage a({3, 3, 3}, 3);
  Storage result({3, 3, 3});
  operator_sigmoid(&a, &result);

  ASSERT_TRUE(device_vector_equals_vector(
      result.get_data(), std::vector<float>(27, 1 / (1 + exp(-3)))));
}

TEST(Sigmoid, Backward) {
  Storage a({3, 3, 3}, 3);
  Storage o({3, 3, 3}, 3);
  Storage result({3, 3, 3});

  std::unordered_map<std::string, std::unique_ptr<Storage>> temp;
  operator_d_sigmoid(&o, &a, &result, temp);

  float x = 1 / (1 + exp(-3));
  float xx = (1 - x) * x;
  ASSERT_TRUE(device_vector_equals_vector(result.get_data(),
                                          std::vector<float>(27, xx * 3)));
}