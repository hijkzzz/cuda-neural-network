#include <test_tools.h>
#include <storage.cuh>

#include <gtest/gtest.h>
#include <thrust/copy.h>

#include <iostream>

TEST(StorageTest, Constructor) {
  std::vector<float> temp(27, 1);

  Storage a({3, 3, 3}, 1);
  ASSERT_EQ(a.get_data().size(), 27);
  ASSERT_TRUE(device_vector_equals_vector(a.get_data(), temp));

  Storage d({3, 3, 3}, temp);
  ASSERT_EQ(d.get_data().size(), 27);
  ASSERT_TRUE(device_vector_equals_vector(d.get_data(), temp));
}

TEST(StorageTest, Reshape) {
  std::vector<int> temp{9, 3};

  Storage a({3, 3, 3}, 1);
  a.reshape(temp);
  ASSERT_EQ(a.get_shape(), temp);
  ASSERT_EXIT(a.reshape({9, 1}), ::testing::ExitedWithCode(1), "error");
}

TEST(StorageTest, Xavier) {
  Storage a({3, 3, 3});
  a.xavier(128, 128);

  device_vector_cout(a.get_data());
}