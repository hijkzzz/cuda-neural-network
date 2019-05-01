#include <storage.cuh>
#include <test_tools.h>

#include <gmock/gmock.h>
#include <thrust/copy.h>

#include <iostream>

TEST(StorageTest, Constructor) {
  std::vector<float> temp(27, 1);

  Storage a({3, 3, 3}, 1);
  ASSERT_EQ(a.data.size(), 27);
  ASSERT_TRUE(device_vector_equals_vector(a.data, temp));

  thrust::device_vector<int> shape(3, 3);
  thrust::device_vector<float> data(3 * 3 * 3, 1);
  Storage b(shape, std::move(data));
  ASSERT_EQ(data.size(), 0);
  ASSERT_TRUE(device_vector_equals_vector(a.data, temp));

  thrust::device_vector<float> data2(3 * 3 * 3, 1);
  Storage c({3, 3, 3}, data2.begin(), data2.end());
  ASSERT_TRUE(device_vector_equals_vector(a.data, temp));
}

TEST(StorageTest, Reshape) {
  std::vector<int> temp{9, 3};

  Storage a({3, 3, 3}, 1);
  a.reshape({9, 3});
  ASSERT_TRUE(device_vector_equals_vector(a.shape, temp));
  try {
    a.reshape({9, 1});
    ASSERT_EQ(0, 1);
  } catch (const std::exception&) {
  }
}

TEST(StorageTest, Xavier) {
  Storage a({3, 3, 3});
  a.xavier(128, 128);

  device_vector_cout(a.data);
}