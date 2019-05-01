#include <gtest/gtest.h>
#include <storage.cuh>
#include <iostream>

TEST(StorageTest, constructor) {
  Storage a({3, 3, 3}, 1);
  ASSERT_EQ(a.data.size(), 3 * 3 * 3);
  for (unsigned int i = 0; i < 3 * 3 * 3; i++) {
    ASSERT_EQ(a.data[i], 1);
  }

  thrust::host_vector<unsigned int> shape(3, 3);
  thrust::device_vector<float> data(3 * 3 * 3, 2);
  Storage b(shape, std::move(data));
  ASSERT_EQ(data.size(), 0);
  for (unsigned int i = 0; i < 3 * 3 * 3; i++) {
    ASSERT_EQ(b.data[i], 2);
  }

  thrust::device_vector<float> data2(3 * 3 * 3, 3);
  Storage c({3, 3, 3}, data2.begin(), data2.end());
  for (unsigned int i = 0; i < 3 * 3 * 3; i++) {
    ASSERT_EQ(c.data[i], 3);
  }
}

TEST(StorageTest, reshape) {
  Storage a({3, 3, 3}, 1);
  a.reshape({9, 3});
  ASSERT_EQ(a.shape[0], 9);
  ASSERT_EQ(a.shape[1], 3);

  try {
    a.reshape({9, 1});
    ASSERT_EQ(0, 1);
  } catch (const std::exception&) {
  }
}

TEST(StorageTest, xavier) {
  Storage a({3, 3, 3});
  a.xavier(128, 128);

  for (unsigned int i = 0; i < 3 * 3 * 3; i++) {
    std::cout << a.data[i] << " ";
    std::cout << std::endl;
  }
}