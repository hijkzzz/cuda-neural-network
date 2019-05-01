#pragma once

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <vector>

template <typename T>
void device_vector_cout(const thrust::device_vector<T> &dv) {
  thrust::copy(dv.begin(), dv.end(),
               std::ostream_iterator<T>(std::cout, ", "));
  std::cout << std::endl;
}

template <typename T>
bool device_vector_equals_vector(const thrust::device_vector<T> &dv,
                                 const std::vector<T> &v) {
  if (dv.size() != v.size()) return false;

  for (int i = 0; i < v.size(); i++)
    if (dv[i] != v[i]) return false;
  return true;
}