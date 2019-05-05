#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class DataSet {
 public:
  DataSet(std::string minist_data_path);

  std::pair<std::vector<std::vector<float>>, std::vector<unsigned char>> get_train_data(
      int batch_size);
  std::pair<std::vector<std::vector<float>>, std::vector<unsigned char>> get_test_data(
      int batch_size);

 private:
  unsigned int reverse_int(unsigned int i); // big endian
  void read_images(std::string file_name, std::vector<std::vector<unsigned char>> & output);
  void read_labels(std::string file_name, std::vector<unsigned char> & output);

  std::vector<std::vector<unsigned char>> train_data;
  std::vector<unsigned char> train_label;
  int train_data_index;

  std::vector<std::vector<unsigned char>> test_data;
  std::vector<unsigned char> test_label;
  int test_data_index;
};
