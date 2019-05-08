#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class DataSet {
 public:
  DataSet(std::string minist_data_path, bool shuffle = false);

  std::pair<std::vector<std::vector<float>>, std::vector<unsigned char>>
  get_train_data(int batch_size);
  std::pair<std::vector<std::vector<float>>, std::vector<unsigned char>>
  get_test_data(int batch_size);

  void reset();

  int get_height() { return this->height; }
  int get_width() { return this->width; }

 private:
  unsigned int reverse_int(unsigned int i);  // big endian
  void read_images(std::string file_name,
                   std::vector<std::vector<float>>& output);
  void read_labels(std::string file_name, std::vector<unsigned char>& output);

  std::vector<std::vector<float>> train_data;
  std::vector<unsigned char> train_label;
  int train_data_index;

  std::vector<std::vector<float>> test_data;
  std::vector<unsigned char> test_label;
  int test_data_index;

  int height;
  int width;
  bool shuffle;
};
