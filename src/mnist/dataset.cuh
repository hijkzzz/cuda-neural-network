#pragma once

#include <layer.cuh>

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class DataSet : public Layer {
 public:
  explicit DataSet(std::string minist_data_path, bool shuffle = false);
  void reset();

  void forward(int batch_size, bool is_train);
  bool has_next(bool is_train);

  int get_height() { return this->height; }
  int get_width() { return this->width; }
  Storage* get_label() { return this->output_label.get(); }

  void print_im();

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
  std::unique_ptr<Storage> output_label;
};
