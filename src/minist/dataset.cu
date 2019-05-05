#include <dataset.cuh>

#include <algorithm>
#include <fstream>

DataSet::DataSet(std::string minist_data_path) {
  // train data
  this->read_images(minist_data_path + "/train-images.idx3-ubyte",
                    this->train_data);
  this->read_labels(minist_data_path + "/train-labels.idx1-ubyte",
                    this->train_label);
  this->train_data_index = 0;

  // test data
  this->read_images(minist_data_path + "/t10k-images.idx3-ubyte",
                    this->test_data);
  this->read_labels(minist_data_path + "/t10k-labels.idx1-ubyte",
                    this->test_label);
  this->test_data_index = 0;
}

std::pair<std::vector<std::vector<float>>, std::vector<unsigned char>>
DataSet::get_train_data(int batch_size) {
  int start = this->train_data_index;
  int end = std::min(this->train_data_index + batch_size,
                     (int)this->train_data.size());

  std::vector<std::vector<float>> temp_data(this->train_data.begin() + start,
                                            this->train_data.begin() + end);
  // preprocess
  for_each(temp_data.begin(), temp_data.end(),
           [](float& x) { x = x / 255 - 0.5; });
  std::vector<unsigned char> temp_label(this->train_label.begin() + start,
                                        this->train_label.begin() + end);

  this->train_data_index = end;
  return {temp_data, temp_label};
}

std::pair<std::vector<std::vector<float>>, std::vector<unsigned char>>
DataSet::get_test_data(int batch_size) {
  int start = this->test_data_index;
  int end =
      std::min(this->test_data_index + batch_size, (int)this->test_data.size());

  std::vector<std::vector<float>> temp_data(this->test_data.begin() + start,
                                            this->test_data.begin() + end);
  // preprocess
  for_each(temp_data.begin(), temp_data.end(),
           [](float& x) { x = x / 255 - 0.5; });
  std::vector<unsigned char> temp_label(this->test_label.begin() + start,
                                        this->test_label.begin() + end);

  this->test_data_index = end;
  return {temp_data, temp_label};
}

unsigned int DataSet::reverse_int(unsigned int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((unsigned int)ch1 << 24) + ((unsigned int)ch2 << 16) +
         ((unsigned int)ch3 << 8) + ch4;
}

void DataSet::read_images(std::string file_name,
                          std::vector<std::vector<unsigned char>>& output) {
  std::ifstream file(file_name, std::ios::binary);
  if (file.is_open()) {
    unsigned int magic_number = 0;
    unsigned int number_of_images = 0;
    unsigned int n_rows = 0;
    unsigned int n_cols = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));
    file.read((char*)&n_rows, sizeof(n_rows));
    file.read((char*)&n_cols, sizeof(n_cols));
    magic_number = this->reverse_int(magic_number);
    number_of_images = this->reverse_int(number_of_images);
    n_rows = this->reverse_int(n_rows);
    n_cols = this->reverse_int(n_cols);

    std::cout << "magic number = " << magic_number << std::endl;
    std::cout << "number of images = " << number_of_images << std::endl;
    std::cout << "rows = " << n_rows << std::endl;
    std::cout << "cols = " << n_cols << std::endl;

    for (int i = 0; i < number_of_images; i++) {
      std::vector<unsigned char> image(n_rows * n_cols);
      file.read((char*)&image[0], sizeof(unsigned char) * n_rows * n_cols);
      output.push_back(image);
    }
  }
}

void DataSet::read_labels(std::string file_name,
                          std::vector<unsigned char>& output) {
  std::ifstream file(file_name, std::ios::binary);
  if (file.is_open()) {
    unsigned int magic_number = 0;
    unsigned int number_of_images = 0;
    file.read((char*)&magic_number, sizeof(magic_number));
    file.read((char*)&number_of_images, sizeof(number_of_images));

    magic_number = this->reverse_int(magic_number);
    number_of_images = this->reverse_int(number_of_images);
    std::cout << "magic number = " << magic_number << std::endl;
    std::cout << "number of images = " << number_of_images << std::endl;

    for (int i = 0; i < number_of_images; i++) {
      unsigned char label = 0;
      file.read((char*)&label, sizeof(label));
      output.push_back(label);
    }
  }
}