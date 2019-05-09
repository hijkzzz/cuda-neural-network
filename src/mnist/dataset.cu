#include <dataset.cuh>

#include <algorithm>
#include <fstream>
#include <random>
#include <chrono>

DataSet::DataSet(std::string mnist_data_path, bool shuffle) : shuffle(shuffle) {
  // train data
  this->read_images(mnist_data_path + "/train-images.idx3-ubyte",
                    this->train_data);
  this->read_labels(mnist_data_path + "/train-labels.idx1-ubyte",
                    this->train_label);
  // test data
  this->read_images(mnist_data_path + "/t10k-images.idx3-ubyte",
                    this->test_data);
  this->read_labels(mnist_data_path + "/t10k-labels.idx1-ubyte",
                    this->test_label);
  // init
  this->reset();
}

void DataSet::reset() {
  this->train_data_index = 0;
  this->test_data_index = 0;

  if (shuffle) {
    // keep random seed same
    unsigned seed = std::chrono::system_clock::now ().time_since_epoch ().count ();

    std::shuffle(this->train_data.begin(), this->train_data.end(),
                 std::default_random_engine(seed));
    std::shuffle(this->train_label.begin(), this->train_label.end(),
                 std::default_random_engine(seed));
  }
}

void DataSet::forward(int batch_size, bool is_train) {
  if (is_train) {
    int start = this->train_data_index;
    int end = std::min(this->train_data_index + batch_size,
                       (int)this->train_data.size());
    this->train_data_index = end;
    int size = end - start;

    // init device memory
    std::vector<int> output_shape{size, 1, this->height, this->width};
    if (this->output.get() == nullptr ||
        this->output->get_shape() != output_shape) {
      this->output.reset(new Storage(output_shape));
    }
    this->output_label.reset(new Storage({size, 10}, 0));

    // copy to device memory
    int im_stride = 1 * this->height * this->width;
    int one_hot_stride = 10;

    std::vector<float> train_data_buffer;
    train_data_buffer.reserve(size * im_stride);

    for (int i = start; i < end; i++) {
      train_data_buffer.insert(train_data_buffer.end(),
                               this->train_data[i].begin(),
                               this->train_data[i].end());
      this->output_label
          ->get_data()[(i - start) * one_hot_stride + this->train_label[i]] = 1;
    }
    thrust::copy(train_data_buffer.begin(), train_data_buffer.end(),
                 this->output->get_data().begin());

  } else {
    int start = this->test_data_index;
    int end = std::min(this->test_data_index + batch_size,
                       (int)this->test_data.size());
    this->test_data_index = end;
    int size = end - start;

    // init device memory
    std::vector<int> output_shape{size, 1, this->height, this->width};
    if (this->output.get() == nullptr ||
        this->output->get_shape() != output_shape) {
      this->output.reset(new Storage(output_shape));
    }
    this->output_label.reset(new Storage({size, 10}, 0));

    // copy to device memory
    int im_stride = 1 * this->height * this->width;
    int one_hot_stride = 10;

    std::vector<float> test_data_buffer;
    test_data_buffer.reserve(size * im_stride);

    for (int i = start; i < end; i++) {
      test_data_buffer.insert(test_data_buffer.end(),
                              this->test_data[i].begin(),
                              this->test_data[i].end());
      this->output_label
          ->get_data()[(i - start) * one_hot_stride + this->test_label[i]] = 1;
    }
    thrust::copy(test_data_buffer.begin(), test_data_buffer.end(),
                 this->output->get_data().begin());
  }
}

bool DataSet::has_next(bool is_train) {
  if (is_train) {
    return this->train_data_index < this->train_data.size();
  } else {
    return this->test_data_index < this->test_data.size();
  }
}

void DataSet::print_im() {
  int size = this->output->get_shape()[0];
  int im_stride = 1 * height * width;

  for (int k = 0; k < size; k++) {
    int max_pos = -1;
    float max_value = -FLT_MAX;
    for (int i = 0; i < 10; i++) {
      float val = this->output_label->get_data()[k * 10 + i];
      if (val > max_value) {
        max_value = val;
        max_pos = i;
      }
    }

    std::cout << max_pos << std::endl;
    auto& data = this->output->get_data();
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        std::cout << (data[k * im_stride + i * width + j] > 0 ? "* " : "  ");
      }
      std::cout << std::endl;
    }
  }
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
                          std::vector<std::vector<float>>& output) {
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

    std::cout << file_name << std::endl;
    std::cout << "magic number = " << magic_number << std::endl;
    std::cout << "number of images = " << number_of_images << std::endl;
    std::cout << "rows = " << n_rows << std::endl;
    std::cout << "cols = " << n_cols << std::endl;

    this->height = n_rows;
    this->width = n_cols;

    std::vector<unsigned char> image(n_rows * n_cols);
    std::vector<float> normalized_image(n_rows * n_cols);

    for (int i = 0; i < number_of_images; i++) {
      file.read((char*)&image[0], sizeof(unsigned char) * n_rows * n_cols);

      for (int i = 0; i < n_rows * n_cols; i++) {
        normalized_image[i] = (float)image[i] / 255 - 0.5;
      }
      output.push_back(normalized_image);
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

    std::cout << file_name << std::endl;
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