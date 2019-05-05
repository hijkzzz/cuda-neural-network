#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include <blas.cuh>
#include <conv.cuh>
#include <linear.cuh>
#include <nll_loss.cuh>
#include <relu.cuh>
#include <rmsprop.cuh>
#include <softmax.cuh>
#include <storage.cuh>

class Minist {
 public:
  void train(std::string train_data_path, std::string train_label_path,
             float learing_rate, float l2, int batch_size, int epochs);
  void test(std::string test_data_path, std::string test_label_path,
            int batch_size);

 private:
  // Conv1_5x5     1 * 16
  // MaxPool_2x2
  // Conv2_5x5     16 * 32
  // MaxPool_2x2
  // Conv3_3x3     32 * 64
  // FC1           (64 *  2 * 2) * 128
  // FC2           128 * 10
  // SoftMax
  void init_network();
  void network_forward();
  void network_backward();

  std::unordered_map<std::string, std::shared_ptr<Storage>>
      weights;  // Layer weights
  std::unordered_map<std::string, std::shared_ptr<Storage>>
      parameters;  // Layer parameters
  std::unordered_map<std::string, std::shared_ptr<Storage>>
      outputs;  // Layer outputs
  std::unordered_map<std::string, std::shared_ptr<Storage>>
      grads;  // Layer grads and Weights grads
  std::unordered_map<std::string, std::shared_ptr<Storage>>
      square_grads;  // for RMSProp
};