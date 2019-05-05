#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>

#include <blas.cuh>
#include <conv.cuh>
#include <dataset.cuh>
#include <linear.cuh>
#include <max_pool.cuh>
#include <nll_loss.cuh>
#include <relu.cuh>
#include <rmsprop.cuh>
#include <softmax.cuh>
#include <storage.cuh>

class Minist {
 public:
  Minist(DataSet* dataset);

  void train(float learing_rate, float l2, int batch_size, int epochs,
             float beta);
  void test(int batch_size);

 private:
  // Conv1_5x5     1 * 16
  // MaxPool1_2x2
  // Conv2_5x5     16 * 32
  // MaxPool2_2x2
  // Conv3_3x3     32 * 64
  // FC1           (64 *  3 * 3) * 128
  // FC2           128 * 10
  // SoftMax

  void init_weights();
  void update_weights(float learing_rate, float l2, float beta);

  void network_forward(const Storage* images, const Storage* labels);
  void network_backward(const Storage* images, const Storage* labels);

  int correct_count(
      const std::vector<float>& predict_probs, int label_stride,
      const std::vector<unsigned char>& ground_truth);  // return correct count

  std::unordered_map<std::string, std::shared_ptr<Storage>>
      weights;  // Layer weights
  std::unordered_map<std::string, std::shared_ptr<Storage>>
      outputs;  // Layer outputs
  std::unordered_map<std::string, std::shared_ptr<Storage>>
      grads;  // Layer grads and Weight grads
  std::unordered_map<std::string, std::shared_ptr<Storage>>
      square_grads;  // for RMSProp

  DataSet* dataset;
};
