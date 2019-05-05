#include <minist.cuh>

Minist::Minist(DataSet* dataset) : dataset(dataset) {}

void Minist::train(float learing_rate, float l2, int batch_size, int epochs) {}

void Minist::test(int batch_size) {}

void Minist::init_weights() {
  // Conv1_5x5     1 * 16
  this->weights["Conv1_5x5_filters"] =
      std::shared_ptr<Storage>(new Storage({16, 1, 5, 5}));
  this->weights["Conv1_5x5_bias"] =
      std::shared_ptr<Storage>(new Storage(std::vector<int>{1, 16}));
  this->weights["Conv1_5x5_filters"]->xavier(1 * 28 * 28, 16 * 24 * 24);
  this->weights["Conv1_5x5_bias"]->xavier(1 * 28 * 28, 16 * 24 * 24);

  // Conv2_5x5     16 * 32
  this->weights["Conv2_5x5_filters"] =
      std::shared_ptr<Storage>(new Storage({32, 16, 5, 5}));
  this->weights["Conv2_5x5_bias"] =
      std::shared_ptr<Storage>(new Storage(std::vector<int>{1, 32}));
  this->weights["Conv2_5x5_filters"]->xavier(16 * 14 * 14, 32 * 10 * 10);
  this->weights["Conv2_5x5_bias"]->xavier(16 * 14 * 14, 32 * 10 * 10);

  // Conv3_3x3     32 * 64
  this->weights["Conv3_3x3_filters"] =
      std::shared_ptr<Storage>(new Storage({64, 32, 3, 3}));
  this->weights["Conv3_3x3_bias"] =
      std::shared_ptr<Storage>(new Storage(std::vector<int>{1, 64}));
  this->weights["Conv3_3x3_filters"]->xavier(32 * 5 * 5, 64 * 3 * 3);
  this->weights["Conv3_3x3_bias"]->xavier(32 * 5 * 5, 64 * 3 * 3);

  // FC1           (64 *  2 * 2) * 128
  this->weights["FC1_weights"] =
      std::shared_ptr<Storage>(new Storage(std::vector<int>{64 * 3 * 3, 128}));
  this->weights["FC1_bias"] =
      std::shared_ptr<Storage>(new Storage(std::vector<int>{1, 128}));
  this->weights["FC1_weights"]->xavier(64 * 3 * 3, 128);
  this->weights["FC1_bias"]->xavier(64 * 3 * 3, 128);

  // FC2           128 * 10
  this->weights["FC2_weights"] =
      std::shared_ptr<Storage>(new Storage(std::vector<int>{128, 10}));
  this->weights["FC2_bias"] =
      std::shared_ptr<Storage>(new Storage(std::vector<int>{1, 10}));
  this->weights["FC2_weights"]->xavier(128, 10);
  this->weights["FC2_bias"]->xavier(128, 10);
}

void Minist::network_forward(const Storage* images, const Storage* labels) {
  // Conv1_5x5     1 * 16
  this->outputs["Conv1_5x5_col"] =
      std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->outputs["Conv1_5x5"] = std::shared_ptr<Storage>(
      operator_conv(images, this->weights["Conv1_5x5_filters"].get(),
                    this->outputs["Conv1_5x5_col"].get(), 0, 0, 1, 1));
  this->outputs["Conv1_5x5_bias"] = std::shared_ptr<Storage>(operator_bias(
      this->outputs["Conv1_5x5"].get(), this->weights["Conv1_5x5_bias"].get()));
  this->outputs["Conv1_5x5_relu"] = std::shared_ptr<Storage>(
      operator_relu(this->outputs["Conv1_5x5_bias"].get()));

  // MaxPool1_2x2
  this->outputs["MaxPool1_2x2_mask"] =
      std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->outputs["MaxPool1_2x2"] = std::shared_ptr<Storage>(operator_max_pool(
      this->outputs["Conv1_5x5_relu"].get(),
      this->outputs["MaxPool1_2x2_mask"].get(), 2, 2, 0, 0, 2, 2));

  // Conv2_5x5     16 * 32
  this->outputs["Conv2_5x5_col"] =
      std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->outputs["Conv2_5x5"] = std::shared_ptr<Storage>(
      operator_conv(this->outputs["MaxPool1_2x2"].get(),
                    this->weights["Conv2_5x5_filters"].get(),
                    this->outputs["Conv2_5x5_col"].get(), 0, 0, 1, 1));
  this->outputs["Conv2_5x5_bias"] = std::shared_ptr<Storage>(operator_bias(
      this->outputs["Conv2_5x5"].get(), this->weights["Conv2_5x5_bias"].get()));
  this->outputs["Conv2_5x5_relu"] = std::shared_ptr<Storage>(
      operator_relu(this->outputs["Conv2_5x5_bias"].get()));

  // MaxPool2_2x2
  this->outputs["MaxPool2_2x2_mask"] =
      std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->outputs["MaxPool2_2x2"] = std::shared_ptr<Storage>(operator_max_pool(
      this->outputs["Conv2_5x5_relu"].get(),
      this->outputs["MaxPool2_2x2_mask"].get(), 2, 2, 0, 0, 2, 2));

  // Conv3_3x3     32 * 64
  this->outputs["Conv3_3x3_col"] =
      std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->outputs["Conv3_3x3"] = std::shared_ptr<Storage>(
      operator_conv(this->outputs["MaxPool2_2x2"].get(),
                    this->weights["Conv3_3x3_filters"].get(),
                    this->outputs["Conv3_3x3_col"].get(), 0, 0, 1, 1));
  this->outputs["Conv3_3x3_bias"] = std::shared_ptr<Storage>(operator_bias(
      this->outputs["Conv3_3x3"].get(), this->weights["Conv3_3x3_bias"].get()));
  this->outputs["Conv3_3x3_relu"] = std::shared_ptr<Storage>(
      operator_relu(this->outputs["Conv3_3x3_bias"].get()));

  // Reshape
  int batch_size = *this->outputs["Conv3_3x3_relu"]->shape.begin();
  this->outputs["Conv3_3x3_relu"]->reshape({batch_size, 64 * 3 * 3});

  // FC1           (64 *  3 * 3) * 128
  this->outputs["FC1"] = std::shared_ptr<Storage>(
      operator_linear(this->outputs["Conv3_3x3_relu"].get(),
                      this->weights["FC1_weights"].get()));
  this->outputs["FC1_bias"] = std::shared_ptr<Storage>(operator_bias(
      this->outputs["FC1"].get(), this->weights["FC1_bias"].get()));
  this->outputs["FC1_bias_relu"] =
      std::shared_ptr<Storage>(operator_relu(this->outputs["FC1_bias"].get()));

  // FC2           128 * 10
  this->outputs["FC2"] = std::shared_ptr<Storage>(
      operator_linear(this->outputs["FC1_bias_relu"].get(),
                      this->weights["FC2_weights"].get()));
  this->outputs["FC2_bias"] = std::shared_ptr<Storage>(operator_bias(
      this->outputs["FC2"].get(), this->weights["FC2_bias"].get()));
  this->outputs["FC2_bias_relu"] =
      std::shared_ptr<Storage>(operator_relu(this->outputs["FC2_bias"].get()));

  // LogSoftMax
  this->outputs["LogSoftMax"] = std::shared_ptr<Storage>(
      operator_log_softmax(this->outputs["FC2_bias_relu"].get(), 1));

  // NLLLoss
  this->outputs["NLLLoss"] = std::shared_ptr<Storage>(
      operator_nll_loss(this->outputs["LogSoftMax"].get(), labels));
}

void Minist::network_backward(const Storage* images, const Storage* labels) {
  // NLLLoss
  this->grads["NLLLoss"] =
      std::shared_ptr<Storage>(operator_d_nll_loss(labels));

  // LogSoftMax
  this->grads["LogSoftMax"] = std::shared_ptr<Storage>(operator_d_log_softmax(
      this->grads["NLLLoss"].get(), this->outputs["FC2_bias_relu"].get(), 1));

  // FC2           128 * 10
  this->grads["FC2"] = std::shared_ptr<Storage>(operator_d_relu(
      this->grads["LogSoftMax"].get(), this->outputs["FC2_bias"].get()));

  this->grads["FC2_bias"] = std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["FC2"] = std::shared_ptr<Storage>(operator_d_bias(
      this->grads["FC2"].get(), this->weights["FC2_bias"].get()));

  this->grads["FC2_weights"] = std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["FC2"] = std::shared_ptr<Storage>(operator_d_linear(
      this->grads["FC2"].get(), this->outputs["FC1_bias_relu"].get(),
      this->weights["FC2_weights"].get(), this->grads["FC2_weights"].get()));

  // FC1          (64 *  3 * 3) * 128
  this->grads["FC1"] = std::shared_ptr<Storage>(operator_d_relu(
      this->grads["FC2"].get(), this->outputs["FC1_bias"].get()));

  this->grads["FC1_bias"] = std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["FC1"] = std::shared_ptr<Storage>(operator_d_bias(
      this->grads["FC1"].get(), this->weights["FC1_bias"].get()));

  this->grads["FC1_weights"] = std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["FC1"] = std::shared_ptr<Storage>(operator_d_linear(
      this->grads["FC1"].get(), this->outputs["Conv3_3x3_relu"].get(),
      this->weights["FC1_weights"].get(), this->grads["FC1_weights"].get()));

  // Reshape
  std::vector<int> top_shape(this->outputs["Conv3_3x3_relu"]->shape.begin(),
                             this->outputs["Conv3_3x3_relu"]->shape.end());
  this->grads["FC1"]->reshape(top_shape);

  // Conv3_3x3     32 * 64
  this->grads["Conv3_3x3"] = std::shared_ptr<Storage>(operator_d_relu(
      this->grads["FC1"].get(), this->outputs["Conv3_3x3_bias"].get()));

  this->grads["Conv3_3x3_bias"] =
      std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["Conv3_3x3"] = std::shared_ptr<Storage>(operator_d_conv_bias(
      this->grads["Conv3_3x3"].get(), this->grads["Conv3_3x3_bias"].get()));

  this->grads["Conv3_3x3_weights"] =
      std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["Conv3_3x3"] = std::shared_ptr<Storage>(operator_d_conv(
      this->grads["Conv3_3x3"].get(), this->outputs["MaxPool2_2x2"].get(),
      this->outputs["Conv3_3x3_col"].get(),
      this->weights["Conv3_3x3_weights"].get(), 0, 0, 1, 1,
      this->grads["Conv3_3x3_weights"].get()));

  // MaxPool2_2x2
  this->grads["MaxPool2_2x2"] = std::shared_ptr<Storage>(operator_d_max_pool(
      this->grads["Conv3_3x3"].get(), this->outputs["Conv2_5x5_bias"].get(),
      this->outputs["MaxPool2_2x2_mask"].get(), 2, 2, 0, 0, 2, 2));

  // Conv2_5x5     16 * 32
  this->grads["Conv2_5x5"] = std::shared_ptr<Storage>(
      operator_d_relu(this->grads["MaxPool2_2x2"].get(),
                      this->outputs["Conv2_5x5_bias"].get()));

  this->grads["Conv2_5x5_bias"] =
      std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["Conv2_5x5"] = std::shared_ptr<Storage>(operator_d_conv_bias(
      this->grads["Conv2_5x5"].get(), this->grads["Conv2_5x5_bias"].get()));

  this->grads["Conv2_5x5_weights"] =
      std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["Conv2_5x5"] = std::shared_ptr<Storage>(operator_d_conv(
      this->grads["Conv2_5x5"].get(), this->outputs["MaxPool1_2x2"].get(),
      this->outputs["Conv2_5x5_col"].get(),
      this->weights["Conv2_5x5_weights"].get(), 0, 0, 1, 1,
      this->grads["Conv2_5x5_weights"].get()));

  // MaxPool1_2x2
  this->grads["MaxPool1_2x2"] = std::shared_ptr<Storage>(operator_d_max_pool(
      this->grads["Conv2_5x5"].get(), this->outputs["Conv1_5x5_bias"].get(),
      this->outputs["MaxPool1_2x2_mask"].get(), 2, 2, 0, 0, 2, 2));

  // Conv1_5x5     1 * 16
  this->grads["Conv1_5x5"] = std::shared_ptr<Storage>(
    operator_d_relu(this->grads["MaxPool1_2x2"].get(),
                    this->outputs["Conv1_5x5_bias"].get()));

  this->grads["Conv1_5x5_bias"] =
      std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["Conv1_5x5"] = std::shared_ptr<Storage>(operator_d_conv_bias(
      this->grads["Conv1_5x5"].get(), this->grads["Conv1_5x5_bias"].get()));

  this->grads["Conv1_5x5_weights"] =
      std::shared_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["Conv1_5x5"] = std::shared_ptr<Storage>(operator_d_conv(
      this->grads["Conv1_5x5"].get(), images,
      this->outputs["Conv1_5x5_col"].get(),
      this->weights["Conv1_5x5_weights"].get(), 0, 0, 1, 1,
      this->grads["Conv1_5x5_weights"].get()));
}

int Minist::correct_count(const std::vector<std::vector<float>>& predict_probs,
                          const std::vector<int>& ground_truth) {

  return 0;
}
