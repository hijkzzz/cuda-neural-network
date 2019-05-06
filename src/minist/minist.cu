#include <minist.cuh>

Minist::Minist(DataSet* dataset) : dataset(dataset) {
  this->init_weights(0.01);
}

void Minist::train(float learing_rate, float l2, int batch_size, int epochs,
                   float beta) {
  for (int epoch = 0; epoch < epochs; epoch++) {
    int idx = 0;

    // get data
    this->dataset->reset();
    auto train_data = std::move(this->dataset->get_train_data(batch_size));
    std::vector<std::vector<float>>* images = &train_data.first;
    std::vector<unsigned char>* labels = &train_data.second;

    while (images->size() > 0) {
      // prepare ont-hot data
      int size = images->size();

      std::unique_ptr<Storage> images_tensor(new Storage(
          {size, 1, dataset->get_height(), dataset->get_width()}, 0));
      std::unique_ptr<Storage> labels_tensor(
          new Storage(std::vector<int>{size, 10}, 0));

      int image_stride = 1 * dataset->get_height() * dataset->get_width();
      int label_stride = 10;
      for (int i = 0; i < size; i++) {
        thrust::copy((*images)[i].begin(), (*images)[i].end(),
                     images_tensor->data.begin() + i * image_stride);
        int index = i * label_stride + (*labels)[i];
        labels_tensor->data[index] = 1;
      }

      // forward
      this->network_forward(images_tensor.get(), labels_tensor.get());

      // print nll loss and accuracy
      std::vector<float> predict_probs =
          this->outputs["LogSoftMax"]->get_data();
      int corr_count =
          this->correct_count(predict_probs, label_stride, *labels);
      float accuracy = (float)corr_count / size;
      float nll_loss = this->outputs["NLLLoss"]->get_data()[0];

      std::cout << "Epoch: " << epoch << ", Batch: " << idx
                << ", NLLLoss: " << nll_loss << ", Train Accuracy: " << accuracy
                << std::endl;

      // backward & update
      this->network_backward(images_tensor.get(), labels_tensor.get());
      this->update_weights(learing_rate, l2, beta);

      //// clear outputs/grads
      stl_clear_object(&this->outputs);
      stl_clear_object(&this->grads);

      // get data
      train_data = std::move(this->dataset->get_train_data(batch_size));
      images = &train_data.first;
      labels = &train_data.second;
      idx++;
    }
  }
}

void Minist::test(int batch_size) {
  int idx = 0;
  int total = 0;
  int corr_count = 0;

  // get data
  auto test_data = std::move(this->dataset->get_test_data(batch_size));
  std::vector<std::vector<float>>* images = &test_data.first;
  std::vector<unsigned char>* labels = &test_data.second;

  while (images->size() > 0) {
    // prepare ont-hot data
    int size = images->size();

    std::unique_ptr<Storage> images_tensor(
        new Storage({size, 1, dataset->get_height(), dataset->get_width()}, 0));
    std::unique_ptr<Storage> labels_tensor(
        new Storage(std::vector<int>{size, 10}, 0));

    int image_stride = 1 * dataset->get_height() * dataset->get_width();
    int label_stride = 10;
    for (int i = 0; i < size; i++) {
      thrust::copy((*images)[i].begin(), (*images)[i].end(),
                   images_tensor->data.begin() + i * image_stride);
      int index = i * label_stride + (*labels)[i];
      labels_tensor->data[index] = 1;
    }

    // forward
    this->network_forward(images_tensor.get(), labels_tensor.get());

    // corr_count
    std::vector<float> predict_probs = this->outputs["LogSoftMax"]->get_data();
    int temp = this->correct_count(predict_probs, label_stride, *labels);

    // print test accuracy
    std::cout << "Batch: " << idx
              << ", Batch Accuracy: " << ((float)temp / size) << std::endl;
    corr_count += temp;
    total += size;

    // clear outputs
    stl_clear_object(&this->outputs);

    // get data
    test_data = std::move(this->dataset->get_test_data(batch_size));
    images = &test_data.first;
    labels = &test_data.second;
  }

  // print test accuracy
  std::cout << "Total Accuracy: " << ((float)corr_count / total) << std::endl;
}

void Minist::init_weights(float rms_default) {
  // Conv1_5x5     1 * 16
  this->weights["Conv1_5x5_filters"] =
      std::unique_ptr<Storage>(new Storage({16, 1, 5, 5}));
  this->weights["Conv1_5x5_bias"] =
      std::unique_ptr<Storage>(new Storage(std::vector<int>{1, 16}));
  this->weights["Conv1_5x5_filters"]->xavier(1 * 28 * 28, 16 * 24 * 24);
  this->weights["Conv1_5x5_bias"]->xavier(1 * 28 * 28, 16 * 24 * 24);

  this->square_grads["Conv1_5x5_filters"] =
      std::unique_ptr<Storage>(new Storage({16, 1, 5, 5}, rms_default));
  this->square_grads["Conv1_5x5_bias"] = std::unique_ptr<Storage>(
      new Storage(std::vector<int>{1, 16}, rms_default));

  // Conv2_5x5     16 * 32
  this->weights["Conv2_5x5_filters"] =
      std::unique_ptr<Storage>(new Storage({32, 16, 5, 5}));
  this->weights["Conv2_5x5_bias"] =
      std::unique_ptr<Storage>(new Storage(std::vector<int>{1, 32}));
  this->weights["Conv2_5x5_filters"]->xavier(16 * 12 * 12, 32 * 8 * 8);
  this->weights["Conv2_5x5_bias"]->xavier(16 * 12 * 12, 32 * 8 * 8);

  this->square_grads["Conv2_5x5_filters"] =
      std::unique_ptr<Storage>(new Storage({32, 16, 5, 5}, rms_default));
  this->square_grads["Conv2_5x5_bias"] = std::unique_ptr<Storage>(
      new Storage(std::vector<int>{1, 32}, rms_default));

  // Conv3_3x3     32 * 64
  this->weights["Conv3_3x3_filters"] =
      std::unique_ptr<Storage>(new Storage({64, 32, 3, 3}));
  this->weights["Conv3_3x3_bias"] =
      std::unique_ptr<Storage>(new Storage(std::vector<int>{1, 64}));
  this->weights["Conv3_3x3_filters"]->xavier(32 * 4 * 4, 64 * 2 * 2);
  this->weights["Conv3_3x3_bias"]->xavier(32 * 4 * 4, 64 * 2 * 2);

  this->square_grads["Conv3_3x3_filters"] =
      std::unique_ptr<Storage>(new Storage({64, 32, 3, 3}, rms_default));
  this->square_grads["Conv3_3x3_bias"] = std::unique_ptr<Storage>(
      new Storage(std::vector<int>{1, 64}, rms_default));

  // FC1           (64 *  3 * 3) * 128
  this->weights["FC1_weights"] =
      std::unique_ptr<Storage>(new Storage(std::vector<int>{64 * 2 * 2, 128}));
  this->weights["FC1_bias"] =
      std::unique_ptr<Storage>(new Storage(std::vector<int>{1, 128}));
  this->weights["FC1_weights"]->xavier(64 * 2 * 2, 128);
  this->weights["FC1_bias"]->xavier(64 * 2 * 2, 128);

  this->square_grads["FC1_weights"] = std::unique_ptr<Storage>(
      new Storage(std::vector<int>{64 * 2 * 2, 128}, rms_default));
  this->square_grads["FC1_bias"] = std::unique_ptr<Storage>(
      new Storage(std::vector<int>{1, 128}, rms_default));

  // FC2           128 * 10
  this->weights["FC2_weights"] =
      std::unique_ptr<Storage>(new Storage(std::vector<int>{128, 10}));
  this->weights["FC2_bias"] =
      std::unique_ptr<Storage>(new Storage(std::vector<int>{1, 10}));
  this->weights["FC2_weights"]->xavier(128, 10);
  this->weights["FC2_bias"]->xavier(128, 10);

  this->square_grads["FC2_weights"] = std::unique_ptr<Storage>(
      new Storage(std::vector<int>{128, 10}, rms_default));
  this->square_grads["FC2_bias"] = std::unique_ptr<Storage>(
      new Storage(std::vector<int>{1, 10}, rms_default));
}

void Minist::update_weights(float learing_rate, float l2, float beta) {
  for (auto iter = this->weights.begin(); iter != this->weights.end(); iter++) {
    std::string weights_name = iter->first;

    std::unique_ptr<Storage>& weights_ptr = iter->second;
    std::unique_ptr<Storage>& square_grad_ptr =
        this->square_grads[weights_name];
    std::unique_ptr<Storage>& grad_ptr = this->grads[weights_name];

    rmsprop_update(square_grad_ptr.get(), weights_ptr.get(), grad_ptr.get(),
                   learing_rate, l2, beta);
  }
}

void Minist::network_forward(const Storage* images, const Storage* labels) {
  // Conv1_5x5     1 * 16
  this->outputs["Conv1_5x5_col"] =
      std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->outputs["Conv1_5x5"] = std::unique_ptr<Storage>(
      operator_conv(images, this->weights["Conv1_5x5_filters"].get(),
                    this->outputs["Conv1_5x5_col"].get(), 0, 0, 1, 1));
  this->outputs["Conv1_5x5_bias"] = std::unique_ptr<Storage>(operator_bias(
      this->outputs["Conv1_5x5"].get(), this->weights["Conv1_5x5_bias"].get()));
  this->outputs["Conv1_5x5_bias_relu"] = std::unique_ptr<Storage>(
      operator_relu(this->outputs["Conv1_5x5_bias"].get()));

  // MaxPool1_2x2
  this->outputs["MaxPool1_2x2_mask"] =
      std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->outputs["MaxPool1_2x2"] = std::unique_ptr<Storage>(operator_max_pool(
      this->outputs["Conv1_5x5_bias_relu"].get(),
      this->outputs["MaxPool1_2x2_mask"].get(), 2, 2, 0, 0, 2, 2));

  // Conv2_5x5     16 * 32
  this->outputs["Conv2_5x5_col"] =
      std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->outputs["Conv2_5x5"] = std::unique_ptr<Storage>(
      operator_conv(this->outputs["MaxPool1_2x2"].get(),
                    this->weights["Conv2_5x5_filters"].get(),
                    this->outputs["Conv2_5x5_col"].get(), 0, 0, 1, 1));
  this->outputs["Conv2_5x5_bias"] = std::unique_ptr<Storage>(operator_bias(
      this->outputs["Conv2_5x5"].get(), this->weights["Conv2_5x5_bias"].get()));
  this->outputs["Conv2_5x5_bias_relu"] = std::unique_ptr<Storage>(
      operator_relu(this->outputs["Conv2_5x5_bias"].get()));

  // MaxPool2_2x2
  this->outputs["MaxPool2_2x2_mask"] =
      std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->outputs["MaxPool2_2x2"] = std::unique_ptr<Storage>(operator_max_pool(
      this->outputs["Conv2_5x5_bias_relu"].get(),
      this->outputs["MaxPool2_2x2_mask"].get(), 2, 2, 0, 0, 2, 2));

  // Conv3_3x3     32 * 64
  this->outputs["Conv3_3x3_col"] =
      std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->outputs["Conv3_3x3"] = std::unique_ptr<Storage>(
      operator_conv(this->outputs["MaxPool2_2x2"].get(),
                    this->weights["Conv3_3x3_filters"].get(),
                    this->outputs["Conv3_3x3_col"].get(), 0, 0, 1, 1));
  this->outputs["Conv3_3x3_bias"] = std::unique_ptr<Storage>(operator_bias(
      this->outputs["Conv3_3x3"].get(), this->weights["Conv3_3x3_bias"].get()));
  this->outputs["Conv3_3x3_bias_relu"] = std::unique_ptr<Storage>(
      operator_relu(this->outputs["Conv3_3x3_bias"].get()));

  // Reshape
  int batch_size = *(this->outputs["Conv3_3x3_bias_relu"]->shape.begin());
  this->outputs["Conv3_3x3_bias_relu"]->reshape({batch_size, 64 * 2 * 2});

  // FC1           (64 *  2 * 2) * 128
  this->outputs["FC1"] = std::unique_ptr<Storage>(
      operator_linear(this->outputs["Conv3_3x3_bias_relu"].get(),
                      this->weights["FC1_weights"].get()));
  this->outputs["FC1_bias"] = std::unique_ptr<Storage>(operator_bias(
      this->outputs["FC1"].get(), this->weights["FC1_bias"].get()));
  this->outputs["FC1_bias_relu"] =
      std::unique_ptr<Storage>(operator_relu(this->outputs["FC1_bias"].get()));

  // FC2           128 * 10
  this->outputs["FC2"] = std::unique_ptr<Storage>(
      operator_linear(this->outputs["FC1_bias_relu"].get(),
                      this->weights["FC2_weights"].get()));
  this->outputs["FC2_bias"] = std::unique_ptr<Storage>(operator_bias(
      this->outputs["FC2"].get(), this->weights["FC2_bias"].get()));
  this->outputs["FC2_bias_relu"] =
      std::unique_ptr<Storage>(operator_relu(this->outputs["FC2_bias"].get()));

  // LogSoftMax
  this->outputs["LogSoftMax"] = std::unique_ptr<Storage>(
      operator_log_softmax(this->outputs["FC2_bias_relu"].get(), 1));

  // NLLLoss
  this->outputs["NLLLoss"] = std::unique_ptr<Storage>(
      operator_nll_loss(this->outputs["LogSoftMax"].get(), labels));
}

void Minist::network_backward(const Storage* images, const Storage* labels) {
  // NLLLoss
  this->grads["NLLLoss"] =
      std::unique_ptr<Storage>(operator_d_nll_loss(labels));

  // LogSoftMax
  this->grads["LogSoftMax"] = std::unique_ptr<Storage>(operator_d_log_softmax(
      this->grads["NLLLoss"].get(), this->outputs["FC2_bias_relu"].get(), 1));

  // FC2           128 * 10
  this->grads["FC2_bias_relu"] = std::unique_ptr<Storage>(operator_d_relu(
      this->grads["LogSoftMax"].get(), this->outputs["FC2_bias"].get()));

  this->grads["FC2_bias"] = std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["FC2_bias_"] = std::unique_ptr<Storage>(operator_d_bias(
      this->grads["FC2_bias_relu"].get(), this->grads["FC2_bias"].get()));

  this->grads["FC2_weights"] = std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["FC2"] = std::unique_ptr<Storage>(operator_d_linear(
      this->grads["FC2_bias_"].get(), this->outputs["FC1_bias_relu"].get(),
      this->weights["FC2_weights"].get(), this->grads["FC2_weights"].get()));

  // FC1          (64 *  2 * 2) * 128
  this->grads["FC1_bias_relu"] = std::unique_ptr<Storage>(operator_d_relu(
      this->grads["FC2"].get(), this->outputs["FC1_bias"].get()));

  this->grads["FC1_bias"] = std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["FC1_bias_"] = std::unique_ptr<Storage>(operator_d_bias(
      this->grads["FC1_bias_relu"].get(), this->grads["FC1_bias"].get()));

  this->grads["FC1_weights"] = std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["FC1"] = std::unique_ptr<Storage>(operator_d_linear(
      this->grads["FC1_bias_"].get(),
      this->outputs["Conv3_3x3_bias_relu"].get(),
      this->weights["FC1_weights"].get(), this->grads["FC1_weights"].get()));

  // Reshape
  std::vector<int> top_shape(this->outputs["Conv3_3x3"]->shape.begin(),
                             this->outputs["Conv3_3x3"]->shape.end());
  this->grads["FC1"]->reshape(top_shape);

  // Conv3_3x3     32 * 64
  this->grads["Conv3_3x3_bias_relu"] = std::unique_ptr<Storage>(operator_d_relu(
      this->grads["FC1"].get(), this->outputs["Conv3_3x3_bias"].get()));

  this->grads["Conv3_3x3_bias"] =
      std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["Conv3_3x3_bias_"] = std::unique_ptr<Storage>(
      operator_d_conv_bias(this->grads["Conv3_3x3_bias_relu"].get(),
                           this->grads["Conv3_3x3_bias"].get()));

  this->grads["Conv3_3x3_filters"] =
      std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["Conv3_3x3"] = std::unique_ptr<Storage>(operator_d_conv(
      this->grads["Conv3_3x3_bias_"].get(), this->outputs["MaxPool2_2x2"].get(),
      this->outputs["Conv3_3x3_col"].get(),
      this->weights["Conv3_3x3_filters"].get(), 0, 0, 1, 1,
      this->grads["Conv3_3x3_filters"].get()));

  // MaxPool2_2x2
  this->grads["MaxPool2_2x2"] = std::unique_ptr<Storage>(operator_d_max_pool(
      this->grads["Conv3_3x3"].get(),
      this->outputs["Conv2_5x5_bias_relu"].get(),
      this->outputs["MaxPool2_2x2_mask"].get(), 2, 2, 0, 0, 2, 2));

  // Conv2_5x5     16 * 32
  this->grads["Conv2_5x5_bias_relu"] = std::unique_ptr<Storage>(
      operator_d_relu(this->grads["MaxPool2_2x2"].get(),
                      this->outputs["Conv2_5x5_bias"].get()));

  this->grads["Conv2_5x5_bias"] =
      std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["Conv2_5x5_bias_"] = std::unique_ptr<Storage>(
      operator_d_conv_bias(this->grads["Conv2_5x5_bias_relu"].get(),
                           this->grads["Conv2_5x5_bias"].get()));

  this->grads["Conv2_5x5_filters"] =
      std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["Conv2_5x5"] = std::unique_ptr<Storage>(operator_d_conv(
      this->grads["Conv2_5x5_bias_"].get(), this->outputs["MaxPool1_2x2"].get(),
      this->outputs["Conv2_5x5_col"].get(),
      this->weights["Conv2_5x5_filters"].get(), 0, 0, 1, 1,
      this->grads["Conv2_5x5_filters"].get()));

  // MaxPool1_2x2
  this->grads["MaxPool1_2x2"] = std::unique_ptr<Storage>(operator_d_max_pool(
      this->grads["Conv2_5x5"].get(),
      this->outputs["Conv1_5x5_bias_relu"].get(),
      this->outputs["MaxPool1_2x2_mask"].get(), 2, 2, 0, 0, 2, 2));

  // Conv1_5x5     1 * 16
  this->grads["Conv1_5x5_bias_relu"] = std::unique_ptr<Storage>(
      operator_d_relu(this->grads["MaxPool1_2x2"].get(),
                      this->outputs["Conv1_5x5_bias"].get()));

  this->grads["Conv1_5x5_bias"] =
      std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["Conv1_5x5_bias_"] = std::unique_ptr<Storage>(
      operator_d_conv_bias(this->grads["Conv1_5x5_bias_relu"].get(),
                           this->grads["Conv1_5x5_bias"].get()));

  this->grads["Conv1_5x5_filters"] =
      std::unique_ptr<Storage>(new Storage({1, 1, 1}));
  this->grads["Conv1_5x5"] = std::unique_ptr<Storage>(
      operator_d_conv(this->grads["Conv1_5x5_bias_"].get(), images,
                      this->outputs["Conv1_5x5_col"].get(),
                      this->weights["Conv1_5x5_filters"].get(), 0, 0, 1, 1,
                      this->grads["Conv1_5x5_filters"].get()));
}

int Minist::correct_count(const std::vector<float>& predict_probs,
                          int label_stride,
                          const std::vector<unsigned char>& ground_truth) {
  int count = 0;
  for (int i = 0; i < ground_truth.size(); i++) {
    int max_pos = -1;
    float max_value = -FLT_MAX;
    for (int j = 0; j < label_stride; j++) {
      int index = i * label_stride + j;
      if (predict_probs[index] > max_value) {
        max_value = predict_probs[index];
        max_pos = j;
      }
    }

    if (max_pos == ground_truth[i]) ++count;
  }
  return count;
}
