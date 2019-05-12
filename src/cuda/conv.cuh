#pragma once

#include <blas.cuh>
#include <layer.cuh>
#include <unordered_map>

#ifdef DEBUG

// High Performance Convolutional Neural Networks for Document Processing
// https://hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf

void im2col(const float *data_im, const int batch_size, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, float *data_col);
void col2im(const float *data_col, const int batch_size, const int channels,
            const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h, const int pad_w,
            const int stride_h, const int stride_w, float *data_im);

void operator_conv(const Storage *inputs, Storage *filters, Storage *cols,
                   const int pad_h, const int pad_w, const int stride_h,
                   const int stride_w, Storage *output);
void operator_d_conv(
    Storage *outputs_grad, const Storage *inputs, const Storage *cols,
    Storage *filters, const int pad_h, const int pad_w, const int stride_h,
    const int stride_w, Storage *filters_grad, Storage *inputs_grad,
    std::unordered_map<std::string, std::unique_ptr<Storage>> &temp);

void operator_conv_bias(const Storage *inputs, const Storage *bias,
                        Storage *output);
void operator_d_conv_bias(
    const Storage *outputs_grad, Storage *bias_grad,
    std::unordered_map<std::string, std::unique_ptr<Storage>> &temp);

#endif

class Conv : public Layer {
 public:
  explicit Conv(int height, int width, int channel_in, int channel_out,
                int kernel_h, int kernel_w, int pad_h, int pad_w, int stride_h,
                int stride_w, bool is_bias);

  void forward();
  void backward();
  std::vector<std::pair<Storage *, Storage *>> parameters();

 private:
  std::unique_ptr<Storage> filters;
  std::unique_ptr<Storage> filters_grad;
  std::unique_ptr<Storage> bias;
  std::unique_ptr<Storage> bias_grad;
  std::unique_ptr<Storage> cols;

  std::unordered_map<std::string, std::unique_ptr<Storage>> temp;

  int height;
  int width;
  int channel_in;
  int channel_out;
  int kernel_h;
  int kernel_w;
  int pad_w;
  int pad_h;
  int stride_w;
  int stride_h;
  bool is_bias;
};
