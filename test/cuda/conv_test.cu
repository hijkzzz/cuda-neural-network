#include <test_tools.h>
#include <conv.cuh>

#include <gtest/gtest.h>
#include <thrust/copy.h>

#include <iostream>

TEST(ConvTest, im2col) {
  int channel_in = 2;
  int width = 5;
  int height = 5;

  int kernel_w = 3;
  int kernel_h = 3;

  int pad_h = 1;
  int pad_w = 1;

  int stride_h = 1;
  int stride_w = 1;

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  Storage im({2, 5, 5}, {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                         39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49});
  std::vector<int> shape{channel_in * kernel_h * kernel_w,
                         height_col * width_col};
  Storage col(shape, 0);
  const float *im_ptr = thrust::raw_pointer_cast(im.data.data());
  float *col_ptr = thrust::raw_pointer_cast(col.data.data());

  im2col(im_ptr, channel_in, height, width, kernel_h, kernel_w, pad_h, pad_w,
         stride_h, stride_w, col_ptr);
  device_vector_cout(col.data);
}

TEST(ConvTest, col2im) {
  int channel_in = 2;
  int width = 5;
  int height = 5;

  int kernel_w = 3;
  int kernel_h = 3;

  int pad_h = 1;
  int pad_w = 1;

  int stride_h = 1;
  int stride_w = 1;

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  Storage im({2, 5, 5}, 0);
  Storage col(
      {channel_in * kernel_h * kernel_w, height_col * width_col},
      {0,  0,  0,  0,  0,  0,  0,  1,  2,  3,  0,  5,  6,  7,  8,  0,  10, 11,
       12, 13, 0,  15, 16, 17, 18, 0,  0,  0,  0,  0,  0,  1,  2,  3,  4,  5,
       6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 0,  0,  0,  0,
       0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0,  11, 12, 13, 14, 0,  16, 17,
       18, 19, 0,  0,  0,  1,  2,  3,  0,  5,  6,  7,  8,  0,  10, 11, 12, 13,
       0,  15, 16, 17, 18, 0,  20, 21, 22, 23, 0,  1,  2,  3,  4,  5,  6,  7,
       8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1,
       2,  3,  4,  0,  6,  7,  8,  9,  0,  11, 12, 13, 14, 0,  16, 17, 18, 19,
       0,  21, 22, 23, 24, 0,  0,  5,  6,  7,  8,  0,  10, 11, 12, 13, 0,  15,
       16, 17, 18, 0,  20, 21, 22, 23, 0,  0,  0,  0,  0,  5,  6,  7,  8,  9,
       10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0,  0,  0,
       0,  0,  6,  7,  8,  9,  0,  11, 12, 13, 14, 0,  16, 17, 18, 19, 0,  21,
       22, 23, 24, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  25, 26, 27,
       28, 0,  30, 31, 32, 33, 0,  35, 36, 37, 38, 0,  40, 41, 42, 43, 0,  0,
       0,  0,  0,  25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
       40, 41, 42, 43, 44, 0,  0,  0,  0,  0,  26, 27, 28, 29, 0,  31, 32, 33,
       34, 0,  36, 37, 38, 39, 0,  41, 42, 43, 44, 0,  0,  25, 26, 27, 28, 0,
       30, 31, 32, 33, 0,  35, 36, 37, 38, 0,  40, 41, 42, 43, 0,  45, 46, 47,
       48, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
       42, 43, 44, 45, 46, 47, 48, 49, 26, 27, 28, 29, 0,  31, 32, 33, 34, 0,
       36, 37, 38, 39, 0,  41, 42, 43, 44, 0,  46, 47, 48, 49, 0,  0,  30, 31,
       32, 33, 0,  35, 36, 37, 38, 0,  40, 41, 42, 43, 0,  45, 46, 47, 48, 0,
       0,  0,  0,  0,  30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
       44, 45, 46, 47, 48, 49, 0,  0,  0,  0,  0,  31, 32, 33, 34, 0,  36, 37,
       38, 39, 0,  41, 42, 43, 44, 0,  46, 47, 48, 49, 0,  0,  0,  0,  0,  0});
  float *im_ptr = thrust::raw_pointer_cast(im.data.data());
  const float *col_ptr = thrust::raw_pointer_cast(col.data.data());

  col2im(col_ptr, channel_in, height, width, kernel_h, kernel_w, pad_h, pad_w,
         stride_h, stride_w, im_ptr);
  device_vector_cout(im.data);
}

TEST(ConvTest, ConvForward) {
  int batch_size = 2;
  int channel_in = 2;
  int width = 5;
  int height = 5;

  int channel_out = 3;
  int kernel_w = 3;
  int kernel_h = 3;

  int pad_h = 1;
  int pad_w = 1;

  int stride_h = 1;
  int stride_w = 1;

  Storage input(
      {batch_size, channel_in, height, width},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 0,  1,  2,  3,  4,  5,  6,  7,  8,
       9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0,
       1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
       10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  Storage filter({channel_out, channel_in, kernel_h, kernel_w},
                 {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8});

  Storage cols({1, 1, 1});

  std::unique_ptr<Storage> output(
      operator_conv(&input, &filter, &cols, pad_h, pad_w, stride_h, stride_h));
  ASSERT_TRUE(device_vector_equals_vector(
      output->shape, {batch_size, channel_out, height, width}));
  device_vector_cout(output->data);
  // test with scipy.signal.convolve2d(input, np.rot90(np.rot90(filter)),
  // "same")

  ASSERT_TRUE(device_vector_equals_vector(
      cols.shape,
      {batch_size, channel_in * kernel_h * kernel_w, height * width}));
  device_vector_cout(cols.data);
}

TEST(ConvTest, ConvBackward) {
  int batch_size = 2;
  int channel_in = 2;
  int width = 5;
  int height = 5;

  int channel_out = 3;
  int kernel_w = 3;
  int kernel_h = 3;

  int pad_h = 0;
  int pad_w = 0;

  int stride_h = 1;
  int stride_w = 1;

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  Storage input(
      {batch_size, channel_in, height, width},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 0,  1,  2,  3,  4,  5,  6,  7,  8,
       9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0,
       1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
       10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  Storage output_grad({batch_size, channel_out, height_col, width_col},
                      {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                       0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8});

  Storage filter({channel_out, channel_in, kernel_h, kernel_w},
                 {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8});

  // im2col
  Storage cols(
      {batch_size, channel_in * kernel_h * kernel_w, height_col, width_col});
  const float *im_ptr = thrust::raw_pointer_cast(input.data.data());
  float *col_ptr = thrust::raw_pointer_cast(cols.data.data());

  im2col(im_ptr, channel_in, height, width, kernel_h, kernel_w, pad_h, pad_w,
         stride_h, stride_w, col_ptr);
  device_vector_cout(cols.data);

  // backward
  Storage filters_grad({1, 1, 1});
  std::unique_ptr<Storage> input_grad(
      operator_d_conv(&output_grad, &input, &cols, &filter, pad_h, pad_w,
                      stride_h, stride_w, &filters_grad));

  ASSERT_TRUE(device_vector_equals_vector(
      input_grad->shape, {batch_size, channel_in, height, width}));
  device_vector_cout(input_grad->data);
  // Y = conv_2d(X, W)
  // dL/dX = conv_2d(dL/dY, rot180(W), "full")
  // test with scipy.signal.convolve2d(output_grad, filter, "full")

  ASSERT_TRUE(device_vector_equals_vector(
      filters_grad.shape,
      {batch_size, channel_out, channel_in, kernel_h, kernel_w}));
  device_vector_cout(filters_grad.data);
  // dL/dW = conv_2d(X, dL/dY, "valid")
  // test with scipy.signal.convolve2d(input, np.rot90(np.rot90(output_grad),
  // "valid")
}

TEST(ConvTest, ConvBiasForward) {
  Storage input({1, 2, 3, 3},
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  Storage bias({1, 1, 2}, {1, 2});

  std::unique_ptr<Storage> result(operator_conv_bias(&input, &bias));
  ASSERT_TRUE(device_vector_equals_vector(
      result->data,
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
}

TEST(ConvTest, ConvBiasBackward) {
  Storage output_grad(
      {2, 2, 3, 3},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  Storage bias_grad({2, 3, 2});

  std::unique_ptr<Storage> result(
      operator_d_conv_bias(&output_grad, &bias_grad));

  ASSERT_TRUE(device_vector_equals_vector(bias_grad.shape, {2, 2}));
  ASSERT_TRUE(device_vector_equals_vector(bias_grad.data, {36, 117, 36, 117}));
}
