#include <test_tools.h>
#include <conv.cuh>

#include <gtest/gtest.h>
#include <thrust/copy.h>

#include <iostream>

TEST(ConvTest, im2col) {
  int batch_size = 1;
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

  Storage im(
      {batch_size, 2, 5, 5},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
       34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49});
  Storage col(
      {batch_size, channel_in * kernel_h * kernel_w, height_col * width_col});
  const float *im_ptr = thrust::raw_pointer_cast(im.get_data().data());
  float *col_ptr = thrust::raw_pointer_cast(col.get_data().data());

  im2col(im_ptr, batch_size, channel_in, height, width, kernel_h, kernel_w,
         pad_h, pad_w, stride_h, stride_w, col_ptr);
  std::cout << "im2col" << std::endl;
  device_vector_cout(col.get_data());
}

TEST(ConvTest, col2im) {
  int batch_size = 1;
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

  Storage im({batch_size, 2, 5, 5});
  Storage col(
      {batch_size, channel_in * kernel_h * kernel_w, height_col * width_col},
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
  float *im_ptr = thrust::raw_pointer_cast(im.get_data().data());
  const float *col_ptr = thrust::raw_pointer_cast(col.get_data().data());

  col2im(col_ptr, batch_size, channel_in, height, width, kernel_h, kernel_w,
         pad_h, pad_w, stride_h, stride_w, im_ptr);
  std::cout << "col2im" << std::endl;
  device_vector_cout(im.get_data());
}

TEST(ConvTest, ConvForward) {
  int batch_size = 1;
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

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  Storage input(
      {batch_size, channel_in, height, width},
      {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
       17, 18, 19, 20, 21, 22, 23, 24, 0,  1,  2,  3,  4,  5,  6,  7,  8,
       9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  Storage filter({channel_out, channel_in, kernel_h, kernel_w},
                 {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8});

  Storage cols(
      {batch_size, channel_in * kernel_h * kernel_w, height_col * width_col});

  Storage output({batch_size, channel_out, height_col, width_col});
  operator_conv(&input, &filter, &cols, pad_h, pad_w, stride_h, stride_h,
                &output);

  std::cout << "conv" << std::endl;
  ASSERT_TRUE(device_vector_equals_vector(
      output.get_data(),
      {176,  284,  350,  416,  272,  420,  624, 696, 768, 480, 690,  984,  1056,
       1128, 690,  960,  1344, 1416, 1488, 900, 464, 608, 638, 668,  368,  176,
       284,  350,  416,  272,  420,  624,  696, 768, 480, 690, 984,  1056, 1128,
       690,  960,  1344, 1416, 1488, 900,  464, 608, 638, 668, 368,  176,  284,
       350,  416,  272,  420,  624,  696,  768, 480, 690, 984, 1056, 1128, 690,
       960,  1344, 1416, 1488, 900,  464,  608, 638, 668, 368}));
  // test with scipy.signal.convolve2d(input, np.rot90(np.rot90(filter)),
  // "same")
}

TEST(ConvTest, ConvBackward) {
  int batch_size = 1;
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
       9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});

  Storage output_grad({batch_size, channel_out, height_col, width_col},
                      {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4,
                       5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8});

  Storage filter({channel_out, channel_in, kernel_h, kernel_w},
                 {0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                  0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 1, 2, 3, 4, 5, 6, 7, 8});

  // im2col
  Storage cols(
      {batch_size, channel_in * kernel_h * kernel_w, height_col * width_col});
  const float *im_ptr = thrust::raw_pointer_cast(input.get_data().data());
  float *col_ptr = thrust::raw_pointer_cast(cols.get_data().data());

  im2col(im_ptr, batch_size, channel_in, height, width, kernel_h, kernel_w,
         pad_h, pad_w, stride_h, stride_w, col_ptr);

  // backward
  std::unordered_map<std::string, std::unique_ptr<Storage>> temp;
  Storage input_grad({batch_size, channel_in, height, width});
  Storage filters_grad({channel_out, channel_in, kernel_h, kernel_w});
  operator_d_conv(&output_grad, &input, &cols, &filter, pad_h, pad_w, stride_h,
                  stride_w, &filters_grad, &input_grad, temp);

  std::cout << "conv_d input" << std::endl;
  ASSERT_TRUE(device_vector_equals_vector(
      input_grad.get_data(),
      {0,   0,   3,   12,  12,  0,   18,  60,  78,  60,  27,  108, 252,
       252, 171, 108, 270, 492, 402, 240, 108, 252, 435, 336, 192, 0,
       0,   3,   12,  12,  0,   18,  60,  78,  60,  27,  108, 252, 252,
       171, 108, 270, 492, 402, 240, 108, 252, 435, 336, 192}));
  // Y = conv_2d(X, W)
  // dL/dX = conv_2d(dL/dY, rot180(W), "full")
  // test with scipy.signal.convolve2d(output_grad, filter, "full")

  std::cout << "conv_d filters" << std::endl;
  ASSERT_TRUE(device_vector_equals_vector(
      filters_grad.get_data(),
      {312, 348, 384, 492, 528, 564, 672, 708, 744, 312, 348, 384, 492, 528,
       564, 672, 708, 744, 312, 348, 384, 492, 528, 564, 672, 708, 744, 312,
       348, 384, 492, 528, 564, 672, 708, 744, 312, 348, 384, 492, 528, 564,
       672, 708, 744, 312, 348, 384, 492, 528, 564, 672, 708, 744}));
  // dL/dW = conv_2d(X, dL/dY, "valid")
  // test with scipy.signal.convolve2d(input, np.rot90(np.rot90(output_grad)),
  // "valid")
}

TEST(ConvTest, ConvBiasForward) {
  Storage input({1, 2, 3, 3},
                {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});
  Storage bias({1, 1, 2}, {1, 2});

  Storage result({1, 2, 3, 3});
  operator_conv_bias(&input, &bias, &result);
  ASSERT_TRUE(device_vector_equals_vector(
      result.get_data(),
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19}));
}

TEST(ConvTest, ConvBiasBackward) {
  Storage output_grad(
      {2, 2, 3, 3},
      {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
       0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17});

  Storage bias_grad({1, 2});
  std::unordered_map<std::string, std::unique_ptr<Storage>> temp;
  operator_d_conv_bias(&output_grad, &bias_grad, temp);
  ASSERT_TRUE(device_vector_equals_vector(bias_grad.get_data(), {72, 234}));
}
