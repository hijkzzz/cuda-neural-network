#pragma once

#include <blas.cuh>

Storage *operator_conv(const Storage *inputs, const Storage *filters);

Storage *operator_d_conv(const Storage *outputs_grad, const Storage *filters);

// High Performance Convolutional Neural Networks for Document Processing 
// https://hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf

void im2col(const float *data_im, const unsigned int channels,
            const unsigned int height, const unsigned int width,
            const unsigned int kernel_h, const unsigned int kernel_w,
            const unsigned int pad_h, const unsigned int pad_w,
            const unsigned int stride_h, const unsigned int stride_w,
            float *data_col);

void col2im(const float *data_col, const unsigned int channels,
            const unsigned int height, const unsigned int width,
            const unsigned int kernel_h, const unsigned int kernel_w,
            const unsigned int pad_h, const unsigned int pad_w,
            const unsigned int stride_h, const unsigned int stride_w,
            float *data_im) {