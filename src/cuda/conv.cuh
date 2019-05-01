#pragma once

#include <blas.cuh>

Storage *operator_conv(const Storage *inputs, const Storage *filters,
                       const int pad_h, const int pad_w, const int stride_h,
                       const int stride_w);

Storage *operator_d_conv(const Storage *outputs_grad, const Storage *inputs,
                         const Storage *cols, const Storage *filters,
                         const int pad_h, const int pad_w, const int stride_h,
                         const int stride_w, Storage *filters_grad);

// High Performance Convolutional Neural Networks for Document Processing
// https://hal.inria.fr/file/index/docid/112631/filename/p1038112283956.pdf

void im2col(const float *data_im, const int channels, const int height,
            const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, float *data_col);

void col2im(const float *data_col, const int channels, const int height,
            const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, float *data_im);