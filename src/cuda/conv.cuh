#pragma once

#include <blas.cuh>

Storage *operator_conv(const Storage *inputs, const Storage *filters);

Storage *operator_d_conv(const Storage *outputs_grad, const Storage *filters);

void im2col(const float *data_im, const std::size_t channels,
            const std::size_t height, const std::size_t width,
            const std::size_t kernel_h, const std::size_t kernel_w,
            const std::size_t pad_h, const std::size_t pad_w,
            const std::size_t stride_h, const std::size_t stride_w,
            float *data_col);

void col2im(const float *data_col, const std::size_t channels,
            const std::size_t height, const std::size_t width,
            const std::size_t kernel_h, const std::size_t kernel_w,
            const std::size_t pad_h, const std::size_t pad_w,
            const std::size_t stride_h, const std::size_t stride_w,
            float *data_im) {