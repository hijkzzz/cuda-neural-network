#pragma once

#include <blas.cuh>

Storage* operator_max_pool(const Storage* inputs, Storage* mask, int kernel_h,
                           int kernel_w, int pad_h, int pad_w, int stride_h,
                           int stride_w);

Storage* operator_d_max_pool(const Storage* output_grads, const Storage* inputs,
                             const Storage* mask, int kernel_h, int kernel_w,
                             int pad_h, int pad_w, int stride_h, int stride_w);