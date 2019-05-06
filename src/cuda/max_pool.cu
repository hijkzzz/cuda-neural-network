#include <max_pool.cuh>

__global__ void operator_max_pool_h(
    const int nthreads, const float* const bottom_data, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    float* const top_data, float* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // output location
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;

    // pooled range
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = fminf(hstart + kernel_h, height);
    const int wend = fminf(wstart + kernel_w, width);
    hstart = fmaxf(hstart, 0);
    wstart = fmaxf(wstart, 0);

    // get max value postion
    float maxval = -FLT_MAX;
    int maxidx = -1;
    const float* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    // output
    top_data[index] = maxval;

    // record idx
    mask[index] = maxidx;
  }
}

__global__ void operator_d_max_pool_h(
    const int nthreads, const float* const top_diff, const float* const mask,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, float* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;

    // pooled range
    const int phstart =
        (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = fminf((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
        (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = fminf((w + pad_w) / stride_w + 1, pooled_width);
    float gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const float* const top_diff_slice = top_diff + offset;

    // get max value idx
    const float* const mask_slice = mask + offset;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        if (mask_slice[ph * pooled_width + pw] == h * width + w) {
          gradient += top_diff_slice[ph * pooled_width + pw];
        }
      }
    }

    bottom_diff[index] = gradient;
  }
}

void operator_max_pool(const Storage* inputs, Storage* mask, int kernel_h,
                       int kernel_w, int pad_h, int pad_w, int stride_h,
                       int stride_w, Storage* output) {
  CHECK_EQ(inputs->shape.size(), 4, "operator_max_pool: inputs shape error");

  int width = *(inputs->shape.rbegin());
  int height = *(inputs->shape.rbegin() + 1);
  int channels = *(inputs->shape.rbegin() + 2);
  int batch_size = *(inputs->shape.rbegin() + 3);

  int pooled_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int pooled_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  mask->data.resize({batch_size * channels * pooled_height * pooled_width});
  mask->reshape({batch_size, channels, pooled_height, pooled_width});

  output->data.resize(batch_size * channels * pooled_height * pooled_width);
  output->reshape({batch_size, channels, pooled_height, pooled_width});

  const float* inputs_data_ptr = thrust::raw_pointer_cast(inputs->data.data());
  float* outputs_data_ptr = thrust::raw_pointer_cast(output->data.data());
  float* mask_data_ptr = thrust::raw_pointer_cast(mask->data.data());

  int num_kernels = batch_size * channels * pooled_height * pooled_width;
  int grid_size = ceil((float)num_kernels / BLOCK_SIZE);

  operator_max_pool_h<<<grid_size, BLOCK_SIZE>>>(
      num_kernels, inputs_data_ptr, channels, height, width, pooled_height,
      pooled_width, kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w,
      outputs_data_ptr, mask_data_ptr);

  CUDA_POST_KERNEL_CHECK;
}

void operator_d_max_pool(const Storage* output_grads, const Storage* inputs,
                         const Storage* mask, int kernel_h, int kernel_w,
                         int pad_h, int pad_w, int stride_h, int stride_w,
                         Storage* inputs_grad) {
  CHECK_EQ(output_grads->shape.size(), 4,
           "operator_d_max_pool: output_grads shape error");
  CHECK_EQ(inputs->shape.size(), 4, "operator_d_max_pool: inputs shape error");
  CHECK_EQ(mask->shape.size(), 4, "operator_d_max_pool: mask shape error");

  int width = *(inputs->shape.rbegin());
  int height = *(inputs->shape.rbegin() + 1);
  int channels = *(inputs->shape.rbegin() + 2);
  int batch_size = *(inputs->shape.rbegin() + 3);

  int pooled_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int pooled_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  inputs_grad->data.resize(batch_size * channels * height * width);
  inputs_grad->reshape({batch_size, channels, height, width});

  const float* inputs_data_ptr = thrust::raw_pointer_cast(inputs->data.data());
  const float* outputs_grad_ptr =
      thrust::raw_pointer_cast(output_grads->data.data());
  const float* mask_data_ptr = thrust::raw_pointer_cast(mask->data.data());
  float* inputs_grad_ptr = thrust::raw_pointer_cast(inputs_grad->data.data());

  int num_kernels = batch_size * channels * height * width;
  int grid_size = ceil((float)num_kernels / BLOCK_SIZE);

  operator_d_max_pool_h<<<grid_size, BLOCK_SIZE>>>(
      num_kernels, outputs_grad_ptr, mask_data_ptr, channels, height, width,
      pooled_height, pooled_width, kernel_h, kernel_w, stride_h, stride_w,
      pad_h, pad_w, inputs_grad_ptr);

  CUDA_POST_KERNEL_CHECK;
}