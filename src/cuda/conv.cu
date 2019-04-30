#include <conv.cuh>

#include <math_functions.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include <cstdlib>
#include <memory>
#include <vector>

// inputs: N*C*H*W
// filters: C_out*C_in*K_h*K_w
Storage *operator_conv(const Storage *inputs, const Storage *filters,
                       const unsigned int pad_h, const unsigned int pad_w,
                       const unsigned int stride_h,
                       const unsigned int stride_w) {
  unsigned int width = *inputs->shape.rbegin();
  unsigned int height = *(inputs->shape.rbegin() + 1);
  unsigned int channel_in = *(inputs->shape.rbegin() + 2);
  unsigned int batch_size = *(inputs->shape.rbegin() + 3);

  unsigned int kernel_w = *filters->shape.rbegin();
  unsigned int kernel_h = *(filters->shape.rbegin() + 1);
  unsigned int channel_out = *(filters->shape.rbegin() + 3);

  unsigned int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  unsigned int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  // im2col
  // [batch_size*(C_in*k_h*k_w)*(height_col * width_col)]
  std::unique_ptr<Storage> cols(new Storage(
      {batch_size, channel_in * kernel_h * kernel_w, height_col * width_col}));

  unsigned int batch_im_stride = channel_in * height * width;
  unsigned int batch_col_stride =
      channel_in * kernel_h * kernel_w * height_col * width_col;

  const float *inputs_ptr = thrust::raw_pointer_cast(inputs->data.data());
  const float *filters_ptr = thrust::raw_pointer_cast(filters->data.data());
  float *cols_ptr = thrust::raw_pointer_cast(cols->data.data());
  for (unsigned int i = 0; i < batch_size; i++) {
    im2col(inputs_ptr + i * batch_im_stride, channel_in, height, width,
           kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
           cols_ptr + i * batch_col_stride);
  }

  // matmul
  // Y = F * im
  // [C_out*(C_in*k_h*k_w)] * [(C_in*k_h*k_w)*(height_col*width_col)]
  std::unique_ptr<Storage> new_filters(new Storage(*filters));
  new_filters->reshape({channel_out, channel_in * kernel_h * kernel_w});

  // [batch_size * channel_out * (height_col * width_col)]
  Storage *outputs =
      new Storage({batch_size, channel_out, height_col, width_col});
  unsigned int output_stride = channel_out * height_col * width_col;

  for (unsigned int i = 0; i < batch_size; ++i) {
    auto cur_output_iter = outputs->data.begin() + i * output_stride;
    auto cur_col_iter = cols->data.begin() + i * batch_col_stride;
    thrust::device_vector<float> cur_cols_data(cur_col_iter,
                                               cur_col_iter + batch_col_stride);
    thrust::host_vector<unsigned int> cur_col_shape{
        channel_in * kernel_h * kernel_w, height_col * width_col};
    Storage cur_col(cur_col_shape, std::move(cur_cols_data));

    std::unique_ptr<Storage> temp(operator_matmul(new_filters.get(), &cur_col));
    thrust::copy(temp->data.begin(), temp->data.end(), cur_output_iter);
  }
  return outputs;
}

// Y = F * col
// dL/d_col = F^T * dL/dY
// dL/d_im = col2im(dL/d_col)
// dL/dF = dL/dY * col^T
Storage *operator_d_conv(const Storage *outputs_grad, const Storage *inputs,
                         const Storage *cols, const Storage *filters,
                         const unsigned int pad_h, const unsigned int pad_w,
                         const unsigned int stride_h,
                         const unsigned int stride_w, Storage *filters_grad) {
  Storage *inputs_grad = new Storage(inputs->shape);

  unsigned int width = *inputs->shape.rbegin();
  unsigned int height = *(inputs->shape.rbegin() + 1);
  unsigned int channel_in = *(inputs->shape.rbegin() + 2);
  unsigned int batch_size = *(inputs->shape.rbegin() + 3);

  unsigned int kernel_w = *filters->shape.rbegin();
  unsigned int kernel_h = *(filters->shape.rbegin() + 1);
  unsigned int channel_out = *(filters->shape.rbegin() + 3);

  unsigned int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  unsigned int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  // F^T
  std::unique_ptr<Storage> filters1(new Storage(*filters));
  filters1->reshape({channel_out, channel_in * kernel_h * kernel_w});
  std::unique_ptr<Storage> filters_t(operator_transpose(filters1.get(), 0, 1));
  filters1.reset();

  // filters grad
  filters_grad->reshape(
      {batch_size, channel_out, channel_in, kernel_h, kernel_w});
  filters_grad->data.resize(batch_size * channel_out * channel_in * kernel_h *
                            kernel_w);

  // stride
  unsigned int batch_im_stride = channel_in * height * width;
  unsigned int batch_col_stride =
      channel_in * kernel_h * kernel_w * height_col * width_col;

  unsigned int batch_inputs_grad_stride = channel_in * height_col * width_col;
  unsigned int batch_filters_grad_stride =
      channel_out * channel_in * kernel_h * kernel_w;
  unsigned int batch_outputs_grad_stride = channel_in * height_col * width_col;

  for (unsigned int i = 0; i < batch_size; ++i) {
    std::unique_ptr<Storage> dl_dy(
        new Storage({channel_out, height_col * width_col},
                    outputs_grad->data.begin() + i * batch_outputs_grad_stride,
                    outputs_grad->data.begin() + i * batch_outputs_grad_stride +
                        batch_outputs_grad_stride));
    // dL/d_col = F^T * dL/dY
    std::unique_ptr<Storage> dl_dcol(
        operator_matmul(filters_t.get(), dl_dy.get()));
    // dL/d_im = col2im(dL/d_col)
    std::unique_ptr<Storage> dl_dim(new Storage({channel_in, height, width}));
    const float *dl_dcol_ptr = thrust::raw_pointer_cast(dl_dcol->data.data());
    float *dl_dim_ptr = thrust::raw_pointer_cast(dl_dim->data.data());
    col2im(dl_dcol_ptr, channel_in, height, width, kernel_h, kernel_w, pad_h,
           pad_w, stride_h, stride_w, dl_dim_ptr);
    thrust::copy(dl_dim->data.begin(), dl_dim->data.end(),
                 inputs_grad->data.begin() + i * batch_inputs_grad_stride);

    // dL/dF = dL/dY * col^T
    std::unique_ptr<Storage> col(new Storage(
        {channel_in * height * width, height_col * width_col},
        cols->data.begin() + i * batch_col_stride,
        cols->data.begin() + i * batch_col_stride + batch_col_stride));
    std::unique_ptr<Storage> col_t(operator_transpose(col.get(), 0, 1));
    col.release();
    std::unique_ptr<Storage> dl_df(operator_matmul(dl_dy.get(), col_t.get()));
    thrust::copy(dl_df->data.begin(), dl_df->data.end(),
                 filters_grad->data.begin() + i * batch_filters_grad_stride);
  }

  return inputs_grad;
}

__global__ void im2col_h(const unsigned int n, const float *data_im,
                         const unsigned int height, const unsigned int width,
                         const unsigned int kernel_h,
                         const unsigned int kernel_w, const unsigned int pad_h,
                         const unsigned int pad_w, const unsigned int stride_h,
                         const unsigned int stride_w,
                         const unsigned int height_col,
                         const unsigned int width_col, float *data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const unsigned int h_index = index / width_col;
    const unsigned int h_col = h_index % height_col;
    const unsigned int w_col = index % width_col;
    const unsigned int c_im = h_index / height_col;
    const unsigned int c_col = c_im * kernel_h * kernel_w;
    const unsigned int h_offset = h_col * stride_h - pad_h;
    const unsigned int w_offset = w_col * stride_w - pad_w;
    float *data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const float *data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (unsigned int i = 0; i < kernel_h; ++i) {
      for (unsigned int j = 0; j < kernel_w; ++j) {
        unsigned int h_im = h_offset + i;
        unsigned int w_im = w_offset + j;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                ? data_im_ptr[i * width + j]
                : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void im2col(const float *data_im, const unsigned int channels,
            const unsigned int height, const unsigned int width,
            const unsigned int kernel_h, const unsigned int kernel_w,
            const unsigned int pad_h, const unsigned int pad_w,
            const unsigned int stride_h, const unsigned int stride_w,
            float *data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  unsigned int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  unsigned int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  unsigned int num_kernels = channels * height_col * width_col;
  unsigned int grid_size = ceil((float)(num_kernels / BLOCK_SIZE));

  im2col_h<<<grid_size, BLOCK_SIZE>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h, pad_w,
      stride_h, stride_w, height_col, width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void col2im_h(const unsigned int n, const float *data_col,
                         const unsigned int height, const unsigned int width,
                         const unsigned int channels,
                         const unsigned int kernel_h,
                         const unsigned int kernel_w, const unsigned int pad_h,
                         const unsigned int pad_w, const unsigned int stride_h,
                         const unsigned int stride_w,
                         const unsigned int height_col,
                         const unsigned int width_col, float *data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = 0;
    const unsigned int w_im = index % width + pad_w;
    const unsigned int h_im = (index / width) % height + pad_h;
    const unsigned int c_im = index / (width * height);

    // compute the start and end of the output
    const unsigned int w_col_start =
        (w_im < kernel_w) ? 0 : (w_im - kernel_w) / stride_w + 1;
    const unsigned int w_col_end = min(w_im / stride_w + 1, width_col);
    const unsigned int h_col_start =
        (h_im < kernel_h) ? 0 : (h_im - kernel_h) / stride_h + 1;
    const unsigned int h_col_end = min(h_im / stride_h + 1, height_col);

    for (unsigned int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (unsigned int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        unsigned int h_k = (h_im - h_col * stride_h);
        unsigned int w_k = (w_im - w_col * stride_w);
        unsigned int data_col_index =
            (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col) *
                width_col +
            w_col;
        val += data_col[data_col_index];
      }
    }
    data_im[index] = val;
  }
}

void col2im(const float *data_col, const unsigned int channels,
            const unsigned int height, const unsigned int width,
            const unsigned int kernel_h, const unsigned int kernel_w,
            const unsigned int pad_h, const unsigned int pad_w,
            const unsigned int stride_h, const unsigned int stride_w,
            float *data_im) {
  unsigned int height_col = height + 2 * pad_h - kernel_h / stride_h + 1;
  unsigned int width_col = width + 2 * pad_w - kernel_w / stride_w + 1;
  unsigned int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  unsigned int grid_size = ceil((float)(num_kernels / BLOCK_SIZE);
  col2im_h<<<grid_size, BLOCK_SIZE>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}