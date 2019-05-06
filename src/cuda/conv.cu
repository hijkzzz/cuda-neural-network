#include <conv.cuh>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdlib>
#include <memory>
#include <vector>

// inputs: N*C*H*W
// filters: C_out*C_in*K_h*K_w
void operator_conv(const Storage *inputs, const Storage *filters, Storage *cols,
                   const int pad_h, const int pad_w, const int stride_h,
                   const int stride_w, Storage *output) {
  CHECK_EQ(inputs->shape.size(), 4, "operator_conv: inputs shape error");
  CHECK_EQ(filters->shape.size(), 4, "operator_conv: filters shape error");

  int width = *(inputs->shape.rbegin());
  int height = *(inputs->shape.rbegin() + 1);
  int channel_in = *(inputs->shape.rbegin() + 2);
  int batch_size = *(inputs->shape.rbegin() + 3);

  int kernel_w = *(filters->shape.rbegin());
  int kernel_h = *(filters->shape.rbegin() + 1);
  int channel_out = *(filters->shape.rbegin() + 3);

  CHECK_EQ(*(filters->shape.rbegin() + 2), channel_in,
           "operator_conv: channel size error");

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  int batch_im_stride = channel_in * height * width;
  int batch_col_stride =
      channel_in * kernel_h * kernel_w * height_col * width_col;

  // im2col
  // [batch_size*(C_in*k_h*k_w)*(height_col * width_col)]
  cols->data.resize(batch_size * channel_in * kernel_h * kernel_w * height_col *
                    width_col);
  cols->reshape(
      {batch_size, channel_in * kernel_h * kernel_w, height_col * width_col});

  const float *inputs_ptr = thrust::raw_pointer_cast(inputs->data.data());
  const float *filters_ptr = thrust::raw_pointer_cast(filters->data.data());
  float *cols_ptr = thrust::raw_pointer_cast(cols->data.data());
  for (int i = 0; i < batch_size; i++) {
    im2col(inputs_ptr + i * batch_im_stride, channel_in, height, width,
           kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
           cols_ptr + i * batch_col_stride);
  }

  // matmul
  // Y = F * col
  // [C_out*(C_in*k_h*k_w)] * [(C_in*k_h*k_w)*(height_col*width_col)]
  Storage temp_filters(*filters);
  temp_filters.reshape(
      std::vector<int>{channel_out, channel_in * kernel_h * kernel_w});

  // [batch_size * channel_out * (height_col * width_col)]
  output->data.resize(batch_size * channel_out * height_col * width_col);
  output->reshape({batch_size, channel_out, height_col, width_col});
  int batch_output_stride = channel_out * height_col * width_col;

  for (int i = 0; i < batch_size; ++i) {
    auto cols_iter = cols->data.begin() + i * batch_col_stride;
    Storage col(std::vector<int>{channel_in * kernel_h * kernel_w,
                                 height_col * width_col},
                cols_iter, cols_iter + batch_col_stride);

    Storage y_temp;
    operator_matmul(&temp_filters, &col, &y_temp);

    auto outputs_iter = output->data.begin() + i * batch_output_stride;
    assert(y_temp.data.size() == batch_output_stride);
    thrust::copy(y_temp.data.begin(), y_temp.data.end(), outputs_iter);
  }
}

// Y = F * col
// dL/d_col = F^T * dL/dY
// dL/d_im = col2im(dL/d_col)
// dL/dF = dL/dY * col^T
void operator_d_conv(const Storage *outputs_grad, const Storage *inputs,
                     const Storage *cols, const Storage *filters,
                     const int pad_h, const int pad_w, const int stride_h,
                     const int stride_w, Storage *filters_grad,
                     Storage *inputs_grad) {
  CHECK_EQ(outputs_grad->shape.size(), 4,
           "operator_conv: outputs_grad shape error");
  CHECK_EQ(inputs->shape.size(), 4, "operator_conv: inputs shape error");
  CHECK_EQ(cols->shape.size(), 3, "operator_conv: cols shape error");
  CHECK_EQ(filters->shape.size(), 4, "operator_conv: filters shape error");

  Storage *inputs_grad = new Storage(inputs->shape);

  int width = *(inputs->shape.rbegin());
  int height = *(inputs->shape.rbegin() + 1);
  int channel_in = *(inputs->shape.rbegin() + 2);
  int batch_size = *(inputs->shape.rbegin() + 3);

  int kernel_w = *(filters->shape.rbegin());
  int kernel_h = *(filters->shape.rbegin() + 1);
  int channel_out = *(filters->shape.rbegin() + 3);

  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  // F^T
  Storage filters_temp(*filters);
  filters_temp.reshape(
      std::vector<int>{channel_out, channel_in * kernel_h * kernel_w});
  Storage filters_trans;
  operator_transpose(&filters_temp, 0, 1, &filters_trans);

  // filters grad
  filters_grad->data.resize(batch_size * channel_out * channel_in * kernel_h *
                            kernel_w);
  filters_grad->reshape(
      {batch_size, channel_out, channel_in, kernel_h, kernel_w});

  // inputs grad
  inputs_grad->data.resize(inputs->data.size());
  inputs_grad->reshape(inputs->shape);

  // stride
  // int batch_im_stride = channel_in * height * width;
  int batch_col_stride =
      channel_in * kernel_h * kernel_w * height_col * width_col;

  int batch_inputs_grad_stride = channel_in * height * width;
  int batch_filters_grad_stride =
      channel_out * channel_in * kernel_h * kernel_w;
  int batch_outputs_grad_stride = channel_out * height_col * width_col;

  for (int i = 0; i < batch_size; ++i) {
    Storage dl_dy(
        std::vector<int>{channel_out, height_col * width_col},
        outputs_grad->data.begin() + i * batch_outputs_grad_stride,
        outputs_grad->data.begin() + (i + 1) * batch_outputs_grad_stride);
    // dL/d_col = F^T * dL/dY
    Storage dl_dcol;
    operator_matmul(&filters_trans, &dl_dy, &dl_dcol);

    // dL/d_im = col2im(dL/d_col)
    Storage dl_dim({channel_in, height, width});
    const float *dl_dcol_ptr = thrust::raw_pointer_cast(dl_dcol.data.data());
    float *dl_dim_ptr = thrust::raw_pointer_cast(dl_dim.data.data());
    col2im(dl_dcol_ptr, channel_in, height, width, kernel_h, kernel_w, pad_h,
           pad_w, stride_h, stride_w, dl_dim_ptr);
    assert(dl_dim.data.size() == batch_inputs_grad_stride);
    thrust::copy(dl_dim.data.begin(), dl_dim.data.end(),
                 inputs_grad->data.begin() + i * batch_inputs_grad_stride);

    // dL/dF = dL/dY * col^T
    Storage col(std::vector<int>{channel_in * kernel_h * kernel_w,
                                 height_col * width_col},
                cols->data.begin() + i * batch_col_stride,
                cols->data.begin() + (i + 1) * batch_col_stride);
    Storage col_t;
    operator_transpose(&col, 0, 1, &col_t);

    Storage dl_df;
    operator_matmul(&dl_dy, &col_t, &dl_df);
    assert(dl_df.data.size() == batch_filters_grad_stride);
    thrust::copy(dl_df.data.begin(), dl_df.data.end(),
                 filters_grad->data.begin() + i * batch_filters_grad_stride);
  }
}

__global__ void operator_conv_bias_h(const float *inputs, const float *bias,
                                     float *output, int channel_size,
                                     int channel_stride, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    int col = (index / channel_stride) % channel_size;
    output[index] = inputs[index] + bias[col];
  }
}

void operator_conv_bias(const Storage *inputs, const Storage *bias,
                        Storage *output) {
  CHECK_EQ(bias->data.size(), *(inputs->shape.begin() + 1),
           "operator_conv_bias: size error");

  const float *inputs_ptr = thrust::raw_pointer_cast(inputs->data.data());
  const float *bias_ptr = thrust::raw_pointer_cast(bias->data.data());
  float *output_ptr = thrust::raw_pointer_cast(output->data.data());

  output->data.resize(inputs->data.size());
  output->reshape(inputs->shape);

  int channel_stride =
      *(inputs->shape.rbegin()) * *(inputs->shape.rbegin() + 1);

  int size = inputs->data.size();
  int grid_size = ceil((float)(size) / BLOCK_SIZE);
  operator_conv_bias_h<<<grid_size, BLOCK_SIZE>>>(inputs_ptr, bias_ptr,
                                                  output_ptr, bias->data.size(),
                                                  channel_stride, size);

  CUDA_POST_KERNEL_CHECK;
}

void operator_d_conv_bias(const Storage *outputs_grad, Storage *bias_grad,
                          Storage *inputs_grad) {
  // N*C*H*W ==> N*C
  Storage sum3;
  operator_sum(outputs_grad, 3, &sum3);
  Storage sum2;
  operator_sum(&sum3, 2, &sum2);

  *bias_grad = std::move(sum2);
  *inputs_grad = *outputs_grad;
}

// C*H*W >> (C_out*k_h*k_w) * (height_col * width_col)
__global__ void im2col_h(const int n, const float *data_im, const int height,
                         const int width, const int kernel_h,
                         const int kernel_w, const int pad_h, const int pad_w,
                         const int stride_h, const int stride_w,
                         const int height_col, const int width_col,
                         float *data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const int h_index = index / width_col;
    const int h_col = h_index % height_col;
    const int w_col = index % width_col;
    const int c_im = h_index / height_col;
    const int c_col = c_im * kernel_h * kernel_w;
    const int h_offset = h_col * stride_h - pad_h;
    const int w_offset = w_col * stride_w - pad_w;

    // channel offset
    float *data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const float *data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;

    // copy to col
    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int h_im = h_offset + i;
        int w_im = w_offset + j;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                ? data_im_ptr[i * width + j]
                : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void im2col(const float *data_im, const int channels, const int height,
            const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, float *data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int num_kernels = channels * height_col * width_col;
  int grid_size = ceil((float)num_kernels / BLOCK_SIZE);

  im2col_h<<<grid_size, BLOCK_SIZE>>>(
      num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h, pad_w,
      stride_h, stride_w, height_col, width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

// (C_out*k_h*k_w) * (height_col * width_col) >> C*H*W
__global__ void col2im_h(const int n, const float *data_col, const int height,
                         const int width, const int channels,
                         const int kernel_h, const int kernel_w,
                         const int pad_h, const int pad_w, const int stride_h,
                         const int stride_w, const int height_col,
                         const int width_col, float *data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = 0;
    const int w_im = index % width + pad_w;
    const int h_im = (index / width) % height + pad_h;
    const int c_im = index / (width * height);

    // compute the start and end of the col
    const int w_col_start =
        (w_im < kernel_w) ? 0 : (w_im - kernel_w) / stride_w + 1;
    const int w_col_end = fminf(w_im / stride_w + 1, width_col);
    const int h_col_start =
        (h_im < kernel_h) ? 0 : (h_im - kernel_h) / stride_h + 1;
    const int h_col_end = fminf(h_im / stride_h + 1, height_col);

    // copy to im
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        int data_col_index =
            (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col) *
                width_col +
            w_col;
        val += data_col[data_col_index];
      }
    }
    data_im[index] = val;
  }
}

void col2im(const float *data_col, const int channels, const int height,
            const int width, const int kernel_h, const int kernel_w,
            const int pad_h, const int pad_w, const int stride_h,
            const int stride_w, float *data_im) {
  int height_col = height + 2 * pad_h - kernel_h / stride_h + 1;
  int width_col = width + 2 * pad_w - kernel_w / stride_w + 1;
  int num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  int grid_size = ceil((float)num_kernels / BLOCK_SIZE);
  col2im_h<<<grid_size, BLOCK_SIZE>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}
