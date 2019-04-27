#include <conv.cuh>

#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include <memory>

// inputs: N*C*H*W
// filters: C_out*C_in*K_h*K_w
Storage *operator_conv(const Storage *inputs, const Storage *filters,
                       const std::size_t pad_h, const std::size_t pad_w,
                       const std::size_t stride_h, const std::size_t stride_w) {
  std::size_t width = *inputs->shape.rbegin();
  std::size_t height = *(inputs->shape.rbegin() + 1);
  std::size_t channel_in = *(inputs->shape.rbegin() + 2);
  std::size_t batch_size = *(inputs->shape.rbegin() + 3);

  std::size_t kernel_w = *filters->shape.rbegin();
  std::size_t kernel_h = *(filters->shape.rbegin() + 1);
  std::size_t channel_out = *(filters->shape.rbegin() + 3);

  std::size_t height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  std::size_t width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;

  // im2col
  // [batch_size*(C_in*k_h*k_w)*(height_col * width_col)]
  std::unique_ptr<Storage> cols(new Storage(
      {batch_size, channel_in * kernel_h * kernel_w, height_col * width_col}));

  std::size_t batch_im_stride = channel_in * height * width;
  std::size_t batch_col_stride =
      channel_in * kernel_h * kernel_w * height_col * width_col;

  float *inputs_ptr = thrust::raw_pointer_cast(inputs->data.data());
  float *filters_ptr = thrust::raw_pointer_cast(filters->data.data());
  float *cols_ptr = thrust::raw_pointer_cast(cols->data.data());
  for (std::size_t i = 0; i < batch_size; i++) {
    im2col(inputs_ptr + i * batch_im_stride, channel_in, height, width,
           kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
           cols_ptr + i * batch_col_stride);
  }

  // matmul
  // Y = F * im
  // [C_out*(C_in*k_h*k_w)] * [(C_in*k_h*k_w)*(height_col*width_col)]
  std::unique_ptr<Storage> new_filters(new Storage(*filters));
  new_filters->shape = {channel_out, channel_in * kernel_h * kernel_w};

  // [batch_size * channel_out * (height_col * width_col)]
  Storage *outputs = new Storage({batch_size, channel_out, height_col, width_col}));
  std::size_t output_stride = channel_out * height_col * width_col;

  for (std::size_t i = 0; i < batch_size; ++i) {
    std::unique_ptr<Storage> temp(
        operator_matmul(new_filters.get(), cols_ptr + i * batch_col_stride));
    thrust::copy(temp->data.begin(), temp->data.end(),
                 outputs.begin() + i * output_stride);
  }
  return output;
}

// Y = F * im
// dL/dF = dL/dY * im^T
// dL/d_im = F^T * dL/dY
Storage *operator_d_conv(const Storage *outputs_grad, const Storage *filters,
                         Storage *filters_grad) {
  Storage *inputs_grad = new Storage({});

                         }

__global__ im2col_h(const std::size_t n, const float *data_im,
                    const std::size_t height, const std::size_t width,
                    const std::size_t kernel_h, const std::size_t kernel_w,
                    const std::size_t pad_h, const std::size_t pad_w,
                    const std::size_t stride_h, const std::size_t stride_w,
                    const std::size_t height_col, const std::size_t width_col,
                    float *data_col) {
  CUDA_KERNEL_LOOP(index, n) {
    const std::size_t h_index = index / width_col;
    const std::size_t h_col = h_index % height_col;
    const std::size_t w_col = index % width_col;
    const std::size_t c_im = h_index / height_col;
    const std::size_t c_col = c_im * kernel_h * kernel_w;
    const std::size_t h_offset = h_col * stride_h - pad_h;
    const std::size_t w_offset = w_col * stride_w - pad_w;
    float *data_col_ptr = data_col;
    data_col_ptr += (c_col * height_col + h_col) * width_col + w_col;
    const float *data_im_ptr = data_im;
    data_im_ptr += (c_im * height + h_offset) * width + w_offset;
    for (std::size_t i = 0; i < kernel_h; ++i) {
      for (std::size_t j = 0; j < kernel_w; ++j) {
        std::size_t h_im = h_offset + i;
        std::size_t w_im = w_offset + j;
        *data_col_ptr =
            (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width)
                ? data_im_ptr[i * width + j]
                : 0;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

void im2col(const float *data_im, const std::size_t channels,
            const std::size_t height, const std::size_t width,
            const std::size_t kernel_h, const std::size_t kernel_w,
            const std::size_t pad_h, const std::size_t pad_w,
            const std::size_t stride_h, const std::size_t stride_w,
            float *data_col) {
  // We are going to launch channels * height_col * width_col kernels, each
  // kernel responsible for copying a single-channel grid.
  std::size_t height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  std::size_t width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  std::size_t num_kernels = channels * height_col * width_col;
  std::size_t grid_size = ceil((float)(num_kernels / BLOCK_SIZE);

  im2col_h<<<grid_size, BLOCK_SIZE>>>(
          num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h, pad_w,
          stride_h, stride_w, height_col, width_col, data_col);
  CUDA_POST_KERNEL_CHECK;
}

__global__ void col2im_h(const std::size_t n, const float *data_col,
                         const std::size_t height, const std::size_t width,
                         const std::size_t channels, const std::size_t kernel_h,
                         const std::size_t kernel_w, const std::size_t pad_h,
                         const std::size_t pad_w, const std::size_t stride_h,
                         const std::size_t stride_w,
                         const std::size_t height_col,
                         const std::size_t width_col, float *data_im) {
  CUDA_KERNEL_LOOP(index, n) {
    float val = 0;
    const std::size_t w_im = index % width + pad_w;
    const std::size_t h_im = (index / width) % height + pad_h;
    const std::size_t c_im = index / (width * height);

    // compute the start and end of the output
    const std::size_t w_col_start =
        (w_im < kernel_w) ? 0 : (w_im - kernel_w) / stride_w + 1;
    const std::size_t w_col_end = min(w_im / stride_w + 1, width_col);
    const std::size_t h_col_start =
        (h_im < kernel_h) ? 0 : (h_im - kernel_h) / stride_h + 1;
    const std::size_t h_col_end = min(h_im / stride_h + 1, height_col);

    for (std::size_t h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (std::size_t w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        std::size_t h_k = (h_im - h_col * stride_h);
        std::size_t w_k = (w_im - w_col * stride_w);
        std::size_t data_col_index =
            (((c_im * kernel_h + h_k) * kernel_w + w_k) * height_col + h_col) *
                width_col +
            w_col;
        val += data_col[data_col_index];
      }
    }
    data_im[index] = val;
  }
}

void col2im(const float *data_col, const std::size_t channels,
            const std::size_t height, const std::size_t width,
            const std::size_t kernel_h, const std::size_t kernel_w,
            const std::size_t pad_h, const std::size_t pad_w,
            const std::size_t stride_h, const std::size_t stride_w,
            float *data_im) {
  std::size_t height_col = height + 2 * pad_h - kernel_h / stride_h + 1;
  std::size_t width_col = width + 2 * pad_w - kernel_w / stride_w + 1;
  std::size_t num_kernels = channels * height * width;
  // To avoid involving atomic operations, we will launch one kernel per
  // bottom dimension, and then in the kernel add up the top dimensions.
  std::size_t grid_size = ceil((float)(num_kernels / BLOCK_SIZE);
  col2im_h<<<grid_size, BLOCK_SIZE>>>(
      num_kernels, data_col, height, width, channels, kernel_h, kernel_w, pad_h,
      pad_w, stride_h, stride_w, height_col, width_col, data_im);
  CUDA_POST_KERNEL_CHECK;
}