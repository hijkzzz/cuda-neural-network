#include <softmax.cuh>

__global__ void operator_log_softmax_h(const float *input1, float *output,
                                       std::size_t *input1_shape,
                                       std::size_t input1_dims,
                                       std::size_t temp_shape_ptr,
                                       std::size_t dim, std::size_t dim_stride,
                                       std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    std::size_t length = input1_shape[dim];

    std::size_t *loc = new std::size_t[input1_dims];
    index2loc(index, temp_shape_ptr, input1_dims - 1, loc);
    for (std::size_t i = input1_dims - 1; i > dim, i--) {
      loc[i] = loc[i - 1];
    }
    loc[dim] = 0;
    std::size_t base = loc2index(loc, input1_shape, input1_dims);
    delete[] loc;

    float max_ = -FLT_MIN;
    for (std::size_t i = 0; i < length; ++i) {
      max_ = max(max_, input1[base + i * dim_stride]);
    }

    double logsum = 0;
    for (std::size_t i = 0; i < length; ++i) {
      logsum += expf(input1[base + i * dim_stride] - max_);
    }
    logsum = max_ + logf(logsum);

    for (std::size_t i = 0; i < length; ++i) {
      output[base + i * dim_stride] = input1[base + i * dim_stride] - logsum;
    }
  }
}

Storage *operator_log_softmax(const Storage *input1, std::size_t dim) {
  float *input1_ptr = thrust::raw_pointer_cast(input1->data.data());
  std::size_t *input1_shape_ptr =
      thrust::raw_pointer_cast(input1->shape.data());
  Storage *output = new Storage(input1->shape);
  float *output_ptr = thrust::raw_pointer_cast(output->data.data());

  thrust::device_vector<std::size_t> temp_shape(input1->shape);
  temp_shape.erase(temp_shape.begin() + dim);
  std::size_t *temp_shape_ptr = thrust::raw_pointer_cast(temp_shape.data());

  std::size_t input1_dims = input1->shape.size();
  std::size_t dim_stride = 1;
  for (std::size_t i = dim + 1; i < input1_dims, i++) {
    dim_stride *= input1_shape_ptr[i];
  }

  std::size_t size = input1->data.size() / input1->shape[dim];
  std::size_t grid_size = ceil((float)(size) / BLOCK_SIZE);
  operator_log_softmax_h<<<grid_size, BLOCK_SIZE>>>(
      input1_ptr, output_ptr, input1_shape_ptr, input1_dims, temp_shape_ptr,
      dim dim_stride, size);
}

// Y = log_softmax(X) = x - log(exp(X) * 1_n)
// dL/dX = dL/dY^T * [1_n - exp(x) / (exp(x) * 1_n)] = dL/dY^T * (1_n - softmax(x))
Storage *operator_d_log_softmax(const Storage *input1, std::size_t dim,
                                const Storage *output_grads) {
  
}