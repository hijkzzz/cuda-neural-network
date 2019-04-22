#include <tensor.cuh>

#include <cuda_runtime.h>
#include <thrust/transform.h>

Storage *tensor_add(const Storage *a, const Storage *b) {
  if (a->data.size() != b->data.size()) {
    throw "a.data.size != b.data.size";
  }

  Storage *c = new Storage(a->shape);
  thrust::transform(a->begin(), b->end(), c->begin(), thrust::plus<float>());
  return c;
}

Storage *tensor_sub(const Storage *a, const Storage *b) {
  if (a->data.size() != b->data.size()) {
    throw "a.data.size != b.data.size";
  }

  Storage *c = new Storage(a->shape);
  thrust::transform(a->begin(), b->end(), c->begin(), thrust::minus<float>());
  return c;
}

Storage *tensor_mul(const Storage *a, const Storage *b) {
  if (a->data.size() != b->data.size()) {
    throw "a.data.size != b.data.size";
  }

  Storage *c = new Storage(a->shape);
  thrust::transform(a->begin(), b->end(), c->begin(),
                    thrust::multiplies<float>());
  return c;
}

Storage *tensor_div(const Storage *a, const Storage *b) {
  if (a->data.size() != b->data.size()) {
    throw "a.data.size != b.data.size";
  }

  Storage *c = new Storage(a->shape);
  thrust::transform(a->begin(), b->end(), c->begin(), thrust::divides<float>());
  return c;
}

Storage *tensor_pow(const Storage *a, float e) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE);
  tensor_pow_h<<<block_size, BLOCK_SIZE>>>(a_ptr, c_ptr, e, size);

  return c;
}

__global__ void tensor_pow_h(const float *a, float *c, float e,
                             std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = powf(a[index], e);
  }
}

Storage *tensor_log(const Storage *a) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE);
  tensor_log_h<<<block_size, BLOCK_SIZE>>>(a_ptr, c_ptr, size);

  return c;
}

__global__ void tensor_log_h(const float *a, float *c, std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = logf(a[index]);
  }
}

Storage *tensor_exp(const Storage *a) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE);
  tensor_exp_h<<<block_size, BLOCK_SIZE>>>(a_ptr, c_ptr, size);

  return c;
}

__global__ void tensor_exp_h(const float *a, float *c, std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = expf(a[index]);
  }
}

Storage *tensor_sigmoid(const Storage *a) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE);
  tensor_sigmoid_h<<<block_size, BLOCK_SIZE>>>(a_ptr, c_ptr, size);

  return c;
}

__global__ void tensor_sigmoid_h(const float *a, float *c, std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = 1 / (1 + expf(-a[index]));
  }
}

Storage *tensor_tanh(const Storage *a) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE);
  tensor_tanh_h<<<block_size, BLOCK_SIZE>>>(a_ptr, c_ptr, size);

  return c;
}

__global__ void tensor_tanh_h(const float *a, float *c, std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = tanhf(a[index]);
  }
}

Storage *tensor_matmul(const Storage *a, const Storage *b);
__global__ void tensor_matmul_h(const float *a, const float *b, float *c,
                                std::size_t width, std::size_t k,
                                std::size_t height);

Storage *tensor_transpose(const Storage *a);
__global__ void tensor_transpose_h(const float *a, float *c, std::size_t width,
                                   std::size_t height);

Storage *tensor_log_softmax(const Storage *a) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = c->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE);
  std::size_t stride = a->shape.back();
  tensor_mean_h<<<block_size, BLOCK_SIZE>>>(a_ptr, c_ptr, size, stride);
}

__global__ void tensor_log_softmax_h(const float *a, float *c,
                                     std::size_t stride) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    real max_input = -THInf;
    for (d = 0; d < LOG_SOFTMAX_CAST_TYPE dim_size; d++)
      max_input = THMax(max_input, input_data[d * dim_stride]);

    accreal logsum = 0;
    for (d = 0; d < LOG_SOFTMAX_CAST_TYPE dim_size; d++)
      logsum += exp(input_data[d * dim_stride] - max_input);
    logsum = max_input + logf(logsum);

    for (d = 0; d < LOG_SOFTMAX_CAST_TYPE dim_size; d++)
      output_data[d * dim_stride] = input_data[d * dim_stride] - logsum;
  }
}

Storage *tensor_mean(const Storage *a) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  thrust::host_vector<std::size_t> new_shape(a->shape.begin(),
                                             a->shape.end() - 1);
  Storage *c = new Storage(new_shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = c->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE);
  std::size_t stride = a->shape.back();
  tensor_mean_h<<<block_size, BLOCK_SIZE>>>(a_ptr, c_ptr, size, stride);
}

__global__ void tensor_mean_h(const float *a, float *c, std::size_t size,
                              std::size_t stride) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    float temp = 0;
    for (std::size_t i = 0; i < stride; ++i) {
      temp += a[index * stride + i];
    }
    c[index] = temp / stride;
  }
}