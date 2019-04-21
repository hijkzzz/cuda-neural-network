#include <tensor.cuh>

#include <cuda_runtime.h>

#include <vector>

Storage* tensor_add(const Storage *a, const Storage *b) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *b_ptr = thrust::raw_pointer_cast(b->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_add_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, b_ptr, c_ptr, size);

  return c;
}

__global__ void tensor_add_h(const float *a, const float *b, float *c,
                             std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = a[index] + b[index];
  }
}

Storage* tensor_sub(const Storage *a, const Storage *b) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *b_ptr = thrust::raw_pointer_cast(b->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_sub_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, b_ptr, c_ptr, size);

  return c;
}

__global__ void tensor_sub_h(const float *a, const float *b, float *c,
                             std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = a[index] - b[index];
  }
}

Storage* tensor_mul(const Storage *a, const Storage *b) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *b_ptr = thrust::raw_pointer_cast(b->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_mul_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, b_ptr, c_ptr, size);

  return c;
}

__global__ void tensor_mul_h(const float *a, const float *b, float *c,
                             std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = a[index] * b[index];
  }
}

Storage* tensor_div(const Storage *a, const Storage *b) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *b_ptr = thrust::raw_pointer_cast(b->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_div_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, b_ptr, c_ptr, size);

  return c;
}

__global__ void tensor_div_h(const float *a, const float *b, float *c,
                             std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = a[index] / b[index];
  }
}

Storage* tensor_matmul(const Storage *a, const Storage *b) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *b_ptr = thrust::raw_pointer_cast(b->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t dims = a->shape.size();
  float *shape_a = thrust::raw_pointer_cast(a->shape.data());
  float *shape_b = thrust::raw_pointer_cast(b->shape.data());

  return c;
}

__global__ void tensor_matmul_h(const float *a, const float *b, float *c,
                                std::size_t *shape_a, std::size_t *shape_b,
                                std::size_t dims) {}

Storage* tensor_transpose(const Storage *a, unsigned int dim0, unsigned int dim1,
                      Storage *c) {}

__global__ void tensor_transpose_h(const float *a, unsigned int dim0,
                                   unsigned int dim1, float *c,
                                   std::size_t *shape_a, std::size_t dims) {}

Storage* tensor_log_softmax(const Storage *a, unsigned int dim) {}

__global__ void tensor_log_softmax_h(const float *a, unsigned int dim, float *c,
                                     std::size_t *shape_a, std::size_t dims) {}

Storage* tensor_mean(const Storage *a, unsigned int dim) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  std::size_t *a_shapre_ptr = thrust::raw_pointer_cast(a->shape.data());
  
  thrust::device<float> new_shape(a->shape);
  new_shape.erase(new_shape.begin() + dim);
  Storage *c = new Storage(new_shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t block_size = ceil(std::static_cast<double>(c->data.size()) / BLOCK_SIZE_1D);
  tensor_mean_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, c_ptr, shape_a, a->shape.size());
}

__global__ void tensor_mean_h(const float *a, float *c, std::size_t *shape_a, std::size_t dims) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    float temp = 0;
    std::size_t length = shape_a[dim];
    for (std::size_t i = 0; i < length; ++i) {
      temp += a[]
    }
    c[index] = temp / length;
  }
  
}

Storage* tensor_pow(const Storage *a, float e) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_pow_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, c_ptr, e, size);

  return c;
}

__global__ void tensor_pow_h(const float *a, float *c, float e,
                             std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = powf(a[index], e);
  }
}

Storage* tensor_log(const Storage *a) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_log_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, c_ptr, size);

  return c;
}

__global__ void tensor_log_h(const float *a, float *c, std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = logf(a[index]);
  }
}

Storage* tensor_exp(const Storage *a) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_exp_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, c_ptr, size);

  return c;
}

__global__ void tensor_exp_h(const float *a, float *c, std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = expf(a[index]);
  }
}

Storage* tensor_sigmoid(const Storage *a) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_sigmoid_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, c_ptr, size);

  return c;
}

__global__ void tensor_sigmoid_h(const float *a, float *c, std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = 1 / (1 + expf(-a[index]));
  }
}

Storage* tensor_tanh(const Storage *a) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_tanh_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, c_ptr, size);

  return c;
}

__global__ void tensor_tanh_h(const float *a, float *c, std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = tanhf(a[index]);
  }
}