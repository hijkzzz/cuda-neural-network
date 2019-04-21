#include <tensor.cuh>
#include <cuda_runtime.h>

void tensor_add(const Storage *a, const Storage *b, Storage *c) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *b_ptr = thrust::raw_pointer_cast(b->data.data());
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_add_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, b_ptr, c_ptr, size)
}

__global__ void tensor_add_h(const float *a, const float *b, float *c,
                             std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = a[index] + b[index];
  }
}

void tensor_sub(const Storage *a, const Storage *b, Storage *c) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *b_ptr = thrust::raw_pointer_cast(b->data.data());
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_sub_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, b_ptr, c_ptr, size)
}

__global__ void tensor_sub_h(const float *a, const float *b, float *c,
                             std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = a[index] - b[index];
  }
}

void tensor_mul(const Storage *a, const Storage *b, Storage *c) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *b_ptr = thrust::raw_pointer_cast(b->data.data());
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_mul_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, b_ptr, c_ptr, size)
}

__global__ void tensor_mul_h(const float *a, const float *b, float *c,
                             std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = a[index] * b[index];
  }
}

void tensor_div(const Storage *a, const Storage *b, Storage *c) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *b_ptr = thrust::raw_pointer_cast(b->data.data());
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t size = a->data.size();
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_div_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, b_ptr, c_ptr, size)
}

__global__ void tensor_div_h(const float *a, const float *b, float *c,
                             std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = a[index] / b[index];
  }
}

void tensor_matmul(const Storage *a, const Storage *b, Storage *c) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *b_ptr = thrust::raw_pointer_cast(b->data.data());
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t dims = a->shape.size();
  float *shape_a = thrust::raw_pointer_cast(a->shape.data());
  float *shape_b = thrust::raw_pointer_cast(b->shape.data());
}

__global__ void tensor_matmul_h(const float *a, const float *b, float *c,
                                std::size_t *shape_a, std::size_t *shape_b,
                                std::size_t dims) {}

void tensor_transpose(const Storage *a, unsigned int dim0, unsigned int dim1,
                      Storage *c);
__global__ void tensor_transpose_h(const float *a, unsigned int dim0,
                                   unsigned int dim1, float *c,
                                   std::size_t *shape_a, std::size_t dims);

void tensor_log_softmax(const Storage *a, unsigned int dim, Storage *c);
__global__ void tensor_log_softmax_h(const float *a, unsigned int dim, float *c,
                                     std::size_t *shape_a, std::size_t dims);

void tensor_mean(const Storage *a, unsigned int dim, Storage *c);
__global__ void tensor_mean_h(const float *a, unsigned int dim, float *c,
                              std::size_t *shape_a, std::size_t dims);

void tensor_pow(const Storage *a, float e, Storage *c) {
  std::size_t size = a->data.size();
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_pow_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, c_ptr, e, size);
}

__global__ void tensor_pow_h(const float *a, float *c, float e, std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = powf(a[index], e);
  }
}

void tensor_log(const Storage *a, Storage *c) {
  std::size_t size = a->data.size();
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_log_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, c_ptr, size);
}

__global__ void tensor_log_h(const float *a, float *c, std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = logf(a[index]);
  }
}

void tensor_exp(const Storage *a, Storage *c) {
  std::size_t size = a->data.size();
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_exp_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, c_ptr, size);
}

__global__ void tensor_exp_h(const float *a, float *c, std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = expf(a[index]);
  }
}

void tensor_sigmoid(const Storage *a, Storage *c) {
  std::size_t size = a->data.size();
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_sigmoid_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, c_ptr, size);
}

__global__ void tensor_sigmoid_h(const float *a, float *c, std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = 1 / (1 + expf(-a[index]));
  }
}

void tensor_tanh(const Storage *a, Storage *c) {
  std::size_t size = a->data.size();
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());
  std::size_t block_size = ceil(std::static_cast<double>(size) / BLOCK_SIZE_1D);
  tensor_tanh_h<<<block_size, BLOCK_SIZE_1D>>>(a_ptr, c_ptr, size);
}

__global__ void tensor_tanh_h(const float *a, float *c, std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size) {
    c[index] = tanhf(a[index]);
  }
}