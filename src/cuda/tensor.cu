#include <tensor.cuh>

#include <cuda_runtime.h>
#include <thrust/transform.h>

#include <cfloat>

Storage *tensor_add(const Storage *a, const Storage *b) {
  if (a->data.size() != b->data.size()) {
    throw "tensor_add: a->data.size() != b->data.size()";
  }

  Storage *c = new Storage(a->shape);
  thrust::transform(a->data.begin(), a->data.end(), b->data.begin(),
                    c->data.begin(), thrust::plus<float>());
  return c;
}

struct add_functor {
  const float e;
  pow_functor(float _e) : e(_e) {}
  __host__ __device__ float operator()(const float &x) const { return x + e; }
};

Storage *tensor_add(const Storage *a, float value) {
  Storage *c = new Storage(a->shape);
  add_functor f(value);
  thrust::transform(a->data.begin(), a->data.end(), c->data.begin(), f);

  return c;
}

Storage *tensor_sub(const Storage *a, const Storage *b) {
  if (a->data.size() != b->data.size()) {
    throw "tensor_sub: a->data.size() != b->data.size()";
  }

  Storage *c = new Storage(a->shape);
  thrust::transform(a->data.begin(), a->data.end(), b->data.begin(),
                    c->data.begin(), thrust::minus<float>());
  return c;
}

Storage *tensor_mul(const Storage *a, const Storage *b) {
  if (a->data.size() != b->data.size()) {
    throw "tensor_mul: a->data.size() != b->data.size()";
  }

  Storage *c = new Storage(a->shape);
  thrust::transform(a->data.begin(), a->data.end(), b->data.begin(),
                    c->data.begin(), thrust::multiplies<float>());
  return c;
}

struct mul_functor {
  const float e;
  mul_functor(float _e) : e(_e) {}
  __host__ __device__ float operator()(const float &x) const { return x * e; }
};

Storage *tensor_mul(const Storage *a, float value) {
  Storage *c = new Storage(a->shape);
  mul_functor f(value);
  thrust::transform(a->data.begin(), a->data.end(), c->data.begin(), f);

  return c;
}

Storage *tensor_div(const Storage *a, const Storage *b) {
  if (a->data.size() != b->data.size()) {
    throw "tensor_div: a->data.size() != b->data.size()";
  }

  Storage *c = new Storage(a->shape);
  thrust::transform(a->data.begin(), a->data.end(), b->data.begin(),
                    c->data.begin(), thrust::divides<float>());
  return c;
}

struct pow_functor {
  const float e;
  pow_functor(float _e) : e(_e) {}
  __host__ __device__ float operator()(const float &x) const {
    return powf(x, e);
  }
};

Storage *tensor_pow(const Storage *a, float e) {
  Storage *c = new Storage(a->shape);
  pow_functor f(e);
  thrust::transform(a->data.begin(), a->data.end(), c->data.begin(), f);

  return c;
}

struct log_functor {
  __host__ __device__ float operator()(const float &x) const { return logf(x); }
};

Storage *tensor_log(const Storage *a) {
  Storage *c = new Storage(a->shape);
  log_functor f;
  thrust::transform(a->data.begin(), a->data.end(), c->data.begin(), f);

  return c;
}

struct exp_functor {
  __host__ __device__ float operator()(const float &x) const { return expf(x); }
};

Storage *tensor_exp(const Storage *a) {
  Storage *c = new Storage(a->shape);
  exp_functor f;
  thrust::transform(a->data.begin(), a->data.end(), c->data.begin(), f);

  return c;
}

struct sigmoid_functor {
  __host__ __device__ float operator()(const float &x) const {
    return 1 / (1 + expf(-x));
  }
};

Storage *tensor_sigmoid(const Storage *a) {
  Storage *c = new Storage(a->shape);
  sigmoid_functor f;
  thrust::transform(a->data.begin(), a->data.end(), c->data.begin(), f);

  return c;
}

struct tanh_functor {
  __host__ __device__ float operator()(const float &x) const {
    return tanhf(x);
  }
};

Storage *tensor_tanh(const Storage *a) {
  Storage *c = new Storage(a->shape);
  tanh_functor f;
  thrust::transform(a->data.begin(), a->data.end(), c->data.begin(), f);

  return c;
}

Storage *tensor_matmul(const Storage *a, const Storage *b) {
  if (a->shape.size() != 2 || b->shape.size() != 2) {
    throw "tensor_matmul: only support 2D Tensor";
  }

  if (a->shape[1] != b->shape[0]) {
    throw "tensor_matmul: a->shape[1] != b->shape[0]";
  }

  std::size_t height = a->shape[0];
  std::size_t k = a->shape[1];
  std::size_t width = b->shape[1];

  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  float *b_ptr = thrust::raw_pointer_cast(b->data.data());
  thrust::host_vector<std::size_t> new_shape{height, width};
  Storage *c = Storage(new_shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  dim3 dimBlock(TILED_SIZE, TILED_SIZE);
  dim3 dimGrid(ceil((float)(height / TILE_WIDTH)), ceil((float)(width / TILE_WIDTH));
  tensor_matmul_h<<<dimBlock, dimGrid>>>(a_ptr, b_ptr, c_ptr, height, k, width);
}

__global__ void tensor_matmul_h(const float *a, const float *b, float *c,
                                std::size_t height, std::size_t k,
                                std::size_t width) {

  __shared__ float shared_a[TILE_WIDTH][TILE_WIDTH];
  __shared__ float shared_b[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = bx * TILE_WIDTH + tx;
  int col = by * TILE_WIDTH + ty;
  float v = 0;

  for (std::sizt_t i = 0; i < (std::sizt_t)(ceil((float)k / TILE_WIDTH)); i++) {
    if (i * TILE_WIDTH + ty < k && row < height)
      shared_a[tx][ty] = a[row * k + i * TILE_WIDTH + ty];
    else
      shared_a[tx][ty] = 0;

    if (i * TILE_WIDTH + tx < k && col < width)
      shared_b[tx][ty] = b[(i * TILE_WIDTH + tx) * width + col];
    else
      shared_b[tx][ty] = 0;
    __syncthreads();

    for (int j = 0; j < TILE_WIDTH; j++)
      v += shared_a[tx][j] * shared_b[j][ty];
    __syncthreads();
  }

  if (row < height && col < width)
    c[row * height + col] = v;
}

Storage *tensor_transpose(const Storage *a) {
  if (a->shape.size() != 2) {
    throw "tensor_transpose: only support 2D Tensor";
  }

  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  std::host_vector<std::sizt_t> new_shape(a->data.rbegin().a->data.rend());
  Storage *c = new Storage(new_shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t height = a->shape[0];
  std::size_t width = a->shape[1];

  dim3 dimBlock(TILED_SIZE, TILED_SIZE);
  dim3 dimGrid(ceil((float)(height / TILE_WIDTH)), ceil((float)(width / TILE_WIDTH));
  tensor_transpose_h<<<dimGrid, dimBlock>>>(a_ptr, c_ptr, size, stride);
}
__global__ void tensor_transpose_h(const float *a, float *c, std::size_t height,
                                   std::size_t width) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < height && j < width) {
    c[j * height + i] = a[i * width + j];
  }
}

Storage *tensor_log_softmax(const Storage *a) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  Storage *c = new Storage(a->shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t stride = a->shape.back();
  std::size_t size = a->data.size() / stride;
  std::size_t block_size = ceil((float)(size) / BLOCK_SIZE);
  tensor_mean_h<<<block_size, BLOCK_SIZE>>>(a_ptr, c_ptr, size, stride);
}

__global__ void tensor_log_softmax_h(const float *a, float *c, std::size_t size,
                                     std::size_t stride) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    std::size_t base = index * stride;

    float max_input = -FLT_MIN;
    for (std::size_t i = 0; i < stride; ++i) {
      max_input = max(max_input, a[base + i]);
    }

    double logsum = 0;
    for (std::size_t i = 0; i < stride; ++i) {
      logsum += expf(a[base + i] - max_input);
    }
    logsum = max_input + logf(logsum);

    for (std::size_t i = 0; i < stride; ++i) {
      c[base + i] = a[base + i] - logsum;
    }
  }
}

Storage *tensor_mean(const Storage *a) {
  float *a_ptr = thrust::raw_pointer_cast(a->data.data());
  thrust::host_vector<std::size_t> new_shape(a->shape.begin(),
                                             a->shape.end() - 1);
  Storage *c = new Storage(new_shape);
  float *c_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t stride = a->shape.back();
  std::size_t size = a->data.size() / stride;
  std::size_t block_size = ceil((float)(size) / BLOCK_SIZE);
  tensor_mean_h<<<block_size, BLOCK_SIZE>>>(a_ptr, c_ptr, size, stride);
}

__global__ void tensor_mean_h(const float *a, float *c, std::size_t size,
                              std::size_t stride) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    std::size_t base = index * stride;
    double total = 0;
    for (std::size_t i = 0; i < stride; ++i) {
      total += a[base + i];
    }

    c[index] = total / stride;
  }
}