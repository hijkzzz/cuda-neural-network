#include <operator.cuh>

#include <cuda_runtime.h>
#include <thrust/transform.h>

#include <cfloat>

Storage *operator_add(const Storage *input1, const Storage *input2) {
  if (input1->data.size() != input2->data.size()) {
    throw "operatoradd: input1->data.size() != input2->data.size()";
  }

  Storage *output = new Storage(input1->shape);
  thrust::transform(input1->data.begin(), input1->data.end(), input2->data.begin(),
                    c->data.begin(), thrust::plus<float>());
  return c;
}

struct add_functor {
  const float e;
  pow_functor(float _e) : e(_e) {}
  __host__ __device__ float operator()(const float &x) const { return x + e; }
};

Storage *operator_add(const Storage *input1, float value) {
  Storage *output = new Storage(input1->shape);
  add_functor f(value);
  thrust::transform(input1->data.begin(), input1->data.end(), c->data.begin(), f);

  return c;
}

Storage *operator_sub(const Storage *input1, const Storage *input2) {
  if (input1->data.size() != input2->data.size()) {
    throw "operatorsub: input1->data.size() != input2->data.size()";
  }

  Storage *output = new Storage(input1->shape);
  thrust::transform(input1->data.begin(), input1->data.end(), input2->data.begin(),
                    c->data.begin(), thrust::minus<float>());
  return c;
}

Storage *operator_mul(const Storage *input1, const Storage *input2) {
  if (input1->data.size() != input2->data.size()) {
    throw "operatormul: input1->data.size() != input2->data.size()";
  }

  Storage *output = new Storage(input1->shape);
  thrust::transform(input1->data.begin(), input1->data.end(), input2->data.begin(),
                    c->data.begin(), thrust::multiplies<float>());
  return c;
}

struct mul_functor {
  const float e;
  mul_functor(float _e) : e(_e) {}
  __host__ __device__ float operator()(const float &x) const { return x * e; }
};

Storage *operator_mul(const Storage *input1, float value) {
  Storage *output = new Storage(input1->shape);
  mul_functor f(value);
  thrust::transform(input1->data.begin(), input1->data.end(), c->data.begin(), f);

  return c;
}

Storage *operator_div(const Storage *input1, const Storage *input2) {
  if (input1->data.size() != input2->data.size()) {
    throw "operatordiv: input1->data.size() != input2->data.size()";
  }

  Storage *output = new Storage(input1->shape);
  thrust::transform(input1->data.begin(), input1->data.end(), input2->data.begin(),
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

Storage *operator_pow(const Storage *input1, float e) {
  Storage *output = new Storage(input1->shape);
  pow_functor f(e);
  thrust::transform(input1->data.begin(), input1->data.end(), c->data.begin(), f);

  return c;
}

struct log_functor {
  __host__ __device__ float operator()(const float &x) const { return logf(x); }
};

Storage *operator_log(const Storage *input1) {
  Storage *output = new Storage(input1->shape);
  log_functor f;
  thrust::transform(input1->data.begin(), input1->data.end(), c->data.begin(), f);

  return c;
}

struct exp_functor {
  __host__ __device__ float operator()(const float &x) const { return expf(x); }
};

Storage *operator_exp(const Storage *input1) {
  Storage *output = new Storage(input1->shape);
  exp_functor f;
  thrust::transform(input1->data.begin(), input1->data.end(), c->data.begin(), f);

  return c;
}

struct sigmoid_functor {
  __host__ __device__ float operator()(const float &x) const {
    return 1 / (1 + expf(-x));
  }
};

Storage *operator_sigmoid(const Storage *input1) {
  Storage *output = new Storage(input1->shape);
  sigmoid_functor f;
  thrust::transform(input1->data.begin(), input1->data.end(), c->data.begin(), f);

  return c;
}

struct tanh_functor {
  __host__ __device__ float operator()(const float &x) const {
    return tanhf(x);
  }
};

Storage *operator_tanh(const Storage *input1) {
  Storage *output = new Storage(input1->shape);
  tanh_functor f;
  thrust::transform(input1->data.begin(), input1->data.end(), c->data.begin(), f);

  return c;
}

Storage *operator_matmul(const Storage *input1, const Storage *input2) {
  if (input1->shape.size() != 2 || input2->shape.size() != 2) {
    throw "operatormatmul: only support 2D Tensor";
  }

  if (input1->shape[1] != input2->shape[0]) {
    throw "operatormatmul: input1->shape[1] != input2->shape[0]";
  }

  std::size_t height = input1->shape[0];
  std::size_t k = input1->shape[1];
  std::size_t width = input2->shape[1];

  float *input1_ptr = thrust::raw_pointer_cast(input1->data.data());
  float *input2_ptr = thrust::raw_pointer_cast(input2->data.data());
  thrust::host_vector<std::size_t> new_shape{height, width};
  Storage *output = Storage(new_shape);
  float *output_ptr = thrust::raw_pointer_cast(c->data.data());

  dim3 dimBlock(TILED_SIZE, TILED_SIZE);
  dim3 dimGrid(ceil((float)(height / TILE_WIDTH)), ceil((float)(width / TILE_WIDTH));
  operator_matmul_h<<<dimBlock, dimGrid>>>(a_ptr, b_ptr, c_ptr, height, k, width);
}

__global__ void operator_matmul_h(const float *input1, const float *input2, float *output,
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

Storage *operator_transpose(const Storage *input1) {
  if (input1->shape.size() != 2) {
    throw "operatortranspose: only support 2D Tensor";
  }

  float *input1_ptr = thrust::raw_pointer_cast(input1->data.data());
  std::host_vector<std::sizt_t> new_shape(input1->data.rbegin().input1->data.rend());
  Storage *output = new Storage(new_shape);
  float *output_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t height = input1->shape[0];
  std::size_t width = input1->shape[1];

  dim3 dimBlock(TILED_SIZE, TILED_SIZE);
  dim3 dimGrid(ceil((float)(height / TILE_WIDTH)), ceil((float)(width / TILE_WIDTH));
  operator_transpose_h<<<dimGrid, dimBlock>>>(a_ptr, c_ptr, size, stride);
}
__global__ void operator_transpose_h(const float *input1, float *output, std::size_t height,
                                   std::size_t width) {
  std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  std::size_t j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < height && j < width) {
    c[j * height + i] = a[i * width + j];
  }
}

Storage *operator_log_softmax(const Storage *input1) {
  float *input1_ptr = thrust::raw_pointer_cast(input1->data.data());
  Storage *output = new Storage(input1->shape);
  float *output_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t stride = input1->shape.back();
  std::size_t size = input1->data.size() / stride;
  std::size_t block_size = ceil((float)(size) / BLOCK_SIZE);
  operator_log_softmax_h<<<block_size, BLOCK_SIZE>>>(a_ptr, c_ptr, size, stride);
}

__global__ void operator_log_softmax_h(const float *input1, float *output, std::size_t size,
                                     std::size_t stride) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    std::size_t base = index * stride;

    float max_*input = -FLT_MIN;
    for (std::size_t i = 0; i < stride; ++i) {
      max_*input = max(max_*input, a[base + i]);
    }

    double logsum = 0;
    for (std::size_t i = 0; i < stride; ++i) {
      logsum += expf(a[base + i] - max_*input);
    }
    logsum = max_*input + logf(logsum);

    for (std::size_t i = 0; i < stride; ++i) {
      c[base + i] = a[base + i] - logsum;
    }
  }
}

Storage *operator_mean(const Storage *input1, std::size_t dim) {
  float *input1_ptr = thrust::raw_pointer_cast(input1->data.data());
  thrust::host_vector<std::size_t> new_shape(input1->shape.begin(),
                                             input1->shape.end() - 1);
  Storage *output = new Storage(new_shape);
  float *output_ptr = thrust::raw_pointer_cast(c->data.data());

  std::size_t stride = input1->shape.back();
  std::size_t size = input1->data.size() / stride;
  std::size_t block_size = ceil((float)(size) / BLOCK_SIZE);
  operator_mean_h<<<block_size, BLOCK_SIZE>>>(a_ptr, c_ptr, size, stride);
}

__global__ void operator_mean_h(const float *input1, float *output, std::size_t size,
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