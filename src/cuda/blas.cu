#include <blas.cuh>
#include <utils.cuh>

#include <cuda_runtime.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <algorithm>
#include <cfloat>

void operator_add(const Storage *input1, const Storage *input2,
                  Storage *outputs) {
  CHECK_EQ(input1->get_data().size(), input2->get_data().size(),
           "operator_add: size error");

  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    input2->get_data().begin(), outputs->get_data().begin(),
                    thrust::plus<float>());
}

struct add_functor {
  const float e;
  add_functor(float _e) : e(_e) {}
  __host__ __device__ float operator()(const float &x) const { return x + e; }
};

void operator_add(const Storage *input1, float value, Storage *outputs) {
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    outputs->get_data().begin(), add_functor(value));
}

void operator_sub(const Storage *input1, const Storage *input2,
                  Storage *outputs) {
  CHECK_EQ(input1->get_data().size(), input2->get_data().size(),
           "operator_sub: size error");

  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    input2->get_data().begin(), outputs->get_data().begin(),
                    thrust::minus<float>());
}

void operator_mul(const Storage *input1, const Storage *input2,
                  Storage *outputs) {
  CHECK_EQ(input1->get_data().size(), input2->get_data().size(),
           "operator_mul: size error");

  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    input2->get_data().begin(), outputs->get_data().begin(),
                    thrust::multiplies<float>());
}

struct mul_functor {
  const float e;
  mul_functor(float _e) : e(_e) {}
  __host__ __device__ float operator()(const float &x) const { return x * e; }
};

void operator_mul(const Storage *input1, float value, Storage *outputs) {
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    outputs->get_data().begin(), mul_functor(value));
}

void operator_div(const Storage *input1, const Storage *input2,
                  Storage *outputs) {
  CHECK_EQ(input1->get_data().size(), input2->get_data().size(),
           "operator_div: size error");

  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    input2->get_data().begin(), outputs->get_data().begin(),
                    thrust::divides<float>());
}

struct pow_functor {
  const float e;
  pow_functor(float _e) : e(_e) {}
  __host__ __device__ float operator()(const float &x) const {
    return powf(x, e);
  }
};

void operator_pow(const Storage *input1, float e, Storage *outputs) {
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    outputs->get_data().begin(), pow_functor(e));
}

struct log_functor {
  __host__ __device__ float operator()(const float &x) const { return logf(x); }
};

void operator_log(const Storage *input1, Storage *outputs) {
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    outputs->get_data().begin(), log_functor());
}

struct exp_functor {
  __host__ __device__ float operator()(const float &x) const { return expf(x); }
};

void operator_exp(const Storage *input1, Storage *outputs) {
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    outputs->get_data().begin(), exp_functor());
}

__global__ void operator_matmul_h(const float *input1, const float *input2,
                                  float *output, int height, int k, int width,
                                  int broadcast) {
  __shared__ float shared_input1[TILE_SIZE][TILE_SIZE];
  __shared__ float shared_input2[TILE_SIZE][TILE_SIZE];

  int batch_idx = blockIdx.z;
  if (broadcast != 1) input1 += batch_idx * height * k;
  if (broadcast != 2) input2 += batch_idx * k * width;
  output += batch_idx * height * width;

  int bx = blockIdx.y;
  int by = blockIdx.x;
  int tx = threadIdx.y;
  int ty = threadIdx.x;

  int row = bx * TILE_SIZE + tx;
  int col = by * TILE_SIZE + ty;
  float v = 0;

  for (int i = 0; i < (int)(ceil((float)k / TILE_SIZE)); i++) {
    if (i * TILE_SIZE + ty < k && row < height)
      shared_input1[tx][ty] = input1[row * k + i * TILE_SIZE + ty];
    else
      shared_input1[tx][ty] = 0;

    if (i * TILE_SIZE + tx < k && col < width)
      shared_input2[tx][ty] = input2[(i * TILE_SIZE + tx) * width + col];
    else
      shared_input2[tx][ty] = 0;
    __syncthreads();

    for (int j = 0; j < TILE_SIZE; j++)
      v += shared_input1[tx][j] * shared_input2[j][ty];
    __syncthreads();
  }

  if (row < height && col < width) output[row * width + col] = v;
}

void operator_matmul(const Storage *input1, const Storage *input2,
                     Storage *outputs, int broadcast) {
  int height = *(input1->get_shape().rbegin() + 1);
  int k = *(input1->get_shape().rbegin());
  int width = *(input2->get_shape().rbegin());
  CHECK_EQ(k, *(input2->get_shape().rbegin() + 1),
           "operator_matmul: shape error");

  // batch size
  std::vector<int> base_shape =
      input1->get_shape().size() > input2->get_shape().size()
          ? input1->get_shape()
          : input2->get_shape();
  int batch_size = 1;
  for (auto i = base_shape.rbegin() + 2; i != base_shape.rend(); i++) {
    batch_size *= *i;
  }

  // pointer
  const float *input1_ptr = RAW_PTR(input1->get_data());
  const float *input2_ptr = RAW_PTR(input2->get_data());
  float *output_ptr = RAW_PTR(outputs->get_data());

  dim3 dim_block(TILE_SIZE, TILE_SIZE);
  dim3 dim_grid(ceil((float)width / TILE_SIZE), ceil((float)height / TILE_SIZE),
                batch_size);
  operator_matmul_h<<<dim_grid, dim_block>>>(input1_ptr, input2_ptr, output_ptr,
                                             height, k, width, broadcast);

  CUDA_POST_KERNEL_CHECK;
}

__global__ void operator_transpose_h(const float *in, float *out, int height,
                                     int width) {
  __shared__ float tile[TILE_SIZE][TILE_SIZE];

  int batch_idx = blockIdx.z;
  in += batch_idx * height * width;
  out += batch_idx * height * width;

  int bx = blockIdx.y;
  int by = blockIdx.x;
  int tx = threadIdx.y;
  int ty = threadIdx.x;

  int row = bx * TILE_SIZE + tx;
  int col = by * TILE_SIZE + ty;

  if (row < height && col < width) {
    // coalesced read from global mem, TRANSPOSED write into shared mem:
    tile[tx][ty] = in[row * width + col];

    __syncthreads();

    // read from shared mem, coalesced write to global mem
    out[col * height + row] = tile[tx][ty];
  }
}

void operator_transpose(const Storage *input1, Storage *outputs) {
  int height = *(input1->get_shape().rbegin() + 1);
  int width = *(input1->get_shape().rbegin());
  int batch_size = 1;
  for (auto iter = input1->get_shape().rbegin() + 2;
       iter != input1->get_shape().rend(); iter++) {
    batch_size *= *iter;
  }

  // input
  const float *input1_ptr = RAW_PTR(input1->get_data());
  float *output_ptr = RAW_PTR(outputs->get_data());

  dim3 dim_block(TILE_SIZE, TILE_SIZE);
  dim3 dim_grid(ceil((float)width / TILE_SIZE), ceil((float)height / TILE_SIZE),
                batch_size);
  operator_transpose_h<<<dim_grid, dim_block>>>(input1_ptr, output_ptr, height,
                                                width);

  CUDA_POST_KERNEL_CHECK;
}

__global__ void operator_mean_h(const float *input1, float *output,
                                const int *input1_shape, int input1_dims,
                                const int *temp_shape, int dim, int dim_stride,
                                int size) {
  extern __shared__ int shared[];

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    int *loc = (int *)shared + threadIdx.x * input1_dims;

    index2loc(index, temp_shape, input1_dims - 1, loc);
    for (int i = input1_dims - 1; i > dim; i--) {
      loc[i] = loc[i - 1];
    }
    loc[dim] = 0;

    int base = loc2index(loc, input1_shape, input1_dims);

    int length = input1_shape[dim];
    double total = 0;
    for (int i = 0; i < length; i++) {
      total += input1[base + i * dim_stride];
    }

    output[index] = total / length;
  }
}

void operator_mean(const Storage *input1, int dim, Storage *outputs) {
  // input
  const float *input1_ptr = RAW_PTR(input1->get_data());
  thrust::device_vector<int> input_shape = input1->get_shape();
  const int *input1_shape_ptr = RAW_PTR(input_shape);
  int input1_dims = input1->get_shape().size();

  // output
  float *output_ptr = RAW_PTR(outputs->get_data());
  thrust::device_vector<int> temp_shape = input_shape;
  temp_shape.erase(temp_shape.begin() + dim);
  int *temp_shape_ptr = RAW_PTR(temp_shape);

  // stride
  int dim_stride = 1;
  for (int i = dim + 1; i < input1_dims; i++) {
    dim_stride *= input1->get_shape()[i];
  }

  int size = input1->get_data().size() / input1->get_shape()[dim];
  int grid_size = ceil((float)(size) / BLOCK_SIZE);
  int shared_memory_size = BLOCK_SIZE * input1_dims * sizeof(int);

  operator_mean_h<<<grid_size, BLOCK_SIZE, shared_memory_size>>>(
      input1_ptr, output_ptr, input1_shape_ptr, input1_dims, temp_shape_ptr,
      dim, dim_stride, size);

  CUDA_POST_KERNEL_CHECK;
}

__global__ void operator_sum_h(const float *input1, float *output,
                               const int *input1_shape, int input1_dims,
                               const int *temp_shape, int dim, int dim_stride,
                               int size) {
  extern __shared__ int shared[];

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    int *loc = (int *)shared + threadIdx.x * input1_dims;

    index2loc(index, temp_shape, input1_dims - 1, loc);
    for (int i = input1_dims - 1; i > dim; i--) {
      loc[i] = loc[i - 1];
    }
    loc[dim] = 0;
    int base = loc2index(loc, input1_shape, input1_dims);

    int length = input1_shape[dim];
    double total = 0;
    for (int i = 0; i < length; i++) {
      total += input1[base + i * dim_stride];
    }

    output[index] = total;
  }
}

void operator_sum(const Storage *input1, int dim, Storage *outputs) {
  // input
  const float *input1_ptr = RAW_PTR(input1->get_data());
  thrust::device_vector<int> input_shape = input1->get_shape();
  const int *input1_shape_ptr = RAW_PTR(input_shape);
  int input1_dims = input1->get_shape().size();

  // output
  float *output_ptr = RAW_PTR(outputs->get_data());
  thrust::device_vector<int> temp_shape = input_shape;
  temp_shape.erase(temp_shape.begin() + dim);
  int *temp_shape_ptr = RAW_PTR(temp_shape);

  // stride
  int dim_stride = 1;
  for (int i = dim + 1; i < input1_dims; i++) {
    dim_stride *= input1->get_shape()[i];
  }

  int size = input1->get_data().size() / input1->get_shape()[dim];
  int grid_size = ceil((float)(size) / BLOCK_SIZE);
  int shared_memory_size = BLOCK_SIZE * input1_dims * sizeof(int);

  operator_sum_h<<<grid_size, BLOCK_SIZE, shared_memory_size>>>(
      input1_ptr, output_ptr, input1_shape_ptr, input1_dims, temp_shape_ptr,
      dim, dim_stride, size);

  CUDA_POST_KERNEL_CHECK;
}