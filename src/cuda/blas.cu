#include <blas.cuh>
#include <utils.cuh>

#include <cuda_runtime.h>
#include <device_functions.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <cfloat>

void operator_add(const Storage *input1, const Storage *input2,
                  Storage *outputs) {
  CHECK_EQ(input1->get_data().size(), input2->get_data().size(),
           "operator_add: error size");

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
  add_functor f(value);
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    outputs->get_data().begin(), f);
}

void operator_sub(const Storage *input1, const Storage *input2,
                  Storage *outputs) {
  CHECK_EQ(input1->get_data().size(), input2->get_data().size(),
           "operator_sub: error size");

  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    input2->get_data().begin(), outputs->get_data().begin(),
                    thrust::minus<float>());
}

void operator_mul(const Storage *input1, const Storage *input2,
                  Storage *outputs) {
  CHECK_EQ(input1->get_data().size(), input2->get_data().size(),
           "operator_mul: error size");

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
  mul_functor f(value);
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    outputs->get_data().begin(), f);
}

void operator_div(const Storage *input1, const Storage *input2,
                  Storage *outputs) {
  CHECK_EQ(input1->get_data().size(), input2->get_data().size(),
           "operator_div: error size");

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
  pow_functor f(e);
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    outputs->get_data().begin(), f);
}

struct log_functor {
  __host__ __device__ float operator()(const float &x) const { return logf(x); }
};

void operator_log(const Storage *input1, Storage *outputs) {
  log_functor f;
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    outputs->get_data().begin(), f);
}

struct exp_functor {
  __host__ __device__ float operator()(const float &x) const { return expf(x); }
};

void operator_exp(const Storage *input1, Storage *outputs) {
  exp_functor f;
  thrust::transform(input1->get_data().begin(), input1->get_data().end(),
                    outputs->get_data().begin(), f);
}

__global__ void operator_matmul_h(const float *input1, const float *input2,
                                  float *output, int height, int k, int width,
                                  int broadcast) {
  __shared__ float shared_input1[TILE_SIZE][TILE_SIZE];
  __shared__ float shared_input2[TILE_SIZE][TILE_SIZE];

  int batch_idx = blockIdx.x;
  if (!broadcast == 1) input1 += batch_idx * height * k;
  if (!broadcast == 2) input2 += batch_idx * k * width;
  output += batch_idx * height * width;

  int bx = blockIdx.y;
  int by = blockIdx.z;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

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

  int batch_size = 1;
  for (auto i = input1->get_shape().rbegin() + 2;
       i != input1->get_shape().rend(); i++) {
    batch_size *= *i;
  }

  const float *input1_ptr = thrust::raw_pointer_cast(input1->get_data().data());
  const float *input2_ptr = thrust::raw_pointer_cast(input2->get_data().data());
  float *output_ptr = thrust::raw_pointer_cast(outputs->get_data().data());

  dim3 dim_block(TILE_SIZE, TILE_SIZE);
  dim3 dim_grid(batch_size, ceil((float)height / TILE_SIZE),
                ceil((float)width / TILE_SIZE));
  operator_matmul_h<<<dim_grid, dim_block>>>(input1_ptr, input2_ptr, output_ptr,
                                             height, k, width, broadcast);

  CUDA_POST_KERNEL_CHECK;
}

__global__ void operator_transpose_h(const float *input1, float *output,
                                     const int *input1_shape, int input1_dims,
                                     const int *output_shape, int dim0,
                                     int dim1, int size, int *loc) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    loc += index * input1_dims;
    index2loc(index, output_shape, input1_dims, loc);
    swap(loc[dim0], loc[dim1]);
    int target_index = loc2index(loc, input1_shape, input1_dims);

    output[index] = input1[target_index];
  }
}

void operator_transpose(const Storage *input1, int dim0, int dim1,
                        Storage *outputs) {
  // input
  const float *input1_ptr = thrust::raw_pointer_cast(input1->get_data().data());
  const int *input1_shape_ptr =
      thrust::raw_pointer_cast(input1->get_shape().data());
  int input_dims = input1->get_shape().size();

  // output
  thrust::device_vector<int> output_shape(input1->get_shape().begin(),
                                          input1->get_shape().end());
  swap(output_shape[dim0], output_shape[dim1]);
  int *output_shape_ptr = thrust::raw_pointer_cast(output_shape.data());
  float *output_ptr = thrust::raw_pointer_cast(outputs->get_data().data());

  int size = input1->get_data().size();
  int grid_size = ceil((float)(size) / BLOCK_SIZE);

  // loc buffer
  thrust::device_vector<int> loc(size * input_dims);
  int *loc_ptr = thrust::raw_pointer_cast(loc.data());

  operator_transpose_h<<<grid_size, BLOCK_SIZE>>>(
      input1_ptr, output_ptr, input1_shape_ptr, input_dims, output_shape_ptr,
      dim0, dim1, size, loc_ptr);

  CUDA_POST_KERNEL_CHECK;
}

__global__ void operator_mean_h(const float *input1, float *output,
                                const int *input1_shape, int input1_dims,
                                const int *temp_shape, int dim, int dim_stride,
                                int size, int *loc) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    int length = input1_shape[dim];

    loc += index * input1_dims;
    index2loc(index, temp_shape, input1_dims - 1, loc);
    for (int i = input1_dims - 1; i > dim; i--) {
      loc[i] = loc[i - 1];
    }
    loc[dim] = 0;
    int base = loc2index(loc, input1_shape, input1_dims);

    double total = 0;
    for (int i = 0; i < length; i++) {
      total += input1[base + i * dim_stride];
    }

    output[index] = total / length;
  }
}

void operator_mean(const Storage *input1, int dim, Storage *outputs) {
  // input
  const float *input1_ptr = thrust::raw_pointer_cast(input1->get_data().data());
  const int *input1_shape_ptr =
      thrust::raw_pointer_cast(input1->get_shape().data());
  int input1_dims = input1->get_shape().size();

  // output
  float *output_ptr = thrust::raw_pointer_cast(outputs->get_data().data());
  thrust::device_vector<int> temp_shape(input1->get_shape());
  temp_shape.erase(temp_shape.begin() + dim);
  int *temp_shape_ptr = thrust::raw_pointer_cast(temp_shape.data());

  // stride
  int dim_stride = 1;
  for (int i = dim + 1; i < input1_dims; i++) {
    dim_stride *= input1->get_shape()[i];
  }

  int size = input1->get_data().size() / input1->get_shape()[dim];
  int grid_size = ceil((float)(size) / BLOCK_SIZE);

  // loc buffer
  thrust::device_vector<int> loc(size * input1_dims);
  int *loc_ptr = thrust::raw_pointer_cast(loc.data());

  operator_mean_h<<<grid_size, BLOCK_SIZE>>>(
      input1_ptr, output_ptr, input1_shape_ptr, input1_dims, temp_shape_ptr,
      dim, dim_stride, size, loc);

  CUDA_POST_KERNEL_CHECK;
}

__global__ void operator_sum_h(const float *input1, float *output,
                               const int *input1_shape, int input1_dims,
                               const int *temp_shape, int dim, int dim_stride,
                               int size, int *loc) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    int length = input1_shape[dim];

    loc += index * input1_dims;
    index2loc(index, temp_shape, input1_dims - 1, loc);
    for (int i = input1_dims - 1; i > dim; i--) {
      loc[i] = loc[i - 1];
    }
    loc[dim] = 0;
    int base = loc2index(loc, input1_shape, input1_dims);

    double total = 0;
    for (int i = 0; i < length; i++) {
      total += input1[base + i * dim_stride];
    }

    output[index] = total;
  }
}

void operator_sum(const Storage *input1, int dim, Storage *outputs) {
  // input
  const float *input1_ptr = thrust::raw_pointer_cast(input1->get_data().data());
  const int *input1_shape_ptr =
      thrust::raw_pointer_cast(input1->get_shape().data());
  int input1_dims = input1->get_shape().size();

  // output
  float *output_ptr = thrust::raw_pointer_cast(outputs->get_data().data());
  thrust::device_vector<int> temp_shape(input1->get_shape());
  temp_shape.erase(temp_shape.begin() + dim);
  int *temp_shape_ptr = thrust::raw_pointer_cast(temp_shape.data());

  // stride
  int dim_stride = 1;
  for (int i = dim + 1; i < input1_dims; i++) {
    dim_stride *= input1->get_shape()[i];
  }

  int size = input1->get_data().size() / input1->get_shape()[dim];
  int grid_size = ceil((float)(size) / BLOCK_SIZE);

  // loc buffer
  thrust::device_vector<int> loc(size * input1_dims);
  int *loc_ptr = thrust::raw_pointer_cast(loc.data());

  operator_sum_h<<<grid_size, BLOCK_SIZE>>>(
      input1_ptr, output_ptr, input1_shape_ptr, input1_dims, temp_shape_ptr,
      dim, dim_stride, size, loc);

  CUDA_POST_KERNEL_CHECK;
}