#include <blas.cuh>
#include <utils.cuh>

#include <cuda_runtime.h>
#include <device_functions.h>
#include <thrust/functional.h>
#include <thrust/transform.h>

#include <cfloat>

Storage *operator_add(const Storage *input1, const Storage *input2) {
  Storage *output = new Storage(input1->shape);
  thrust::transform(input1->data.begin(), input1->data.end(),
                    input2->data.begin(), output->data.begin(),
                    thrust::plus<float>());
  return output;
}

struct add_functor {
  const float e;
  add_functor(float _e) : e(_e) {}
  __host__ __device__ float operator()(const float &x) const { return x + e; }
};

Storage *operator_add(const Storage *input1, float value) {
  Storage *output = new Storage(input1->shape);
  add_functor f(value);
  thrust::transform(input1->data.begin(), input1->data.end(),
                    output->data.begin(), f);

  return output;
}

Storage *operator_sub(const Storage *input1, const Storage *input2) {
  Storage *output = new Storage(input1->shape);
  thrust::transform(input1->data.begin(), input1->data.end(),
                    input2->data.begin(), output->data.begin(),
                    thrust::minus<float>());
  return output;
}

Storage *operator_mul(const Storage *input1, const Storage *input2) {
  Storage *output = new Storage(input1->shape);
  thrust::transform(input1->data.begin(), input1->data.end(),
                    input2->data.begin(), output->data.begin(),
                    thrust::multiplies<float>());
  return output;
}

struct mul_functor {
  const float e;
  mul_functor(float _e) : e(_e) {}
  __host__ __device__ float operator()(const float &x) const { return x * e; }
};

Storage *operator_mul(const Storage *input1, float value) {
  Storage *output = new Storage(input1->shape);
  mul_functor f(value);
  thrust::transform(input1->data.begin(), input1->data.end(),
                    output->data.begin(), f);

  return output;
}

Storage *operator_div(const Storage *input1, const Storage *input2) {
  Storage *output = new Storage(input1->shape);
  thrust::transform(input1->data.begin(), input1->data.end(),
                    input2->data.begin(), output->data.begin(),
                    thrust::divides<float>());
  return output;
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
  thrust::transform(input1->data.begin(), input1->data.end(),
                    output->data.begin(), f);

  return output;
}

struct log_functor {
  __host__ __device__ float operator()(const float &x) const { return logf(x); }
};

Storage *operator_log(const Storage *input1) {
  Storage *output = new Storage(input1->shape);
  log_functor f;
  thrust::transform(input1->data.begin(), input1->data.end(),
                    output->data.begin(), f);

  return output;
}

struct exp_functor {
  __host__ __device__ float operator()(const float &x) const { return expf(x); }
};

Storage *operator_exp(const Storage *input1) {
  Storage *output = new Storage(input1->shape);
  exp_functor f;
  thrust::transform(input1->data.begin(), input1->data.end(),
                    output->data.begin(), f);

  return output;
}

__global__ void operator_matmul_h(const float *input1, const float *input2,
                                  float *output, unsigned int height,
                                  unsigned int k, unsigned int width) {
  __shared__ float shared_input1[TILE_SIZE][TILE_SIZE];
  __shared__ float shared_input2[TILE_SIZE][TILE_SIZE];

  unsigned int batch_idx = blockIdx.x;
  input1 += batch_idx * height * k;
  input2 += batch_idx * k * width;
  output += batch_idx * height * width;

  unsigned int bx = blockIdx.y;
  unsigned int by = blockIdx.z;
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;

  unsigned int row = bx * TILE_SIZE + tx;
  unsigned int col = by * TILE_SIZE + ty;
  float v = 0;

  for (unsigned int i = 0; i < (unsigned int)(ceil((float)k / TILE_SIZE)); i++) {
    if (i * TILE_SIZE + ty < k && row < height)
      shared_input1[tx][ty] = input1[row * k + i * TILE_SIZE + ty];
    else
      shared_input1[tx][ty] = 0;

    if (i * TILE_SIZE + tx < k && col < width)
      shared_input2[tx][ty] = input2[(i * TILE_SIZE + tx) * width + col];
    else
      shared_input2[tx][ty] = 0;
    __syncthreads();

    for (unsigned int j = 0; j < TILE_SIZE; j++)
      v += shared_input1[tx][j] * shared_input2[j][ty];
    __syncthreads();
  }

  if (row < height && col < width) output[row * height + col] = v;
}

Storage *operator_matmul(const Storage *input1, const Storage *input2) {
  unsigned int height = *(input1->shape.rbegin + 1);
  unsigned int k = input1->shape.back();
  unsigned int width = input2->shape.back();
  unsigned int batch_size = 1;
  for (auto i = input1->shape.rbegin() + 2; i < input1->shape.rend(); i++) {
    batch_size *= *i;
  }

  const float *input1_ptr = thrust::raw_pointer_cast(input1->data.data());
  const float *input2_ptr = thrust::raw_pointer_cast(input2->data.data());
  thrust::host_vector<unsigned int> new_shape(input1->shape);
  new_shape.back() = width;
  Storage *output = new Storage(new_shape);
  float *output_ptr = thrust::raw_pointer_cast(output->data.data());

  dim3 dim_block(TILE_SIZE, TILE_SIZE);
  dim3 dim_grid(batch_size, ceil((float)(height / TILE_SIZE)), ceil((float)(width / TILE_SIZE));
  operator_matmul_h<<<dim_grid, dim_block>>>(input1_ptr, input2_ptr, output_ptr, height, k, width);

  CUDA_POST_KERNEL_CHECK;
}

__global__ void operator_transpose_h(const float *input1, float *output,
                                     unsigned int *input1_shape,
                                     unsigned int input1_dims,
                                     unsigned int output_shape, unsigned int dim0,
                                     unsigned int dim1, unsigned int size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    unsigned int *loc = new unsigned int[input1_dims];
    index2loc(index, input1_shape, input1_dims, loc);
    swap(loc[dim0], loc[dim1]);
    unsigned int target_index = loc2index(loc, input1_shape, input1_dims);
    delete[] loc;

    output[target_index] = input1[index];
  }
}

Storage *operator_transpose(const Storage *input1, unsigned int dim0,
                            unsigned int dim1) {
  const float *input1_ptr = thrust::raw_pointer_cast(input1->data.data());
  const unsigned int *input1_shape_ptr =
      thrust::raw_pointer_cast(input1->shape.data());

  thrust::host_vector<unsigned int> new_shape(input1->shape);
  std::swap(new_shape[dim0], new_shape[dim1]);
  Storage *output = new Storage(new_shape);
  float *output_ptr = thrust::raw_pointer_cast(output->data.data());
  unsigned int *output_shape_ptr =
      thrust::raw_pointer_cast(output->shape.data());

  unsigned int size = input1->data.size();
  unsigned int grid_size = ceil((float)(size) / BLOCK_SIZE);
  operator_transpose_h<<<grid_size, BLOCK_SIZE>>>(
      input1_ptr, output_ptr, input1_shape_ptr, input1->shape.size(),
      output_shape_ptr, dim0, dim1, size);

  CUDA_POST_KERNEL_CHECK;
  return output;
}

__global__ void operator_mean_h(const float *input1, float *output,
                                unsigned int *input1_shape,
                                unsigned int input1_dims,
                                unsigned int *output_shape, unsigned int dim,
                                unsigned int dim_stride, unsigned int size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    unsigned int length = input1_shape[dim];

    unsigned int *loc = new unsigned int[input1_dims];
    index2loc(index, output_shape, input1_dims - 1, loc);
    for (unsigned int i = input1_dims - 1; i > dim, i--) {
      loc[i] = loc[i - 1];
    }
    loc[dim] = 0;
    unsigned int base = loc2index(loc, input1_shape, input1_dims);
    delete[] loc;

    double total = 0;
    for (unsigned int i = 0; i < length; i++) {
      total += input1[base + i * dim_stride];
    }

    output[index] = total / length;
  }
}

Storage *operator_mean(const Storage *input1, unsigned int dim) {
  const float *input1_ptr = thrust::raw_pointer_cast(input1->data.data());
  const unsigned int *input1_shape_ptr =
      thrust::raw_pointer_cast(input1->shape.data());

  thrust::host_vector<unsigned int> new_shape(input1->shape);
  new_shape.erase(new_shape.begin() + dim);
  Storage *output = new Storage(new_shape);
  float *output_ptr = thrust::raw_pointer_cast(output->data.data());
  unsigned int *output_shape_ptr =
      thrust::raw_pointer_cast(output->shape.data());

  unsigned int input1_dims = input1->shape.size();
  unsigned int dim_stride = 1;
  for (unsigned int i = dim + 1; i < input1_dims; i++) {
    dim_stride *= input1_shape_ptr[i];
  }

  unsigned int size = output->data.size();
  unsigned int grid_size = ceil((float)(size) / BLOCK_SIZE);
  operator_mean_h<<<grid_size, BLOCK_SIZE>>>(
      input1_ptr, output_ptr, input1_shape_ptr, input1_dims, output_shape_ptr,
      dim, dim_stride, size);

  CUDA_POST_KERNEL_CHECK;
  return output;
}

__global__ void operator_sum_h(const float *input1, float *output,
                               unsigned int *input1_shape,
                               unsigned int input1_dims,
                               unsigned int *output_shape, unsigned int dim,
                               unsigned int dim_stride, unsigned int size) {
  unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    unsigned int length = input1_shape[dim];

    unsigned int *loc = new unsigned int[input1_dims];
    index2loc(index, output_shape, input1_dims - 1, loc);
    for (unsigned int i = input1_dims - 1; i > dim; i--) {
      loc[i] = loc[i - 1];
    }
    loc[dim] = 0;
    unsigned int base = loc2index(loc, input1_shape, input1_dims);
    delete[] loc;

    double total = 0;
    for (unsigned int i = 0; i < length; i++) {
      total += input1[base + i * dim_stride];
    }

    output[index] = total;
  }
}

Storage *operator_sum(const Storage *input1, unsigned int dim) {
  const float *input1_ptr = thrust::raw_pointer_cast(input1->data.data());
  const unsigned int *input1_shape_ptr =
      thrust::raw_pointer_cast(input1->shape.data());

  thrust::host_vector<unsigned int> new_shape(input1->shape);
  new_shape.erase(new_shape.begin() + dim);
  Storage *output = new Storage(new_shape);
  float *output_ptr = thrust::raw_pointer_cast(output->data.data());
  unsigned int *output_shape_ptr =
      thrust::raw_pointer_cast(output->shape.data());

  unsigned int input1_dims = input1->shape.size();
  unsigned int dim_stride = 1;
  for (unsigned int i = dim + 1; i < input1_dims; i++) {
    dim_stride *= input1_shape_ptr[i];
  }

  unsigned int size = output->data.size();
  unsigned int grid_size = ceil((float)(size) / BLOCK_SIZE);
  operator_mean_h<<<grid_size, BLOCK_SIZE>>>(
      input1_ptr, output_ptr, input1_shape_ptr, input1_dims, output_shape_ptr,
      dim, dim_stride, size);

  CUDA_POST_KERNEL_CHECK;
  return output;
}