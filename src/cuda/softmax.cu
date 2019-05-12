#include <softmax.cuh>

__global__ void operator_log_softmax_h(const float *input1, float *output,
                                       const int *input1_shape, int input1_dims,
                                       const int *temp_shape, int dim,
                                       int dim_stride, int size) {
  extern __shared__ int shared[];
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    int length = input1_shape[dim];

    int *loc = (int *)shared + threadIdx.x * input1_dims;
    index2loc(index, temp_shape, input1_dims - 1, loc);
    for (int i = input1_dims - 1; i > dim; i--) {
      loc[i] = loc[i - 1];
    }
    loc[dim] = 0;
    int base = loc2index(loc, input1_shape, input1_dims);

    float max_ = -FLT_MAX;
    for (int i = 0; i < length; ++i) {
      max_ = fmaxf(max_, input1[base + i * dim_stride]);
    }

    double logsum = 0;
    for (int i = 0; i < length; ++i) {
      logsum += expf(input1[base + i * dim_stride] - max_);
    }
    logsum = max_ + logf(logsum);

    for (int i = 0; i < length; ++i) {
      output[base + i * dim_stride] = input1[base + i * dim_stride] - logsum;
    }
  }
}

void operator_log_softmax(const Storage *input1, int dim, Storage *outputs) {
  // input
  const float *input1_ptr = RAW_PTR(input1->get_data());
  thrust::device_vector<int> input_shape = input1->get_shape();
  const int *input1_shape_ptr = RAW_PTR(input_shape);
  float *output_ptr = RAW_PTR(outputs->get_data());

  thrust::device_vector<int> temp_shape = input_shape;
  temp_shape.erase(temp_shape.begin() + dim);
  int *temp_shape_ptr = RAW_PTR(temp_shape);

  // stride
  int input1_dims = input1->get_shape().size();
  int dim_stride = 1;
  for (int i = dim + 1; i < input1_dims; i++) {
    dim_stride *= input1->get_shape()[i];
  }

  int size = input1->get_data().size() / input1->get_shape()[dim];
  int grid_size = ceil((float)(size) / BLOCK_SIZE);
  int shared_memory_size = BLOCK_SIZE * input1_dims * sizeof(int);

  operator_log_softmax_h<<<grid_size, BLOCK_SIZE, shared_memory_size>>>(
      input1_ptr, output_ptr, input1_shape_ptr, input1_dims, temp_shape_ptr,
      dim, dim_stride, size);

  CUDA_POST_KERNEL_CHECK;
}

__global__ void operator_d_log_softmax_h(const float *output_grads,
                                         const float *input1,
                                         const int *input1_shape,
                                         const int *temp_shape, int input1_dims,
                                         int dim, int dim_stride, int size,
                                         float *input1_grads) {
  extern __shared__ int shared[];
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    int length = input1_shape[dim];

    int *loc = (int *)shared + threadIdx.x * input1_dims;
    index2loc(index, temp_shape, input1_dims - 1, loc);
    for (int i = input1_dims - 1; i > dim; i--) {
      loc[i] = loc[i - 1];
    }
    loc[dim] = 0;
    int base = loc2index(loc, input1_shape, input1_dims);

    float max_ = -FLT_MAX;
    for (int i = 0; i < length; ++i) {
      max_ = fmaxf(max_, input1[base + i * dim_stride]);
    }

    double logsum = 0;
    for (int i = 0; i < length; ++i) {
      logsum += expf(input1[base + i * dim_stride] - max_);
    }
    logsum = max_ + logf(logsum);

    // sum(dL/dY) = dL/dY * 1_n
    double dldysum = 0;
    for (int i = 0; i < length; ++i) {
      dldysum += output_grads[base + i * dim_stride];
    }

    // dL/dY - sum(dL/dY) * exp(x) / sum(exp(x))
    for (int i = 0; i < length; ++i) {
      float x = input1[base + i * dim_stride];
      input1_grads[base + i * dim_stride] =
          output_grads[base + i * dim_stride] - dldysum * expf(x - logsum);
    }
  }
}

// Y = log_softmax(X) = x - log(exp(X) * 1_n) * 1_n^T
// dL/dX = dL/dY - (dL/dY * 1_n * exp(x)) / (exp(x) * 1_n)
void operator_d_log_softmax(const Storage *output_grads, const Storage *input1,
                            int dim, Storage *inputs_grad) {
  const float *input1_ptr = RAW_PTR(input1->get_data());
  thrust::device_vector<int> input_shape = input1->get_shape();
  const int *input1_shape_ptr = RAW_PTR(input_shape);
  const float *output_grads_ptr = RAW_PTR(output_grads->get_data());
  float *input1_grads_ptr = RAW_PTR(inputs_grad->get_data());

  thrust::device_vector<int> temp_shape = input_shape;
  temp_shape.erase(temp_shape.begin() + dim);
  int *temp_shape_ptr = RAW_PTR(temp_shape);

  int input1_dims = input1->get_shape().size();
  int dim_stride = 1;
  for (int i = dim + 1; i < input1_dims; i++) {
    dim_stride *= input1->get_shape()[i];
  }

  int size = input1->get_data().size() / input1->get_shape()[dim];
  int grid_size = ceil((float)(size) / BLOCK_SIZE);
  int shared_memory_size = BLOCK_SIZE * input1_dims * sizeof(int);

  operator_d_log_softmax_h<<<grid_size, BLOCK_SIZE, shared_memory_size>>>(
      output_grads_ptr, input1_ptr, input1_shape_ptr, temp_shape_ptr,
      input1_dims, dim, dim_stride, size, input1_grads_ptr);

  CUDA_POST_KERNEL_CHECK;
}

void LogSoftmax::forward() {
  const Storage *input = this->pre->get_output();

  INIT_STORAGE(this->output, input->get_shape());
  operator_log_softmax(input, this->dim, this->output.get());
}

void LogSoftmax::backward() {
  const Storage *input = this->pre->get_output();
  const Storage *output_grad = this->next->get_grad();

  INIT_STORAGE(this->grad, input->get_shape());
  operator_d_log_softmax(output_grad, input, this->dim, this->grad.get());
}