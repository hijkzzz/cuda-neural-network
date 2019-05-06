#include <linear.cuh>
#include <memory>

void operator_linear(const Storage *inputs, const Storage *weights,
                     Storage *output) {
  operator_matmul(inputs, weights, output);
}

void operator_d_linear(const Storage *outputs_grad, const Storage *inputs,
                       const Storage *weights, Storage *weights_grad,
                       Storage *inputs_grad) {
  Storage weights_transpose;
  operator_transpose(weights, 0, 1, &weights_transpose);
  Storage inputs_transpose;
  operator_transpose(inputs, 0, 1, &inputs_transpose);

  // Y = X * W
  // dL/dX = dL/dY * W^T
  // dL/dW = X^T * dL/dY
  operator_matmul(outputs_grad, &weights_transpose, inputs_grad);

  Storage w_grad;
  operator_matmul(&inputs_transpose, outputs_grad, &w_grad);
  *weights_grad = std::move(w_grad);
}

__global__ void operator_bias_h(const float *inputs, const float *bias,
                                float *output, int width, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    int col = index % width;
    output[index] = inputs[index] + bias[col];
  }
}

void operator_bias(const Storage *inputs, const Storage *bias,
                   Storage *output) {
  const float *inputs_ptr = thrust::raw_pointer_cast(inputs->data.data());
  const float *bias_ptr = thrust::raw_pointer_cast(bias->data.data());

  output->data.resize(inputs->data.size());
  output->reshape(inputs->shape);
  float *output_ptr = thrust::raw_pointer_cast(output->data.data());

  int size = inputs->data.size();
  int grid_size = ceil((float)(size) / BLOCK_SIZE);
  operator_bias_h<<<grid_size, BLOCK_SIZE>>>(inputs_ptr, bias_ptr, output_ptr,
                                             bias->data.size(), size);

  CUDA_POST_KERNEL_CHECK;
}

void operator_d_bias(const Storage *outputs_grad, Storage *bias_grad,
                     Storage *inputs_grad) {
  *bias_grad = *outputs_grad;
  *inputs_grad = *outputs_grad;
}