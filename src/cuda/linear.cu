#include <linear.cuh>
#include <memory>

Storage *operator_linear(const Storage *inputs, const Storage *weights) {
  return operator_matmul(inputs, weights);
}

Storage *operator_d_linear(const Storage *outputs_grad, const Storage *inputs,
                           const Storage *weights, Storage *weights_grad) {
  std::unique_ptr<Storage> weights_transpose(operator_transpose(weights, 0, 1));
  std::unique_ptr<Storage> inputs_transpose(operator_transpose(inputs, 0, 1));

  // Y = X * W
  // dL/dX = dL/dY * W^T
  // dL/dW = X^T * dL/dY
  Storage *inputs_grad = operator_matmul(outputs_grad, weights_transpose.get());
  std::unique_ptr<Storage> w_grad(
      operator_matmul(inputs_transpose.get(), outputs_grad));
  *weights_grad = std::move(*w_grad.get());

  return inputs_grad;
}

__global__ void operator_bias_h(const float *inputs, const float *bias,
                                float *output, std::size_t width,
                                std::size_t size) {
  std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    std::size_t col = index % width;
    output[index] = inputs[index] + bias[col];
  }
}

Storage *operator_bias(const Storage *inputs, const Storage *bias) {
  const float *inputs_ptr = thrust::raw_pointer_cast(input1->data.data());
  const float *bias_ptr = thrust::raw_pointer_cast(bias->data.data());
  Storage *output = new Storage(input1->shape);
  float *output_ptr = thrust::raw_pointer_cast(output->data.data());

  std::size_t size = input1->data.size();
  std::size_t grid_size = ceil((float)(size) / BLOCK_SIZE);
  operator_bias_h<<<grid_size, BLOCK_SIZE>>>(inputs_ptr, bias_ptr, output_ptr,
                                             bias->data.size(), size);

  CUDA_POST_KERNEL_CHECK;
  return output;
}

Storage *operator_d_bias(const Storage *outputs_grad, Storage *bias_grad) {
  *bias_grad = *outputs_grad;
}