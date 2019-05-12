#include <linear.cuh>
#include <memory>

void operator_linear(const Storage *inputs, const Storage *weights,
                     Storage *output) {
  operator_matmul(inputs, weights, output);
}

void operator_d_linear(
    const Storage *outputs_grad, const Storage *inputs, const Storage *weights,
    Storage *weights_grad, Storage *inputs_grad,
    std::unordered_map<std::string, std::unique_ptr<Storage>> &temp) {
  // W^T
  std::vector<int> weights_t_shape{weights->get_shape()[1],
                                   weights->get_shape()[0]};
  INIT_TEMP(temp, "weights_t", weights_t_shape);
  operator_transpose(weights, temp["weights_t"].get());

  // X^T
  std::vector<int> inputs_t_shape(
      {inputs->get_shape()[1], inputs->get_shape()[0]});
  INIT_TEMP(temp, "inputs_t", inputs_t_shape);
  operator_transpose(inputs, temp["inputs_t"].get());

  // Y = X * W
  // dL/dX = dL/dY * W^T
  // dL/dW = X^T * dL/dY
  operator_matmul(outputs_grad, temp["weights_t"].get(), inputs_grad);
  operator_matmul(temp["inputs_t"].get(), outputs_grad, weights_grad);
}

__global__ void operator_bias_h(const float *inputs, const float *bias,
                                float *output, int width, int size) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < size) {
    int col = index % width;
    output[index] = inputs[index] + bias[col];
  }
}

void operator_linear_bias(const Storage *inputs, const Storage *bias,
                          Storage *output) {
  const float *inputs_ptr = RAW_PTR(inputs->get_data());
  const float *bias_ptr = RAW_PTR(bias->get_data());
  float *output_ptr = RAW_PTR(output->get_data());

  int size = inputs->get_data().size();
  int grid_size = ceil((float)(size) / BLOCK_SIZE);
  int width = bias->get_data().size();
  operator_bias_h<<<grid_size, BLOCK_SIZE>>>(inputs_ptr, bias_ptr, output_ptr,
                                             width, size);

  CUDA_POST_KERNEL_CHECK;
}

void operator_d_linear_bias(const Storage *outputs_grad, Storage *bias_grad) {
  operator_sum(outputs_grad, 0, bias_grad);
}

Linear::Linear(int in_size, int out_size, bool is_bias)
    : in_size(in_size), out_size(out_size), is_bias(is_bias) {
  this->weights.reset(new Storage({in_size, out_size}));
  this->weights_grad.reset(new Storage({in_size, out_size}));
  this->weights->xavier(in_size, out_size);

  if (this->is_bias) {
    this->bias.reset(new Storage({1, out_size}));
    this->bias_grad.reset(new Storage({1, out_size}));
    this->bias->xavier(in_size, out_size);
  }
}

std::vector<std::pair<Storage *, Storage *>> Linear::parameters() {
  if (this->is_bias) {
    return {std::make_pair(this->weights.get(), this->weights_grad.get()),
            std::make_pair(this->bias.get(), this->bias_grad.get())};
  } else {
    return {std::make_pair(this->weights.get(), this->weights_grad.get())};
  }
}

void Linear::forward() {
  const Storage *input = this->pre->get_output();
  std::vector<int> output_shape = {input->get_shape()[0], this->out_size};

  INIT_STORAGE(this->output, output_shape);

  operator_linear(input, this->weights.get(), this->output.get());
  if (this->bias) {
    operator_linear_bias(this->output.get(), this->bias.get(),
                         this->output.get());
  }
}

void Linear::backward() {
  const Storage *input = this->pre->get_output();
  const Storage *output_grad = this->next->get_grad();

  INIT_STORAGE(this->grad, input->get_shape());

  if (this->bias) {
    operator_d_linear_bias(output_grad, this->bias_grad.get());
  }

  operator_d_linear(output_grad, input, this->weights.get(),
                    this->weights_grad.get(), this->grad.get(), this->temp);
}