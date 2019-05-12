#include <memory>
#include <nll_loss.cuh>

// L = mean(sum(-log_P element_mul Y, 1), 0)
void operator_nll_loss(
    const Storage *log_p, const Storage *y, Storage *output,
    std::unordered_map<std::string, std::unique_ptr<Storage>> &temp) {
  INIT_TEMP(temp, "nll_loss_batch", y->get_shape());
  operator_mul(log_p, y, temp["nll_loss_batch"].get());

  std::vector<int> sum_shape{y->get_shape()[0], 1};
  INIT_TEMP(temp, "nll_loss_sum", sum_shape);
  operator_sum(temp["nll_loss_batch"].get(), 1, temp["nll_loss_sum"].get());

  operator_mean(temp["nll_loss_sum"].get(), 0, output);
  output->get_data()[0] *= -1;
}

// L = 1_n^T * ((-log_P element_mul Y) * 1_k) / N
// dL/d(log_P) = -Y / N
void operator_d_nll_loss(const Storage *y, Storage *inputs_grad) {
  int batch_size = *y->get_shape().begin();
  operator_mul(y, (float)-1 / batch_size, inputs_grad);
}

void NLLLoss::forward(const Storage *y) {
  const Storage *input = this->pre->get_output();
  this->y = y;

  operator_nll_loss(input, y, this->output.get(), this->temp);
}

void NLLLoss::backward() {
  const Storage *input = this->pre->get_output();

  INIT_STORAGE(this->grad, input->get_shape());
  operator_d_nll_loss(this->y, this->grad.get());
}
