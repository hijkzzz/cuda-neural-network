#include <memory>
#include <nll_loss.cuh>

// L = mean(sum(-log_P element_mul Y, 1), 0)
void operator_nll_loss(const Storage *log_p, const Storage *y,
                       Storage *output) {
  Storage nll_loss_batch(y->get_shape());
  operator_mul(log_p, y, &nll_loss_batch);

  Storage nll_loss_sum({nll_loss_batch.get_shape()[0], 1});
  operator_sum(&nll_loss_batch, 1, &nll_loss_sum);

  operator_mean(&nll_loss_sum, 0, output);
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

  operator_nll_loss(input, y, this->output.get());
}

void NLLLoss::backward() {
  const Storage *input = this->pre->get_output();

  INIT_STORAGE(this->grad, input->get_shape());
  operator_d_nll_loss(this->y, this->grad.get());
}
