#include <memory>
#include <nll_loss.cuh>

// L = mean(sum(-log_P element_mul Y, 1), 0)
void operator_nll_loss(const Storage *log_p, const Storage *y,
                       Storage *output) {
  Storage nll_loss_batch;
  operator_mul(log_p, y, &nll_loss_batch);

  Storage nll_loss_sum;
  operator_sum(&nll_loss_batch, 1, &nll_loss_sum);

  Storage nll_loss_mean;
  operator_mean(&nll_loss_sum, 0, &nll_loss_mean);

  operator_mul(&nll_loss_mean, -1, output);
}

// L = 1_n^T * ((-log_P element_mul Y) * 1_k) / N
// dL/d(log_P) = -Y / N
void operator_d_nll_loss(const Storage *y, Storage *inputs_grad) {
  int batch_size = *y->shape.begin();
  operator_mul(y, (float)-1 / batch_size, inputs_grad);
}