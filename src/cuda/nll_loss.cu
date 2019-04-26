#include <memory>
#include <nll_loss.cuh>

// L = -log_P * Y^T
// dL/d(log_P) = -Y

Storage *operator_nll_loss(const Storage *log_p, const Storage *y) {
  std::unique_ptr<Storage> y_transpose(operator_transpose(y, 0, 1));
  std::unique_ptr<Storage> nll_loss_batch(
      operator_matmul(log_p, y_transpose.get()));
  return operator_mean(nll_loss_batch.get(), 0);
}

Storage *operator_d_nll_loss(const Storage *y) {
  return operator_mul(y, (float)-1);
}