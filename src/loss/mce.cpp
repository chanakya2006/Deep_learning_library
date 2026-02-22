// mce.cpp

#include "loss/mce.hpp"
#include "tensor/tensor.hpp"

MCE::MCE(int feature)
    : log_on_pred({1, feature}, {}, true),
      target_mul_log_on_pred({1, feature}, {}, true), sum({1, 1}, {}, true),
      out({1, 1}, {}, true) {}

Tensor MCE::apply(Tensor &pred, Tensor &target) {
  log_on_pred = pred.log_on_Tensor();
  target_mul_log_on_pred = target * log_on_pred;
  sum = target_mul_log_on_pred.sum();
  out = sum * -1;
  return out;
}
