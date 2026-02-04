#include "loss/bce.hpp"
#include <stdexcept>

Tensor Binary_cross_entropy::apply(Tensor &pred, Tensor &target) {
  if (pred.get_data().size() != 1 || target.get_data().size() != 1)
    throw std::runtime_error("BCE expects scalar prediction");

  pred_log = pred.log_on_Tensor();
  term1 = target * pred_log;
  one_minus_target = one - target;
  one_minus_pred = one - pred;
  one_minus_pred_log = one_minus_pred.log_on_Tensor();
  term2 = one_minus_target * one_minus_pred_log;
  term1_plus_term2 = term1 + term2;
  sum = term1_plus_term2.sum();
  out = sum * (-1.0f / pred.get_data().size());
  return out;
}
