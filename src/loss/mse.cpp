#include "loss/mse.hpp"

MSE::MSE(int feature)
    : diff({1, feature}, {}, true),
      sq({1, feature}, {}, true),
      sum({1, feature}, {}, true),
      out({1, 1}, {}, true) {}

Tensor MSE::apply(Tensor &pred, Tensor &target) {
  diff = pred - target;
  sq = diff * diff;
  sum = sq.sum();
  out = sum * (1.0f / sq.get_data().size());
  return out;
}
