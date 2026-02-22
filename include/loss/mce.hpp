#pragma once
#include "loss.hpp"
#include "tensor/tensor.hpp"

// multi class cross entropy
class MCE : public Loss_function {
private:
  Tensor log_on_pred;
  Tensor target_mul_log_on_pred;
  Tensor sum;
  Tensor out;

public:
  explicit MCE(int feature);
  Tensor apply(Tensor &pred, Tensor &target) override;
};
