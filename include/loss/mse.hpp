#pragma once
#include "loss.hpp"

class MSE : public Loss_function {
private:
  Tensor diff, sq, sum, out;

public:
  explicit MSE(int feature);
  Tensor apply(Tensor &pred, Tensor &target) override;
};
