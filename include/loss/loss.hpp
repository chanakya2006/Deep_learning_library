#pragma once
#include "tensor/tensor.hpp"

class Loss_function {
public:
  virtual Tensor apply(Tensor &pred, Tensor &target) = 0;
  virtual ~Loss_function() {}
};
