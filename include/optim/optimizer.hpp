#pragma once
#include "tensor/tensor.hpp"
#include <vector>

class Optimizer {
public:
  virtual void step(std::vector<Tensor *> &parameters) = 0;
  virtual void zero_grad(std::vector<Tensor *> &parameters){
    for (Tensor *t : parameters)
      t->zero_grad();
  }
  virtual ~Optimizer() {}
};
