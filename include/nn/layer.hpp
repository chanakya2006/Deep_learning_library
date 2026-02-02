#pragma once
#include "tensor/tensor.hpp"
#include <vector>

class Layer {
public:
  virtual Tensor forward(Tensor &input) = 0;
  virtual std::vector<Tensor *> parameters() { return {}; }
  virtual ~Layer() {}
};
