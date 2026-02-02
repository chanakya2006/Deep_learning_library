#pragma once
#include "optimizer.hpp"

class SGD : public Optimizer {
  float lr;

public:
  explicit SGD(float lr);
  void step(std::vector<Tensor *> &parameters) override;
};
