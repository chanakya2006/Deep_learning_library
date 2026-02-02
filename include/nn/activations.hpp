#pragma once
#include "layer.hpp"

class ReLU : public Layer {
public:
  Tensor forward(Tensor &input) override;
};

class Sigmoid : public Layer {
public:
  Tensor forward(Tensor &input) override;
};
