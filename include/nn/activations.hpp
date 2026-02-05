#pragma once
#include "layer.hpp"
#include "tensor/tensor.hpp"

class ReLU : public Layer {
public:
  Tensor forward(Tensor &input) override;
};

class Sigmoid : public Layer {
public:
  Tensor forward(Tensor &input) override;
};

class LeakyReLU : public Layer {
private:
  float alpha;

public:
  LeakyReLU(float alpha_ = 0.001f) : alpha(alpha_) {};
  Tensor forward(Tensor &input) override;
};

class Tanh : public Layer {
public:
  Tensor forward(Tensor &input) override;
};
