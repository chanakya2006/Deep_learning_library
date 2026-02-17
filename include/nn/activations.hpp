#pragma once
#include "layer.hpp"
#include "tensor/tensor.hpp"
#include <fstream>

class ReLU : public Layer {
public:
  Tensor forward(Tensor &input) override;
  void save_layer(std::ofstream &out) override;
};

class Sigmoid : public Layer {
public:
  Tensor forward(Tensor &input) override;
  void save_layer(std::ofstream &out) override;
};

class LeakyReLU : public Layer {
public:
  float alpha;

  LeakyReLU(float alpha_ = 0.001f) : alpha(alpha_) {};
  Tensor forward(Tensor &input) override;
  void save_layer(std::ofstream &out) override;
};

class Tanh : public Layer {
public:
  Tensor forward(Tensor &input) override;
  void save_layer(std::ofstream &out) override;
};
