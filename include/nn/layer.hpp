#pragma once
#include "tensor/tensor.hpp"
#include <fstream>
#include <vector>

#define SIZE_OF_LAYER_CHAR_TAG 10

class Layer {
public:
  virtual Tensor forward(Tensor &input) = 0;
  virtual std::vector<Tensor *> parameters() { return {}; }
  virtual void save_layer(std::ofstream &out) = 0;
  virtual ~Layer() {}
};
