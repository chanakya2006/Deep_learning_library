#include "nn/activations.hpp"

Tensor ReLU::forward(Tensor &input) {
  return input.relu();
}

Tensor Sigmoid::forward(Tensor &input) {
  return input.sigmoid();
}
