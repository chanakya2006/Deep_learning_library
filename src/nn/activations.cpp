#include "nn/activations.hpp"
#include "tensor/tensor.hpp"

Tensor ReLU::forward(Tensor &input) { return input.relu(); }

Tensor Sigmoid::forward(Tensor &input) { return input.sigmoid(); }

Tensor LeakyReLU::forward(Tensor &input) { return input.leaky_relu(alpha); }

Tensor Tanh::forward(Tensor &input) { return input.tanh(); }
