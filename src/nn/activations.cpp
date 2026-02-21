#include "nn/activations.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"
#include <fstream>

Tensor ReLU::forward(Tensor &input) { return input.relu(); }

void ReLU::save_layer(std::ofstream &out) {
  char tag[SIZE_OF_LAYER_CHAR_TAG] = "relu";
  out.write(reinterpret_cast<const char *>(tag), SIZE_OF_LAYER_CHAR_TAG);
}

Tensor Sigmoid::forward(Tensor &input) { return input.sigmoid(); }

void Sigmoid::save_layer(std::ofstream &out) {
  char tag[SIZE_OF_LAYER_CHAR_TAG] = "sigmoid";
  out.write(reinterpret_cast<const char *>(tag), SIZE_OF_LAYER_CHAR_TAG);
}

Tensor LeakyReLU::forward(Tensor &input) { return input.leaky_relu(alpha); }

void LeakyReLU::save_layer(std::ofstream &out) {
  char tag[SIZE_OF_LAYER_CHAR_TAG] = "leakyrelu";
  out.write(reinterpret_cast<const char *>(tag), SIZE_OF_LAYER_CHAR_TAG);

  // Writing alpha value
  out.write(reinterpret_cast<const char *>(&alpha), sizeof(float));
}

Tensor Tanh::forward(Tensor &input) { return input.tanh(); }

void Tanh::save_layer(std::ofstream &out) {
  char tag[SIZE_OF_LAYER_CHAR_TAG] = "tanh";
  out.write(reinterpret_cast<const char *>(tag), SIZE_OF_LAYER_CHAR_TAG);
}

Tensor Softmax::forward(Tensor &input) { return input.softmax(); }

void Softmax::save_layer(std::ofstream &out) {
  char tag[SIZE_OF_LAYER_CHAR_TAG] = "softmax";
  out.write(reinterpret_cast<const char *>(tag), SIZE_OF_LAYER_CHAR_TAG);
}
