#include "nn/activations.hpp"
#include "nn/dense.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"
#include "utils/helpers.hpp"
#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

using namespace std;

int main() {

  Dense dense{4, 5};
  ReLU relu;
  Sigmoid sig;
  LeakyReLU leakyrelu{0.2};
  Tanh tanh;

  // Writing to file
  std::ofstream out("temp.dat", std::ios::binary);

  dense.save_layer(out);
  relu.save_layer(out);
  sig.save_layer(out);
  leakyrelu.save_layer(out);
  tanh.save_layer(out);

  out.close();

  // Reading from file
  std::ifstream in("temp.dat", std::ios::binary);

  unique_ptr<Layer> dense_copy = load_layer(in);
  unique_ptr<Layer> relu_copy = load_layer(in);
  unique_ptr<Layer> sig_copy = load_layer(in);
  unique_ptr<Layer> leakyrelu_copy = load_layer(in);
  unique_ptr<Layer> tanh_copy = load_layer(in);

  in.close();

  return 0;
}

// As each layer has different number of tensor we must know beforehand what
// type layer we are going to read, so when saving the layer I write the type of
// layer it is (tag) and when reading i first check which type of layer it is
// then allocate tensors appropriately

// Implement :
// RSMprop
// Save and load model from a raw binary file

// loading functions of model, layers, and tensors are defined in helpers.hpp
// and Implementend in helpers.cpp
//
// If needed move them to appropriate files
