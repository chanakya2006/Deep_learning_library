#include "nn/dense.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"
#include "utils/helpers.hpp"
#include <climits>
#include <fstream>
#include <iostream>
#include <limits>
#include <vector>

using namespace std;

int main() {

  // When defining and Implementing the function for loading a layer from file
  // return the layer using
  //
  // unique_ptr<Layer>
  //
  // as this not only line ups with way they are stored in model class
  // but simplifies the way we have to handle differnt layer.

  Dense d{4, 5};

  // Writing to file
  std::ofstream out("temp.dat", std::ios::binary);

  d.save_layer(out);

  out.close();

  // Reading from file

  // reading tag
  char tag[SIZE_OF_LAYER_CHAR_TAG];

  std::ifstream in("temp.dat", std::ios::binary);

  in.read(reinterpret_cast<char *>(tag), SIZE_OF_LAYER_CHAR_TAG);

  // loading the tensor appropriately
  Tensor W = load_tensor(in);
  Tensor b = load_tensor(in);
  Tensor input_matmul_W = load_tensor(in);
  Tensor W_plus_bias = load_tensor(in);

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
