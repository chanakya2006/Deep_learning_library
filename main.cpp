#include "loss/bce.hpp"
#include "model/sequential.hpp"
#include "nn/activations.hpp"
#include "nn/dense.hpp"
#include "optim/Adam.hpp"
#include "tensor/tensor.hpp"
#include "utils/helpers.hpp"
#include <chrono>
#include <climits>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

using namespace std;

int main() {
  // Saving a Tensor to a binary file
  Tensor t({2, 2}, {2, 3, 4, 5}, true);

  std::ofstream out("temp.dat", std::ios::binary);

  t.save_tensor(out);

  out.close();

  // Loading the Tensor frome the same file
  std::ifstream in("temp.dat", std::ios::binary);

  Tensor t_loaded = load_tensor(in);

  in.close();

  return 0;
}

// Implement :
// RSMprop
// Save and load model from a raw binary file

// in layer's the Tensor is begin returned as copy
// instead return the pointer to the answer which
// is being stored as a member variable.
