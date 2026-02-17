// sequential.hpp

#pragma once

#include "nn/layer.hpp"
#include "tensor/tensor.hpp"
#include <memory>
#include <vector>
using namespace std;

class Sequential {
private:
  vector<unique_ptr<Layer>> layers;
  vector<Tensor> out_tensors;

public:
  Sequential() = default;
  void add(unique_ptr<Layer> layer);
  Tensor forward(Tensor &input);
  void save(string file_name);
  vector<Tensor *> get_parameters();
};
