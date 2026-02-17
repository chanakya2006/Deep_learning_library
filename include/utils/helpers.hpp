#pragma once
#include "model/sequential.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"
#include <memory>
#include <vector>

using namespace std;

std::vector<Tensor *> get_all_parameters(std::vector<Layer *> &layers);

Tensor load_tensor(std::ifstream &in);

unique_ptr<Layer> load_layer(std::ifstream &in);

Sequential load_Sequential_model(string file);
