#pragma once
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"
#include <fstream>
#include <vector>

using namespace std;

std::vector<Tensor *> get_all_parameters(std::vector<Layer *> &layers);

Tensor load_tensor(std::ifstream &in);
