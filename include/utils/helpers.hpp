#pragma once
#include "nn/layer.hpp"
#include <vector>

std::vector<Tensor *> get_all_parameters(std::vector<Layer *> &layers);
