#include "utils/helpers.hpp"

std::vector<Tensor *> get_all_parameters(std::vector<Layer *> &layers) {
  std::vector<Tensor *> params;
  for (Layer *l : layers) {
    std::vector<Tensor *> parameters_of_layer_l = l->parameters();
    params.insert(params.end(), parameters_of_layer_l.begin(), parameters_of_layer_l.end());
  }
  return params;
}
