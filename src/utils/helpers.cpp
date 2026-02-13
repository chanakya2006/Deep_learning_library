#include "utils/helpers.hpp"
#include "tensor/tensor.hpp"

std::vector<Tensor *> get_all_parameters(std::vector<Layer *> &layers) {
  std::vector<Tensor *> params;
  for (Layer *l : layers) {
    std::vector<Tensor *> parameters_of_layer_l = l->parameters();
    params.insert(params.end(), parameters_of_layer_l.begin(),
                  parameters_of_layer_l.end());
  }
  return params;
}

Tensor load_tensor(std::ifstream &in) {
  size_t size_of_shape;
  in.read(reinterpret_cast<char *>(&size_of_shape), sizeof(size_of_shape));

  vector<int> shape(size_of_shape);
  in.read(reinterpret_cast<char *>(shape.data()), size_of_shape * sizeof(int));

  size_t size_of_data;
  in.read(reinterpret_cast<char *>(&size_of_data), sizeof(size_of_data));

  vector<float> data(size_of_data);
  in.read(reinterpret_cast<char *>(data.data()), size_of_data * sizeof(float));

  bool requires_grad = false;
  in.read(reinterpret_cast<char *>(&requires_grad), sizeof(requires_grad));

  Tensor t(shape, data, requires_grad);

  return t;
}
