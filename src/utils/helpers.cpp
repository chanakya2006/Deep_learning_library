#include "utils/helpers.hpp"
#include "nn/activations.hpp"
#include "nn/dense.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"
#include <cstddef>
#include <cstring>
#include <fstream>
#include <memory>

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

unique_ptr<Layer> load_layer(std::ifstream &in) {
  char tag[SIZE_OF_LAYER_CHAR_TAG];
  in.read(reinterpret_cast<char *>(tag), SIZE_OF_LAYER_CHAR_TAG);

  if (strcmp(tag, "dense") == 0) {
    unique_ptr<Dense> dense = make_unique<Dense>(0, 0);
    dense->W = load_tensor(in);
    dense->b = load_tensor(in);
    dense->input_matmul_W = load_tensor(in);
    dense->W_plus_bias = load_tensor(in);

    return dense;
  }
  // for activation layers
  else if (strcmp(tag, "relu") == 0) {
    unique_ptr<ReLU> relu = make_unique<ReLU>();
    return relu;
  }

  else if (strcmp(tag, "sigmoid") == 0) {
    unique_ptr<Sigmoid> sigmoid = make_unique<Sigmoid>();
    return sigmoid;
  }

  else if (strcmp(tag, "leakyrelu") == 0) {
    // reading the alpha value
    float alpha;
    in.read(reinterpret_cast<char *>(&alpha), sizeof(float));

    unique_ptr<LeakyReLU> leakyrelu = make_unique<LeakyReLU>(alpha);
    return leakyrelu;
  }

  else if (strcmp(tag, "tanh") == 0) {
    unique_ptr<Tanh> tanh = make_unique<Tanh>();
    return tanh;
  }

  return nullptr; // Throw corrupted file exception here
}
