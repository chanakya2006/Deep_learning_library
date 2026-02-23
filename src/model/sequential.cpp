// sequential.cpp

#include "model/sequential.hpp"
#include "tensor/tensor.hpp"
#include "utils/helpers.hpp"
#include <cstddef>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

void Sequential::add(unique_ptr<Layer> layer) {
  layers.push_back(std::move(layer));
  out_tensors.resize(layers.size(), Tensor({}, {}, false));
}

Tensor Sequential::forward(Tensor &input) {
  out_tensors[0] = layers[0]->forward(input);
  for (size_t i = 1; i < layers.size(); i++) {
    out_tensors[i] = layers[i]->forward(out_tensors[i - 1]);
  }
  return out_tensors[out_tensors.size() - 1];
}

vector<Tensor *> Sequential::get_parameters() {
  vector<Layer *> out_layer;
  for (size_t i = 0; i < layers.size(); i++) {
    out_layer.push_back(layers[i].get());
  }
  return get_all_parameters(out_layer);
}

void Sequential::save(string file) {
  std::ofstream out(file, std::ios::binary);

  if (!out.is_open()) {
    throw std::runtime_error("File not found or permission denied : " + file);
  }

  // writing the number of layers
  size_t num_of_layer = layers.size();
  out.write(reinterpret_cast<const char *>(&num_of_layer),
            sizeof(num_of_layer));

  for (size_t i = 0; i < layers.size(); i++) {
    layers[i]->save_layer(out);
  }

  out.close();
}
