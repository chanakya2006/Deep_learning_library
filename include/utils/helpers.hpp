#pragma once
#include "model/sequential.hpp"
#include "nn/layer.hpp"
#include "tensor/tensor.hpp"
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

std::vector<Tensor *> get_all_parameters(std::vector<Layer *> &layers);

Tensor load_tensor(std::ifstream &in);

unique_ptr<Layer> load_layer(std::ifstream &in);

Sequential load_Sequential_model(string file);

struct data {
  vector<float> input;
  vector<float> output;
};

class csv_loader {
public:
  std::ifstream in;
  size_t split_at;
  size_t num_of_rows = 0;

  csv_loader(string file, size_t split_at_)
      : in(std::ifstream(file)), split_at(split_at_) {

    if (!in.is_open()) {
      throw std::runtime_error("File not found or permission denied : " + file);
    }

    string line;
    while (std::getline(in, line)) {
      num_of_rows++;
    }
    in.clear();
    in.seekg(0, std::ios::beg);
  }

  ::data load_next();

  void reset();

  ~csv_loader() { in.close(); }
};
