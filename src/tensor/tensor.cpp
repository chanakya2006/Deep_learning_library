#include "tensor/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <cstdint>

using namespace std;

Tensor::Tensor(const tensor_shape &input_shape,
               const tensor_data &input_data,
               grad_flag input_grad_flag)
    : shape(input_shape.shape),
      data(input_data.data),
      requires_grad(input_grad_flag.flag) {

  uint_least64_t expected_size = 1;

  for (auto dimension : shape) 
    expected_size *= dimension;

  if (!data.empty()) {
    if (data.size() != expected_size)
      throw runtime_error("Constructor: shape/data mismatch");
  } else {
    data.resize(expected_size, 0.0f);
  }

  if (requires_grad)
    grad.resize(expected_size, 0.0f);
}

Tensor::Tensor(const vector<int> &input_shape,
               const vector<float> &input_data,
               bool grad_flag)
    : shape(input_shape),
      data(input_data),
      requires_grad(grad_flag) {

  uint_least64_t expected_size = 1;
  
  for (auto dimension : shape) 
    expected_size *= dimension;

  if (!data.empty()) {
    if (data.size() != expected_size)
      throw runtime_error("Constructor: shape/data mismatch");
  } else {
    data.resize(expected_size, 0.0f);
  }

  if (requires_grad)
    grad.resize(expected_size, 0.0f);
}

vector<int> Tensor::get_shape() { return shape; }
vector<float> Tensor::get_data() { return data; }
vector<float> *Tensor::get_data_pointer() { return &data; }

int Tensor::flatten(const vector<int> &indices) const {
  if (indices.size() != shape.size())
    throw runtime_error("flatten: dimension mismatch");

  int flat_index = 0;
  int stride = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    if (indices[i] < 0 || indices[i] >= shape[i])
      throw runtime_error("flatten: index out of bounds");
    flat_index += indices[i] * stride;
    stride *= shape[i];
  }
  return flat_index;
}

float &Tensor::operator()(vector<int> idx) {
  return data[flatten(vector<int>(idx))];
}

float Tensor::operator()(vector<int> idx) const {
  return data[flatten(vector<int>(idx))];
}

void Tensor::build_topo(Tensor *t, vector<Tensor *> &topo,
                        unordered_set<Tensor *> &visited) {
  if (visited.count(t)) return;
  visited.insert(t);
  for (auto parents : t->parents)
    build_topo(parents, topo, visited);
  topo.push_back(t);
}

void Tensor::backward() {
  // Only scalar tensors should start backprop
  if (data.size() != 1)
    throw runtime_error("backward() can only be called on Tensor with scalar "
                        "value, use .sum() first ");

  grad[0] = 1.0f;
  vector<Tensor *> topo;
  unordered_set<Tensor *> visited;
  build_topo(this, topo, visited);

  reverse(topo.begin(), topo.end());

  for (Tensor *t : topo)
    if (t->backward_fn)
      t->backward_fn(*t);
}

void Tensor::zero_grad() {
  if (requires_grad)
    fill(grad.begin(), grad.end(), 0.0f);
}

Tensor::~Tensor() {}