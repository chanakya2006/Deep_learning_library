#include "nn/dense.hpp"
#include <cstdlib>

using namespace std;

Dense::Dense(int in_features, int out_features)
    : W({in_features, out_features}, {}, true),
      b({1, out_features}, {}, true),
      input_matmul_W({1, out_features}, {}, true),
      W_plus_bias({1, out_features}, {}, true) {

  for (float &x : *W.get_data_pointer())
    x = ((float)rand() / RAND_MAX) - 0.5f;
  for (float &x : *b.get_data_pointer())
    x = ((float)rand() / RAND_MAX) - 0.5f;
}

Tensor Dense::forward(Tensor &input) {
  input_matmul_W = input.matmul_2D(W);
  W_plus_bias = input_matmul_W + b;
  return W_plus_bias;
}

vector<Tensor *> Dense::parameters() {
  return {&W, &b};
}
