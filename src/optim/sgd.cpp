#include "optim/sgd.hpp"
#include <cstddef>

SGD::SGD(float lr_) : lr(lr_) {}

void SGD::step(std::vector<Tensor *> &parameters) {
  for (Tensor *t : parameters) {
    auto &w = *t->get_data_pointer();
    auto &g = t->grad;
    for (size_t i = 0; i < w.size(); i++)
      w[i] -= lr * g[i];
  }
}
