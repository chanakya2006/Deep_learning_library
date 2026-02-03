#include "optim/sgdmomentum.hpp"
#include "tensor/tensor.hpp"
#include <cstddef>
#include <vector>

SGDMomentum::SGDMomentum(float lr_, float beta_) : lr(lr_), beta(beta_) {};

// v = βv + ∇θ

// θ −= ηv

// or in simpler terms : θ := θ - η∇θ - ηβv

// For more info on SGDMomentum :

// https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum

void SGDMomentum::step(vector<Tensor *> &parameters) {
  for (Tensor *param : parameters) {
    vector<float> &velocity = velocities[param];
    if (velocity.empty()) {
      velocity.resize(param->grad.size(), 0.0f);
    }

    vector<float> &para_data_poitner = *param->get_data_pointer();
    vector<float> &param_grad = param->grad;

    for (size_t i = 0; i < para_data_poitner.size(); i++) {
      velocity[i] = beta * velocity[i] + param_grad[i];
      para_data_poitner[i] -= lr * velocity[i];
    }
  }
}
