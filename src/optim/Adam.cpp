#include "optim/Adam.hpp"
#include "tensor/tensor.hpp"
#include <cmath>
#include <cstddef>
#include <vector>

Adam::Adam(float lr_, float b1_, float b2_, float eps_)
    : lr(lr_), b1(b1_), b2(b2_), eps(eps_) {};

// For more information:
// https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam

void Adam::step(vector<Tensor *> &parameters) {
  iterations++;

  for (Tensor *param : parameters) {

    vector<float> &moment = moments[param];
    vector<float> &velocity = velocities[param];

    if (moment.empty()) {
      moment.resize(param->grad.size(), 0.0f);
      velocity.resize(param->grad.size(), 0.0f);
    }

    vector<float> &param_data_pointer = *param->get_data_pointer();
    vector<float> &param_grad = param->grad;

    for (size_t i = 0; i < param_data_pointer.size(); i++) {
      moment[i] = (b1 * moment[i]) + (1 - b1) * (param_grad[i]);
      velocity[i] =
          (b2 * velocity[i]) + (1 - b2) * (param_grad[i]) * (param_grad[i]);

      float m_hat = moment[i] / (1 - pow(b1, iterations));
      float v_hat = velocity[i] / (1 - pow(b2, iterations));

      param_data_pointer[i] -= (lr * (m_hat / (pow(v_hat, 0.5) + eps)));
    }
  }
};
