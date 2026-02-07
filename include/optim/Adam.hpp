#pragma once

#include "optim/optimizer.hpp"
#include <unordered_map>
#include <vector>

using namespace std;

class Adam : public Optimizer {
private:
  float lr, b1, b2, eps;
  unordered_map<Tensor *, vector<float>> moments, velocities;
  int iterations = 0;

public:
  explicit Adam(float lr_ = 0.001f, float b1_ = 0.9f, float b2_ = 0.999f,
                float eps_ = 1e-8f);
  void step(vector<Tensor *> &parameters) override;
};
