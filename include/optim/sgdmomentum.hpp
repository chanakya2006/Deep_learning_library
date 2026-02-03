#pragma once

#include "optim/optimizer.hpp"
#include <unordered_map>
#include <vector>

using namespace std;

class SGDMomentum : public Optimizer {
private:
  float lr, beta;
  unordered_map<Tensor *, vector<float>> velocities;

public:
  explicit SGDMomentum(float lr_, float beta_ = 0.9f);
  void step(vector<Tensor *> &parameters) override;
};
