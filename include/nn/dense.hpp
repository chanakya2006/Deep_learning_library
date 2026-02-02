#pragma once
#include "layer.hpp"

class Dense : public Layer {
private:
  Tensor W;
  Tensor b;
  Tensor input_matmul_W;
  Tensor W_plus_bias;

public:
  Dense(int in_features, int out_features);
  Tensor forward(Tensor &input) override;
  std::vector<Tensor *> parameters() override;
};
