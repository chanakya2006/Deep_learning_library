#pragma once
#include "loss.hpp"

class Binary_cross_entropy : public Loss_function {
private:
  Tensor one{{1, 1}, {1.0f}, false};
  Tensor pred_log{{1, 1}, {}, true};
  Tensor term1{{1, 1}, {}, true};
  Tensor one_minus_target{{1, 1}, {}, true};
  Tensor one_minus_pred{{1, 1}, {}, true};
  Tensor one_minus_pred_log{{1, 1}, {}, true};
  Tensor term2{{1, 1}, {}, true};
  Tensor term1_plus_term2{{1, 1}, {}, true};
  Tensor sum{{1, 1}, {}, true};
  Tensor out{{1, 1}, {}, true};

public:
  // Binary_cross_entropy();
  Tensor apply(Tensor &pred, Tensor &target) override;
};
