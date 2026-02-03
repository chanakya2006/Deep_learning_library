#include "loss/bce.hpp"
#include "nn/activations.hpp"
#include "nn/dense.hpp"
#include "optim/sgd.hpp"
#include "optim/sgdmomentum.hpp"
#include "utils/helpers.hpp"
#include <iostream>

using namespace std;

int main() {
  vector<vector<float>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  vector<vector<float>> Y = {{0}, {1}, {1}, {0}};

  Dense fc1(2, 4);
  Sigmoid sig1;
  Dense fc2(4, 1);
  Sigmoid sig2;

  vector<Layer *> model = {&fc1, &fc2};
  vector<Tensor *> parameters = get_all_parameters(model);

  SGDMomentum sgdmomentum(1e-3f);

  Binary_cross_entropy bce;

  for (int epoch = 0; epoch < 1392200; epoch++) {
    float epoch_loss = 0;

    for (int i = 0; i < 4; i++) {
      Tensor x({1, 2}, X[i], false);
      Tensor y_true({1, 1}, Y[i], false);

      Tensor y = fc1.forward(x);
      Tensor y1 = sig1.forward(y);
      Tensor y2 = fc2.forward(y1);
      Tensor y3 = sig2.forward(y2);

      Tensor loss = bce.apply(y3, y_true);
      epoch_loss += loss.get_data()[0];

      loss.backward();
      sgdmomentum.step(parameters);
      sgdmomentum.zero_grad(parameters);
    }

    if (epoch % 200 == 0)
      cout << "Epoch " << epoch << " loss = " << epoch_loss / 4 << endl;
  }

  // Model prediction for [0,0]
  Tensor x({1, 2}, {0, 0}, false);

  Tensor y = fc1.forward(x);
  Tensor y1 = sig1.forward(y);
  Tensor y2 = fc2.forward(y1);
  Tensor y3 = sig2.forward(y2);

  cout << "[0,0] -> " << y3.get_data()[0] << " -> ~0" << endl;

  // Model prediction for [0,1]
  x = Tensor({1, 2}, {0, 1}, false);

  y = fc1.forward(x);
  y1 = sig1.forward(y);
  y2 = fc2.forward(y1);
  y3 = sig2.forward(y2);

  cout << "[0,1] -> " << y3.get_data()[0] << " -> ~1" << endl;

  // Model prediction for [1,0]
  x = Tensor({1, 2}, {1, 0}, false);

  y = fc1.forward(x);
  y1 = sig1.forward(y);
  y2 = fc2.forward(y1);
  y3 = sig2.forward(y2);

  cout << "[1,0] -> " << y3.get_data()[0] << " -> ~1" << endl;

  // Model prediction for [1,1]
  x = Tensor({1, 2}, {1, 1}, false);

  y = fc1.forward(x);
  y1 = sig1.forward(y);
  y2 = fc2.forward(y1);
  y3 = sig2.forward(y2);

  cout << "[1,1] -> " << y3.get_data()[0] << " -> ~0" << endl;
  return 0;
}

// I have set bias to random values instead of 0 revert this change,
// if needed

// Add the ability to save and load models from files.

// Arrange all of the classes in proper file structure.

// Implement :
// SGD + Momentum
// ADAM
// Leaky ReLU
// Tanh

// also add runtime_error when you call Binary_cross_entropy when you pass a
// Tensor of more than size 1.

// in layer's the Tensor is begin returned as copy
// instead return the pointer to the answer which
// is being stored as a member variable.
