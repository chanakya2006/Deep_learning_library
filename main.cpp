#include "loss/bce.hpp"
#include "model/sequential.hpp"
#include "nn/activations.hpp"
#include "nn/dense.hpp"
#include "nn/layer.hpp"
#include "optim/Adam.hpp"
#include "tensor/tensor.hpp"
#include "utils/helpers.hpp"
#include <climits>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

using namespace std;

int main() {

  Sequential model;
  model.add(make_unique<Dense>(2, 4));
  model.add(make_unique<Tanh>());
  model.add(make_unique<Dense>(4, 1));
  model.add(make_unique<Sigmoid>());

  vector<Tensor *> parameters = model.get_parameters();

  Binary_cross_entropy bce;

  Adam adam;

  vector<vector<float>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  vector<vector<float>> Y = {{0}, {1}, {1}, {}};

  for (int epoch = 0; epoch < 10000; epoch++) {
    float epoch_loss = 0;

    for (int i = 0; i < 4; i++) {
      Tensor x({1, 2}, X[i], false);
      Tensor y_true({1, 1}, Y[i], false);

      Tensor y_pred = model.forward(x);

      Tensor loss = bce.apply(y_pred, y_true);
      epoch_loss += loss.get_data()[0];

      loss.backward();
      adam.step(parameters);
      adam.zero_grad(parameters);
    }

    if (epoch % 200 == 0)
      cout << "Epoch " << epoch << " loss = " << epoch_loss / 4 << endl;
  }

  // Saving model
  model.save("temp.dat");

  // Loading the same model from file
  Sequential model_copy = load_Sequential_model("temp.dat");

  // Model prediction for [0,0]
  Tensor x({1, 2}, {0, 0}, false);

  Tensor y_pred = model_copy.forward(x);

  cout << "[0,0] -> " << y_pred.get_data()[0] << " -> ~0" << endl;

  // Model prediction for [0,1]
  x = Tensor({1, 2}, {0, 1}, false);

  y_pred = model_copy.forward(x);

  cout << "[0,1] -> " << y_pred.get_data()[0] << " -> ~1" << endl;

  // Model prediction for [1,0]
  x = Tensor({1, 2}, {1, 0}, false);

  y_pred = model_copy.forward(x);

  cout << "[1,0] -> " << y_pred.get_data()[0] << " -> ~1" << endl;

  // Model prediction for [1,1]
  x = Tensor({1, 2}, {1, 1}, false);

  y_pred = model_copy.forward(x);

  cout << "[1,1] -> " << y_pred.get_data()[0] << " -> ~0" << endl;

  return 0;
}

// As each layer has different number of tensor we must know beforehand what
// type layer we are going to read, so when saving the layer I write the type of
// layer it is (tag) and when reading i first check which type of layer it is
// then allocate tensors appropriately

// Implement :
// RSMprop
// Save and load model from a raw binary file

// loading functions of model, layers, and tensors are defined in helpers.hpp
// and Implementend in helpers.cpp
//
// If needed move them to appropriate files
