#include "loss/bce.hpp"
#include "loss/mse.hpp"
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
#include <ratio>
#include <vector>

using namespace std;

vector<float> label_to_encoding(float label) {
  vector<float> out(10, 0);
  out[label] = 1;
  return out;
}

int encoding_to_label(vector<float> encoding) {
  for (size_t i = 0; i < encoding.size(); i++) {
    if (encoding[i] == 1)
      return i;
  }
  return 0;
}

void train() {
  Sequential model;

  model.add(make_unique<Dense>(783, 256));
  model.add(make_unique<LeakyReLU>(0.1));
  model.add(make_unique<Dense>(256, 128));
  model.add(make_unique<LeakyReLU>(0.1));
  model.add(make_unique<Dense>(128, 10));
  model.add(make_unique<LeakyReLU>(0.1));
  model.add(make_unique<Softmax>());

  vector<Tensor *> parameters = model.get_parameters();

  MSE mse(10);

  Adam adam;

  csv_loader loader("mnist_train.csv", 2);

  for (int epoch = 0; epoch < 1; epoch++) {
    float epoch_loss = 0;

    for (size_t i = 0; i < loader.num_of_rows / 60; i++) {

      ::data pred_and_input = loader.load_next();
      ::data input_and_pred{pred_and_input.output,
                            label_to_encoding(pred_and_input.input[0])};

      Tensor x({1, int(input_and_pred.input.size())}, input_and_pred.input,
               false);
      Tensor y_true({1, int(input_and_pred.output.size())},
                    input_and_pred.output, false);

      Tensor y_pred = model.forward(x);

      Tensor loss = mse.apply(y_pred, y_true);
      epoch_loss += loss.get_data()[0];

      cout << "It ran :) " << i << endl;

      loss.backward();
      adam.step(parameters);
      adam.zero_grad(parameters);
    }

    loader.reset();

    if (epoch % 200 == 0) {
      cout << "Epoch " << epoch
           << " loss = " << int(epoch_loss / (loader.num_of_rows / 60)) << endl;
    }
  }

  // Saving model
  // model.save("temp.dat");
}

void predict() {
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
}

int main() {
  train();
  // predict();
  return 0;
}

// Implement :
// RSMprop

// loading functions of model, layers, and tensors are defined in
// helpers.hpp and Implementend in helpers.cpp
//
// If needed move them to appropriate files
