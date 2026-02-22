#include "loss/bce.hpp"
#include "loss/mce.hpp"
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
  float max_value = 0;
  int max_value_at = 0;
  for (size_t i = 0; i < encoding.size(); i++) {
    if (encoding[i] > max_value) {
      max_value = encoding[i];
      max_value_at = i;
    }
  }
  return max_value_at;
}

void train() {
  bool is_model_saved = false;

  Sequential model;

  if (is_model_saved) {
    model = load_Sequential_model("temp.dat");
  } else {
    model.add(make_unique<Dense>(783, 256));
    model.add(make_unique<Tanh>());
    model.add(make_unique<Dense>(256, 128));
    model.add(make_unique<Tanh>());
    model.add(make_unique<Dense>(128, 10));
    model.add(make_unique<Softmax>());
  }

  vector<Tensor *> parameters = model.get_parameters();

  MCE mce(10);

  Adam adam;

  csv_loader loader("mnist_train.csv", 2);

  for (int epoch = 0; epoch < 1; epoch++) {
    float epoch_loss = 0;

    for (size_t i = 0; i < loader.num_of_rows; i++) {

      ::data output_and_input = loader.load_next();
      ::data input_and_output{output_and_input.output,
                              label_to_encoding(output_and_input.input[0])};

      Tensor x({1, int(input_and_output.input.size())}, input_and_output.input,
               false);
      Tensor y_true({1, int(input_and_output.output.size())},
                    input_and_output.output, false);

      Tensor y_pred = model.forward(x);

      Tensor loss = mce.apply(y_pred, y_true);
      epoch_loss += loss.get_data()[0];

      cout << "Row : " << i << "\tThe loss was : " << loss.get_data()[0]
           << endl;

      loss.backward();
      adam.step(parameters);
      adam.zero_grad(parameters);
    }

    loader.reset();

    if (epoch % 200 == 0) {
      cout << "Epoch " << epoch << " loss = " << epoch_loss / loader.num_of_rows
           << endl;
    }
  }

  // Saving model
  model.save("temp.dat");
}

float get_accuracy() {
  size_t sample_size = 6000;

  Sequential model = load_Sequential_model("temp.dat");

  csv_loader loader("mnist_train.csv", 2);

  float correct = 0;

  for (size_t i = 0; i < sample_size; i++) {
    ::data output_and_input = loader.load_next();
    ::data input_and_output{output_and_input.output,
                            label_to_encoding(output_and_input.input[0])};

    Tensor x{
        {1, int(input_and_output.input.size())}, input_and_output.input, false};
    Tensor y_pred = model.forward(x);

    int pred_num = encoding_to_label(y_pred.get_data());
    int true_num = output_and_input.input[0];

    if (pred_num == true_num) {
      correct++;
    }
  }
  loader.reset();

  return float(correct / sample_size) * 100;
}

int main() {
  // train();

  float accuracy = get_accuracy();
  cout << "accuracy :  " << accuracy << endl;

  return 0;
}

// Implement :
// RSMprop

// loading functions of model, layers, and tensors are defined in
// helpers.hpp and Implementend in helpers.cpp
//
// If needed move them to appropriate files
