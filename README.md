# Deep_learning_library

## 1. Tensor operations
 
1.1 Transpose (2D Tensor):

```

Tensor t(tensor_shape{{2,3}},tensor_data{{1,2,3,4,5,6}});

Tensor t1 = t.transpose();

```

1.2 Matrix multiplication (2D):

```

Tensor t(tensor_shape{{3,3}}, tensor_data{{1,2,3,4,5,6,7,8,9}});
Tensor t1(tensor_shape{{3,2}}, tensor_data{{1,2,3,4,5,6}});

Tensor t2 = t.matmul_2D(t1);

```

1.3 Scalar Multiplication:

```

Tensor t1(tensor_shape{{2,2}}, tensor_data{{1,2,3,4}});
Tensor t2 = t1 * 3.0f;

```

## 2. Automatic Differentiation

2.1 Topological Graph build:

```

Tensor t1{{2,2},{1,2,3,4},true};
Tensor t2{{2,2},{1,2,3,4},true};

Tensor t3 = t1 + t2;
Tensor t4 = t3 * 3.0f;
Tensor t5{{2,2},{1,2,3,4},true};
Tensor t6 = t4 * t5;

vector<Tensor*> topo;
unordered_set<Tensor*> visited;

build_topo(&t6, topo, visited);
reverse(topo.begin(), topo.end());

```

2.2 Gradient Example :

``` 

Tensor t1{{3},{1,2,3},true};
Tensor t2 = t1 * t1;
Tensor t3 = t2.sum();

t3.backward();

```

## 3. Matrix Multiplication Gradients

3.1 Square Matrix Case:

Forward :
```
@ -> matmul

A (2x2)
B (2x2)

Z = A @ B -> (2x2)
```

Backpropagation :

```
Suppose dZ are the gradients of the Z Tensor

dA = dZ @ transpose(B)
dB = transpose(A) @ dZ

where dA and dB are the gradients of Tensor and A and B respectively.   
```

```
 
Tensor A{{2,2},{1,2,3,4},true};
Tensor B{{2,2},{5,6,7,8},true};

Tensor Z = A.matmul_2D(B);
Tensor Z_temp = Z.sum();

Z_temp.backward();

```

3.2 Non - Square Case:

```
Tensor A{{2,3},{1,0,2,-1,3,1},true};
Tensor B{{3,2},{2,1,0,-1,3,4},true};

Tensor Z = A.matmul_2D(B);
Tensor Z_temp = Z.sum();

Z_temp.backward();
```

## 4. Saving and Loading

4.1 Tensor

```
  // Saving a Tensor to a binary file
  Tensor t({2, 2}, {2, 3, 4, 5}, true);

  std::ofstream out("temp.dat", std::ios::binary);

  t.save_tensor(out);

  out.close();

  // Loading the Tensor from the same file
  std::ifstream in("temp.dat", std::ios::binary);

  Tensor t_loaded = load_tensor(in);

  in.close();
```

4.2 Layers

```
  Dense dense{4, 5};
  ReLU relu;
  Sigmoid sig;
  LeakyReLU leakyrelu{0.2};
  Tanh tanh;

  // Writing to file
  std::ofstream out("temp.dat", std::ios::binary);

  dense.save_layer(out);
  relu.save_layer(out);
  sig.save_layer(out);
  leakyrelu.save_layer(out);
  tanh.save_layer(out);

  out.close();

  // Reading from file
  std::ifstream in("temp.dat", std::ios::binary);

  unique_ptr<Layer> dense_copy = load_layer(in);
  unique_ptr<Layer> relu_copy = load_layer(in);
  unique_ptr<Layer> sig_copy = load_layer(in);
  unique_ptr<Layer> leakyrelu_copy = load_layer(in);
  unique_ptr<Layer> tanh_copy = load_layer(in);

  in.close();
```
## 5. File Format Specification

Sequential Model format

```
number_of_layers (size_t)
+ each_layer.save_layer()
```

Layer format

```
layer_tag (char array)
+ optional tensor data
```

Special case :
* LeakyReLU  =>  stores + tag

Tensor Format

```
size_of_shape (size_t)
shape_data (vector<int>)
size_of_data (size_t)
data (vector<float>)
requires_grad (bool)
```

## XOR example with model saving and loading and with csv file loader :

```
In XOR.csv:

0,0,0
0,1,1
1,0,1
1,1,0

```

```
  Sequential model;

  model.add(make_unique<Dense>(2, 4));
  model.add(make_unique<Tanh>());
  model.add(make_unique<Dense>(4, 1));
  model.add(make_unique<Sigmoid>());

  vector<Tensor *> parameters = model.get_parameters();

  Binary_cross_entropy bce;

  Adam adam;

  csv_loader loader("XOR.csv", 2);

  for (int epoch = 0; epoch < 10000; epoch++) {
    float epoch_loss = 0;

    for (size_t i = 0; i < loader.num_of_rows; i++) {

      ::data input_and_pred = loader.load_next();

      Tensor x({1, 2}, input_and_pred.input, false);
      Tensor y_true({1, 1}, input_and_pred.output, false);

      Tensor y_pred = model.forward(x);

      Tensor loss = bce.apply(y_pred, y_true);
      epoch_loss += loss.get_data()[0];

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
```

## MNIST :

```
string path_to_model = "trained_models/MNIST_sequential.dat";
string path_to_dataset = "datasets/mnist_train.csv";
bool use_saved_model = false;

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
  Sequential model;

  if (use_saved_model) {
    model = load_Sequential_model(path_to_model);
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

  csv_loader loader(path_to_dataset, 2);

  for (int epoch = 0; epoch < 2; epoch++) {
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

    cout << "Epoch " << epoch << " loss = " << epoch_loss / loader.num_of_rows
         << endl;
  }

  // Saving model
  model.save(path_to_model);
}

float get_accuracy() {
  size_t sample_size = 6000;

  Sequential model = load_Sequential_model(path_to_model);

  csv_loader loader(path_to_dataset, 2);

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

  // To download the mnist_train.csv dataset please visit
  // https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_train.csv

  // NOTE : Please be sure to remove the first row in the datatset

  train();

  float accuracy = get_accuracy();
  cout << "accuracy :  " << accuracy << endl;

  return 0;
}
```
