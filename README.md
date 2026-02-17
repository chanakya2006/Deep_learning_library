# Deep_learning_library

To transpose a 2 dimensional tensor (Matrix):

    Tensor t(tensor_shape{{2,3}},tensor_data{{1,2,3,4,5,6}});
    std::cout << t({0,0}) << " " << t({0,1}) << " " << t({0,2}) << std::endl;
    std::cout << t({1,0}) << " " << t({1,1}) << " " << t({1,2}) << std::endl;

    std::cout << "\n";

    Tensor t1 = t.transpose();
    std::cout << t1({0,0}) << " " << t1({0,1}) << std::endl;
    std::cout << t1({1,0}) << " " << t1({1,1}) << std::endl;
    std::cout << t1({2,0}) << " " << t1({2,1}) << std::endl;


To matrix multiply 2 two dimensional Tensor (Matrix):

    Tensor t(tensor_shape{{3,3}},tensor_data{{1,2,3,4,5,6,7,8,9}});
    std::cout << t({0,0}) << " " << t({0,1}) << " " << t({0,2}) << std::endl;
    std::cout << t({1,0}) << " " << t({1,1}) << " " << t({1,2}) << std::endl;
    std::cout << t({2,0}) << " " << t({2,1}) << " " << t({2,2}) << std::endl;

    std::cout << "\n";

    Tensor t1(tensor_shape{{3,2}},tensor_data{{1,2,3,4,5,6}});
    std::cout << t1({0,0}) << " " << t1({0,1}) << std::endl;
    std::cout << t1({1,0}) << " " << t1({1,1}) << std::endl;
    std::cout << t1({2,0}) << " " << t1({2,1}) << std::endl;
    
    std::cout << "\n";

    Tensor t2 = t.matmul_2D(t1);
    std::cout << t2({0,0}) << " " << t2({0,1}) << std::endl;
    std::cout << t2({1,0}) << " " << t2({1,1}) << std::endl;
    std::cout << t2({2,0}) << " " << t2({2,1}) << std::endl;


Scalar Multiplication:

    Tensor t1(tensor_shape{{2,2}},tensor_data{{1,2,3,4}});

    std::cout << t1({0,0}) << " " << t1({0,1}) << std::endl;
    std::cout << t1({1,0}) << " " << t1({1,1}) << std::endl;

    t1 = t1 * 3.0f;

    std::cout << "\n";
    std::cout << t1({0,0}) << " " << t1({0,1}) << std::endl;
    std::cout << t1({1,0}) << " " << t1({1,1}) << std::endl;

build_topo in action :

    Tensor t1{tensor_shape{{2,2}}, tensor_data{{1,2,3,4}},grad_flag{true}};
    Tensor t2{tensor_shape{{2,2}}, tensor_data{{1,2,3,4}},grad_flag{true}};
    Tensor t3 = t1 + t2;
    Tensor t4 = t3 * 3.0f;
    Tensor t5{tensor_shape{{2,2}}, tensor_data{{1,2,3,4}},grad_flag{true}};
    Tensor t6 = t4 * t5;
    std::cout << "Address of t1 : " << &t1 << std::endl;
    std::cout << "Address of t2 : " << &t2 << std::endl;
    std::cout << "Address of t3 : " << &t3 << std::endl;
    std::cout << "Address of t4 : " << &t4 << std::endl;
    std::cout << "Address of t5 : " << &t5 << std::endl;
    std::cout << "Address of t6 : " << &t6 << std::endl;

    std::vector<Tensor*> topo;
    std::unordered_set<Tensor*> visited;
    build_topo(&t6, topo, visited);
    std::reverse(topo.begin(), topo.end());
    for(auto x : topo){
        std::cout << x << " " << std::endl;
    }


calculating gradient :

    Tensor t{{3},{1,2,3},true};
    Tensor t_half = t * t;
    Tensor t1 = t_half.sum();
    cout << "\n\n";
    cout << "t address : " << &t << endl;
    cout << "t_half address : " << &t_half << endl;
    cout << "t1 address : " << &t1 << endl;
    cout << "t1 parents address : " << t1.parents[0] << endl;
    cout << "t_half parents address : ";
    for(Tensor* parent : t_half.parents) cout << parent << " ";
    cout << "\n\n "<<endl;
    t1.backward();
    cout << "\n\ngrad values of t_half are : ";
    for(auto x : t_half.grad){
        cout << x << " ";
    }
    cout << "\n";
    cout << "grad values of t are : ";
    for(auto x : t.grad){
        std::cout << x << " "; 
    }

matmul grad working (square matrix): 

    Tensor A{ tensor_shape{{2,2}}, tensor_data{{1,2,3,4}}, grad_flag{true} };
    Tensor B{ tensor_shape{{2,2}}, tensor_data{{5,6,7,8}}, grad_flag{true} };

    Tensor Z = A.matmul_2D(B);
    Tensor Z_temp = Z.sum();
    Z_temp.backward();

    for(auto x : A.grad){
        cout << x << " ";
    }
    cout << endl;
    for(auto x : B.grad){
        cout << x << " ";
    }
    cout << endl;

matmul grad working ( non - square matrix):

```
A = [1  0  2               A = 2 x 3
    -1  3  1]

B = [2  1                  B = 3 x 2
     0 -1
     3  4]

Z = A @ B                  Z = 2 x 2

```
When backpropagation begins : 
```
dZ = [1  1
      1  1]


dA = dZ @ transpose_2D(B)

dB = transpose_2D(A) @ dZ

dA = [3 -1  7
      3 -1  7]

dB = [0  0
      3  3
      3  3]
```

```
    Tensor A{ tensor_shape{{2,3}}, tensor_data{{1,0,2,-1,3,1}}, grad_flag{true} };
    Tensor B{ tensor_shape{{3,2}}, tensor_data{{2,1,0,-1,3,4}}, grad_flag{true} };

    Tensor Z = A.matmul_2D(B);

    cout << "Z : ";
    for(float x : Z.get_data()){
        cout << x << " ";
    }
    cout << endl;

    Tensor Z_temp = Z.sum();
    Z_temp.backward();

    cout << "A.grad : ";
    for(auto x : A.grad){
        cout << x << " ";
    }
    cout << endl;
    
    cout << "B.grad : ";
    for(auto x : B.grad){
        cout << x << " ";
    }
    cout << endl;
```

Dense layer basic example : 

```
    Tensor input({1,3},{1,2,3},true);
    Dense D1(3,3);

    Tensor out = D1.forward(input);

    for(auto x : out.get_data()) cout << x << " ";
```

Tensor relu() example :
```
  Tensor t{{2, 2}, {-2, 20, 0.2, -100}, true};
  Tensor t1 = t.relu();

  cout << "Address of t : " << &t << endl;
  cout << "Parents of t1 : ";
  for (Tensor *parent : t1.parents)
    cout << parent << endl;
  cout << endl;

  cout << "data of t1 : ";
  for (float x : t1.get_data()) {
    cout << x << " ";
  }
  cout << endl;

  t1.sum().backward();

  cout << "gradeints of t : ";
  for (float grads : t.grad)
    cout << grads << " ";
  cout << endl;
```

Simple single loop model :

```
  Tensor x({1, 2}, {1.0f, 2.0f}, true);

  Dense fc1(2, 3);
  ReLU relu1;
  Dense fc2(3, 1);
  Sigmoid sig1;

  Tensor y = fc1.forward(x);
  Tensor y1 = relu1.forward(y);
  Tensor y2 = fc2.forward(y1);
  Tensor y3 = sig1.forward(y2);

  Tensor loss = y3.sum();
  loss.backward();
  
  x.zero_grad();
```

XOR problem prediction :

```
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
```

XOR prediction using model class :

```
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
  // Model prediction for [0,0]
  Tensor x({1, 2}, {0, 0}, false);

  Tensor y_pred = model.forward(x);

  cout << "[0,0] -> " << y_pred.get_data()[0] << " -> ~0" << endl;

  // Model prediction for [0,1]
  x = Tensor({1, 2}, {0, 1}, false);

  y_pred = model.forward(x);

  cout << "[0,1] -> " << y_pred.get_data()[0] << " -> ~1" << endl;

  // Model prediction for [1,0]
  x = Tensor({1, 2}, {1, 0}, false);

  y_pred = model.forward(x);

  cout << "[1,0] -> " << y_pred.get_data()[0] << " -> ~1" << endl;

  // Model prediction for [1,1]
  x = Tensor({1, 2}, {1, 1}, false);

  y_pred = model.forward(x);
  
  cout << "[1,1] -> " << y_pred.get_data()[0] << " -> ~0" << endl;

```

Saving and loading a Tensor from file

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

Saving and loading layers from a binary file 

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

XOR example using loading and saving model from binary

```
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

  // Loading the same model from file and predicting using it
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
