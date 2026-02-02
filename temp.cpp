#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <iterator>
#include <mutex>
#include <numeric>
#include <ostream>
#include <ratio>
#include <type_traits>
#include <unordered_set>
#include <vector>

using namespace std;

struct tensor_shape {
  vector<int> shape;
};

struct tensor_data {
  vector<float> data;
};

struct grad_flag {
  bool flag;
};

class Tensor {
private:
  vector<int> shape;
  vector<float> data;

  bool requires_grad = false;

public:
  vector<float> grad;
  vector<Tensor *> parents;
  function<void(Tensor &)> backward_fn;

  Tensor(const tensor_shape &input_shape,
         const tensor_data &input_data = tensor_data{{}},
         grad_flag input_grad_flag = grad_flag{false})
      : shape(input_shape.shape), data(input_data.data),
        requires_grad(input_grad_flag.flag) {
    uint_least64_t expected_size = 1;
    for (auto dimension : shape)
      expected_size *= dimension;

    if (data.size()) {
      if (data.size() != expected_size)
        throw runtime_error(
            "Strong type Constructor : Mismatch in input shape and input data");
    } else {
      data.resize(expected_size, 0.0f);
    }
    if (requires_grad)
      grad.resize(expected_size, 0.0f);
  }

  Tensor(const vector<int> &input_shape, const vector<float> &input_data = {},
         bool grad_flag = false)
      : shape(input_shape), data(input_data), requires_grad(grad_flag) {
    uint_least64_t expected_size = 1;
    for (auto dimension : shape)
      expected_size *= dimension;

    if (data.size()) {
      if (data.size() != expected_size)
        throw runtime_error(
            "Weak type Constructor : Mismatch in input shape and input data");
    } else {
      data.resize(expected_size, 0.0f);
    }
    if (requires_grad)
      grad.resize(expected_size, 0.0f);
  }

  vector<int> get_shape() { return shape; }

  vector<float> get_data() { return this->data; }

  vector<float> *get_data_pointer() { return &this->data; }

  Tensor matmul_2D(Tensor &other) {
    if ((shape.size() != 2) || (other.shape.size() != 2))
      throw runtime_error("matmul_2D : This function is to be only used when "
                          "the tensors are 2 Dimensional.");
    if (shape[1] != other.shape[0])
      throw runtime_error("matmul_2D : The Number of rows in the first tensor "
                          "is not equal to the columns of the second tensor.");

    int matrix_1_columns = shape[0], matrix_1_rows = shape[1],
        matrix_2_rows = other.shape[1];
    vector<float> ans_data(matrix_1_columns * matrix_2_rows);
    // matrix_1 = m x n
    // matrix_2 = n x k

    for (int m = 0; m < matrix_1_columns; m++) {
      for (int k = 0; k < matrix_2_rows; k++) {
        float sum = 0;
        for (int n = 0; n < matrix_1_rows; n++) {
          sum += this->data[m * matrix_1_rows + n] *
                 other.data[n * matrix_2_rows + k];
        }
        ans_data[m * matrix_2_rows + k] = sum;
      }
    }
    Tensor ans{tensor_shape{{shape[0], other.shape[1]}}, tensor_data{ans_data},
               grad_flag{this->requires_grad || other.requires_grad}};

    if (this->requires_grad || other.requires_grad) {
      ans.parents = {this, &other};

      ans.backward_fn = [](Tensor &self) {
        // Z = A @ B;
        // dA = dZ @ B.transpose_2D()
        // dB = A.transpose_2D() @ dZ

        // Z = A.matmul(B);
        // dA = dZ.matmul(B.transpose_2D());
        // dB = A.transpose_2D().matmul(dZ);

        Tensor &A = *self.parents[0];
        Tensor &B = *self.parents[1];

        Tensor dZ{tensor_shape{{A.shape[0], B.shape[1]}},
                  tensor_data{self.grad}};
        if (A.requires_grad) {
          Tensor other_transpose = B.transpose_2D();
          Tensor dA = dZ.matmul_2D(other_transpose);
          vector<float> &dA_data_pointer = *dA.get_data_pointer();
          for (size_t i = 0; i < dA_data_pointer.size(); i++) {
            A.grad[i] += dA_data_pointer[i];
          }
        }
        if (B.requires_grad) {
          Tensor dB = A.transpose_2D().matmul_2D(dZ);
          vector<float> &dB_data_pointer = *dB.get_data_pointer();
          for (size_t i = 0; i < dB_data_pointer.size(); i++) {
            B.grad[i] += dB_data_pointer[i];
          }
        }
      };
    }

    return ans;
  }

  Tensor transpose_2D() const {
    if (shape.size() != 2)
      throw runtime_error("transpose : The tensor should be 2 dimensional.");

    int matrix_row = shape[1], matrix_col = shape[0];
    vector<float> ans_data(matrix_col * matrix_row);
    // matrix = m x n

    for (int n = 0; n < matrix_row; n++) {
      for (int m = 0; m < matrix_col; m++) {
        ans_data[m + n * matrix_col] = data[n + m * matrix_row];
      }
    }

    Tensor ans{tensor_shape{{shape[1], shape[0]}}, tensor_data{ans_data},
               grad_flag{this->requires_grad}};
    return ans;
  }

  // Converts N-dimensional indices to 1D index
  int flatten(const vector<int> &indices) const {
    if (indices.size() != shape.size())
      throw runtime_error("flatten : Index dimensions mismatch");

    int flat_index = 0;
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      if (indices[i] >= shape[i] || indices[i] < 0)
        throw runtime_error("flatten : Index out of bounds");

      flat_index += indices[i] * stride;
      stride *= shape[i];
    }
    return flat_index;
  }

  // operator() for read/write access
  float &operator()(vector<int> idx) { // Used to update the value
    return data[flatten(vector<int>(idx))];
  }

  float operator()(vector<int> idx) const { // Returns Value
    return data[flatten(vector<int>(idx))];
  }

  Tensor operator+(Tensor &other) {
    if (this->shape.size() != other.shape.size())
      throw runtime_error("operator + : Index Dimensions Mismatch");

    for (uint_least64_t i = 0; i < this->shape.size(); i++) {
      if (this->shape[i] != other.shape[i])
        throw runtime_error("operator + : Tensors have same dimensionality but "
                            "different Indexes");
    }

    Tensor ans(tensor_shape{this->shape}, tensor_data{{}},
               grad_flag{this->requires_grad || other.requires_grad});
    for (uint_least64_t i = 0; i < this->data.size(); i++) {
      ans.data[i] = this->data[i] + other.data[i];
    }

    if (this->requires_grad || other.requires_grad) {
      ans.parents = {this, &other};

      ans.backward_fn = [](Tensor &self) {
        Tensor &A = *self.parents[0];
        Tensor &B = *self.parents[1];

        for (uint_least64_t i = 0; i < self.grad.size(); i++) {
          if (A.requires_grad)
            A.grad[i] += self.grad[i];
          if (B.requires_grad)
            B.grad[i] += self.grad[i];
        }
      };
    }

    return ans;
  }

  Tensor operator-(Tensor &other) {
    if (this->shape.size() != other.shape.size())
      throw runtime_error("operator - : Index Dimensions Mismatch");

    for (uint_least64_t i = 0; i < this->shape.size(); i++) {
      if (this->shape[i] != other.shape[i])
        throw runtime_error("operator - : Tensors have same dimensionality but "
                            "different Indexes");
    }

    Tensor ans(tensor_shape{this->shape}, tensor_data{{}},
               grad_flag{this->requires_grad || other.requires_grad});
    for (uint_least64_t i = 0; i < this->data.size(); i++) {
      ans.data[i] = this->data[i] - other.data[i];
    }

    if (requires_grad || other.requires_grad) {
      ans.parents = {this, &other};

      ans.backward_fn = [](Tensor &self) {
        Tensor &A = *self.parents[0];
        Tensor &B = *self.parents[1];

        for (uint_least64_t i = 0; i < self.grad.size(); i++) {
          if (A.requires_grad)
            A.grad[i] += self.grad[i];
          if (B.requires_grad)
            B.grad[i] -= self.grad[i];
        }
      };
    }
    return ans;
  }

  Tensor operator*(float Scalar_value) {
    Tensor ans(tensor_shape{this->shape}, tensor_data{{}},
               grad_flag{this->requires_grad});
    for (uint_least64_t i = 0; i < this->data.size(); i++) {
      ans.data[i] = this->data[i] * Scalar_value;
    }
    if (this->requires_grad) {
      ans.parents = {this};

      ans.backward_fn = [Scalar_value](Tensor &self) {
        Tensor &A = *self.parents[0];

        for (uint_least64_t i = 0; i < self.grad.size(); i++) {
          A.grad[i] += Scalar_value * self.grad[i];
        }
      };
    }
    return ans;
  }

  Tensor operator*(Tensor &other) { // Element wise multiplication , for matrix
                                    // multiplication use matmul
    if (this->shape.size() != other.shape.size())
      throw runtime_error("operator * :Index Dimensions Mismatch");

    for (uint_least64_t i = 0; i < this->shape.size(); i++) {
      if (this->shape[i] != other.shape[i])
        throw runtime_error("operator * : Tensors have same dimensionality but "
                            "different Indexes");
    }

    Tensor ans(tensor_shape{other.shape}, tensor_data{{}},
               grad_flag{this->requires_grad || other.requires_grad});
    for (uint_least64_t i = 0; i < this->data.size(); i++) {
      ans.data[i] = this->data[i] * other.data[i];
    }

    if (this->requires_grad || other.requires_grad) {
      ans.parents = {this, &other};

      ans.backward_fn = [](Tensor &self) {
        Tensor &A = *self.parents[0];
        Tensor &B = *self.parents[1];

        for (uint_least64_t i = 0; i < self.grad.size(); i++) {
          if (A.requires_grad)
            A.grad[i] += B.data[i] * self.grad[i];
          if (B.requires_grad)
            B.grad[i] += A.data[i] * self.grad[i];
        }
      };
    }
    return ans;
  }

  Tensor sum() {
    Tensor out{tensor_shape{{1}}, tensor_data{{}}, grad_flag{requires_grad}};
    out.data[0] = accumulate(data.begin(), data.end(), 0.0f);

    if (requires_grad) {
      out.parents = {this};

      out.backward_fn = [](Tensor &self) {
        Tensor &A = *self.parents[0];

        for (uint_least64_t i = 0; i < A.grad.size(); i++) {
          A.grad[i] += self.grad[0];
        }
      };
    }
    return out;
  }

  void build_topo(Tensor *t, vector<Tensor *> &topo,
                  unordered_set<Tensor *> &visited) {
    if (visited.count(t))
      return;
    visited.insert(t);
    for (auto parents : t->parents) {
      build_topo(parents, topo, visited);
    }
    topo.push_back(t);
  }

  void backward() {
    // Only scalar tensors should start backprop
    if (data.size() != 1) {
      throw runtime_error("backward() can only be called on Tensor with scalar "
                          "value, use .sum() first ");
    }
    grad[0] = 1.0f;
    vector<Tensor *> topo;
    unordered_set<Tensor *> visited;
    build_topo(this, topo, visited);

    reverse(topo.begin(), topo.end());

    for (Tensor *t : topo) {
      if (t->backward_fn) {
        t->backward_fn(*t);
      }
    }
  }

  Tensor relu() {
    Tensor ans{tensor_shape{shape}, tensor_data{{}}, grad_flag{requires_grad}};

    vector<float> &ans_data = *ans.get_data_pointer();

    for (size_t i = 0; i < ans_data.size(); i++) {
      ans_data[i] = max(0.0f, data[i]);
    }

    if (requires_grad) {

      ans.parents = {this};

      ans.backward_fn = [](Tensor &self) {
        Tensor &A = *self.parents[0];

        for (size_t i = 0; i < A.grad.size(); i++) {
          A.grad[i] += (A.data[i] > 0 ? 1.0f : 0.0f) * self.grad[i];
        }
      };
    }

    return ans;
  }

  Tensor sigmoid() {
    Tensor ans(tensor_shape{shape}, tensor_data{{}}, grad_flag{requires_grad});

    vector<float> &ans_data = *ans.get_data_pointer();

    for (size_t i = 0; i < ans_data.size(); i++) {
      ans_data[i] = 1.0f / (1.0f + exp(-data[i]));
    }

    if (requires_grad) {
      ans.parents = {this};

      ans.backward_fn = [](Tensor &self) {
        Tensor &A = *self.parents[0];

        for (size_t i = 0; i < A.grad.size(); i++) {
          float s = self.data[i];
          A.grad[i] += (s * (1 - s)) * self.grad[i];
        }
      };
    }
    return ans;
  }

  void zero_grad() {
    if (requires_grad) {
      fill(grad.begin(), grad.end(), 0.0f);
    }
  }

  Tensor log_on_Tensor() {
    Tensor ans(tensor_shape{shape}, tensor_data{{}}, grad_flag{true});
    vector<float> &ans_data_pointer = *ans.get_data_pointer();
    for (uint_least64_t i = 0; i < ans_data_pointer.size(); i++) {
      ans_data_pointer[i] = log(data[i] + 1e-8f);
    }

    if (requires_grad) {
      ans.parents = {this};

      ans.backward_fn = [](Tensor &self) {
        Tensor &A = *self.parents[0];
        for (uint_least64_t i = 0; i < A.grad.size(); i++) {
          A.grad[i] += (1.0f / A.data[i] + 1e-8f) * self.grad[i];
        }
      };
    }
    return ans;
  }

  ~Tensor() { // destructor
  } // free memory is any allocated
};

class Layer {
public:
  virtual Tensor forward(Tensor &input) = 0;
  virtual vector<Tensor *> parameters() { return {}; };
  virtual ~Layer() {}
};

class Dense : public Layer {
private:
  Tensor W;
  Tensor b;
  Tensor input_matmul_W;
  Tensor W_plus_base;

public:
  Dense(int in_feature, int out_feature)
      : W(tensor_shape{{in_feature, out_feature}}, tensor_data{{}},
          grad_flag{true}),

        b(tensor_shape{{1, out_feature}}, tensor_data{{}}, grad_flag{true}),

        input_matmul_W(tensor_shape{{1, out_feature}}, tensor_data{{}},
                       grad_flag{true}),

        W_plus_base(tensor_shape{{1, out_feature}}, tensor_data{{}},
                    grad_flag{true}) {
    for (float &x : *W.get_data_pointer())
      x = ((float)rand() / RAND_MAX) - 0.5f;
    for (float &x : *b.get_data_pointer())
      x = ((float)rand() / RAND_MAX) - 0.5f;
  }

  Tensor forward(Tensor &input) {
    input_matmul_W = input.matmul_2D(W);
    W_plus_base = input_matmul_W + b;
    return W_plus_base;
  }

  vector<Tensor *> parameters() { return {&W, &b}; }
};

class ReLU : public Layer {
public:
  Tensor forward(Tensor &input) { return input.relu(); }
};

class Sigmoid : public Layer {
public:
  Tensor forward(Tensor &input) { return input.sigmoid(); }
};

class Loss_function {
public:
  virtual Tensor apply(Tensor &pred, Tensor &target) = 0;
};

class MSE : public Loss_function {
private:
  Tensor diff;
  Tensor sq;
  Tensor sum;
  Tensor out;

public:
  MSE(int feature)
      : diff(tensor_shape{{1, feature}}, tensor_data{{}}, grad_flag{true}),
        sq(tensor_shape{{1, feature}}, tensor_data{{}}, grad_flag{true}),
        sum(tensor_shape{{1, feature}}, tensor_data{{}}, grad_flag{true}),
        out(tensor_shape{{1, 1}}, tensor_data{{}}, grad_flag{true}) {}

  Tensor apply(Tensor &pred, Tensor &target) {
    diff = pred - target;
    sq = diff * diff;
    sum = sq.sum();
    out = sum * (1.0f / sq.get_data().size());
    return out;
  }
};

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
  Binary_cross_entropy() {};

  Tensor apply(Tensor &pred, Tensor &target) {
    pred_log = pred.log_on_Tensor();
    term1 = target * pred_log;
    one_minus_target = one - target;
    one_minus_pred = one - pred;
    one_minus_pred_log = one_minus_pred.log_on_Tensor();
    term2 = one_minus_target * one_minus_pred_log;
    term1_plus_term2 = term1 + term2;
    sum = term1_plus_term2.sum();
    out = sum * (-1.0f / pred.get_data().size());
    return out;
  }
};

class Optimizer {
public:
  virtual void step(vector<Tensor *> &parameters) = 0;
  virtual void zero_grad(vector<Tensor *> &parameters) {
    for (Tensor *t : parameters)
      t->zero_grad();
  }
};

class SGD : public Optimizer {
  float lr;

public:
  SGD(float lr_) : lr(lr_) {}

  void step(vector<Tensor *> &parameters) {
    for (Tensor *t : parameters) {
      vector<float> &w = *t->get_data_pointer();
      vector<float> g = t->grad;
      for (uint_least64_t i = 0; i < w.size(); i++) {
        w[i] -= lr * g[i];
      }
    }
  }
};

vector<Tensor *> get_all_parameters(vector<Layer *> &layers) {
  vector<Tensor *> params;
  for (Layer *l : layers) {
    vector<Tensor *> parameters_of_layer_l = l->parameters();
    params.insert(params.end(), parameters_of_layer_l.begin(),
                  parameters_of_layer_l.end());
  }
  return params;
};

int main() {
  vector<vector<float>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
  vector<vector<float>> Y = {{0}, {1}, {1}, {0}};

  Dense fc1(2, 1);
  Sigmoid sig1;
  Dense fc2(1, 1);
  Sigmoid sig2;

  vector<Layer *> model = {&fc1, &fc2};
  vector<Tensor *> parameters = get_all_parameters(model);

  SGD sgd(1e-3f);

  Binary_cross_entropy bce;

  for (int epoch = 0; epoch < 1000000; epoch++) {
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
      sgd.step(parameters);
      sgd.zero_grad(parameters);
    }

    if (epoch % 200 == 0) {
      cout << "Epoch " << epoch << " loss = " << epoch_loss / 4 << endl;
    }
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

// I have set bias to random values instead of 0 revert this change, if needed.

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
