#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <numeric>
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
          A.grad = dA.get_data();
        }
        if (B.requires_grad) {
          Tensor dB = A.transpose_2D().matmul_2D(dZ);
          B.grad = dB.get_data();
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

  Tensor operator-(const Tensor &other) const {
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

      ans.backward_fn = [&](Tensor &self) {
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
          A.grad[i] = (A.data[i] > 0 ? 1.0f : 0.0f) * self.grad[i];
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

int main() {
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

  // the variable ans in the backward_fn of matmul_2D, operator+, operator* is
  // captured with refernce to that member function but when the function
  // return, the refernce to the ans variable becomes a dangling refernce
  // because it has been deleted from memory.
  //
  // Solution : The defination should become more like
  //
  //            ans.backward_fn = [this](Tensor& self)
  //
  //            more in chatgpt.
  return 0;
}
