#include "tensor/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <cstdint>

using namespace std;

Tensor Tensor::matmul_2D(Tensor &other) {
  if (shape.size() != 2 || other.get_shape().size() != 2){
    throw runtime_error("matmul_2D: tensors must be 2D");
  }
  if (shape[1] != other.get_shape()[0]){
    throw runtime_error("matmul_2D: dimension mismatch");
  }

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

Tensor Tensor::transpose_2D() const {
    
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

Tensor Tensor::operator+(Tensor &other) {
    if (shape != other.get_shape())
        throw runtime_error("operator+: shape mismatch");

    Tensor ans(tensor_shape{this->shape}, tensor_data{{}},grad_flag{this->requires_grad || other.requires_grad});

    vector<float> &ans_data_pointer = *ans.get_data_pointer();
    
    for (size_t i = 0; i < data.size(); i++)
        ans_data_pointer[i] = data[i] + other.data[i];

  if (requires_grad || other.requires_grad) {
    ans.parents = {this, &other};
    ans.backward_fn = [](Tensor &self) {
      Tensor &A = *self.parents[0];
      Tensor &B = *self.parents[1];
      for (size_t i = 0; i < self.grad.size(); i++) {
        if (A.requires_grad) A.grad[i] += self.grad[i];
        if (B.requires_grad) B.grad[i] += self.grad[i];
      }
    };
  }
  return ans;
}

Tensor Tensor::operator-(Tensor &other) {
    if (this->shape.size() != other.shape.size())
      throw runtime_error("operator - : Index Dimensions Mismatch");

    for (uint_least64_t i = 0; i < this->shape.size(); i++) {
      if (this->shape[i] != other.shape[i])
        throw runtime_error("operator - : Tensors have same dimensionality but "
                            "different Indexes");
    }

    Tensor ans(tensor_shape{this->shape}, tensor_data{{}},
               grad_flag{this->requires_grad || other.requires_grad});

    vector<float> &ans_data_pointer = *ans.get_data_pointer();

    for (uint_least64_t i = 0; i < this->data.size(); i++) {
      ans_data_pointer[i] = this->data[i] - other.data[i];
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

Tensor Tensor::operator*(float Scalar_value) {
    Tensor ans(tensor_shape{this->shape}, tensor_data{{}},
               grad_flag{this->requires_grad});

    vector<float>& ans_data_pointer = *ans.get_data_pointer();
    for (uint_least64_t i = 0; i < this->data.size(); i++) {
      ans_data_pointer[i] = this->data[i] * Scalar_value;
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

Tensor Tensor::operator*(Tensor &other) {// Element wise multiplication , for matrix
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

    vector<float>& ans_data_pointer = *ans.get_data_pointer();

    for (uint_least64_t i = 0; i < this->data.size(); i++) {
      ans_data_pointer[i] = this->data[i] * other.data[i];
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

Tensor Tensor::sum() {
  Tensor out(tensor_shape{{1}}, tensor_data{{}}, grad_flag{requires_grad});
  
  (*out.get_data_pointer())[0] =
      accumulate(data.begin(), data.end(), 0.0f);

  if (requires_grad) {
    out.parents = {this};
    out.backward_fn = [](Tensor &self) {
      Tensor &A = *self.parents[0];
      for (size_t i = 0; i < A.grad.size(); i++)
        A.grad[i] += self.grad[0];
    };
  }
  return out;
}

Tensor Tensor::relu() {
  Tensor ans(tensor_shape{shape}, tensor_data{{}}, grad_flag{requires_grad});
  
  vector<float> &ans_data_pointer = *ans.get_data_pointer();

  for (size_t i = 0; i < ans_data_pointer.size(); i++)
    ans_data_pointer[i] = max(0.0f, data[i]);

  if (requires_grad) {
    ans.parents = {this};
    ans.backward_fn = [](Tensor &self) {
      Tensor &A = *self.parents[0];
      for (size_t i = 0; i < A.grad.size(); i++)
        A.grad[i] += (A.data[i] > 0 ? 1.0f : 0.0f) * self.grad[i];
    };
  }
  return ans;
}

Tensor Tensor::sigmoid() {
  Tensor ans(tensor_shape{shape}, tensor_data{{}}, grad_flag{requires_grad});
  
  vector<float> &ans_data_pointer = *ans.get_data_pointer();

  for (size_t i = 0; i < ans_data_pointer.size(); i++)
    ans_data_pointer[i] = 1.0f / (1.0f + exp(-data[i]));

  if (requires_grad) {
    ans.parents = {this};
    ans.backward_fn = [](Tensor &self) {
      Tensor &A = *self.parents[0];
      for (size_t i = 0; i < A.grad.size(); i++) {
        float s = self.data[i];
        A.grad[i] += s * (1 - s) * self.grad[i];
      }
    };
  }
  
  return ans;
}

Tensor Tensor::log_on_Tensor() {
  Tensor ans(tensor_shape{shape}, tensor_data{{}}, grad_flag{requires_grad});

  vector<float> &ans_data_pointer = *ans.get_data_pointer();

  for (size_t i = 0; i < ans_data_pointer.size(); i++)
    ans_data_pointer[i] = log(data[i] + 1e-8f);

  if (requires_grad) {
    ans.parents = {this};
    ans.backward_fn = [](Tensor &self) {
      Tensor &A = *self.parents[0];
      for (size_t i = 0; i < A.grad.size(); i++)
        A.grad[i] += (1.0f / (A.data[i] + 1e-8f)) * self.grad[i];
    };
  }
  return ans;
}