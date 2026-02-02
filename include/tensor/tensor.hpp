#pragma once

#include <vector>
#include <functional>
#include <unordered_set>
#include <vector>

struct tensor_shape {
  std::vector<int> shape;
};

struct tensor_data {
  std::vector<float> data;
};

struct grad_flag {
  bool flag;
};

class Tensor {
private:
  std::vector<int> shape;
  std::vector<float> data;
  bool requires_grad = false;

public:
  std::vector<float> grad;
  std::vector<Tensor *> parents;
  std::function<void(Tensor &)> backward_fn;

  Tensor(const tensor_shape &input_shape,
         const tensor_data &input_data = tensor_data{{}},
         grad_flag input_grad_flag = grad_flag{false});

  Tensor(const std::vector<int> &input_shape,
         const std::vector<float> &input_data = {},
         bool grad_flag = false);

  std::vector<int> get_shape();
  std::vector<float> get_data();
  std::vector<float> *get_data_pointer();

  Tensor matmul_2D(Tensor &other);
  Tensor transpose_2D() const;

  int flatten(const std::vector<int> &indices) const;

  float &operator()(std::vector<int> idx);
  float operator()(std::vector<int> idx) const;

  Tensor operator+(Tensor &other);
  Tensor operator-(Tensor &other);
  Tensor operator*(float scalar);
  Tensor operator*(Tensor &other);
  
  Tensor relu();
  Tensor sigmoid();
  Tensor log_on_Tensor();
  
  Tensor sum();

  void build_topo(Tensor *t, std::vector<Tensor *> &topo,
                  std::unordered_set<Tensor *> &visited);



  void backward();
  void zero_grad();

  ~Tensor();
};