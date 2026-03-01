#include "tensor/tensor.hpp"
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

#define DEFAULT_SIZE 2048

int size_of_mat;

void randomize(Tensor &t) {
  vector<float> &data = *t.get_data_pointer();
  for (float &x : data) {
    x = ((float)rand() / RAND_MAX) - 0.5f;
  }
}

Tensor matmul_v1(Tensor &t1, Tensor &t2) {

  vector<float> &matrix_1_data = *t1.get_data_pointer();
  vector<float> &matrix_2_data = *t2.get_data_pointer();

  int matrix_1_columns = t1.get_shape()[0];
  int matrix_1_rows = t1.get_shape()[1];
  int matrix_2_rows = t2.get_shape()[1];

  vector<float> ans_data(matrix_1_columns * matrix_2_rows, 0.0f);

  // matrix_1 = m x n
  // matrix_2 = n x k

  // Matmul logic goes here
  for (int m = 0; m < matrix_1_columns; m++) {
    for (int n = 0; n < matrix_1_rows; n++) {
      for (int k = 0; k < matrix_2_rows; k++) {

        ans_data[m * matrix_1_columns + k] +=
            matrix_1_data[m * matrix_1_rows + n] *
            matrix_2_data[n * matrix_2_rows + k];
      }
    }
  }

  Tensor ans{{matrix_1_columns, matrix_2_rows}, ans_data};
  return ans;
}

int main(int argc, char *argv[]) {
  if (argc == 2)
    size_of_mat = stoi(argv[1]);
  else
    size_of_mat = DEFAULT_SIZE;

  Tensor t1{{size_of_mat, size_of_mat}};
  Tensor t2{{size_of_mat, size_of_mat}};
  randomize(t1);
  randomize(t2);

  // Tensor t1{{2, 3}, {1, 2, 3, 4, 5, 6}};
  // Tensor t2{{3, 2}, {1, 2, 3, 4, 5, 6}};

  // Tensor t1{{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}};
  // Tensor t2{{3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}};

  cout << "Matmul begins" << endl;
  Tensor out = matmul_v1(t1, t2);
  // Tensor out = t1.matmul_2D(t2);
  cout << "Matmul ended" << endl;

  return 0;
}

// command :
// perf stat -e cycles,instructions,cache-references,cache-misses ./train

// matmul_v1 tries to maximize its cache-hits hence utilizing cache to its
// maximum potential

// Learned cache utilization from :
// https://michalpitr.substack.com/p/optimizing-matrix-multiplication

/*
 1) buildin tensor matmul_2D

    173,936,681,879      cycles
    61,185,768,557      instructions  #    0.35  insn per cycle
     8,673,611,266      cache-references
     4,056,200,010      cache-misses  #   46.76% of all cache refs

      39.665592651 seconds time elapsed

      39.635121000 seconds user
       0.023996000 seconds sys

2) matmul_v1

    18,247,823,959      cycles
    69,496,379,584      instructions #    3.81  insn per cycle
       524,961,725      cache-references
       156,402,541      cache-misses #   29.79% of all cache refs

       4.165514517 seconds time elapsed

       4.145482000 seconds user
       0.018997000 seconds sys
*/

// Implement :
// RSMprop

// loading functions of model, layers, and tensors are defined in
// helpers.hpp and Implementend in helpers.cpp
//
// If needed move them to appropriate files
